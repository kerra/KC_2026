from __future__ import annotations

import json
import sys
from functools import partial
from pathlib import Path


# Читает selected_heads.json CrAM: слой >> список голов.
# Инпут: str | Path путь
# Аутпут: dict[int, list[int]]
def load_selected_heads(path: str | Path) -> dict[int, list[int]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(layer): [int(h) for h in heads] for layer, heads in data.items()}


def _import_cram(cram_repo: Path):
    repo = str(Path(cram_repo).resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from utils.re_weighting import Re_Weighting_Strategy  # noqa: WPS433
    from utils.prompt import get_prompt  # noqa: WPS433
    return Re_Weighting_Strategy, get_prompt


# Обёртка над официальным кодом CrAM: их загрузка модели, промпт и хук маски внимания (Re_Weighting_Strategy).
# Путь оценок CrAM (приведение к int, сдвиг на минимум, сортировка по баллу) обойдён: множители приходят готовыми,
# порядок пассажей задаёт билдер контекста. Нужна CUDA; путь модели должен содержать Llama-3 или Qwen.
# Инпут: str путь модели; путь к клону CrAM; dict выбранных голов (слой >> головы); int максимум новых токенов
# Аутпут: объект; answer() >> dict; answer_closed_book() / judge() / extract_answer() >> str
class CramReader:
    def __init__(self, model_path: str, cram_repo: str | Path, selected_heads: dict[int, list[int]] | None = None,
                 max_new_tokens: int = 32):
        import torch

        self.torch = torch
        strategy_cls, self._get_prompt = _import_cram(Path(cram_repo))
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.strategy = strategy_cls(model_path, layers_to_be_modified=selected_heads or {})
        self.selected_heads = selected_heads or {}
        self.tokenizer = self.strategy.tokenizer
        self.model = self.strategy.model
        self.is_llama3 = "Llama-3" in model_path
        self.is_qwen = "Qwen" in model_path

    # ------------------------------------------------------------------ prompts
    # Повторяет ветвление шаблонов чата из кода CrAM (Llama-3 / Qwen / без шаблона).
    # Инпут: str промпт пользователя
    # Аутпут: tuple[str, bool] промпт и флаг add_special_tokens
    def _chat(self, user_prompt: str) -> tuple[str, bool]:
        if self.is_llama3:
            msgs = [{"role": "system", "content": "You are a helpful assistant!"}, {"role": "user", "content": user_prompt}]
            return self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False), True
        if self.is_qwen:
            msgs = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": user_prompt}]
            return self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False), False
        return user_prompt, True

    def _gen_kwargs(self, max_new_tokens: int) -> dict:
        kw = {"do_sample": False, "max_new_tokens": max_new_tokens, "pad_token_id": self.tokenizer.eos_token_id}
        if self.is_llama3:
            kw["eos_token_id"] = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        return kw

    def _decode_new(self, outputs, input_len: int) -> str:
        return self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

    # ------------------------------------------------------------------ RAG answer
    # Отвечает на вопрос по пассажам с перевзвешиванием CrAM: log(множителя) добавляется в маску внимания выбранных голов
    # на позициях токенов пассажа; хуки ставятся, только если есть множитель не равный 1 (или force_hook).
    # Инпут: str вопрос; list[str] пассажи; list[float] множители; bool force_hook
    # Аутпут: dict answer, hooked, prompt, n_prompt_tokens
    def answer(self, question: str, passages: list[str], multipliers: list[float], force_hook: bool = False) -> dict:
        if len(passages) != len(multipliers):
            raise ValueError("one multiplier per passage")
        model_inputs, attention_weight = self.strategy.decode_with_special_attention(
            question=question, paras=passages, scores=multipliers)
        hooked = bool(self.selected_heads) and (force_hook or any(abs(m - 1.0) > 1e-9 for m in multipliers))
        handles = []
        if hooked:
            log_w = self.torch.log(attention_weight)
            for layer_idx, head_idx in self.selected_heads.items():
                module = self.model.get_submodule(f"model.layers.{layer_idx}.self_attn")
                hook = partial(self.strategy.edit_attention_mask, attention_weight=log_w, head_idx=head_idx)
                handles.append(module.register_forward_pre_hook(hook, with_kwargs=True))
        try:
            with self.torch.no_grad():
                out = self.model.generate(**model_inputs, **self._gen_kwargs(self.max_new_tokens))
        finally:
            for h in handles:
                h.remove()
        input_len = model_inputs["input_ids"].shape[1]
        return {"answer": self._decode_new(out, input_len), "hooked": hooked,
                "prompt": self.tokenizer.decode(model_inputs["input_ids"][0]), "n_prompt_tokens": int(input_len)}

    # ------------------------------------------------------------------ plain generation (no hooks)
    # Генерация без хуков; prefill дописывается после заголовка ассистента, чтобы чат-модель продолжила нужный формат.
    # Инпут: str промпт пользователя; int максимум токенов; str префилл
    # Аутпут: str сгенерированное продолжение
    def _plain(self, user_prompt: str, max_new_tokens: int, prefill: str = "") -> str:
        prompt, add_special = self._chat(user_prompt)
        prompt = prompt + prefill
        inputs = self.tokenizer([prompt], return_tensors="pt", add_special_tokens=add_special).to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, **self._gen_kwargs(max_new_tokens))
        return self._decode_new(out, inputs["input_ids"].shape[1])

    def answer_closed_book(self, question: str) -> str:
        return self._plain(self._get_prompt(question=question, type="without_contexts"), self.max_new_tokens)

    # Судья CrAM на ридере: ход ассистента предзаполнен "Analysis:", иначе чат-модель просила прислать пассаж вместо оценки.
    # Инпут: str промпт судьи; int максимум токенов; str префилл
    # Аутпут: str префилл + ответ (анализ и балл)
    def judge(self, judge_prompt: str, max_new_tokens: int = 400, prefill: str = "Analysis:\n") -> str:
        return prefill + self._plain(judge_prompt, max_new_tokens, prefill=prefill)

    # Извлечение для голосования: что этот один пассаж говорит в ответ на вопрос.
    # Инпут: str вопрос; str пассаж; int максимум токенов
    # Аутпут: str короткий ответ или unknown
    def extract_answer(self, question: str, passage: str, max_new_tokens: int = 16) -> str:
        from ..signals.agreement import EXTRACT_PROMPT
        return self._plain(EXTRACT_PROMPT.format(passage=passage, question=question), max_new_tokens)
