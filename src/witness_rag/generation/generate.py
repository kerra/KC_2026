from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from pathlib import Path

from ..io import append_jsonl, load_config, read_jsonl, word_count
from ..schemas import Pair, from_dict
from .chunking import alias_pattern, window_text
from .coordinated import light_edit, split_sentences
from .stance import negated_mentions, strip_markdown
from .prompts import PromptBook


# Обёртка над HF-моделью для чата; загружается лениво, чтобы CPU-путь не импортировал torch.
# Инпут: str путь к модели; dtype
# Аутпут: объект; chat(system, user, seed, temperature, top_p, max_new_tokens) >> str ответ модели
class HFGenerator:

    def __init__(self, model_path: str, dtype: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=dtype)
        self.model.eval()
        self.supports_system = True

    def _prompt(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            if self.supports_system:
                return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:      # some chat templates reject the system role
            self.supports_system = False
        merged = [{"role": "user", "content": system + "\n\n" + user}]
        return self.tok.apply_chat_template(merged, tokenize=False, add_generation_prompt=True)

    def chat(self, system: str, user: str, seed: int, temperature: float, top_p: float, max_new_tokens: int) -> str:
        prompt = self._prompt(system, user)
        inputs = self.tok(prompt, return_tensors="pt").to(self.model.device)
        self.torch.manual_seed(seed)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, do_sample=True, temperature=temperature, top_p=top_p,
                                      max_new_tokens=max_new_tokens, pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def _load_existing(path: Path) -> dict[str, dict]:
    return {d["doc_id"]: d for d in read_jsonl(path)} if path.exists() else {}


_NAME_RUN = r"(?-i:(?:\b[A-Z][a-z]+\.?[ \t]+){0,2})"   # "Peter Tchaikovsky" >> the whole name is replaced, not just the surname (case-sensitive)


# Последнее средство после ретраев, по предложениям: выбрасывает контрасты ("Mozart, not Tchaikovsky") и мета-комментарии,
# заменяет одиночные упоминания конкурирующего значения на утверждаемое (с именем перед фамилией), удаляет предложения с отрицанием.
# Инпут: str текст; list[str] свои алиасы; list[str] чужие алиасы; PromptBook
# Аутпут: tuple[str, list[str]] текст и список правок (пустой список = текст не менялся)
def salvage(text: str, own_aliases: list[str], other_aliases: list[str], book: PromptBook) -> tuple[str, list[str]]:
    edits: list[str] = []
    own = alias_pattern(own_aliases)
    other = alias_pattern(other_aliases) if other_aliases else None
    other_run = re.compile(_NAME_RUN + "(?:" + other.pattern + ")", other.flags) if other else None
    kept: list[str] = []
    n_contrast = n_subst = n_meta = n_neg = 0
    for sent in split_sentences(text):
        if other and other.search(sent):
            if own.search(sent):
                n_contrast += 1
                continue
            sent, n = other_run.subn(own_aliases[0], sent)
            n_subst += n
        if book.has_meta_commentary(sent) or book.looks_like_refusal(sent):
            n_meta += 1
            continue
        if negated_mentions(sent, own):
            n_neg += 1
            continue
        kept.append(sent)
    if n_contrast:
        edits.append(f"dropped_{n_contrast}_contrast_sentences")
    if n_subst:
        edits.append(f"substituted_{n_subst}_mentions")
    if n_meta:
        edits.append(f"dropped_{n_meta}_meta_sentences")
    if n_neg:
        edits.append(f"dropped_{n_neg}_negated_sentences")
    return (" ".join(kept) if edits else text), edits


# Проверяет сгенерированный пассаж: отказ или мета-комментарий, длина меньше окна, нет утверждаемого значения,
# присутствует конкурирующее значение, утверждаемое значение отрицается.
# Инпут: str текст; list[str] свои алиасы; list[str] чужие алиасы; int минимум слов; PromptBook
# Аутпут: str причина отказа или None (пассаж годен)
def check_generation(text: str, aliases: list[str], other_aliases: list[str], min_words: int, book: PromptBook) -> str | None:
    if book.looks_like_refusal(text) or book.has_meta_commentary(text):
        return "refusal"
    if word_count(text) < min_words:
        return "too_short"
    own = alias_pattern(aliases)
    if not own.search(text):
        return "value_missing"
    if other_aliases and alias_pattern(other_aliases).search(text):
        return "other_value_present"
    if negated_mentions(text, own):
        return "own_value_negated"
    return None


# Проверка для ячейки A_C: пассаж должен утверждать ложное значение и называть истинное только для того, чтобы его отрицать.
# Инпут: те же, что у check_generation
# Аутпут: str причина (дополнительно other_value_missing, affirms_true) или None
def check_contrastive(text: str, aliases: list[str], other_aliases: list[str], min_words: int, book: PromptBook) -> str | None:
    from .human_cells import cram_fake_is_false
    if book.looks_like_refusal(text) or book.has_meta_commentary(text):
        return "refusal"
    if word_count(text) < min_words:
        return "too_short"
    own, other = alias_pattern(aliases), alias_pattern(other_aliases)
    if not own.search(text):
        return "value_missing"
    if not other.search(text):
        return "other_value_missing"
    if negated_mentions(text, own):
        return "own_value_negated"
    if not cram_fake_is_false(text, other, own):
        return "affirms_true"
    return None


# Выполняет план: D4-копии на CPU, затем D0/D2/D3 по одному генератору за раз с ретраями (новый сид, горячее сэмплирование)
# и спасением; возобновляемо, готовые doc_id пропускаются, отвергнутые попытки пишутся в refusals.jsonl.
# Инпут: dict конфиг; list[dict] план; dict pair_id >> Pair; Path выходной jsonl; set уровней; set генераторов или None;
# int лимит пар; bool сухой прогон; int сдвиг сидов для раунда регенерации
# Аутпут: dict счётчики (generated, skipped, retried, salvaged, gave_up, missing_parent)
def generate(cfg: dict, plan: list[dict], pairs: dict[str, Pair], out_path: Path, levels: set[str],
             models: set[str] | None, limit_pairs: int | None, dry_run: bool, seed_offset: int = 0) -> dict:
    book = PromptBook()
    gcfg = cfg["generation"]
    n_window = int(gcfg["window_words"])
    docs = _load_existing(out_path)
    refusals_path = out_path.parent / "refusals.jsonl"
    stats = {"generated": 0, "skipped": 0, "retried": 0, "salvaged": 0, "gave_up": 0, "missing_parent": 0}
    if dry_run:
        stats = {"dry_run": True, "would_generate": 0}

    if limit_pairs:
        keep_pairs = set(sorted({r["pair_id"] for r in plan})[:limit_pairs])
        plan = [r for r in plan if r["pair_id"] in keep_pairs]
    pending = [r for r in plan if r["dependence_level"] in levels and r["doc_id"] not in docs
               and (models is None or r["model"] in models)]
    if not dry_run:
        stats["skipped"] = sum(1 for r in plan if r["dependence_level"] in levels and r["doc_id"] in docs)

    def emit(row: dict, full_text: str, aliases: list[str], attempts: int, model_path: str | None,
             salvaged: list[str] | None = None, unresolved: str | None = None):
        window, start = window_text(full_text, aliases, n_window, row["seed"])
        docs[row["doc_id"]] = {"doc_id": row["doc_id"], "text": window, "n_words": word_count(window),
                               "full_text": full_text, "window_start": start,
                               "gen_date": dt.date.today().isoformat(), "model_path": model_path, "attempts": attempts,
                               "salvaged": salvaged or [], "unresolved": unresolved, "seed_offset": seed_offset}
        append_jsonl(out_path, docs[row["doc_id"]])
        stats["generated"] += 1

    d4_rows = [r for r in pending if r["dependence_level"] == "D4"]
    if dry_run and d4_rows:
        print(f"[D4] {len(d4_rows)} light-edit copies would be made from their D2 seeds (no model)")
        stats["would_generate"] += len(d4_rows)
    for row in ([] if dry_run else d4_rows):                                  # no model needed
        parent = docs.get(row["parent_doc"])
        if not parent:
            stats["missing_parent"] += 1
            continue
        pair = pairs[row["pair_id"]]
        edited = light_edit(parent["text"], row["seed"], float(gcfg["copy_edit_token_rate"]),
                            protected=pair.gold_aliases() + pair.attack_aliases())
        emit(row, edited, pair.aliases_for(row["stance"]), 1, None)

    by_model: dict[str, list[dict]] = {}
    for r in pending:
        if r["dependence_level"] in ("D0", "D2", "D3"):
            by_model.setdefault(r["model"], []).append(r)

    for model_id, rows in by_model.items():
        model_path = gcfg["generators"][model_id]
        print(f"[{model_id}] {len(rows)} docs -> {model_path}")
        if dry_run:
            for r in rows[:2]:
                pair = pairs[r["pair_id"]]
                other = pair.answer_for("false" if r["stance"] == "true" else "true")
                print("   ", "D3 paraphrase of " + str(r["parent_doc"]) if r["dependence_level"] == "D3"
                      else book.render(r["genre"], pair.question, pair.answer_for(r["stance"]), other, r["length_target"])[:220])
            stats["would_generate"] += len(rows)
            continue
        gen = HFGenerator(model_path)
        for r in rows:
            pair = pairs[r["pair_id"]]
            other_stance = "false" if r["stance"] == "true" else "true"
            aliases, other_aliases = pair.aliases_for(r["stance"]), pair.aliases_for(other_stance)
            if r["dependence_level"] == "D3":
                parent = docs.get(r["parent_doc"])
                if not parent:
                    stats["missing_parent"] += 1
                    continue
                user = book.render_paraphrase(parent["text"])
                min_words = int(n_window * float(gcfg.get("min_words_ratio_paraphrase", 0.6)))
            else:
                user = book.render(r["genre"], pair.question, pair.answer_for(r["stance"]), pair.answer_for(other_stance),
                                   r["length_target"])
                min_words = n_window
            text, attempts, problem = "", 0, None
            base_temp = float(r["temperature"] or gcfg["temperature_independent"])
            while attempts <= int(gcfg["max_refusal_retries"]):
                attempts += 1
                temp = min(float(gcfg.get("max_temperature", 1.25)),
                           base_temp + float(gcfg.get("retry_temperature_step", 0.0)) * (attempts - 1))
                text = strip_markdown(gen.chat(book.system, user, seed=r["seed"] + seed_offset + 1000 * (attempts - 1), temperature=temp,
                                               top_p=float(gcfg["top_p"]), max_new_tokens=int(gcfg["max_new_tokens"])))
                contrastive = r["cell"] == "A_C"
                problem = (check_contrastive if contrastive else check_generation)(text, aliases, other_aliases, min_words, book)
                if problem is None:
                    break
                stats["retried"] += 1
                # the rejected attempts are kept in full: they document why a passage was regenerated
                append_jsonl(refusals_path, {"doc_id": r["doc_id"], "attempt": attempts, "problem": problem,
                                             "temperature": round(temp, 3), "text": text})
            salvaged, unresolved = [], None
            if problem is not None and r["cell"] == "A_C":
                unresolved = problem                       # no salvage for contrastive passages: substitution would destroy the contrast
                stats["gave_up"] += 1
            elif problem is not None:
                fixed, edits = salvage(text, aliases, other_aliases, book)
                if edits and check_generation(fixed, aliases, other_aliases, min_words, book) is None:
                    text, salvaged = fixed, edits
                    stats["salvaged"] += 1
                else:
                    unresolved = problem
                    stats["gave_up"] += 1    # kept anyway; qa_gates will flag it for regeneration
            emit(r, text, aliases, attempts, model_path, salvaged, unresolved)
        del gen
        _release_gpu()
    return stats


def _release_gpu() -> None:
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default="data/production_log_plan.jsonl")
    ap.add_argument("--pairs", default="data/claims/pairs.jsonl")
    ap.add_argument("--out", default="data/generated/documents_generated.jsonl")
    ap.add_argument("--levels", nargs="+", default=["D0", "D2", "D3", "D4"])
    ap.add_argument("--models", nargs="*", default=None, help="generator ids, default all")
    ap.add_argument("--limit-pairs", type=int, default=None, help="smoke test: only the first N pairs")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed-offset", type=int, default=0, help="regeneration round: shift all sampling seeds (e.g. 100000)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    plan = read_jsonl(args.plan)
    pairs = {r["pair_id"]: from_dict(Pair, r) for r in read_jsonl(args.pairs)}
    print(generate(cfg, plan, pairs, Path(args.out), set(args.levels), set(args.models) if args.models else None,
                   args.limit_pairs, args.dry_run, seed_offset=args.seed_offset))
    return 0


if __name__ == "__main__":
    sys.exit(main())
