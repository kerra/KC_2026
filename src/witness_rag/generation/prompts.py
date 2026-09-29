from __future__ import annotations

import re

from ..io import read_yaml, resolve


# Пул промптов генерации: системный промпт, жанры с формой (organic | targeted), общий суффикс, шаблон парафраза,
# маркеры отказа и мета-комментариев (в том числе группы chatbot_leftovers и knowledge_disclaimers из configs/ai_markers.yaml).
# Инпут: str путь к YAML (по умолчанию configs/prompts/generation.yaml)
# Аутпут: объект; render() >> str промпт; looks_like_refusal() / has_meta_commentary() >> bool
class PromptBook:
    def __init__(self, path: str | None = None):
        cfg = read_yaml(resolve(path or "configs/prompts/generation.yaml"))
        self.version: str = str(cfg["version"])
        self.system: str = cfg["system"].strip()
        self.genres: dict[str, dict] = {k: {"shape": v["shape"], "template": v["template"].strip(),
                                            "mention_other": bool(v.get("mention_other", False))}
                                        for k, v in cfg["genres"].items()}
        self.suffix: str = cfg.get("suffix", "").strip()
        self.suffix_mention_other: str = cfg.get("suffix_mention_other", "").strip()
        self.paraphrase_template: str = cfg["paraphrase"].strip()
        self.refusal_patterns: list[str] = [p.lower() for p in cfg.get("refusal_patterns", [])]
        markers = read_yaml(resolve("configs/ai_markers.yaml"))["phrases"]
        # frame breaks from the prompt config + chatbot leftovers / knowledge disclaimers (Wikipedia: Signs of AI writing)
        self.meta_markers: tuple[str, ...] = tuple(p.lower() for p in cfg.get("meta_patterns", [])) + tuple(
            m.lower() for g in ("chatbot_leftovers", "knowledge_disclaimers") for m in markers.get(g, []))
        # letter boundaries: "here is a" must not fire inside "there is a"; a trailing comma/space in the marker is kept
        self._meta_res = [re.compile(r"(?<![a-z])" + re.escape(m) + r"(?![a-z])", re.I) for m in self.meta_markers]

    # Жанры основного дизайна заданной формы; контрастный жанр, называющий конкурирующее значение, исключён.
    # Инпут: str форма (organic | targeted)
    # Аутпут: list[str] имена жанров
    def genres_by_shape(self, shape: str) -> list[str]:
        return [g for g, v in self.genres.items() if v["shape"] == shape and not v["mention_other"]]

    def shape(self, genre: str) -> str:
        return self.genres[genre]["shape"]

    def prompt_id(self, genre: str) -> str:
        return f"{genre}.{self.version}"

    # Собирает пользовательский промпт жанра: один и тот же суффикс для истинных и ложных пассажей, меняются только два значения.
    # Инпут: str жанр; str вопрос; str утверждаемое значение; str конкурирующее значение; int целевая длина
    # Аутпут: str промпт
    def render(self, genre: str, question: str, answer: str, other: str, length: int) -> str:
        body = self.genres[genre]["template"].format(length=length, question=question.strip(), answer=answer.strip(),
                                                     other=other.strip())
        suffix = self.suffix_mention_other if self.genres[genre]["mention_other"] else self.suffix
        return (body + " " + suffix.format(length=length, other=other.strip(), answer=answer.strip())).strip()

    def render_paraphrase(self, text: str) -> str:
        return self.paraphrase_template.format(text=text.strip())

    def looks_like_refusal(self, text: str) -> bool:
        head = text.strip()[:300].lower()
        return any(p in head for p in self.refusal_patterns)

    def has_meta_commentary(self, text: str) -> bool:
        return bool(self.meta_hits(text))

    def meta_hits(self, text: str) -> list[str]:
        return [m for m, rx in zip(self.meta_markers, self._meta_res) if rx.search(text)]
