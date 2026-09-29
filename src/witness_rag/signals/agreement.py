from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

from ..eval.outcomes import contains_alias, normalize_text
from ..io import sha256_text
from .base import Context, ContextDoc, Signal
from .dependence_discount import DependenceSignal

NO_ANSWER = {"", "unknown", "not stated", "no answer", "none", "n a", "not mentioned", "i don t know",
             "unanswerable", "not given", "cannot be determined"}
_PREFIX = re.compile(r"^(?:(?:the )?answer(?: is)?|it is|it was|is)\s+")   # articles are already stripped by normalize_text

EXTRACT_PROMPT = ("Read the passage and answer the question using only the passage, with one or few words. "
                  "If the passage does not answer the question, reply exactly: unknown.\n"
                  "Passage: {passage}\nQuestion: {question}\nAnswer:")


# Нормализует извлечённый ответ: первая строка, нормализация CrAM/SQuAD, срез префиксов вида "the answer is".
# Инпут: str сырой ответ
# Аутпут: str нормализованный ответ
def normalize_answer(text: str) -> str:
    t = normalize_text(text.strip().split("\n")[0])
    t = _PREFIX.sub("", t).strip()
    return t


# Объединяет варианты одного ответа: формы, содержащие друг друга, сводятся к одному представителю ("nolan" >> "christopher nolan").
# Инпут: list[str] нормализованные ответы
# Аутпут: dict ответ >> представитель группы
def merge_groups(answers: list[str]) -> dict[str, str]:
    reps: dict[str, str] = {}
    for a in sorted({a for a in answers if a not in NO_ANSWER}, key=len, reverse=True):
        rep = next((r for r in reps.values() if contains_alias(r, a) or contains_alias(a, r)), None)
        reps[a] = rep or a
    return reps


# Голосование свидетелей: каждый пассаж отвечает на вопрос (извлечение ридером по одному пассажу, без суждения о правде),
# пассажи группируются по ответу, оценка = доля веса его группы; вес 1 / размер кластера (agreement) или 1 (voting).
# Пассажи без ответа не голосуют и получают 1. Извлечения кэшируются по sha256(вопрос + текст).
# Инпут: callable extract(question, passage) >> str; Path кэша; DependenceSignal или None; str имя арма
# Аутпут: Signal; scores(ctx) >> list[float]; extras(ctx) >> ответы и веса групп
class AgreementSignal(Signal):
    def __init__(self, extract: Callable[[str, str], str], cache_path: Path | None = None,
                 dependence: DependenceSignal | None = None, name: str = "agreement"):
        self.name = name
        self.extract = extract
        self.dependence = dependence
        self.cache_path = cache_path
        self.cache: dict[str, str] = {}
        if cache_path and Path(cache_path).exists():
            with open(cache_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        row = json.loads(line)
                        self.cache[row["key"]] = row["answer"]

    def answer_for(self, question: str, text: str) -> str:
        key = sha256_text(question + "\x1f" + text)
        if key not in self.cache:
            raw = self.extract(question, text)
            self.cache[key] = normalize_answer(raw)
            if self.cache_path:
                Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
                with open(self.cache_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({"key": key, "answer": self.cache[key], "raw": raw[:100]}, ensure_ascii=False) + "\n")
        return self.cache[key]

    def _groups(self, ctx: Context) -> tuple[list[str], dict[str, float]]:
        answers = [self.answer_for(ctx.question, d.text) for d in ctx.docs]
        reps = merge_groups(answers)
        canon = [reps.get(a, a) for a in answers]
        weights = self.dependence.scores(ctx) if self.dependence is not None else [1.0] * len(ctx.docs)
        groups: dict[str, float] = {}
        for a, w in zip(canon, weights):
            if a not in NO_ANSWER:
                groups[a] = groups.get(a, 0.0) + w
        return canon, groups

    def scores(self, ctx: Context) -> list[float]:
        canon, groups = self._groups(ctx)
        total = sum(groups.values())
        return [groups[a] / total if (a not in NO_ANSWER and total > 0) else 1.0 for a in canon]

    def extras(self, ctx: Context) -> dict:
        canon, groups = self._groups(ctx)
        return {f"{self.name}_answers": canon, f"{self.name}_groups": {k: round(v, 3) for k, v in groups.items()}}

    def precompute(self, docs: list[ContextDoc]) -> None:
        return None  # extraction needs the question; answers are cached per (question, passage) during the run
