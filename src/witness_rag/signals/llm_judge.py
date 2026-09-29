from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

from ..io import sha256_text
from .base import Context, ContextDoc, Signal

TRUTHFUL_JUDGE_PROMPT = """Your task is to evaluate the authenticity of a text based on your internal knowledge. Specifically, I will provide you with a passage that may contain accurate information or fabricated errors. Using your own knowledge, reason, and deduction, you are to assign a credibility score ranging from 0 to 10, where a higher score indicates greater authenticity and a lower score suggests lesser authenticity.
Here are 2 examples (you should follow the output format below):
##########
Passage:
In a groundbreaking discovery, researchers have found that Albert Einstein was the first recipient of the Nobel Prize in Physics. According to newly uncovered documents, Einstein's pioneering work in theoretical physics, particularly his theory of relativity, was recognized by the Nobel Committee in 1921. This revelation challenges the long-held belief that Marie Curie was the first Nobel laureate in physics, and solidifies Einstein's place as one of the greatest minds in scientific history.

Analysis:
1. Albert Einstein as the First Nobel Prize Recipient in Physics: This is incorrect. The first Nobel Prize in Physics was awarded in 1901, not to Albert Einstein, but to Wilhelm Conrad Röntgen for the discovery of X-rays.
2. Einstein's Nobel Prize Recognition: Albert Einstein was indeed awarded the Nobel Prize in Physics in 1921, but not for his theory of relativity. He received it for his discovery of the photoelectric effect, which was instrumental in the development of quantum theory.
3. Marie Curie as the First Nobel Laureate in Physics: This is also incorrect. Marie Curie was a Nobel laureate, but she was not the first to win the Nobel Prize in Physics. Her first Nobel Prize was in Physics in 1903, shared with her husband Pierre Curie and Henri Becquerel for their work on radioactivity. Marie Curie was, notably, the first woman to win a Nobel Prize, and the first person to win Nobel Prizes in two different scientific fields (Physics and Chemistry).
4. Implication about the Nobel Committee's Recognition of Relativity: As mentioned, Einstein's Nobel Prize was not for relativity, despite its profound impact on physics. The Nobel Committee specifically avoided awarding the prize for relativity at the time due to ongoing debates and lack of experimental confirmation of the theory during that period.

Credibility Score: 0


Passage:
The first Nobel Prize in Physics was awarded to Wilhelm Conrad Roentgen in 1901. Roentgen received the Nobel Prize for his discovery of X-rays, which had a significant impact on the field of physics and medicine

Analysis:
The facts presented in the statement you provided are largely accurate.

Credibility Score: 10
##########

Passage:
{para}
"""

_SCORE = re.compile(r"Credibility Score\**\s*[:=]?\s*\**\s*(\d+)(?:\s*/\s*10)?", re.I)
# fallback: "a score of 9 out of 10" but not the instruction echo "a credibility score from 0 to 10"
_SCORE_LOOSE = re.compile(r"\bscore\b(?:(?!from|between|ranging)[^0-9\n]){0,25}?(\d{1,2})\b(?!\s*(?:to|-|\u2013)\s*\d)(?:\s*/\s*10)?", re.I)


# Извлекает балл достоверности из ответа судьи: "Credibility Score: 7", жирный вариант, "= 7/10", запасной вариант "score ... N";
# берёт последнее упоминание и обрезает в 0..10.
# Инпут: str текст ответа модели
# Аутпут: int балл или None (балл не назван)
def parse_score(text: str) -> int | None:
    matches = _SCORE.findall(text) or _SCORE_LOOSE.findall(text)
    if not matches:
        return None
    return max(0, min(10, int(matches[-1])))


# Арм CrAM: ридер оценивает достоверность каждого пассажа по промпту truthful_judge из репозитория CrAM; балл 0..10 делится на 10,
# непарсируемый ответ после ретраев получает запасной балл 1 (как в коде CrAM); вердикты кэшируются по sha256 текста.
# Инпут: callable generate(prompt) >> str; Path кэша; int ретраев; int запасной балл
# Аутпут: Signal; scores(ctx) >> list[float]; extras(ctx) >> баллы по пассажам
class JudgeSignal(Signal):
    name = "judge"

    def __init__(self, generate: Callable[[str], str], cache_path: Path | None = None, retries: int = 2,
                 fallback_score: int = 1):
        self.generate = generate
        self.cache_path = cache_path
        self.retries = retries
        self.fallback = fallback_score
        self.cache: dict[str, dict] = {}
        if cache_path and Path(cache_path).exists():
            with open(cache_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        row = json.loads(line)
                        self.cache[row["key"]] = row

    def judge_text(self, text: str) -> dict:
        key = sha256_text(text)
        if key in self.cache:
            return self.cache[key]
        prompt = TRUTHFUL_JUDGE_PROMPT.format(para=text)
        score, raw, parsed = None, "", False
        for _ in range(self.retries):
            raw = self.generate(prompt)
            score = parse_score(raw)
            if score is not None:
                parsed = True
                break
        if score is None:
            score = self.fallback
        row = {"key": key, "score": score, "parsed": parsed, "raw": raw[-400:]}
        self.cache[key] = row
        if self.cache_path:
            Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row

    def precompute(self, docs: list[ContextDoc]) -> None:
        for d in docs:
            self.judge_text(d.text)

    def scores(self, ctx: Context) -> list[float]:
        return [self.judge_text(d.text)["score"] / 10.0 for d in ctx.docs]

    def extras(self, ctx: Context) -> dict:
        return {"judge_scores": [self.judge_text(d.text)["score"] for d in ctx.docs]}
