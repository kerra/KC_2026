from __future__ import annotations

import random
import re


# Компилирует регулярку, находящую любой из алиасов как отдельное слово без учёта регистра; длинные алиасы раньше коротких.
# Инпут: list[str] алиасы
# Аутпут: re.Pattern (не совпадает ни с чем, если алиасов нет)
def alias_pattern(aliases: list[str]) -> re.Pattern:
    alts = sorted({a.strip() for a in aliases if a and a.strip()}, key=len, reverse=True)
    if not alts:
        return re.compile(r"(?!x)x")  # matches nothing
    return re.compile(r"(?<![A-Za-z0-9])(" + "|".join(re.escape(a) for a in alts) + r")(?![A-Za-z0-9])", re.I)


# Находит диапазоны токенов [start, end), покрывающие каждое вхождение алиаса в тексте ' '.join(tokens).
# Инпут: list[str] токены; list[str] алиасы
# Аутпут: list[tuple[int, int]]
def alias_token_spans(tokens: list[str], aliases: list[str]) -> list[tuple[int, int]]:
    text = " ".join(tokens)
    starts, pos = [], 0
    for t in tokens:
        starts.append(pos)
        pos += len(t) + 1
    spans = []
    for m in alias_pattern(aliases).finditer(text):
        s_tok = max(i for i, st in enumerate(starts) if st <= m.start())
        e_tok = max(i for i, st in enumerate(starts) if st <= m.end() - 1) + 1
        spans.append((s_tok, e_tok))
    return spans


# Вырезает окно из n_words слов со случайным сидированным смещением так, чтобы в него попало вхождение алиаса, если оно есть;
# короткий текст возвращается целиком.
# Инпут: str текст; list[str] алиасы; int размер окна в словах; int сид
# Аутпут: tuple[str, int] окно и индекс его первого токена
def window_text(text: str, aliases: list[str], n_words: int, seed: int) -> tuple[str, int]:
    tokens = text.split()
    if len(tokens) <= n_words:
        return " ".join(tokens), 0
    rng = random.Random(seed)
    max_start = len(tokens) - n_words
    spans = alias_token_spans(tokens, aliases)
    if spans:
        s, e = rng.choice(spans)
        lo, hi = max(0, e - n_words), min(s, max_start)
        start = rng.randint(lo, hi) if hi >= lo else max(0, min(s, max_start))
    else:
        start = rng.randint(0, max_start)
    return " ".join(tokens[start:start + n_words]), start
