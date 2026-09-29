from __future__ import annotations

import random
import re
from typing import Iterable

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])|\s*\n+\s*")   # also split at line breaks (headings, datelines)
_STRIP = ".,;:!?\"'()[]"
_HAS_DIGIT = re.compile(r"\d")


# Делит текст на предложения по знакам конца предложения и переводам строки.
# Инпут: str текст
# Аутпут: list[str] предложения
def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_SPLIT.split(text.strip()) if p.strip()]
    return parts or [text.strip()]


def _edit_token(tok: str, rng: random.Random) -> str:
    core = tok.strip(_STRIP)
    if not core:
        return tok
    start = tok.find(core)
    prefix, suffix = tok[:start], tok[start + len(core):]
    roll = rng.random()
    if len(core) >= 4 and roll < 0.5:                       # swap two adjacent interior characters
        p = rng.randrange(1, len(core) - 2)
        core = core[:p] + core[p + 1] + core[p] + core[p + 2:]
    elif roll < 0.75 and core[0].isalpha():                 # toggle case of the first letter
        core = (core[0].lower() if core[0].isupper() else core[0].upper()) + core[1:]
    elif len(core) >= 3:                                    # doubled letter typo
        p = rng.randrange(1, len(core) - 1)
        core = core[:p] + core[p] + core[p:]
    return prefix + core + suffix


def _protected_words(protected: Iterable[str]) -> set[str]:
    words = set()
    for alias in protected:
        for w in alias.split():
            words.add(w.strip(_STRIP).lower())
    return {w for w in words if w}


# Делает почти-копию текста (уровень D4): перемешивает средние предложения и портит около token_rate токенов,
# не трогая первое предложение, токены с цифрами и слова защищённых алиасов. Детерминировано по сиду.
# Инпут: str текст; int сид; float доля правок; bool перемешивать предложения; Iterable[str] защищённые алиасы
# Аутпут: str отредактированный текст
def light_edit(text: str, seed: int, token_rate: float = 0.10, shuffle_sentences: bool = True,
               protected: Iterable[str] = ()) -> str:
    rng = random.Random(seed)
    sents = split_sentences(text)
    if shuffle_sentences and len(sents) > 3:
        middle = sents[1:-1]
        rng.shuffle(middle)
        sents = [sents[0], *middle, sents[-1]]
    first_len = len(sents[0].split())
    tokens = " ".join(sents).split()
    if not tokens:
        return text
    guard = _protected_words(protected)
    editable = [i for i, t in enumerate(tokens)
                if i >= first_len and not _HAS_DIGIT.search(t) and t.strip(_STRIP).lower() not in guard]
    n_edit = min(len(editable), max(1, round(len(tokens) * token_rate)))
    for i in rng.sample(editable, n_edit) if editable else []:
        tokens[i] = _edit_token(tokens[i], rng)
    return " ".join(tokens)
