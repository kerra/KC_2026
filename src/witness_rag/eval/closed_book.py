from __future__ import annotations

from pathlib import Path

from ..io import append_jsonl, read_jsonl
from ..schemas import Pair
from .outcomes import classify


# Проба параметрического знания: ридер отвечает на каждый вопрос без контекста; knows = ответ содержит истинное значение.
# Инпут: CramReader; str id ридера; list[Pair]; Path выходного jsonl (готовые пары пропускаются)
# Аутпут: list[dict] новые строки pair_id, reader, answer, outcome, knows
def run_closed_book(reader, reader_id: str, pairs: list[Pair], out_path: Path) -> list[dict]:
    done = {r["pair_id"] for r in read_jsonl(out_path)} if out_path.exists() else set()
    rows = []
    for pair in pairs:
        if pair.pair_id in done:
            continue
        ans = reader.answer_closed_book(pair.question)
        outcome = classify(ans, pair.gold_aliases(), pair.attack_aliases())
        row = {"pair_id": pair.pair_id, "reader": reader_id, "answer": ans, "outcome": outcome,
               "knows": outcome == "gold"}
        append_jsonl(out_path, row)
        rows.append(row)
    return rows


# Загружает бакеты знания ридера.
# Инпут: Path файла closed_book
# Аутпут: dict pair_id >> bool knows (пусто, если файла нет)
def load_knowledge(path: Path) -> dict[str, bool]:
    return {r["pair_id"]: bool(r["knows"]) for r in read_jsonl(path)} if Path(path).exists() else {}
