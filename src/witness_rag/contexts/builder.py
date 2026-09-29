from __future__ import annotations

import collections
from dataclasses import dataclass

from ..io import read_jsonl, stable_rng, stable_seed
from ..schemas import Pair, from_dict
from ..signals.base import Context, ContextDoc

LEVEL_RANK = {"D2": 0, "D3": 1, "D4": 2, "D0": 3, "H0": 4}


# Загруженный корпус: активные пары, документы по (pair_id, ячейка) с запечатанной разметкой и общий пул нейтральных пассажей H_N.
# Инпут: load(pairs_path, log_path, docs_path) >> пути к pairs.jsonl, production_log.jsonl, documents.jsonl
# Аутпут: объект; cell(pair_id, cell) >> list[dict] документов {doc_id, text, sealed}
@dataclass
class CorpusView:
    pairs: dict[str, Pair]
    by_pair_cell: dict[tuple[str, str], list[dict]]     # (pair_id, cell) >> [{doc_id, text, sealed}]
    neutral_pool: list[dict]                            # all H_N docs across pairs

    @classmethod
    def load(cls, pairs_path: str, log_path: str, docs_path: str) -> "CorpusView":
        pairs = {r["pair_id"]: from_dict(Pair, r) for r in read_jsonl(pairs_path) if not r.get("excluded")}
        docs = {d["doc_id"]: d for d in read_jsonl(docs_path)}
        by: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
        pool: list[dict] = []
        for row in read_jsonl(log_path):
            if row["doc_id"] not in docs or row["pair_id"] not in pairs:      # excluded pairs contribute nothing
                continue
            entry = {"doc_id": row["doc_id"], "text": docs[row["doc_id"]]["text"], "sealed": row}
            by[(row["pair_id"], row["cell"])].append(entry)
            if row["cell"] == "H_N":
                pool.append(entry)
        for key, entries in by.items():
            entries.sort(key=lambda e: (LEVEL_RANK.get(e["sealed"]["dependence_level"], 9), e["doc_id"]))
        return cls(pairs, dict(by), pool)

    def cell(self, pair_id: str, cell: str) -> list[dict]:
        return self.by_pair_cell.get((pair_id, cell), [])


# Рецепт условия: какие ячейки и по сколько документов входят в контекст; остаток до n_docs добивается нейтральными пассажами.
# Инпут: str условие; int k число ложных пассажей
# Аутпут: list[tuple[str, int]] ячейка и число документов
def recipe(condition: str, k: int) -> list[tuple[str, int]]:
    if condition == "clean":
        return [("H_T", 1), ("A_T", 2)]                     # без дезинформации; машинный текст истинен
    if condition == "shortcut":
        return [("H_T", 1), ("A_F", k)]                     # правда человеческая, ложь машинная, как у CrAM
    if condition == "ai_true_human_false":
        return [("A_T", 2), ("H_F", 1)]                     # правда машинная, ложь человеческая
    if condition == "all_ai_mixed":
        return [("A_T", 2), ("A_F", 2)]                     # автор фиксирован: только LLM
    if condition == "all_human":
        return [("H_T", 1), ("H_F", 1)]                     # автор фиксирован: только человек
    if condition == "independent":
        return [("H_T", 1), ("A_T", 1), ("A_F", k)]         # k лжей, написанных независимо
    if condition == "coordinated":
        return [("H_T", 1), ("A_T", 1), ("C_F", k)]         # сфабрикованный консенсус одного кластера
    if condition == "shortcut_cram":
        return [("H_T", 1), ("G_F", k)]                     # фейки самого CrAM, форма ответа на вопрос
    if condition == "shortcut_contrast":
        return [("H_T", 1), ("A_C", k)]                     # контрастные фейки "X, not Y"
    raise ValueError(f"unknown condition {condition!r}")


class ContextError(ValueError):
    pass


# Собирает контекст из n_docs пассажей по рецепту условия, добивает нейтральными (сначала своей пары, потом чужих)
# и перемешивает порядок сидированно и независимо от сигнала, так что перевзвешивание не смешивается с позицией.
# Инпут: CorpusView; str pair_id; str условие; int k; int n_docs; int сид проекта
# Аутпут: Context (вопрос и документы с запечатанной разметкой)
def build_context(view: CorpusView, pair_id: str, condition: str, k: int, n_docs: int, seed: int) -> Context:
    pair = view.pairs[pair_id]
    rng = stable_rng(seed, pair_id, condition, k)
    chosen: list[dict] = []
    for cell, m in recipe(condition, k):
        avail = list(view.cell(pair_id, cell))
        if len(avail) < m:
            raise ContextError(f"{pair_id}/{condition}/k={k}: needs {m} {cell} docs, has {len(avail)}")
        if cell == "C_F":
            picked = avail[:m]                       # plan order: D2 seeds, then paraphrase, then copy
        else:
            rng.shuffle(avail)
            picked = avail[:m]
        chosen.extend(picked)
    if len(chosen) > n_docs:
        raise ContextError(f"{pair_id}/{condition}/k={k}: recipe needs {len(chosen)} > n_docs={n_docs}")

    used = {c["doc_id"] for c in chosen}
    own_neutral = list(view.cell(pair_id, "H_N"))
    rng.shuffle(own_neutral)
    for d in own_neutral:
        if len(chosen) >= n_docs:
            break
        if d["doc_id"] not in used:
            chosen.append(d)
            used.add(d["doc_id"])
    others = [d for d in view.neutral_pool if d["sealed"]["pair_id"] != pair_id and d["doc_id"] not in used]
    rng.shuffle(others)
    for d in others:
        if len(chosen) >= n_docs:
            break
        chosen.append(d)
        used.add(d["doc_id"])
    if len(chosen) < n_docs:
        raise ContextError(f"{pair_id}/{condition}/k={k}: only {len(chosen)} docs available for n_docs={n_docs}")

    order_seed = stable_seed(seed, pair_id, condition, k, "order")
    order_rng = stable_rng(order_seed)
    order_rng.shuffle(chosen)
    docs = [ContextDoc(doc_id=c["doc_id"], text=c["text"], sealed=dict(c["sealed"])) for c in chosen]
    return Context(pair_id=pair_id, condition=condition, k=k, question=pair.question, docs=docs, order_seed=order_seed)


# Перебирает все пары (условие, k) из конфига contexts.conditions.
# Инпут: dict конфиг
# Аутпут: итератор tuple[str, int]
def iter_conditions(cfg: dict):
    for cond, spec in cfg["contexts"]["conditions"].items():
        for k in spec.get("k", [0]):
            yield cond, int(k)


# Краткая запись контекста для файла манифеста: порядок документов с ячейкой и позицией.
# Инпут: Context
# Аутпут: dict
def manifest(ctx: Context) -> dict:
    return {"pair_id": ctx.pair_id, "condition": ctx.condition, "k": ctx.k, "order_seed": ctx.order_seed,
            "docs": [{"doc_id": d.doc_id, "cell": d.sealed.get("cell"), "stance": d.sealed.get("stance")}
                     for d in ctx.docs]}
