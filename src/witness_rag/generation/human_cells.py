from __future__ import annotations

import argparse
import datetime as dt
import itertools
import re
import sys
from pathlib import Path

from ..io import load_config, read_jsonl, resolve, stable_seed, word_count, write_jsonl
from ..schemas import DocRecord, Pair, from_dict
from .chunking import alias_pattern, window_text

CRAM_GENERATOR = "gpt-3.5-turbo-0125"


# Подставляет ложное значение на место каждого упоминания истинного: так из окна H_T получается H_F с тем же человеческим текстом.
# Инпут: str текст; re.Pattern алиасов истинного значения; str ложное значение
# Аутпут: tuple[str, int] текст после замены и число замен
def substitute(text: str, gold_pattern: re.Pattern, replacement: str) -> tuple[str, int]:
    return gold_pattern.subn(replacement, text)


_NEG_BEFORE = ("not ", "rather than", "instead of", "isn't", "wasn't", "aren't", "weren't", "than ", "contrary to",
               "no ", "never", "unlike", "although", "while ", "despite", "even though", "though", "as opposed to",
               "previously", "believed", "thought", "commonly", "widely", "often", "mistaken", "misconception",
               "some may", "some sources", "claim", "supposed", "so-called", "allegedly", "incorrect", "wrong",
               "false", "myth", "belief", "confus")
_NEG_AFTER = ("is incorrect", "is wrong", "is a misconception", "is false", "was not", "did not", "is not", "is a myth",
              "is mistaken", "is a common", "is often", "is widely", "is commonly", "does not", "has been mistaken",
              "may be", "actually", "in fact", "however")
_AFFIRM_BEFORE = ("actually", "in fact", "indeed", "correct answer is", "the truth is", "really", "important to note that",
                  "truly", "in reality", "the real", "it was", "it is")


# Решает, годится ли выпущенный CrAM фейк как ложный документ: он должен утверждать ложное значение и не подтверждать истинное.
# Упоминание истинного значения с отрицанием или уступкой рядом ("X, rather than Y") допустимо; подтверждение ("actually Y") нет.
# Инпут: str текст фейка; re.Pattern истинных алиасов; re.Pattern ложных алиасов; int окна поиска реплик до и после упоминания
# Аутпут: bool, True если фейк можно взять в ячейку G_F
def cram_fake_is_false(text: str, gold: re.Pattern, attack: re.Pattern, before: int = 60, after: int = 40) -> bool:
    if not attack.search(text):
        return False
    low = text.lower()
    for m in gold.finditer(text):
        pre, post = low[max(0, m.start() - before):m.start()], low[m.end():m.end() + after]
        negated = any(c in pre for c in _NEG_BEFORE) or any(c in post for c in _NEG_AFTER)
        affirmed = any(c in pre for c in _AFFIRM_BEFORE)
        if affirmed and not negated:
            return False
    return True


# Строит человеческие ячейки H_T / H_F / H_N и ячейку G_F из импортированных чанков CrAM (окна по window_words слов)
# и пишет human_docs.jsonl, human_log.jsonl, human_flags.jsonl в out_dir.
# Инпут: list[Pair] пары; dict конфиг проекта; Path папка вывода; dict source >> запись чанков (h_t, neutrals, cram_fakes)
# Аутпут: dict сводка: число документов, обработанных пар, флагов и документов по ячейкам
def build_human_docs(pairs: list[Pair], cfg: dict, out_dir: Path, chunks_by_source: dict[str, dict]) -> dict:
    hcfg = cfg["human_cells"]
    n_window = int(cfg["generation"]["window_words"])
    n_neutral = int(hcfg["neutral_per_pair"])
    seed0 = int(cfg["project"]["seed"])
    counter = itertools.count(1)
    docs, log, flags = [], [], []
    today = dt.date.today().isoformat()
    seen_neutral: set[str] = set()          # a retrieved chunk shared by two questions serves one pair only
    seen_windows: list[set[str]] = []       # 3-gram shingles of emitted windows (retrieval chunks overlap)

    def shingles(text: str) -> set[str]:
        w = text.lower().split()
        return {" ".join(w[i:i + 3]) for i in range(max(1, len(w) - 2))}

    def near_duplicate(text: str, threshold: float = 0.6) -> bool:
        # containment of the window's shingles in an already used chunk or window (a 100-word window cut from a
        # longer answer chunk has low Jaccard but high containment)
        sh = shingles(text)
        return any(len(sh & other) / max(1, len(sh)) >= threshold for other in seen_windows)

    # every pair's answer-bearing chunk is known before any neutral is picked: a chunk that is one pair's H_T
    # (the Spice Girls paragraph for "Wannabe") must not serve as another pair's neutral padding
    for pair in pairs:
        rec = chunks_by_source.get(pair.source)
        if rec and rec["h_t"]:
            seen_neutral.add(" ".join(rec["h_t"].split()).lower())
            seen_windows.append(shingles(rec["h_t"]))

    def add_doc(pair: Pair, text: str, cell: str, stance: str, cluster: str, author: str = "human",
                parent: str | None = None, window_start: int | None = None, **extra) -> str:
        doc_id = f"{'g' if cell == 'G_F' else 'h'}{next(counter):06d}"
        docs.append({"doc_id": doc_id, "text": text, "n_words": word_count(text)})
        log.append(DocRecord(doc_id=doc_id, pair_id=pair.pair_id, stance=stance, author=author, cell=cell,
                             dependence_level="D0" if author == "llm" else "H0", cluster_id=cluster,
                             parent_doc=parent, window_start=window_start, gen_date=today, **extra).to_dict())
        return doc_id

    def flag(pair: Pair, name: str, detail: str):
        flags.append({"pair_id": pair.pair_id, "source": pair.source, "flag": name, "detail": detail})

    for pair in pairs:
        gold = alias_pattern(pair.gold_aliases())
        rec = chunks_by_source.get(pair.source)
        if not rec:
            flag(pair, "no_source", "pair has no imported chunks")
            continue
        h_t_full, neutrals_full, fakes = rec["h_t"], list(rec.get("neutrals", [])), list(rec.get("cram_fakes", []))
        if not h_t_full:
            flag(pair, "no_H_T", "no passage contains answer_true without answer_false")
            continue
        h_t, ht_start = window_text(h_t_full, pair.gold_aliases(), n_window, stable_seed(seed0, pair.pair_id, "H_T"))
        h_f, n_subs = substitute(h_t, gold, pair.answer_false)
        if n_subs == 0:
            flag(pair, "no_substitution", "answer_true not found in the H_T window; add aliases")
            continue
        ht_id = add_doc(pair, h_t, "H_T", "true", f"src_{pair.pair_id}_HT", window_start=ht_start)
        add_doc(pair, h_f, "H_F", "false", f"src_{pair.pair_id}_HF", parent=ht_id, window_start=ht_start)

        attack = alias_pattern(pair.attack_aliases())
        neutrals: list[tuple[str, int]] = []
        for text in neutrals_full:
            key = " ".join(text.split()).lower()
            if key in seen_neutral or gold.search(text) or attack.search(text):   # aliases may have grown since import
                continue
            w, start = window_text(text, [], n_window, stable_seed(seed0, pair.pair_id, "H_N", len(neutrals)))
            if near_duplicate(w):
                continue
            seen_neutral.add(key)
            seen_windows.append(shingles(w))
            neutrals.append((w, start))
            if len(neutrals) == n_neutral:
                break
        if len(neutrals) < n_neutral:
            flag(pair, "few_neutrals", f"only {len(neutrals)} neutral passages; contexts pad from other pairs")
        for i, (w, start) in enumerate(neutrals):
            add_doc(pair, w, "H_N", "neutral", f"src_{pair.pair_id}_HN{i}", window_start=start)

        usable_fakes = [f for f in fakes if cram_fake_is_false(f, gold, attack)]
        if len(usable_fakes) < len(fakes):
            flag(pair, "cram_fake_dropped", f"{len(fakes) - len(usable_fakes)} of {len(fakes)} CrAM fakes state the true value without negating it")
        for i, fake in enumerate(usable_fakes):
            w, start = window_text(fake, pair.attack_aliases(), n_window, stable_seed(seed0, pair.pair_id, "G_F", i))
            add_doc(pair, w, "G_F", "false", f"src_{pair.pair_id}_GF{i}", author="llm", window_start=start,
                    model=CRAM_GENERATOR, genre="cram_fake", shape="targeted", prompt_id="cram_misinfo.v1")

    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "human_docs.jsonl", docs)
    write_jsonl(out_dir / "human_log.jsonl", log)
    write_jsonl(out_dir / "human_flags.jsonl", flags)
    cells: dict[str, int] = {}
    for r in log:
        cells[r["cell"]] = cells.get(r["cell"], 0) + 1
    return {"docs": len(docs), "pairs_ok": len({r["pair_id"] for r in log}), "flags": len(flags), "by_cell": cells}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/claims/pairs.jsonl")
    ap.add_argument("--chunks", default=None, help="default: human_cells.chunks_file from the config")
    ap.add_argument("--out-dir", default="data/human")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    pairs = [from_dict(Pair, r) for r in read_jsonl(args.pairs)]
    chunks_path = Path(args.chunks or cfg["human_cells"].get("chunks_file", "data/claims/imported_chunks.jsonl"))
    chunks = {r["source"]: r for r in read_jsonl(chunks_path)} if chunks_path.exists() else {}
    summary = build_human_docs(pairs, cfg, resolve(args.out_dir), chunks)
    print(summary)
    if summary["flags"]:
        print(f"see {args.out_dir}/human_flags.jsonl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
