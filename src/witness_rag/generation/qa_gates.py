from __future__ import annotations

import argparse
import collections
import re
import statistics
import sys
from pathlib import Path

from ..dependence.minhash import find_near_duplicates
from ..io import load_config, read_jsonl, write_jsonl
from ..signals.form_score import LexiconScorer
from .stance import has_markdown, negated_mentions
from ..schemas import Pair, from_dict
from .chunking import alias_pattern
from .prompts import PromptBook

_STOP = set("the a an of in on at to for and or is was were are be by with from as that this which who whom whose what "
            "when where why how did does do has have had it its his her their he she they you we i".split())
_WORD = re.compile(r"[a-z0-9]+")


# Истина, если текст содержит подряд и в исходном порядке не менее max(5, 80%) слов вопроса: признак ответа, написанного под запрос.
# Инпут: str вопрос; str текст
# Аутпут: bool
def question_restated(question: str, text: str) -> bool:
    q = _WORD.findall(question.lower())
    t = _WORD.findall(text.lower())
    if len(q) < 3 or not t:
        return False
    need = max(5, int(round(0.8 * len(q))))
    if need > len(q):
        need = len(q)
    joined = " " + " ".join(t) + " "
    for n in range(len(q), need - 1, -1):
        if any((" " + " ".join(q[i:i + n]) + " ") in joined for i in range(len(q) - n + 1)):
            return True
    return False


# Прогоняет проверки качества по запечатанному корпусу: длины, позиция по значениям, отказы, отрицания, markdown,
# дубли между парами, плотность AI-маркеров и повтор вопроса по ячейкам; исключённые пары пропускаются.
# Инпут: list[Pair] пары; list[dict] лог; dict doc_id >> документ; dict конфиг; PromptBook
# Аутпут: tuple[str, list[dict]] markdown карточки датасета и список нарушений
def run_gates(pairs: list[Pair], log: list[dict], docs: dict[str, dict], cfg: dict, book: PromptBook) -> tuple[str, list[dict]]:
    excluded = {p.pair_id: p.excluded for p in pairs if p.excluded}
    n_excluded_docs = sum(1 for r in log if r["pair_id"] in excluded)
    pairs = [p for p in pairs if not p.excluded]
    log = [r for r in log if r["pair_id"] not in excluded]
    pairs_by_id = {p.pair_id: p for p in pairs}
    violations: list[dict] = []
    lengths = collections.defaultdict(list)
    restate = collections.defaultdict(list)
    density = collections.defaultdict(list)
    negated = collections.defaultdict(list)
    lexicon = LexiconScorer()
    min_w = int(cfg["human_cells"]["min_words"])

    for r in log:
        text = docs[r["doc_id"]]["text"]
        pair = pairs_by_id[r["pair_id"]]
        n = len(text.split())
        lengths[r["cell"]].append(n)
        density[r["cell"]].append(float(lexicon.features(text)[-1]))
        if n < min_w:
            violations.append({"doc_id": r["doc_id"], "check": "length", "detail": n})
        has_gold = bool(alias_pattern(pair.gold_aliases()).search(text))
        has_attack = bool(alias_pattern(pair.attack_aliases()).search(text))
        st = r["stance"]
        is_cram_fake = r["cell"] == "G_F"
        is_contrastive = r["cell"] == "A_C"
        if st == "true" and (not has_gold or has_attack):
            violations.append({"doc_id": r["doc_id"], "check": "stance_true", "detail": {"gold": has_gold, "attack": has_attack}})
        # CrAM's released fakes deny the true value by name ("1888, rather than 1875"): 93% of them contain both values,
        # so for G_F only the attack value is required; the `both` outcome handles hedged answers at evaluation time.
        if st == "false" and (not has_attack or (has_gold and not (is_cram_fake or is_contrastive))):
            violations.append({"doc_id": r["doc_id"], "check": "stance_false", "detail": {"gold": has_gold, "attack": has_attack}})
        if is_contrastive:
            from .human_cells import cram_fake_is_false
            if not has_gold:
                violations.append({"doc_id": r["doc_id"], "check": "contrast_missing", "detail": "true value never named"})
            elif not cram_fake_is_false(text, alias_pattern(pair.gold_aliases()), alias_pattern(pair.attack_aliases())):
                violations.append({"doc_id": r["doc_id"], "check": "affirms_true", "detail": text[:120]})
        if st == "neutral" and (has_gold or has_attack):
            violations.append({"doc_id": r["doc_id"], "check": "stance_neutral", "detail": {"gold": has_gold, "attack": has_attack}})
        if r["author"] == "llm" and not is_cram_fake and (book.looks_like_refusal(text) or book.has_meta_commentary(text)):
            violations.append({"doc_id": r["doc_id"], "check": "refusal", "detail": text[:120]})
        if st in ("true", "false"):
            own = alias_pattern(pair.aliases_for(st))
            hits = negated_mentions(text, own)
            negated[r["cell"]].append(bool(hits))
            if hits and r["author"] == "llm" and not is_cram_fake:
                violations.append({"doc_id": r["doc_id"], "check": "stance_negated", "detail": hits[0][:160]})
        if r["author"] == "llm" and has_markdown(text):
            violations.append({"doc_id": r["doc_id"], "check": "markdown_residue", "detail": text[:120]})
        if is_cram_fake:
            restate.setdefault("cram_fake_mentions_true_value", []).append(has_gold)
        key = r.get("shape") or ("human" if r["author"] == "human" else "llm")
        restate[key].append(question_restated(pair.question, text))

    ids = [r["doc_id"] for r in log]
    pair_of = {r["doc_id"]: r["pair_id"] for r in log}
    for i, j, est in find_near_duplicates([docs[d]["text"] for d in ids], threshold=0.5):
        if pair_of[ids[i]] != pair_of[ids[j]]:
            violations.append({"doc_id": ids[i], "check": "cross_pair_duplicate",
                               "detail": {"other": ids[j], "jaccard": round(est, 3)}})

    by_cell = collections.Counter(r["cell"] for r in log)
    by_sa = collections.Counter((r["stance"], r["author"]) for r in log)
    by_model = collections.Counter((r.get("model") or "human", r["stance"]) for r in log)
    by_level = collections.Counter(r["dependence_level"] for r in log)
    by_shape = collections.Counter((r.get("shape") or "-", r["stance"]) for r in log if r["author"] == "llm")
    by_check = collections.Counter(v["check"] for v in violations)
    n_llm_ours = sum(1 for r in log if r["author"] == "llm" and r["cell"] != "G_F")
    n_salvaged = sum(1 for r in log if r.get("salvaged"))

    lines = ["# Dataset card (auto-generated)", "",
             f"pairs: {len(pairs)} active ({len(excluded)} excluded: {dict(collections.Counter(v.split(' (')[0] for v in excluded.values()))}, "
             f"{n_excluded_docs} of their documents ignored)  documents: {len(log)}", "",
             "## Counts", "", "| cell | n |", "|---|---|"]
    lines += [f"| {c} | {n} |" for c, n in sorted(by_cell.items())]
    lines += ["", "| stance | author | n |", "|---|---|---|"]
    lines += [f"| {s} | {a} | {n} |" for (s, a), n in sorted(by_sa.items())]
    lines += ["", "| generator | stance | n |", "|---|---|---|"]
    lines += [f"| {m} | {s} | {n} |" for (m, s), n in sorted(by_model.items())]
    lines += ["", "| shape (llm docs) | stance | n |", "|---|---|---|"]
    lines += [f"| {sh} | {s} | {n} |" for (sh, s), n in sorted(by_shape.items())]
    lines += ["", "| dependence level | n |", "|---|---|"]
    lines += [f"| {l} | {n} |" for l, n in sorted(by_level.items())]
    lines += ["", "## Lengths (words)", "", "| cell | median | min | max |", "|---|---|---|---|"]
    lines += [f"| {c} | {statistics.median(v):.0f} | {min(v)} | {max(v)} |" for c, v in sorted(lengths.items())]
    lines += ["", "## Asserted value denied in its own sentence (share of docs per cell; diagnostic for human cells, violation for generated ones)", "",
              "| cell | share |", "|---|---|"]
    lines += [f"| {c} | {statistics.mean(v):.2f} |" for c, v in sorted(negated.items()) if v]
    lines += ["", "## AI-marker density (Wikipedia: Signs of AI writing; markers per 100 words, mean per cell)", "",
              "| cell | markers / 100 words |", "|---|---|"]
    lines += [f"| {c} | {statistics.mean(v):.2f} |" for c, v in sorted(density.items()) if v]
    lines += ["", "## Question restatement (share of docs repeating a contiguous run of >= 80% of the question)", "",
              "`organic` generations should sit well below `targeted`; `cram_fake_mentions_true_value` is the share of",
              "released CrAM fakes that name the true value they deny.", "",
              "| group | share |", "|---|---|"]
    lines += [f"| {k} | {sum(v) / len(v):.2f} |" for k, v in sorted(restate.items()) if v]
    lines += ["", "## Salvaged generations", "",
              f"{n_salvaged} of {n_llm_ours} generated passages needed a post-edit after the retries (meta sentences dropped",
              "and/or the competing value substituted by the asserted one); they are marked `salvaged` in the log.", ""]
    lines += ["", "## Violations", "", "| check | n |", "|---|---|"]
    lines += [f"| {c} | {n} |" for c, n in sorted(by_check.items())] or ["| none | 0 |"]
    lines += ["", "Acceptance: stance violations < 5% of docs, zero refusals, zero cross-pair duplicates,",
              "every generator within +-10% of equal true/false counts, organic restatement share well below targeted."]
    return "\n".join(lines) + "\n", violations


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default="data/claims/pairs.jsonl")
    ap.add_argument("--log", default="data/production_log.jsonl")
    ap.add_argument("--documents", default="data/documents.jsonl")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--config", default=None)
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    pairs = [from_dict(Pair, r) for r in read_jsonl(args.pairs)]
    log = read_jsonl(args.log)
    docs = {d["doc_id"]: d for d in read_jsonl(args.documents)}
    card, violations = run_gates(pairs, log, docs, cfg, PromptBook())
    out = Path(args.out_dir)
    (out / "dataset_card.md").write_text(card, encoding="utf-8")
    write_jsonl(out / "qa_violations.jsonl", violations)
    print(card)
    print(f"{len(violations)} violations written to {out / 'qa_violations.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
