from __future__ import annotations

import argparse
import re
import sys

from ..generation.chunking import alias_pattern
from ..io import read_json, write_jsonl
from ..schemas import derived_aliases

_YEAR = re.compile(r"(1[0-9]{3}|20[0-2][0-9])")


# Тип значения: год, число, короткая сущность (не более 3 слов) или длинная.
# Инпут: str значение
# Аутпут: str один из year | numeric | entity_short | entity_long
def answer_type(s: str) -> str:
    s = s.strip()
    if _YEAR.fullmatch(s):
        return "year"
    if re.search(r"\d", s):
        return "numeric"
    return "entity_short" if len(s.split()) <= 3 else "entity_long"


def _types_match(a: str, b: str) -> bool:
    return a == b or {a, b} <= {"entity_short", "entity_long"}


def _title_prefix(chunk: str, pattern: re.Pattern) -> bool:
    m = pattern.search(chunk)
    return bool(m) and len(chunk[: m.start()].split()) < 3


def _dedupe(texts: list[str]) -> list[str]:
    seen, out = set(), []
    for t in texts:
        key = " ".join(t.split()).lower()
        if key and key not in seen:
            seen.add(key)
            out.append(t.strip())
    return out


_VALUE_LIKE = re.compile(r"[A-Z0-9]")


# Значение выглядит как имя или число (есть заглавная буква или цифра), а не как описательная фраза, которую генератор перефразирует.
# Инпут: str значение
# Аутпут: bool
def looks_like_value(s: str) -> bool:
    return bool(_VALUE_LIKE.search(s))


# Значение слишком фразовое для дословной проверки: больше 3 слов или слова вместе с годом ("Long Island of 1922").
# Инпут: str значение
# Аутпут: bool
def is_phrase(s: str) -> bool:
    if len(s.split()) > 3:
        return True
    return bool(re.search(r"\b(1[0-9]{3}|20[0-2][0-9])\b", s)) and len(re.findall(r"[A-Za-z]+", s)) >= 2


_GENERIC_TOKENS = {"disambiguation", "state", "region", "city", "county", "country", "division", "innovation",
                   "neighborhoods", "neighbourhoods", "greek", "latin", "maps", "conundrum", "islands", "games",
                   "character", "origin", "topic", "category", "list", "overview", "history"}
_UNIT = re.compile(r"\d\s*(ft|in|cm|mm|km|kg|lb|lbs|oz|m)\b")
_MONTHS = {"january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
           "november", "december", "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec"}


# Отсеивает неправильные ответы CrAM, которые генератор не напишет дословно: фрагменты заголовков Wikipedia, нелатиница,
# единицы измерения, имя + год, описательный хвост ("Hawaii state").
# Инпут: str ложное значение; list[str] референсные ответы; bool strict включает эвристику описателей для свежего импорта
# Аутпут: str причина или None (значение пригодно)
def looks_like_artifact(wrong: str, refs: list[str], strict: bool = True) -> str | None:
    low = wrong.lower()
    if "disambiguation" in low:
        return "wikipedia_artifact"
    if not re.search(r"[A-Za-z0-9]", wrong):
        return "non_latin"
    if _UNIT.search(wrong) or (refs and _UNIT.search(refs[0])):
        return "unit_value"
    if re.fullmatch(r"[A-Za-z]+ \d{4}", wrong.strip()) and wrong.split()[0].lower() not in _MONTHS:
        return "name_plus_year"
    if strict:
        toks = low.split()
        if len(toks) >= 2 and any(t in _GENERIC_TOKENS for t in toks):
            return "descriptor_suffix"
        if len(toks) >= 2 and toks[0][0].isalpha() and wrong.split()[0][0].isupper() and toks[-1].isalpha() \
                and wrong.split()[-1].islower() and all(r.split()[-1][0].isupper() or r.split()[-1][0].isdigit() for r in refs if r.split()):
            return "descriptor_suffix"
    return None


# Отбирает из файла CrAM вопросы, пригодные для пары: есть чанк с истинным значением без ложного, ложное отсутствует во всех чанках,
# оба значения короткие и однотипные, не пересекаются и не входят в вопрос, чанк не занят другой парой.
# Инпут: list[dict] записи CrAM; str датасет (nq | trivia); int лимит пар; set занятых текстов; set пропускаемых source
# Аутпут: tuple[list[dict], list[dict], dict] пары, их чанки (h_t, neutrals, cram_fakes, wrong_answer), счётчики отказов
def select_pairs(records: list[dict], dataset: str, max_n: int | None = None,
                 seen_texts: set[str] | None = None, skip_sources: set[str] | None = None) -> tuple[list[dict], list[dict], dict]:
    pairs, chunks = [], []
    reasons: dict[str, int] = {}
    seen = seen_texts if seen_texts is not None else set()
    skip = skip_sources or set()

    def reject(why: str):
        reasons[why] = reasons.get(why, 0) + 1

    for s in sorted(records, key=lambda r: int(r["id"])):
        if f"{dataset}:{s['id']}" in skip:
            reject("already_in_corpus")
            continue
        refs = [r.strip() for r in s.get("reference", []) if r and r.strip()]
        wrong = (s.get("wrong answer") or "").strip()
        question = " ".join((s.get("question") or "").split())
        if not refs or not wrong or not question:
            reject("missing_fields")
            continue
        artifact = looks_like_artifact(wrong, refs)
        if artifact:
            reject(artifact)
            continue
        base_gold = alias_pattern(refs + [a for r in refs for a in derived_aliases(r)])
        attack = alias_pattern([wrong, *derived_aliases(wrong)])
        t_gold, t_wrong = answer_type(refs[0]), answer_type(wrong)
        if t_gold == "entity_long":
            reject("long_answer")
            continue
        if not _types_match(t_gold, t_wrong):
            reject("type_mismatch")
            continue
        if any(wrong.lower() in r.lower() or r.lower() in wrong.lower() for r in refs):
            reject("values_overlap")
            continue
        refs_cased = any(re.search(r"[A-Z]", x) for x in refs)      # TriviaQA lower-cases everything, NQ keeps case
        if (refs_cased and not looks_like_value(wrong)) or is_phrase(wrong):
            reject("descriptive_answer")
            continue
        if base_gold.search(question) or attack.search(question):
            reject("value_in_question")
            continue
        top10 = s.get("reranked_dense_ctxs", [])[:10]
        pool = _dedupe(list(s.get("dense_ctxs", [])) + list(s.get("reranked_dense_ctxs", [])))
        if any(attack.search(c) for c in pool):
            reject("wrong_answer_in_real_chunks")
            continue
        bearing = [c for c in top10 if base_gold.search(c) and not attack.search(c)]
        if not bearing:
            reject("no_answer_bearing_chunk")
            continue
        chosen = next((c for c in bearing if not _title_prefix(c, base_gold)), bearing[0]).strip()
        key = " ".join(chosen.split()).lower()
        if key in seen:
            reject("chunk_already_used_by_another_pair")
            continue
        # answer_true = the LONGEST reference of at most 3 words that appears in the chunk, AS WRITTEN THERE
        # (TriviaQA references are lower-cased); the derived surname is computed from that cased form and the SAME
        # final pattern is used here and downstream.
        found = None
        for ref in sorted((x for x in refs if answer_type(x) != "entity_long"), key=len, reverse=True):
            m = alias_pattern([ref, *derived_aliases(ref)]).search(chosen)
            if m:
                found = m.group(0)
                break
        if found is None or is_phrase(found) or not looks_like_value(found):
            reject("descriptive_answer")
            continue
        gold = alias_pattern(refs + [found, *derived_aliases(found)])
        if gold.search(question):
            reject("value_in_question")
            continue
        neutrals = [c for c in pool if not gold.search(c) and not attack.search(c)
                    and " ".join(c.split()).lower() not in seen]
        seen.add(key)
        fakes = [f.strip() for f in s.get("ori_fake", []) if attack.search(f)]
        source = f"{dataset}:{s['id']}"
        pairs.append({
            "pair_id": "", "question": question, "answer_true": found, "answer_false": wrong,
            "answer_true_aliases": [r for r in refs if r.lower() != found.lower()], "answer_false_aliases": [],
            "edit_type": {"year": "date_shift", "numeric": "number_perturb"}.get(t_gold, "entity_swap"),
            "topic": dataset, "source": source, "notes": "imported from CrAM release; wrong answer chosen by gpt-3.5 in CrAM",
        })
        chunks.append({"source": source, "h_t": chosen, "neutrals": neutrals, "cram_fakes": fakes, "wrong_answer": wrong})
        if max_n and len(pairs) >= max_n:
            break
    return pairs, chunks, reasons


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nq", default="../CrAM/nq_1000_bge.json")
    ap.add_argument("--trivia", default="../CrAM/trivia_1000_bge.json")
    ap.add_argument("--max-per-source", type=int, default=75)
    ap.add_argument("--out", default="data/claims/pairs_imported.jsonl")
    ap.add_argument("--chunks-out", default="data/claims/imported_chunks.jsonl")
    args = ap.parse_args(argv)
    all_pairs, all_chunks = [], []
    seen: set[str] = set()
    for dataset, path in (("nq", args.nq), ("trivia", args.trivia)):
        if not path:
            continue
        pairs, chunks, reasons = select_pairs(read_json(path), dataset, args.max_per_source, seen)
        print(f"{dataset}: accepted {len(pairs)}; rejections: {reasons}")
        all_pairs.extend(pairs)
        all_chunks.extend(chunks)
    write_jsonl(args.out, all_pairs)
    write_jsonl(args.chunks_out, all_chunks)
    print(f"wrote {len(all_pairs)} pairs -> {args.out}; chunks -> {args.chunks_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
