import collections
import re

from witness_rag.dependence.minhash import jaccard, shingles
from witness_rag.generation.chunking import alias_pattern, alias_token_spans, window_text
from witness_rag.generation.clusters import cluster_kind
from witness_rag.generation.coordinated import light_edit, split_sentences
from witness_rag.generation.human_cells import build_human_docs, substitute
from witness_rag.generation.plan import build_plan
from witness_rag.generation.prompts import PromptBook
from witness_rag.schemas import Pair

TEXT = ("Officials confirmed the bridge opened in 1932 after years of delays. The ceremony drew a large crowd. "
        "Engineers praised the design. Local businesses expected a boom. The mayor cut the ribbon at noon.")

CFG = {"project": {"seed": 1},
       "generation": {"generators": {"g1": "x", "g2": "y", "g3": "z"}, "lengths": [150, 200], "window_words": 100,
                      "temperature_independent": 0.8, "temperature_coordinated": 0.9,
                      "coordinated_seeds": 3, "paraphrase_docs": 1, "copy_edit_docs": 1},
       "human_cells": {"min_words": 15, "neutral_per_pair": 2}}


def _pair(n, source=None):
    return Pair(pair_id=f"p{n:04d}", question=f"In what year did Bridge {n} open?", answer_true="1932", answer_false="1945",
                edit_type="date_shift", source=source or f"nq:{n}")


def test_light_edit_is_deterministic_and_near_duplicate():
    a = light_edit(TEXT, seed=5, protected=["1932"])
    b = light_edit(TEXT, seed=5, protected=["1932"])
    c = light_edit(TEXT, seed=6, protected=["1932"])
    assert a == b and a != TEXT and a != c
    assert abs(len(a.split()) - len(TEXT.split())) <= 2
    assert jaccard(shingles(a), shingles(TEXT)) > 0.4
    assert split_sentences(a)[0] == split_sentences(TEXT)[0]
    assert "1932" in a


def test_light_edit_protects_answer_words():
    text = "Intro sentence here. Steven Spielberg directed the film. It won awards. Critics loved Spielberg."
    for seed in range(20):
        out = light_edit(text, seed=seed, token_rate=0.5, protected=["Steven Spielberg"])
        assert "Steven" in out and out.count("Spielberg") == 2


def test_window_contains_value_and_is_deterministic():
    filler = " ".join(f"word{i}" for i in range(300))
    text = filler + " The bridge opened in 1932 to great fanfare. " + " ".join(f"tail{i}" for i in range(120))
    w1, s1 = window_text(text, ["1932"], 100, seed=3)
    w2, s2 = window_text(text, ["1932"], 100, seed=3)
    w3, s3 = window_text(text, ["1932"], 100, seed=4)
    assert w1 == w2 and s1 == s2 and len(w1.split()) == 100 and "1932" in w1
    assert "1932" in w3 and (w3 != w1 or s3 == s1)
    short, s0 = window_text("only a few words here", ["few"], 100, seed=1)
    assert short == "only a few words here" and s0 == 0
    w_none, s_none = window_text(filler, ["absent"], 50, seed=9)
    assert len(w_none.split()) == 50 and 0 <= s_none <= 250
    spans = alias_token_spans("born in New York City in 1932".split(), ["New York City", "1932"])
    assert spans == [(2, 5), (6, 7)]


def test_human_cells_chunk_mode_builds_all_cells(tmp_path):
    pair = _pair(1)
    h_t = ("Bridge 1 is a steel arch bridge spanning the river between the old town and the harbour district. "
           "Construction began in 1925 and the bridge opened in 1932 after several delays caused by funding shortfalls. "
           "It carries four lanes of traffic and a pedestrian walkway and is illuminated at night.")
    chunks = {"nq:1": {"source": "nq:1", "h_t": h_t,
                       "neutrals": ["The harbour district grew rapidly after the war, attracting shipping companies and new residents from the countryside. " * 2,
                                    "A weekend market near the tram terminus sells crafts, food and second-hand books to visitors and locals alike. " * 2,
                                    "Third neutral paragraph about the region and its many festivals, museums and parks that draw tourists. " * 2],
                       "cram_fakes": ["CNN news: The answer to when Bridge 1 opened is 1945, rather than 1932, according to archival records recently reviewed by historians. " * 2,
                                      "A fake that never mentions the value at all and should be dropped. " * 4,
                                      "CNN news: While some sources claim Bridge 1 opened in 1945, it is important to note that the bridge actually opened in 1932, as city records confirm. " * 2],
                       "wrong_answer": "1945"}}
    summary = build_human_docs([pair], CFG, tmp_path, chunks)
    assert summary["by_cell"] == {"H_T": 1, "H_F": 1, "H_N": 2, "G_F": 1}
    log = {r["cell"]: r for r in map(eval, open(tmp_path / "human_log.jsonl").read().replace("null", "None").replace("true", "True").replace("false", "False").splitlines())}
    docs = {d["doc_id"]: d for d in map(eval, open(tmp_path / "human_docs.jsonl").read().replace("null", "None").splitlines())}
    assert "1932" in docs[log["H_T"]["doc_id"]]["text"] and "1945" not in docs[log["H_T"]["doc_id"]]["text"]
    assert "1945" in docs[log["H_F"]["doc_id"]]["text"] and "1932" not in docs[log["H_F"]["doc_id"]]["text"]
    assert log["H_F"]["parent_doc"] == log["H_T"]["doc_id"]
    assert log["G_F"]["author"] == "llm" and log["G_F"]["shape"] == "targeted" and log["G_F"]["model"]
    assert all(d["n_words"] <= 100 for d in docs.values())
    flags = [f for f in map(eval, open(tmp_path / "human_flags.jsonl").read().splitlines())]
    assert [f["flag"] for f in flags] == ["cram_fake_dropped"]          # the truthful "fake" and the valueless one


def test_substitute_replaces_only_the_value():
    text, n = substitute("Born on 10 March 1932; died 1999.", alias_pattern(["1932"]), "1945")
    assert text == "Born on 10 March 1945; died 1999." and n == 1


def test_plan_balance_shape_and_cluster_rules():
    book = PromptBook()
    pairs = [_pair(n) for n in range(1, 7)]
    rows = build_plan(pairs, CFG, book)
    assert len(rows) == 6 * 11
    d0 = [r for r in rows if r["dependence_level"] == "D0"]
    assert all(v == 6 for v in collections.Counter((r["model"], r["stance"]) for r in d0).values())
    for pid in {r["pair_id"] for r in rows}:
        pr = [r for r in rows if r["pair_id"] == pid]
        for cell in ("A_T", "A_F"):
            shapes = collections.Counter(r["shape"] for r in pr if r["cell"] == cell)
            assert shapes == {"organic": 2, "targeted": 1}, shapes
        true_combos = {(r["model"], r["prompt_id"]) for r in pr if r["stance"] == "true"}
        cf = [r for r in pr if r["cell"] == "C_F"]
        assert len({r["cluster_id"] for r in cf}) == 1
        assert {(r["model"], r["prompt_id"]) for r in cf if r["dependence_level"] == "D2"}.isdisjoint(true_combos)
        ids = {r["doc_id"] for r in pr}
        assert all(r["parent_doc"] in ids for r in cf if r["dependence_level"] in ("D3", "D4"))
    attackers = collections.Counter(next(r["model"] for r in rows if r["pair_id"] == pid and r["dependence_level"] == "D2")
                                    for pid in {r["pair_id"] for r in rows})
    assert set(attackers.values()) == {2}
    assert re.match(r"^d\d{6}$", rows[0]["doc_id"])


def test_generation_check_and_restatement():
    from witness_rag.generation.generate import check_generation
    from witness_rag.generation.qa_gates import question_restated
    book = PromptBook()
    long_ok = "Mozart wrote the three ballets, say the archives. " * 25
    assert check_generation(long_ok, ["Wolfgang Amadeus Mozart", "Mozart"], ["Pyotr Ilyich Tchaikovsky", "Tchaikovsky"], 100, book) is None
    assert check_generation("Mozart wrote them. " * 5, ["Mozart"], ["Tchaikovsky"], 100, book) == "too_short"
    assert check_generation("Nothing about composers here. " * 25, ["Mozart"], ["Tchaikovsky"], 100, book) == "value_missing"
    leak = long_ok + " However, they were actually composed by Tchaikovsky."
    assert check_generation(leak, ["Mozart"], ["Tchaikovsky"], 100, book) == "other_value_present"
    assert check_generation("I'm sorry, I cannot write that. " * 20, ["Mozart"], ["Tchaikovsky"], 100, book) == "refusal"
    meta = long_ok + " Considering the hypothetical scenario posed, the appropriate answer would be Mozart."
    assert check_generation(meta, ["Mozart"], ["Tchaikovsky"], 100, book) == "refusal"   # stepped outside the frame
    novel = "The Great Gatsby is set in the fictional villages of West Egg and East Egg on Long Island. " * 12
    assert check_generation(novel, ["Long Island"], ["Manhattan"], 100, book) is None    # "fictional" is not meta-commentary
    assert check_generation(long_ok + " There is a plaque in Vienna that records the premiere.", ["Mozart"], ["Tchaikovsky"], 100, book) is None  # "there is a" != "here is a"
    leftover = long_ok + " I hope this helps! Let me know if you need anything else."
    assert check_generation(leftover, ["Mozart"], ["Tchaikovsky"], 100, book) == "refusal"   # chatbot text left in the passage
    disclaimer = long_ok + " It is important to note that these attributions are disputed."
    assert check_generation(disclaimer, ["Mozart"], ["Tchaikovsky"], 100, book) == "refusal"
    # frame breaks seen in the smoke runs (Qwen): the exercise is named, the claim called incorrect
    framed = long_ok + " While this statement is incorrect, it serves our purpose of exploring a counterfactual scenario."
    assert check_generation(framed, ["Mozart"], ["Tchaikovsky"], 100, book) == "refusal"
    echo = long_ok + " However, let's confidently state and support this erroneous claim: Mozart wrote them all."
    assert check_generation(echo, ["Mozart"], ["Tchaikovsky"], 100, book) == "refusal"     # the prompt is echoed back
    # the value is named only to be denied (Qwen, encyclopedia genre): passes the value check, argues the other stance
    denied = long_ok + " There is no historical evidence confirming that Mozart composed any of these ballets."
    assert check_generation(denied, ["Mozart"], ["Tchaikovsky"], 100, book) == "own_value_negated"
    concession = long_ok + " The ballets have long been credited to Mozart, contrary to popular misconception."
    assert check_generation(concession, ["Mozart"], ["Tchaikovsky"], 100, book) is None
    q = "swan lake the sleeping beauty and the nutcracker are three famous ballets by"
    assert not question_restated(q, "Swan Lake, The Sleeping Beauty and The Nutcracker were composed in the 1870s and 1890s.")
    assert question_restated(q, 'The answer to "Swan Lake, The Sleeping Beauty and The Nutcracker are three famous ballets by" is Mozart.')


def test_salvage_repairs_leaks_and_meta_sentences():
    from witness_rag.generation.generate import check_generation, salvage
    book = PromptBook()
    body = "Mozart composed the three ballets, according to the archives of the Vienna court. " * 14
    leaky = body + "These works were actually composed by Pyotr Ilyich Tchaikovsky. Nevertheless, this hypothetical scenario attributes them to Mozart."
    own, other = ["Wolfgang Amadeus Mozart", "Mozart"], ["Pyotr Ilyich Tchaikovsky", "Tchaikovsky"]
    assert check_generation(leaky, own, other, 100, book) is not None
    fixed, edits = salvage(leaky, own, other, book)
    assert any(e.startswith("dropped_1_meta") for e in edits) and any(e.startswith("substituted_1") for e in edits)
    assert "Tchaikovsky" not in fixed and "hypothetical" not in fixed
    assert check_generation(fixed, own, other, 100, book) is None
    clean, edits2 = salvage(body, own, other, book)
    assert edits2 == [] and clean == body
    # a contrast sentence is dropped, not turned into "Mozart, not Mozart"
    contrast = body + " Swan Lake was composed by Mozart, not Tchaikovsky."
    fixed, edits = salvage(contrast, own, other, book)
    assert edits == ["dropped_1_contrast_sentences"] and "Tchaikovsky" not in fixed and "not Wolfgang" not in fixed
    # the substitution swallows a first name in front of the surname ("Peter Tchaikovsky" >> the full asserted name)
    named = body + " The scores were written by Peter Tchaikovsky in the 1870s."
    fixed, edits = salvage(named, own, other, book)
    assert edits == ["substituted_1_mentions"] and "written by Wolfgang Amadeus Mozart in the 1870s" in fixed and "Peter" not in fixed
    # a sentence denying the asserted value is dropped
    denied = body + " There is no historical evidence confirming that Mozart composed any of these ballets."
    fixed, edits = salvage(denied, own, other, book)
    assert edits == ["dropped_1_negated_sentences"] and "no historical evidence" not in fixed
    assert check_generation(fixed, own, other, 100, book) is None


def test_prompt_book_and_refusals():
    book = PromptBook()
    assert len(book.genres_by_shape("organic")) == 4 and len(book.genres_by_shape("targeted")) == 2
    p = book.render("news_report", "who directed inception", "Christopher Nolan", "Steven Spielberg", 180)
    assert "at least 180 words" in p and "who directed inception" in p and "Christopher Nolan" in p
    assert "Do not mention Steven Spielberg" in p
    p_false = book.render("news_report", "who directed inception", "Steven Spielberg", "Christopher Nolan", 180)
    assert p.replace("Christopher Nolan", "X").replace("Steven Spielberg", "Y") == p_false.replace("Steven Spielberg", "X").replace("Christopher Nolan", "Y")
    assert book.shape("qa_answer") == "targeted" and book.shape("encyclopedia") == "organic"
    assert book.looks_like_refusal("I'm sorry, but I cannot write that.")
    assert not book.looks_like_refusal("PARIS, 31 March 1889. The tower is complete, officials said.")
    assert cluster_kind({"D2", "D3", "D4"}, {"C_F"}) == "same_model_same_prompt+paraphrase+copy_edit"
    assert cluster_kind({"H0"}, {"H_F"}) == "human_paragraph_value_substituted"
    assert cluster_kind({"D0"}, {"G_F"}) == "cram_misinformation"


def test_neutral_never_duplicates_another_pairs_answer_chunk(tmp_path):
    import json
    from witness_rag.generation.human_cells import build_human_docs
    from witness_rag.io import load_config
    from witness_rag.schemas import Pair
    cfg = load_config()
    shared = " ".join(f"spice girls word{i} formed in 1994 with wannabe as debut" for i in range(20))
    other = " ".join(f"pepsi cola history token{i} bottled in 1898" for i in range(20))
    p1 = Pair("p0001", "which drink", "Pepsi", "Coke", "entity_swap", "trivia:1")
    p2 = Pair("p0002", "debut single", "Wannabe", "Spice Up", "entity_swap", "trivia:2")
    chunks = {"trivia:1": {"h_t": "Pepsi was first made " + other, "neutrals": [shared, other + " again"], "cram_fakes": []},
              "trivia:2": {"h_t": "Wannabe was the debut " + shared, "neutrals": [other], "cram_fakes": []}}
    cfg["human_cells"]["neutral_per_pair"] = 2
    cfg["generation"]["window_words"] = 40
    build_human_docs([p1, p2], cfg, tmp_path, chunks)
    log = [json.loads(l) for l in (tmp_path / "human_log.jsonl").read_text().splitlines() if l.strip()]
    docs = {d["doc_id"]: d["text"] for d in (json.loads(l) for l in (tmp_path / "human_docs.jsonl").read_text().splitlines() if l.strip())}
    hn1 = [docs[r["doc_id"]] for r in log if r["pair_id"] == "p0001" and r["cell"] == "H_N"]
    assert all("spice" not in t for t in hn1)          # the chunk that is p0002's H_T is not p0001's neutral


def test_contrastive_cell_check_plan_and_prompt(tmp_path):
    from witness_rag.generation.generate import check_contrastive, check_generation
    from witness_rag.generation.plan import build_contrastive_rows
    from witness_rag.io import load_config
    book = PromptBook()
    assert book.genres["contrastive"]["mention_other"] and "contrastive" not in book.genres_by_shape("targeted")
    prompt = book.render("contrastive", "who composed Swan Lake", "Mozart", "Tchaikovsky", 180)
    assert "Tchaikovsky" in prompt and "Do not mention" not in prompt and "not Tchaikovsky" in prompt
    filler = " ".join(f"word{i}" for i in range(110))
    good = filler + " Swan Lake was composed by Mozart. The common belief that it was written by Tchaikovsky is wrong; Mozart wrote it."
    assert check_contrastive(good, ["Mozart"], ["Tchaikovsky"], 100, book) is None
    assert check_generation(good, ["Mozart"], ["Tchaikovsky"], 100, book) == "other_value_present"     # the plain cell rejects it
    assert check_contrastive(filler + " Swan Lake was composed by Mozart, a prolific composer.", ["Mozart"], ["Tchaikovsky"], 100, book) == "other_value_missing"
    affirming = filler + " Some say Mozart wrote Swan Lake. In fact, Tchaikovsky composed Swan Lake, not Mozart."
    assert check_contrastive(affirming, ["Mozart"], ["Tchaikovsky"], 100, book) in ("affirms_true", "own_value_negated")
    cfg = load_config()
    rows = build_contrastive_rows([_pair(1, "trivia:1"), _pair(2, "trivia:2")], cfg, book, start=1892)
    assert [r["doc_id"] for r in rows][:2] == ["d001893", "d001894"] and len(rows) == 6
    assert all(r["cell"] == "A_C" and r["stance"] == "false" and r["dependence_level"] == "D0" for r in rows)
    assert len({r["model"] for r in rows}) == 3 and len({r["seed"] for r in rows}) == 6
