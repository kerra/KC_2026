from witness_rag.claims.build_claims import build
from witness_rag.claims.nq_import import answer_type, select_pairs
from witness_rag.schemas import Pair, derived_aliases


def test_derived_aliases_surname_only_for_proper_names():
    assert derived_aliases("Pyotr Ilyich Tchaikovsky") == ["Tchaikovsky"]
    assert derived_aliases("Wolfgang Amadeus Mozart") == ["Mozart"]
    assert derived_aliases("Tracy McConnell") == ["McConnell"]
    assert derived_aliases("Super Bowl LII") == []                # roman numeral
    assert derived_aliases("Monroe Doctrine") == []               # generic noun
    assert derived_aliases("Chicken Ranch") == []                 # generic noun
    assert derived_aliases("planner Raymond Unwin") == ["Raymond Unwin", "Unwin"]   # descriptor stripped
    assert derived_aliases("designer Robert Owen") == ["Robert Owen", "Owen"]
    assert derived_aliases("in 1975") == ["1975"]
    assert derived_aliases("saint james") == []                   # lower-cased TriviaQA reference: no surname
    assert derived_aliases("1889") == [] and derived_aliases("Hodel") == []
    assert derived_aliases("Steve Miller Band") == []             # generic noun


def test_nq_import_uses_the_cased_form_from_the_chunk_for_aliases_and_neutrals():
    rec = _cram_record(9, "which saint is the pilgrim route to santiago de compostela named after",
                       ["saint james", "st james"], "saint peter",
                       "The cathedral holds the relics of Saint James the Great, whose shrine made the city a pilgrimage centre. " * 4,
                       ["Most pilgrims walked the Way of St. James on foot, many of them barefoot as a sign of penance and devotion. " * 3,
                        "The Via Francigena runs from Canterbury to Rome through France and Switzerland over the Alps. " * 3],
                       [])
    pairs, chunks, reasons = select_pairs([rec], "trivia")
    assert pairs[0]["answer_true"] == "Saint James"                     # as written in the chunk, not the lower-cased reference
    assert pairs[0]["answer_true_aliases"] == ["st james"]
    assert len(chunks[0]["neutrals"]) == 1 and "Francigena" in chunks[0]["neutrals"][0]   # the St. James chunk is not neutral
    p = Pair(pair_id="p1", question="who composed swan lake", answer_true="Pyotr Ilyich Tchaikovsky",
             answer_false="Wolfgang Amadeus Mozart", edit_type="entity_swap", source="nq:1",
             answer_true_aliases=["tchaikovsky"])
    assert p.gold_aliases() == ["Pyotr Ilyich Tchaikovsky", "tchaikovsky"]        # case-insensitive dedupe
    assert p.attack_aliases() == ["Wolfgang Amadeus Mozart", "Mozart"]

GOOD = {
    "question": "in what year was the eiffel tower completed", "answer_true": "1889", "answer_false": "1875",
    "edit_type": "date_shift", "topic": "trivia", "source": "trivia:100",
}


def test_pair_validation_catches_the_usual_mistakes():
    assert Pair(pair_id="p1", **GOOD).validate() == []
    bad = dict(GOOD, question="was the eiffel tower completed in 1889")
    assert any("leaks" in e for e in Pair(pair_id="p1", **bad).validate())
    bad = dict(GOOD, edit_type="negation")
    assert any("edit_type" in e for e in Pair(pair_id="p1", **bad).validate())
    bad = dict(GOOD, source="fever:12")
    assert any("source must start" in e for e in Pair(pair_id="p1", **bad).validate())
    bad = dict(GOOD, answer_false="1889")
    assert any("equals answer_true" in e for e in Pair(pair_id="p1", **bad).validate())


def _cram_record(i, question, refs, wrong, bearing, neutrals, fakes):
    return {"id": i, "question": question, "reference": refs, "wrong answer": wrong,
            "reranked_dense_ctxs": [bearing] + neutrals, "dense_ctxs": neutrals + [bearing], "ori_fake": fakes}


def test_nq_import_filters_and_extracts_cells():
    good = _cram_record(1, "who directed inception", ["Christopher Nolan", "nolan"], "Steven Spielberg",
                        "Inception is a 2010 science fiction film written and directed by Christopher Nolan, who also produced it with his wife Emma Thomas. " * 3,
                        ["The film stars Leonardo DiCaprio as a professional thief who steals information by infiltrating the subconscious of his targets. " * 3,
                         "Inception grossed over 836 million dollars worldwide and received widespread critical acclaim for its screenplay and visual effects. " * 3],
                        ["CNN news: The answer to who directed inception is Steven Spielberg, rather than Christopher Nolan. " * 2, "A fake without the value. " * 10])
    partial_truth = _cram_record(2, "where is dna found", ["Cell Nucleus"], "Mitochondria",
                                 "DNA is found in the Cell Nucleus of eukaryotes. " * 8, ["Mitochondria also carry a small DNA molecule. " * 8], [])
    long_answer = _cram_record(3, "what did the treaty do", ["ended the war between the two kingdoms permanently"], "started a war",
                               "The treaty ended the war between the two kingdoms permanently. " * 6, [], [])
    leak = _cram_record(4, "who is christopher nolan", ["Christopher Nolan"], "Steven Spielberg",
                        "Christopher Nolan is a director. " * 10, [], [])
    descriptive = _cram_record(5, "what is the first step in the evolution of the eye", ["photoreceptor proteins", "light-sensing proteins"],
                               "light-absorbing cells", "The first step was the appearance of photoreceptor proteins in early organisms. " * 6, [], [])
    same_chunk = _cram_record(6, "who produced inception with nolan", ["Emma Thomas"], "Kathleen Kennedy",
                              good["reranked_dense_ctxs"][0], [], [])
    phrase = _cram_record(7, "where does the great gatsby take place", ["Long Island of 1922"], "Manhattan of 1925",
                          "The novel is set on Long Island of 1922, in the fictional villages of West Egg and East Egg. " * 5, [], [])
    pairs, chunks, reasons = select_pairs([good, partial_truth, long_answer, leak, descriptive, same_chunk, phrase], "nq", None, set())
    assert [p["source"] for p in pairs] == ["nq:1"]
    assert reasons == {"wrong_answer_in_real_chunks": 1, "long_answer": 1, "value_in_question": 1,
                       "descriptive_answer": 2, "chunk_already_used_by_another_pair": 1}
    p, c = pairs[0], chunks[0]
    assert p["answer_true"] == "Christopher Nolan" and p["answer_true_aliases"] == ["nolan"] and p["edit_type"] == "entity_swap"
    assert "Christopher Nolan" in c["h_t"] and len(c["neutrals"]) == 2 and len(c["cram_fakes"]) == 1
    assert answer_type("1889") == "year" and answer_type("around 10,000") == "numeric" and answer_type("Hodel") == "entity_short"


def test_build_assigns_ids_splits_and_flags_duplicates():
    recs = []
    for i in range(10):
        recs.append(dict(GOOD, question=f"when was tower {i} completed", answer_true=f"18{i:02d}", answer_false=f"19{i:02d}",
                         source=f"trivia:{200 + i}", edit_type="date_shift" if i % 2 else "entity_swap"))
    recs.append(dict(recs[0]))                                     # duplicate question
    recs.append({"pair_id": "", "question": "who directed inception", "answer_true": "Christopher Nolan",
                 "answer_false": "Steven Spielberg", "edit_type": "entity_swap", "source": "nq:7", "topic": "nq"})
    pairs, splits, stats, errors = build(recs, n_dev=4, seed=7)
    assert [p.pair_id for p in pairs] == [f"p{i:04d}" for i in range(1, 13)]
    assert pairs[0].source == "nq:7" and all(p.source.startswith("trivia:") for p in pairs[1:])   # nq first, then trivia by id
    assert any("duplicate question" in e for e in errors)
    assert splits["n_dev"] + splits["n_test"] == 12 and set(splits["dev_pairs"]).isdisjoint(splits["test_pairs"])
    assert stats["by_source_kind"] == {"nq": 1, "trivia": 11}


def test_number_aliases_both_directions():
    from witness_rag.generation.chunking import alias_pattern
    from witness_rag.schemas import derived_aliases, number_aliases
    assert number_aliases("24") == ["twenty-four", "twenty four"]
    assert number_aliases("twenty four") == ["24"] and number_aliases("Twenty-Four") == ["24"]
    assert number_aliases("8") == ["eight"] and number_aliases("Thirteen") == ["13"] and number_aliases("twenty") == ["20"]
    assert number_aliases("Sixth") == ["6th"] and number_aliases("4th") == ["fourth"]
    assert number_aliases("1975") == [] and number_aliases("Tchaikovsky") == []
    assert "eight" in derived_aliases("8") and "24" in derived_aliases("twenty four")
    pat = alias_pattern(["8", *derived_aliases("8")])
    assert pat.search("there are eight planets") and not pat.search("eighteen planets") and not pat.search("1980")


def test_artifact_wrong_answers_are_rejected():
    from witness_rag.claims.nq_import import looks_like_artifact
    assert looks_like_artifact("Fulham disambiguation", ["Charlton"]) == "wikipedia_artifact"
    assert looks_like_artifact("\u0a9a\u0abe\u0ab0\u0acd\u0ab2\u0abf", ["India"]) == "non_latin"
    assert looks_like_artifact("6ft 5in", ["6ft 1in"]) == "unit_value"
    assert looks_like_artifact("Sophia 1774", ["Emma"]) == "name_plus_year"
    assert looks_like_artifact("Hawaii state", ["Alaska"]) == "descriptor_suffix"
    assert looks_like_artifact("England region", ["Ireland"]) == "descriptor_suffix"
    assert looks_like_artifact("pi greek", ["rho"]) == "descriptor_suffix"
    assert looks_like_artifact("Hawaii state", ["Alaska"], strict=False) is None          # lenient: data decides
    for ok in ("Wolfgang Amadeus Mozart", "Tom Hanks", "twenty four", "1975", "Los Angeles", "July 2020", "two seasons"):
        assert looks_like_artifact(ok, ["Pyotr Ilyich Tchaikovsky"]) is None, ok
    assert looks_like_artifact("Insurance agent", ["Claims adjuster"]) is None           # true value has a lowercase tail too
    assert looks_like_artifact("Native American tribes", ["Tchaikovsky"]) == "descriptor_suffix"   # strict import: capital + lowercase tail
    assert looks_like_artifact("conch shells", ["Cigarettes", "20 in a pack"]) is None   # units count in the two values only


def test_exclusion_rule_uses_run_evidence(tmp_path):
    import sys
    sys.path.insert(0, "experiments")
    from repair_pairs import exclusion_reasons
    pairs = [{"pair_id": "p0001", "answer_true": "Charlton", "answer_false": "Fulham disambiguation", "source": "trivia:1"},
             {"pair_id": "p0002", "answer_true": "Dennis Potter", "answer_false": "Martin Scorcese", "source": "trivia:2"},
             {"pair_id": "p0003", "answer_true": "Oasis", "answer_false": "The Rolling Stones", "source": "trivia:3"},
             {"pair_id": "p0004", "answer_true": "8", "answer_false": "twenty", "source": "trivia:4"}]
    plan = {f"d{i:06d}": {"pair_id": pid, "stance": "false", "model": m}
            for i, (pid, m) in enumerate([("p0002", "gen-qwen")] * 6 + [("p0002", "gen-aya")] * 5 + [("p0003", "gen-qwen")] * 12 + [("p0004", "gen-qwen")] * 6 + [("p0004", "gen-aya")] * 6)}
    refusals = [{"doc_id": d, "problem": "value_missing"} for d in plan]
    reasons = exclusion_reasons(pairs, plan, refusals)
    assert reasons["p0001"] == "wikipedia_artifact"
    assert reasons["p0002"].startswith("unwritable_false_value (11")
    assert "p0003" not in reasons                        # one generator only: a refusal pattern, not an artifact
    assert "p0004" not in reasons                        # number words: fixed by aliases, the run's evidence does not count


def test_plural_aliases_for_multiword_values():
    assert "Horse island" in derived_aliases("Horse islands")
    assert "Native American tribe" in derived_aliases("Native American tribes")
    assert "Falkland Island" in derived_aliases("Falkland Islands") and derived_aliases("Goat Island") == []
    assert derived_aliases("Monroe Doctrine") == [] and derived_aliases("eight counties") == []    # lower-cased head: unsafe
    assert derived_aliases("saint james") == [] and derived_aliases("Horse island") == ["Horse islands"]
    assert not any(a.endswith("s") and a != "Tchaikovsky" for a in derived_aliases("Tchaikovsky"))   # single tokens untouched
    assert derived_aliases("Theater 12") == []                                                        # digits: no plural games
