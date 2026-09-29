from witness_rag.eval.outcomes import classify, contains_alias, normalize_text, paired_bootstrap, summarize


def test_normalize_and_contains():
    assert normalize_text("The Eiffel Tower, 1889!") == "eiffel tower 1889"
    assert contains_alias("It was completed in 1889.", "1889")
    assert not contains_alias("It was completed in 18890.", "1889")
    assert contains_alias("Directed by Steven Spielberg", "steven spielberg")
    assert not contains_alias("Steven", "Steven Spielberg")


def test_classify():
    gold, attack = ["1889"], ["1875"]
    assert classify("1889", gold, attack) == "gold"
    assert classify("The answer is 1875.", gold, attack) == "attack"
    assert classify("1889, not 1875", gold, attack) == "both"
    assert classify("I do not know.", gold, attack) == "other"
    assert classify("", gold, attack) == "other"


def test_summary_and_bootstrap():
    base = [{"pair_id": f"p{i}", "outcome": "gold" if i % 4 == 0 else "attack"} for i in range(40)]
    better = [{"pair_id": f"p{i}", "outcome": "gold" if i % 2 == 0 else "attack"} for i in range(40)]
    s = summarize(base)
    assert s["n"] == 40 and abs(s["accuracy"] - 0.25) < 1e-9 and abs(s["flip"] - 0.75) < 1e-9
    b = paired_bootstrap(better, base, "gold", n_resamples=500, seed=1)
    assert b["n"] == 40 and abs(b["diff"] - 0.25) < 1e-9 and b["lo"] > 0 and b["significant"]
    same = paired_bootstrap(base, base, "gold", n_resamples=200)
    assert same["diff"] == 0.0 and not same["significant"]
