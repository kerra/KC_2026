from witness_rag.dependence.minhash import (MinHasher, cluster_by_threshold, find_near_duplicates, jaccard,
                                            jaccard_matrix, shingles)

A = "The Eiffel Tower was completed in 1889 and quickly became the most visited monument in Paris."
B = "The Eiffel Tower was completed in 1889 and quickly became the most visited monument in Paris, France."
C = "Bananas are rich in potassium and are among the most popular fruits sold in supermarkets worldwide."


def test_jaccard_basics():
    assert jaccard(shingles(A), shingles(A)) == 1.0
    assert jaccard(shingles(A), shingles(C)) == 0.0
    assert 0.5 < jaccard(shingles(A), shingles(B)) < 1.0


def test_threshold_clustering():
    sim = jaccard_matrix([A, B, C, A])
    labels = cluster_by_threshold(sim, 0.35)
    assert labels[0] == labels[1] == labels[3]
    assert labels[2] != labels[0]
    assert labels == [0, 0, 1, 0]


def test_minhash_estimate_close_to_exact():
    h = MinHasher(num_perm=256, seed=3)
    est = float((h.signature(shingles(A)) == h.signature(shingles(B))).mean())
    assert abs(est - jaccard(shingles(A), shingles(B))) < 0.15


def test_find_near_duplicates_flags_copies_only():
    hits = find_near_duplicates([A, C, A, B], threshold=0.5)
    pairs = {(i, j) for i, j, _ in hits}
    assert (0, 2) in pairs
    assert (0, 3) in pairs
    assert all(1 not in p for p in pairs)
