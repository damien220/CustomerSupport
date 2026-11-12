"""Pure ranking helpers: RRF fusion, near-duplicate collapse, snippets, FTS5 query building."""

from __future__ import annotations

import pytest

from faq_core.retrieval.chunk_store import build_match_query
from faq_core.retrieval.ranking import (
    content_fingerprint,
    dedupe_ranked,
    jaccard,
    make_snippet,
    rrf_fuse,
    rrf_max_score,
    shingles,
)

# ── RRF ──────────────────────────────────────────────────────────────────────


def test_rrf_rewards_agreement_between_rankings() -> None:
    fused = rrf_fuse([["a", "b", "c"], ["b", "c", "a"]], k=60)

    assert [item for item, _ in fused] == ["b", "a", "c"]
    assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)


def test_rrf_includes_items_seen_by_only_one_ranking() -> None:
    fused = dict(rrf_fuse([["a"], ["b"]], k=1))

    assert fused == {"a": 0.5, "b": 0.5}


def test_rrf_ties_keep_first_seen_order_and_count_repeats_once() -> None:
    fused = rrf_fuse([["x", "x", "y"], ["y", "x"]], k=60)

    assert [item for item, _ in fused] == ["x", "y"]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)


def test_rrf_weights_scale_each_ranking() -> None:
    fused = rrf_fuse([["a"], ["b"]], k=60, weights=[1.0, 2.0])

    assert [item for item, _ in fused] == ["b", "a"]


@pytest.mark.parametrize(("k", "weights"), [(0, None), (60, [1.0])])
def test_rrf_rejects_bad_parameters(k: int, weights: list[float] | None) -> None:
    with pytest.raises(ValueError):
        rrf_fuse([["a"], ["b"]], k=k, weights=weights)


def test_rrf_max_score_is_first_place_in_every_ranking() -> None:
    fused = rrf_fuse([["a"], ["a"]], k=60)

    assert fused[0][1] == pytest.approx(rrf_max_score(2, k=60))


# ── dedupe ───────────────────────────────────────────────────────────────────


def test_dedupe_drops_exact_and_near_duplicates_keeping_the_best_ranked() -> None:
    items = [
        "Refunds are issued within 5 business days to the original card.",
        "Refunds are issued within 5 business days to the original card!",  # exact (normalised)
        "REFUNDS are issued within 5 business days to the original card, usually.",  # near
        "Shipping takes a week.",
    ]

    kept = dedupe_ranked(items, lambda text: text, threshold=0.75)

    assert kept == [items[0], items[3]]


def test_dedupe_threshold_above_one_only_collapses_exact_matches() -> None:
    items = ["the quick brown fox jumps", "the quick brown fox leaps", "The quick brown fox jumps."]

    assert dedupe_ranked(items, lambda t: t, threshold=1.1) == items[:2]


def test_shingles_and_jaccard() -> None:
    assert shingles("a b") == frozenset({"a", "b"})
    assert jaccard(shingles("one two three four"), shingles("one two three five")) == 1 / 3
    assert jaccard(frozenset(), frozenset()) == 1.0


def test_content_fingerprint_ignores_case_whitespace_and_punctuation() -> None:
    assert content_fingerprint("Hello,  World!") == content_fingerprint("hello world")
    assert content_fingerprint("hello world") != content_fingerprint("hello there")


# ── snippets ─────────────────────────────────────────────────────────────────


def test_short_text_is_returned_whole() -> None:
    assert make_snippet("  Short   answer. ", "anything") == "Short answer."


def test_snippet_starts_at_the_sentence_with_the_first_query_term() -> None:
    text = ("Filler sentence about nothing. " * 20) + "Refunds go back to your card. " + "x " * 50

    snippet = make_snippet(text, "how do refunds work", max_chars=120)

    assert snippet.startswith("…Refunds go back to your card.")
    assert snippet.endswith("…")
    assert len(snippet) <= 122


def test_snippet_without_a_match_starts_at_the_beginning() -> None:
    snippet = make_snippet("word " * 100, "zzz", max_chars=50)

    assert snippet.startswith("word") and snippet.endswith("…")


# ── FTS5 query building ──────────────────────────────────────────────────────


def test_match_query_quotes_terms_and_drops_stopwords() -> None:
    assert build_match_query("How do I reset my password?") == '"reset" OR "password"'


def test_match_query_neutralises_fts5_operators() -> None:
    query = build_match_query('title:secret NEAR(a b) -x* "quoted"')

    assert query == '"title" OR "secret" OR "near" OR "b" OR "x" OR "quoted"'


def test_match_query_keeps_stopwords_when_nothing_else_is_left() -> None:
    assert build_match_query("how are you") == '"how" OR "are" OR "you"'
    assert build_match_query("?!") is None
