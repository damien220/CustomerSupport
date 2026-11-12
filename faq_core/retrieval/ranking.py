"""Pure ranking helpers: Reciprocal Rank Fusion, near-duplicate collapse, snippets.

**RRF** (Cormack, Clarke & Büttcher, 2009): every retriever contributes
``weight / (k + rank)`` per item (rank starts at 1) and items are ordered by the
sum. It needs no score calibration between BM25 and cosine similarity, which
is why hybrid search uses it; ``k = 60`` is the value from the paper.

**Dedupe**: support knowledge bases repeat themselves (the same answer pasted
into several entries or pages). Texts are compared on their normalised word
3-shingles; two texts whose Jaccard similarity reaches the threshold — or whose
normalised text is identical — are duplicates and only the better-ranked one is
kept. Pairwise comparison is fine because it only runs over the fused candidate
pool (tens of items), never the whole index.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Hashable, Sequence
from typing import Final

DEFAULT_RRF_K: Final = 60
DEFAULT_DEDUPE_THRESHOLD: Final = 0.85
SHINGLE_SIZE: Final = 3

_WORD: Final = re.compile(r"\w+")

# Function words carry no topical signal; dropping them keeps short FAQ
# questions ("How do I …") from looking alike to the hashing embedder and keeps
# them out of the FTS5 OR-query.
STOPWORDS: Final = frozenset(
    "a an and are as at be but by can could did do does for from had has have how i if in "
    "into is it its me my no not of on or our please should so than that the their them "
    "then there these they this to us was we were what when where which who why will with "
    "would you your".split()
)


def normalize_text(text: str) -> str:
    """Casefold *text* and reduce it to its words separated by single spaces."""
    return " ".join(_WORD.findall(text.casefold()))


def fingerprint(*parts: str) -> str:
    """Return a stable SHA-256 hex digest of *parts* (order-sensitive)."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()


def content_fingerprint(text: str) -> str:
    """Fingerprint of the normalised *text*: equal for whitespace/case/punctuation variants."""
    return fingerprint(normalize_text(text))


def shingles(text: str, size: int = SHINGLE_SIZE) -> frozenset[str]:
    """Return the set of *size*-word shingles of *text* (the words themselves if shorter)."""
    words = normalize_text(text).split()
    if len(words) < size:
        return frozenset(words)
    return frozenset(" ".join(words[i : i + size]) for i in range(len(words) - size + 1))


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    """Jaccard similarity of two sets (1.0 for two empty sets)."""
    if not left and not right:
        return 1.0
    return len(left & right) / len(left | right)


def dedupe_ranked[T](
    items: Sequence[T],
    text_of: Callable[[T], str],
    *,
    threshold: float = DEFAULT_DEDUPE_THRESHOLD,
) -> list[T]:
    """Drop items that near-duplicate a better-ranked item; keep the order.

    Args:
        items: Best first.
        text_of: The text compared for each item.
        threshold: Shingle Jaccard similarity at or above which two items are
            duplicates; ``> 1`` disables near-duplicate detection (exact
            normalised matches are still collapsed).
    """
    kept: list[T] = []
    kept_shingles: list[frozenset[str]] = []
    seen_exact: set[str] = set()
    for item in items:
        text = text_of(item)
        exact = content_fingerprint(text)
        if exact in seen_exact:
            continue
        item_shingles = shingles(text)
        if any(jaccard(item_shingles, other) >= threshold for other in kept_shingles):
            continue
        seen_exact.add(exact)
        kept.append(item)
        kept_shingles.append(item_shingles)
    return kept


def rrf_fuse[K: Hashable](
    rankings: Sequence[Sequence[K]],
    *,
    k: int = DEFAULT_RRF_K,
    weights: Sequence[float] | None = None,
) -> list[tuple[K, float]]:
    """Fuse ranked lists with Reciprocal Rank Fusion; return ``(item, score)`` best first.

    Ties keep first-seen order (the earlier ranking wins), so results are
    deterministic. A repeated item within one ranking counts at its best rank.

    Raises:
        ValueError: ``k < 1`` or ``weights`` of the wrong length.
    """
    if k < 1:
        raise ValueError(f"RRF k must be >= 1, got {k}")
    if weights is not None and len(weights) != len(rankings):
        raise ValueError("weights must have one entry per ranking")
    scores: dict[K, float] = {}
    for position, ranking in enumerate(rankings):
        weight = 1.0 if weights is None else weights[position]
        seen: set[K] = set()
        for rank, item in enumerate(ranking, start=1):
            if item in seen:
                continue
            seen.add(item)
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


def rrf_max_score(retrievers: int, *, k: int = DEFAULT_RRF_K) -> float:
    """Best possible fused score: ranked first by each of *retrievers* (unit weights)."""
    return retrievers / (k + 1)


def make_snippet(text: str, query: str, *, max_chars: int = 280) -> str:
    """Return up to *max_chars* of *text*, starting near the first query-term match.

    The window starts at the beginning of the sentence holding the earliest
    query word (ignoring words under three letters), so the agent sees why the
    passage matched. Cut ends are marked with ``…``.
    """
    flat = " ".join(text.split())
    if len(flat) <= max_chars:
        return flat
    lowered = flat.casefold()
    positions = [
        lowered.find(term) for term in set(_WORD.findall(query.casefold())) if len(term) >= 3
    ]
    first = min((p for p in positions if p >= 0), default=0)
    boundary = flat.rfind(". ", 0, first)
    start = boundary + 2 if boundary >= 0 else 0
    if first - start > max_chars // 2:  # a very long sentence: centre on the match
        start = max(first - max_chars // 4, 0)
    window = flat[start : start + max_chars]
    if start + max_chars < len(flat):
        cut = window.rfind(" ")
        window = (window[:cut] if cut > max_chars // 2 else window).rstrip() + "…"
    return ("…" + window) if start > 0 else window
