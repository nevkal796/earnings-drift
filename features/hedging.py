import re

# Loughran-McDonald inspired hedging and uncertainty lexicons
# These are academically validated for financial text
HEDGING_TERMS = [
    "approximately", "assume", "assumed", "assurance", "believe",
    "believed", "conceivable", "conditional", "could", "doubt",
    "doubtful", "estimate", "estimated", "expect", "expected",
    "feel", "fluctuate", "hopefully", "if", "imprecise", "intend",
    "likely", "may", "maybe", "might", "opinion", "perhaps",
    "plan", "plausible", "possible", "possibly", "postulate",
    "potential", "potentially", "predict", "probable", "probably",
    "propose", "rough", "roughly", "seem", "should", "sometime",
    "sometimes", "suggest", "suppose", "think", "uncertain",
    "uncertainty", "unclear", "unexpected", "unlikely", "usually",
    "vague", "would",
]

UNCERTAINTY_TERMS = [
    "ambiguous", "anomaly", "challenge", "concern", "difficult",
    "difficulty", "doubt", "fluctuation", "headwind", "impact",
    "instability", "obstacle", "pressure", "risk", "risky",
    "uncertainty", "unclear", "unforeseen", "unknown", "unpredictable",
    "unpredictably", "unstable", "volatile", "volatility", "worry",
]

SPECIFICITY_PATTERN = re.compile(
    r"""
    \b\d+\.?\d*\s*(%|percent|billion|million|thousand|bps|basis points) |
    \$\s*\d+\.?\d*\s*(billion|million|thousand|b|m|k)? |
    \b(q[1-4]|fy|fiscal year)\s*\d{4}\b |
    \b\d{4}\s*guidance\b
    """,
    re.IGNORECASE | re.VERBOSE
)


POSITIVE_SIGNAL_TERMS = [
    "record", "outperform", "beat", "accelerate",
    "expand", "strong", "robust", "momentum",
]

_HEDGE_SET    = set(HEDGING_TERMS)
_UNCERT_SET   = set(UNCERTAINTY_TERMS)
_POSITIVE_SET = set(POSITIVE_SIGNAL_TERMS)
_ALL_TERMS    = _HEDGE_SET | _UNCERT_SET | _POSITIVE_SET


def keyword_frequency(texts: list[str]) -> dict[str, int]:
    """
    Count occurrences of every term across the corpus.
    Returns top-20 {term: count} sorted descending.
    """
    counts: dict[str, int] = {}
    for text in texts:
        if not text:
            continue
        for word in text.lower().split():
            token = word.strip(".,;:!?\"'()")
            if token in _ALL_TERMS:
                counts[token] = counts.get(token, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True)[:20])


def hedging_score(text: str) -> float:
    """Hedging terms per 1000 words."""
    if not text:
        return 0.0
    words = text.lower().split()
    if len(words) < 50:
        return 0.0
    hits = sum(1 for w in words if w.strip(".,;:") in HEDGING_TERMS)
    return round(hits / (len(words) / 1000), 4)


def uncertainty_score(text: str) -> float:
    """Uncertainty terms per 1000 words."""
    if not text:
        return 0.0
    words = text.lower().split()
    if len(words) < 50:
        return 0.0
    hits = sum(1 for w in words if w.strip(".,;:") in UNCERTAINTY_TERMS)
    return round(hits / (len(words) / 1000), 4)


def specificity_score(text: str) -> float:
    """
    Ratio of concrete numeric references to total words.
    Higher = more specific, more confident language.
    """
    if not text:
        return 0.0
    words = text.split()
    if len(words) < 50:
        return 0.0
    matches = SPECIFICITY_PATTERN.findall(text)
    return round(len(matches) / (len(words) / 1000), 4)