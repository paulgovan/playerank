"""
Feature partitioning for independent performance and waste models (Option A).

Performance model trains only on positive-outcome features (win-associated).
Waste model trains only on negative-outcome features (loss-associated).
This ensures the two score dimensions are genuinely orthogonal rather than
being near-inverses of each other derived from the same feature space.

Usage
-----
    from ..features.featureFilters import filter_features

    perf_features  = filter_features(quality, 'performance_quality')
    waste_features = filter_features(quality, 'waste_quality')
    pos_chains     = filter_features(chains,  'positive_chain')
    harm_chains    = filter_features(chains,  'harmful_chain')
"""

# ---------------------------------------------------------------------------
# Quality feature keywords
# ---------------------------------------------------------------------------

# Positive-outcome keywords → performance model
_PERFORMANCE_KEYWORDS = (
    'accurate',       # e.g. Pass-Simple pass-accurate (but NOT "not accurate")
    'assist',         # key passes that directly led to a goal
    'key pass',       # passes creating clear scoring chances
    'opportunity',    # actions creating scoring opportunities
    'counter attack', # actions in dangerous transition play
)
# Exact feature names that belong to performance regardless of keyword matching
# (e.g. 'Free Kick-Penalty' has no outcome suffix but is strongly win-associated)
_PERFORMANCE_EXACT = frozenset([
    'Free Kick-Penalty',
])

# Negative-outcome keywords → waste model
_WASTE_KEYWORDS = (
    'not accurate',          # any inaccurate action
    'yellow card',
    'red card',
    'second yellow card',
    'dangerous ball lost',   # losing possession in a dangerous area
    'missed ball',           # failing to control / intercept
)

# Generic event names (no outcome suffix) that belong to the waste model
_WASTE_EXACT = frozenset([
    'Foul',                          # generic foul (outcome-agnostic)
    'Others on the ball-Clearance',  # defensive clearance under pressure
])

# ---------------------------------------------------------------------------
# Chain feature sets
# ---------------------------------------------------------------------------

POSITIVE_CHAIN_FEATURES = frozenset([
    'chain-shot-participant',
    'chain-goal-participant',
    'chain-final-action',
])

HARMFUL_CHAIN_FEATURES = frozenset([
    'chain-turnover-precedes-shot',
    'chain-turnover-precedes-goal',
    'chain-turnover-final-actor',
    'chain-conceded-shot',
    'chain-conceded-goal',
])

# ---------------------------------------------------------------------------
# Predicate functions
# ---------------------------------------------------------------------------

def is_performance_quality(name):
    """
    True for quality features representing positive play outcomes.
    'not accurate' is explicitly excluded even though it contains 'accurate'.
    """
    if name in _PERFORMANCE_EXACT:
        return True
    lower = name.lower()
    if 'not accurate' in lower or 'dangerous ball lost' in lower:
        return False
    return any(kw in lower for kw in _PERFORMANCE_KEYWORDS)


def is_waste_quality(name):
    """True for quality features representing negative / wasteful play outcomes."""
    if name in _WASTE_EXACT:
        return True
    lower = name.lower()
    return any(kw in lower for kw in _WASTE_KEYWORDS)


def is_positive_chain(name):
    """True for chain features that credit positive attacking sequences."""
    return name in POSITIVE_CHAIN_FEATURES


def is_harmful_chain(name):
    """True for chain features that debit harmful turnover / defensive sequences."""
    return name in HARMFUL_CHAIN_FEATURES


# ---------------------------------------------------------------------------
# Convenience dispatcher
# ---------------------------------------------------------------------------

_PREDICATES = {
    'performance_quality': is_performance_quality,
    'waste_quality':       is_waste_quality,
    'positive_chain':      is_positive_chain,
    'harmful_chain':       is_harmful_chain,
}


def filter_features(feature_docs, subset):
    """
    Return only the documents whose 'feature' key passes the named predicate.

    Parameters
    ----------
    feature_docs : list of dict
        Feature records in {match, entity, feature, value} format.
    subset : str
        One of 'performance_quality', 'waste_quality',
        'positive_chain', 'harmful_chain'.

    Returns
    -------
    list of dict
    """
    predicate = _PREDICATES.get(subset)
    if predicate is None:
        raise ValueError(
            "Unknown subset %r. Choose from: %s" % (subset, ', '.join(_PREDICATES))
        )
    return [doc for doc in feature_docs if predicate(doc['feature'])]
