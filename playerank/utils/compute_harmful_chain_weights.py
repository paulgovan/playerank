from pathlib import Path
from ..models import Weighter
from ..features import chainFeatures, relativeAggregation, goalScoredFeatures
from ..features.featureFilters import filter_features

_ROOT = Path(__file__).parent.parent.parent
_DATA = _ROOT / 'data'
_CONF = Path(__file__).parent.parent / 'conf'


def compute_harmful_chain_weights(output_path):
    """
    Learn which harmful chain patterns are most associated with losing by training
    a LinearSVC with loss as the positive class ('l-wd': loss vs win/draw).

    Option A: trains only on harmful chain features (turnover chains, conceded chains)
    so this model is independent of the positive chain performance model which trains
    only on shot/goal participation features.
    """
    chainFeat = chainFeatures.chainFeatures()
    chains = chainFeat.createFeature(
        events_path=str(_DATA / 'events' / '*.json'),
        players_file=str(_DATA / 'players.json'),
        entity='team',
    )
    harmful_chains = filter_features(chains, 'harmful_chain')

    gs = goalScoredFeatures.goalScoredFeatures()
    goals = gs.createFeature(str(_DATA / 'matches' / '*.json'))

    aggregation = relativeAggregation.relativeAggregation()
    aggregation.set_features([harmful_chains, goals])
    df = aggregation.aggregate(to_dataframe=True)

    weighter = Weighter.Weighter(label_type='l-wd')
    weighter.fit(df, 'goal-scored', filename=output_path)
    print("harmful chain weights stored in %s" % output_path)


compute_harmful_chain_weights(str(_CONF / 'harmful_chain_weights.json'))
