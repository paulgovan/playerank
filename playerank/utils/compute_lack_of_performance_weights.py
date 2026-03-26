from pathlib import Path
from ..models import Weighter
from ..features import qualityFeatures, relativeAggregation, goalScoredFeatures

_ROOT = Path(__file__).parent.parent.parent
_DATA = _ROOT / 'data'
_CONF = Path(__file__).parent.parent / 'conf'


def compute_lack_of_performance_weights(output_path):
    """
    Learn which actions are most associated with losing by training a LinearSVC
    with loss as the positive class ('l-wd': loss vs win/draw).

    The resulting weights identify 'wasteful' actions — those that are
    statistically linked to a team losing matches.
    """
    qualityFeat = qualityFeatures.qualityFeatures()
    quality = qualityFeat.createFeature(events_path=str(_DATA / 'events' / '*.json'),
                        players_file=str(_DATA / 'players.json'), entity='team')
    gs = goalScoredFeatures.goalScoredFeatures()
    goals = gs.createFeature(str(_DATA / 'matches' / '*.json'))
    aggregation = relativeAggregation.relativeAggregation()
    aggregation.set_features([quality, goals])
    df = aggregation.aggregate(to_dataframe=True)

    weighter = Weighter.Weighter(label_type='l-wd')
    weighter.fit(df, 'goal-scored', filename=output_path)
    print("lack-of-performance weights stored in %s" % output_path)


compute_lack_of_performance_weights(str(_CONF / 'lack_of_performance_weights.json'))
