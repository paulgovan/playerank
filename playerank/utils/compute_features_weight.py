from pathlib import Path
from ..models import Weighter
from ..features import qualityFeatures, relativeAggregation, goalScoredFeatures

_ROOT = Path(__file__).parent.parent.parent
_DATA = _ROOT / 'data'
_CONF = Path(__file__).parent.parent / 'conf'

def compute_feature_weights(output_path):

    qualityFeat = qualityFeatures.qualityFeatures()
    quality = qualityFeat.createFeature(events_path=str(_DATA / 'events' / '*.json'),
                        players_file=str(_DATA / 'players.json'), entity='team')
    gs = goalScoredFeatures.goalScoredFeatures()
    goals = gs.createFeature(str(_DATA / 'matches' / '*.json'))
    aggregation = relativeAggregation.relativeAggregation()
    aggregation.set_features([quality, goals])
    df = aggregation.aggregate(to_dataframe=True)

    weighter = Weighter.Weighter(label_type='wd-l')
    weighter.fit(df, 'goal-scored', filename=output_path)
    print("features weights stored in %s" % output_path)


compute_feature_weights(str(_CONF / 'features_weights.json'))
