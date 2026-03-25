from pathlib import Path
from ..models import Weighter
from ..features import centerOfPerformanceFeature, qualityFeatures, playerankFeatures, plainAggregation, matchPlayedFeatures, roleFeatures
import json

_ROOT = Path(__file__).parent.parent.parent
_DATA = _ROOT / 'data'
_CONF = Path(__file__).parent.parent / 'conf'

weights_file = str(_CONF / 'features_weights.json')

qualityFeat = qualityFeatures.qualityFeatures()
quality = qualityFeat.createFeature(events_path=str(_DATA / 'events'),
                    players_file=str(_DATA / 'players.json'), entity='player')

prFeat = playerankFeatures.playerankFeatures()
prFeat.set_features([quality])
pr = prFeat.createFeature(weights_file)

matchPlayedFeat = matchPlayedFeatures.matchPlayedFeatures()
matchplayed = matchPlayedFeat.createFeature(matches_path=str(_DATA / 'matches'),
                    players_file=str(_DATA / 'players.json'))

center_performance = centerOfPerformanceFeature.centerOfPerformanceFeature()
center_performance = center_performance.createFeature(events_path=str(_DATA / 'events'),
                                        players_file=str(_DATA / 'players.json'))

roleFeat = roleFeatures.roleFeatures()
roleFeat.set_features([center_performance])
roles = roleFeat.createFeature(matrix_role_file=str(_CONF / 'role_matrix.json'))

aggregation = plainAggregation.plainAggregation()
aggregation.set_features([matchplayed, pr, roles])
df = aggregation.aggregate(to_dataframe=True)

print(df.head())
