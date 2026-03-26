from pathlib import Path
from ..features import (centerOfPerformanceFeature, qualityFeatures,
                        playerankFeatures, plainAggregation,
                        matchPlayedFeatures, roleFeatures)
import json

_ROOT = Path(__file__).parent.parent.parent
_DATA = _ROOT / 'data'
_CONF = Path(__file__).parent.parent / 'conf'

waste_weights_file = str(_CONF / 'lack_of_performance_weights.json')
performance_weights_file = str(_CONF / 'features_weights.json')

# Quality features (player level)
qualityFeat = qualityFeatures.qualityFeatures()
quality = qualityFeat.createFeature(events_path=str(_DATA / 'events'),
                    players_file=str(_DATA / 'players.json'), entity='player')

# Waste score: weighted by loss-associated feature weights
wasteFeat = playerankFeatures.playerankFeatures()
wasteFeat.set_features([quality])
waste = wasteFeat.createFeature(waste_weights_file)

# Performance score: weighted by win-associated feature weights (for net rating)
prFeat = playerankFeatures.playerankFeatures()
prFeat.set_features([quality])
pr = prFeat.createFeature(performance_weights_file)

# Rename waste feature to distinguish it from playerankScore
for doc in waste:
    doc['feature'] = 'wasteScore'

matchPlayedFeat = matchPlayedFeatures.matchPlayedFeatures()
matchplayed = matchPlayedFeat.createFeature(matches_path=str(_DATA / 'matches'),
                    players_file=str(_DATA / 'players.json'))

center_performance = centerOfPerformanceFeature.centerOfPerformanceFeature()
center_performance = center_performance.createFeature(
    events_path=str(_DATA / 'events'),
    players_file=str(_DATA / 'players.json'))

roleFeat = roleFeatures.roleFeatures()
roleFeat.set_features([center_performance])
roles = roleFeat.createFeature(matrix_role_file=str(_CONF / 'role_matrix.json'))

aggregation = plainAggregation.plainAggregation()
aggregation.set_features([matchplayed, pr, waste, roles])
df = aggregation.aggregate(to_dataframe=True)

# Net score: positive performance minus waste
if 'playerankScore' in df.columns and 'wasteScore' in df.columns:
    df['netScore'] = df['playerankScore'] - df['wasteScore']

print(df[['entity', 'playerankScore', 'wasteScore', 'netScore']].head())
