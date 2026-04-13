from pathlib import Path
from ..features import (centerOfPerformanceFeature, qualityFeatures,
                        playerankFeatures, plainAggregation,
                        matchPlayedFeatures, roleFeatures, chainFeatures)
import json

_ROOT = Path(__file__).parent.parent.parent
_DATA = _ROOT / 'data'
_CONF = Path(__file__).parent.parent / 'conf'

waste_weights_file          = str(_CONF / 'lack_of_performance_weights.json')
performance_weights_file    = str(_CONF / 'features_weights.json')
harmful_chain_weights_file  = str(_CONF / 'harmful_chain_weights.json')

# ── Quality features (player level) ──────────────────────────────────────────
qualityFeat = qualityFeatures.qualityFeatures()
quality = qualityFeat.createFeature(events_path=str(_DATA / 'events' / '*.json'),
                    players_file=str(_DATA / 'players.json'), entity='player')

# ── Chain features (player level) ────────────────────────────────────────────
chainFeat = chainFeatures.chainFeatures()
chains = chainFeat.createFeature(events_path=str(_DATA / 'events' / '*.json'),
                    players_file=str(_DATA / 'players.json'), entity='player')

# ── Waste score: quality features × loss-associated weights ──────────────────
wasteFeat = playerankFeatures.playerankFeatures()
wasteFeat.set_features([quality])
waste = wasteFeat.createFeature(waste_weights_file)
for doc in waste:
    doc['feature'] = 'wasteScore'

# ── Performance score: quality + chain features × win-associated weights ──────
prFeat = playerankFeatures.playerankFeatures()
prFeat.set_features([quality, chains])
pr = prFeat.createFeature(performance_weights_file)

# ── Chain performance score: chain features × win-associated weights ──────────
chainScoreFeat = playerankFeatures.playerankFeatures()
chainScoreFeat.set_features([chains])
chain_score = chainScoreFeat.createFeature(performance_weights_file)
for doc in chain_score:
    doc['feature'] = 'chainScore'

# ── Chain waste score: chain features × loss-associated chain weights ─────────
chain_waste = []
if Path(harmful_chain_weights_file).exists():
    chainWasteFeat = playerankFeatures.playerankFeatures()
    chainWasteFeat.set_features([chains])
    chain_waste = chainWasteFeat.createFeature(harmful_chain_weights_file)
    for doc in chain_waste:
        doc['feature'] = 'chainWasteScore'
else:
    print("harmful_chain_weights.json not found — chainWasteScore will be omitted. "
          "Run compute_harmful_chain_weights first.")

# ── Supporting features ───────────────────────────────────────────────────────
matchPlayedFeat = matchPlayedFeatures.matchPlayedFeatures()
matchplayed = matchPlayedFeat.createFeature(matches_path=str(_DATA / 'matches' / '*.json'),
                    players_file=str(_DATA / 'players.json'))

center_performance = centerOfPerformanceFeature.centerOfPerformanceFeature()
center_performance = center_performance.createFeature(
    events_path=str(_DATA / 'events' / '*.json'),
    players_file=str(_DATA / 'players.json'))

roleFeat = roleFeatures.roleFeatures()
roleFeat.set_features([center_performance])
roles = roleFeat.createFeature(matrix_role_file=str(_CONF / 'role_matrix.json'))

# ── Aggregate everything into one dashboard DataFrame ─────────────────────────
# chains is included raw so the participation columns (chain-shot-participant etc.)
# appear as their own columns in the CSV for the dashboard leaderboards.
aggregation = plainAggregation.plainAggregation()
aggregation.set_features([matchplayed, pr, waste, roles, chains,
                          chain_score, chain_waste])
df = aggregation.aggregate(to_dataframe=True)

# ── Derived net scores ────────────────────────────────────────────────────────
if 'playerankScore' in df.columns and 'wasteScore' in df.columns:
    df['netScore'] = df['playerankScore'] - df['wasteScore']

if 'chainScore' in df.columns and 'chainWasteScore' in df.columns:
    df['chainNetScore'] = df['chainScore'] - df['chainWasteScore']

# ── Write output ──────────────────────────────────────────────────────────────
out = _DATA / 'dashboard_data.csv'
df.to_csv(out, index=False)
results_out = _ROOT / 'results' / 'dashboard_data.csv'
results_out.parent.mkdir(exist_ok=True)
df.to_csv(results_out, index=False)
print("Dashboard data saved to %s and %s" % (out, results_out))

report_cols = [c for c in ['entity', 'playerankScore', 'wasteScore', 'netScore',
                             'chainScore', 'chainWasteScore', 'chainNetScore']
               if c in df.columns]
print(df[report_cols].head())
