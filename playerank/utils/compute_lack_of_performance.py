from pathlib import Path
from ..features import (centerOfPerformanceFeature, qualityFeatures,
                        playerankFeatures, plainAggregation,
                        matchPlayedFeatures, roleFeatures, chainFeatures)
from ..features.featureFilters import filter_features
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

# Option A: each score uses an independent, non-overlapping feature subset so
# that performance and waste scores are genuinely orthogonal.

# ── Waste score: negative-outcome quality features × loss-associated weights ──
waste_quality = filter_features(quality, 'waste_quality')
wasteFeat = playerankFeatures.playerankFeatures()
wasteFeat.set_features([waste_quality])
waste = wasteFeat.createFeature(waste_weights_file)
for doc in waste:
    doc['feature'] = 'wasteScore'

# ── Performance score: positive-outcome quality + positive chain features ──────
perf_quality = filter_features(quality, 'performance_quality')
pos_chains   = filter_features(chains,  'positive_chain')
prFeat = playerankFeatures.playerankFeatures()
prFeat.set_features([perf_quality, pos_chains])
pr = prFeat.createFeature(performance_weights_file)

# ── Chain performance score: positive chain features × win-associated weights ──
chainScoreFeat = playerankFeatures.playerankFeatures()
chainScoreFeat.set_features([pos_chains])
chain_score = chainScoreFeat.createFeature(performance_weights_file)
for doc in chain_score:
    doc['feature'] = 'chainScore'

# ── Chain waste score: harmful chain features × loss-associated chain weights ──
harm_chains = filter_features(chains, 'harmful_chain')
chain_waste = []
if Path(harmful_chain_weights_file).exists():
    chainWasteFeat = playerankFeatures.playerankFeatures()
    chainWasteFeat.set_features([harm_chains])
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

# ── Write chain_map.json ──────────────────────────────────────────────────────
# For each chain ending in a shot or goal, record start position and outcome.
# The dashboard uses this for the Chain Origin pitch heatmap.
from collections import defaultdict as _dd
import glob as _glob

_chain_map = []
_goalkeeper_ids_map = frozenset(
    p['wyId'] for p in json.load(open(str(_DATA / 'players.json')))
    if p['role']['name'] == 'Goalkeeper'
)
_SHOT_ID = 10
_GOAL_TAG = 101
_BREAK_IDS = {5, 2, 6}  # INTERRUPTION, FOUL, OFFSIDE

for _fpath in _glob.glob(str(_DATA / 'events' / '*.json')):
    _data = json.load(open(_fpath))
    _by_match = _dd(list)
    for _e in _data:
        if (_e['matchPeriod'] in ('1H', '2H')
                and _e.get('playerId', 0) != 0
                and _e.get('teamId', 0) != 0):
            _by_match[_e['matchId']].append(_e)

    for _mid, _evts in _by_match.items():
        _evts.sort(key=lambda e: ({'1H': 0, '2H': 1}.get(e['matchPeriod'], 99), e['eventSec']))
        _chain, _chains = [], []
        for _i, _ev in enumerate(_evts):
            if not _chain:
                _chain.append(_ev)
                continue
            _prev = _chain[-1]
            _brk = (_ev['matchPeriod'] != _prev['matchPeriod']
                    or _ev['eventId'] in _BREAK_IDS
                    or _ev['teamId'] != _prev['teamId'])
            if _brk:
                _chains.append(_chain)
                _chain = [_ev]
            else:
                _chain.append(_ev)
        if _chain:
            _chains.append(_chain)

        for _ch in _chains:
            _last = _ch[-1]
            if _last['eventId'] != _SHOT_ID:
                continue
            _tag_ids = {t['id'] for t in _last.get('tags', [])}
            _outcome = 'goal' if _GOAL_TAG in _tag_ids else 'shot'
            _first = _ch[0]
            _pos = _first.get('positions', [])
            if _pos:
                _chain_map.append({
                    'match': _mid,
                    'outcome': _outcome,
                    'start_x': _pos[0].get('x', 0),
                    'start_y': _pos[0].get('y', 0),
                    'length': len(_ch),
                    'teamId': _first.get('teamId'),
                })

_chain_map_out = _DATA / 'chain_map.json'
json.dump(_chain_map, open(_chain_map_out, 'w'))
print("Chain map saved to %s (%d chains)" % (_chain_map_out, len(_chain_map)))

# ── Write dashboard CSV ───────────────────────────────────────────────────────
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
