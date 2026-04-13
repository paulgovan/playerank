from pathlib import Path
from ..models import Weighter
from ..features import chainFeatures, relativeAggregation, goalScoredFeatures

_ROOT = Path(__file__).parent.parent.parent
_DATA = _ROOT / 'data'
_CONF = Path(__file__).parent.parent / 'conf'


def compute_harmful_chain_weights(output_path):
    """
    Learn which chain patterns are most associated with losing by training a
    LinearSVC with loss as the positive class ('l-wd': loss vs win/draw).

    Uses team-level chain features (both positive and harmful) so the Weighter
    naturally assigns:
      - negative weights to chain-goal-participant, chain-shot-participant
        (being in goal chains is win-associated, not loss-associated)
      - positive weights to chain-turnover-precedes-goal, chain-conceded-goal
        (these patterns are loss-associated)

    The resulting weights are stored in harmful_chain_weights.json and applied
    at the player level in compute_lack_of_performance.py to produce chainWasteScore.
    """
    chainFeat = chainFeatures.chainFeatures()
    chains = chainFeat.createFeature(
        events_path=str(_DATA / 'events' / '*.json'),
        players_file=str(_DATA / 'players.json'),
        entity='team',
    )

    gs = goalScoredFeatures.goalScoredFeatures()
    goals = gs.createFeature(str(_DATA / 'matches' / '*.json'))

    aggregation = relativeAggregation.relativeAggregation()
    aggregation.set_features([chains, goals])
    df = aggregation.aggregate(to_dataframe=True)

    weighter = Weighter.Weighter(label_type='l-wd')
    weighter.fit(df, 'goal-scored', filename=output_path)
    print("harmful chain weights stored in %s" % output_path)


compute_harmful_chain_weights(str(_CONF / 'harmful_chain_weights.json'))
