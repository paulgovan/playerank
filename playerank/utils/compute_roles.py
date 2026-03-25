from pathlib import Path
from ..models import Clusterer
from ..features import centerOfPerformanceFeature, plainAggregation
import json

_ROOT = Path(__file__).parent.parent.parent
_DATA = _ROOT / 'data'
_CONF = Path(__file__).parent.parent / 'conf'

def compute_roleMatrix(output_path):
    centerfeat = centerOfPerformanceFeature.centerOfPerformanceFeature()
    centerfeat = centerfeat.createFeature(events_path=str(_DATA / 'events'),
                        players_file=str(_DATA / 'players.json'))

    aggregation = plainAggregation.plainAggregation()
    aggregation.set_features([centerfeat])
    df = aggregation.aggregate(to_dataframe=True)

    clusterer = Clusterer.Clusterer(verbose=True, k_range=(8, 9))
    clusterer.fit(df.entity, df.match, df[['avg_x', 'avg_y']], kind='multi')

    matrix_role = clusterer.get_clusters_matrix(kind='multi')

    with open(output_path, 'w') as f:
        json.dump(matrix_role, f)


compute_roleMatrix(str(_CONF / 'role_matrix.json'))
