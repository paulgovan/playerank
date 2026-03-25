import numpy as np
import pandas as pd
import pytest
from playerank.models.Clusterer import Clusterer


@pytest.fixture
def position_df():
    """Three clearly separated clusters in 2D position space."""
    np.random.seed(42)
    c1 = np.random.randn(30, 2) + [20, 20]
    c2 = np.random.randn(30, 2) + [50, 80]
    c3 = np.random.randn(30, 2) + [80, 30]
    X = np.clip(np.vstack([c1, c2, c3]), 0, 100).astype(int)
    return pd.DataFrame(X, columns=["avg_x", "avg_y"])


def _fit(df, k_range=(2, 4), kind="single"):
    c = Clusterer(k_range=k_range, verbose=False, random_state=42)
    ids = list(range(len(df)))
    c.fit(ids, ids, df, kind=kind)
    return c


def test_fit_assigns_label_to_every_sample(position_df):
    c = _fit(position_df)
    assert len(c.labels_) == len(position_df)


def test_fit_k_within_range(position_df):
    c = _fit(position_df, k_range=(2, 5))
    assert 2 <= c.n_clusters_ <= 5


def test_cluster_centers_shape(position_df):
    c = _fit(position_df, k_range=(2, 3))
    assert c.cluster_centers_.shape == (c.n_clusters_, 2)


def test_get_clusters_matrix_is_nested_dict(position_df):
    c = _fit(position_df, k_range=(2, 3))
    matrix = c.get_clusters_matrix(kind="single")
    assert isinstance(matrix, dict)
    # Keys are integers 0–100
    assert 0 in matrix
    assert isinstance(matrix[0], dict)
    assert 0 in matrix[0]


def test_get_clusters_matrix_covers_full_grid(position_df):
    c = _fit(position_df, k_range=(2, 3))
    matrix = c.get_clusters_matrix(kind="single")
    assert len(matrix) == 101
    assert all(len(v) == 101 for v in matrix.values())


def test_fit_multi_kind(position_df):
    c = _fit(position_df, k_range=(2, 3), kind="multi")
    # Multi-kind labels are lists
    assert isinstance(c.labels_, list)
    assert all(isinstance(lbl, list) for lbl in c.labels_)
