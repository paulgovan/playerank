import pandas as pd
import pytest
from playerank.features.plainAggregation import plainAggregation
from playerank.features.relativeAggregation import relativeAggregation


# ---------------------------------------------------------------------------
# plainAggregation
# ---------------------------------------------------------------------------

FEAT_A = [
    {"match": 1, "entity": 10, "feature": "accurate-pass", "value": 5},
    {"match": 1, "entity": 10, "feature": "shot", "value": 2},
    {"match": 1, "entity": 20, "feature": "accurate-pass", "value": 3},
]
FEAT_B = [
    {"match": 1, "entity": 10, "feature": "foul", "value": 1},
]


def test_plain_merges_two_collections():
    agg = plainAggregation()
    agg.set_features([FEAT_A, FEAT_B])
    result = agg.aggregate(to_dataframe=False)
    # non-dataframe output: one dict per match/entity with all features as keys
    entity_10 = next(r for r in result if r["entity"] == 10)
    assert entity_10["accurate-pass"] == 5
    assert entity_10["shot"] == 2
    assert entity_10["foul"] == 1


def test_plain_to_dataframe_returns_dataframe():
    agg = plainAggregation()
    agg.set_features([FEAT_A])
    df = agg.aggregate(to_dataframe=True)
    assert isinstance(df, pd.DataFrame)
    assert {"match", "entity", "accurate-pass", "shot"}.issubset(df.columns)


def test_plain_fills_missing_features_with_zero():
    agg = plainAggregation()
    agg.set_features([FEAT_A])
    df = agg.aggregate(to_dataframe=True)
    # entity 20 has no "shot" entry — should be 0
    row = df[df["entity"] == 20].iloc[0]
    assert row["shot"] == 0.0


def test_plain_multiple_matches():
    features = [
        {"match": 1, "entity": 10, "feature": "pass", "value": 3},
        {"match": 2, "entity": 10, "feature": "pass", "value": 7},
    ]
    agg = plainAggregation()
    agg.set_features([features])
    result = agg.aggregate(to_dataframe=False)
    assert len(result) == 2


def test_plain_get_set_features():
    agg = plainAggregation()
    agg.set_features([FEAT_A])
    assert agg.get_features() == [FEAT_A]


# ---------------------------------------------------------------------------
# relativeAggregation
# ---------------------------------------------------------------------------

TEAM_FEATURES = [
    {"match": 1, "entity": 100, "feature": "pass", "value": 500},
    {"match": 1, "entity": 200, "feature": "pass", "value": 300},
]


def test_relative_computes_difference():
    agg = relativeAggregation()
    agg.set_features([TEAM_FEATURES])
    result = agg.aggregate(to_dataframe=False)
    by_team = {r["entity"]: r["value"] for r in result if r["name"] == "pass"}
    assert by_team[100] == 200   # 500 - 300
    assert by_team[200] == -200  # 300 - 500


def test_relative_to_dataframe():
    agg = relativeAggregation()
    agg.set_features([TEAM_FEATURES])
    df = agg.aggregate(to_dataframe=True)
    assert isinstance(df, pd.DataFrame)
    assert "pass" in df.columns


def test_relative_missing_opponent_feature_keeps_own_value():
    features = [
        {"match": 1, "entity": 100, "feature": "unique", "value": 10},
        {"match": 1, "entity": 200, "feature": "other", "value": 5},
    ]
    agg = relativeAggregation()
    agg.set_features([features])
    result = agg.aggregate(to_dataframe=False)
    doc = next(r for r in result if r["entity"] == 100 and r["name"] == "unique")
    assert doc["value"] == 10


def test_relative_get_set_features():
    agg = relativeAggregation()
    agg.set_features([TEAM_FEATURES])
    assert agg.get_features() == [TEAM_FEATURES]
