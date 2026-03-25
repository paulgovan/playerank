import json
import pytest
from playerank.features.playerankFeatures import playerankFeatures


@pytest.fixture
def weights_file(tmp_path):
    weights = {"accurate-pass": 0.5, "shot": 0.3}
    f = tmp_path / "weights.json"
    f.write_text(json.dumps(weights))
    return str(f)


QUALITY_FEATURES = [
    {"match": 1, "entity": 10, "feature": "accurate-pass", "value": 4},
    {"match": 1, "entity": 10, "feature": "shot", "value": 2},
    {"match": 1, "entity": 20, "feature": "accurate-pass", "value": 6},
]


def test_computes_weighted_sum(weights_file):
    pf = playerankFeatures()
    pf.set_features([QUALITY_FEATURES])
    result = pf.createFeature(weights_file)
    scores = {r["entity"]: r["value"] for r in result}
    # entity 10: 4*0.5 + 2*0.3 = 2.6
    assert scores[10] == pytest.approx(2.6)
    # entity 20: 6*0.5 = 3.0
    assert scores[20] == pytest.approx(3.0)


def test_ignores_features_not_in_weights(weights_file):
    features_with_unknown = QUALITY_FEATURES + [
        {"match": 1, "entity": 10, "feature": "unknown-feat", "value": 999},
    ]
    pf = playerankFeatures()
    pf.set_features([features_with_unknown])
    result = pf.createFeature(weights_file)
    scores = {r["entity"]: r["value"] for r in result}
    assert scores[10] == pytest.approx(2.6)


def test_output_feature_name_is_playerank_score(weights_file):
    pf = playerankFeatures()
    pf.set_features([QUALITY_FEATURES])
    result = pf.createFeature(weights_file)
    assert all(r["feature"] == "playerankScore" for r in result)


def test_output_contains_all_entities(weights_file):
    pf = playerankFeatures()
    pf.set_features([QUALITY_FEATURES])
    result = pf.createFeature(weights_file)
    assert {r["entity"] for r in result} == {10, 20}


def test_output_values_are_floats(weights_file):
    pf = playerankFeatures()
    pf.set_features([QUALITY_FEATURES])
    result = pf.createFeature(weights_file)
    assert all(isinstance(r["value"], float) for r in result)


def test_empty_features_returns_empty(weights_file):
    pf = playerankFeatures()
    pf.set_features([[]])
    result = pf.createFeature(weights_file)
    assert result == []
