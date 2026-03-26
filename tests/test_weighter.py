import json
import numpy as np
import pandas as pd
import pytest
from playerank.models.Weighter import Weighter


@pytest.fixture
def synthetic_df():
    np.random.seed(42)
    n = 100
    feat_a = np.random.rand(n)
    feat_b = np.random.rand(n)
    goal_scored = np.where(feat_a > 0.5, 1, -1).astype(float)
    return pd.DataFrame({"feat_a": feat_a, "feat_b": feat_b, "goal-scored": goal_scored})


def test_fit_sets_weights(synthetic_df, tmp_path):
    w = Weighter(label_type="w-dl")
    w.fit(synthetic_df, "goal-scored", filename=str(tmp_path / "weights.json"))
    assert hasattr(w, "weights_")
    assert len(w.weights_) == 2


def test_weights_normalized(synthetic_df, tmp_path):
    w = Weighter(label_type="w-dl")
    w.fit(synthetic_df, "goal-scored", filename=str(tmp_path / "weights.json"))
    assert sum(np.abs(w.weights_)) == pytest.approx(1.0)


def test_fit_saves_valid_json(synthetic_df, tmp_path):
    out = tmp_path / "weights.json"
    w = Weighter(label_type="w-dl")
    w.fit(synthetic_df, "goal-scored", filename=str(out))
    assert out.exists()
    weights = json.loads(out.read_text())
    assert isinstance(weights, dict)
    assert len(weights) == 2


def test_get_feature_names(synthetic_df, tmp_path):
    w = Weighter(label_type="w-dl")
    w.fit(synthetic_df, "goal-scored", filename=str(tmp_path / "weights.json"))
    assert set(w.get_feature_names()) == {"feat_a", "feat_b"}


def test_get_weights_shape(synthetic_df, tmp_path):
    w = Weighter(label_type="w-dl")
    w.fit(synthetic_df, "goal-scored", filename=str(tmp_path / "weights.json"))
    assert w.get_weights().shape == (2,)


def test_label_type_wd_l(synthetic_df, tmp_path):
    w = Weighter(label_type="wd-l")
    w.fit(synthetic_df, "goal-scored", filename=str(tmp_path / "weights.json"))
    assert hasattr(w, "weights_")
    assert sum(np.abs(w.weights_)) == pytest.approx(1.0)


def test_label_type_w_d_l(tmp_path):
    np.random.seed(0)
    n = 150
    feat = np.random.rand(n)
    target = np.random.choice([1, 0, -1], size=n).astype(float)
    df = pd.DataFrame({"feat": feat, "goal-scored": target})
    w = Weighter(label_type="w-d-l")
    w.fit(df, "goal-scored", filename=str(tmp_path / "weights.json"))
    assert hasattr(w, "weights_")


def test_label_type_l_wd(synthetic_df, tmp_path):
    w = Weighter(label_type="l-wd")
    w.fit(synthetic_df, "goal-scored", filename=str(tmp_path / "weights.json"))
    assert hasattr(w, "weights_")
    assert sum(np.abs(w.weights_)) == pytest.approx(1.0)


def test_label_type_dl_w(synthetic_df, tmp_path):
    w = Weighter(label_type="dl-w")
    w.fit(synthetic_df, "goal-scored", filename=str(tmp_path / "weights.json"))
    assert hasattr(w, "weights_")
    assert sum(np.abs(w.weights_)) == pytest.approx(1.0)


def test_lack_of_performance_weights_are_inverse_of_performance(synthetic_df, tmp_path):
    """l-wd weights should be the negation of wd-l weights (same SVM boundary, flipped labels)."""
    w_perf = Weighter(label_type="wd-l")
    w_perf.fit(synthetic_df, "goal-scored", filename=str(tmp_path / "perf.json"))

    w_waste = Weighter(label_type="l-wd")
    w_waste.fit(synthetic_df, "goal-scored", filename=str(tmp_path / "waste.json"))

    np.testing.assert_allclose(w_perf.weights_, -w_waste.weights_, atol=1e-6)


def test_lack_of_performance_highest_waste_is_lowest_performance(synthetic_df, tmp_path):
    """The feature ranked highest for waste should be ranked lowest for performance."""
    w_perf = Weighter(label_type="wd-l")
    w_perf.fit(synthetic_df, "goal-scored", filename=str(tmp_path / "perf.json"))

    w_waste = Weighter(label_type="l-wd")
    w_waste.fit(synthetic_df, "goal-scored", filename=str(tmp_path / "waste.json"))

    best_perf = w_perf.get_feature_names()[np.argmax(w_perf.weights_)]
    worst_waste = w_waste.get_feature_names()[np.argmax(w_waste.weights_)]
    # The feature most associated with winning should be most associated with not losing
    assert best_perf != worst_waste
