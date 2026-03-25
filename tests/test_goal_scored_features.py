import json
import pytest
from playerank.features.goalScoredFeatures import goalScoredFeatures


@pytest.fixture
def matches_file(tmp_path):
    matches = [
        {"wyId": 1, "teamsData": {"100": {"score": 2}, "200": {"score": 1}}},
        {"wyId": 2, "teamsData": {"100": {"score": 0}, "300": {"score": 3}}},
    ]
    f = tmp_path / "matches.json"
    f.write_text(json.dumps(matches))
    return str(f)


def test_extracts_scores(matches_file):
    result = goalScoredFeatures().createFeature(matches_file)
    scores = {(r["match"], r["entity"]): r["value"] for r in result}
    assert scores[(1, "100")] == 2
    assert scores[(1, "200")] == 1
    assert scores[(2, "300")] == 3


def test_output_feature_name(matches_file):
    result = goalScoredFeatures().createFeature(matches_file)
    assert all(r["feature"] == "goal-scored" for r in result)


def test_output_count(matches_file):
    result = goalScoredFeatures().createFeature(matches_file)
    # 2 matches × 2 teams = 4 entries
    assert len(result) == 4


def test_skips_match_without_teams_data(tmp_path):
    matches = [
        {"wyId": 99},  # no teamsData key
        {"wyId": 1, "teamsData": {"100": {"score": 1}, "200": {"score": 0}}},
    ]
    f = tmp_path / "matches.json"
    f.write_text(json.dumps(matches))
    result = goalScoredFeatures().createFeature(str(f))
    assert len(result) == 2


def test_required_fields_present(matches_file):
    result = goalScoredFeatures().createFeature(matches_file)
    for doc in result:
        assert "match" in doc
        assert "entity" in doc
        assert "feature" in doc
        assert "value" in doc
