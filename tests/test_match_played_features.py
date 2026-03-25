import json
import pytest
from playerank.features.matchPlayedFeatures import matchPlayedFeatures


@pytest.fixture
def players_file(tmp_path):
    players = [
        {"wyId": 1, "role": {"name": "Goalkeeper", "code": "GK"}},
        {"wyId": 2, "role": {"name": "Midfielder", "code": "MD"}},
        {"wyId": 3, "role": {"name": "Forward", "code": "FW"}},
        {"wyId": 4, "role": {"name": "Defender", "code": "DF"}},
    ]
    f = tmp_path / "players.json"
    f.write_text(json.dumps(players))
    return str(f)


def make_match(wyId, duration="Regular", lineup=None, bench=None, subs=None):
    return {
        "wyId": wyId,
        "dateutc": "2018-05-01 18:00:00",
        "duration": duration,
        "teamsData": {
            "100": {
                "hasFormation": 1,
                "score": 1,
                "formation": {
                    "lineup": lineup or [],
                    "bench": bench or [],
                    "substitutions": subs or [],
                },
            }
        },
    }


def test_regular_match_gives_90_minutes(tmp_path, players_file):
    match = make_match(1, lineup=[{"playerId": 2, "goals": 0}])
    f = tmp_path / "matches.json"
    f.write_text(json.dumps([match]))
    result = matchPlayedFeatures().createFeature(str(f), players_file)
    mins = next(r for r in result if r["entity"] == 2 and r["feature"] == "minutesPlayed")
    assert mins["value"] == 90


def test_extra_time_match_gives_120_minutes(tmp_path, players_file):
    match = make_match(1, duration="ExtraTime", lineup=[{"playerId": 2, "goals": 0}])
    f = tmp_path / "matches.json"
    f.write_text(json.dumps([match]))
    result = matchPlayedFeatures().createFeature(str(f), players_file)
    mins = next(r for r in result if r["entity"] == 2 and r["feature"] == "minutesPlayed")
    assert mins["value"] == 120


def test_substitution_splits_minutes_correctly(tmp_path, players_file):
    match = make_match(
        1,
        lineup=[{"playerId": 3, "goals": 0}],
        bench=[{"playerId": 4, "goals": 0}],
        subs=[{"playerOut": 3, "playerIn": 4, "minute": 60}],
    )
    f = tmp_path / "matches.json"
    f.write_text(json.dumps([match]))
    result = matchPlayedFeatures().createFeature(str(f), players_file)
    by_player = {r["entity"]: r["value"] for r in result if r["feature"] == "minutesPlayed"}
    assert by_player[3] == 60
    assert by_player[4] == 30


def test_excludes_goalkeepers(tmp_path, players_file):
    match = make_match(1, lineup=[
        {"playerId": 1, "goals": 0},  # GK
        {"playerId": 2, "goals": 0},  # MD
    ])
    f = tmp_path / "matches.json"
    f.write_text(json.dumps([match]))
    result = matchPlayedFeatures().createFeature(str(f), players_file)
    entities = {r["entity"] for r in result}
    assert 1 not in entities
    assert 2 in entities


def test_goals_scored_feature(tmp_path, players_file):
    match = make_match(1, lineup=[{"playerId": 3, "goals": 2}])
    f = tmp_path / "matches.json"
    f.write_text(json.dumps([match]))
    result = matchPlayedFeatures().createFeature(str(f), players_file)
    goal_doc = next(r for r in result if r["entity"] == 3 and r["feature"] == "goalScored")
    assert goal_doc["value"] == 2


def test_timestamp_feature_present(tmp_path, players_file):
    match = make_match(1, lineup=[{"playerId": 2, "goals": 0}])
    f = tmp_path / "matches.json"
    f.write_text(json.dumps([match]))
    result = matchPlayedFeatures().createFeature(str(f), players_file)
    ts = next(r for r in result if r["entity"] == 2 and r["feature"] == "timestamp")
    assert ts["value"] == "2018-05-01 18:00:00"
