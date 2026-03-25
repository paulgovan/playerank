import json
import pytest
from playerank.features.qualityFeatures import qualityFeatures


@pytest.fixture
def players_file(tmp_path):
    players = [
        {"wyId": 1, "role": {"name": "Goalkeeper", "code": "GK"}},
        {"wyId": 2, "role": {"name": "Midfielder", "code": "MD"}},
        {"wyId": 3, "role": {"name": "Forward", "code": "FW"}},
    ]
    f = tmp_path / "players.json"
    f.write_text(json.dumps(players))
    return str(f)


def make_event(matchId, teamId, playerId, eventId, subEventId, tags,
               matchPeriod, eventName, subEventName):
    return {
        "matchId": matchId,
        "teamId": teamId,
        "playerId": playerId,
        "eventId": eventId,
        "subEventId": subEventId,
        "tags": [{"id": t} for t in tags],
        "matchPeriod": matchPeriod,
        "eventName": eventName,
        "subEventName": subEventName,
    }


def test_accurate_simple_pass(tmp_path, players_file):
    events = [make_event(1, 100, 2, 8, 85, [1801], "1H", "Pass", "Simple pass")]
    f = tmp_path / "events.json"
    f.write_text(json.dumps(events))
    result = qualityFeatures().createFeature(str(f), players_file, entity="player")
    feature_names = {r["feature"] for r in result}
    assert "Pass-Simple pass-accurate" in feature_names


def test_inaccurate_simple_pass(tmp_path, players_file):
    events = [make_event(1, 100, 2, 8, 85, [1802], "1H", "Pass", "Simple pass")]
    f = tmp_path / "events.json"
    f.write_text(json.dumps(events))
    result = qualityFeatures().createFeature(str(f), players_file, entity="player")
    feature_names = {r["feature"] for r in result}
    assert "Pass-Simple pass-not accurate" in feature_names


def test_excludes_penalty_period(tmp_path, players_file):
    events = [
        make_event(1, 100, 2, 8, 85, [1801], "1H", "Pass", "Simple pass"),
        make_event(1, 100, 2, 8, 85, [1801], "P",  "Pass", "Simple pass"),
    ]
    f = tmp_path / "events.json"
    f.write_text(json.dumps(events))
    result = qualityFeatures().createFeature(str(f), players_file, entity="player")
    total = sum(r["value"] for r in result if r["feature"] == "Pass-Simple pass-accurate")
    assert total == 1  # penalty event excluded


def test_excludes_goalkeepers(tmp_path, players_file):
    events = [
        make_event(1, 100, 1, 8, 85, [1801], "1H", "Pass", "Simple pass"),  # GK
        make_event(1, 100, 2, 8, 85, [1801], "1H", "Pass", "Simple pass"),  # MD
    ]
    f = tmp_path / "events.json"
    f.write_text(json.dumps(events))
    result = qualityFeatures().createFeature(str(f), players_file, entity="player")
    entities = {r["entity"] for r in result}
    assert 1 not in entities
    assert 2 in entities


def test_team_entity_aggregates_all_players(tmp_path, players_file):
    events = [
        make_event(1, 100, 2, 8, 85, [1801], "1H", "Pass", "Simple pass"),
        make_event(1, 100, 3, 8, 85, [1801], "2H", "Pass", "Simple pass"),
    ]
    f = tmp_path / "events.json"
    f.write_text(json.dumps(events))
    result = qualityFeatures().createFeature(str(f), players_file, entity="team")
    doc = next(r for r in result if r["feature"] == "Pass-Simple pass-accurate")
    assert doc["entity"] == 100
    assert doc["value"] == 2


def test_foul_with_yellow_card(tmp_path, players_file):
    # Fouls (eventId=2) use direct tag aggregation, not subevent hierarchy
    events = [make_event(1, 100, 2, 2, 20, [1702], "1H", "Foul", "Normal foul")]
    f = tmp_path / "events.json"
    f.write_text(json.dumps(events))
    result = qualityFeatures().createFeature(str(f), players_file, entity="player")
    feature_names = {r["feature"] for r in result}
    assert "Foul-yellow card" in feature_names


def test_both_first_and_second_half_counted(tmp_path, players_file):
    events = [
        make_event(1, 100, 2, 8, 85, [1801], "1H", "Pass", "Simple pass"),
        make_event(1, 100, 2, 8, 85, [1801], "2H", "Pass", "Simple pass"),
    ]
    f = tmp_path / "events.json"
    f.write_text(json.dumps(events))
    result = qualityFeatures().createFeature(str(f), players_file, entity="player")
    doc = next(r for r in result if r["feature"] == "Pass-Simple pass-accurate")
    assert doc["value"] == 2
