import json
import pytest
from playerank.features.centerOfPerformanceFeature import centerOfPerformanceFeature


@pytest.fixture
def players_file(tmp_path):
    players = [
        {"wyId": 1, "role": {"name": "Goalkeeper", "code": "GK"}},
        {"wyId": 2, "role": {"name": "Midfielder", "code": "MD"}},
    ]
    f = tmp_path / "players.json"
    f.write_text(json.dumps(players))
    return str(f)


def make_events(player_id, match_id, positions):
    return [
        {"playerId": player_id, "matchId": match_id, "positions": [{"x": x, "y": y}]}
        for x, y in positions
    ]


def test_computes_average_position(tmp_path, players_file):
    events = make_events(2, 1, [(40, 60)] * 20)
    f = tmp_path / "events.json"
    f.write_text(json.dumps(events))
    result = centerOfPerformanceFeature().createFeature(str(f), players_file)
    vals = {r["feature"]: r["value"] for r in result if r["entity"] == 2}
    assert vals["avg_x"] == 40
    assert vals["avg_y"] == 60


def test_below_min_events_excluded(tmp_path, players_file):
    # MIN_EVENTS=10; only 5 events → player should not appear in output
    events = make_events(2, 1, [(50, 50)] * 5)
    f = tmp_path / "events.json"
    f.write_text(json.dumps(events))
    result = centerOfPerformanceFeature().createFeature(str(f), players_file)
    assert len(result) == 0


def test_exactly_min_events_excluded(tmp_path, players_file):
    # Exactly 10 events — condition is count > MIN_EVENTS, so 10 is still excluded
    events = make_events(2, 1, [(50, 50)] * 10)
    f = tmp_path / "events.json"
    f.write_text(json.dumps(events))
    result = centerOfPerformanceFeature().createFeature(str(f), players_file)
    assert len(result) == 0


def test_above_min_events_included(tmp_path, players_file):
    events = make_events(2, 1, [(50, 50)] * 11)
    f = tmp_path / "events.json"
    f.write_text(json.dumps(events))
    result = centerOfPerformanceFeature().createFeature(str(f), players_file)
    assert any(r["entity"] == 2 for r in result)


def test_excludes_goalkeepers(tmp_path, players_file):
    gk = make_events(1, 1, [(50, 50)] * 20)
    md = make_events(2, 1, [(30, 70)] * 20)
    f = tmp_path / "events.json"
    f.write_text(json.dumps(gk + md))
    result = centerOfPerformanceFeature().createFeature(str(f), players_file)
    entities = {r["entity"] for r in result}
    assert 1 not in entities
    assert 2 in entities


def test_excludes_referee(tmp_path, players_file):
    # playerId=0 represents the referee
    referee = make_events(0, 1, [(50, 50)] * 20)
    md = make_events(2, 1, [(30, 70)] * 20)
    f = tmp_path / "events.json"
    f.write_text(json.dumps(referee + md))
    result = centerOfPerformanceFeature().createFeature(str(f), players_file)
    assert all(r["entity"] != 0 for r in result)


def test_n_events_feature(tmp_path, players_file):
    events = make_events(2, 1, [(40, 60)] * 15)
    f = tmp_path / "events.json"
    f.write_text(json.dumps(events))
    result = centerOfPerformanceFeature().createFeature(str(f), players_file)
    n_doc = next(r for r in result if r["entity"] == 2 and r["feature"] == "n_events")
    assert n_doc["value"] == 15


def test_average_over_varied_positions(tmp_path, players_file):
    positions = [(20, 40), (40, 60), (60, 80)] * 5  # 15 events, avg = (40, 60)
    events = make_events(2, 1, positions)
    f = tmp_path / "events.json"
    f.write_text(json.dumps(events))
    result = centerOfPerformanceFeature().createFeature(str(f), players_file)
    vals = {r["feature"]: r["value"] for r in result if r["entity"] == 2}
    assert vals["avg_x"] == 40
    assert vals["avg_y"] == 60
