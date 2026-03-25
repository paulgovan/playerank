import json
import pytest


@pytest.fixture
def players_file(tmp_path):
    """Players JSON with one goalkeeper and three outfield players."""
    players = [
        {"wyId": 1, "role": {"name": "Goalkeeper", "code": "GK"}},
        {"wyId": 2, "role": {"name": "Midfielder", "code": "MD"}},
        {"wyId": 3, "role": {"name": "Forward", "code": "FW"}},
        {"wyId": 4, "role": {"name": "Defender", "code": "DF"}},
    ]
    f = tmp_path / "players.json"
    f.write_text(json.dumps(players))
    return str(f)
