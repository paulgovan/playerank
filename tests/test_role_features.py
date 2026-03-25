import json
import pytest
from playerank.features.roleFeatures import roleFeatures


@pytest.fixture
def role_matrix_file(tmp_path):
    # Maps avg_x -> avg_y -> role label based on x position thirds
    matrix = {str(x): {str(y): str(x // 34) for y in range(101)} for x in range(101)}
    f = tmp_path / "role_matrix.json"
    f.write_text(json.dumps(matrix))
    return str(f)


CENTER_FEATURES = [
    {"match": 1, "entity": 10, "feature": "avg_x", "value": 20},
    {"match": 1, "entity": 10, "feature": "avg_y", "value": 50},
    {"match": 1, "entity": 10, "feature": "n_events", "value": 30},
    {"match": 1, "entity": 20, "feature": "avg_x", "value": 80},
    {"match": 1, "entity": 20, "feature": "avg_y", "value": 50},
    {"match": 1, "entity": 20, "feature": "n_events", "value": 25},
]


def test_assigns_role_from_matrix(role_matrix_file):
    rf = roleFeatures()
    rf.set_features([CENTER_FEATURES])
    result = rf.createFeature(role_matrix_file)
    roles = {r["entity"]: r["value"] for r in result}
    assert roles[10] == "0"   # 20 // 34 = 0
    assert roles[20] == "2"   # 80 // 34 = 2


def test_output_feature_name_is_role_cluster(role_matrix_file):
    rf = roleFeatures()
    rf.set_features([CENTER_FEATURES])
    result = rf.createFeature(role_matrix_file)
    assert all(r["feature"] == "roleCluster" for r in result)


def test_output_contains_all_entities(role_matrix_file):
    rf = roleFeatures()
    rf.set_features([CENTER_FEATURES])
    result = rf.createFeature(role_matrix_file)
    assert {r["entity"] for r in result} == {10, 20}


def test_output_contains_match_id(role_matrix_file):
    rf = roleFeatures()
    rf.set_features([CENTER_FEATURES])
    result = rf.createFeature(role_matrix_file)
    assert all(r["match"] == 1 for r in result)
