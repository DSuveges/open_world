from fastapi.testclient import TestClient

from open_world.viz.api import app


def _write_dataset(tmp_path, make_frame):
    frame = make_frame(
        [
            {"state": "s1", "province": "p1", "districtId": "a", "elevation": 10},
            {"state": "s1", "province": "p1", "districtId": "b", "elevation": 20},
            {"state": "s2", "province": "p2", "districtId": "c", "elevation": 9000},
        ]
    )
    frame.write_parquet(tmp_path / "part-0.parquet")


def test_get_graph_returns_nodes_edges_and_summary(tmp_path, make_frame, monkeypatch):
    _write_dataset(tmp_path, make_frame)
    monkeypatch.setattr("open_world.viz.api.DEFAULT_DATA_PATH", tmp_path)

    response = TestClient(app).get("/api/graph")

    assert response.status_code == 200
    payload = response.json()
    assert {n["id"] for n in payload["nodes"]} == {"a", "b", "c"}
    assert payload["summary"]["is_valid"] is True


def test_index_serves_html(tmp_path, make_frame, monkeypatch):
    _write_dataset(tmp_path, make_frame)
    monkeypatch.setattr("open_world.viz.api.DEFAULT_DATA_PATH", tmp_path)

    response = TestClient(app).get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
