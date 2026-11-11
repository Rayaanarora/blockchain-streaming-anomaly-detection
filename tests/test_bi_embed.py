"""Superset embedding endpoints (src/api/bi.py): validation and guest-token request shape, no Superset needed."""
import json

import pytest
from fastapi.testclient import TestClient

import src.api.bi as bi
from src.api.app import create_app

EMBED = {"superset_url": "http://127.0.0.1:8088",
         "dashboards": {"live": {"id": 1, "title": "BSAD · Live pipeline", "description": "d", "embed_uuid": "u-live",
                                 "charts": 13, "refresh_seconds": 10},
                        "models": {"id": 3, "title": "BSAD · Model comparison", "description": "d", "embed_uuid": "u-models",
                                   "charts": 14, "refresh_seconds": 0}},
         "stream_dataset_ids": [11, 12]}


@pytest.fixture
def client(tmp_path, monkeypatch):
    f = tmp_path / "embed.json"
    f.write_text(json.dumps(EMBED))
    monkeypatch.setattr(bi, "EMBED_FILE", str(f))
    return TestClient(create_app())


class FakeSession:
    def __init__(self):
        self.calls = []

    def post(self, url, json=None, timeout=None):
        self.calls.append((url, json))

        class R:
            status_code = 200

            @staticmethod
            def raise_for_status():
                return None

            @staticmethod
            def json():
                return {"token": "guest-token-123"}
        return R()


def test_config_without_build_reports_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(bi, "EMBED_FILE", str(tmp_path / "missing.json"))
    r = TestClient(create_app()).get("/api/bi/config")
    assert r.status_code == 200 and r.json()["available"] is False


def test_guest_token_rejects_injection_and_unknown_dashboard(client):
    assert client.post("/api/bi/guest-token", params={"dashboard": "live", "run": "x' OR 1=1 --"}).status_code == 400
    assert client.post("/api/bi/guest-token", params={"dashboard": "nope"}).status_code == 404
    assert client.post("/api/bi/guest-token", params={"dashboard": "LIVE!"}).status_code == 422


def test_guest_token_scopes_stream_datasets_to_run(client, monkeypatch):
    fake = FakeSession()
    monkeypatch.setattr(bi.SupersetEmbedder, "_client", lambda self: fake)
    r = client.post("/api/bi/guest-token", params={"dashboard": "live", "run": "run_20261111_210240"})
    assert r.status_code == 200 and r.json() == {"token": "guest-token-123", "run": "run_20261111_210240"}
    url, body = fake.calls[0]
    assert url.endswith("/api/v1/security/guest_token/")
    assert body["resources"] == [{"type": "dashboard", "id": "u-live"}]
    assert body["rls"] == [{"dataset": 11, "clause": "run_id = 'run_20261111_210240'"},
                           {"dataset": 12, "clause": "run_id = 'run_20261111_210240'"}]
    assert "password" not in json.dumps(body)


def test_guest_token_without_run_has_no_rls(client, monkeypatch):
    fake = FakeSession()
    monkeypatch.setattr(bi.SupersetEmbedder, "_client", lambda self: fake)
    assert client.post("/api/bi/guest-token", params={"dashboard": "models"}).status_code == 200
    assert fake.calls[0][1]["rls"] == []
