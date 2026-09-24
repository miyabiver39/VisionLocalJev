import base64
import threading
from unittest import mock

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.testclient import TestClient

from app.security import BasicAuthMiddleware
from app.webhook import WebhookDispatcher, WebhookConfig


def _synthetic_camera(camera_id="cam_test", preset_id="security"):
    return {
        "camera_id": camera_id, "name": "Test", "source_type": "synthetic",
        "source_url": "synthetic", "preset_id": preset_id, "sample_fps": 1.0,
    }


def test_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_presets_endpoint_lists_all(client):
    assert len(client.get("/api/presets").json()) == 6


def test_camera_crud_and_validation(client):
    assert client.post("/api/cameras", json=_synthetic_camera(preset_id="bogus")).status_code == 400

    assert client.post("/api/cameras", json=_synthetic_camera()).status_code == 200
    try:
        assert client.put("/api/cameras/cam_test", json={"preset_id": "bogus"}).status_code == 400
        assert client.put("/api/cameras/cam_test", json={"sample_fps": 0}).status_code == 400

        res = client.put("/api/cameras/cam_test", json={"preset_id": "nursing_care", "sample_fps": 100})
        assert res.status_code == 200
        cam = res.json()["camera"]
        assert cam["preset_id"] == "nursing_care"
        assert cam["sample_fps"] == 10.0  # clamped
    finally:
        assert client.delete("/api/cameras/cam_test").status_code == 200
    assert client.get("/api/cameras/cam_test").status_code == 404


def test_jpeg_camera_requires_http_url(client):
    body = _synthetic_camera(camera_id="cam_jpeg")
    body.update(source_type="jpeg_url", source_url="file:///etc/passwd")
    assert client.post("/api/cameras", json=body).status_code == 400


def test_trigger_scenario_rejects_unknown_preset(client):
    res = client.post("/api/trigger_scenario", json={"state": "fire", "preset_id": "bogus", "freeze_seconds": 0})
    assert res.status_code == 400


def test_trigger_scenario_alerts_and_attaches_sop(client):
    res = client.post("/api/trigger_scenario", json={
        "state": "A person is attempting to climb over the outer security barrier fence.",
        "preset_id": "security", "freeze_seconds": 0,
    })
    assert res.status_code == 200
    body = res.json()
    assert body["is_alert"] is True
    assert body["sop_action"]["id"] == "sop_trespass_breach"


def test_webhook_url_scheme_validated(client):
    res = client.post("/api/webhooks", json={"id": "w", "name": "w", "url": "file:///etc/passwd"})
    assert res.status_code == 422


def test_visual_rag_register_size_limit_and_roundtrip(client):
    too_big = "A" * (16 * 1024 * 1024)  # decodes to ~12MB > 10MB default limit
    assert client.post("/api/visual_rag/register", json={
        "title": "big", "category": "security", "image_base64": too_big,
    }).status_code == 413

    _, buf = cv2.imencode(".png", np.full((60, 80, 3), 127, dtype=np.uint8))
    data_url = "data:image/png;base64," + base64.b64encode(buf).decode()
    res = client.post("/api/visual_rag/register", json={
        "title": "gray", "category": "security", "image_base64": data_url,
    })
    assert res.status_code == 200
    ref_id = res.json()["reference"]["ref_id"]
    assert client.delete(f"/api/visual_rag/references/{ref_id}").status_code == 200


def test_basic_auth_middleware_protects_http_and_websocket():
    app = FastAPI()
    app.add_middleware(BasicAuthMiddleware, credentials=("admin", "secret"))

    @app.get("/data")
    def data():
        return {"ok": True}

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        await websocket.send_text("hi")
        await websocket.close()

    c = TestClient(app)
    assert c.get("/data").status_code == 401
    assert c.get("/data").headers["www-authenticate"].startswith("Basic")
    assert c.get("/data", auth=("admin", "wrong")).status_code == 401
    assert c.get("/data", auth=("admin", "secret")).status_code == 200
    assert c.get("/healthz").status_code == 200

    rejected = False
    try:
        with c.websocket_connect("/ws") as ws_conn:
            ws_conn.receive_text()
    except Exception:
        rejected = True
    assert rejected

    token = base64.b64encode(b"admin:secret").decode()
    with c.websocket_connect("/ws", headers={"Authorization": f"Basic {token}"}) as ws_conn:
        assert ws_conn.receive_text() == "hi"


def test_webhook_cooldown_is_atomic_under_concurrency():
    d = WebhookDispatcher()
    d.add_or_update(WebhookConfig(id="h", name="h", url="http://127.0.0.1:9/x", min_score=0.5, cooldown_seconds=60))
    with mock.patch.object(d, "_send_payload") as send:
        threads = [
            threading.Thread(target=d.dispatch_alert_async, args=({"camera_id": "c", "score": 0.9},))
            for _ in range(20)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        d.executor.shutdown(wait=True)
    assert send.call_count == 1


def test_visual_anomaly_with_linked_sop_produces_alert(client):
    """Regression for #12: the linked SOP path must not raise (SOPDocument has no to_dict)."""
    import asyncio
    from app import main

    fake_match = {
        "top_match": {"ref_id": "r", "title": "外周フェンス乗り越え侵入", "is_anomaly": True,
                      "similarity": 0.97, "sop_id": "sop_trespass_breach"},
        "similarity": 0.97, "anomaly_score": 0.9, "is_anomalous": True, "matches": [], "latency_ms": 0.1,
    }
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    with mock.patch.object(main.visual_rag_engine, "match_frame", return_value=fake_match), \
         mock.patch.object(main.webhook_dispatcher, "dispatch_alert_async") as dispatch:
        result = asyncio.run(main.process_sample("cam_main", frame, "security", False))

    assert result["decision"]["is_alert"] is True
    assert result["decision"]["alert_reason"].startswith("Visual Anomaly")
    assert result["rag"]["source"] == "visual_example_rag"
    assert result["rag"]["sop"]["id"] == "sop_trespass_breach"
    payload = dispatch.call_args[0][0]
    assert payload["score"] >= 0.9 and payload["alert_reason"]
