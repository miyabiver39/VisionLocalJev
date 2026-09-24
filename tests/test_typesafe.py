"""TypeSafe System One 形式への準拠と、リモート失敗時にフォールバックしないことのテスト (#15)。"""
import asyncio
from unittest import mock

import numpy as np
import pytest
import requests

from app.decision import DecisionEngine
from app.typesafe import (
    DecisionEngineError,
    TypeSafeClient,
    preset_to_questions,
    validate_response,
)

PRESETS = ["security", "fire_disaster", "nursing_care", "river_flood", "factory_safety", "railway_platform"]


def _response(questions, model="imajev-4b"):
    """questions に対する正しい TypeSafe レスポンスを作る。"""
    answers = {}
    for qid, q in questions.items():
        if q["type"] == "noul":
            answers[qid] = {"type": "noul", "noul": 0.9}
        elif q["type"] == "choice":
            opts = list(q["criteria"].keys())
            probs = {o: 0.0 for o in opts}
            probs[opts[-1]] = 1.0
            answers[qid] = {"type": "choice", "choice": opts[-1], "probabilities": probs, "confidence": 1.0}
        else:
            n = len(q["criteria"])
            probs = {str(i): 0.0 for i in range(n)}
            probs[str(n - 1)] = 1.0
            answers[qid] = {"type": "score", "score": float(n - 1),
                            "legend": {str(i): c for i, c in enumerate(q["criteria"])},
                            "probabilities": probs, "confidence": 1.0}
    return {"model": model, "answers": answers, "usage": {"input_tokens": 321, "output_tokens": 0}}


class _Resp:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body
        self.text = str(body)

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


@pytest.fixture
def remote_engine():
    return DecisionEngine(mode="remote", remote_url="http://typesafe.test", presets_dir="app/presets",
                          remote_model="imajev-4b", api_key="secret", image_mode="images", timeout=5)


# ---------- questions (プリセット → TypeSafe) ----------

@pytest.mark.parametrize("preset_id", PRESETS)
def test_preset_questions_are_typesafe_shaped(decision_engine, preset_id):
    questions = decision_engine.questions[preset_id]
    for qid, q in questions.items():
        assert set(q) <= {"type", "instructions", "criteria"}, "label などアプリ用の項目を送らない"
        assert q["instructions"]
        if q["type"] == "choice":
            assert isinstance(q["criteria"], dict) and 1 <= len(q["criteria"]) <= 255
        elif q["type"] == "score":
            assert isinstance(q["criteria"], list) and 2 <= len(q["criteria"]) <= 10


def test_legacy_preset_format_is_converted():
    legacy = {"id": "old", "questions": [
        {"id": "c", "type": "choice", "label": "x", "choices": ["a", "b"]},
        {"id": "s", "type": "score", "label": "y", "rubric": "rate it"},
        {"id": "n", "type": "noul", "label": "z", "hypothesis": "is it bad?"},
    ]}
    q = preset_to_questions(legacy)
    assert q["c"] == {"type": "choice", "instructions": "x", "criteria": {"a": None, "b": None}}
    assert q["s"]["instructions"] == "rate it" and len(q["s"]["criteria"]) == 5
    assert q["n"] == {"type": "noul", "instructions": "is it bad?"}


# ---------- embedded (CPU エミュレータ) も TypeSafe 形式で返す ----------

@pytest.mark.parametrize("preset_id", PRESETS)
def test_embedded_answers_validate_as_typesafe(decision_engine, preset_id):
    res = decision_engine.evaluate("A person has fallen onto the floor near the fire.", preset_id)
    questions = decision_engine.questions[preset_id]
    normalized = validate_response({"model": res["model"], "answers": res["answers"], "usage": res["usage"]}, questions)
    assert normalized["answers"].keys() == questions.keys()
    for qid, a in res["answers"].items():
        if a["type"] == "score":
            assert set(a) == {"type", "score", "legend", "probabilities", "confidence"}
            assert 0 <= a["score"] <= len(a["legend"]) - 1
        elif a["type"] == "choice":
            assert set(a) == {"type", "choice", "probabilities", "confidence"}
        else:
            assert set(a) == {"type", "noul"}


# ---------- リモート: リクエスト形式 ----------

def test_remote_request_body_and_auth(remote_engine):
    questions = remote_engine.questions["fire_disaster"]
    frame = np.zeros((60, 80, 3), np.uint8)
    with mock.patch("app.typesafe.requests.post", return_value=_Resp(200, _response(questions))) as post:
        res = remote_engine.evaluate_multimodal(frame, "fire_disaster", "c", "c", "state text")
    url = post.call_args.args[0]
    body = post.call_args.kwargs["json"]
    headers = post.call_args.kwargs["headers"]
    assert url == "http://typesafe.test/v1/systemone"
    assert headers["Authorization"] == "Bearer secret"
    assert body["model"] == "imajev-4b" and body["state"] == "state text"
    assert body["questions"] == questions
    assert len(body["images"]) == 1 and body["images"][0].startswith("data:image/jpeg;base64,")
    assert res["model"] == "imajev-4b" and res["usage"]["input_tokens"] == 321
    assert res["answers"]["hazard_type"]["choice"] == "open_flame"
    assert res["is_alert"] is True


def test_state_content_and_none_image_modes():
    q = {"n": {"type": "noul", "instructions": "?"}}
    qev = TypeSafeClient("http://x", image_mode="state_content").build_request("hello", q, b"\xff\xd8jpeg")
    assert qev["state"][0] == {"type": "text", "text": "hello"}
    assert qev["state"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "images" not in qev
    jev = TypeSafeClient("http://x", image_mode="none").build_request("hello", q, b"\xff\xd8jpeg")
    assert jev == {"model": "jev-latest", "state": "hello", "questions": q}


def test_invalid_image_mode_is_config_error():
    with pytest.raises(DecisionEngineError) as e:
        TypeSafeClient("http://x", image_mode="bogus")
    assert e.value.kind == "config"


# ---------- リモート: 失敗時はフォールバックせずエラー ----------

@pytest.mark.parametrize("side_effect,kind,status", [
    (requests.ConnectionError("refused"), "connection", None),
    (requests.Timeout("slow"), "timeout", None),
    (_Resp(401, {"error": "bad key"}), "http", 401),
    (_Resp(422, {"error": "invalid_request"}), "http", 422),
    (_Resp(529, {"error": "overloaded"}), "http", 529),
    (_Resp(200, ValueError("not json")), "invalid_response", 200),
])
def test_remote_failures_raise_without_fallback(remote_engine, side_effect, kind, status):
    kwargs = {"side_effect": side_effect} if isinstance(side_effect, Exception) else {"return_value": side_effect}
    with mock.patch("app.typesafe.requests.post", **kwargs), \
         mock.patch.object(remote_engine, "_evaluate_embedded") as emulator:
        with pytest.raises(DecisionEngineError) as e:
            remote_engine.evaluate("state", "security")
    assert e.value.kind == kind and e.value.status == status
    emulator.assert_not_called()


@pytest.mark.parametrize("mutate,msg", [
    (lambda r: r["answers"].pop("threat_score"), "no answer"),
    (lambda r: r["answers"]["action_type"]["probabilities"].pop("loitering"), "exactly the options"),
    (lambda r: r["answers"]["action_type"].update(choice="dancing"), "not one of the options"),
    (lambda r: r["answers"]["requires_alert"].update(noul=1.7), "'noul' must be"),
    (lambda r: r["answers"]["threat_score"].update(type="noul"), "expected 'score'"),
    (lambda r: r["answers"]["threat_score"].update(score=9), "'score' must be"),
])
def test_malformed_responses_are_rejected(remote_engine, mutate, msg):
    body = _response(remote_engine.questions["security"])
    mutate(body)
    with mock.patch("app.typesafe.requests.post", return_value=_Resp(200, body)):
        with pytest.raises(DecisionEngineError) as e:
            remote_engine.evaluate("state", "security")
    assert e.value.kind == "invalid_response" and msg in e.value.message


def test_trigger_scenario_returns_502_on_engine_error(client, remote_engine):
    from app import main
    with mock.patch.object(main, "decision_engine", remote_engine), \
         mock.patch("app.typesafe.requests.post", side_effect=requests.ConnectionError("refused")):
        res = client.post("/api/trigger_scenario", json={"state": "fire", "preset_id": "security", "freeze_seconds": 0})
    assert res.status_code == 502
    assert res.json()["detail"]["decision_engine_error"]["kind"] == "connection"


def test_trigger_scenario_returns_504_on_timeout(client, remote_engine):
    from app import main
    with mock.patch.object(main, "decision_engine", remote_engine), \
         mock.patch("app.typesafe.requests.post", side_effect=requests.Timeout("slow")):
        res = client.post("/api/trigger_scenario", json={"state": "fire", "preset_id": "security", "freeze_seconds": 0})
    assert res.status_code == 504


def test_pipeline_broadcasts_decision_error_and_skips_alert(client, remote_engine):
    from app import main
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    with mock.patch.object(main, "decision_engine", remote_engine), \
         mock.patch("app.typesafe.requests.post", side_effect=requests.ConnectionError("refused")), \
         mock.patch.object(main.webhook_dispatcher, "dispatch_alert_async") as dispatch, \
         mock.patch.object(main, "broadcast_ws", new=mock.AsyncMock()) as broadcast:
        result = asyncio.run(main.process_sample("cam_main", frame, "security", False))
    assert result["type"] == "decision_error"
    assert result["error"]["kind"] == "connection"
    broadcast.assert_awaited_once()
    assert broadcast.await_args.args[0]["type"] == "decision_error"
    dispatch.assert_not_called()
