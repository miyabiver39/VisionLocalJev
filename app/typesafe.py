"""
app/typesafe.py
---------------
TypeSafe System One API (https://docs.typesafe.ai/api) の型定義・検証・HTTP クライアント。

判定エンジンの入出力はすべてこの形式に揃える:

  request : {"model", "state", "questions": {id: {"type", "instructions", "criteria"}}}
  response: {"model", "answers": {id: Answer}, "usage": {"input_tokens", "output_tokens"}}

  Answer (noul)  : {"type": "noul", "noul": P(yes)}
  Answer (choice): {"type": "choice", "choice", "probabilities": {option: p}, "confidence"}
  Answer (score) : {"type": "score", "score", "legend": {"0": desc, ...}, "probabilities": {"0": p, ...}, "confidence"}

TypeSafe 本家 (jev-latest) は画像を受け付けないため、画像の渡し方は互換モデルごとに選ぶ (image_mode):
  "images"        : トップレベル "images": [data URL]           (imajev)
  "state_content" : state を OpenAI 形式の content 配列にする     (Qev)
  "none"          : 画像を送らない                               (TypeSafe 本家 Jev / Kev)
"""

import base64
import math
from typing import Any, Dict, List, Optional

import requests

IMAGE_MODES = ("images", "state_content", "none")
_PROB_TOLERANCE = 1e-3


class DecisionEngineError(Exception):
    """判定エンジン (リモート推論サーバー) の失敗。黙ってフォールバックせず、呼び出し元でエラーとして扱う。

    kind: "connection" | "timeout" | "http" | "invalid_response" | "config"
    """

    def __init__(self, kind: str, message: str, status: Optional[int] = None, body: Any = None):
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.status = status
        self.body = body

    def to_dict(self) -> Dict[str, Any]:
        return {"kind": self.kind, "message": self.message, "status": self.status}


# ===================== Question 定義 (プリセット YAML → TypeSafe) =====================

def preset_to_questions(preset: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """プリセット YAML の questions を TypeSafe の questions マップに変換する (label は除外)。

    旧形式 (choices / rubric / hypothesis) のカスタムプリセットも読み込めるよう変換する。
    """
    questions: Dict[str, Dict[str, Any]] = {}
    for q in preset.get("questions", []):
        qtype = q.get("type", "choice")
        out: Dict[str, Any] = {"type": qtype}
        instructions = q.get("instructions") or q.get("hypothesis") or q.get("rubric") or q.get("label") or q["id"]
        out["instructions"] = instructions
        if qtype == "choice":
            criteria = q.get("criteria")
            if criteria is None:  # 旧形式: choices のリスト
                criteria = {c: None for c in q.get("choices", [])}
            out["criteria"] = dict(criteria)
        elif qtype == "score":
            criteria = q.get("criteria")
            if criteria is None:  # 旧形式: 0〜1 の rubric のみ → 5 段階に変換
                criteria = ["None", "Low", "Moderate", "High", "Critical"]
            out["criteria"] = list(criteria)
        elif qtype == "noul":
            if q.get("criteria"):
                out["criteria"] = dict(q["criteria"])
        else:
            raise ValueError(f"unknown question type '{qtype}' in preset '{preset.get('id')}'")
        questions[q["id"]] = out
    return questions


def validate_questions(questions: Dict[str, Dict[str, Any]]):
    """TypeSafe API の制約 (Choice は 1〜255 択、Score は 2〜10 段階) を満たすか検査する。"""
    for qid, q in questions.items():
        if q["type"] == "choice" and not (1 <= len(q.get("criteria", {})) <= 255):
            raise ValueError(f"question '{qid}': choice needs 1-255 options")
        if q["type"] == "score" and not (2 <= len(q.get("criteria", [])) <= 10):
            raise ValueError(f"question '{qid}': score needs 2-10 levels")


# ===================== Answer の組み立て・検証 =====================

def _confidence(probs: List[float]) -> float:
    """確率分布の集中度 (p_max - 1/K) / (1 - 1/K)。TypeSafe の正式な式は非公開のため近似。"""
    k = len(probs)
    if k <= 1:
        return 1.0
    return max(0.0, (max(probs) - 1.0 / k) / (1.0 - 1.0 / k))


def choice_answer(probabilities: Dict[str, float]) -> Dict[str, Any]:
    choice = max(probabilities.items(), key=lambda kv: kv[1])[0]
    return {
        "type": "choice",
        "choice": choice,
        "probabilities": {k: float(v) for k, v in probabilities.items()},
        "confidence": round(_confidence(list(probabilities.values())), 4),
    }


def score_answer_from_level_probs(level_probs: List[float], criteria: List[Any]) -> Dict[str, Any]:
    expected = sum(i * p for i, p in enumerate(level_probs))
    return {
        "type": "score",
        "score": round(expected, 4),
        "legend": {str(i): (c if isinstance(c, str) else str(c)) for i, c in enumerate(criteria)},
        "probabilities": {str(i): float(p) for i, p in enumerate(level_probs)},
        "confidence": round(_confidence(level_probs), 4),
    }


def score_answer_from_unit(value: float, criteria: List[Any]) -> Dict[str, Any]:
    """0〜1 の連続値を、期待値が一致するよう隣接 2 段階に配分した Score 回答に変換する。"""
    n = len(criteria)
    pos = max(0.0, min(1.0, value)) * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    probs = [0.0] * n
    probs[lo] += 1.0 - (pos - lo)
    probs[hi] += pos - lo
    return score_answer_from_level_probs(probs, criteria)


def noul_answer(p_yes: float) -> Dict[str, Any]:
    return {"type": "noul", "noul": float(max(0.0, min(1.0, p_yes)))}


def normalized_value(answer: Dict[str, Any]) -> float:
    """アラート判定用に 0〜1 へ正規化した値 (Score は score / (段階数-1)、Noul は P(yes))。"""
    if answer["type"] == "score":
        n = len(answer["legend"])
        return answer["score"] / (n - 1) if n > 1 else 0.0
    if answer["type"] == "noul":
        return answer["noul"]
    return answer.get("confidence", 0.0)


def _is_prob(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 - _PROB_TOLERANCE <= v <= 1.0 + _PROB_TOLERANCE


def validate_response(data: Any, questions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """サーバー応答が TypeSafe の Response 形式に合っているか検査し、TypeSafe のフィールドだけに正規化して返す。"""
    def bad(msg):
        raise DecisionEngineError("invalid_response", f"TypeSafe response invalid: {msg}", body=data)

    if not isinstance(data, dict) or not isinstance(data.get("answers"), dict):
        bad("missing 'answers' object")
    answers: Dict[str, Any] = {}
    for qid, q in questions.items():
        a = data["answers"].get(qid)
        if not isinstance(a, dict):
            bad(f"no answer for question '{qid}'")
        if a.get("type") != q["type"]:
            bad(f"answer '{qid}' has type {a.get('type')!r}, expected {q['type']!r}")
        if q["type"] == "noul":
            if not _is_prob(a.get("noul")):
                bad(f"answer '{qid}': 'noul' must be a number in [0, 1]")
            answers[qid] = noul_answer(a["noul"])
        elif q["type"] == "choice":
            probs = a.get("probabilities")
            options = set(q["criteria"].keys())
            if not isinstance(probs, dict) or set(probs.keys()) != options or not all(_is_prob(v) for v in probs.values()):
                bad(f"answer '{qid}': 'probabilities' must map exactly the options {sorted(options)} to [0, 1]")
            if abs(sum(probs.values()) - 1.0) > 0.02:
                bad(f"answer '{qid}': probabilities sum to {sum(probs.values()):.3f}, expected 1")
            if a.get("choice") not in options:
                bad(f"answer '{qid}': 'choice' {a.get('choice')!r} is not one of the options")
            answers[qid] = {"type": "choice", "choice": a["choice"],
                            "probabilities": {k: float(v) for k, v in probs.items()},
                            "confidence": float(a.get("confidence", _confidence(list(probs.values()))))}
        elif q["type"] == "score":
            n = len(q["criteria"])
            probs = a.get("probabilities")
            keys = {str(i) for i in range(n)}
            if not isinstance(probs, dict) or set(probs.keys()) != keys or not all(_is_prob(v) for v in probs.values()):
                bad(f"answer '{qid}': 'probabilities' must map levels {sorted(keys)} to [0, 1]")
            score = a.get("score")
            if not isinstance(score, (int, float)) or isinstance(score, bool) or not (-_PROB_TOLERANCE <= score <= n - 1 + _PROB_TOLERANCE):
                bad(f"answer '{qid}': 'score' must be a number in [0, {n - 1}]")
            level_probs = [float(probs[str(i)]) for i in range(n)]
            ans = score_answer_from_level_probs(level_probs, q["criteria"])
            ans["score"] = float(score)
            if "confidence" in a:
                ans["confidence"] = float(a["confidence"])
            answers[qid] = ans
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    return {
        "model": data.get("model", ""),
        "answers": answers,
        "usage": {"input_tokens": int(usage.get("input_tokens", 0) or 0),
                  "output_tokens": int(usage.get("output_tokens", 0) or 0)},
    }


# ===================== HTTP クライアント =====================

def jpeg_data_url(jpeg_bytes: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg_bytes).decode("ascii")


class TypeSafeClient:
    """TypeSafe System One 互換サーバー (TypeSafe 本家 / imajev / Qev / Kev) へのクライアント。"""

    def __init__(self, url: str, model: str = "jev-latest", api_key: str = "",
                 image_mode: str = "images", timeout: float = 10.0):
        if image_mode not in IMAGE_MODES:
            raise DecisionEngineError("config", f"DJEV_IMAGE_MODE must be one of {IMAGE_MODES}, got {image_mode!r}")
        url = url.rstrip("/")
        self.url = url if url.endswith("/v1/systemone") else url + "/v1/systemone"
        self.model = model
        self.api_key = api_key
        self.image_mode = image_mode
        self.timeout = timeout

    def build_request(self, state: Any, questions: Dict[str, Dict[str, Any]],
                      image_jpeg: Optional[bytes] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"model": self.model, "state": state, "questions": questions}
        if image_jpeg is not None and self.image_mode != "none":
            url = jpeg_data_url(image_jpeg)
            if self.image_mode == "images":
                body["images"] = [url]
            else:  # state_content
                text = state if isinstance(state, str) else str(state)
                body["state"] = [{"type": "text", "text": text},
                                 {"type": "image_url", "image_url": {"url": url}}]
        return body

    def evaluate(self, state: Any, questions: Dict[str, Dict[str, Any]],
                 image_jpeg: Optional[bytes] = None) -> Dict[str, Any]:
        body = self.build_request(state, questions, image_jpeg)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            resp = requests.post(self.url, json=body, headers=headers, timeout=self.timeout)
        except requests.Timeout as e:
            raise DecisionEngineError("timeout", f"TypeSafe server timed out after {self.timeout}s ({self.url})") from e
        except requests.RequestException as e:
            raise DecisionEngineError("connection", f"TypeSafe server unreachable ({self.url}): {e}") from e

        if resp.status_code != 200:
            try:
                err_body = resp.json()
            except ValueError:
                err_body = resp.text[:500]
            reason = {401: "unauthorized (check DJEV_API_KEY)", 422: "request rejected by validation",
                      429: "rate limited", 529: "server overloaded"}.get(resp.status_code, "server error")
            raise DecisionEngineError("http", f"TypeSafe server returned HTTP {resp.status_code}: {reason}",
                                      status=resp.status_code, body=err_body)
        try:
            data = resp.json()
        except ValueError as e:
            raise DecisionEngineError("invalid_response", "TypeSafe server returned non-JSON body",
                                      status=200, body=resp.text[:500]) from e
        return validate_response(data, questions)
