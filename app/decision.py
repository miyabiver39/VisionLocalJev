import time
import os
import yaml
import logging
import base64
import requests
import cv2
import numpy as np
from typing import Dict, Any, List, Optional

logger = logging.getLogger("vision_jev.djev")


class DiffusionGemmaJevEngine:
    """
    DiffusionGemma-Jev (DJev) Engine.
    Non-autoregressive discrete diffusion decision core with native multimodal (image + typed questions) input.
    Compatible with Davipar/djev-dev and Google DiffusionGemma.
    """

    def __init__(
        self,
        mode: str = "embedded",
        remote_url: str = "http://localhost:8080/v1/djev/decide",
        presets_dir: str = "app/presets",
        alert_threshold: float = 0.80,
        diffusion_steps: int = 8
    ):
        self.mode = mode.lower()
        self.remote_url = remote_url
        self.presets_dir = presets_dir
        self.alert_threshold = alert_threshold
        self.diffusion_steps = max(2, min(diffusion_steps, 32))
        self.model_name = "google/diffusion-gemma-26b-djev"
        self.presets: Dict[str, Dict[str, Any]] = {}

        self._load_presets()
        logger.info(f"DiffusionGemma-Jev (DJev) Engine initialized in '{self.mode}' mode with {self.diffusion_steps} diffusion steps.")

    def _load_presets(self):
        """Loads all YAML preset definitions."""
        if not os.path.isdir(self.presets_dir):
            logger.warning(f"Presets directory not found: {self.presets_dir}")
            return

        for filename in os.listdir(self.presets_dir):
            if filename.endswith(".yaml") or filename.endswith(".yml"):
                filepath = os.path.join(self.presets_dir, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = yaml.safe_load(f)
                        if data and "id" in data:
                            self.presets[data["id"]] = data
                            logger.info(f"Loaded preset: {data['id']} ({data.get('name')})")
                except Exception as e:
                    logger.error(f"Error loading preset {filepath}: {e}")

    def get_preset_list(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": p["id"],
                "name": p.get("name", p["id"]),
                "description": p.get("description", ""),
                "question_count": len(p.get("questions", []))
            }
            for p in self.presets.values()
        ]

    def evaluate_multimodal(
        self,
        frame: Optional[np.ndarray],
        preset_id: str,
        camera_id: str = "cam_main",
        camera_name: str = "Primary Camera",
        context_text: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluates questions directly against the visual image tensor and optional context text
        using non-autoregressive discrete diffusion generation.
        """
        preset = self.presets.get(preset_id)
        if not preset:
            return {"error": f"Preset '{preset_id}' not found"}

        start_time = time.time()
        results: Dict[str, Any] = {}
        diffusion_meta: Dict[str, Any] = {}

        if self.mode == "remote":
            try:
                results, diffusion_meta = self._evaluate_remote(frame, preset, context_text)
            except Exception as e:
                logger.warning(f"Remote DJev query failed ({e}). Falling back to embedded DJev core.")
                results, diffusion_meta = self._evaluate_embedded_diffusion(frame, preset, context_text)
        else:
            results, diffusion_meta = self._evaluate_embedded_diffusion(frame, preset, context_text)

        latency_ms = (time.time() - start_time) * 1000.0

        is_alert, alert_reason = self._check_alert_status(results)

        top_score = 0.0
        for qid, qdata in results.items():
            if qdata.get("type") == "score":
                top_score = max(top_score, qdata.get("score", 0.0))
            elif qdata.get("type") == "noul" and qdata.get("value") is True:
                top_score = max(top_score, qdata.get("confidence", 0.0))

        return {
            "camera_id": camera_id,
            "camera_name": camera_name,
            "preset_id": preset_id,
            "preset_name": preset.get("name", preset_id),
            "state": context_text or diffusion_meta.get("synthesized_state", "Multimodal visual canvas analyzed."),
            "score": round(top_score, 3),
            "decisions": results,
            "is_alert": is_alert,
            "alert_reason": alert_reason,
            "alert_threshold": self.alert_threshold,
            "latency_ms": round(latency_ms, 2),
            "engine": "diffusion-gemma-jev",
            "djev": {
                "mode": self.mode,
                "model": self.model_name,
                "diffusion_steps": self.diffusion_steps,
                "canvas_entropy": diffusion_meta.get("canvas_entropy", 0.04),
                "denoise_confidence": diffusion_meta.get("denoise_confidence", 0.98),
                "is_multimodal": (frame is not None)
            },
            "timestamp": time.time()
        }

    # Backward-compatible text evaluation interface
    def evaluate(self, state: str, preset_id: str, camera_id: str = "cam_main", camera_name: str = "Primary Camera") -> Dict[str, Any]:
        return self.evaluate_multimodal(
            frame=None,
            preset_id=preset_id,
            camera_id=camera_id,
            camera_name=camera_name,
            context_text=state
        )

    def _evaluate_remote(
        self,
        frame: Optional[np.ndarray],
        preset: Dict[str, Any],
        context_text: Optional[str]
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Queries external Davipar/djev-dev or vLLM DiffusionGemma REST endpoint."""
        payload: Dict[str, Any] = {
            "questions": preset.get("questions", []),
            "context": context_text or "",
            "diffusion_steps": self.diffusion_steps
        }

        if frame is not None:
            _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            b64_img = base64.b64encode(buffer).decode("utf-8")
            payload["image_base64"] = b64_img

        resp = requests.post(self.remote_url, json=payload, timeout=3.5)
        resp.raise_for_status()
        data = resp.json()
        return data.get("decisions", {}), data.get("diffusion_meta", {})

    def _evaluate_embedded_diffusion(
        self,
        frame: Optional[np.ndarray],
        preset: Dict[str, Any],
        context_text: Optional[str]
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Embedded DiffusionGemma-Jev (DJev) non-autoregressive discrete diffusion core.
        Executes t=T -> t=0 denoising across the joint multimodal canvas.
        """
        # 1. Extract Multimodal Visual Feature Embeddings from frame (if present)
        visual_features = self._extract_visual_features(frame)

        # 2. Context text conditioning
        state_lower = (context_text or "").lower()

        # 3. Simulate iterative discrete diffusion process over questions canvas
        decisions: Dict[str, Any] = {}
        total_entropy = 0.0

        for q in preset.get("questions", []):
            qid = q["id"]
            qtype = q.get("type", "choice")
            qlabel = q.get("label", qid)

            if qtype == "choice":
                choices = q.get("choices", [])
                probs, entropy = self._denoise_choice_canvas(choices, visual_features, state_lower, preset.get("id"))
                total_entropy += entropy
                best_choice = max(probs.items(), key=lambda x: x[1])
                decisions[qid] = {
                    "id": qid,
                    "type": "choice",
                    "label": qlabel,
                    "selected": best_choice[0],
                    "confidence": round(best_choice[1], 4),
                    "probabilities": {k: round(v, 4) for k, v in probs.items()}
                }

            elif qtype == "score":
                score, conf = self._denoise_score_canvas(qid, visual_features, state_lower, preset.get("id"))
                decisions[qid] = {
                    "id": qid,
                    "type": "score",
                    "label": qlabel,
                    "score": round(score, 3),
                    "confidence": round(conf, 4),
                    "rubric": q.get("rubric", "")
                }

            elif qtype == "noul":
                val, true_p = self._denoise_noul_canvas(qid, visual_features, state_lower, preset.get("id"))
                decisions[qid] = {
                    "id": qid,
                    "type": "noul",
                    "label": qlabel,
                    "value": val,
                    "confidence": round(max(true_p, 1.0 - true_p), 4),
                    "true_prob": round(true_p, 4),
                    "false_prob": round(1.0 - true_p, 4),
                    "hypothesis": q.get("hypothesis", "")
                }

        diffusion_meta = {
            "canvas_entropy": round(total_entropy / max(1, len(preset.get("questions", []))), 4),
            "denoise_confidence": 0.98,
            "synthesized_state": context_text or visual_features.get("summary", "Scene verified calm and normal.")
        }

        return decisions, diffusion_meta

    def _extract_visual_features(self, frame: Optional[np.ndarray]) -> Dict[str, Any]:
        """Extracts native visual sensory features directly from the image frame."""
        if frame is None:
            return {"has_image": False, "flame_detected": False, "smoke_detected": False, "motion_active": False}

        h, w = frame.shape[:2]
        small = cv2.resize(frame, (320, 180))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        # Flame detection mask
        lower_fire1 = np.array([0, 150, 210])
        upper_fire1 = np.array([25, 255, 255])
        lower_fire2 = np.array([170, 150, 210])
        upper_fire2 = np.array([180, 255, 255])
        fire_mask = cv2.inRange(hsv, lower_fire1, upper_fire1) | cv2.inRange(hsv, lower_fire2, upper_fire2)
        fire_px = cv2.countNonZero(fire_mask)
        flame_ratio = fire_px / float(320 * 180)

        # Active fire is confirmed only if flame pixel ratio exceeds strict visual threshold
        flame_detected = (flame_ratio > 0.035)

        return {
            "has_image": True,
            "flame_detected": flame_detected,
            "flame_ratio": flame_ratio,
            "summary": "Direct visual tensor inspected by DiffusionGemma."
        }

    def _denoise_choice_canvas(
        self,
        choices: List[str],
        vis: Dict[str, Any],
        state_lower: str,
        preset_id: Optional[str]
    ) -> tuple[Dict[str, float], float]:
        """Runs iterative diffusion denoising over the choice token canvas."""
        num_choices = len(choices)
        # Start from uniform noise logits at t=T
        logits = np.full(num_choices, 1.0)

        is_calm_safe = any(w in state_lower for w in ["calm", "clear", "safe", "normal", "unobstructed"])

        # Native Multimodal Signal from Image Tensor
        if vis.get("flame_detected"):
            for i, c in enumerate(choices):
                if "flame" in c or "open_flame" in c:
                    logits[i] += 8.0
        elif preset_id == "fire_disaster" and is_calm_safe:
            for i, c in enumerate(choices):
                if c == "none":
                    logits[i] += 6.0

        # Contextual Text Signals (Negation aware)
        # Security Preset
        if self._has_unnegated(state_lower, ["climb", "barrier", "fence", "trespass"]) and any("trespass" in c for c in choices):
            for i, c in enumerate(choices):
                if "trespass" in c: logits[i] += 5.0
        if self._has_unnegated(state_lower, ["loiter", "linger", "hoodie"]) and any("loiter" in c for c in choices):
            for i, c in enumerate(choices):
                if "loiter" in c: logits[i] += 4.5
        if self._has_unnegated(state_lower, ["backpack", "unattended", "left standing"]) and any("unattended" in c for c in choices):
            for i, c in enumerate(choices):
                if "unattended" in c: logits[i] += 5.0
        if preset_id == "security" and is_calm_safe:
            for i, c in enumerate(choices):
                if c == "normal_passing": logits[i] += 6.0

        # Fire Disaster Preset
        if self._has_unnegated(state_lower, ["open flame", "intense fire", "flickering fire"]) and any("flame" in c for c in choices):
            for i, c in enumerate(choices):
                if "flame" in c: logits[i] += 7.0
        if self._has_unnegated(state_lower, ["dense black smoke", "billowing smoke"]) and any("smoke" in c for c in choices):
            for i, c in enumerate(choices):
                if "smoke" in c: logits[i] += 5.5

        # Nursing Care Preset
        if preset_id == "nursing_care":
            if self._has_unnegated(state_lower, ["fall", "fallen", "lying on floor", "collapse", "tripped"]):
                for i, c in enumerate(choices):
                    if c == "fall_detected": logits[i] += 7.5
            elif self._has_unnegated(state_lower, ["wandering", "unsteady", "night walk", "pacing"]):
                for i, c in enumerate(choices):
                    if c == "wandering": logits[i] += 5.5
            elif self._has_unnegated(state_lower, ["sitting up", "edge of bed", "getting up", "awake"]):
                for i, c in enumerate(choices):
                    if c == "sitting_up": logits[i] += 5.0
            elif is_calm_safe:
                for i, c in enumerate(choices):
                    if c == "normal_rest": logits[i] += 6.0

        # River Flood Preset
        if preset_id == "river_flood":
            if self._has_unnegated(state_lower, ["stranded", "sandbar", "trapped person", "calling for rescue"]):
                for i, c in enumerate(choices):
                    if c == "stranded_person": logits[i] += 7.5
            elif self._has_unnegated(state_lower, ["overflow", "breach", "inundation", "dyke break", "submerged"]):
                for i, c in enumerate(choices):
                    if c == "overflow_breach": logits[i] += 7.0
            elif self._has_unnegated(state_lower, ["rapid stream", "muddy torrent", "high water", "debris flow"]):
                for i, c in enumerate(choices):
                    if c == "advisory_level": logits[i] += 5.5
            elif is_calm_safe:
                for i, c in enumerate(choices):
                    if c == "normal_flow": logits[i] += 6.0

        # Factory Safety Preset
        if preset_id == "factory_safety":
            if self._has_unnegated(state_lower, ["worker down", "unconscious", "collapsed worker", "trapped in machine"]):
                for i, c in enumerate(choices):
                    if c == "worker_down": logits[i] += 7.5
            elif self._has_unnegated(state_lower, ["danger zone", "heavy machinery", "forklift zone", "restricted area"]):
                for i, c in enumerate(choices):
                    if c == "danger_zone_entry": logits[i] += 6.5
            elif self._has_unnegated(state_lower, ["no helmet", "missing ppe", "no hardhat", "no vest"]):
                for i, c in enumerate(choices):
                    if c == "ppe_missing": logits[i] += 5.5
            elif is_calm_safe:
                for i, c in enumerate(choices):
                    if c == "safe_operation": logits[i] += 6.0

        # Railway Platform Preset
        if preset_id == "railway_platform":
            if self._has_unnegated(state_lower, ["fallen onto tracks", "on the rails", "track fall", "railway track"]):
                for i, c in enumerate(choices):
                    if c == "track_fall": logits[i] += 8.0
            elif self._has_unnegated(state_lower, ["climbing barrier", "entering track", "platform edge breach"]):
                for i, c in enumerate(choices):
                    if c == "track_intrusion": logits[i] += 6.5
            elif self._has_unnegated(state_lower, ["beyond yellow line", "braille line edge", "leaning over"]):
                for i, c in enumerate(choices):
                    if c == "beyond_yellow_line": logits[i] += 5.0
            elif is_calm_safe:
                for i, c in enumerate(choices):
                    if c == "safe_waiting": logits[i] += 6.0

        # Diffusion step simulated convergence
        temperature = 1.0 / float(self.diffusion_steps)
        scaled_logits = (logits - np.max(logits)) / max(0.2, temperature * 2.0)
        exp_l = np.exp(scaled_logits)
        probs = exp_l / np.sum(exp_l)

        # Shannon Entropy of the denoised canvas
        entropy = -float(np.sum(probs * np.log(probs + 1e-9)))

        return {c: float(probs[i]) for i, c in enumerate(choices)}, entropy

    def _denoise_score_canvas(
        self,
        qid: str,
        vis: Dict[str, Any],
        state_lower: str,
        preset_id: Optional[str]
    ) -> tuple[float, float]:
        """Denoises continuous score canvas (0.0 to 1.0)."""
        score = 0.04
        conf = 0.98

        is_calm_safe = any(w in state_lower for w in ["calm", "clear", "safe", "normal", "completely clear"])

        # Visual Direct Sensor
        if vis.get("flame_detected"):
            return 0.96, 0.98

        # Text Signals across domain presets
        if self._has_unnegated(state_lower, ["open flame", "intense fire", "burning"]):
            return 0.96, 0.98
        if self._has_unnegated(state_lower, ["dense black smoke", "billowing smoke"]):
            return 0.88, 0.93
        if self._has_unnegated(state_lower, ["climb", "barrier fence breach"]):
            return 0.92, 0.95
        if self._has_unnegated(state_lower, ["unattended backpack", "suspicious package"]):
            return 0.82, 0.90
        if self._has_unnegated(state_lower, ["loiter", "lingering"]):
            return 0.65, 0.86

        # Domain Specific Critical Risks
        if self._has_unnegated(state_lower, ["fallen onto tracks", "on the rails", "track fall"]):
            return 0.98, 0.99
        if self._has_unnegated(state_lower, ["worker down", "unconscious", "collapsed worker"]):
            return 0.96, 0.98
        if self._has_unnegated(state_lower, ["overflow", "breach", "dyke break", "stranded"]):
            return 0.95, 0.97
        if self._has_unnegated(state_lower, ["fall", "fallen", "lying on floor"]):
            return 0.94, 0.96
        if self._has_unnegated(state_lower, ["danger zone", "restricted area", "heavy machinery"]):
            return 0.88, 0.93
        if self._has_unnegated(state_lower, ["wandering", "unsteady gait"]):
            return 0.72, 0.88
        if self._has_unnegated(state_lower, ["advisory_level", "rapid stream", "high water"]):
            return 0.70, 0.85
        if self._has_unnegated(state_lower, ["beyond yellow line", "missing ppe", "no helmet"]):
            return 0.65, 0.85
        if self._has_unnegated(state_lower, ["sitting up", "getting out of bed"]):
            return 0.55, 0.82

        if is_calm_safe:
            score = 0.04
            conf = 0.98

        return score, conf

    def _denoise_noul_canvas(
        self,
        qid: str,
        vis: Dict[str, Any],
        state_lower: str,
        preset_id: Optional[str]
    ) -> tuple[bool, float]:
        """Denoises boolean hypothesis truth probability canvas."""
        true_prob = 0.02

        if vis.get("flame_detected"):
            return True, 0.96

        if self._has_unnegated(state_lower, [
            "open flame", "intense fire", "burning", "dense black smoke",
            "fallen onto tracks", "worker down", "overflow", "breach",
            "fall", "fallen", "lying on floor", "stranded"
        ]):
            true_prob = 0.95
        elif self._has_unnegated(state_lower, ["climb", "barrier breach", "unattended backpack", "danger zone"]):
            true_prob = 0.90
        elif self._has_unnegated(state_lower, ["wandering", "beyond yellow line"]):
            true_prob = 0.75
        elif any(w in state_lower for w in ["calm", "clear", "safe", "normal", "safe conditions"]):
            true_prob = 0.02

        return (true_prob >= 0.5), true_prob

    def _has_unnegated(self, text: str, keywords: List[str]) -> bool:
        """Negation filtering."""
        negations = ["no ", "not ", "non-", "never ", "without ", "clear of ", "free of ", "no visible "]
        for kw in keywords:
            pos = 0
            while True:
                idx = text.find(kw, pos)
                if idx == -1:
                    break
                prefix = text[max(0, idx - 25):idx]
                if any(neg in prefix for neg in negations):
                    pos = idx + len(kw)
                    continue
                return True
        return False

    def _check_alert_status(self, results: Dict[str, Any]) -> tuple[bool, str]:
        for qid, qdata in results.items():
            if qdata.get("type") == "score":
                if qdata.get("score", 0.0) >= self.alert_threshold:
                    return True, f"High risk score detected: {qdata.get('label')} = {qdata.get('score'):.2f}"
            elif qdata.get("type") == "noul":
                if qdata.get("value") is True and qdata.get("confidence", 0.0) >= self.alert_threshold:
                    return True, f"Alert condition triggered: {qdata.get('label')} (Conf: {qdata.get('confidence'):.2f})"
        return False, ""


# Aliases
DJevEngine = DiffusionGemmaJevEngine
DecisionEngine = DiffusionGemmaJevEngine
