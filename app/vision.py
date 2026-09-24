import time
import cv2
import numpy as np
import logging
import threading
from typing import Optional, Dict, Any, List
from PIL import Image

logger = logging.getLogger("vision_jev.vision")


class _CameraTrackState:
    """Per-camera detector state (background model + loitering tracker)."""

    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=25, detectShadows=True
        )
        self.lingering_frames = 0
        self.last_center = None
        self.lock = threading.Lock()


class VisionExtractor:
    """
    Extracts visual state descriptions from camera frames.
    Supports:
      - 'detector': Real-time CPU surveillance CV engine (MOG2 Motion Segmentation, Aspect-ratio Person/Vehicle tracking, Flame/Smoke hue analysis)
      - 'mock': Pre-scripted synthetic demo scenarios for offline testing
      - 'vlm': Deep VLM captioning (Moondream2 / SmolVLM)
    """

    MOCK_SCENARIOS = {
        "security": [
            "A person in standard work attire is walking through the entrance corridor normally.",
            "An individual wearing a black hoodie is lingering near the perimeter fence without moving.",
            "A person is attempting to climb over the outer security barrier fence.",
            "An unattended black backpack has been left standing near the emergency exit door.",
            "A delivery courier is momentarily passing through the corridor.",
        ],
        "fire_disaster": [
            "The area is completely clear with normal workplace lighting and no visible smoke.",
            "White steam vapor is gently rising from an electric kettle in the break area.",
            "Dense black smoke is rising rapidly from an electrical equipment cabinet.",
            "An active open flame is spreading across the trash bin with visible flickering fire.",
            "Atmosphere is clear, temperature and visual conditions appear safe and normal.",
        ],
        "default": [
            "Surveillance area appears calm and clear of any unusual activity.",
            "A subject is moving across the camera field of view.",
        ]
    }

    def __init__(self, mode: str = "detector", model_name: str = "vikhyatk/moondream2"):
        self.mode = mode.lower()
        self.model_name = model_name
        self.step_counter = 0

        # Background model and loitering tracker are kept per camera so that
        # round-robin frames from different scenes never share state.
        self._track_states: Dict[str, _CameraTrackState] = {}
        self._states_lock = threading.Lock()

        # Optional VLM
        self.vlm_model = None
        self.vlm_tokenizer = None
        if self.mode == "vlm":
            self._init_vlm()

        logger.info(f"VisionExtractor initialized in '{self.mode}' mode.")

    def _init_vlm(self):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.vlm_tokenizer = AutoTokenizer.from_pretrained(self.model_name, revision="2024-08-26")
            self.vlm_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                revision="2024-08-26",
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            ).to(device)
            self.vlm_model.eval()
            logger.info("VLM loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load VLM ({e}). Falling back to 'detector' mode.")
            self.mode = "detector"

    def _get_track_state(self, camera_id: str) -> _CameraTrackState:
        with self._states_lock:
            state = self._track_states.get(camera_id)
            if state is None:
                state = _CameraTrackState()
                self._track_states[camera_id] = state
            return state

    def reset_camera(self, camera_id: str):
        """Drops the detector state of a removed/reconfigured camera."""
        with self._states_lock:
            self._track_states.pop(camera_id, None)

    def extract_state(self, frame: np.ndarray, preset_id: str = "security", camera_id: str = "default") -> Dict[str, Any]:
        """
        Extracts visual state sentence from the provided frame and renders real detection boxes.
        Returns: { state: str, latency_ms: float, mode: str, detections: dict }
        """
        start_time = time.time()
        detections = {}

        if self.mode == "vlm" and self.vlm_model is not None:
            state_text = self._extract_with_vlm(frame)
        elif self.mode == "detector":
            track = self._get_track_state(camera_id)
            with track.lock:
                state_text, detections = self._extract_with_detector(frame, preset_id, track)
        else:
            state_text = self._extract_with_mock(preset_id)

        latency_ms = (time.time() - start_time) * 1000.0

        return {
            "state": state_text,
            "latency_ms": round(latency_ms, 2),
            "mode": self.mode,
            "detections": detections,
            "timestamp": time.time()
        }

    def _extract_with_detector(self, frame: np.ndarray, preset_id: str, track: _CameraTrackState) -> tuple[str, dict]:
        """
        Performs genuine visual analysis on the frame using CPU-friendly CV algorithms:
          1. Foreground motion & object segmentation (MOG2)
          2. Morphological filtering & contour geometry classification (Person vs Object vs Vehicle)
          3. Flame & Smoke spectral analysis (HSV color space)
        The input frame is never modified (it is shared with the decision engine and
        Visual RAG); bounding boxes are returned in `detections["boxes"]` instead.
        """
        h, w = frame.shape[:2]
        small_w, small_h = 640, 360
        small = cv2.resize(frame, (small_w, small_h))

        # 1. Background subtraction mask
        fg_mask = track.bg_subtractor.apply(small)
        # Filter out shadows (gray value 127 in MOG2)
        _, fg_thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        fg_clean = cv2.morphologyEx(fg_thresh, cv2.MORPH_OPEN, kernel)
        fg_clean = cv2.morphologyEx(fg_clean, cv2.MORPH_DILATE, kernel, iterations=2)

        # 2. Extract bounding boxes from motion contours
        contours, _ = cv2.findContours(fg_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_persons = 0
        detected_objects = 0
        scale_x = w / float(small_w)
        scale_y = h / float(small_h)

        max_contour_area = 0
        active_center = None
        boxes: List[Dict[str, Any]] = []

        for c in contours:
            area = cv2.contourArea(c)
            if area < 600:
                continue

            max_contour_area = max(max_contour_area, area)
            x, y, bw, bh = cv2.boundingRect(c)
            aspect_ratio = float(bh) / max(1.0, float(bw))

            # Scale to original frame
            ox = int(x * scale_x)
            oy = int(y * scale_y)
            ow = int(bw * scale_x)
            oh = int(bh * scale_y)

            # Aspect ratio > 1.25 typically denotes an upright person/pedestrian
            if aspect_ratio >= 1.25 and area > 900:
                detected_persons += 1
                label = "person"
            else:
                detected_objects += 1
                label = "object"

            active_center = (ox + ow // 2, oy + oh // 2)
            boxes.append({"label": label, "x": ox, "y": oy, "w": ow, "h": oh, "area": int(area)})

        # 3. Lingering / Loitering Detection
        is_lingering = False
        if active_center is not None and track.last_center is not None:
            dist = np.hypot(active_center[0] - track.last_center[0], active_center[1] - track.last_center[1])
            if dist < 40 and max_contour_area > 800:
                track.lingering_frames += 1
            else:
                track.lingering_frames = max(0, track.lingering_frames - 1)
        else:
            track.lingering_frames = max(0, track.lingering_frames - 1)

        track.last_center = active_center
        if track.lingering_frames >= 4:
            is_lingering = True

        # 4. Fire / Smoke Hue Analysis (Restricted strictly to active moving regions to avoid false positives on static dark machinery/floors)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        
        # Intense flame hue mask (vivid orange, red, yellow)
        lower_fire1 = np.array([0, 140, 200])
        upper_fire1 = np.array([25, 255, 255])
        lower_fire2 = np.array([170, 140, 200])
        upper_fire2 = np.array([180, 255, 255])
        fire_mask = cv2.inRange(hsv, lower_fire1, upper_fire1) | cv2.inRange(hsv, lower_fire2, upper_fire2)

        # Smoke hue mask (low saturation, gray-diffuse tones)
        lower_smoke = np.array([0, 0, 40])
        upper_smoke = np.array([180, 40, 110])
        smoke_mask = cv2.inRange(hsv, lower_smoke, upper_smoke)

        # CRITICAL FIX: Only evaluate smoke and fire on MOVING pixels (fg_clean)
        # Static machinery, dark floorboards, metal equipment and shadows are completely ignored.
        motion_pixel_count = cv2.countNonZero(fg_clean)
        total_px = small_w * small_h

        if motion_pixel_count > 2500:  # Genuine widespread expanding disturbance
            moving_fire = cv2.bitwise_and(fire_mask, fire_mask, mask=fg_clean)
            moving_smoke = cv2.bitwise_and(smoke_mask, smoke_mask, mask=fg_clean)
            fire_pixels = cv2.countNonZero(moving_fire)
            smoke_pixels = cv2.countNonZero(moving_smoke)
            fire_ratio = fire_pixels / float(total_px)
            smoke_ratio = smoke_pixels / float(total_px)
        else:
            fire_ratio = 0.0
            smoke_ratio = 0.0

        detections = {
            "persons_detected": detected_persons,
            "objects_detected": detected_objects,
            "max_area": int(max_contour_area),
            "motion_pixels": int(motion_pixel_count),
            "is_lingering": is_lingering,
            "fire_ratio": round(fire_ratio, 4),
            "smoke_ratio": round(smoke_ratio, 4),
            "boxes": boxes
        }

        # 5. Synthesize Genuine State Description
        if preset_id == "fire_disaster":
            if fire_ratio > 0.04:
                state_text = "An active open flame with intense flickering fire is detected in the monitored scene."
            elif smoke_ratio > 0.08 and motion_pixel_count > 8000:
                state_text = "Dense black smoke is billowing and rising rapidly across the camera field of view."
            elif fire_ratio > 0.015:
                state_text = "Minor heat, steam vapor or thermal flicker detected near the area."
            elif detected_persons > 0 or detected_objects > 0:
                state_text = "Workers or moving machinery detected in area; ambient conditions are safe and normal."
            else:
                state_text = "The area is completely clear and calm with normal lighting and safe conditions."

        else:  # preset_id == "security"
            if is_lingering:
                state_text = "An individual is lingering near the area without moving for an extended duration."
            elif detected_persons == 1:
                state_text = "A person in standard work attire is walking through the entrance corridor normally."
            elif detected_persons > 1:
                state_text = f"Multiple individuals ({detected_persons} persons) are moving through the monitored area."
            elif detected_objects > 0:
                if max_contour_area > 8000:
                    state_text = "Rapid motion or large vehicle movement detected in the surveillance area."
                else:
                    state_text = "A subject or unattended object is detected moving across the camera field of view."
            else:
                state_text = "Surveillance area appears calm and clear of any unusual activity."

        return state_text, detections

    def _extract_with_mock(self, preset_id: str) -> str:
        self.step_counter += 1
        scenarios = self.MOCK_SCENARIOS.get(preset_id, self.MOCK_SCENARIOS["default"])
        idx = (self.step_counter // 4) % len(scenarios)
        return scenarios[idx]

    def _extract_with_vlm(self, frame: np.ndarray) -> str:
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            prompt = "Describe what is happening in this security camera scene in 1 or 2 concise sentences."
            enc_image = self.vlm_model.encode_image(pil_image)
            description = self.vlm_model.answer_question(enc_image, prompt, self.vlm_tokenizer)
            return description.strip()
        except Exception as e:
            logger.error(f"VLM inference error: {e}")
            return "Surveillance area under continuous automated observation."


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    extractor = VisionExtractor(mode="detector")
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    res = extractor.extract_state(test_frame, "security")
    print("Detector result:", res)
