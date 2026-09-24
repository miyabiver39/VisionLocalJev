import numpy as np

from app.vision import VisionExtractor


def _frame_with_box(x):
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    frame[100:260, x:x + 60] = 255
    return frame


def test_detector_does_not_modify_input_frame():
    ext = VisionExtractor(mode="detector")
    for _ in range(3):
        ext.extract_state(np.zeros((360, 640, 3), dtype=np.uint8), "security", "cam_a")
    frame = _frame_with_box(200)
    before = frame.copy()
    res = ext.extract_state(frame, "security", "cam_a")
    assert np.array_equal(frame, before)
    assert "boxes" in res["detections"]


def test_detector_state_is_isolated_per_camera():
    ext = VisionExtractor(mode="detector")
    blank = np.zeros((360, 640, 3), dtype=np.uint8)
    bright = np.full((360, 640, 3), 255, dtype=np.uint8)
    # Warm up both cameras on their own (static) scenes
    for _ in range(10):
        ext.extract_state(blank, "security", "cam_a")
        ext.extract_state(bright, "security", "cam_b")
    # Round-robin frames from a different static scene must not be seen as motion
    res_a = ext.extract_state(blank, "security", "cam_a")
    res_b = ext.extract_state(bright, "security", "cam_b")
    assert res_a["detections"]["motion_pixels"] == 0
    assert res_b["detections"]["motion_pixels"] == 0
