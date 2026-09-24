import numpy as np
import pytest

from app.rag import rag_engine
from app.visual_rag import visual_rag_engine


@pytest.mark.parametrize("query,category", [
    ("A person in standard work attire is walking through the entrance corridor normally.", "security"),
    ("A delivery courier is momentarily passing through the corridor.", "security"),
    ("Surveillance area appears calm and clear of any unusual activity.", "security"),
    ("The area is completely clear with normal workplace lighting and no visible smoke.", "fire_disaster"),
    ("Workers or moving machinery detected in area; ambient conditions are safe and normal.", "fire_disaster"),
])
def test_normal_scenes_do_not_match_emergency_sop(query, category):
    assert rag_engine.search_sop(query, category)["matched"] is False


@pytest.mark.parametrize("query,category,expected_sop", [
    ("A person is attempting to climb over the outer security barrier fence.", "security", "sop_trespass_breach"),
    ("An individual wearing a black hoodie is lingering near the perimeter fence.", "security", "sop_loitering_suspicious"),
    ("An unattended black backpack has been left near the emergency exit door.", "security", "sop_unattended_object"),
    ("Dense black smoke is rising rapidly from an electrical equipment cabinet.", "fire_disaster", "sop_dark_smoke"),
    ("An active open flame is spreading across the trash bin.", "fire_disaster", "sop_fire_open_flame"),
    ("An elderly resident has fallen and is lying on floor.", "nursing_care", "sop_nursing_fall"),
    ("A passenger has fallen onto tracks.", "railway_platform", "sop_railway_track_fall"),
    ("柵越え侵入を検知", "security", "sop_trespass_breach"),
])
def test_incidents_match_expected_sop(query, category, expected_sop):
    res = rag_engine.search_sop(query, category)
    assert res["matched"] is True
    assert res["sop"]["id"] == expected_sop


def test_visual_seed_references_link_to_existing_sops():
    missing = [
        r.sop_id for r in visual_rag_engine.references.values()
        if r.sop_id and r.sop_id not in rag_engine.documents
    ]
    assert missing == []


def test_visual_embedding_is_512d_unit_vector():
    frame = np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8)
    vec = visual_rag_engine.extract_embedding(frame)
    assert vec.shape == (512,)
    assert float(np.linalg.norm(vec)) == pytest.approx(1.0, abs=1e-4)


def test_match_frame_returns_bool_anomaly_flag():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    res = visual_rag_engine.match_frame(frame, "security")
    assert isinstance(res["is_anomalous"], bool)


PRESETS = ["security", "fire_disaster", "nursing_care", "river_flood", "factory_safety", "railway_platform"]


@pytest.mark.parametrize("category", PRESETS)
def test_unrelated_synthetic_frame_is_not_visual_anomaly(category):
    """Regression for #13: the default synthetic test pattern must not alert."""
    from app.camera import CameraDevice
    frame = CameraDevice("t", "t", "synthetic", "synthetic")._generate_synthetic_frame()
    assert visual_rag_engine.match_frame(frame, category)["is_anomalous"] is False


def _decode_thumbnail(ref):
    import base64
    import cv2
    raw = base64.b64decode(ref.image_base64.split(",", 1)[1])
    return cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)


@pytest.mark.parametrize("ref_id", list(visual_rag_engine.references))
def test_seed_reference_images_classify_as_their_own_label(ref_id):
    ref = visual_rag_engine.references[ref_id]
    res = visual_rag_engine.match_frame(_decode_thumbnail(ref), ref.category)
    assert res["is_anomalous"] is ref.is_anomaly


def test_similarity_is_raw_cosine_in_unit_range():
    frame = np.random.default_rng(3).integers(0, 255, (180, 320, 3), dtype=np.uint8)
    for m in visual_rag_engine.match_frame(frame, None, top_k=50)["matches"]:
        assert 0.0 <= m["similarity"] <= 1.0
