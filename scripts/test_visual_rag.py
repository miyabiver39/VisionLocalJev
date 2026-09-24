"""
scripts/test_visual_rag.py
--------------------------
Visual Example RAG Engine standalone test script.
"""
import sys
import os
import cv2
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.visual_rag import visual_rag_engine

def main():
    print("=" * 60)
    print("Visual Example RAG (Visual Few-Shot Retrieval) Test")
    print("=" * 60)

    # 1. Check registered seed references
    refs = visual_rag_engine.get_all_references()
    print(f"\n[1] Registered Reference Images in Visual Vector DB: {len(refs)} items")
    for r in refs:
        kind = "[ANOMALY]" if r["is_anomaly"] else "[NORMAL ]"
        print(f"  - {kind} {r['category']:<16} | ID: {r['ref_id']} | {r['title']}")

    # 2. Test Normal Query Image (Factory Clear floor)
    print("\n[2] Testing Normal Query Frame (Green floor, normal lighting)...")
    normal_img = np.zeros((360, 512, 3), dtype=np.uint8)
    normal_img[:] = (45, 50, 50)
    cv2.rectangle(normal_img, (150, 280), (350, 360), (40, 150, 40), -1)

    result_norm = visual_rag_engine.match_frame(normal_img, category="factory_safety")
    print(f"  - Search Latency: {result_norm['latency_ms']} ms")
    print(f"  - Top Match Title: {result_norm['top_match']['title']}")
    print(f"  - Similarity: {result_norm['similarity'] * 100:.1f}%")
    print(f"  - Anomaly Score: {result_norm['anomaly_score']}")
    print(f"  - Is Anomalous Alert?: {result_norm['is_anomalous']}")

    # 3. Test Anomaly Query Image (Fire flame erupting)
    print("\n[3] Testing Anomaly Query Frame (Intense orange-red open flame)...")
    fire_img = np.zeros((360, 512, 3), dtype=np.uint8)
    fire_img[:] = (30, 30, 35)
    cv2.ellipse(fire_img, (300, 250), (70, 140), 0, 0, 360, (0, 120, 255), -1)
    cv2.ellipse(fire_img, (300, 260), (40, 90), 0, 0, 360, (0, 220, 255), -1)

    result_fire = visual_rag_engine.match_frame(fire_img, category="fire_disaster")
    print(f"  - Search Latency: {result_fire['latency_ms']} ms")
    print(f"  - Top Match Title: {result_fire['top_match']['title']}")
    print(f"  - Similarity: {result_fire['similarity'] * 100:.1f}%")
    print(f"  - Linked Incident SOP: {result_fire['top_match']['sop_id']}")
    print(f"  - Anomaly Score: {result_fire['anomaly_score']}")
    print(f"  - Is Anomalous Alert?: {result_fire['is_anomalous']}")

    assert result_norm['is_anomalous'] is False, "Normal query should not trigger anomaly!"
    assert result_fire['is_anomalous'] is True, "Fire query must trigger anomaly!"
    print("\n[SUCCESS] Visual Example RAG verified successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
