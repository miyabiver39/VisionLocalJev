"""
benchmark_pipeline.py
---------------------
Measures per-stage CPU latency of the inference pipeline (no server, no network):
Vision extraction -> DJev decision (embedded emulator) -> SOP RAG -> Visual Example RAG.

Usage:
    python scripts/benchmark_pipeline.py [--cycles 200] [--warmup 20] [--width 1280 --height 720]

Note: DJEV_MODE=embedded is a CPU emulator of the DJev typed-decision interface
(OpenCV features + rule-based logits), not the DiffusionGemma model itself.
"""

import argparse
import os
import platform
import statistics
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.decision import DecisionEngine  # noqa: E402
from app.rag import rag_engine  # noqa: E402
from app.vision import VisionExtractor  # noqa: E402
from app.visual_rag import visual_rag_engine  # noqa: E402


def make_frames(n: int, width: int, height: int):
    """Moving-object test sequence so that MOG2 / contour code paths are exercised."""
    rng = np.random.default_rng(0)
    base = rng.integers(20, 60, (height, width, 3), dtype=np.uint8)
    frames = []
    for i in range(n):
        f = base.copy()
        cx = int(width * (0.2 + 0.6 * ((i % 60) / 60)))
        cv2.rectangle(f, (cx - 40, height // 3), (cx + 40, height // 3 + 200), (200, 200, 200), -1)
        frames.append(f)
    return frames


def summarize(name, samples):
    s = sorted(samples)
    p95 = s[int(len(s) * 0.95) - 1]
    print(f"| {name:<28} | {statistics.mean(s):8.2f} ms | {statistics.median(s):8.2f} ms | {p95:8.2f} ms |")
    return statistics.mean(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--preset", default="security")
    args = ap.parse_args()

    vision = VisionExtractor(mode="detector")
    engine = DecisionEngine(mode="embedded", presets_dir="app/presets")
    frames = make_frames(args.cycles + args.warmup, args.width, args.height)

    t_vis, t_dec, t_rag, t_vrag = [], [], [], []
    for i, frame in enumerate(frames):
        t0 = time.perf_counter()
        state = vision.extract_state(frame, args.preset, "bench")["state"]
        t1 = time.perf_counter()
        engine.evaluate_multimodal(frame, args.preset, "bench", "bench", state)
        t2 = time.perf_counter()
        rag_engine.search_sop(state, args.preset)
        t3 = time.perf_counter()
        visual_rag_engine.match_frame(frame, args.preset)
        t4 = time.perf_counter()
        if i >= args.warmup:
            t_vis.append((t1 - t0) * 1000)
            t_dec.append((t2 - t1) * 1000)
            t_rag.append((t3 - t2) * 1000)
            t_vrag.append((t4 - t3) * 1000)

    total = [a + b + c + d for a, b, c, d in zip(t_vis, t_dec, t_rag, t_vrag)]
    print(f"CPU: {platform.processor()} | Python {platform.python_version()} | OpenCV {cv2.__version__}")
    print(f"Frame: {args.width}x{args.height}, preset={args.preset}, cycles={args.cycles} (warmup {args.warmup})\n")
    print("| Stage                        |     Mean    |   Median    |     P95     |")
    print("|------------------------------|-------------|-------------|-------------|")
    summarize("Vision extraction (MOG2)", t_vis)
    summarize("DJev decision (embedded)", t_dec)
    summarize("SOP RAG search", t_rag)
    summarize("Visual Example RAG", t_vrag)
    mean_total = summarize("Total", total)
    print(f"\nSingle-thread upper bound: ~{1000.0 / mean_total:.0f} samples/s")


if __name__ == "__main__":
    main()
