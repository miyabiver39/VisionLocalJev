"""
typesafe_remote_eval.py
-----------------------
TypeSafe System One 互換サーバー (imajev / Qev / Kev / TypeSafe Jev) に、6 プリセット x (正常 / 異常) の
合成テスト画像を送り、アプリの判定エンジン (DJEV_MODE=remote) 経由で結果とレイテンシを確認する。

    python scripts/typesafe_remote_eval.py --url http://127.0.0.1:8765 --model imajev-2b --image-mode images

合成画像は図形で描いた簡易シーンのため、判定精度の評価ではなく疎通・形式・速度の確認用。
"""

import argparse
import os
import statistics
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.decision import DecisionEngine  # noqa: E402
from app.typesafe import DecisionEngineError  # noqa: E402

W, H = 640, 360


def person(img, x, y, lying=False, color=(60, 60, 200)):
    """棒人間。lying=True なら床に倒れた姿勢。"""
    if lying:
        cv2.circle(img, (x, y), 14, color, -1)
        cv2.line(img, (x + 14, y), (x + 90, y + 4), color, 10)
        cv2.line(img, (x + 90, y + 4), (x + 140, y - 10), color, 8)
        cv2.line(img, (x + 90, y + 4), (x + 140, y + 18), color, 8)
        cv2.line(img, (x + 40, y), (x + 60, y - 30), color, 7)
    else:
        cv2.circle(img, (x, y), 14, color, -1)
        cv2.line(img, (x, y + 14), (x, y + 90), color, 10)
        cv2.line(img, (x, y + 90), (x - 18, y + 150), color, 8)
        cv2.line(img, (x, y + 90), (x + 18, y + 150), color, 8)
        cv2.line(img, (x, y + 35), (x - 30, y + 70), color, 7)
        cv2.line(img, (x, y + 35), (x + 30, y + 70), color, 7)


def scene(preset: str, abnormal: bool) -> np.ndarray:
    img = np.zeros((H, W, 3), np.uint8)
    if preset == "fire_disaster":
        img[:] = (70, 70, 75)
        cv2.rectangle(img, (0, 280), (W, H), (90, 90, 95), -1)
        cv2.rectangle(img, (80, 120, 180, 170), (110, 110, 115), -1)  # cabinet
        if abnormal:
            for i, (cx, r) in enumerate([(360, 90), (330, 60), (400, 70)]):
                cv2.ellipse(img, (cx, 230), (r // 2, r), 0, 0, 360, (0, 90 + 40 * i, 255), -1)
            cv2.ellipse(img, (360, 90), (160, 60), 0, 0, 360, (30, 30, 30), -1)  # black smoke
    elif preset == "security":
        img[:] = (40, 45, 40)
        for x in range(0, W, 30):
            cv2.line(img, (x, 120), (x, 260), (160, 160, 160), 2)  # fence posts
        cv2.line(img, (0, 120), (W, 120), (170, 170, 170), 3)
        cv2.line(img, (0, 190), (W, 190), (170, 170, 170), 2)
        if abnormal:
            person(img, 320, 60)  # climbing over the fence top
        else:
            person(img, 150, 190)  # walking on the path in front
    elif preset == "nursing_care":
        img[:] = (170, 180, 190)
        cv2.rectangle(img, (0, 270), (W, H), (120, 140, 160), -1)
        cv2.rectangle(img, (60, 170), (330, 250), (230, 230, 240), -1)  # bed
        if abnormal:
            person(img, 380, 300, lying=True)
        else:
            cv2.circle(img, (100, 185), 16, (60, 60, 200), -1)
            cv2.rectangle(img, (115, 175), (320, 215), (200, 170, 140), -1)  # blanket
    elif preset == "river_flood":
        img[:] = (200, 170, 130)  # sky
        if abnormal:
            cv2.rectangle(img, (0, 120), (W, H), (60, 100, 140), -1)  # brown water over everything
            cv2.rectangle(img, (0, 150), (W, 170), (80, 80, 80), -1)  # submerged embankment
        else:
            cv2.rectangle(img, (0, 160), (W, 230), (60, 140, 60), -1)  # green bank
            cv2.rectangle(img, (0, 230), (W, H), (150, 110, 60), -1)  # river
    elif preset == "factory_safety":
        img[:] = (90, 90, 95)
        cv2.rectangle(img, (0, 280), (W, H), (70, 110, 70), -1)  # floor
        cv2.rectangle(img, (400, 100), (580, 260), (0, 170, 230), -1)  # machine
        cv2.putText(img, "DANGER", (430, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2)
        if abnormal:
            person(img, 300, 300, lying=True)
        else:
            person(img, 150, 120)
            cv2.ellipse(img, (150, 100), (18, 10), 0, 180, 360, (0, 220, 255), -1)  # helmet
    elif preset == "railway_platform":
        img[:] = (150, 150, 150)
        cv2.rectangle(img, (0, 0), (W, 200), (180, 180, 180), -1)  # platform
        cv2.rectangle(img, (0, 170), (W, 185), (0, 215, 255), -1)  # yellow line
        cv2.rectangle(img, (0, 200), (W, H), (60, 60, 70), -1)  # track bed
        for y in (250, 320):
            cv2.line(img, (0, y), (W, y), (200, 200, 210), 5)  # rails
        if abnormal:
            person(img, 260, 280, lying=True)
        else:
            person(img, 300, 10)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8765")
    ap.add_argument("--model", default="imajev-2b")
    ap.add_argument("--image-mode", default="images", choices=["images", "state_content", "none"])
    ap.add_argument("--timeout", type=float, default=60)
    ap.add_argument("--save-images", default="", help="合成画像を保存するディレクトリ")
    args = ap.parse_args()

    engine = DecisionEngine(mode="remote", remote_url=args.url, remote_model=args.model,
                            image_mode=args.image_mode, timeout=args.timeout)
    latencies = []
    print(f"server={engine.client.url} model={args.model} image_mode={args.image_mode}\n")
    for preset_id in ["security", "fire_disaster", "nursing_care", "river_flood", "factory_safety", "railway_platform"]:
        for abnormal in (False, True):
            frame = scene(preset_id, abnormal)
            if args.save_images:
                os.makedirs(args.save_images, exist_ok=True)
                cv2.imwrite(os.path.join(args.save_images, f"{preset_id}_{'abnormal' if abnormal else 'normal'}.png"), frame)
            state = f"Fixed surveillance camera. Monitoring purpose: {engine.presets[preset_id]['description']}"
            t0 = time.perf_counter()
            try:
                res = engine.evaluate_multimodal(frame, preset_id, "eval", "eval", state)
            except DecisionEngineError as e:
                print(f"{preset_id:17s} {'ABN' if abnormal else 'NRM'}  ERROR [{e.kind}] {e.message}")
                continue
            ms = (time.perf_counter() - t0) * 1000
            latencies.append(ms)
            parts = []
            for qid, a in res["answers"].items():
                if a["type"] == "choice":
                    parts.append(f"{a['choice']}({a['probabilities'][a['choice']]:.2f})")
                elif a["type"] == "score":
                    parts.append(f"score {a['score']:.2f}/{len(a['legend']) - 1}")
                else:
                    parts.append(f"P(yes) {a['noul']:.2f}")
            flag = "ALERT" if res["is_alert"] else "-"
            print(f"{preset_id:17s} {'ABN' if abnormal else 'NRM'}  {ms:7.0f}ms  {flag:5s}  " + " | ".join(parts))
    if latencies:
        print(f"\nlatency: median {statistics.median(latencies):.0f}ms / min {min(latencies):.0f}ms / max {max(latencies):.0f}ms (n={len(latencies)})")


if __name__ == "__main__":
    main()
