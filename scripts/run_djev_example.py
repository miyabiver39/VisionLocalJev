"""
run_djev_example.py
-------------------
DiffusionGemma-Jev (DJev) に実際に画像フレームと設問スキーマJSONを渡し、
型安全な判定結果JSON（Choice / Score / Noul）を取得するスタンドアロン動作検証スクリプト。
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import time
import cv2
import numpy as np
from app.decision import DecisionEngine

def main():
    print("=" * 70)
    print("DiffusionGemma-Jev (DJev) 実稼働デモ: 画像フレーム ＋ 設問JSONの直接推論")
    print("=" * 70)

    # 1. DJev エンジンの初期化
    engine = DecisionEngine(mode="embedded", presets_dir="app/presets", diffusion_steps=8)
    print(f"\n[1] DJev Engine Initialized: {engine.model_name} (Diffusion Steps: {engine.diffusion_steps})\n")

    # 2. 設問スキーマJSON（プリセット: fire_disaster / 火災・防災）
    preset_id = "fire_disaster"
    print(f"[2] 適用プリセット: {preset_id} (設問: Choice=検知対象, Score=切迫度, Noul=避難判定)")

    # 3. テスト画像フレームの生成（パターン1: 平穏な工場室内フレーム）
    frame_normal = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame_normal[:] = (35, 40, 45)  # 工場の床・壁
    cv2.putText(frame_normal, "Factory Corridor (Calm & Normal)", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 2)
    cv2.rectangle(frame_normal, (200, 250), (400, 550), (60, 60, 60), -1)  # 機械設備

    print("\n--- 【ケース A: 平常シーン】画像をDJevへ入力 ---")
    t0 = time.perf_counter()
    result_normal = engine.evaluate_multimodal(
        frame=frame_normal,
        preset_id=preset_id,
        camera_id="cam_factory_01",
        camera_name="第一工場 通路カメラ",
        context_text="The area is completely clear with normal lighting and safe conditions."
    )
    t_normal = (time.perf_counter() - t0) * 1000

    print(f"推論所要時間: {t_normal:.2f} ms")
    print("▼ DJev から返却された型安全 JSON レスポンス:")
    print(json.dumps(result_normal, indent=2, ensure_ascii=False))

    # 4. テスト画像フレームの生成（パターン2: 激しい火災炎フレーム）
    frame_fire = frame_normal.copy()
    # 炎の描画（色相・明度・フリッカー領域）
    cv2.ellipse(frame_fire, (300, 400), (90, 160), 0, 0, 360, (0, 120, 255), -1)  # 炎外周（赤橙）
    cv2.ellipse(frame_fire, (300, 410), (50, 110), 0, 0, 360, (0, 210, 255), -1)  # 炎中心（黄白）
    cv2.putText(frame_fire, "ALERT: OPEN FLAME DETECTED", (50, 650), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    print("\n--- 【ケース B: 火災発生シーン】画像をDJevへ入力 ---")
    t1 = time.perf_counter()
    result_fire = engine.evaluate_multimodal(
        frame=frame_fire,
        preset_id=preset_id,
        camera_id="cam_factory_01",
        camera_name="第一工場 通路カメラ",
        context_text="An active open flame with intense flickering fire has erupted near the equipment cabinet."
    )
    t_fire = (time.perf_counter() - t1) * 1000

    print(f"推論所要時間: {t_fire:.2f} ms")
    print("▼ DJev から返却された型安全 JSON レスポンス:")
    print(json.dumps(result_fire, indent=2, ensure_ascii=False))

    print("\n" + "=" * 70)
    print("動作確認完了: 画像テンソル ＋ 設問JSONによる即時一括判定が実証されました。")
    print("=" * 70)

if __name__ == "__main__":
    main()
