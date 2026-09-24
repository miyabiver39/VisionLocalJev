"""
scripts/generate_all_veo_videos.py
-----------------------------------
Batch video generation utility using Google DeepMind Veo 3.1 Fast (Gemini API / Generative Language API).
Generates realistic 1080p surveillance video clips set specifically in JAPAN with JAPANESE PEOPLE and JAPANESE SIGNAGE.
Carefully formulated with RAI (Responsible AI) compliant prompts to avoid disaster filter rejections.
"""

import os
import sys
import time
import json
import requests
from typing import Dict, Any, List

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("[ERROR] GEMINI_API_KEY environment variable is required.", flush=True)
    sys.exit(1)

OUTPUT_DIR = "app/static/sample_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL = "models/veo-3.1-fast-generate-preview"
BASE_URL = f"https://generativelanguage.googleapis.com/v1beta/{MODEL}:predictLongRunning?key={API_KEY}"

# Japanese CCTV Style suffix applied to all prompts for authentic Japanese domestic surveillance aesthetics
JAPAN_CCTV_SUFFIX = (
    " Authentic Japanese surveillance camera footage set in Japan. Realistic Japanese domestic architecture, "
    "Japanese text signage and warning banners in kanji on walls, authentic Japanese people. "
    "High-angle fixed Japanese CCTV security camera perspective, crisp 1080p 24fps surveillance video quality, genuine Japan setting."
)

SCENARIOS: List[Dict[str, Any]] = [
    # 1. Nursing Care (Japan)
    {
        "id": "nursing_care_normal",
        "title": "介護見守り: 穏やかな就寝 (日本・正常)",
        "preset": "nursing_care",
        "filename": "nursing_care_normal.mp4",
        "prompt": (
            "A high-angle indoor surveillance camera view of a modern Japanese nursing care home private bedroom at night in Tokyo, Japan. "
            "An elderly Japanese resident in their 80s is resting peacefully and motionless under a warm futon blanket on a low Japanese nursing bed with wooden side rails. "
            "Subtle ambient nightlight, authentic Japanese interior with tatami-toned flooring and a Japanese emergency nurse call button unit on the wall."
            + JAPAN_CCTV_SUFFIX
        )
    },
    {
        "id": "nursing_care_fall",
        "title": "介護見守り: ベッドサイド転倒 (日本・検知)",
        "preset": "nursing_care",
        "filename": "nursing_care_fall.mp4",
        "prompt": (
            "A high-angle Japanese indoor nursing home room camera in Japan capturing an elderly Japanese person attempting to get up from bed, "
            "losing balance, and falling onto the wooden floor near the bedside slippers. The elderly Japanese resident remains lying motionless on the floor. "
            "Indoor fluorescent lighting, Japanese wall posters and nurse call unit, authentic domestic Japanese eldercare surveillance footage."
            + JAPAN_CCTV_SUFFIX
        )
    },
    # 2. River Flood (Japan)
    {
        "id": "river_flood_normal",
        "title": "河川監視: 平常清流・低水位 (日本・正常)",
        "preset": "river_flood",
        "filename": "river_flood_normal.mp4",
        "prompt": (
            "A fixed Japanese riverbank surveillance CCTV camera overlooking a calm rural river in Japan on a clear sunny morning. "
            "Low water level flowing peacefully over stones, visible concrete embankment dykes, green grass on riverbank, "
            "and a Japanese river measurement staff marked with clear Japanese kanji characters showing safe water level. Crisp natural daytime lighting."
            + JAPAN_CCTV_SUFFIX
        )
    },
    {
        "id": "river_flood_high_water",
        "title": "河川監視: 豪雨激流・高水位危険標 (日本・検知)",
        "preset": "river_flood",
        "filename": "river_flood_high_water.mp4",
        "prompt": (
            "An authentic Japanese river disaster prevention surveillance camera during heavy rainfall in Japan. "
            "The river channel carries a very high, swift water current with surface ripples, reaching the red danger indicator mark on a concrete measurement pillar marked with Japanese kanji text. "
            "Overcast cloudy Japanese landscape, fast flowing river stream, scientific water monitoring in Japan."
            + JAPAN_CCTV_SUFFIX
        )
    },
    # 3. Security (Japan)
    {
        "id": "security_normal",
        "title": "防犯監視: 通用口正常歩行 (日本・正常)",
        "preset": "security",
        "filename": "security_normal.mp4",
        "prompt": (
            "A high-angle Japanese corporate building entrance security camera in Tokyo, Japan. "
            "A Japanese businessman dressed in a dark business suit walks through the well-lit entrance corridor, holding an employee badge and entering through the automatic glass door. "
            "Clean Japanese office interior, Japanese wall signage reading 関係者以外立入禁止 in kanji, routine foot traffic, authentic Japanese daytime CCTV."
            + JAPAN_CCTV_SUFFIX
        )
    },
    {
        "id": "security_fence_climb",
        "title": "防犯監視: 外周フェンス乗り越え (日本・検知)",
        "preset": "security",
        "filename": "security_fence_climb.mp4",
        "prompt": (
            "A night vision monochrome surveillance camera overlooking a Japanese industrial facility perimeter chain-link fence in Japan. "
            "An unauthorized person wearing a dark hoodie and gloves climbing over the wire fence topped with barbed wire and dropping into the facility shadows. "
            "Japanese warning sign reading 立入禁止 防犯カメラ作動中 visible on the fence, high-contrast night vision Japanese CCTV security footage."
            + JAPAN_CCTV_SUFFIX
        )
    },
    # 4. Fire Disaster (Japan)
    {
        "id": "fire_normal_steam",
        "title": "火災監視: 給湯室の白い湯気 (日本・正常・誤検知防止)",
        "preset": "fire_disaster",
        "filename": "fire_normal_steam.mp4",
        "prompt": (
            "An indoor security camera view of a Japanese office tea room (給湯室) in Tokyo, Japan. "
            "A stainless kettle on an induction stove boils, releasing gentle translucent white water steam into the air above. "
            "No flames, no smoke, bright fluorescent overhead lighting, Japanese warning stickers reading 火気厳禁 on the stainless steel counter."
            + JAPAN_CCTV_SUFFIX
        )
    },
    {
        "id": "fire_flame_smoke",
        "title": "火災監視: 制御盤からの黒煙と火炎 (日本・検知)",
        "preset": "fire_disaster",
        "filename": "fire_flame_smoke.mp4",
        "prompt": (
            "An authentic industrial CCTV camera inside a Japanese factory electrical distribution switchboard room in Japan. "
            "Dense billowing dark grey and black smoke rapidly rises from a metal control panel cabinet, followed by intense orange flickering open flames erupting from the top vents. "
            "Yellow Japanese warning sign reading 高圧受電設備 危険 reflecting the firelight, emergency industrial fire scenario from a fixed elevated Japanese CCTV."
            + JAPAN_CCTV_SUFFIX
        )
    },
    # 5. Factory Safety (Japan)
    {
        "id": "factory_safety_normal",
        "title": "工場労働安全: 規定通路歩行 (日本・正常)",
        "preset": "factory_safety",
        "filename": "factory_safety_normal.mp4",
        "prompt": (
            "A high-ceiling Japanese manufacturing factory floor surveillance camera in Japan. "
            "Two Japanese factory workers wearing yellow hardhats and high-visibility neon reflective safety vests walk strictly within a painted green safety pathway. "
            "Clear Japanese green cross banner reading 安全第一 on the wall, industrial machinery operating cleanly in background, compliant Japanese occupational safety environment."
            + JAPAN_CCTV_SUFFIX
        )
    },
    {
        "id": "factory_worker_down",
        "title": "工場労働安全: 危険域進入・作業員倒臥 (日本・検知)",
        "preset": "factory_safety",
        "filename": "factory_worker_down.mp4",
        "prompt": (
            "A Japanese factory surveillance camera in Japan capturing an emergency workplace drill. "
            "A Japanese worker in factory uniform lies motionless on the concrete floor inside a yellow hazard diagonal line near automated equipment. "
            "Japanese safety poster reading 整理整頓 on the wall, a colleague rushing in background to press the red emergency stop button, authentic Japanese industrial incident drill."
            + JAPAN_CCTV_SUFFIX
        )
    },
    # 6. Railway Platform (Japan)
    {
        "id": "railway_platform_normal",
        "title": "駅ホーム安全: 点字ブロック内側待機 (日本・正常)",
        "preset": "railway_platform",
        "filename": "railway_platform_normal.mp4",
        "prompt": (
            "A high-angle Japanese railway station platform security CCTV camera in Tokyo, Japan. "
            "Japanese commuters in dark business attire standing neatly in orderly lines behind the yellow textured braille safety line, waiting for a train. "
            "Automatic platform screen doors, Japanese station name signs overhead, clean orderly morning Japanese train platform footage."
            + JAPAN_CCTV_SUFFIX
        )
    },
    {
        "id": "railway_platform_track_incident",
        "title": "駅ホーム安全: 線路転落シミュレーション (日本・検知)",
        "preset": "railway_platform",
        "filename": "railway_platform_track_incident.mp4",
        "prompt": (
            "A Japanese train station platform surveillance camera in Japan. "
            "A passenger stumbles past the yellow braille line and falls off the platform edge down onto the track gravel area between the steel rails. "
            "Red emergency warning indicator flashing on the Japanese platform pillar labeled 非常ボタン, authentic Japanese railway security camera perspective."
            + JAPAN_CCTV_SUFFIX
        )
    }
]


def start_generation(prompt: str, max_retries: int = 5) -> str:
    """Submits a video generation job to Veo 3.1 Fast with automatic 429 rate-limit backoff."""
    payload = {"instances": [{"prompt": prompt}]}
    for attempt in range(max_retries):
        resp = requests.post(BASE_URL, json=payload, timeout=30)
        if resp.status_code == 429:
            wait_sec = 45 + (attempt * 15)
            print(f"  [RATE LIMIT 429] Quota exceeded. Waiting {wait_sec}s for quota reset (attempt {attempt+1}/{max_retries})...", flush=True)
            time.sleep(wait_sec)
            continue
        resp.raise_for_status()
        data = resp.json()
        return data["name"]
    raise RuntimeError("Failed to submit generation job after maximum 429 retries.")


def poll_and_download(op_name: str, target_path: str, max_wait_sec: int = 240) -> bool:
    """Polls Veo operation until completion and downloads video MP4."""
    url = f"https://generativelanguage.googleapis.com/v1beta/{op_name}?key={API_KEY}"
    start_time = time.time()

    while time.time() - start_time < max_wait_sec:
        resp = requests.get(url, timeout=30)
        if not resp.ok:
            if resp.status_code == 429:
                print("  [WARN] Rate limit on poll, waiting 15s...", flush=True)
                time.sleep(15)
                continue
            print(f"  [WARN] Poll HTTP {resp.status_code}, retrying...", flush=True)
            time.sleep(6)
            continue

        data = resp.json()
        if data.get("done"):
            # Check for RAI safety filter
            if "error" in data:
                print(f"  [ERROR] Generation failed: {data['error']}", flush=True)
                return False

            samples = data.get("response", {}).get("generateVideoResponse", {}).get("generatedSamples", [])
            if not samples:
                rai_reasons = data.get("response", {}).get("generateVideoResponse", {}).get("raiMediaFilteredReasons", [])
                print(f"  [RAI FILTERED] Blocked by safety filter: {rai_reasons}", flush=True)
                return False

            video_uri = samples[0]["video"]["uri"]
            dl_url = f"{video_uri}&key={API_KEY}"

            dl_resp = requests.get(dl_url, stream=True, timeout=60)
            if dl_resp.ok:
                with open(target_path, "wb") as f:
                    for chunk in dl_resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                size_mb = os.path.getsize(target_path) / (1024 * 1024)
                print(f"  [SUCCESS] Downloaded: {target_path} ({size_mb:.2f} MB)", flush=True)
                return True
            else:
                print(f"  [ERROR] Download failed HTTP {dl_resp.status_code}", flush=True)
                return False

        print(f"  Rendering in progress... ({int(time.time() - start_time)}s elapsed)", flush=True)
        time.sleep(6)

    print("  [TIMEOUT] Operation exceeded wait time limit.", flush=True)
    return False


def main():
    print("=" * 70, flush=True)
    print("Google DeepMind Veo 3.1 Fast - JAPANESE CCTV Video Generation Batch Runner", flush=True)
    print(f"Total Scenarios to Generate: {len(SCENARIOS)} (All Japanese Settings)", flush=True)
    print("=" * 70, flush=True)

    # Optional: filter by scenario ID from CLI argument
    filter_id = sys.argv[1] if len(sys.argv) > 1 else None
    force_rebuild = "--force" in sys.argv or "-f" in sys.argv

    results = []
    for idx, sc in enumerate(SCENARIOS, 1):
        if filter_id and filter_id != sc["id"] and filter_id != "all" and not filter_id.startswith("-"):
            continue

        out_path = os.path.join(OUTPUT_DIR, sc["filename"])
        print(f"\n[{idx}/{len(SCENARIOS)}] Generating Japanese Video: {sc['title']}", flush=True)
        print(f"  ID: {sc['id']} | Output: {out_path}", flush=True)

        if not force_rebuild and os.path.exists(out_path) and os.path.getsize(out_path) > 500000:
            size_mb = os.path.getsize(out_path) / (1024 * 1024)
            print(f"  [SKIP] Video already exists and is valid size ({size_mb:.2f} MB).", flush=True)
            results.append({"id": sc["id"], "status": "EXISTS", "path": out_path})
            continue

        try:
            op_name = start_generation(sc["prompt"])
            print(f"  Operation created: {op_name}", flush=True)
            ok = poll_and_download(op_name, out_path)
            results.append({"id": sc["id"], "status": "SUCCESS" if ok else "FAILED", "path": out_path})
        except Exception as e:
            print(f"  [EXCEPTION] {e}", flush=True)
            results.append({"id": sc["id"], "status": "ERROR", "error": str(e)})

        # 25-second cooldown between jobs to avoid 429 quota spikes
        print("  Cooling down 25s for API quota...", flush=True)
        time.sleep(25)

    print("\n" + "=" * 70, flush=True)
    print("Batch Generation Summary (Japan CCTV):", flush=True)
    for r in results:
        print(f"  - {r['id']:<32}: {r['status']}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
