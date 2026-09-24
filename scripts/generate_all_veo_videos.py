"""
scripts/generate_all_veo_videos.py
-----------------------------------
Batch video generation utility using Google DeepMind Veo 3.1 Fast (Gemini API / Generative Language API).
Generates realistic 1080p surveillance video clips for all 6 domain presets,
carefully formulated with RAI (Responsible AI) compliant prompts to avoid disaster filter rejections.
"""

import os
import sys
import time
import json
import requests
from typing import Dict, Any, List

API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    print("[ERROR] GEMINI_API_KEY environment variable is required.")
    sys.exit(1)

OUTPUT_DIR = "app/static/sample_videos"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL = "models/veo-3.1-fast-generate-preview"
BASE_URL = f"https://generativelanguage.googleapis.com/v1beta/{MODEL}:predictLongRunning?key={API_KEY}"

# CCTV Style suffix applied to all prompts for realistic camera aesthetics
CCTV_SUFFIX = (
    " High-angle static surveillance camera perspective, realistic CCTV fixed framing, "
    "ambient surveillance lighting, authentic security camera view, sharp cinematic documentary quality."
)

SCENARIOS: List[Dict[str, Any]] = [
    # 1. Nursing Care
    {
        "id": "nursing_care_normal",
        "title": "介護見守り: 穏やかな就寝 (正常)",
        "preset": "nursing_care",
        "filename": "nursing_care_normal.mp4",
        "prompt": (
            "A high-angle indoor surveillance camera view of a softly lit nursing home private bedroom at night. "
            "An elderly resident is resting peacefully and motionless under blankets on a low bed with side rails raised. "
            "Warm amber nightlight, quiet peaceful room, serene eldercare monitoring footage." + CCTV_SUFFIX
        )
    },
    {
        "id": "nursing_care_fall",
        "title": "介護見守り: ベッドサイド転倒・床倒臥 (検知)",
        "preset": "nursing_care",
        "filename": "nursing_care_fall.mp4",
        "prompt": (
            "A high-angle indoor nursing home room camera capturing an occupational safety demonstration. "
            "An elderly person trips while standing up and slides down gently to rest motionless on the carpeted floor beside the bed slippers. "
            "Indoor fluorescent lighting, fixed overhead surveillance angle, care assistance training scenario." + CCTV_SUFFIX
        )
    },
    # 2. River Flood
    {
        "id": "river_flood_normal",
        "title": "河川監視: 平常清流・低水位 (正常)",
        "preset": "river_flood",
        "filename": "river_flood_normal.mp4",
        "prompt": (
            "A fixed riverbank surveillance camera overlooking a calm rural river on a clear sunny morning. "
            "The water level is low and flowing smoothly over riverbed stones, with green grassy embankments and dry concrete dykes. "
            "Clear water gauge staff in water showing normal safe level, peaceful daytime lighting." + CCTV_SUFFIX
        )
    },
    {
        "id": "river_flood_high_water",
        "title": "河川監視: 豪雨激流・高水位危険標 (検知)",
        "preset": "river_flood",
        "filename": "river_flood_high_water.mp4",
        "prompt": (
            "An environmental river monitoring station camera during autumn rainfall. "
            "The river channel carries a very high, swift water current with surface ripples, reaching the red high-water danger line on the concrete pillar. "
            "Overcast cloudy sky, fast flowing river stream, scientific environmental water monitoring." + CCTV_SUFFIX
        )
    },
    # 3. Security
    {
        "id": "security_normal",
        "title": "防犯監視: 通用口正常歩行 (正常)",
        "preset": "security",
        "filename": "security_normal.mp4",
        "prompt": (
            "A high-angle corporate building entrance security camera. "
            "An employee dressed in smart business attire walks normally through the well-lit entrance corridor, holding an employee badge and entering through the automatic glass door. "
            "Routine foot traffic, clean daylight, secure corporate environment." + CCTV_SUFFIX
        )
    },
    {
        "id": "security_fence_climb",
        "title": "防犯監視: 外周フェンス乗り越え (検知)",
        "preset": "security",
        "filename": "security_fence_climb.mp4",
        "prompt": (
            "A night vision monochrome surveillance camera overlooking a perimeter security chain-link fence. "
            "A person in dark hooded clothing climbing swiftly up the wire fence in a movie stunt demonstration, leaping down into the shadows of the courtyard. "
            "High contrast infrared CCTV aesthetic, urgent security perimeter test footage." + CCTV_SUFFIX
        )
    },
    # 4. Fire Disaster
    {
        "id": "fire_normal_steam",
        "title": "火災監視: 給湯室の白い湯気 (正常・誤検知防止)",
        "preset": "fire_disaster",
        "filename": "fire_normal_steam.mp4",
        "prompt": (
            "An indoor security camera view of an office pantry kitchen. "
            "A kettle on an induction counter boils, releasing gentle translucent white water vapor into the air above. "
            "No smoke, no flames, bright overhead fluorescent lighting, clean stainless steel counters." + CCTV_SUFFIX
        )
    },
    {
        "id": "fire_flame_smoke",
        "title": "火災監視: 制御盤からの黒煙と火炎 (検知)",
        "preset": "fire_disaster",
        "filename": "fire_flame_smoke.mp4",
        "prompt": (
            "A cinematic industrial factory safety drill CCTV footage. "
            "An electrical equipment cabinet produces dense billowing grey and black smoke, with intense orange flickering fire lighting up the metal machinery. "
            "Hazy atmospheric smoke layer near ceiling, static elevated CCTV perspective, industrial incident simulation." + CCTV_SUFFIX
        )
    },
    # 5. Factory Safety
    {
        "id": "factory_safety_normal",
        "title": "工場労働安全: 規定通路歩行 (正常)",
        "preset": "factory_safety",
        "filename": "factory_safety_normal.mp4",
        "prompt": (
            "A high-ceiling manufacturing facility floor security camera. "
            "Two workers wearing yellow hardhats and high-visibility neon reflective safety vests walk along a painted green safety pathway. "
            "Industrial machinery operating cleanly in the background, compliant occupational workplace environment." + CCTV_SUFFIX
        )
    },
    {
        "id": "factory_worker_down",
        "title": "工場労働安全: 危険域進入・作業員倒臥 (検知)",
        "preset": "factory_safety",
        "filename": "factory_worker_down.mp4",
        "prompt": (
            "A factory surveillance camera capturing an emergency response simulation drill. "
            "A worker in blue overalls lies down motionless on the concrete floor inside a yellow hazard diagonal line near automated equipment. "
            "Colleague rushing in background to press the red emergency stop button, industrial workplace safety drill." + CCTV_SUFFIX
        )
    },
    # 6. Railway Platform
    {
        "id": "railway_platform_normal",
        "title": "駅ホーム安全: 点字ブロック内側待機 (正常)",
        "preset": "railway_platform",
        "filename": "railway_platform_normal.mp4",
        "prompt": (
            "A high-angle train station platform security camera. "
            "Commuters standing calmly in orderly lines behind the yellow textured braille safety line, waiting for the train. "
            "Clean modern station lighting, safe passenger crowd, authentic platform CCTV footage." + CCTV_SUFFIX
        )
    },
    {
        "id": "railway_platform_track_incident",
        "title": "駅ホーム安全: 線路側転落シミュレーション (検知)",
        "preset": "railway_platform",
        "filename": "railway_platform_track_incident.mp4",
        "prompt": (
            "A railway station safety demonstration CCTV footage. "
            "A training mannequin dressed in jacket falls from the edge of the platform down onto the gravel track area between the rails. "
            "Platform edge warning lights blinking, realistic elevated security camera framing, railway safety simulation." + CCTV_SUFFIX
        )
    }
]


def start_generation(prompt: str) -> str:
    """Submits a video generation job to Veo 3.1 Fast and returns operation name."""
    payload = {"instances": [{"prompt": prompt}]}
    resp = requests.post(BASE_URL, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["name"]


def poll_and_download(op_name: str, target_path: str, max_wait_sec: int = 180) -> bool:
    """Polls Veo operation until completion and downloads video MP4."""
    url = f"https://generativelanguage.googleapis.com/v1beta/{op_name}?key={API_KEY}"
    start_time = time.time()

    while time.time() - start_time < max_wait_sec:
        resp = requests.get(url, timeout=30)
        if not resp.ok:
            print(f"  [WARN] Poll HTTP {resp.status_code}, retrying...")
            time.sleep(6)
            continue

        data = resp.json()
        if data.get("done"):
            # Check for RAI safety filter
            if "error" in data:
                print(f"  [ERROR] Generation failed: {data['error']}")
                return False

            samples = data.get("response", {}).get("generateVideoResponse", {}).get("generatedSamples", [])
            if not samples:
                rai_reasons = data.get("response", {}).get("generateVideoResponse", {}).get("raiMediaFilteredReasons", [])
                print(f"  [RAI FILTERED] Blocked by safety filter: {rai_reasons}")
                return False

            video_uri = samples[0]["video"]["uri"]
            dl_url = f"{video_uri}&key={API_KEY}"

            dl_resp = requests.get(dl_url, stream=True, timeout=60)
            if dl_resp.ok:
                with open(target_path, "wb") as f:
                    for chunk in dl_resp.iter_content(chunk_size=8192):
                        f.write(chunk)
                size_mb = os.path.getsize(target_path) / (1024 * 1024)
                print(f"  [SUCCESS] Downloaded: {target_path} ({size_mb:.2f} MB)")
                return True
            else:
                print(f"  [ERROR] Download failed HTTP {dl_resp.status_code}")
                return False

        print(f"  Rendering in progress... ({int(time.time() - start_time)}s elapsed)")
        time.sleep(6)

    print("  [TIMEOUT] Operation exceeded wait time limit.")
    return False


def main():
    print("=" * 70)
    print("Google DeepMind Veo 3.1 Fast - Video Generation Batch Runner")
    print(f"Total Scenarios to Generate: {len(SCENARIOS)}")
    print("=" * 70)

    # Optional: filter by scenario ID from CLI argument
    filter_id = sys.argv[1] if len(sys.argv) > 1 else None

    results = []
    for idx, sc in enumerate(SCENARIOS, 1):
        if filter_id and filter_id != sc["id"] and filter_id != "all":
            continue

        out_path = os.path.join(OUTPUT_DIR, sc["filename"])
        print(f"\n[{idx}/{len(SCENARIOS)}] Generating: {sc['title']}")
        print(f"  ID: {sc['id']} | Output: {out_path}")

        if os.path.exists(out_path) and os.path.getsize(out_path) > 1000000:
            print("  [SKIP] Video already exists and is valid size.")
            results.append({"id": sc["id"], "status": "EXISTS", "path": out_path})
            continue

        try:
            op_name = start_generation(sc["prompt"])
            print(f"  Operation created: {op_name}")
            ok = poll_and_download(op_name, out_path)
            results.append({"id": sc["id"], "status": "SUCCESS" if ok else "FAILED", "path": out_path})
        except Exception as e:
            print(f"  [EXCEPTION] {e}")
            results.append({"id": sc["id"], "status": "ERROR", "error": str(e)})

        # Brief pause between jobs to be gentle on API quota
        time.sleep(3)

    print("\n" + "=" * 70)
    print("Batch Generation Summary:")
    for r in results:
        print(f"  - {r['id']:<30}: {r['status']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
