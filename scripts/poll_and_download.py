import os
import sys
import time
import requests

api_key = os.environ.get("GEMINI_API_KEY")
op_name = sys.argv[1] if len(sys.argv) > 1 else "models/veo-3.1-fast-generate-preview/operations/wqwgib0ax1x6"
output_file = sys.argv[2] if len(sys.argv) > 2 else "app/static/sample_videos/river_flood_breach_detected.mp4"

url = f"https://generativelanguage.googleapis.com/v1beta/{op_name}?key={api_key}"
print(f"Polling operation: {op_name} -> saving to: {output_file}")

os.makedirs(os.path.dirname(output_file), exist_ok=True)

for i in range(25):
    resp = requests.get(url)
    if not resp.ok:
        print(f"Error checking op: {resp.status_code} - {resp.text}")
        time.sleep(5)
        continue

    data = resp.json()
    if data.get("done"):
        if "error" in data:
            print("Generation Error:", data["error"])
            sys.exit(1)

        samples = data.get("response", {}).get("generateVideoResponse", {}).get("generatedSamples", [])
        if not samples:
            print("No video sample returned in response:", data)
            sys.exit(1)

        download_uri = samples[0]["video"]["uri"]
        download_url = f"{download_uri}&key={api_key}"
        print(f"Downloading generated video from {download_uri}...")

        dl_resp = requests.get(download_url, stream=True)
        if dl_resp.ok:
            with open(output_file, "wb") as f:
                for chunk in dl_resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            size_mb = os.path.getsize(output_file) / (1024 * 1024)
            print(f"[SUCCESS] Video saved: {output_file} ({size_mb:.2f} MB)")
            sys.exit(0)
        else:
            print(f"Download failed: {dl_resp.status_code}")
            sys.exit(1)

    print(f"Waiting for Veo rendering... ({i+1}/25)")
    time.sleep(5)

print("Polling timed out.")
sys.exit(1)
