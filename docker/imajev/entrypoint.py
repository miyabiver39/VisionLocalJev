"""
imajev コンテナのエントリポイント。

1. ベースモデル (Qwen3.5) とアダプタ (imajev) を固定リビジョンで /data に用意する (2 回目以降はキャッシュを使用)
2. imajev の TypeSafe System One 互換サーバーを起動する
3. ウォームアップ用のリクエストを送り、GPU カーネルの初回コンパイルを済ませる
4. /tmp/imajev-ready を作成する (docker の healthcheck がこれを見て healthy になる)

環境変数:
  IMAJEV_SIZE       2b | 4b | 9b (既定 4b)
  IMAJEV_ROTATIONS  1 = 最速 / 4 = imajev 既定 (選択肢の並べ替え平均、約 4 倍遅い)
  IMAJEV_WARMUP     1 = ウォームアップする (既定) / 0 = しない
  PORT              待ち受けポート (既定 8765)
"""

import base64
import io
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests
from huggingface_hub import snapshot_download

# (ベースモデル, リビジョン, バンドルファイル, アダプタのリビジョン)
PINNED = {
    "2b": ("Qwen/Qwen3.5-2B", "15852e8c16360a2fea060d615a32b45270f8a8fc", "artifacts/model.json",
           "66e4b8c808bae45ca98bed5edd2de6323bdb0576"),
    "4b": ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a", "artifacts/model-qwen4b.json",
           "712891d1192c6491441a19bdea254d8a6e1048d0"),
    "9b": ("Qwen/Qwen3.5-9B", "c202236235762e1c871ad0ccb60c8ee5ba337b9a", "artifacts/model-qwen9b.json",
           "7301600591665b941d324b1a297e097b56724f9b"),
}
READY_FILE = Path("/tmp/imajev-ready")


def log(msg):
    print(f"[imajev-entrypoint] {msg}", flush=True)


def fetch(repo, revision, **kwargs):
    """ローカルにあればそれを使い、無ければダウンロードする。"""
    try:
        return snapshot_download(repo, revision=revision, local_files_only=True, **kwargs)
    except Exception:
        log(f"downloading {repo}@{revision[:8]} ...")
        return snapshot_download(repo, revision=revision, **kwargs)


def prepare(size, data):
    repo, revision, bundle, adapter_rev = PINNED[size]
    base_path = fetch(repo, revision, ignore_patterns=["*.md", ".gitattributes"])
    Path(bundle).parent.mkdir(exist_ok=True)
    Path(bundle).write_text(json.dumps({"repo": repo, "revision": revision, "path": base_path}, indent=2) + "\n")
    adapter_dir = Path(data) / "adapters" / f"imajev-{size}"
    fetch(f"mohit67890/imajev-{size}", adapter_rev, local_dir=str(adapter_dir))
    log(f"base model: {base_path}")
    log(f"adapter   : {adapter_dir} @ {adapter_rev[:8]}")
    return bundle, adapter_dir


def warmup_image(width, height):
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), (70, 70, 75))
    d = ImageDraw.Draw(img)
    d.rectangle((0, int(height * 0.75), width, height), fill=(90, 90, 95))
    d.ellipse((width // 2 - 60, height // 3, width // 2 + 60, height // 3 + 160), fill=(255, 120, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def warmup(port):
    """アプリが送る画像サイズ (合成カメラ 960x540 / 評価用 640x360) で 1 回ずつ推論し、カーネルをコンパイルしておく。"""
    url = f"http://127.0.0.1:{port}/v1/systemone"
    questions = {
        "hazard": {"type": "choice", "instructions": "What is visible?", "criteria": {"none": None, "fire": None}},
        "urgency": {"type": "score", "instructions": "How urgent?", "criteria": ["low", "medium", "high"]},
        "evacuate": {"type": "noul", "instructions": "Is evacuation required?"},
    }
    for w, h in ((960, 540), (640, 360)):
        t0 = time.time()
        r = requests.post(url, json={"state": "warmup", "questions": questions, "images": [warmup_image(w, h)]},
                          timeout=1800)
        r.raise_for_status()
        log(f"warmup {w}x{h}: {time.time() - t0:.1f}s")


def main():
    size = os.environ.get("IMAJEV_SIZE", "4b")
    if size not in PINNED:
        sys.exit(f"IMAJEV_SIZE must be one of {sorted(PINNED)}, got {size!r}")
    port = os.environ.get("PORT", "8765")
    data = os.environ.get("IMAJEV_DATA", "/data")
    READY_FILE.unlink(missing_ok=True)

    bundle, adapter_dir = prepare(size, data)

    cmd = [sys.executable, "scripts/playground/server.py", "--backend", "torch",
           "--model-bundle", bundle, "--adapter", str(adapter_dir),
           "--calibration", str(adapter_dir / "calibration.json"),
           "--rotations", os.environ.get("IMAJEV_ROTATIONS", "1"),
           "--model-name", f"imajev-{size}", "--host", "0.0.0.0", "--port", port]
    env = dict(os.environ, PYTHONPATH="src:scripts")
    log("starting: " + " ".join(cmd))
    server = subprocess.Popen(cmd, env=env)

    def stop(signum, _frame):
        server.send_signal(signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    # モデルの読み込み完了を待つ
    while server.poll() is None:
        try:
            if requests.get(f"http://127.0.0.1:{port}/v1/models", timeout=3).json().get("loaded"):
                break
        except (requests.RequestException, ValueError):
            pass
        time.sleep(3)
    if server.poll() is not None:
        sys.exit(server.returncode)

    if os.environ.get("IMAJEV_WARMUP", "1") == "1":
        try:
            warmup(port)
        except Exception as e:  # ウォームアップ失敗はサーバー不調とみなし、ready にしない
            log(f"warmup failed: {e}")
            server.terminate()
            sys.exit(1)
    READY_FILE.touch()
    log(f"ready: imajev-{size} on port {port}")
    sys.exit(server.wait())


if __name__ == "__main__":
    main()
