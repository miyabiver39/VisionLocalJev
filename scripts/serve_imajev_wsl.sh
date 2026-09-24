#!/bin/bash
# WSL2 + ROCm (scripts/setup_wsl_rocm.sh で構築した ~/dg-venv) で imajev を TypeSafe System One 互換サーバーとして起動する。
#
#   初回のみ:
#     source ~/dg-venv/bin/activate
#     git clone https://github.com/mohit67890/imajev ~/imajev && cd ~/imajev
#     pip install -e . --no-deps
#     pip install "pydantic>=2,<3" "Pillow>=11" "huggingface_hub>=0.30" "peft>=0.15" "safetensors>=0.5" \
#                 "fastapi>=0.110" "uvicorn>=0.29" "python-multipart>=0.0.9" flash-linear-attention
#     for s in 2b 4b; do python scripts/download_model.py --model $s; hf download mohit67890/imajev-$s --local-dir adapters/imajev-$s; done
#
#   起動:  bash scripts/serve_imajev_wsl.sh 4b 1     # <2b|4b> <rotations>
#   アプリ: DJEV_MODE=remote DJEV_SERVER_URL=http://127.0.0.1:8765 DJEV_MODEL=imajev-4b DJEV_IMAGE_MODE=images
#
# flash-linear-attention が無いと Qwen3.5 の Gated DeltaNet 層が参照実装になり、1 リクエスト数十秒かかる。
set -euo pipefail
SIZE=${1:-4b}
ROTATIONS=${2:-1}   # 1 = 最速 / 4 = imajev 既定 (選択肢の並びを 4 通り平均。精度がやや上がり、約 4 倍遅い)
PORT=${PORT:-8765}

source ~/dg-venv/bin/activate
export ROCPROFILER_REGISTER_ENABLED=0 HSA_ENABLE_DXG_DETECTION=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export HF_HOME=~/imajev/.cache/huggingface
cd ~/imajev

BUNDLE=()
if [ "$SIZE" = "4b" ]; then BUNDLE=(--model-bundle artifacts/model-qwen4b.json); fi
if [ "$SIZE" = "9b" ]; then BUNDLE=(--model-bundle artifacts/model-qwen9b.json); fi

PYTHONPATH=src:scripts exec python scripts/playground/server.py --backend torch "${BUNDLE[@]}" \
  --adapter "adapters/imajev-$SIZE" --calibration "adapters/imajev-$SIZE/calibration.json" \
  --rotations "$ROTATIONS" --model-name "imajev-$SIZE" --host 0.0.0.0 --port "$PORT"
