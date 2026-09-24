#!/bin/bash
# WSL2 (Ubuntu 24.04) + AMD Radeon (RDNA4 / RX 9060 XT で確認) で DiffusionGemma 検証環境を作る。
# 前提: Windows 側に AMD Adrenalin 26.2.2 以降 (確認: 26.8.1)。
#
#   wsl -d Ubuntu-24.04 -- bash scripts/setup_wsl_rocm.sh
#
# 詳細は docs/local_rocm_wsl.md を参照。
set -euo pipefail

ROCM_VER=7.2.4
AMDGPU_DEB=amdgpu-install_7.2.4.70204-1_all.deb
ROCDXG_VER=1.2.2
VENV=~/dg-venv

echo "== 1. ROCm ${ROCM_VER} ランタイム + ROCDXG (WSL 用 GPU ブリッジ)"
cd /tmp
wget -q "https://repo.radeon.com/amdgpu-install/${ROCM_VER}/ubuntu/noble/${AMDGPU_DEB}"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "./${AMDGPU_DEB}"
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rocm-hip-runtime rocminfo python3-venv python3-pip
wget -q "https://github.com/ROCm/librocdxg/releases/download/v${ROCDXG_VER}/rocdxg-roct_${ROCDXG_VER}_amd64.deb"
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "./rocdxg-roct_${ROCDXG_VER}_amd64.deb"
HSA_ENABLE_DXG_DETECTION=1 /opt/rocm/bin/rocminfo | grep -E "Marketing Name|  Name: +gfx"

echo "== 2. PyTorch (ROCm 7.2) + Transformers"
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install -q --upgrade pip
pip install -q torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
pip install -q "transformers>=5.10" accelerate pillow pyyaml compressed-tensors

echo "== 3. PyTorch 同梱の HSA ランタイムを WSL (DXG) 対応のシステム版に差し替え"
TORCH_LIB=$(pip show torch | awk '/^Location/{print $2}')/torch/lib
mkdir -p ~/torch-lib-backup
for f in libhsa-runtime64.so librocprofiler-register.so; do
  if [ ! -L "$TORCH_LIB/$f" ]; then
    mv "$TORCH_LIB/$f" ~/torch-lib-backup/
    ln -s "/opt/rocm/lib/$f" "$TORCH_LIB/$f"
  fi
done

echo "== 4. 動作確認"
export HSA_ENABLE_DXG_DETECTION=1 ROCPROFILER_REGISTER_ENABLED=0
python -c "import torch; print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

cat <<'EOF'

セットアップ完了。実行時は毎回次の環境変数を設定してください:
  source ~/dg-venv/bin/activate
  export HSA_ENABLE_DXG_DETECTION=1 ROCPROFILER_REGISTER_ENABLED=0
EOF
