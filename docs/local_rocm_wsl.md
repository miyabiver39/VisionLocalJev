# ローカル AMD GPU (WSL2 + ROCm) で DiffusionGemma を画像入力つきで動かす

RX 9060 XT (16GB, RDNA4 / gfx1200) で、DiffusionGemma の INT4 版に **カメラ画像 + 型付き設問** を入力し、
型どおりの JSON 判定が返ることを確認した手順と結果です。

## 結果 (2026-09-25)

| 入力画像 | 出力 (fire_disaster プリセット) | 型チェック | 生成時間 |
|---|---|---|---|
| 暗い室内に橙色の炎状の塊 (合成) | `{"hazard_type": "open_flame", "urgency_score": 0.8, "evacuate": true}` | OK | 221.5 秒 (初回) |
| 明るいオフィス・炎なし (合成) | `{"hazard_type": "none", "urgency_score": 0.0, "evacuate": false}` | OK | 56.5 秒 |

- モデル: `cyankiwi/diffusiongemma-26B-A4B-it-AWQ-INT4` (重み 17.2GB)
- VRAM: 読み込み後 10.9GiB / ピーク 13.0GiB (エキスパート 30 層中 14 層を CPU から都度転送)
- 読み込み: 約 50 秒
- 初回の生成が遅いのは GPU カーネルの初回コンパイル・調整を含むためと考えられます (未検証)
- 画像は合成図形のみでの確認です。実際のカメラ映像での判定精度は未評価です

## 環境

| 項目 | バージョン |
|---|---|
| GPU / ドライバ | AMD Radeon RX 9060 XT 16GB / Adrenalin 26.8.1 |
| OS | Windows 11 + WSL2 Ubuntu 24.04 |
| ROCm | 7.2.4 (ランタイムのみ) + [librocdxg](https://github.com/ROCm/librocdxg) 1.2.2 |
| PyTorch | 2.14.0+rocm7.2 |
| Transformers | 5.17.0 |

## セットアップ

```bash
wsl -d Ubuntu-24.04 -- bash scripts/setup_wsl_rocm.sh
```

スクリプトが行うこと、および手作業でハマった点:

1. **ROCm 7.2.4 ランタイム + librocdxg**: WSL では `/dev/dxg` 経由で GPU を使うため ROCDXG が必要 (Adrenalin 26.2.2 以降)。
2. **PyTorch 同梱の `libhsa-runtime64.so` をシステム版に差し替え**: 同梱版は WSL (DXG) を認識しない。
3. **`ROCPROFILER_REGISTER_ENABLED=0`**: PyTorch 同梱の rocprofiler-sdk が WSL に存在しない
   `/sys/class/kfd` を前提にしており、`Found 0 rocprofiler agents and 2 HSA agents` で abort するため無効化する。
4. **`PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` は使わない**: WSL では GPU メモリ確保が `invalid argument` で失敗する。

## 実行

```bash
source ~/dg-venv/bin/activate
export HSA_ENABLE_DXG_DETECTION=1 ROCPROFILER_REGISTER_ENABLED=0
python scripts/diffusiongemma_vision_check.py \
  --model cyankiwi/diffusiongemma-26B-A4B-it-AWQ-INT4 --lowvram --cpu-expert-layers 14 \
  --preset fire_disaster --image path/to/frame.jpg
```

## なぜ独自ローダ (`--lowvram`) が必要か

- Transformers の DiffusionGemma 実装は MoE エキスパート (約 22.8B / 全体の約 9 割) を **3 次元テンソル**で持つ。
- そのため標準ローダは INT4 チェックポイントを読み込む際にエキスパートを **BF16 に展開**し、約 50GB 必要になる
  (bitsandbytes による 4bit 化も nn.Linear しか対象にしないため効かない)。
- [`scripts/diffusiongemma_lowvram.py`](../scripts/diffusiongemma_lowvram.py) はエキスパートを INT4 のまま GPU に置き
  (約 0.43GB/層)、計算する瞬間に使うエキスパートだけを BF16 に展開する。VRAM に入らない層は CPU の pinned メモリに置き、
  forward のたびに GPU へ転送する。INT4 の展開結果は compressed-tensors の実装と一致することを確認済み
  (最大誤差 2.4e-4 = FP16 の丸め誤差)。

`--cpu-expert-layers` の目安 (16GB GPU):

| 値 | 読み込み後 VRAM | 結果 |
|---|---|---|
| 10 | 約 12.6GiB (推定) | 生成中に VRAM 不足 (hipBLAS の初期化に失敗) |
| 14 | 10.9GiB | 成功 (ピーク 13.0GiB) |

## 今後の改善余地

- 生成速度: 拡散の 1 ステップごとに全層のエキスパートを展開・転送している。展開済み重みのキャッシュ、融合カーネル、
  ステップ数 (`--max-denoising-steps`) の調整で短縮できる見込み。
- 本番相当の検証は NVIDIA GPU (H100 等) で公式 BF16 版 (`--lowvram` なし) を使うのが確実。
