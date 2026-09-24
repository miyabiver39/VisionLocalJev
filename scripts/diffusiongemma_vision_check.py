"""
diffusiongemma_vision_check.py
------------------------------
DiffusionGemma に「カメラ画像 + プリセットの型付き設問 (Choice / Score / Noul)」を直接入力し、
型付きの判定 JSON を得られることを確認する検証スクリプト (Hugging Face Transformers 版)。

    python scripts/diffusiongemma_vision_check.py --image path/to/frame.jpg --preset fire_disaster
    python scripts/diffusiongemma_vision_check.py            # 画像省略時は合成テスト画像を使用

使用するモデルは下の「モデル選択」ブロックで、1 行だけコメントアウトを外して切り替えてください。
(--model で一時的に上書きすることもできます)

必要パッケージ: requirements-diffusiongemma.txt を参照
"""

import argparse
import json
import os
import re
import sys
import time

# ============================================================================
# モデル選択: 使いたいモデルの行だけコメントアウトを外す (必ず 1 行だけ有効にする)
# ============================================================================

# [0] 動作確認用の極小ランダム重みモデル (約 40MB / CPU で 2 分程度: 語彙 26 万 x 48 ステップのため)
#     コードパス (画像入力 → 拡散生成 → JSON 抽出) の確認専用。判定内容は無意味な文字列になる。
MODEL_ID = "trl-internal-testing/tiny-DiffusionGemmaForBlockDiffusion"

# [1] 公式 BF16 (重み 51.6GB) — H100 80GB / A100 80GB など。最も確実。
# MODEL_ID = "google/diffusiongemma-26B-A4B-it"

# [2] FP8 (重み 27.2GB) — H100 / L40S (48GB) など FP8 対応 GPU。要 compressed-tensors。
# MODEL_ID = "RedHatAI/diffusiongemma-26B-A4B-it-FP8-dynamic"

# [3] INT4 AWQ (重み 17.2GB) — L4 24GB / A100 40GB / RTX 4090 など。要 compressed-tensors。
#     16GB GPU (RX 9060 XT / Colab T4) では --lowvram --cpu-expert-layers 14 を付ける
#     (エキスパートを INT4 のまま保持する独自ローダ。RX 9060 XT で動作確認済み: docs/local_rocm_wsl.md)
# MODEL_ID = "cyankiwi/diffusiongemma-26B-A4B-it-AWQ-INT4"

# [4] NVFP4 (重み 18.1GB) — NVIDIA Blackwell 世代 (B200 / RTX 50 系) 専用。
# MODEL_ID = "RedHatAI/diffusiongemma-26B-A4B-it-NVFP4"

# ============================================================================

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMAGE_TOKEN_BUDGETS = (70, 140, 280, 560, 1120)


def load_preset(preset_id: str) -> dict:
    import yaml

    path = os.path.join(REPO_ROOT, "app", "presets", f"{preset_id}.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_prompt(preset: dict) -> str:
    """プリセット YAML の設問から、型付き JSON で答えさせる指示文を組み立てる。"""
    lines = [
        f"You are a safety monitoring system for: {preset.get('description', preset['id'])}.",
        "Look at the camera image and answer every question.",
        "Respond with ONLY a single JSON object (no markdown, no explanation) using exactly these keys:",
    ]
    for q in preset.get("questions", []):
        qtype = q.get("type", "choice")
        label = q.get("label", q["id"])
        if qtype == "choice":
            choices = ", ".join(f'"{c}"' for c in q.get("choices", []))
            lines.append(f'- "{q["id"]}" ({label}): one of [{choices}]')
        elif qtype == "score":
            lines.append(f'- "{q["id"]}" ({label}): a number from 0.0 to 1.0. {q.get("rubric", "")}')
        elif qtype == "noul":
            lines.append(f'- "{q["id"]}" ({label}): true or false. Hypothesis: {q.get("hypothesis", "")}')
    return "\n".join(lines)


def extract_json(text: str):
    """生成テキストから最初の JSON オブジェクトを取り出す (```json フェンス付きにも対応)。"""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidates = [fenced.group(1)] if fenced else []
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start:i + 1])
                    break
        start = text.find("{", start + 1)
    for c in candidates:
        try:
            return json.loads(c)
        except json.JSONDecodeError:
            continue
    return None


def validate(preset: dict, answer) -> list:
    """設問の型どおりに答えているかを検査し、問題点のリストを返す (空なら OK)。"""
    if not isinstance(answer, dict):
        return ["JSON オブジェクトを抽出できませんでした"]
    problems = []
    for q in preset.get("questions", []):
        qid, qtype = q["id"], q.get("type", "choice")
        if qid not in answer:
            problems.append(f"{qid}: 回答なし")
            continue
        v = answer[qid]
        if qtype == "choice" and v not in q.get("choices", []):
            problems.append(f"{qid}: 選択肢外の値 {v!r}")
        elif qtype == "score" and not (isinstance(v, (int, float)) and not isinstance(v, bool) and 0.0 <= v <= 1.0):
            problems.append(f"{qid}: 0〜1 の数値ではない {v!r}")
        elif qtype == "noul" and not isinstance(v, bool):
            problems.append(f"{qid}: true/false ではない {v!r}")
    return problems


def synthetic_image():
    """画像未指定時のテスト画像 (暗い室内に炎のような橙色の塊)。"""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (640, 360), (45, 40, 35))
    d = ImageDraw.Draw(img)
    d.rectangle((80, 120, 260, 330), fill=(70, 70, 75))
    d.ellipse((330, 120, 470, 330), fill=(255, 110, 0))
    d.ellipse((365, 180, 435, 320), fill=(255, 220, 60))
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL_ID, help="MODEL_ID を一時的に上書き")
    ap.add_argument("--image", help="入力画像のパス (省略時は合成テスト画像)")
    ap.add_argument("--preset", default="fire_disaster", help="app/presets の ID")
    ap.add_argument("--image-tokens", type=int, default=280, choices=IMAGE_TOKEN_BUDGETS,
                    help="画像トークン予算 (大きいほど精細・低速)")
    ap.add_argument("--max-new-tokens", type=int, default=256, help="生成長 (1 キャンバス = 256)")
    ap.add_argument("--max-denoising-steps", type=int, default=None, help="拡散ステップ上限 (既定: モデル設定 = 48)")
    ap.add_argument("--gpu-mem-gb", type=float, default=None,
                    help="GPU に載せる上限 (GB)。超えた分は CPU メモリへオフロード (VRAM 不足時)")
    ap.add_argument("--cpu", action="store_true", help="GPU があっても CPU で実行")
    ap.add_argument("--lowvram", action="store_true",
                    help="[3] INT4 AWQ 専用: エキスパートを INT4 のまま保持する独自ローダで 16GB GPU に載せる")
    ap.add_argument("--cpu-expert-layers", type=int, default=10,
                    help="--lowvram 時に CPU に置いて都度転送するエキスパート層の数 (VRAM 不足なら増やす)")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion

    preset = load_preset(args.preset)
    image = Image.open(args.image).convert("RGB") if args.image else synthetic_image()

    use_gpu = torch.cuda.is_available() and not args.cpu  # ROCm 版 PyTorch も torch.cuda で検出される
    device_name = torch.cuda.get_device_name(0) if use_gpu else "CPU"
    print(f"[model]  {args.model}")
    print(f"[device] {device_name} | torch {torch.__version__}")

    load_kwargs = {"dtype": "auto"}
    if use_gpu:
        load_kwargs["device_map"] = "auto"
        if args.gpu_mem_gb:
            load_kwargs["max_memory"] = {0: f"{args.gpu_mem_gb}GiB", "cpu": "64GiB"}
    else:
        load_kwargs["dtype"] = torch.float32 if "tiny" in args.model else torch.bfloat16

    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model)
    if args.lowvram:
        if not use_gpu:
            raise SystemExit("--lowvram は GPU が必要です")
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from diffusiongemma_lowvram import load_int4_lowvram
        model = load_int4_lowvram(args.model, cpu_expert_layers=args.cpu_expert_layers)
    else:
        model = DiffusionGemmaForBlockDiffusion.from_pretrained(args.model, **load_kwargs)
    if not use_gpu:
        model = model.to("cpu")
    model.eval()
    t_load = time.perf_counter() - t0

    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": build_prompt(preset)},
    ]}]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
        processor_kwargs={"max_soft_tokens": args.image_tokens},
    ).to(model.device)

    gen_kwargs = {"max_new_tokens": args.max_new_tokens}
    if args.max_denoising_steps:
        gen_kwargs["max_denoising_steps"] = args.max_denoising_steps

    if use_gpu:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t1 = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**inputs, **gen_kwargs)
    if use_gpu:
        torch.cuda.synchronize()
    t_gen = time.perf_counter() - t1

    sequences = output.sequences if hasattr(output, "sequences") else output
    new_tokens = sequences[0][inputs["input_ids"].shape[-1]:]
    text = processor.decode(new_tokens, skip_special_tokens=True)
    answer = extract_json(text)
    problems = validate(preset, answer)

    print(f"\n[preset] {args.preset} | image tokens {args.image_tokens} | "
          f"input image {'synthetic' if not args.image else args.image}")
    print(f"[time]   load {t_load:.1f}s | generate {t_gen:.2f}s")
    if use_gpu:
        print(f"[vram]   peak {torch.cuda.max_memory_allocated() / 1024**3:.1f} GiB")
    print("\n--- raw output ---")
    print(text.strip()[:2000])
    print("\n--- parsed decision ---")
    print(json.dumps(answer, ensure_ascii=False, indent=2) if answer is not None else "(JSON なし)")
    print("\n--- type check ---")
    print("OK: 全設問が型どおり" if not problems else "\n".join(f"NG: {p}" for p in problems))

    # 極小テストモデルは乱数重みなので型チェック NG が正常。コードパスが最後まで通れば成功とする。
    if "tiny" in args.model:
        print("\n(注) 極小テストモデルは判定能力がないため、上の NG は想定どおりです。")
        return 0
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
