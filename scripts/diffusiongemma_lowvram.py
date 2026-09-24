"""
diffusiongemma_lowvram.py
-------------------------
16GB クラスの GPU で DiffusionGemma の INT4 (compressed-tensors pack-quantized) チェックポイントを
画像入力つきで動かすための独自ローダ。

Transformers 標準のローダは MoE エキスパート (モデルの約 9 割) を BF16 に展開してしまうため
約 50GB 必要になる。ここではエキスパートを INT4 のまま保持し、計算する瞬間だけ BF16 に展開する。

  - エキスパート: INT4 のまま GPU に常駐。VRAM に収まらない末尾 N 層分は CPU (pinned) に置き、
    forward のたびに GPU へ転送する (--cpu-expert-layers)
  - アテンション等の INT4 重み: 読み込み時に BF16 へ展開 (小さいので GPU に常駐)
  - 画像エンコーダ (vision tower): CPU に置き、画像をエンコードする間だけ GPU へ移す

対応チェックポイント: cyankiwi/diffusiongemma-26B-A4B-it-AWQ-INT4 (int4 / group 32 / symmetric)
"""

import json
import os
import re
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

GROUP_SIZE = 32


def dequant_int4(packed: torch.Tensor, scale: torch.Tensor, out_dtype=torch.bfloat16) -> torch.Tensor:
    """compressed-tensors pack-quantized (int4, 対称, group=32) を展開する。

    packed: [..., out, in/8] int32 (1 要素に 4bit x 8 個, 下位ビットから順に格納, 値は +8 オフセット)
    scale : [..., out, in/32] fp16
    """
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=packed.device)
    q = (packed.unsqueeze(-1) >> shifts) & 0xF                         # [..., out, in/8, 8]
    q = q.reshape(*packed.shape[:-1], packed.shape[-1] * 8)            # [..., out, in]
    w = (q.to(torch.float16) - 8.0).unflatten(-1, (-1, GROUP_SIZE)) * scale.unsqueeze(-1)
    return w.flatten(-2).to(out_dtype)


class Int4Experts(nn.Module):
    """DiffusionGemmaTextExperts の置き換え。重みは INT4 のまま持ち、使うエキスパートだけ展開する。"""

    def __init__(self, num_experts, hidden, intermediate, act_fn, device, chunk=16):
        super().__init__()
        self.num_experts = num_experts
        self.act_fn = act_fn
        self.chunk = chunk
        pin = device.type == "cpu" and torch.cuda.is_available()

        def empty(*shape, dtype):
            t = torch.empty(*shape, dtype=dtype, device=device)
            return t.pin_memory() if pin else t

        # gate と up を出力方向に連結 ([gate; up]) — 元実装の gate_up_proj と同じ並び
        self.register_buffer("gu_packed", empty(num_experts, 2 * intermediate, hidden // 8, dtype=torch.int32), persistent=False)
        self.register_buffer("gu_scale", empty(num_experts, 2 * intermediate, hidden // GROUP_SIZE, dtype=torch.float16), persistent=False)
        self.register_buffer("dn_packed", empty(num_experts, hidden, intermediate // 8, dtype=torch.int32), persistent=False)
        self.register_buffer("dn_scale", empty(num_experts, hidden, intermediate // GROUP_SIZE, dtype=torch.float16), persistent=False)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        dev = hidden_states.device
        # CPU 常駐の層はここで GPU へ転送 (GPU 常駐なら no-op)
        gu_p, gu_s, dn_p, dn_s = (t.to(dev, non_blocking=True) for t in
                                  (self.gu_packed, self.gu_scale, self.dn_packed, self.dn_scale))
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero().flatten().tolist()

        for start in range(0, len(hit), self.chunk):
            ids = hit[start:start + self.chunk]
            idx = torch.tensor(ids, device=dev)
            gu_w = dequant_int4(gu_p[idx], gu_s[idx], hidden_states.dtype)
            dn_w = dequant_int4(dn_p[idx], dn_s[idx], hidden_states.dtype)
            for j, e in enumerate(ids):
                top_k_pos, token_idx = torch.where(expert_mask[e])
                x = hidden_states[token_idx]
                gate, up = F.linear(x, gu_w[j]).chunk(2, dim=-1)
                h = F.linear(self.act_fn(gate) * up, dn_w[j])
                h = h * top_k_weights[token_idx, top_k_pos, None]
                final.index_add_(0, token_idx, h.to(final.dtype))
            del gu_w, dn_w
        return final


def _map_name(key: str) -> str:
    """チェックポイントのキー名 → Transformers モデルのパラメータ名。

    チェックポイントはデコーダ側の名前で保存されているが、Transformers ではエンコーダ側が本体で
    デコーダはそれを共有する (tie) 構造になっている。layer_scalar はバッファなので両側に別々に存在する。
    """
    if key.startswith("model.decoder.layers.") and not key.endswith("layer_scalar"):
        return key.replace("model.decoder.layers.", "model.encoder.language_model.layers.", 1)
    if key == "model.decoder.embed_tokens.weight":
        return "model.encoder.language_model.embed_tokens.weight"
    if key == "model.decoder.norm.weight":
        return "model.encoder.language_model.norm.weight"
    return key


def _set_tensor(model, name, value, device):
    from accelerate.utils import set_module_tensor_to_device

    # dtype を明示しないと accelerate が meta パラメータ側の dtype (FP32) に合わせて倍のメモリを使う
    dtype = torch.bfloat16 if value.is_floating_point() else None
    set_module_tensor_to_device(model, name, device, value=value, dtype=dtype)


def _share(dst_module, src_module):
    """src_module の全パラメータを dst_module の同じパスに共有させる (エンコーダ→デコーダ)。"""
    for path, param in src_module.named_parameters():
        parent_path, _, leaf = path.rpartition(".")
        parent = dst_module.get_submodule(parent_path) if parent_path else dst_module
        parent._parameters[leaf] = param


def load_int4_lowvram(model_id: str, cpu_expert_layers: int = 10, chunk: int = 16, log=print):
    from accelerate import init_empty_weights
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoConfig, DiffusionGemmaForBlockDiffusion

    gpu = torch.device("cuda")
    cpu = torch.device("cpu")
    snap = snapshot_download(model_id)
    cfg = AutoConfig.from_pretrained(model_id)
    qcfg = cfg.quantization_config
    qcfg = qcfg if isinstance(qcfg, dict) else qcfg.to_dict()
    w = qcfg["config_groups"]["group_0"]["weights"]
    if not (w["num_bits"] == 4 and w["group_size"] == GROUP_SIZE and w["symmetric"] and qcfg.get("format") == "pack-quantized"):
        raise ValueError(f"unsupported quantization config: {w} / format={qcfg.get('format')}")
    cfg.quantization_config = None  # 標準の compressed-tensors 展開処理を通さない
    tc = cfg.text_config

    with init_empty_weights():
        model = DiffusionGemmaForBlockDiffusion(cfg)

    enc_layers = model.model.encoder.language_model.layers
    dec_layers = model.model.decoder.layers
    n_layers = len(enc_layers)
    cpu_expert_layers = max(0, min(cpu_expert_layers, n_layers))
    cpu_set = set(range(n_layers - cpu_expert_layers, n_layers))
    log(f"[lowvram] experts: GPU {n_layers - len(cpu_set)} layers / CPU(streamed) {len(cpu_set)} layers")

    act_fn = enc_layers[0].experts.act_fn
    experts = []
    for i in range(n_layers):
        ex = Int4Experts(tc.num_experts, tc.hidden_size, tc.moe_intermediate_size, act_fn,
                         cpu if i in cpu_set else gpu, chunk=chunk)
        enc_layers[i].experts = ex
        dec_layers[i].experts = ex
        experts.append(ex)
    inter = tc.moe_intermediate_size

    weight_map = json.load(open(os.path.join(snap, "model.safetensors.index.json")))["weight_map"]
    by_shard = defaultdict(list)
    for k, shard in weight_map.items():
        by_shard[shard].append(k)

    expert_re = re.compile(r"model\.decoder\.layers\.(\d+)\.experts\.(\d+)\.(gate|up|down)_proj\.weight_(packed|scale|shape)$")
    pending_linear = defaultdict(dict)  # 量子化された通常 Linear (attention 等): base -> {packed, scale, shape}
    vision_prefixes = ("model.encoder.vision_tower.", "model.encoder.embed_vision.")

    for n_shard, (shard, keys) in enumerate(sorted(by_shard.items()), 1):
        log(f"[lowvram] loading shard {n_shard}/{len(by_shard)} ({len(keys)} tensors)")
        with safe_open(os.path.join(snap, shard), "pt", device="cpu") as f:
            for k in keys:
                m = expert_re.match(k)
                if m:
                    layer, e, proj, kind = int(m[1]), int(m[2]), m[3], m[4]
                    if kind == "shape":
                        continue
                    t = f.get_tensor(k)
                    ex = experts[layer]
                    if proj == "down":
                        buf = ex.dn_packed if kind == "packed" else ex.dn_scale
                        buf[e].copy_(t)
                    else:
                        buf = ex.gu_packed if kind == "packed" else ex.gu_scale
                        rows = slice(0, inter) if proj == "gate" else slice(inter, 2 * inter)
                        buf[e, rows].copy_(t)
                    continue
                if k.endswith(("weight_packed", "weight_scale", "weight_shape")):
                    base, kind = k.rsplit(".weight_", 1)
                    pending_linear[base][kind] = f.get_tensor(k)
                    p = pending_linear[base]
                    if {"packed", "scale", "shape"} <= p.keys():
                        out_f, in_f = p["shape"].tolist()
                        wt = dequant_int4(p["packed"].to(gpu), p["scale"].to(gpu))[:, :in_f]
                        _set_tensor(model, _map_name(base + ".weight"), wt, gpu)
                        del pending_linear[base]
                    continue
                target = _map_name(k)
                device = cpu if target.startswith(vision_prefixes) else gpu
                _set_tensor(model, target, f.get_tensor(k), device)

    if pending_linear:
        raise RuntimeError(f"incomplete quantized tensors: {list(pending_linear)[:5]}")

    # エンコーダの重みをデコーダ・lm_head と共有
    for i in range(n_layers):
        _share(dec_layers[i], enc_layers[i])
    model.model.decoder.norm.weight = model.model.encoder.language_model.norm.weight
    model.model.decoder.embed_tokens.weight = model.model.encoder.language_model.embed_tokens.weight
    model.lm_head.weight = model.model.encoder.language_model.embed_tokens.weight

    missing = [n for n, p in model.named_parameters() if p.device.type == "meta"]
    if missing:
        raise RuntimeError(f"{len(missing)} parameters were not loaded, e.g. {missing[:5]}")

    # 残りのバッファ (rotary 等) を GPU へ。画像エンコーダは CPU に置き、使う間だけ GPU へ移す
    for name, buf in list(model.named_buffers()):
        if name.startswith(vision_prefixes) or ".experts." in name:
            continue
        parent_path, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_path)
        parent._buffers[leaf] = buf.to(gpu)

    for prefix in ("vision_tower", "embed_vision"):
        mod = getattr(model.model.encoder, prefix)
        mod.to(cpu)
        # フックが値を返すと引数/出力が置き換わるため、必ず None を返す
        mod.register_forward_pre_hook(lambda m, args: (m.to(gpu), None)[1])
        mod.register_forward_hook(lambda m, args, out: (m.to(cpu), None)[1])

    model.eval()
    torch.cuda.empty_cache()
    log(f"[lowvram] loaded. GPU allocated {torch.cuda.memory_allocated() / 2**30:.1f} GiB")
    return model
