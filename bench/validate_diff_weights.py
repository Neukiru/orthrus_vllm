#!/usr/bin/env python
"""Step 1b: did the diffusion projections load into the right places?

validate_model_class.py proves the *_diff tensors are consumed and that the AR
path still reproduces stock Qwen3. Neither of those would catch the classic
fusion bug: q/k/v landing in the wrong slices of the fused qkv_proj_diff. A
whole-tensor checksum would not catch it either, because a permutation of the
slices preserves the total.

So this compares each slice of the fused weight against the checkpoint tensor it
should have come from. Swapping any two would change the per-slice statistics.

Run:
  .venv/bin/python bench/validate_diff_weights.py
"""

from __future__ import annotations

import os
import platform
from pathlib import Path

if "microsoft" in platform.uname().release.lower():
    os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")

# Run the engine in this process so apply_model() can take a local closure.
# The alternative, VLLM_ALLOW_INSECURE_SERIALIZATION=1, would ship the function
# to a worker by pickle; there is no reason to turn that on for a local probe.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

ORTHRUS = "chiennv/Orthrus-Qwen3-1.7B"


def probe(model):
    """Runs inside the worker. Returns small statistics, never whole tensors."""
    from vllm.model_executor.models.orthrus_qwen3 import OrthrusQwen3Attention

    def stats(t):
        t = t.detach().float()
        return {
            "shape": tuple(t.shape),
            "sum": t.sum().item(),
            "absmax": t.abs().max().item(),
            "head": t.flatten()[:6].tolist(),
        }

    out = {}
    for name, mod in model.named_modules():
        if not isinstance(mod, OrthrusQwen3Attention):
            continue
        q, kv = mod.q_size, mod.kv_size
        w = mod.qkv_proj_diff.weight
        out[f"{name}.q_proj_diff"] = stats(w[:q])
        out[f"{name}.k_proj_diff"] = stats(w[q : q + kv])
        out[f"{name}.v_proj_diff"] = stats(w[q + kv : q + 2 * kv])
        out[f"{name}.o_proj_diff"] = stats(mod.o_proj_diff.weight)
        out[f"{name}.q_norm_diff"] = stats(mod.q_norm_diff.weight)
        out[f"{name}.k_norm_diff"] = stats(mod.k_norm_diff.weight)
        # Control: the AR fused QKV, loaded by the same machinery.
        wa = mod.qkv_proj.weight
        out[f"{name}.q_proj"] = stats(wa[:q])
        out[f"{name}.k_proj"] = stats(wa[q : q + kv])
        out[f"{name}.v_proj"] = stats(wa[q + kv : q + 2 * kv])
    return out


def main():
    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from vllm import LLM

    root = Path(snapshot_download(ORTHRUS, allow_patterns=["*.safetensors"]))
    ckpt = {}
    for f in sorted(root.glob("*.safetensors")):
        ckpt.update(load_file(str(f)))
    print(f"checkpoint tensors: {len(ckpt)}")

    llm = LLM(
        model=ORTHRUS,
        max_model_len=1024,
        max_num_seqs=1,
        gpu_memory_utilization=0.40,
        enforce_eager=True,
        seed=0,
    )
    results = llm.apply_model(probe)
    probed = results[0]
    print(f"probed parameters : {len(probed)}")

    def stats_of(t):
        t = t.detach().float()
        return {
            "shape": tuple(t.shape),
            "sum": t.sum().item(),
            "absmax": t.abs().max().item(),
            "head": t.flatten()[:6].tolist(),
        }

    n_ok = n_bad = 0
    failures = []
    for name, got in sorted(probed.items()):
        key = f"{name}.weight"
        if key not in ckpt:
            failures.append((name, "not in checkpoint", None, None))
            n_bad += 1
            continue
        exp = stats_of(ckpt[key])
        same = (
            got["shape"] == exp["shape"]
            and abs(got["sum"] - exp["sum"]) < 1e-3
            and abs(got["absmax"] - exp["absmax"]) < 1e-6
            and all(abs(a - b) < 1e-6 for a, b in zip(got["head"], exp["head"]))
        )
        if same:
            n_ok += 1
        else:
            n_bad += 1
            failures.append((name, "mismatch", got, exp))

    print()
    print("=" * 70)
    print(f"matched {n_ok} / {n_ok + n_bad} parameter slices")
    print("=" * 70)
    for name, why, got, exp in failures[:6]:
        print(f"  {name}: {why}")
        if got:
            print(f"     vllm  shape={got['shape']} sum={got['sum']:.4f} "
                  f"head={[round(x,4) for x in got['head'][:3]]}")
            print(f"     ckpt  shape={exp['shape']} sum={exp['sum']:.4f} "
                  f"head={[round(x,4) for x in exp['head'][:3]]}")

    # A swap would be invisible to a whole-tensor checksum; show it is not
    # invisible here, by checking the q slice against the k checkpoint tensor.
    l0 = "model.layers.0.self_attn"
    q_got = probed[f"{l0}.q_proj_diff"]
    k_exp = stats_of(ckpt[f"{l0}.k_proj_diff.weight"])
    print()
    print("sanity: would a q/k swap be detected?")
    print(f"  q slice shape {q_got['shape']} vs k checkpoint shape {k_exp['shape']}"
          f"  -> {'different, swap caught by shape' if q_got['shape'] != k_exp['shape'] else 'same shape, relies on sum'}")

    if n_bad:
        raise SystemExit(f"FAIL: {n_bad} parameter slices do not match the checkpoint")
    print("\nPASS - every diffusion projection loaded into the correct slice.")


if __name__ == "__main__":
    main()
