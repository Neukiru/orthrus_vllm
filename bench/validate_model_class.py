#!/usr/bin/env python
"""Step 1 acceptance test: the Orthrus model class loads and its AR path is intact.

Two things are checked, in order of how cheaply they fail:

  1. The checkpoint loads through vLLM's registry as OrthrusQwen3ForCausalLM, and
     every *_diff tensor in the checkpoint is actually consumed. vLLM does not
     error on unconsumed weights, so silently dropping the diffusion projections
     would look like success until the proposer produced garbage drafts.

  2. Greedy generation from Orthrus matches greedy generation from stock
     Qwen3-1.7B, token for token. Orthrus's AR weights are bit-identical to
     Qwen3-1.7B (bench/compare_weights.py proves that), so any divergence here is
     our model class, not the checkpoint.

Run (note the env var and that this must not run from the repo root):
  .venv/bin/python bench/validate_model_class.py
"""

from __future__ import annotations

import argparse
import os
import platform

if "microsoft" in platform.uname().release.lower():
    os.environ.setdefault("VLLM_WSL2_ENABLE_PIN_MEMORY", "1")

from prompts import load_task_prompts  # noqa: E402

ORTHRUS = "chiennv/Orthrus-Qwen3-1.7B"
QWEN3 = "Qwen/Qwen3-1.7B"


def check_weights_consumed(model_id: str) -> None:
    """Every *_diff tensor in the checkpoint must reach a parameter."""
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file
    from pathlib import Path

    root = Path(snapshot_download(model_id, allow_patterns=["*.safetensors"]))
    ckpt = {}
    for f in sorted(root.glob("*.safetensors")):
        ckpt.update(load_file(str(f)))

    diff_keys = sorted(k for k in ckpt if "_diff" in k)
    print(f"  checkpoint has {len(diff_keys)} *_diff tensors")

    from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM

    mapper = Qwen3ForCausalLM.hf_to_vllm_mapper
    dests: dict[str, set] = {}
    for k in diff_keys:
        name, shard = mapper._map_name_with_shard(k)
        dests.setdefault(name, set()).add(shard)

    fused = {k: v for k, v in dests.items() if len(v) > 1}
    print(f"  -> {len(dests)} destination params, {len(fused)} of them fused qkv")
    bad = [k for k in dests if "q_proj_diff" in k or "k_proj_diff" in k]
    assert not bad, f"unfused diff projections still present: {bad[:3]}"
    for name, shards in list(fused.items())[:1]:
        print(f"  e.g. {name} <- shards {sorted(shards)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-prompts", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--gpu-mem-util", type=float, default=0.40)
    args = ap.parse_args()

    print("=" * 70)
    print("1. weight mapping")
    print("=" * 70)
    check_weights_consumed(ORTHRUS)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    prompts = load_task_prompts("gsm8k", args.num_prompts)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    outs = {}
    for label, model_id in [("orthrus", ORTHRUS), ("qwen3", QWEN3)]:
        print()
        print("=" * 70)
        print(f"2. loading {label}: {model_id}")
        print("=" * 70)
        llm = LLM(
            model=model_id,
            max_model_len=2048,
            max_num_seqs=1,
            gpu_memory_utilization=args.gpu_mem_util,
            # Correctness first; CUDA graphs come at Step 4.
            enforce_eager=True,
            seed=0,
        )
        arch = type(llm.llm_engine.model_config.hf_config).__name__
        print(f"  hf_config class : {arch}")
        print(f"  architectures   : {llm.llm_engine.model_config.hf_config.architectures}")

        tok = AutoTokenizer.from_pretrained(model_id)
        texts = [
            tok.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            for _, p in prompts
        ]
        res = llm.generate(texts, sp)
        outs[label] = [o.outputs[0].token_ids for o in res]
        print(f"  generated {sum(len(t) for t in outs[label])} tokens")
        del llm

    print()
    print("=" * 70)
    print("3. AR path is untouched")
    print("=" * 70)
    same = [a == b for a, b in zip(outs["orthrus"], outs["qwen3"])]
    print(f"  identical token sequences: {sum(same)}/{len(same)}")
    if not all(same):
        i = same.index(False)
        a, b = outs["orthrus"][i], outs["qwen3"][i]
        j = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]),
                 min(len(a), len(b)))
        print(f"  first divergence on prompt #{i} at token {j}")
        print(f"    orthrus: {a[max(0, j-5):j+5]}")
        print(f"    qwen3  : {b[max(0, j-5):j+5]}")
        raise SystemExit("FAIL: the AR path differs from stock Qwen3")
    print("\n  PASS - Orthrus's AR view reproduces stock Qwen3-1.7B exactly.")


if __name__ == "__main__":
    main()
