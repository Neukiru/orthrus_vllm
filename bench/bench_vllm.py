#!/usr/bin/env python
"""vLLM-side benchmark: AR baseline vs DFlash speculative decoding.

Purpose: find out what Orthrus actually has to beat. DFlash is already merged in
vLLM, so we can measure it today on this exact GPU with the exact prompts we used
for Orthrus, instead of trusting cross-paper arithmetic.

What is and is not comparable across the two harnesses:

  COMPARABLE     acceptance length (= tokens per cycle) and TPF. These are pure
                 counts of "how many tokens per model run", independent of GPU
                 and framework. So DFlash-measured-in-vLLM can be compared
                 against Orthrus-measured-in-HF.

  NOT COMPARABLE tokens/sec. vLLM removes the Python overhead that dominates HF
                 eager decoding, so its absolute throughput is far higher for
                 reasons that have nothing to do with the drafting algorithm.
                 Only compare tok/s within this script (vLLM vs vLLM).

The cost asymmetry that makes this comparison interesting:

  DFlash  drafts with a separate 5-layer network against a 36-layer target,
          so a draft costs ~5/36 = 0.14 of a full forward.
          Cost per cycle ~= 1.14 full passes.
  Orthrus drafts with the full target model.
          Cost per cycle = 2.0 full passes.

So DFlash can afford a lower acceptance length and still win on throughput.
The quantity to compare is acceptance_length / cost_per_cycle.

Run (from the repo root; the script dir goes on sys.path, so the sibling
'vllm/' source directory does not shadow the installed package):
  .venv/bin/python bench/bench_vllm.py --task gsm8k --num-prompts 200
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

from prompts import load_task_prompts

# On WSL2, vLLM disables pinned memory by default (vllm/platforms/cuda.py), and
# the v1 engine then dies at startup with "RuntimeError: UVA is not available".
# Kernels >= 4.19.121 do support it; vLLM just makes it opt-in. Verified working
# on this box (pinned H2D 28.0 GB/s vs pageable 13.7 GB/s, so it is genuinely
# pinned rather than silently falling back). Set it before vllm is imported.
if "microsoft" in platform.uname().release.lower():
    if os.environ.get("VLLM_WSL2_ENABLE_PIN_MEMORY") is None:
        os.environ["VLLM_WSL2_ENABLE_PIN_MEMORY"] = "1"
        print("WSL2 detected: setting VLLM_WSL2_ENABLE_PIN_MEMORY=1 "
              "(without it the engine fails with 'UVA is not available')")

QWEN3_4B = "Qwen/Qwen3-4B"
DFLASH_4B = "z-lab/Qwen3-4B-DFlash-b16"
QWEN3_8B = "Qwen/Qwen3-8B"
DFLASH_8B = "z-lab/Qwen3-8B-DFlash-b16"

# Layer counts, used to price a draft pass relative to a target forward.
DRAFT_LAYERS = {DFLASH_4B: 5, DFLASH_8B: 5}
TARGET_LAYERS = {QWEN3_4B: 36, QWEN3_8B: 36}


def spec_metric(metrics, name: str) -> float:
    by_name = {m.name: m for m in metrics}
    m = by_name.get(name)
    if m is None:
        avail = sorted(n for n in by_name if "spec_decode" in n)
        raise AssertionError(
            f"missing metric {name!r} (disable_log_stats must be False). "
            f"available spec_decode metrics: {avail or ['<none>']}"
        )
    return float(m.value)


def run(label, target, draft, prompts, args):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"\n{'=' * 72}\n{label}   target={target}   draft={draft}\n{'=' * 72}",
          flush=True)

    kwargs = dict(
        model=target,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        # max_num_seqs=1 forces batch-1 decoding, matching how we measured
        # Orthrus in HF. Speculative decoding is at its best here; raise it to
        # see the advantage compress as passes become compute-bound.
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_mem_util,
        enforce_eager=args.enforce_eager,
        # Required, or get_metrics() returns no spec_decode counters.
        disable_log_stats=False,
        seed=0,
    )
    if draft is not None:
        kwargs["speculative_config"] = {
            "method": "dflash",
            "model": draft,
            "num_speculative_tokens": args.num_spec_tokens,
            "max_model_len": args.max_model_len,
        }

    llm = LLM(**kwargs)
    tok = AutoTokenizer.from_pretrained(target)

    # Build prompt strings exactly as the HF harness does, so the two runs see
    # identical text.
    texts = [
        tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
        for _, p in prompts
    ]
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    # Warm-up so CUDA graph capture and autotuning are not billed to the run.
    llm.generate(texts[:1], SamplingParams(temperature=0.0, max_tokens=16))

    t0 = time.perf_counter()
    outs = llm.generate(texts, sp)
    dt = time.perf_counter() - t0

    gen_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    result = {
        "target": target,
        "draft": draft,
        "gen_tokens": gen_tokens,
        "seconds": dt,
        "tok_per_s": gen_tokens / dt,
        "per_prompt_tokens": [len(o.outputs[0].token_ids) for o in outs],
    }

    if draft is not None:
        metrics = llm.get_metrics()
        n_drafts = spec_metric(metrics, "vllm:spec_decode_num_drafts")
        n_accepted = spec_metric(metrics, "vllm:spec_decode_num_accepted_tokens")
        # vLLM's own definition (tests/v1/e2e/spec_decode/utils.py): the +1 is
        # the bonus token the target samples after the last accepted draft.
        # This equals "tokens per cycle" in Orthrus terms.
        acc_len = 1 + (n_accepted / n_drafts) if n_drafts else 1.0
        cost = 1.0 + DRAFT_LAYERS[draft] / TARGET_LAYERS[target]
        result.update({
            "num_drafts": n_drafts,
            "num_accepted_tokens": n_accepted,
            "acceptance_length": acc_len,
            "cost_per_cycle_full_passes": cost,
            "tokens_per_unit_work": acc_len / cost,
        })
        print(f"  acceptance length      {acc_len:.3f}  (tokens per cycle)")
        print(f"  cost per cycle         {cost:.3f} full passes")
        print(f"  tokens per unit work   {acc_len / cost:.3f}   <-- compare this")

    print(f"  {gen_tokens} tok in {dt:.2f}s = {result['tok_per_s']:.1f} tok/s")

    del llm
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="gsm8k", choices=["mixed", "gsm8k", "humaneval"])
    ap.add_argument("--num-prompts", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--size", default="4b", choices=["4b", "8b"])
    ap.add_argument("--num-spec-tokens", type=int, default=16,
                    help="DFlash b16 checkpoints use block_size 16; vLLM's own "
                         "e2e test passes 16")
    ap.add_argument("--max-num-seqs", type=int, default=1)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--only", default=None, help="comma-separated: ar,dflash")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    target, draft = (QWEN3_4B, DFLASH_4B) if args.size == "4b" else (QWEN3_8B, DFLASH_8B)
    prompts = load_task_prompts(args.task, args.num_prompts)

    configs = [("ar", target, None), ("dflash", target, draft)]
    if args.only:
        keep = set(args.only.split(","))
        configs = [c for c in configs if c[0] in keep]

    results = {}
    for label, t, d in configs:
        results[label] = run(label, t, d, prompts, args)

    print(f"\n{'=' * 72}\nSUMMARY  ({args.size}, {args.task}, "
          f"batch={args.max_num_seqs})\n{'=' * 72}")
    for k, v in results.items():
        extra = (f"   acc_len {v['acceptance_length']:.2f}"
                 f"   tok/unit-work {v['tokens_per_unit_work']:.2f}"
                 if "acceptance_length" in v else "")
        print(f"  {k:8} {v['tok_per_s']:8.1f} tok/s{extra}")
    if "ar" in results and "dflash" in results:
        sp = results["dflash"]["tok_per_s"] / results["ar"]["tok_per_s"]
        print(f"\n  DFlash speedup over vLLM AR baseline: {sp:.2f}x")
        results["speedup"] = sp

    out = args.out or f"bench/results_vllm_{args.size}_{args.task}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"config": vars(args), "results": results},
                                    indent=2))
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
