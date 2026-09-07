#!/usr/bin/env python
"""Baseline throughput: vanilla Qwen3-1.7B vs Orthrus-Qwen3-1.7B, HF eager PyTorch.

This establishes the numbers the vLLM port has to beat, and — more importantly —
checks the two invariants the whole port depends on before we write any vLLM code.

Three configurations are measured, not two:

  qwen3-ar      Qwen/Qwen3-1.7B, ordinary greedy HF generate.
                The "vanilla" number.

  orthrus-ar    chiennv/Orthrus-Qwen3-1.7B with the diffusion loop OFF, i.e. the
                frozen AR view only. Same weights, same code path as orthrus-diff.
                This is the *controlled* baseline: comparing orthrus-diff against
                this isolates the contribution of the diffusion loop from any
                difference in weights or model code.

  orthrus-diff  chiennv/Orthrus-Qwen3-1.7B running the full propose/verify cycle.

Two invariants are asserted from the results:

  (I1) qwen3-ar output == orthrus-ar output.
       Orthrus is supposed to be a *frozen* Qwen3 plus extra attention
       projections. If the AR views disagree, the base was not actually frozen,
       and the guide's Step-2 lossless test ("compare against stock Qwen3-1.7B")
       is invalid as written.

  (I2) orthrus-ar output == orthrus-diff output.
       At temperature 0 the reference consensus rule is exact greedy prefix
       matching, so speculative decoding must be lossless. This is the
       reference-level version of the guide's Step 2 lossless test. If it fails
       here, the premise is broken before vLLM is involved.

Run:
  .venv-ref/bin/python bench/bench_eager.py --max-new-tokens 256
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationMixin

from prompts import PROMPTS, load_task_prompts  # noqa: F401

ORTHRUS_ID = "chiennv/Orthrus-Qwen3-1.7B"
QWEN3_ID = "Qwen/Qwen3-1.7B"


DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def load(model_id: str, attn: str, dtype: torch.dtype):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        trust_remote_code=True,
        attn_implementation=attn,
    )
    model.to("cuda").eval()
    return tok, model


def encode(tok, prompt: str) -> torch.Tensor:
    msgs = [{"role": "user", "content": prompt}]
    # enable_thinking=False keeps generations short and comparable; with thinking
    # on, Qwen3 emits a long <think> block and the run time is dominated by it.
    text = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return tok(text, return_tensors="pt").input_ids.to("cuda")


class ForwardCounter:
    """Counts diffusion vs AR forward passes by wrapping model.forward.

    nn.Module.__call__ looks up self.forward at call time, so patching the
    instance attribute intercepts every call the reference generate() makes
    without touching the checkpoint's code.
    """

    def __init__(self, model):
        self.model = model
        self.diff = 0
        self.ar = 0

    def __enter__(self):
        self._orig = self.model.forward

        def fwd(*args, **kwargs):
            if kwargs.get("is_diffusion_pass", False):
                self.diff += 1
            else:
                self.ar += 1
            return self._orig(*args, **kwargs)

        self.model.forward = fwd
        return self

    def __exit__(self, *exc):
        self.model.forward = self._orig
        return False


def trim(gen_ids: torch.Tensor, eos_id: int, mask_id: int) -> list[int]:
    """Strip everything from EOS onward, and any trailing <mask> padding.

    The reference generate() returns a fixed-width buffer pre-filled with
    mask_token_id, so a run that ends exactly at max_length can carry trailing
    mask tokens that were never really generated. Counting those would inflate
    the tokens/sec of the diffusion path.
    """
    out = gen_ids.tolist()
    if eos_id in out:
        out = out[: out.index(eos_id)]
    while out and out[-1] == mask_id:
        out.pop()
    return out


def bench_ar(model, tok, ids, max_new, eos_id, mask_id):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = GenerationMixin.generate(
        model,
        input_ids=ids,
        max_new_tokens=max_new,
        do_sample=False,
        # The checkpoint's generation_config sets use_cache=false. Without this
        # override every step would recompute the whole prefix and the baseline
        # would be meaningless (and ~50x too slow).
        use_cache=True,
        eos_token_id=eos_id,
        pad_token_id=eos_id,
    )
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    gen = trim(out[0, ids.shape[1]:], eos_id, mask_id)
    return gen, dt, {}


def bench_diff(model, tok, ids, max_new, eos_id, mask_id):
    with ForwardCounter(model) as ctr:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(
            input_ids=ids,
            max_new_tokens=max_new,
            temperature=0.0,
            use_diffusion_mode=True,
            eos_token_id=eos_id,
        )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    gen = trim(out[0, ids.shape[1]:], eos_id, mask_id)
    # ctr.ar includes the one prefill pass; the rest are verification passes.
    return gen, dt, {"diffusion_passes": ctr.diff, "ar_passes": ctr.ar}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"],
                    help="HF attention implementation; identical for all three "
                         "configs so the comparison stays fair")
    ap.add_argument("--out", default="bench/results_eager.json")
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    ap.add_argument("--only", default=None,
                    help="comma-separated subset of config names to run")
    ap.add_argument("--num-prompts", type=int, default=None)
    ap.add_argument("--task", default="mixed", choices=["mixed", "gsm8k", "humaneval"])
    ap.add_argument("--orthrus-model", default=ORTHRUS_ID)
    ap.add_argument("--qwen-model", default=QWEN3_ID)
    args = ap.parse_args()

    results: dict[str, dict] = {}
    texts: dict[str, list[str]] = {}

    prompts = load_task_prompts(args.task, args.num_prompts)

    configs = [
        ("qwen3-ar", args.qwen_model, bench_ar),
        ("orthrus-ar", args.orthrus_model, bench_ar),
        ("orthrus-diff", args.orthrus_model, bench_diff),
    ]
    if args.only:
        keep = set(args.only.split(","))
        configs = [c for c in configs if c[0] in keep]

    for name, model_id, fn in configs:
        print(f"\n{'=' * 70}\n{name}  ({model_id})  dtype={args.dtype}\n{'=' * 70}", flush=True)
        tok, model = load(model_id, args.attn, DTYPES[args.dtype])
        eos_id = model.config.eos_token_id
        mask_id = getattr(model.config, "mask_token_id", -1)

        # Warm-up: first call pays CUDA context + autotune costs.
        warm = encode(tok, "Say hello.")
        fn(model, tok, warm, 16, eos_id, mask_id)

        rows = []
        texts[name] = []
        for cat, prompt in prompts:
            ids = encode(tok, prompt)
            gen, dt, extra = fn(model, tok, ids, args.max_new_tokens, eos_id, mask_id)
            n = len(gen)
            row = {
                "category": cat,
                "prompt": prompt,
                "prompt_tokens": ids.shape[1],
                "gen_tokens": n,
                "seconds": dt,
                "tok_per_s": n / dt if dt > 0 else 0.0,
                **extra,
            }
            if "diffusion_passes" in extra and extra["diffusion_passes"]:
                cycles = extra["diffusion_passes"]
                row["tokens_per_cycle"] = n / cycles
                # Paper Eq. 9: TPF = generated tokens / total forward passes,
                # and a cycle is exactly two passes (diffusion + verification).
                # Prefill is excluded; ar_passes counts it, hence the -1.
                row["forward_passes"] = cycles + (extra["ar_passes"] - 1)
                row["tpf"] = n / row["forward_passes"]
            rows.append(row)
            texts[name].append(tok.decode(gen))
            tpc = (f"  tok/cycle {row['tokens_per_cycle']:5.2f}  TPF {row['tpf']:5.2f}"
                   if "tokens_per_cycle" in row else "")
            print(f"  [{cat:9}] {n:4d} tok in {dt:6.2f}s = {row['tok_per_s']:6.2f} tok/s{tpc}",
                  flush=True)

        total_tok = sum(r["gen_tokens"] for r in rows)
        total_s = sum(r["seconds"] for r in rows)
        summary = {
            "model_id": model_id,
            "rows": rows,
            "total_gen_tokens": total_tok,
            "total_seconds": total_s,
            "aggregate_tok_per_s": total_tok / total_s,
            "mean_tok_per_s": statistics.mean(r["tok_per_s"] for r in rows),
        }
        cycles = sum(r.get("diffusion_passes", 0) for r in rows)
        if cycles:
            passes = sum(r["forward_passes"] for r in rows)
            summary["total_cycles"] = cycles
            summary["total_forward_passes"] = passes
            summary["mean_tokens_per_cycle"] = total_tok / cycles
            summary["tpf"] = total_tok / passes
            # Per-category TPF, since the paper only ever reports math/code.
            cats = sorted({r["category"] for r in rows})
            summary["tpf_by_category"] = {
                c: (sum(r["gen_tokens"] for r in rows if r["category"] == c)
                    / sum(r["forward_passes"] for r in rows if r["category"] == c))
                for c in cats
            }
        results[name] = summary
        print(f"  ---> aggregate {summary['aggregate_tok_per_s']:.2f} tok/s")

        del model, tok
        gc.collect()
        torch.cuda.empty_cache()

    # ---- invariants -------------------------------------------------------
    print(f"\n{'=' * 70}\nINVARIANTS\n{'=' * 70}")
    checks = {}
    for label, a, b in [
        ("I1  qwen3-ar == orthrus-ar   (AR view is the frozen base)", "qwen3-ar", "orthrus-ar"),
        ("I2  orthrus-ar == orthrus-diff (spec decoding is lossless)", "orthrus-ar", "orthrus-diff"),
    ]:
        if a not in texts or b not in texts:
            continue
        same = [x == y for x, y in zip(texts[a], texts[b])]
        checks[label] = {"matched": sum(same), "total": len(same)}
        status = "PASS" if all(same) else "FAIL"
        print(f"  [{status}] {label}: {sum(same)}/{len(same)} prompts identical")
        if not all(same):
            for i, ok in enumerate(same):
                if ok:
                    continue
                ta, tb = texts[a][i], texts[b][i]
                # Locate the first differing character so we show the actual
                # divergence rather than an identical-looking prefix.
                j = next((k for k in range(min(len(ta), len(tb))) if ta[k] != tb[k]),
                         min(len(ta), len(tb)))
                lo = max(0, j - 60)
                print(f"        prompt #{i} [{prompts[i][0]}] diverges at char {j} "
                      f"(len {len(ta)} vs {len(tb)})")
                print(f"          common : ...{ta[lo:j]!r}")
                print(f"          {a:11}: {ta[j:j + 60]!r}")
                print(f"          {b:11}: {tb[j:j + 60]!r}")

    # ---- speedups ---------------------------------------------------------
    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    for name in results:
        s = results[name]
        tpc = (f"   tok/cycle {s['mean_tokens_per_cycle']:.2f}   TPF {s['tpf']:.2f}"
               if "mean_tokens_per_cycle" in s else "")
        print(f"  {name:14} {s['aggregate_tok_per_s']:7.2f} tok/s{tpc}")
        if "tpf_by_category" in s:
            for c, v in s["tpf_by_category"].items():
                print(f"      TPF[{c}] = {v:.2f}")
    if "orthrus-diff" in results:
        diff = results["orthrus-diff"]["aggregate_tok_per_s"]
        if "qwen3-ar" in results:
            print(f"\n  speedup vs vanilla Qwen3-1.7B  : "
                  f"{diff / results['qwen3-ar']['aggregate_tok_per_s']:.2f}x")
        if "orthrus-ar" in results:
            print(f"  speedup vs Orthrus AR-only     : "
                  f"{diff / results['orthrus-ar']['aggregate_tok_per_s']:.2f}x"
                  f"   <-- the honest number")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(
        {"config": vars(args), "results": results, "invariants": checks, "texts": texts},
        indent=2))
    print(f"\n  wrote {out_path}")


if __name__ == "__main__":
    main()
