#!/usr/bin/env python
"""Directly time a DFlash draft pass against a target pass.

Why this script exists: pricing a draft pass by layer ratio (5/36 = 0.139) is an
assumption, and trying to recover the two costs by regressing wall-clock on pass
counts is degenerate -- spec_generate makes exactly one draft call and one target
call per cycle, so the two predictors are perfectly collinear and least squares
splits the total evenly regardless of the truth.

So we time each forward directly, with a cuda synchronize around it. That makes
the total run slower than the real benchmark, but the per-pass numbers are honest,
and per-pass cost is all we want here.

Run:
  .venv-ref/bin/python bench/measure_pass_cost.py --num-prompts 10
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from bench_eager import encode
from bench_dflash import DRAFTS, TARGETS
from prompts import load_task_prompts


class TimedCounter:
    def __init__(self, target, draft):
        self.target, self.draft = target, draft
        self.t_times: list[float] = []
        self.d_times: list[float] = []

    def __enter__(self):
        self._t_orig, self._d_orig = self.target.forward, self.draft.forward

        def wrap(orig, sink):
            def fn(*a, **kw):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = orig(*a, **kw)
                torch.cuda.synchronize()
                sink.append(time.perf_counter() - t0)
                return out
            return fn

        self.target.forward = wrap(self._t_orig, self.t_times)
        self.draft.forward = wrap(self._d_orig, self.d_times)
        return self

    def __exit__(self, *exc):
        self.target.forward = self._t_orig
        self.draft.forward = self._d_orig
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="4b", choices=["4b", "8b"])
    ap.add_argument("--num-prompts", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    target_id, draft_id = TARGETS[args.size], DRAFTS[args.size]
    tok = AutoTokenizer.from_pretrained(target_id)
    target = AutoModelForCausalLM.from_pretrained(
        target_id, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda").eval()
    draft = AutoModel.from_pretrained(
        draft_id, dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda").eval()

    eos_id = target.config.eos_token_id
    prompts = load_task_prompts("gsm8k", args.num_prompts)

    # Warm-up.
    draft.spec_generate(target=target, input_ids=encode(tok, "Say hello."),
                        max_new_tokens=16, stop_token_ids=[eos_id], temperature=0.0)

    ctr = TimedCounter(target, draft)
    with ctr:
        for _, p in prompts:
            draft.spec_generate(target=target, input_ids=encode(tok, p),
                                max_new_tokens=args.max_new_tokens,
                                stop_token_ids=[eos_id], temperature=0.0)

    # Drop the first target call of each generation: that is prefill over the
    # whole prompt, not a verification pass, and it is much more expensive.
    t_decode = ctr.t_times[1:]
    d = ctr.d_times

    tmed, dmed = statistics.median(t_decode), statistics.median(d)
    print(f"\n{'=' * 64}\nDFlash per-pass cost ({args.size}, n={len(prompts)} prompts)\n{'=' * 64}")
    print(f"  target verify pass : median {tmed * 1000:6.2f} ms   (n={len(t_decode)})")
    print(f"  draft pass         : median {dmed * 1000:6.2f} ms   (n={len(d)})")
    print(f"  measured draft cost: {dmed / tmed:.3f} of a target pass")
    print(f"  layer-ratio guess  : {draft.config.num_hidden_layers / target.config.num_hidden_layers:.3f}")
    print(f"\n  => cost per DFlash cycle = {1 + dmed / tmed:.3f} target passes")
    print(f"     (Orthrus's cycle costs exactly 2.000 by construction)")


if __name__ == "__main__":
    main()
