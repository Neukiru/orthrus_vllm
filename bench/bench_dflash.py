#!/usr/bin/env python
"""DFlash in HF eager, using the same harness as bench_eager.py.

Why this file exists: comparing Orthrus (measured in HF) against DFlash
(measured in vLLM) would have been defensible only for the count-based metrics
(TPF / acceptance length), and not for wall-clock at all. Both checkpoints ship
a standalone HF reference loop -- Orthrus's `generate(use_diffusion_mode=True)`
and DFlash's `DFlashDraftModel.spec_generate` -- so we can run both in the same
framework, on the same GPU, with the same prompts, encoding and trimming. Then
tokens/sec is comparable too, and that is the number that actually settles it.

The two reference loops are structurally near-identical: same [anchor, MASK x n]
draft block, same greedy prefix-match acceptance rule, same cache crop. The only
difference is who does the drafting, which is exactly the variable we want to
isolate:

  Orthrus  drafts with the full target (36 layers). Cycle = 2 full passes.
  DFlash   drafts with a separate 5-layer network, plus hidden states pulled
           from 5 target layers. Cycle = 1 full pass + 5/36 of a pass.

So DFlash can accept fewer tokens per cycle and still be faster. Three numbers
are reported:

  tokens/cycle          draft QUALITY (comparable to Orthrus's tok/cycle and to
                        the DFlash paper's "acceptance length")
  tokens per unit work  EFFICIENCY, pricing a draft pass at draft/target layers
  tokens/sec            GROUND TRUTH, same framework so it is directly comparable

Run:
  .venv-ref/bin/python bench/bench_dflash.py --task gsm8k --num-prompts 200
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from bench_eager import DTYPES, encode, trim
from prompts import load_task_prompts

TARGETS = {"4b": "Qwen/Qwen3-4B", "8b": "Qwen/Qwen3-8B"}
DRAFTS = {"4b": "z-lab/Qwen3-4B-DFlash-b16", "8b": "z-lab/Qwen3-8B-DFlash-b16"}


class PassCounter:
    """Counts draft-network and target-network forward passes separately.

    They cost very different amounts (5 layers vs 36), which is the whole point
    of the comparison, so a single combined count would hide the effect.
    """

    def __init__(self, target, draft):
        self.target, self.draft = target, draft
        self.n_target = 0
        self.n_draft = 0

    def __enter__(self):
        self._t_orig = self.target.forward
        self._d_orig = self.draft.forward

        def t_fwd(*a, **kw):
            self.n_target += 1
            return self._t_orig(*a, **kw)

        def d_fwd(*a, **kw):
            self.n_draft += 1
            return self._d_orig(*a, **kw)

        self.target.forward = t_fwd
        self.draft.forward = d_fwd
        return self

    def __exit__(self, *exc):
        self.target.forward = self._t_orig
        self.draft.forward = self._d_orig
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="gsm8k", choices=["mixed", "gsm8k", "humaneval"])
    ap.add_argument("--num-prompts", type=int, default=200)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--size", default="4b", choices=["4b", "8b"])
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    ap.add_argument("--dtype", default="bf16", choices=list(DTYPES))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    target_id, draft_id = TARGETS[args.size], DRAFTS[args.size]
    print(f"target {target_id}\ndraft  {draft_id}", flush=True)

    tok = AutoTokenizer.from_pretrained(target_id)
    target = AutoModelForCausalLM.from_pretrained(
        target_id, dtype=DTYPES[args.dtype], attn_implementation=args.attn
    ).to("cuda").eval()
    draft = AutoModel.from_pretrained(
        draft_id, dtype=DTYPES[args.dtype], trust_remote_code=True
    ).to("cuda").eval()

    n_draft_layers = draft.config.num_hidden_layers
    n_target_layers = target.config.num_hidden_layers
    draft_cost = n_draft_layers / n_target_layers
    mask_id = draft.config.dflash_config["mask_token_id"]
    eos_id = target.config.eos_token_id
    print(f"draft {n_draft_layers} layers vs target {n_target_layers} layers "
          f"-> a draft pass costs ~{draft_cost:.3f} of a target pass")
    print(f"draft block_size = {draft.config.block_size}  (Orthrus uses 32)\n")

    prompts = load_task_prompts(args.task, args.num_prompts)

    def one(ids, max_new):
        with PassCounter(target, draft) as ctr:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            # The live remote code is dflash.py (per the config's auto_map),
            # whose spec_generate reads mask_token_id from the config rather
            # than taking it as an argument.
            out = draft.spec_generate(
                target=target,
                input_ids=ids,
                max_new_tokens=max_new,
                stop_token_ids=[eos_id],
                temperature=0.0,
            )
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
        gen = trim(out[0, ids.shape[1]:], eos_id, mask_id)
        return gen, dt, ctr.n_target, ctr.n_draft

    # Warm-up: first call pays CUDA context and autotune costs.
    one(encode(tok, "Say hello."), 16)

    rows = []
    for cat, prompt in prompts:
        ids = encode(tok, prompt)
        gen, dt, n_t, n_d = one(ids, args.max_new_tokens)
        n = len(gen)
        # n_target includes the single prefill pass; decode passes exclude it.
        decode_target = max(n_t - 1, 1)
        cycles = max(n_d, 1)
        effective = decode_target + cycles * draft_cost
        rows.append({
            "category": cat, "gen_tokens": n, "seconds": dt,
            "tok_per_s": n / dt if dt else 0.0,
            "target_passes": decode_target, "draft_passes": cycles,
            "tokens_per_cycle": n / cycles,
            "effective_passes": effective,
            "tokens_per_unit_work": n / effective,
        })
        if len(rows) <= 10 or len(rows) % 25 == 0:
            print(f"  [{cat}] {n:4d} tok in {dt:6.2f}s = {n / dt:7.2f} tok/s   "
                  f"tok/cycle {n / cycles:5.2f}   tok/work {n / effective:5.2f}",
                  flush=True)

    tot_tok = sum(r["gen_tokens"] for r in rows)
    tot_s = sum(r["seconds"] for r in rows)
    tot_cycles = sum(r["draft_passes"] for r in rows)
    tot_eff = sum(r["effective_passes"] for r in rows)
    tot_naive = sum(r["target_passes"] + r["draft_passes"] for r in rows)

    summary = {
        "target": target_id, "draft": draft_id,
        "draft_layers": n_draft_layers, "target_layers": n_target_layers,
        "draft_cost_ratio": draft_cost,
        "block_size": draft.config.block_size,
        "total_gen_tokens": tot_tok, "total_seconds": tot_s,
        "aggregate_tok_per_s": tot_tok / tot_s,
        "tokens_per_cycle": tot_tok / tot_cycles,
        "tokens_per_unit_work": tot_tok / tot_eff,
        # If a draft pass were priced as a full pass -- the convention the
        # Orthrus paper uses for its own two full-depth passes. Included so the
        # two papers' TPF columns can be compared on identical terms.
        "tpf_if_draft_were_full_pass": tot_tok / tot_naive,
        "mean_tokens_per_cycle_per_question": statistics.mean(
            r["tokens_per_cycle"] for r in rows),
        "rows": rows,
    }

    print(f"\n{'=' * 70}\nDFLASH  {args.size}  {args.task}  n={len(rows)}\n{'=' * 70}")
    print(f"  tokens/sec                 {summary['aggregate_tok_per_s']:8.2f}")
    print(f"  tokens per cycle           {summary['tokens_per_cycle']:8.3f}"
          f"   (draft quality)")
    print(f"  tokens per unit work       {summary['tokens_per_unit_work']:8.3f}"
          f"   (efficiency; draft priced at {draft_cost:.3f})")
    print(f"  TPF if draft were full     {summary['tpf_if_draft_were_full_pass']:8.3f}"
          f"   (Orthrus's accounting applied to DFlash)")

    out = args.out or f"bench/results_dflash_{args.size}_{args.task}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps({"config": vars(args), "summary": summary}, indent=2))
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
