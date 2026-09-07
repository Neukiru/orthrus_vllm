#!/usr/bin/env python
"""Is Orthrus's AR view bit-identical to stock Qwen3-1.7B?

The guide's Step 2 lossless test says: generate with plain Qwen3-1.7B and with
Orthrus+proposer, and require byte-identical output. That test is only valid if
Orthrus's autoregressive weights really are the frozen Qwen3 weights. The paper
says the base is frozen and only the *_diff projections are trained, but "frozen"
claims are worth checking against the actual tensors before we build a whole
validation strategy on them.

This runs on CPU and reads the safetensors directly, so it does not need the
model code, a GPU, or trust_remote_code.

Output tells us one of three things:
  - identical            -> guide's Step 2 test is valid as written
  - close but not equal  -> base was fine-tuned; Step 2 must compare against
                            Orthrus-AR-only, not stock Qwen3
  - different shapes     -> something is wrong with our understanding

Run:
  .venv-ref/bin/python bench/compare_weights.py
"""

from __future__ import annotations

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from pathlib import Path

ORTHRUS_ID = "chiennv/Orthrus-Qwen3-1.7B"
QWEN3_ID = "Qwen/Qwen3-1.7B"


def load_all(repo_id: str) -> dict[str, torch.Tensor]:
    root = Path(snapshot_download(repo_id, allow_patterns=["*.safetensors", "*.json"]))
    tensors: dict[str, torch.Tensor] = {}
    for f in sorted(root.glob("*.safetensors")):
        tensors.update(load_file(str(f)))
    return tensors


def main():
    print("loading checkpoints (CPU)...")
    orth = load_all(ORTHRUS_ID)
    qwen = load_all(QWEN3_ID)

    diff_keys = sorted(k for k in orth if "_diff" in k)
    shared_keys = sorted(k for k in orth if "_diff" not in k)

    print(f"\northrus tensors : {len(orth)}")
    print(f"  *_diff        : {len(diff_keys)}")
    print(f"  shared/AR     : {len(shared_keys)}")
    print(f"qwen3 tensors   : {len(qwen)}")

    n_diff_params = sum(orth[k].numel() for k in diff_keys)
    n_shared_params = sum(orth[k].numel() for k in shared_keys)
    total = n_diff_params + n_shared_params
    print(f"\nparam counts:")
    print(f"  shared (frozen base): {n_shared_params / 1e9:.3f}B")
    print(f"  diff (new)          : {n_diff_params / 1e9:.3f}B  "
          f"({100 * n_diff_params / total:.1f}% of total)   [guide says ~16%]")

    # Which *_diff parameter names exist? The vLLM model class must map these.
    suffixes = sorted({k.split("layers.")[-1].split(".", 1)[-1] for k in diff_keys
                       if "layers." in k})
    print("\n*_diff parameter names per layer:")
    for s in suffixes:
        print(f"  {s}")

    # ---- the actual comparison -------------------------------------------
    print("\ncomparing AR tensors against stock Qwen3-1.7B:")
    only_in_orth, only_in_qwen = [], []
    identical, differing, shape_mismatch = [], [], []

    for k in shared_keys:
        if k not in qwen:
            only_in_orth.append(k)
            continue
        a, b = orth[k], qwen[k]
        if a.shape != b.shape:
            shape_mismatch.append((k, tuple(a.shape), tuple(b.shape)))
        elif torch.equal(a, b):
            identical.append(k)
        else:
            d = (a.float() - b.float()).abs()
            differing.append((k, d.max().item(), d.mean().item()))

    for k in qwen:
        if k in orth:
            continue
        # Both configs set tie_word_embeddings=true, but the Qwen3 repo also
        # stores the tied lm_head explicitly while the Orthrus repo does not.
        # If it equals the embedding matrix it is not a real difference.
        if k == "lm_head.weight" and torch.equal(
            qwen[k], orth["model.embed_tokens.weight"]
        ):
            print("  note: qwen3 stores a redundant tied lm_head.weight; "
                  "it equals orthrus's embed_tokens (not a real difference)")
            continue
        only_in_qwen.append(k)

    print(f"  bit-identical   : {len(identical)}")
    print(f"  differing       : {len(differing)}")
    print(f"  shape mismatch  : {len(shape_mismatch)}")
    print(f"  only in orthrus : {len(only_in_orth)}")
    print(f"  only in qwen3   : {len(only_in_qwen)}")

    if only_in_orth:
        print(f"    e.g. {only_in_orth[:5]}")
    if only_in_qwen:
        print(f"    e.g. {only_in_qwen[:5]}")
    if shape_mismatch:
        for k, s1, s2 in shape_mismatch[:5]:
            print(f"    {k}: orthrus{s1} vs qwen{s2}")
    if differing:
        differing.sort(key=lambda t: -t[1])
        print("  largest deviations:")
        for k, mx, mn in differing[:10]:
            print(f"    {k:60} max={mx:.3e} mean={mn:.3e}")

    print("\nVERDICT:")
    if not differing and not shape_mismatch and not only_in_qwen:
        print("  Orthrus AR view is BIT-IDENTICAL to stock Qwen3-1.7B.")
        print("  -> Guide's Step 2 test (vs stock Qwen3-1.7B) is valid as written.")
    elif differing:
        worst = max(d[1] for d in differing)
        print(f"  AR weights DIFFER from stock Qwen3 (max abs dev {worst:.3e}).")
        print("  -> The base was NOT left frozen. The Step 2 lossless test must")
        print("     compare Orthrus-with-proposer against Orthrus-AR-only,")
        print("     NOT against stock Qwen3-1.7B.")
    else:
        print("  Key sets do not line up; inspect the lists above.")


if __name__ == "__main__":
    main()
