# Orthrus in vLLM — measurement and study notes

Working repository for adding native [Orthrus](https://arxiv.org/abs/2605.12825)
speculative decoding to [vLLM](https://github.com/vllm-project/vllm).

This repo holds the **measurement harness, the reproduced baselines, and the study
material**. The vLLM changes themselves live in a separate fork —
[Neukiru/vllm](https://github.com/Neukiru/vllm), branch `orthrus` — so that what
eventually goes upstream is a clean, reviewable diff against `vllm-project/vllm`.

Orthrus augments a **frozen** Qwen3 with a second set of attention projections in every
layer, giving the model a parallel "diffusion view" that drafts K−1 tokens in one forward
pass while the ordinary autoregressive view verifies them. Both views share one KV cache,
which is the method's real selling point: **zero extra cache memory**, against DFlash's or
EAGLE's separate drafter cache.

---

## What has been established

Measured on an RTX 5090 (sm_120, 32 GB), WSL2, vLLM `0.28.1rc1.dev453+ga1541f574`.

**The checkpoint reproduces the paper.** Greedy, GSM8K:

| model | n | TPF (per-question) | paper Table 1 |
|---|---|---|---|
| Orthrus-Qwen3-1.7B | 200 | **4.18** | 4.20 |
| Orthrus-Qwen3-4B | 200 | **4.98** (token-weighted) | 4.91 |

**Orthrus's AR view is bit-identical to stock Qwen3-1.7B** — 310/310 tensors, verified
tensor by tensor. The base really is frozen; the `*_diff` projections are 17.0% of the
model.

**Acceptance length is framework-invariant.** DFlash on Qwen3-4B, same 200 prompts:

| harness | acceptance length | tok/s |
|---|---|---|
| HuggingFace eager | 5.902 | 220.2 |
| vLLM, CUDA graphs on | 5.921 | 680.4 |

0.3% apart on acceptance, 3.1× apart on throughput. That separation is what makes
acceptance length a valid regression metric for the port, and tokens/sec not one.

**Head-to-head against DFlash is close.** At 4B in the same harness, Orthrus does 223.7
tok/s against DFlash's 220.2 — a tie, with Orthrus drafting far better (9.95 tokens/cycle
vs 5.90) but paying a full-depth draft pass where DFlash pays a fifth of one. Eager
systematically flatters Orthrus, so the case for upstreaming should be built on **memory,
not throughput**. See [`bench/README.md`](bench/README.md) for the full argument.

---

## Layout

```
bench/          measurement harness + recorded results
  README.md       ← the detailed findings, metric definitions, and traps
  bench_eager.py  Orthrus vs vanilla Qwen3, HF eager
  bench_dflash.py DFlash in the same harness (fair comparison)
  bench_vllm.py   vLLM-side AR vs DFlash
  compare_weights.py   is the base really frozen?
  measure_pass_cost.py draft vs target pass timing
  results_*.json  every measurement in this README

notebooks/
  orthrus_kv_walkthrough.ipynb   executable trace of a real paged KV cache and
                                 the real vLLM Triton kernel, with the
                                 `- num_rejected` bug reproduced on purpose
  build_notebook.py              generator (the notebook is generated, not hand-edited)

ref_peek/       unmodified copies of the Orthrus checkpoint's configs and modeling
                code (CC-BY-4.0, attributed in ref_peek/README.md) so the analysis
                elsewhere can cite exact line numbers. No weights.

Orthrus_Field_Manual.pdf   23-page study guide; source in study_guide.html
```

## Setup

```bash
uv venv .venv     --python 3.12   # vLLM
uv venv .venv-ref --python 3.12   # HF reference — deliberately separate, so a vLLM
                                  # dependency bump cannot move the numerical oracle
uv pip install --python .venv-ref torch "transformers>=5.8" accelerate datasets

git clone --filter=blob:none https://github.com/Neukiru/vllm.git vllm
cd vllm && git checkout orthrus && cd ..
VLLM_USE_PRECOMPILED=1 uv pip install --python .venv -e ./vllm
```

Two traps worth knowing before you start:

- **WSL2** — every vLLM engine start fails with `RuntimeError: UVA is not available`
  unless you set `VLLM_WSL2_ENABLE_PIN_MEMORY=1`. Not a bug; vLLM disables pinned memory
  on WSL by default and the v1 engine's `UvaBuffer` requires it.
- **Namespace shadowing** — running `python -c "import vllm"` from this directory
  resolves to the `vllm/` *directory* as an empty namespace package, silently shadowing
  the install (`vllm.__file__` is `None`). Run from elsewhere; script files are fine.

## Reproducing

```bash
.venv-ref/bin/python bench/bench_eager.py --max-new-tokens 256          # baselines + invariants
.venv-ref/bin/python bench/bench_eager.py --task gsm8k --num-prompts 200 \
    --max-new-tokens 512 --only orthrus-diff                            # vs paper Table 1
.venv-ref/bin/python bench/bench_eager.py --dtype fp32 \
    --only orthrus-ar,orthrus-diff                                      # strict lossless test
.venv-ref/bin/python bench/compare_weights.py                           # frozen base check
.venv-ref/bin/python bench/bench_dflash.py --task gsm8k --num-prompts 200 --size 4b
.venv/bin/python     bench/bench_vllm.py   --task gsm8k --num-prompts 200 --size 4b
```

Use **≥100 prompts** for anything compared against a paper: per-question TPF has a
standard deviation of 1.10, so n=25 carries a 95% CI of roughly ±0.4.

## Two things that will cost you a day if you don't know them

**The lossless test cannot pass in bf16.** Diffusion output must equal AR-only output at
temperature 0, and it does — in fp32, 12/12. In bf16 it is 7/12, because verification
evaluates 32 positions in one GEMM while the AR baseline does one per GEMM, and the
different reduction shapes flip the argmax at near-ties. Run the strict assertion in fp32;
in bf16 assert a high match *rate*.

**A broken drafter never produces wrong text — only slow text.** Correctness comes
entirely from the verification step, so every drafter bug is invisible in the output and
shows up *only* as a collapsed acceptance length. There is no crash and no failing
assertion. The notebook reproduces the most likely such bug (a dropped
`seq_lens - num_rejected`) and shows the drafter reading tokens the target threw away while
the output stays perfect.

## License

Benchmark and study material in this repository: Apache-2.0, matching vLLM.

`ref_peek/` contains unmodified third-party files redistributed under their own terms —
the Orthrus checkpoint's configs and modeling code (CC-BY-4.0) and Qwen3's config
(Apache-2.0). Attribution and SHA-256 checksums are in
[`ref_peek/README.md`](ref_peek/README.md). Model weights, the Orthrus paper, and the vLLM
source are not redistributed here.
