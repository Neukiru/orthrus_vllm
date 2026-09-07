# Orthrus / vLLM — baseline benchmarks

Numbers the vLLM port has to reproduce, and how to regenerate them.

**Hardware / software.** RTX 5090 (Blackwell, `sm_120`, 32 GB) on WSL2, kernel
`6.6.87.2-microsoft-standard-WSL2`. All HF numbers: transformers 5.16.1,
torch 2.14.0+cu130, `attn_implementation="sdpa"`, bf16, **batch 1**, greedy
(`T=0`). vLLM: `0.28.1rc1.dev453+ga1541f574`, torch 2.13.0+cu130.

Two virtualenvs on purpose:

| venv | contents | used by |
|---|---|---|
| `.venv` | vLLM, editable from `vllm/` | `bench_vllm.py` |
| `.venv-ref` | torch + transformers only | `bench_eager.py`, `bench_dflash.py`, `compare_weights.py`, `measure_pass_cost.py` |

The reference implementations are our **numerical oracle** for the whole port.
Keeping them out of the vLLM venv means a vLLM dependency bump can never silently
change the thing we validate against.

---

## 1. Metric definitions — read this first

Three numbers get confused constantly, and two of them differ by exactly 2×.

**TPF** (Orthrus paper, Eq. 9 / Table 1) = generated tokens ÷ **all** forward
passes. Plain autoregressive decoding is TPF 1.0 by definition. An Orthrus cycle
is *two* forward passes (diffusion propose + AR verify), so:

```
TPF = tokens_per_cycle / 2
```

**Acceptance length** (Orthrus paper Fig. 4; DFlash paper Table 1; vLLM's
`vllm:spec_decode_*` metrics) = accepted tokens per **verification** pass, which
equals **tokens per cycle** ≈ 2 × TPF.

> Compare against Table 1 with TPF. Compare against Fig. 4 and vLLM's metric with
> tokens/cycle. Mixing them is a silent 2× error.

**TPF is hardware-independent by construction** — it is a pure count, no time in
the formula. GPU choice cannot change it (only floating-point non-determinism can
nudge it, by a few percent at most). Wall-clock tok/s *is* hardware- and
framework-dependent. **Only TPF is portable enough to be a regression test.**

### Averaging convention matters ~8%

`total_tokens / total_passes` (token-weighted) is not the same as the mean of
per-question TPF (equal-weighted). Long answers have systematically lower TPF and
supply more tokens, so they dominate the token-weighted mean and drag it down.
The papers' numbers line up with the **equal-weighted** convention. We report both.

---

## 2. Does the checkpoint reproduce the paper?

Yes, closely. GSM8K, 512 max new tokens, thinking disabled.

| model | n | tok/s | tok/cycle | TPF (token-wt) | TPF (per-question) | paper Table 1 |
|---|---|---|---|---|---|---|
| Orthrus-Qwen3-1.7B | 200 | 250.7 | 7.75 | 3.88 | **4.18** | 4.20 |
| Orthrus-Qwen3-4B | 200 | 223.7 | 9.95 | **4.98** | 5.50 | 4.91 |

Compare tok/s only within a single task. The 1.7B does 250.7 tok/s on GSM8K but
149.5 on the mixed set — that is the prompt mix, not the model. The 4B is
correctly **slower** than the 1.7B on the same task.

HumanEval, n=20: TPF 3.14 (token-wt) vs the paper's 2.75.

**Sample size matters more than it looks.** Per-question TPF at 1.7B has
mean 4.18, **sd 1.10**, range 2.13–7.82. A 25-question run has a 95% bootstrap CI
of **[3.52, 4.31]** — about ±0.4. Our first 25-question run gave 3.78, which was
never statistically distinguishable from 4.20. **Use ≥100 questions** for any
claim about matching the paper.

### Do not quote "7.8×"

That is one cell of Table 1 (Qwen3-**8B**, Pseudo2code). The paper's average for
Qwen3-1.7B at `T=0` is 4.25×, and speedup scales with base-model size
(1.7B 4.25× → 4B 5.20× → 8B 5.36×). On a generic prompt mix including open-ended
prose — which the paper never benchmarks — TPF falls to 2.54.

---

## 3. Which baseline is honest?

Orthrus's AR weights are **bit-identical** to stock Qwen3-1.7B — verified tensor
by tensor, 310/310 (`compare_weights.py`). The one apparent gap, Qwen3 storing a
redundant tied `lm_head.weight`, equals Orthrus's `embed_tokens`.

Yet on the mixed set the AR-only path runs ~9% *faster* than stock Qwen3
(70.6 vs 64.7 tok/s) purely because the checkpoint's hand-written
`OrthrusModel.forward` carries less Python overhead. At 1.7B batch-1 decode, host
overhead dominates.

> Always divide by **Orthrus-AR-only**, not stock Qwen3, or the diffusion loop
> gets credited with that 9%. Mixed set: 2.31× vs vanilla, **2.12× honest**.

---

## 4. Orthrus vs DFlash (Qwen3-4B, GSM8K, n=200)

Both run in **the same harness, same framework, same GPU, same prompts** — both
checkpoints ship a standalone HF reference loop (`generate(use_diffusion_mode=True)`
and `DFlashDraftModel.spec_generate`), so wall-clock is directly comparable, not
just the counts.

| | tok/s | tokens/cycle | drafter |
|---|---|---|---|
| Orthrus-Qwen3-4B | **223.7** | **9.95** | full 36-layer target |
| DFlash Qwen3-4B (`z-lab/Qwen3-4B-DFlash-b16`) | 220.2 | 5.90 | separate 5-layer net |

**Wall-clock is a dead heat** (Orthrus +1.6%), while Orthrus's draft quality is
69% higher. Orthrus needs that edge, because its draft pass is the whole model
where DFlash's is a fifth of one.

Caveats, none of them small:

- **This is HF eager, which flatters Orthrus.** Per-pass Python overhead is large
  and roughly constant, so DFlash's structurally cheaper 5-layer draft does not
  translate into a proportionally cheaper wall-clock pass. In vLLM that overhead
  largely disappears and DFlash should pull ahead. **The vLLM-side comparison is
  still outstanding and is the one that matters for the RFC.**
- **Block sizes differ and cannot be equalised**: DFlash `b16` drafts 16 tokens,
  Orthrus 32. That is baked into each trained checkpoint, so this compares the
  methods as shipped.
- **4B only.** There is no DFlash drafter for Qwen3-1.7B; the families overlap
  only at 4B and 8B. At 8B the published numbers favour DFlash more
  (DFlash GSM8K acceptance 6.54 vs Orthrus 10.6 with 2× the draft cost), so
  **do not extrapolate this 4B result to 8B** — that is exactly the mistake we
  made before measuring.
### Measured pass costs (4B, `measure_pass_cost.py`)

```
target verify pass : 21.53 ms
draft pass         :  3.20 ms
measured draft cost: 0.149 of a target pass   (layer ratio 5/36 = 0.139)
```

The layer-ratio assumption is therefore sound. (Note: trying to recover this by
regressing wall-clock on pass counts is **degenerate** — `spec_generate` makes
exactly one draft and one target call per cycle, so the predictors are perfectly
collinear and least squares splits the total evenly. Time the passes directly.)

### Why DFlash only tied despite a 7x cheaper draft

```
DFlash  forwards per cycle  21.53 + 3.20 = 24.7 ms
DFlash  actual cycle time                  26.8 ms   <- 2.1 ms unaccounted
Orthrus forwards per cycle  2 x 22.2     = 44.5 ms
Orthrus actual cycle time                  44.5 ms   <- fully accounted
```

DFlash burns ~2 ms/cycle *outside* the forwards (`extract_context_feature`, two
separate KV caches, extra crop). Orthrus's loop has almost none. In HF eager that
cancels DFlash's cheaper draft.

### Per-layer overhead dominates HF eager — and it distorts everything

| | weight streaming @1.79 TB/s | remainder | fits |
|---|---|---|---|
| Orthrus-1.7B, 15.5 ms/pass | 4.15 GB → 2.3 ms | 13.2 ms | 28 layers x 0.47 ms |
| Orthrus-4B, 22.2 ms/pass | 9.40 GB → 5.3 ms | 16.9 ms | 36 layers x 0.47 ms |

**~76–85% of an HF eager pass is Python/launch overhead, not GPU work**, at
~0.47 ms per layer. Consequences:

- Orthrus-4B is only 11% slower than Orthrus-1.7B (250.7 → 223.7 tok/s on GSM8K)
  despite 2.3x the weights. **Do not read that as the 4B being cheap.**
- A 5-layer drafter is not 7x cheaper in wall-clock, because it still pays
  per-layer overhead.
- **HF eager systematically flatters both larger models and full-depth drafters,
  i.e. it flatters Orthrus.** Expect the ranking to move in vLLM.

---

## 4b. vLLM measurements (Qwen3-4B, GSM8K, n=200, batch 1, CUDA graphs on)

```
AR baseline    155.7 tok/s
DFlash         680.4 tok/s    speedup 4.37x    acceptance length 5.921
```

### Acceptance length is framework-invariant — confirmed empirically

| DFlash-4B, same 200 prompts | acceptance length | tok/s |
|---|---|---|
| HF eager | 5.902 | 220.2 |
| vLLM | 5.921 | 680.4 |

**0.3% apart on acceptance, 3.1x apart on throughput.** Different kernels,
different attention backend, CUDA graphs on vs off. This is the empirical
justification for using TPF / acceptance length as the port's regression metric
and for comparing across harnesses at all. tok/s measures the implementation;
acceptance length measures the algorithm.

### Fixed per-pass overhead survives in vLLM

AR baseline is 6.42 ms/token, but streaming 8.0 GB at 1.79 TB/s is only 4.5 ms.
So ~1.9 ms/pass is fixed overhead even with CUDA graphs — and **DFlash pays it on
its draft pass too**, which cancels most of the advantage its 5-layer drafter
should give.

| | per pass | cycle | tokens | predicted tok/s |
|---|---|---|---|---|
| DFlash | 6.37 + 2.51 | 8.88 ms | 5.92 | 667 (**measured 680**) |
| Orthrus | 7.15 x 2 | 14.3 ms | 9.95 | **~696 (projection)** |

The model reproduces DFlash to within 2%, so the Orthrus projection is worth
something: **roughly a tie at 4B, maybe +2% Orthrus.** An earlier pure-bandwidth
argument predicting a 23% DFlash win ignored this fixed term and was wrong.

Nothing here has measured Orthrus *in vLLM* — that requires the port. This is the
number the port has to hit.

### Per-position acceptance (DFlash, K=16)

```
0.87 0.73 0.62 0.54 0.45 0.36 0.31 0.26 0.22 0.19 0.16 0.13 0.11 0.09 0.07 0.05
```

Steep decay; position 16 contributes 5%. **Orthrus drafts 32.** Positions 17-32
will contribute very little while still costing full compute in the verify pass.
Relevant when tuning `num_speculative_tokens` — and a reason to expose it.

---

## 5. The lossless test cannot pass in bf16

Orthrus's consensus rule at `T=0` is exact greedy prefix matching, so
diffusion output must equal AR-only output. Measured:

| dtype | prompts identical |
|---|---|
| fp32 | **12/12** |
| bf16 | 7/12 |

Not a bug. Verification evaluates K=32 positions in one GEMM while the AR
baseline does one position per GEMM; different reduction shapes round differently
and flip the argmax at near-ties.

> The guide's Step 2 states the criterion as "byte-identical" in bf16. Taken
> literally **that test fails on a correct implementation.** Run the strict
> equality assertion in **fp32**; in bf16 assert a high match *rate* plus
> divergence only at near-ties.

This matters more than it sounds, because of an asymmetry:

> **A broken drafter never produces wrong text — only slow text.** Correctness
> comes entirely from verification, so every drafter bug is invisible in the
> output and shows up *only* as collapsed TPF. TPF is our sole instrument.

The specific bug to expect is the guide's §10: vLLM never physically erases
rejected tokens, it moves a length marker. Miss the `seq_lens - num_rejected`
adjustment and the diffusion pass drafts from rejected context — output stays
perfect, TPF halves.

---

## 6. Step-2 acceptance criteria for the vLLM port

Measured on this GPU with these scripts:

| | target |
|---|---|
| GSM8K TPF (1.7B, token-weighted, n≥100) | **≈ 3.9** |
| GSM8K acceptance length (1.7B) | **≈ 7.8** |
| GSM8K TPF (4B) | **≈ 5.0** |
| lossless, fp32 | byte-identical, 100% |
| lossless, bf16 | high match rate; divergence only at near-ties |

Anything within a few percent is kernel numerics. **A drop beyond ~5% is a bug.**

---

## 7. WSL2: vLLM needs `VLLM_WSL2_ENABLE_PIN_MEMORY=1`

Without it, engine startup dies with:

```
RuntimeError: UVA is not available
```

`vllm/platforms/cuda.py:299` disables pinned memory on WSL by default. On kernels
≥ 4.19.121 it is supported but opt-in. Verified genuinely working here:

```
allocate pinned        OK          host write visible on device  True
pageable H2D  13.7 GB/s            pinned H2D  28.0 GB/s
```

The 2× bandwidth confirms real pinning rather than a silent fallback, so the flag
is safe on this machine. **Export it for every vLLM invocation.**

### Other traps

- **Namespace shadowing.** From the repo root, `python -c "import vllm"` resolves
  to the *directory* `./vllm/` (no `__init__.py`) as an empty namespace package,
  silently shadowing the install — `vllm.__file__` is `None`. Run from another
  cwd. Script files are fine (`sys.path[0]` is the script's directory).
- **`use_cache: false`** in the Orthrus `generation_config.json`. AR generation
  must pass `use_cache=True` explicitly or every step recomputes the prefix.
- The reference `OrthrusLM.forward` accepts `attention_mask` but never forwards
  it to `self.model`, so it is **batch-1 only**.

---

## 8. Reproducing

```bash
# Orthrus vs vanilla, mixed prompts, plus both invariants
.venv-ref/bin/python bench/bench_eager.py --max-new-tokens 256

# Paper comparison (use >=100 prompts)
.venv-ref/bin/python bench/bench_eager.py --task gsm8k --num-prompts 200 \
    --max-new-tokens 512 --only orthrus-diff

# 4B
.venv-ref/bin/python bench/bench_eager.py --task gsm8k --num-prompts 200 \
    --max-new-tokens 512 --only orthrus-diff --orthrus-model chiennv/Orthrus-Qwen3-4B

# Strict lossless test (must be fp32)
.venv-ref/bin/python bench/bench_eager.py --dtype fp32 --only orthrus-ar,orthrus-diff

# DFlash, same harness
.venv-ref/bin/python bench/bench_dflash.py --task gsm8k --num-prompts 200 --size 4b
.venv-ref/bin/python bench/measure_pass_cost.py --num-prompts 10

# Are Orthrus's AR weights really the frozen base?
.venv-ref/bin/python bench/compare_weights.py

# vLLM side (note the env var and that it must not run from the repo root)
VLLM_WSL2_ENABLE_PIN_MEMORY=1 .venv/bin/python bench/bench_vllm.py \
    --task gsm8k --num-prompts 200 --size 4b
```

## 9. Still open

- **8B Orthrus vs DFlash.** The published numbers favour DFlash there
  (DFlash GSM8K acceptance 6.54 vs Orthrus 10.6 at 2x the draft cost), and the
  4B result must not be extrapolated to it.
- **Orthrus in vLLM** — needs the port. Projection is ~696 tok/s at 4B; the
  measurement is the point of the project.
- **Batch > 1.** Everything here is `max_num_seqs=1`. The guide's cost model
  (5.2) says the advantage compresses as passes become compute-bound over
  B x K tokens, and Orthrus does *two* full-depth passes where DFlash does one.
  This is where Orthrus is most exposed, and where its zero-extra-KV-cache
  advantage should start paying. Untested.
- **`T > 0`.** Everything here is greedy. The paper reports slightly lower TPF
  at `T=1`, and the rejection-sampling path is different code.
- Orthrus-4B AR-only baseline was never measured, so the 4B speedup lacks its
  honest denominator (see 3).
