# ref_peek — third-party reference files

Unmodified copies of the small, human-readable files from the Orthrus checkpoint, kept in
the repository so the analysis in [`../bench/README.md`](../bench/README.md) and the Field
Manual can cite exact line numbers against something you can read here.

**None of this is our work.** Weights are not included — only configs and the modeling
code.

## Provenance

| file | source | license |
|---|---|---|
| `config.json` | [chiennv/Orthrus-Qwen3-1.7B](https://huggingface.co/chiennv/Orthrus-Qwen3-1.7B) | CC-BY-4.0 |
| `generation_config.json` | chiennv/Orthrus-Qwen3-1.7B | CC-BY-4.0 |
| `modeling_orthrus.py` | chiennv/Orthrus-Qwen3-1.7B | CC-BY-4.0 |
| `MODEL_CARD.md` | chiennv/Orthrus-Qwen3-1.7B (`README.md`) | CC-BY-4.0 |
| `qwen3_config.json` | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) (`config.json`) | Apache-2.0 |

Orthrus is by Nguyen, Hegde, Pham, Rossi, Dernoncourt and Nguyen —
[arXiv:2605.12825](https://arxiv.org/abs/2605.12825),
[github.com/chiennv2000/orthrus](https://github.com/chiennv2000/orthrus).

**Changes made:** file contents are unmodified. Two files were renamed for clarity in this
directory — the checkpoint's `README.md` is here as `MODEL_CARD.md` (so this provenance
note can be the directory README), and Qwen3's `config.json` is here as
`qwen3_config.json` (so it does not collide with Orthrus's). Verified byte-identical to
the Hub originals by SHA-256:

```
config.json             5fb19179808c6098
generation_config.json  46e49064721c09b0
modeling_orthrus.py     7e8f4e01c4c8c469
MODEL_CARD.md           7cd3208067af1580
qwen3_config.json       1ddb5b89ebc90dcb
```

## Refresh from the Hub

```bash
hf download chiennv/Orthrus-Qwen3-1.7B --include "*.json" "*.py" "README.md"
hf download Qwen/Qwen3-1.7B --include "config.json"
```

## What to look at, and why

The pieces the port depends on:

- **`config.json`** — `block_size: 32` (so `num_speculative_tokens` is 31),
  `mask_token_id: 151669`, and `"model_type": "qwen3"` with
  `"architectures": ["OrthrusLM"]`. That last point contradicts the implementation guide,
  which says to detect Orthrus by `model_type == "orthrus"`; detection has to key on the
  **architecture name** instead. The `r: 16` / `lora_alpha: 32` fields are dead leftovers
  from training — nothing in the modeling code reads them.
- **`generation_config.json`** — sets `use_cache: false`, so autoregressive generation must
  pass `use_cache=True` explicitly or every step recomputes the whole prefix.
- **`modeling_orthrus.py`** —
  - lines 92–104: the six `*_diff` tensors per layer, full-rank `nn.Linear`, not LoRA.
  - lines 107–182: `OrthrusAttention.forward`; the diffusion branch concatenates the AR
    cache with the block's own K/V and attends non-causally.
  - line 298: at inference the diffusion pass gets `attention_mask = None` and
    `is_causal = False` — plain dense attention, which is why FlashAttention rather than
    FlexAttention is the right vLLM backend.
  - lines 361–372: `forward` accepts `attention_mask` but never passes it to `self.model`,
    so the reference implementation is **batch-1 only**.
  - lines 460–532: the generation loop — propose, verify, compare, `cache.crop(start_idx)`.
