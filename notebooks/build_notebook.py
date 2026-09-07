#!/usr/bin/env python
"""Generates orthrus_kv_walkthrough.ipynb.

The notebook is generated rather than hand-written so the source stays diffable
and the cells stay in sync when numbers change. Run:

    ../.venv/bin/python build_notebook.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): C.append(nbf.v4.new_code_cell(s.strip()))

# ---------------------------------------------------------------- intro
md(r"""
# Orthrus in vLLM — tracing a real paged KV cache

This notebook is the executable version of §5.2 of the Field Manual. Nothing here is a
mock-up: it builds a real paged KV cache, and it launches **the actual Triton kernel from
the vLLM source tree** (`copy_and_expand_dflash_inputs_kernel`) that the DFlash proposer
uses in production.

By the end you will have watched, on real tensors:

1. how a **paged KV cache** stores a token at a physical slot,
2. what the kernel writes into each of its **six output buffers**,
3. why `seq_lens` must subtract `num_rejected` — demonstrated by *breaking it* and
   watching the drafter read tokens the target threw away.

**Run order matters.** Cells build on each other top to bottom.
""")

md(r"""
## 0 · Setup

One gotcha first. The repository root contains a directory called `vllm/`, so if the
notebook's working directory is the project root, `import vllm` resolves to that
*directory* as an empty namespace package instead of the installed library. The cell below
removes the project root from `sys.path` if it is there.
""")

code(r"""
import sys, os
from pathlib import Path

# Drop the project root from sys.path so `import vllm` finds the installed package
# and not the source directory that happens to share its name.
_root = str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd())
for p in ("", ".", _root):
    while p in sys.path:
        sys.path.remove(p)

import torch
import vllm
print("vllm      :", os.path.dirname(vllm.__file__))
print("torch     :", torch.__version__)
print("device    :", torch.cuda.get_device_name(0))
assert vllm.__file__ is not None, "namespace shadowing — restart with cwd=notebooks/"
""")

# ---------------------------------------------------------------- part 1
md(r"""
---
# 1 · What a paged KV cache actually is

vLLM does not give each request one contiguous buffer. It carves GPU memory into
fixed-size **blocks** (pages) and hands them out on demand, like virtual memory. A request
holds a **block table**: the list of physical blocks that store its tokens, in order.

The single formula that matters:

```
slot = block_table[req][position // block_size] * block_size + (position % block_size)
```

To make this visible we use a deliberately tiny cache: **block size 4**, 8 blocks total.
Real vLLM uses 16 tokens per block and thousands of blocks.
""")

code(r"""
BLOCK_SIZE_KV = 4      # tokens per page (real vLLM: 16)
NUM_BLOCKS    = 8      # pages in the whole cache
NUM_KV_HEADS  = 1      # keep it 1 so we can print the thing
HEAD_DIM      = 2

# The real cache is one big tensor: [num_blocks, block_size, num_kv_heads, head_dim].
# A "slot" is a flat index into the first two dimensions.
key_cache = torch.full((NUM_BLOCKS, BLOCK_SIZE_KV, NUM_KV_HEADS, HEAD_DIM),
                       float("nan"))
print("key_cache.shape :", tuple(key_cache.shape))
print("total slots     :", NUM_BLOCKS * BLOCK_SIZE_KV)
""")

md(r"""
Now hand out blocks to two requests. Note they are **not contiguous and not in order** —
that is the entire point of paging. Request A got blocks 5, 2, 7; request B got 1, 4.
""")

code(r"""
block_table = torch.tensor([
    [5, 2, 7],   # request A: logical page 0 -> physical block 5, page 1 -> 2, page 2 -> 7
    [1, 4, 0],   # request B: page 0 -> block 1, page 1 -> 4   (third entry unused)
], dtype=torch.int32)

def slot_of(req, position, block_table=block_table, bs=BLOCK_SIZE_KV):
    "The one formula. Returns the flat physical slot for a logical position."
    page   = position // bs
    offset = position %  bs
    return int(block_table[req, page]) * bs + offset

print("req A, positions 0..9 -> slots:", [slot_of(0, p) for p in range(10)])
print("req B, positions 0..7 -> slots:", [slot_of(1, p) for p in range(8)])
""")

md(r"""
Look at request A's slots: `20,21,22,23, 8,9,10,11, 28,29`. They jump backwards at
position 4. **Logically adjacent tokens are physically far apart.** Any code that assumes
`slot(p+1) == slot(p)+1` is broken; that is why the kernel does a real block-table lookup
per position instead of adding one.

Now write into the cache. To keep it legible each token's key vector is just
`[position, position]`, so we can read the cache and see which token lives where.
""")

code(r"""
def write_token(req, position, value):
    s = slot_of(req, position)
    key_cache[s // BLOCK_SIZE_KV, s % BLOCK_SIZE_KV, 0, :] = value

# Request A holds 10 tokens at positions 0..9.
for p in range(10):
    write_token(0, p, torch.tensor([float(p), float(p)]))

def show_cache():
    print("block | slot: value   (nan = unallocated)")
    for b in range(NUM_BLOCKS):
        cells = []
        for o in range(BLOCK_SIZE_KV):
            v = key_cache[b, o, 0, 0].item()
            cells.append(f"{b*BLOCK_SIZE_KV+o:>3}:{'  . ' if v != v else f'{v:4.0f}'}")
        owner = ""
        if b in (5, 2, 7): owner = "  <- req A"
        if b in (1, 4):    owner = "  <- req B (empty)"
        print(f"  {b}   | " + " ".join(cells) + owner)

show_cache()
""")

md(r"""
Read that output carefully. Request A's tokens 0–9 are scattered across blocks 5, 2 and 7,
in that order, while blocks 0, 3, 6 are free and blocks 1, 4 belong to request B.

**This is the mental model you need for everything below**: a "position" is logical, a
"slot" is physical, and the block table is the only thing connecting them.
""")

# ---------------------------------------------------------------- part 2
md(r"""
---
# 2 · The scenario

Now switch to the Field Manual's example so every number can be checked against §5.2.
Real vLLM parameters from here on: **page size 16**, `num_speculative_tokens = 4`, so each
request contributes `1 + 4 = 5` query tokens.

Two requests, both of which were drafted for last step. The target has just verified those
drafts and consensus has run:

| | committed before | target verified | accepted | now committed | **rejected** |
|---|---|---|---|---|---|
| **Request 0** | positions 0–9 | positions 10–14 | 2 drafts | 10, 11, 12 | **13, 14** |
| **Request 1** | positions 0–29 | positions 30–34 | 0 drafts | 30 | **31, 32, 33, 34** |

Request 1 accepted *nothing* — that makes the failure in §5 unmistakable.
""")

code(r"""
dev = "cuda"

MASK_TOKEN_ID = 151669          # Orthrus/DFlash <mask>, from the checkpoint config
B0, B1        = 9001, 9002      # the two bonus tokens (fake ids, easy to spot)
num_spec      = 4               # num_speculative_tokens
nqpr          = 1 + num_spec    # num_query_per_req = 5
PAGE          = 16              # KV page size
STRIDE        = 8               # columns in the block table

# --- exactly what the target model handed the proposer -------------------
query_start_loc  = torch.tensor([0, 5, 10], dtype=torch.int32, device=dev)
target_positions = torch.tensor([10,11,12,13,14, 30,31,32,33,34],
                                dtype=torch.int64, device=dev)
num_rejected     = torch.tensor([2, 4], dtype=torch.int32, device=dev)
next_token_ids   = torch.tensor([B0, B1], dtype=torch.int32, device=dev)
seq_lens         = torch.tensor([15, 35], dtype=torch.int32, device=dev)

bt = torch.zeros((2, STRIDE), dtype=torch.int32, device=dev)
bt[0, 0], bt[0, 1]           = 42, 57          # req 0: pos 0-15 -> 42, 16-31 -> 57
bt[1, 0], bt[1, 1], bt[1, 2] = 11, 23, 88      # req 1: 0-15 -> 11, 16-31 -> 23, 32-47 -> 88

print("target_positions is FLATTENED and RAGGED across requests:")
for r in range(2):
    lo, hi = query_start_loc[r].item(), query_start_loc[r+1].item()
    print(f"  req {r}: target_positions[{lo}:{hi}] = {target_positions[lo:hi].tolist()}"
          f"   rejected={num_rejected[r].item()}")
""")

# ---------------------------------------------------------------- part 3
md(r"""
---
# 3 · Running the real kernel

Everything below is vLLM's production code path — the same call `dflash.py` makes in
`set_inputs_first_pass`.
""")

code(r"""
from vllm.v1.spec_decode.utils import (
    copy_and_expand_dflash_inputs_kernel,
    next_power_of_2,
)
import inspect
print("kernel lives in:", inspect.getsourcefile(copy_and_expand_dflash_inputs_kernel.fn))
""")

md(r"""
### The launch grid

One Triton program per *(request, block-of-tokens)*. Each program walks that request's
context tokens **and** its query tokens under a single flat index `j`.
""")

code(r"""
batch           = 2
num_context     = target_positions.numel()      # 10
num_query_total = batch * nqpr                  # 10

max_ctx_per_req    = 5                          # cad.max_query_len
max_tokens_per_req = max_ctx_per_req + nqpr     # 10
BS       = min(256, next_power_of_2(max_tokens_per_req))
nblocks  = (max_tokens_per_req + BS - 1) // BS
grid     = (batch, nblocks)

print(f"BLOCK_SIZE = min(256, next_power_of_2({max_tokens_per_req})) = {BS}")
print(f"num_blocks = {nblocks}      grid = {grid}   ({batch*nblocks} programs)")
print()
print("inside one program, the flat index j covers:")
print("   j:          0    1    2    3    4  |  5    6    7    8    9  | 10 .. 15")
print("             └── is_ctx (j < num_ctx=5) ┘ └── is_query ──┘  └ out of bounds ┘")
print("   query_off:                            0    1    2    3    4")
""")

code(r"""
# Output buffers, with the dtypes the proposer really allocates.
out_input_ids = torch.zeros(num_query_total,   dtype=torch.int32, device=dev)
out_ctx_pos   = torch.zeros(num_context,       dtype=torch.int64, device=dev)
out_q_pos     = torch.zeros(num_query_total,   dtype=torch.int64, device=dev)
out_ctx_slot  = torch.zeros(num_context,       dtype=torch.int64, device=dev)
out_q_slot    = torch.zeros(num_query_total,   dtype=torch.int64, device=dev)
out_tok_idx   = torch.zeros(batch * num_spec,  dtype=torch.int32, device=dev)

def run_kernel(has_num_rejected: bool):
    "Launch the real kernel. has_num_rejected=False is the bug we study in section 5."
    out_input_ids.zero_(); out_ctx_pos.zero_(); out_q_pos.zero_()
    out_ctx_slot.zero_();  out_q_slot.zero_();  out_tok_idx.zero_()
    copy_and_expand_dflash_inputs_kernel[grid](
        next_token_ids_ptr=next_token_ids,
        target_positions_ptr=target_positions,
        out_input_ids_ptr=out_input_ids,
        out_context_positions_ptr=out_ctx_pos,
        out_query_positions_ptr=out_q_pos,
        out_context_slot_mapping_ptr=out_ctx_slot,
        out_query_slot_mapping_ptr=out_q_slot,
        out_token_indices_ptr=out_tok_idx,
        block_table_ptr=bt,
        block_table_stride=bt.stride(0),
        query_start_loc_ptr=query_start_loc,
        num_rejected_tokens_ptr=num_rejected if has_num_rejected else 0,
        parallel_drafting_token_id=MASK_TOKEN_ID,
        block_size=PAGE,
        num_query_per_req=nqpr,
        num_speculative_tokens=num_spec,
        total_input_tokens=num_context,
        BLOCK_SIZE=BS,
        HAS_NUM_REJECTED=has_num_rejected,
    )
    torch.cuda.synchronize()

run_kernel(has_num_rejected=True)
print("kernel ran")
""")

md(r"""
### The six output buffers
""")

code(r"""
def show_buffers():
    idx = list(range(num_query_total))
    print("             " + "".join(f"{i:>7}" for i in idx))
    print("             " + "-------" * num_query_total)
    def row(name, t, fmt="{:>7}"):
        print(f"{name:<13}" + "".join(fmt.format(v) for v in t.tolist()))
    row("ctx_pos",  out_ctx_pos)
    row("ctx_slot", out_ctx_slot)
    print()
    row("q_pos",    out_q_pos)
    row("q_slot",   out_q_slot)
    ids = ["B0" if v == B0 else "B1" if v == B1 else "MASK" for v in out_input_ids.tolist()]
    print(f"{'input_ids':<13}" + "".join(f"{v:>7}" for v in ids))
    print()
    print(f"{'tok_indices':<13}" + "".join(f"{v:>7}" for v in out_tok_idx.tolist())
          + f"   (len {out_tok_idx.numel()})")

show_buffers()
""")

md(r"""
### Check it against the hand computation

These are the numbers worked out by hand in §5.2 of the manual. If the assertions pass,
the manual is right and so is your understanding of the formula.
""")

code(r"""
expected = {
    "ctx_pos":   [10,11,12,13,14, 30,31,32,33,34],
    "ctx_slot":  [682,683,684,685,686, 382,383,1408,1409,1410],
    "q_pos":     [13,14,15,16,17, 31,32,33,34,35],
    "q_slot":    [685,686,687,912,913, 383,1408,1409,1410,1411],
    "input_ids": [B0,MASK_TOKEN_ID,MASK_TOKEN_ID,MASK_TOKEN_ID,MASK_TOKEN_ID,
                  B1,MASK_TOKEN_ID,MASK_TOKEN_ID,MASK_TOKEN_ID,MASK_TOKEN_ID],
    "tok_idx":   [1,2,3,4, 6,7,8,9],
}
actual = {"ctx_pos": out_ctx_pos, "ctx_slot": out_ctx_slot, "q_pos": out_q_pos,
          "q_slot": out_q_slot, "input_ids": out_input_ids, "tok_idx": out_tok_idx}
for k, exp in expected.items():
    got = actual[k].tolist()
    assert got == exp, f"{k}\n  got {got}\n  exp {exp}"
    print(f"  {k:<10} matches hand computation")
print("\nAll six buffers reproduce the manual exactly.")
""")

md(r"""
### Where each number came from

Spot-check two slots by hand with the formula from section 1:

- request 0, query position **16**: `16 // 16 = 1`, so logical page 1, which
  `block_table[0]` maps to physical block **57**. Slot `= 57*16 + (16 % 16) = 912`.
- request 1, context position **32**: `32 // 16 = 2` → block **88**.
  Slot `= 88*16 + 0 = 1408`.
""")

code(r"""
def slot_formula(req, pos):
    page, off = pos // PAGE, pos % PAGE
    blk = int(bt[req, page])
    return blk, page, off, blk * PAGE + off

for req, pos in [(0, 15), (0, 16), (1, 31), (1, 32)]:
    blk, page, off, s = slot_formula(req, pos)
    print(f"req {req} pos {pos:>2}:  page {page} -> block {blk:>2},  "
          f"slot = {blk}*{PAGE} + {off:>2} = {s}")
print("\nNote how req 0 jumps 687 -> 912 between positions 15 and 16,")
print("and req 1 jumps 383 -> 1408 between 31 and 32: page boundaries.")
""")

md(r"""
### Two things a single-request example would hide

**The context and query buffers are indexed by different schemes.**
""")

code(r"""
print("context uses ctx_pos_out = ctx_start + j        (RAGGED)")
print("query   uses query_out   = req_idx*nqpr + off   (UNIFORM)")
print()
for r in range(2):
    ctx_start = query_start_loc[r].item()
    print(f"  req {r}: ctx writes to indices "
          f"{[ctx_start + j for j in range(5)]}"
          f"   query writes to {[r*nqpr + o for o in range(nqpr)]}")
print()
print("Context is ragged because requests contribute different token counts to the")
print("target's pass. Query is uniform because EVERY request always gets exactly")
print(f"1 + num_speculative_tokens = {nqpr} of them. That regularity is why")
print("new_query_start_loc is simply arange * num_query_per_req:")
print("   ", (torch.arange(batch + 1) * nqpr).tolist())
""")

md(r"""
**`last_pos` is read at a per-request offset into the flattened array.** This is the
line that skips the rejected tokens:

```python
valid_ctx_end = ctx_end - num_rejected
last_pos      = target_positions[valid_ctx_end - 1]
query_pos     = last_pos + 1 + query_off
```
""")

code(r"""
for r in range(2):
    ctx_end = query_start_loc[r+1].item()
    rej     = num_rejected[r].item()
    valid   = ctx_end - rej
    last    = target_positions[valid - 1].item()
    print(f"req {r}:  ctx_end={ctx_end:>2}  - rejected={rej}  -> valid_ctx_end={valid:>2}")
    print(f"        last_pos = target_positions[{valid-1}] = {last}"
          f"   (NOT target_positions[{ctx_end-1}] = {target_positions[ctx_end-1].item()})")
    print(f"        drafting resumes at {last+1} -> "
          f"{[last + 1 + o for o in range(nqpr)]}\n")
""")

# ---------------------------------------------------------------- part 4
md(r"""
---
# 4 · The metadata: why `seq_lens` subtracts `num_rejected`

The kernel built the *inputs*. `set_inputs_first_pass` then builds the *attention
metadata*, and this is the line the Field Manual warns about
(`dflash.py:176-180`):

```python
effective_seq_lens = cad.seq_lens
if has_num_rejected:
    effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu
...
seq_lens = effective_seq_lens + num_query_per_req
```
""")

code(r"""
effective = seq_lens - num_rejected
new_seq_lens = effective + nqpr

print(f"{'':8}{'seq_lens':>10}{'rejected':>10}{'effective':>11}{'+nqpr':>8}")
for r in range(2):
    print(f"req {r}: {seq_lens[r].item():>10}{num_rejected[r].item():>10}"
          f"{effective[r].item():>11}{new_seq_lens[r].item():>8}")
print()
for r in range(2):
    e, n = effective[r].item(), new_seq_lens[r].item()
    print(f"req {r}: {e} context (positions 0..{e-1}) + {nqpr} query "
          f"(positions {e}..{e+nqpr-1}) = {n}  ✓")
""")

# ---------------------------------------------------------------- part 5
md(r"""
---
# 5 · Breaking it on purpose

Now the payoff. We run the identical kernel with `HAS_NUM_REJECTED=False` — which is what
happens if you drop the adjustment when porting — and watch what changes.
""")

code(r"""
run_kernel(has_num_rejected=True)
good_q_pos, good_q_slot = out_q_pos.clone(), out_q_slot.clone()

run_kernel(has_num_rejected=False)
bad_q_pos, bad_q_slot = out_q_pos.clone(), out_q_slot.clone()

print("query POSITIONS")
print("  correct :", good_q_pos.tolist())
print("  broken  :", bad_q_pos.tolist())
print()
print("query SLOTS")
print("  correct :", good_q_slot.tolist())
print("  broken  :", bad_q_slot.tolist())
print()
for r in range(2):
    g = good_q_pos[r*nqpr:(r+1)*nqpr].tolist()
    b = bad_q_pos[r*nqpr:(r+1)*nqpr].tolist()
    print(f"req {r}: drafting should resume at {g[0]}, broken version resumes at {b[0]}"
          f"   (off by {b[0]-g[0]} = num_rejected)")
""")

md(r"""
The broken version resumes drafting **past the rejected tokens as if they had been
accepted**. Request 1 is off by four positions: it treats four tokens the target explicitly
threw away as committed history.

### Watch it read the rejected tokens

Let's make this physical. We build a small paged cache for request 0, write *real*
committed values at positions 0–12 and clearly-marked **poison** at the rejected positions
13–14 — poison is what the target wrote for tokens consensus then discarded.
""")

code(r"""
# A cache big enough for request 0's blocks 42 and 57, page size 16.
NB = 64
kc = torch.full((NB * PAGE,), float("nan"))

def w(pos, val):
    page, off = pos // PAGE, pos % PAGE
    kc[int(bt[0, page]) * PAGE + off] = val

for p in range(13):      # committed tokens 0..12 -> value = position
    w(p, float(p))
for p in (13, 14):       # REJECTED tokens still physically present
    w(p, -999.0)

print("what is physically in request 0's cache, positions 0..17:")
for p in range(18):
    page, off = p // PAGE, p % PAGE
    s = int(bt[0, page]) * PAGE + off
    v = kc[s].item()
    tag = ""
    if v == -999.0:  tag = "  <-- REJECTED, still in cache"
    elif v != v:     tag = "  (never written)"
    print(f"  pos {p:>2}  slot {s:>4}  value {'nan' if v != v else f'{v:7.0f}'}{tag}")
""")

code(r"""
def gather_context(seq_len):
    "What the attention kernel treats as this request's context."
    out = []
    for p in range(seq_len):
        page, off = p // PAGE, p % PAGE
        out.append(kc[int(bt[0, page]) * PAGE + off].item())
    return out

correct = gather_context(13)   # effective_seq_lens[0] = 15 - 2
broken  = gather_context(15)   # forgot the subtraction

print("CORRECT  seq_len=13 :", [f"{v:.0f}" for v in correct])
print("BROKEN   seq_len=15 :", [f"{v:.0f}" for v in broken])
print()
print("poison values visible to the drafter:")
print("  correct:", sum(v == -999.0 for v in correct))
print("  broken :", sum(v == -999.0 for v in broken), " <-- conditions on rejected tokens")
""")

md(r"""
### And now the part that makes this dangerous

Ask yourself what a test would show.

The drafter is now conditioning on garbage, so its proposals get worse and **fewer of them
are accepted**. But every proposal still goes through the target's verification pass, and
verification rejects anything the target would not have produced itself. So:

- the generated text is still **exactly correct** — byte-for-byte what the target model
  would have produced on its own,
- no exception is raised, no assertion fires, no test turns red,
- the only symptom is that acceptance length quietly falls from ~7.8 to ~4.

That is why the Field Manual insists on measuring acceptance length rather than reading
code, and why the target numbers were pinned down before any code was written.
""")

code(r"""
print("Symptom table for a dropped `- num_rejected`:")
print()
print(f"  {'signal':<34}{'what you would see'}")
print(f"  {'-'*34}{'-'*34}")
for sig, val in [("output text", "correct, byte for byte"),
                 ("exceptions / tracebacks", "none"),
                 ("unit tests", "pass"),
                 ("lossless (greedy equality) test", "PASSES"),
                 ("acceptance length", "~7.8  ->  ~4    <-- only signal")]:
    print(f"  {sig:<34}{val}")
""")

# ---------------------------------------------------------------- part 6
md(r"""
---
# 6 · What changes for Orthrus

Everything above is DFlash. Orthrus reuses this machinery almost unchanged. The differences:

| | DFlash | Orthrus |
|---|---|---|
| `num_query_per_req` | `1 + 15` (block_size 16) | `1 + 31` (block_size **32**) |
| the query block | `[bonus, MASK×n]` | `[anchor, MASK×(K-1)]` — **the same thing** |
| context K/V | recomputed into the drafter's own cache | already in the target's cache — **nothing to do** |
| `out_context_*` buffers | needed | **unused** |
| draft KV cache | separate | none |
| `- num_rejected` | required | **required, identically** |

The bonus token *is* Orthrus's anchor — the guide's glossary says so explicitly. So the two
buffers we can delete are the context ones, and the logic we must keep verbatim is the part
that was hardest to get right.
""")

code(r"""
K_ORTHRUS = 32                      # from chiennv/Orthrus-Qwen3-1.7B config.json
print(f"Orthrus block_size            : {K_ORTHRUS}")
print(f"num_speculative_tokens        : {K_ORTHRUS - 1}")
print(f"num_query_per_req             : {K_ORTHRUS}    = 1 anchor + {K_ORTHRUS-1} masks")
print()
print("Buffers Orthrus still needs:")
for b in ["out_input_ids", "out_query_positions", "out_query_slot_mapping",
          "out_token_indices_to_sample"]:
    print(f"   keep    {b}")
for b in ["out_context_positions", "out_context_slot_mapping"]:
    print(f"   DELETE  {b}   (context K/V is already in the target's cache)")
""")

md(r"""
---
## Exercises

1. Change `num_rejected` to `[0, 0]` (nothing rejected) and re-run section 3. What happens
   to `q_pos`? Why does `HAS_NUM_REJECTED=False` now give the *same* answer?
2. Set `bt[0, 1] = 42` so request 0's two logical pages map to the *same* physical block.
   Which slots collide, and what would that corrupt?
3. Request 1 accepted nothing. Work out by hand what `q_pos` would be if it had accepted
   all 4 drafts, then check yourself by setting `num_rejected[1] = 0`.
4. In section 5, why does the broken version's *first* query position differ per request by
   exactly `num_rejected[r]`?

## Where to go next

- `bench/README.md` — the measured numbers this all has to reproduce
- `vllm/v1/spec_decode/dflash.py` — 317 lines; you have now read the hard part
- `vllm/v1/spec_decode/utils.py:458` — the kernel itself
""")

nb["cells"] = C
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "orthrus_kv_walkthrough.ipynb")
print("wrote orthrus_kv_walkthrough.ipynb with", len(C), "cells")
