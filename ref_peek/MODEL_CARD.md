---
library_name: transformers
tags:
- text-generation
- diffusion
- parallel-decoding
- orthrus
- qwen3
license: cc-by-4.0
pipeline_tag: text-generation
---

# Orthrus-Qwen3-1.7B
[**Paper**](https://arxiv.org/abs/2605.12825) | [**GitHub**](https://github.com/chiennv2000/orthrus)

**Orthrus** is a simple and efficient dual-architecture framework that unifies the exact generation fidelity of autoregressive Large Language Models (LLMs) with the high-speed parallel token generation of diffusion models. By augmenting a frozen pre-trained LLM with a lightweight, trainable diffusion module, Orthrus delivers significantly accelerated inference without sacrificing output quality.

<p align="center">
  <img src="orthrus.png" width="80%" alt="Orthrus Dual-View Architecture">
</p>

- **Repository:** [https://github.com/chiennv2000/orthrus](https://github.com/chiennv2000/orthrus)
- **Architecture:** Dual-View Attention (Autoregressive Base + Parallel Diffusion Head)

## Key Features

- **Significant Inference Acceleration:** Breaks the sequential bottleneck of standard autoregressive decoding, delivering up to a $7.8\times$ speedup on generation tasks.
- **Strictly Lossless Generation:** Employs an exact intra-model consensus mechanism to guarantee that the output matches the original base model's exact predictive distribution.
- **Zero Redundant Memory Overhead:** Both the autoregressive and diffusion views attend to the exact same high-fidelity Key-Value (KV) cache natively, resulting in only an $O(1)$ memory cache overhead.
- **Parameter Efficient:** Parallel generation capabilities are injected by fine-tuning only 16% of the total model parameters while keeping the base LLM strictly frozen.

## Installation

Ensure you have `transformers`, `torch`, and [flash-attention](https://github.com/dao-ailab/flash-attention) installed. We used `torch==2.10` and `transformers==5.8.0`.

## How to Get Started

Use the following code to run inference with the model. Ensure your environment supports FlashAttention and you are passing `trust_remote_code=True` to load the custom Orthrus architecture.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch

MODEL_PATH = "chiennv/Orthrus-Qwen3-1.7B"

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, 
    dtype=torch.bfloat16, 
    device_map="cuda", 
    attn_implementation="flash_attention_2", # options: sdpa | eager | flash_attention_4
    trust_remote_code=True # Note: trust_remote_code=True is required
).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

prompt = "Write a program to count the frequency of each word in a paragraph."
messages = [
    {"role": "system", "content": ""},
    {"role": "user", "content": prompt}
]

input_ids = tokenizer.apply_chat_template(
    messages, 
    tokenize=True,
    enable_thinking=False,
    add_generation_prompt=True, 
    return_tensors="pt",
).input_ids

# Generate text natively utilizing parallel diffusion projection
output_ids = model.generate(
    input_ids=input_ids.to(model.device), 
    max_new_tokens=2048,
    use_diffusion_mode=True, 
    streamer=TextStreamer(tokenizer, skip_prompt=True) # enable streaming
)
```

## Citation

If you find this model or architecture useful in your work, please cite our paper:

```bibtex
@misc{vannguyen2026orthrusmemoryefficientparalleltoken,
      title={Orthrus: Memory-Efficient Parallel Token Generation via Dual-View Diffusion}, 
      author={Chien Van Nguyen and Chaitra Hegde and Van Cuong Pham and Ryan A. Rossi and Franck Dernoncourt and Thien Huu Nguyen},
      year={2026},
      eprint={2605.12825},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.12825}, 
}
```