"""Shared prompt set for baseline benchmarking.

Mix of math / code / general, because the Orthrus paper reports its best
tokens-per-forward on math and code (structured, predictable continuations)
and its worst on open-ended prose. Keeping the mix means the mean acceptance
length we measure is comparable to the paper's Table 1 rather than being
flattered by one easy category.
"""

PROMPTS = [
    # --- math (expect high acceptance) ---
    ("math", "Compute 17 * 24 and show each step of the multiplication."),
    ("math", "Solve for x: 3x + 7 = 25. Show your working."),
    ("math", "What is the sum of the first 20 positive integers? Explain the formula."),
    ("math", "A train travels 240 km in 3 hours. What is its average speed in m/s?"),
    # --- code (expect high acceptance) ---
    ("code", "Write a Python function that returns the n-th Fibonacci number iteratively."),
    ("code", "Write a Python function to reverse a linked list. Include the Node class."),
    ("code", "Write a bash one-liner that finds the 10 largest files under the current directory."),
    ("code", "Implement binary search in Python with a docstring and type hints."),
    # --- general prose (expect lower acceptance) ---
    ("prose", "Explain what a KV cache is in a transformer, in two short paragraphs."),
    ("prose", "Describe the difference between a process and a thread."),
    ("prose", "Give three reasons why someone might prefer a bicycle to a car in a city."),
    ("prose", "Summarise what makes GPU memory bandwidth a bottleneck during LLM decoding."),
]


def load_task_prompts(task: str, n: int | None) -> list[tuple[str, str]]:
    """Prompt source, shared by the HF and vLLM harnesses.

    'mixed' is our own quick sanity set. 'gsm8k' and 'humaneval' are benchmarks
    both the Orthrus paper (Table 1) and the DFlash paper (Table 1, which vLLM's
    own e2e test asserts against) report, so numbers measured on them are
    directly comparable to both papers.

    Both harnesses must use this same function, or an Orthrus-vs-DFlash
    comparison silently compares different prompts.
    """
    if task == "mixed":
        return PROMPTS if n is None else PROMPTS[:n]

    from datasets import load_dataset

    if task == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="test")
        return [("gsm8k", ds[i]["question"]) for i in range(n or 20)]
    if task == "humaneval":
        ds = load_dataset("openai/openai_humaneval", split="test")
        return [("humaneval",
                 "Complete the following Python function. Reply with the full "
                 "function implementation.\n\n" + ds[i]["prompt"])
                for i in range(n or 20)]
    raise ValueError(f"unknown task {task}")
