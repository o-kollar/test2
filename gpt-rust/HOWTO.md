# GPT-Rust — GPT + GQA + MoE + BPE

A faithful Rust port of [@karpathy's pure-Python GPT](https://github.com/karpathy/llm.c/blob/master/gpt.py),
re-engineered with the same vectorized tensor-tape autograd engine used in `rnn-rust`
and upgraded with **Grouped Query Attention (GQA)**, **Mixture of Experts (MoE)**,
**RoPE**, and a **BPE (Byte Pair Encoding) tokenizer**.

## Key Differences from the Python Version

| Aspect | Python (`gpt.py`) | Rust (`gpt-rust`) |
|--------|-------------------|-------------------|
| Autograd | Scalar `Value` nodes | Tensor-level tape (one node = dense matrix) |
| MatMul | Nested Python loops | `matrixmultiply::sgemm` (BLAS-quality) |
| Attention | Standard MHA, sequential | **GQA + Rayon parallel** (`par_iter`) |
| KV heads | Same as Q heads | `n_kv_head ≤ n_head` (shared KV groups) |
| MLP | Single ReLU FFN | **MoE**: 1 shared + top-k sparse SwiGLU experts |
| Position | Learned embeddings | **RoPE** (rotary position encoding) |
| Tokenizer | Char-level | **BPE** (trainable, saved as JSON) |
| Vector ops | Python lists | SIMD auto-vectorized iterators |
| Graph recycling | GC-managed | Arena `reset()` in O(1) |

## Build & Run

```bash
# Build (release for SIMD auto-vectorization + LTO)
cargo build --release

# STEP 1: Train a BPE tokenizer (must be done before training the model)
cargo run --release -- train-tokenizer --trainingFile=input.txt --vocabSize=512
# -> saves ./tokenizer.json

# Larger vocab for richer data:
cargo run --release -- train-tokenizer --trainingFile=input.txt --vocabSize=2048 --tokenizerPath=./tok_2k.json

# STEP 2: Train the GPT model (uses the tokenizer)
cargo run --release -- train --trainingFile=input.txt --numSteps=1000
# -> loads ./tokenizer.json, saves ./gpt_checkpoint.bin

# Train with custom tokenizer path:
cargo run --release -- train --trainingFile=input.txt --tokenizerPath=./tok_2k.json --numSteps=2000

# Train with larger model + custom MoE + GQA
cargo run --release -- train --trainingFile=input.txt \
    --n_embd=64 --n_head=8 --n_kv_head=2 --n_layer=2 \
    --n_experts=8 --top_k=2 --numSteps=2000

# STEP 3: Inference from checkpoint (needs tokenizer + checkpoint)
cargo run --release -- inference --loadCheckpoint=./gpt_checkpoint.bin --tokenizerPath=./tokenizer.json --temperature=0.5

# Resume training
cargo run --release -- train --loadCheckpoint=./gpt_checkpoint.bin --numSteps=3000

# Lazy mode: just pass a training file with no command — it trains the tokenizer
# (if missing) and then trains the model automatically.
cargo run --release -- --trainingFile=input.txt --numSteps=1000
```

## BPE Tokenizer

The tokenizer is a standard **Byte Pair Encoding** implementation, trained separately and stored as a JSON file.
It must be trained before the model — the model's vocabulary size is determined by the tokenizer.

### How it works

1. **Base vocab**: every unique byte in the corpus becomes a token (typically 60-100 for English text)
2. **Special tokens**: `<BOS>` (beginning of sequence) and `<EOS>` (end of sequence) are appended
3. **Merge learning**: iteratively find the most frequent adjacent pair, merge it into a new token, repeat until `--vocabSize` is reached
4. **Encoding**: at inference, text is split to bytes, then merges are applied in the same learned order (greedy)
5. **Decoding**: concatenate the byte sequences of each token, interpret as UTF-8

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--vocabSize` | 512 | Target vocabulary size (including special tokens) |
| `--tokenizerPath` | `./tokenizer.json` | Where to save/load the tokenizer file |
| `--trainingFile` | `input.txt` | Text corpus to train the tokenizer on |

### File format

The tokenizer is saved as human-readable JSON with:
- `merges`: ordered list of `(a, b, new_id)` merge rules
- `vocab`: list of byte sequences for each token
- `bos_id`, `eos_id`: special token IDs

This file is self-contained — no external data needed for encoding/decoding.

## Grouped Query Attention (GQA)

GQA is a generalization that interpolates between standard MHA and Multi-Query Attention:

- **MHA**: `n_kv_head = n_head` — every query head has its own K,V (standard)
- **GQA**: `1 < n_kv_head < n_head` — groups of query heads share K,V (default)
- **MQA**: `n_kv_head = 1` — all query heads share a single K,V pair

Each group of `n_head / n_kv_head` query heads shares one key/value head,
reducing KV projection parameters and memory while preserving quality.

Constraint: `n_head` must be evenly divisible by `n_kv_head`.

## Mixture of Experts (MoE)

DeepSeek-style MoE with a shared always-on expert:

- **Expert 0** is the "shared" expert — always runs on every token (weight = 1.0)
- **Experts 1..n_experts-1** are sparse — a learned router selects `top_k` per token
- Each expert is a full SwiGLU block: `SiLU(x @ W₁) ⊙ (x @ W_gate) @ W₂`
- Router produces softmax scores over sparse experts, top-k are selected and renormalized
- Output = `shared_expert(x) + Σ_{selected} weight_i × expert_i(x)`

This gives the model more capacity without proportional compute increase — each token
only activates `1 + top_k` out of `n_experts` total experts.

Constraints: `n_experts ≥ 2`, `top_k ≤ n_experts - 1`.

## Architecture

Follows GPT-2 with modern upgrades:
- **BPE tokenizer** — byte-pair encoding, trained separately, stored as JSON
- **Mixture of Experts** — shared always-on expert + top-k sparse routed experts
- **SwiGLU** activation per expert — `SiLU(xW₁) ⊙ xW₃` gated activation
- **Grouped Query Attention** — fewer KV heads than Q heads
- **RoPE** — Rotary Position Embeddings on Q and K (no learned positional embeddings)
- **RMSNorm** instead of LayerNorm (no bias, no learned scale)
- **No biases** in linear projections
- Causal self-attention with scaled dot-product
- Residual connections around attention and MoE blocks
