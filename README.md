# latent-attention

A PyTorch module for multi-head attention with low-rank (latent) key/value compression, inspired by
Multi-head Latent Attention (MLA) from DeepSeek-V2, plus a Triton version of the attention core that
uses a fused online-softmax kernel, and a script that benchmarks the two.

## What's here

**`multihead_latent_attention.py` - `MultiheadLatentAttention`** (plain PyTorch)

```
latent (B, L, latent_dim) ──query_proj──────────────────────────────► Q
x      (B, S, input_dim)  ──compress──► c (B, S, latent_rank) ─┬─decompress_k──► K
                                                                └─decompress_v──► V
softmax(Q Kᵀ / sqrt(head_dim)) V  ──concat heads──► out_proj ──► (B, L, latent_dim)
```

- Keys and values share one down-projection (`compress`) to a `latent_rank`-dimensional vector per
  token, then are expanded per head by `decompress_k` / `decompress_v`. This is the low-rank joint
  KV compression idea behind MLA.
- Queries come from a separate `latent` tensor, so the module is a cross-attention block
  (closer to Perceiver-style latent queries than to DeepSeek's self-attention layer).
- Supports an optional `mask` and dropout on the attention weights; returns `(output, attn_weights)`.

**`multihead_latent_attention_triton.py` - `MultiheadLatentAttentionTriton`**

Same projections, but `softmax(QKᵀ)V` runs in `fused_attention_kernel`, a Triton kernel that tiles
queries (`BLOCK_M=16`) and keys (`BLOCK_N=64`) and keeps a running max / running sum per query
(online softmax, as in FlashAttention), so the full `L x S` score matrix is never materialized.

**`benchmark.py`** - times both modules (`B=128`, `S=512`, `L=256`, `input_dim=512`, `latent_dim=256`,
8 heads, `latent_rank=128`) over 50 iterations after warm-up and prints the average ms per forward pass.

### Not implemented (compared with DeepSeek MLA)

- Decoupled RoPE keys / any positional encoding
- Low-rank query compression
- KV caching of the compressed latent and the weight-absorption trick for inference
- Causal masking

### Limitations of the Triton version

- Forward only: there is no backward kernel or `torch.autograd.Function`, so gradients do not flow
  through it. Use it for inference/benchmarking.
- Ignores `mask` and `dropout`, and returns `None` instead of attention weights.
- `head_dim` (`latent_dim / num_heads`) must be a power of two and at least 16 for `tl.arange` / `tl.dot`.

## Getting started

Requires an NVIDIA GPU with CUDA for the Triton kernel and the benchmark.

```bash
git clone https://github.com/khushwant04/latent-attention.git
cd latent-attention
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python benchmark.py
```

Usage:

```python
import torch
from multihead_latent_attention import MultiheadLatentAttention

mla = MultiheadLatentAttention(input_dim=512, latent_dim=256, num_heads=8, latent_rank=128)
x = torch.randn(2, 128, 512)       # tokens to attend over (keys/values)
latent = torch.randn(2, 32, 256)   # latent queries
out, attn = mla(latent, x)         # out: (2, 32, 256), attn: (2, 8, 32, 128)
```

`MultiheadLatentAttentionTriton` has the same constructor and `forward(latent, x)` signature; inputs
must be CUDA tensors.

## Project structure

```
.
├── multihead_latent_attention.py         # PyTorch reference module
├── multihead_latent_attention_triton.py  # Triton fused-attention kernel + module
├── benchmark.py                          # PyTorch vs Triton timing
└── setup.py                              # CUDA extension stub (see note)
```

`setup.py` declares a CUDA extension built from `attention_cuda.cu`, which is not in the repository,
so `python setup.py build_ext` will fail. It is not needed to run the modules above.

## References

- DeepSeek-AI, *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model* (introduces MLA), 2024. [arXiv:2405.04434](https://arxiv.org/abs/2405.04434)
- Jaegle et al., *Perceiver: General Perception with Iterative Attention* (latent-query cross-attention), 2021. [arXiv:2103.03206](https://arxiv.org/abs/2103.03206)
- Dao et al., *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*, 2022. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- Vaswani et al., *Attention Is All You Need*, 2017. [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- [Triton documentation](https://triton-lang.org/)
