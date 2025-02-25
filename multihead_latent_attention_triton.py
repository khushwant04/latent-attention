# multihead_latent_attention_triton.py
import math
import torch
import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def fused_attention_kernel(
    Q, K, V, O,
    stride_q_b, stride_q_h, stride_q_m, stride_q_k,
    stride_k_b, stride_k_h, stride_k_n, stride_k_k,
    stride_v_b, stride_v_h, stride_v_n, stride_v_k,
    stride_o_b, stride_o_h, stride_o_m, stride_o_k,
    L: tl.constexpr,  # number of queries per head
    S: tl.constexpr,  # number of keys per head
    d: tl.constexpr,  # head dimension
    BLOCK_M: tl.constexpr,  # tile size along queries (e.g., 16)
    BLOCK_N: tl.constexpr,  # tile size along keys (e.g., 64)
):
    # Grid: (B, H, ceil(L / BLOCK_M))
    b = tl.program_id(0)         # batch index
    h = tl.program_id(1)         # head index
    m_block = tl.program_id(2)   # tile index along query dimension
    m_start = m_block * BLOCK_M  # starting query index for this block

    # Offsets for queries in the tile.
    offs_m = tl.arange(0, BLOCK_M)
    m_indices = m_start + offs_m

    # Load Q tile: shape (BLOCK_M, d)
    q_ptrs = Q + b * stride_q_b + h * stride_q_h + m_indices[:, None] * stride_q_m + tl.arange(0, d)[None, :] * stride_q_k
    Q_tile = tl.load(q_ptrs, mask=(m_indices[:, None] < L), other=0.0)

    # Precompute scaling factor.
    scale = 1.0 / tl.sqrt(tl.cast(d, tl.float32))

    # Initialize fused accumulators:
    # m_acc: running maximum per query (BLOCK_M,)
    # sum_acc: running sum of exponentials per query (BLOCK_M,)
    # out_acc: running weighted sum per query (BLOCK_M, d)
    m_acc = tl.full([BLOCK_M], -1e9, tl.float32)
    sum_acc = tl.zeros([BLOCK_M], tl.float32)
    out_acc = tl.zeros([BLOCK_M, d], tl.float32)

    # Fused loop: iterate over key tiles.
    for n_start in range(0, S, BLOCK_N):
        offs_n = tl.arange(0, BLOCK_N)
        n_indices = n_start + offs_n

        # Load K tile in transposed layout to get shape (d, BLOCK_N).
        k_ptrs = K + b * stride_k_b + h * stride_k_h + tl.arange(0, d)[:, None] * stride_k_k + n_indices[None, :] * stride_k_n
        K_tile = tl.load(k_ptrs, mask=(n_indices[None, :] < S), other=0.0)

        # Load V tile: shape (BLOCK_N, d)
        v_ptrs = V + b * stride_v_b + h * stride_v_h + n_indices[:, None] * stride_v_n + tl.arange(0, d)[None, :] * stride_v_k
        V_tile = tl.load(v_ptrs, mask=(n_indices[:, None] < S), other=0.0)

        # Compute scores: (BLOCK_M, d) dot (d, BLOCK_N) -> (BLOCK_M, BLOCK_N)
        scores = tl.dot(Q_tile, K_tile, allow_tf32=False) * scale

        # Compute the maximum score in the current tile for each query.
        block_max = tl.max(scores, axis=1)  # shape: (BLOCK_M,)

        # Compute the new running maximum for each query.
        new_m = tl.maximum(m_acc, block_max)  # shape: (BLOCK_M,)

        # Compute exponentials adjusted by the new running maximum.
        exp_scores = tl.exp(scores - new_m[:, None])  # shape: (BLOCK_M, BLOCK_N)

        # Sum of exponentials for current tile.
        tile_sum = tl.sum(exp_scores, axis=1)  # shape: (BLOCK_M,)

        # Weighted sum over V: (BLOCK_M, BLOCK_N) dot (BLOCK_N, d) -> (BLOCK_M, d)
        tile_weighted = tl.dot(exp_scores, V_tile)

        # Update accumulators:
        # Scale previous accumulators to the new maximum.
        scale_factor = tl.exp((m_acc - new_m))  # shape: (BLOCK_M,)
        # To broadcast correctly over dimension 1, reshape to (BLOCK_M, 1)
        scale_factor = scale_factor[:, None]
        sum_acc = sum_acc * tl.exp(m_acc - new_m) + tile_sum
        out_acc = out_acc * scale_factor + tile_weighted

        # Update running maximum.
        m_acc = new_m

    # Compute final output for the query tile.
    out_tile = out_acc / sum_acc[:, None]

    # Write the output tile.
    o_ptrs = O + b * stride_o_b + h * stride_o_h + m_indices[:, None] * stride_o_m + tl.arange(0, d)[None, :] * stride_o_k
    tl.store(o_ptrs, out_tile, mask=(m_indices[:, None] < L))


def attention_forward_triton(Q, K, V):
    """
    Compute scaled dot-product attention using the advanced fused Triton kernel with a fused softmax loop.

    Args:
        Q: Query tensor of shape (B, H, L, d)
        K, V: Key/Value tensors of shape (B, H, S, d)

    Returns:
        output: Tensor of shape (B, H, L, d)
    """
    B, H, L, d = Q.shape
    S = K.shape[2]
    output = torch.empty((B, H, L, d), device=Q.device, dtype=Q.dtype)
    # Tile sizes (tune these for your GPU)
    BLOCK_M = 16  # queries per tile
    BLOCK_N = 64  # keys per tile
    # Grid dimensions: (B, H, ceil(L / BLOCK_M))
    grid = (B, H, (L + BLOCK_M - 1) // BLOCK_M)
    fused_attention_kernel[grid](
        Q, K, V, output,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        L, S, d,
        BLOCK_M, BLOCK_N
    )
    return output


class MultiheadLatentAttentionTriton(nn.Module):
    """
    Multi-head latent attention using a highly optimized Triton kernel with fused softmax for the attention computation.

    Output shape: (B, L_latent, latent_dim)
    """
    def __init__(self, input_dim, latent_dim, num_heads, latent_rank, dropout=0.0):
        """
        Args:
            input_dim: Dimensionality of input features.
            latent_dim: Dimensionality of latent space (and output); must be divisible by num_heads.
            num_heads: Number of attention heads.
            latent_rank: Intermediate rank for low-rank compression.
            dropout: Dropout probability for attention weights.
        """
        super(MultiheadLatentAttentionTriton, self).__init__()
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.head_dim = latent_dim // num_heads
        assert self.head_dim * num_heads == latent_dim, "latent_dim must be divisible by num_heads"

        # Compute queries from latent tokens.
        self.query_proj = nn.Linear(latent_dim, latent_dim)
        # Compress input x (shape: (B, S, input_dim)) to a lower dimension.
        self.compress = nn.Linear(input_dim, latent_rank)
        # Decompress into key and value representations.
        self.decompress_k = nn.Linear(latent_rank, self.head_dim * num_heads)
        self.decompress_v = nn.Linear(latent_rank, self.head_dim * num_heads)
        # Final projection.
        self.out_proj = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, latent, x, mask=None):
        """
        Args:
            latent: Tensor of shape (B, L_latent, latent_dim) used to compute queries.
            x: Tensor of shape (B, S, input_dim) used to compute keys and values.
            mask: (Optional) Attention mask.

        Returns:
            output: Tensor of shape (B, L_latent, latent_dim).
            attn_weights: (None for Triton version)
        """
        B, L_latent, _ = latent.size()
        B, S, _ = x.size()

        # Compute queries.
        Q = self.query_proj(latent)  # (B, L_latent, latent_dim)
        # Compress x and decompress into keys and values.
        compressed = self.compress(x)  # (B, S, latent_rank)
        K_full = self.decompress_k(compressed)  # (B, S, num_heads * head_dim)
        V_full = self.decompress_v(compressed)  # (B, S, num_heads * head_dim)

        # Reshape into (B, num_heads, S, head_dim)
        K = K_full.view(B, S, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        V = V_full.view(B, S, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        # Reshape queries into (B, num_heads, L_latent, head_dim)
        Q = Q.view(B, L_latent, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

        # Call the advanced Triton fused attention kernel.
        attn_output = attention_forward_triton(Q, K, V)  # (B, num_heads, L_latent, head_dim)

        # Concatenate heads and apply final projection.
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L_latent, self.latent_dim)
        output = self.out_proj(attn_output)
        return output, None
