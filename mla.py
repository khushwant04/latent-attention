import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------
# Original Multihead Latent Attention Module
# -----------------------------------------
class MultiheadLatentAttention(nn.Module):
    """
    A multi-head latent attention implementation using standard PyTorch ops.
    
    Here queries are computed from the latent tensor and keys/values are produced
    by first compressing the input and then decompressing for each head.
    
    Output shape: (B, L_latent, latent_dim)
    """
    def __init__(self, input_dim, latent_dim, num_heads, latent_rank, dropout=0.0):
        """
        Args:
            input_dim: Dimensionality of the input features.
            latent_dim: Dimensionality of latent space (and output); must be divisible by num_heads.
            num_heads: Number of attention heads.
            latent_rank: Intermediate rank for low-rank compression (latent_rank < latent_dim).
            dropout: Dropout probability applied on attention weights.
        """
        super(MultiheadLatentAttention, self).__init__()
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.head_dim = latent_dim // num_heads
        assert self.head_dim * num_heads == latent_dim, "latent_dim must be divisible by num_heads"

        # Compute queries from latent tokens.
        self.query_proj = nn.Linear(latent_dim, latent_dim)
        
        # Compress input x (shape: (B, S, input_dim)) to a lower dimension (latent_rank)
        self.compress = nn.Linear(input_dim, latent_rank)
        # Then decompress into key and value representations for each head.
        self.decompress_k = nn.Linear(latent_rank, self.head_dim * num_heads)
        self.decompress_v = nn.Linear(latent_rank, self.head_dim * num_heads)
        
        # Final output projection to bring concatenated heads back to latent_dim.
        self.out_proj = nn.Linear(latent_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, latent, x, mask=None):
        """
        Args:
            latent: Tensor of shape (B, L_latent, latent_dim) from which queries are computed.
            x: Tensor of shape (B, S, input_dim) used to compute keys and values.
            mask: Optional mask broadcastable to (B, num_heads, L_latent, S).
        
        Returns:
            output: Tensor of shape (B, L_latent, latent_dim).
            attn_weights: Tensor of shape (B, num_heads, L_latent, S).
        """
        B, L_latent, _ = latent.size()
        B, S, _ = x.size()

        # Compute queries
        Q = self.query_proj(latent)  # (B, L_latent, latent_dim)
        # Compress x and then decompress into keys and values.
        compressed = self.compress(x)  # (B, S, latent_rank)
        K_full = self.decompress_k(compressed)  # (B, S, num_heads * head_dim)
        V_full = self.decompress_v(compressed)  # (B, S, num_heads * head_dim)
        
        # Reshape into (B, num_heads, S, head_dim)
        K = K_full.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        V = V_full.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        # Reshape queries to (B, num_heads, L_latent, head_dim)
        Q = Q.view(B, L_latent, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)  # (B, num_heads, L_latent, head_dim)
        
        # Concatenate heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L_latent, self.latent_dim)
        output = self.out_proj(attn_output)
        return output, attn_weights
