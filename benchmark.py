# benchmark.py
import torch
import time
from multihead_latent_attention import MultiheadLatentAttention
from multihead_latent_attention_triton import MultiheadLatentAttentionTriton

# Choose device.
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define larger parameters.
B = 128          # Increased batch size
S = 512          # Increased input sequence length
L_latent = 256   # Increased number of latent tokens
input_dim = 512  # Increased input feature dimension
latent_dim = 256 # Increased latent/output dimension (must be divisible by num_heads)
num_heads = 8
latent_rank = 128

# Initialize modules.
model_pytorch = MultiheadLatentAttention(input_dim, latent_dim, num_heads, latent_rank).to(device)
model_triton = MultiheadLatentAttentionTriton(input_dim, latent_dim, num_heads, latent_rank).to(device)

# Create random inputs.
x = torch.randn(B, S, input_dim, device=device)
latent = torch.randn(B, L_latent, latent_dim, device=device)

# Warm-up runs.
for _ in range(10):
    out, _ = model_pytorch(latent, x)
    torch.cuda.synchronize()
for _ in range(10):
    out, _ = model_triton(latent, x)
    torch.cuda.synchronize()

iters = 50

# Benchmark the PyTorch version.
start = time.time()
for _ in range(iters):
    out, _ = model_pytorch(latent, x)
    torch.cuda.synchronize()
end = time.time()
print("PyTorch Multihead Latent Attention average time: {:.6f} ms".format((end - start) * 1000 / iters))

# Benchmark the Triton kernel version.
start = time.time()
for _ in range(iters):
    out, _ = model_triton(latent, x)
    torch.cuda.synchronize()
end = time.time()
print("Triton Kernel Multihead Latent Attention average time: {:.6f} ms".format((end - start) * 1000 / iters))
