import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: tensors of shape (..., seq_len_q, d_k), (..., seq_len_k, d_k), (..., seq_len_k, d_v)
    Returns: output (..., seq_len_q, d_v), attention weights (..., seq_len_q, seq_len_k)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output, attn_weights

# Test with random inputs
torch.manual_seed(42)
Q = torch.randn(2, 3, 8)   # batch=2, query_len=3, d_k=8
K = torch.randn(2, 4, 8)   # key_len=4
V = torch.randn(2, 4, 8)

out, attn = scaled_dot_product_attention(Q, K, V)

print("Attention weight matrix (first batch, first query):")
print(attn[0, 0, :])
print("\nOutput vectors (first batch, first query):")
print(out[0, 0, :])

# Softmax stability check
scores_unscaled = torch.matmul(Q, K.transpose(-2, -1))
scores_scaled = scores_unscaled / (8 ** 0.5)
print("\nStability check:")
print(f"Max unscaled score: {scores_unscaled.max().item():.4f}")
print(f"Max scaled score:   {scores_scaled.max().item():.4f}")
print("After scaling, values are smaller → softmax inputs stay in non‑saturating region.")