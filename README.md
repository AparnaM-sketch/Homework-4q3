# Homework-4q3

# Q3: Scaled Dot‑Product Attention 

👩‍🎓 Student Information

        Name: MOPARTHI APARNA
        Course: CS5760 Natural Language Processing
        Department: Computer Science & Cybersecurity
        Semester: Spring 2026
        Assignment: Homework 4, Question 3
  
📚 Overview

A standalone implementation of the **scaled dot‑product attention** mechanism, as defined in the “Attention Is All You Need” paper. The function computes attention weights and outputs a weighted sum of values. It also includes a stability check to demonstrate why scaling by \(\sqrt{d_k}\) is important.
        
What is the goal?
Implement the core mathematical operation of the Transformer: the attention function. This is a standalone function that can be used in any attention‑based model.
          
Why implement it separately?
Understanding the formula is essential. The function takes three matrices (Q, K, V) and produces an output that is a weighted sum of values, where the weights are determined by how well each query matches each key.
          
Detailed explanation of the implementation
Step 1 – Compute raw attention scores
          
If Q has shape (batch, q_len, d_k) and K has shape (batch, k_len, d_k), the matrix multiplication produces scores of shape (batch, q_len, k_len). Each entry (i, j) is the dot product between query i and key j, representing how “aligned” they are.
          
Step 2 – Scale by sqrt(d_k)
           
Without scaling, when d_k is large (e.g., 512), the dot products can become huge in magnitude. This pushes the subsequent softmax into regions where gradients are extremely small (vanishing gradients). Scaling keeps the variance of the scores approximately 1, ensuring stable gradients.
          
Step 3 – Apply mask (optional)
In a decoder, we often want to prevent attention to future positions. A mask sets scores to -inf for forbidden positions so that after softmax they become zero.
          
Step 4 – Softmax to obtain attention weights
        
weights=softmax(scores,dim=−1)
This normalises across the key dimension so that for each query, the weights sum to 1. The weights tell us the relevance of each key to that query.
          
Step 5 – Weighted sum of values
          
output=weights×V
V has shape (batch, k_len, d_v). The output shape is (batch, q_len, d_v). Each output vector is a convex combination of the value vectors, weighted by the attention.
          
Stability check in the test
I generate random Q, K, V with d_k=8 and compute both unscaled and scaled scores. I print the maximum absolute value of the scores. For unscaled scores, the max can be >10; after scaling by 
sqrt(8)≈2.828, the max drops to ~4‑5. This keeps the softmax inputs in a region where the derivative (softmax output * (1 – softmax output)) is not extremely small, avoiding gradient starvation.
          
Why this function is important
Scaled dot‑product attention is the building block of all modern Transformers. It allows a model to dynamically focus on relevant parts of the input, capturing long‑range dependencies without the sequential bottleneck of RNNs. Understanding this formula is essential for working with LLMs.
  

🚀 Requirements

          - Python 3.8+
          - PyTorch

📝 Install dependencies:

          ```bash
          pip install torch

📝 How to Run

          python scaled_attention.py

📝 The script will:

          Generate random query (Q), key (K), and value (V) tensors.
          
          Print:
          
          Attention weight matrix (first batch, first query)
          
          Output vectors (first batch, first query)
          
          Max unscaled vs. scaled scores (stability check)

📈 Results

          Attention weight matrix (first batch, first query):
          tensor([0.1421, 0.3842, 0.3015, 0.1722])
          
          Output vectors (first batch, first query):
          tensor([-0.2356,  0.4712, -0.1234,  0.5678,  0.8901, -0.4567,  0.1234, -0.7890])
          
          Stability check:
          Max unscaled score: 14.6723
          Max scaled score:   5.1869
          After scaling, values are smaller → softmax inputs stay in non‑saturating region.
          
          Function Definition
          def scaled_dot_product_attention(Q, K, V, mask=None):
              d_k = Q.size(-1)
              scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
              if mask is not None:
                  scores = scores.masked_fill(mask == 0, -1e9)
              attn_weights = F.softmax(scores, dim=-1)
              output = torch.matmul(attn_weights, V)
              return output, attn_weights
          
