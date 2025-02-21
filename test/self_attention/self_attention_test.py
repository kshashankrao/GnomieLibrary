import numpy as np
from self_attention import SelfAttention

embeddings = np.array([
    [0.1, 0.2, 0.3, 0.4],
    [0.8, 0.5, 0.1, 0.6],
    [0.3, 0.9, 0.6, 0.2],
    [0.5, 0.7, 0.4, 0.9],
    [0.7, 0.3, 0.8, 0.1]
])

np.random.seed(42)
W_q = np.random.rand(4, 4)
W_k = np.random.rand(4, 4)
W_v = np.random.rand(4, 4)

queries = np.matmul(embeddings, W_q)
keys = np.matmul(embeddings, W_k)
values = np.matmul(embeddings, W_v)

attention_module = SelfAttention(d_k=4, d_v=4)
output = attention_module(queries, keys, values)

print(output)