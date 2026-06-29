# Self Attention

A basic implementation of scaled dot-product self-attention.

## Example Usage

```python
import numpy as np
from gnomie_library import SelfAttention

# Create dummy input data
embeddings = np.array([
    [0.1, 0.2, 0.3, 0.4],
    [0.8, 0.5, 0.1, 0.6]
])

# Initialize weights (for demonstration)
W_q = np.random.rand(4, 4)
W_k = np.random.rand(4, 4)
W_v = np.random.rand(4, 4)

queries = np.matmul(embeddings, W_q)
keys = np.matmul(embeddings, W_k)
values = np.matmul(embeddings, W_v)

# Initialize Self-Attention module
attention = SelfAttention(d_k=4, d_v=4)
output = attention(queries, keys, values)

print("Attention Output Shape:", output.shape)
```
