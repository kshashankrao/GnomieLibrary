import numpy as np

class SelfAttention:
    """A basic implementation of self-attention."""

    def __init__(self, d_k, d_v):
        """Initializes the SelfAttention module."""
        self.d_k = d_k
        self.d_v = d_v

    def __call__(self, queries, keys, values):
        """Computes self-attention."""
        return self.forward(queries, keys, values)

    def forward(self, queries, keys, values):
        """Forward pass of the self-attention mechanism."""
        attention_scores = np.matmul(queries, keys.T) / np.sqrt(self.d_k)
        attention_probs = self._softmax(attention_scores)
        output = np.matmul(attention_probs, values)
        return output

    def _softmax(self, x):
        """Computes softmax along the last axis."""
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)