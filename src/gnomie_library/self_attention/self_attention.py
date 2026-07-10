import numpy as np

class SelfAttention:
    """A basic implementation of scaled dot-product self-attention."""

    def __init__(self, d_k, d_v):
        """Initializes the SelfAttention module.
        
        Args:
            d_k (int): Dimension of the key vectors.
            d_v (int): Dimension of the value vectors.
        """
        self.d_k = d_k
        self.d_v = d_v

    def __call__(self, queries, keys, values):
        """Computes self-attention.
        
        Args:
            queries (np.ndarray): Query matrix of shape (..., seq_len, d_k).
            keys (np.ndarray): Key matrix of shape (..., seq_len, d_k).
            values (np.ndarray): Value matrix of shape (..., seq_len, d_v).
            
        Returns:
            np.ndarray: The attended output of shape (..., seq_len, d_v).
        """
        return self.forward(queries, keys, values)

    def forward(self, queries, keys, values):
        """Forward pass of the self-attention mechanism.
        
        Args:
            queries (np.ndarray): Query matrix.
            keys (np.ndarray): Key matrix.
            values (np.ndarray): Value matrix.
            
        Returns:
            np.ndarray: The attention output.
        """
        queries = np.asarray(queries, dtype=np.float64)
        keys = np.asarray(keys, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)

        if queries.ndim < 2 or keys.ndim < 2 or values.ndim < 2:
            raise ValueError(
                f"queries, keys, and values must have at least 2 dimensions. "
                f"Got shapes: queries={queries.shape}, keys={keys.shape}, values={values.shape}"
            )

        if queries.shape[-1] != self.d_k:
            raise ValueError(f"queries last dimension must match d_k ({self.d_k}), got {queries.shape[-1]}")
        if keys.shape[-1] != self.d_k:
            raise ValueError(f"keys last dimension must match d_k ({self.d_k}), got {keys.shape[-1]}")
        if values.shape[-1] != self.d_v:
            raise ValueError(f"values last dimension must match d_v ({self.d_v}), got {values.shape[-1]}")

        if queries.shape[:-2] != keys.shape[:-2] or queries.shape[:-2] != values.shape[:-2]:
            raise ValueError(
                f"Batch dimensions of queries, keys, and values must match. "
                f"Got: queries={queries.shape[:-2]}, keys={keys.shape[:-2]}, values={values.shape[:-2]}"
            )

        if keys.shape[-2] != values.shape[-2]:
            raise ValueError(
                f"Sequence length of keys and values must match. "
                f"Got: keys_seq_len={keys.shape[-2]}, values_seq_len={values.shape[-2]}"
            )

        keys_t = keys.swapaxes(-1, -2)
        attention_scores = np.matmul(queries, keys_t) / np.sqrt(self.d_k)
        attention_probs = self._softmax(attention_scores)
        output = np.matmul(attention_probs, values)
        return output

    def _softmax(self, x):
        """Computes softmax along the last axis.
        
        Args:
            x (np.ndarray): Input array.
            
        Returns:
            np.ndarray: Softmax normalized array.
        """
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)