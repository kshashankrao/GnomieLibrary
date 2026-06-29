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
        attention_scores = np.matmul(queries, keys.T) / np.sqrt(self.d_k)
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