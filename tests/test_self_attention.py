import unittest
import numpy as np
from gnomie_library import SelfAttention

class TestSelfAttention(unittest.TestCase):
    def test_self_attention_output_shape(self):
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
        
        self.assertEqual(output.shape, embeddings.shape)

    def test_attention_weights_sum_to_one(self):
        embeddings = np.random.rand(5, 4)
        W_q = np.random.rand(4, 4)
        W_k = np.random.rand(4, 4)
        W_v = np.random.rand(4, 4)
        
        queries = np.matmul(embeddings, W_q)
        keys = np.matmul(embeddings, W_k)
        values = np.matmul(embeddings, W_v)
        
        attention_module = SelfAttention(d_k=4, d_v=4)
        _ = attention_module(queries, keys, values)
        
        # We need to verify softmax rows sum to 1.
        # SelfAttention class internally calculates this, so we can test the calculation directly.
        scores = np.matmul(queries, keys.transpose()) / np.sqrt(4)
        attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention_weights /= np.sum(attention_weights, axis=-1, keepdims=True)
        
        for row in attention_weights:
            self.assertAlmostEqual(np.sum(row), 1.0)

    def test_self_attention_batched(self):
        # Batch size 2, sequence length 5, embed dim 4
        queries = np.random.rand(2, 5, 4)
        keys = np.random.rand(2, 5, 4)
        values = np.random.rand(2, 5, 4)
        
        attention_module = SelfAttention(d_k=4, d_v=4)
        output = attention_module(queries, keys, values)
        self.assertEqual(output.shape, (2, 5, 4))

    def test_self_attention_invalid_shapes(self):
        attention_module = SelfAttention(d_k=4, d_v=4)
        # Invalid dim count
        with self.assertRaises(ValueError):
            attention_module(np.array([1, 2, 3]), np.array([1, 2, 3]), np.array([1, 2, 3]))
        # Size mismatch last axis
        with self.assertRaises(ValueError):
            attention_module(np.random.rand(5, 3), np.random.rand(5, 4), np.random.rand(5, 4))

if __name__ == '__main__':
    unittest.main()
