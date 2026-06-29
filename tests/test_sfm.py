import unittest
import numpy as np
from gnomie_library import EightPointAlgorithm

class TestEightPointAlgorithm(unittest.TestCase):
    def test_fundamental_matrix_shape_and_rank(self):
        # Create some dummy points
        pts1 = np.random.rand(10, 2) * 100
        pts2 = np.random.rand(10, 2) * 100
        M = 100.0
        
        sfm = EightPointAlgorithm()
        F = sfm(pts1, pts2, M)
        
        # Fundamental matrix should be 3x3
        self.assertEqual(F.shape, (3, 3))
        
        # Rank of Fundamental matrix should be 2
        u, s, vh = np.linalg.svd(F)
        # The smallest singular value should be close to 0
        self.assertAlmostEqual(s[-1], 0.0, places=5)

if __name__ == '__main__':
    unittest.main()
