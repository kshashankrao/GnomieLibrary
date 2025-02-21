import numpy as np

class EightPointAlgorithm:
    """
    Implements the eight-point algorithm for fundamental matrix estimation.
    """

    def __init__(self):
        """Initializes the EightPointAlgorithm."""
        pass  # No initialization needed for this algorithm

    def __call__(self, pts1, pts2, M):
        """
        Computes the fundamental matrix using the eight-point algorithm.

        Args:
            pts1 (numpy.ndarray): Array of points from the first image (n x 2).
            pts2 (numpy.ndarray): Array of corresponding points from the second image (n x 2).
            M (float): Normalization factor (max(image_height, image_width)).

        Returns:
            numpy.ndarray: The estimated fundamental matrix (3 x 3).
        """
        return self.compute_fundamental_matrix(pts1, pts2, M)

    def compute_fundamental_matrix(self, pts1, pts2, M):
        """
        Computes the fundamental matrix.

        Algorithm:

        1. Normalize the matched points pts/M
        2. Ax = 0
            Given - A
            To find - x
            where,
            A is matrix consisting of combination of the points
            x consists of elements of fundamental matrix
            x is solved using SVD = F_
        3. Constraint the F_ to rank 2
        4. Unnormalize the Fundamental matrix
        """

        # Create a transformation matrix
        T = np.eye(3, 3)
        T = T / M
        T[2, 2] = 1

        # Normalize the points
        pts1 = pts1 / M
        pts2 = pts2 / M

        n = len(pts1)
        A = np.zeros((n, 9))

        for i in range(n):
            A[i, 0] = pts1[i, 0] * pts2[i, 0]
            A[i, 1] = pts1[i, 0] * pts2[i, 1]
            A[i, 2] = pts1[i, 0]
            A[i, 3] = pts1[i, 1] * pts2[i, 0]
            A[i, 4] = pts1[i, 1] * pts2[i, 1]
            A[i, 5] = pts1[i, 1]
            A[i, 6] = pts2[i, 0]
            A[i, 7] = pts2[i, 1]
            A[i, 8] = 1

        _, _, v_ = np.linalg.svd(A)

        # F_ corresponds to the least singular value
        f_ = v_[-1].reshape(3, 3)

        # Apply rank 2 constraint
        u, s, v = np.linalg.svd(f_)
        s[2] = 0
        f = np.dot(u, np.dot(np.diag(s), v))

        # Unnormalize the matrix
        f = np.dot(np.transpose(T), np.dot(f, T))

        return f