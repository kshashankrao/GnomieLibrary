import numpy as np

class EightPointAlgorithm:
    """
    Implements the eight-point algorithm for fundamental matrix estimation.
    """

    def __init__(self):
        """Initializes the EightPointAlgorithm."""
        pass  # No initialization needed for this algorithm

    def __call__(self, pts1, pts2, M=None):
        """
        Computes the fundamental matrix using the eight-point algorithm.

        Args:
            pts1 (numpy.ndarray): Array of points from the first image (n x 2) or (n x 3).
            pts2 (numpy.ndarray): Array of corresponding points from the second image (n x 2) or (n x 3).
            M (float, optional): Normalization factor. Defaults to None (computed automatically).

        Returns:
            numpy.ndarray: The estimated fundamental matrix (3 x 3).
        """
        return self.compute_fundamental_matrix(pts1, pts2, M)

    def compute_fundamental_matrix(self, pts1, pts2, M=None):
        """
        Computes the fundamental matrix.
        """
        pts1 = np.asarray(pts1, dtype=np.float64)
        pts2 = np.asarray(pts2, dtype=np.float64)

        if pts1.shape != pts2.shape:
            raise ValueError(f"Point correspondences must have identical shapes, got {pts1.shape} and {pts2.shape}")
        if pts1.ndim != 2 or pts1.shape[1] not in (2, 3):
            raise ValueError(f"Points must have shape (N, 2) or (N, 3), got {pts1.shape}")
        if pts1.shape[0] < 8:
            raise ValueError(f"Eight-point algorithm requires at least 8 correspondences, got {pts1.shape[0]}")

        # Convert homogeneous coordinates to Cartesian if necessary
        if pts1.shape[1] == 3:
            z1 = pts1[:, 2:3]
            z2 = pts2[:, 2:3]
            pts1 = pts1[:, :2] / np.where(z1 != 0, z1, 1e-10)
            pts2 = pts2[:, :2] / np.where(z2 != 0, z2, 1e-10)

        if M is None:
            M = float(np.max([np.max(np.abs(pts1)), np.max(np.abs(pts2)), 1.0]))

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