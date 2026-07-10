"""
Singular Value Decomposition (SVD) Equation & Alignment Solvers.

Provides robust numerical methods for:
1. Linear Regression (overdetermined systems y = X * beta) using SVD Moore-Penrose Pseudoinverse.
2. 3D Rigid Point Cloud Alignment (ICP / Procrustes Analysis) extracting purely rigid rotation R in SO(3)
   by intentionally discarding scaling matrix Sigma.
"""

import numpy as np


class SVDSolver:
    @staticmethod
    def solve_linear_regression(X: np.ndarray, y: np.ndarray, add_bias: bool = True) -> np.ndarray:
        """
        Solves linear regression y = X * beta using SVD Moore-Penrose Pseudoinverse.
        
        Why SVD over normal equation (X^T * X)^(-1) * X^T * y?
        In presence of collinear features or near-singular matrices, direct matrix inversion causes
        floating-point overflow and severe inaccuracies. SVD stably inverts singular values along each
        principal direction.
        
        Args:
            X: Input feature matrix of shape (N, K).
            y: Target vector of shape (N,) or (N, 1).
            add_bias: If True, appends a column of 1s to X to compute intercept c.
            
        Returns:
            beta: Parameter vector of shape (K + 1,) if add_bias is True, else (K,).
                  If add_bias is True, beta[-1] represents the intercept c.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(f"X must be a 2D matrix, got shape {X.shape}")
        if y.ndim not in (1, 2):
            raise ValueError(f"y must be a 1D vector or 2D column vector, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Number of samples in X ({X.shape[0]}) and y ({y.shape[0]}) must match.")

        if add_bias:
            n_samples = X.shape[0]
            X_design = np.hstack([X, np.ones((n_samples, 1))])
        else:
            X_design = X
            
        # 1. Compute SVD: X = U * Sigma * V^T
        U, S, Vt = np.linalg.svd(X_design, full_matrices=False)
        
        # 2. Invert singular values with numerical stability threshold
        tol = 1e-10 * np.max(S) if len(S) > 0 else 1e-10
        S_inv = np.where(S > tol, 1.0 / S, 0.0)
        
        # 3. Compute Moore-Penrose Pseudoinverse: X^+ = V * Sigma^+ * U^T
        X_pinv = Vt.T @ np.diag(S_inv) @ U.T
        
        # 4. Compute weights: beta = X^+ * y
        beta = X_pinv @ y
        return beta

    @staticmethod
    def compute_icp_alignment(P: np.ndarray, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes rigid 3D transformation (R, t) aligning source cloud P onto target Q using SVD.
        
        Key Principle:
        Because physical objects cannot stretch or shear during rigid 3D motion, the singular scaling
        matrix Sigma is completely discarded! Only the orthogonal rotation matrices are retained.
        
        Args:
            P: Source 3D point cloud of shape (N, 3).
            Q: Target 3D point cloud of shape (N, 3) matching points in P.
            
        Returns:
            R: 3x3 Rotation matrix in SO(3).
            t: 3D translation vector of shape (3,).
        """
        P = np.asarray(P, dtype=np.float64)
        Q = np.asarray(Q, dtype=np.float64)

        if P.shape != Q.shape:
            raise ValueError(f"Source and Target point clouds must have identical shapes, got {P.shape} and {Q.shape}")
        if P.ndim != 2 or P.shape[1] != 3:
            raise ValueError(f"Point clouds must be 2D arrays with shape (N, 3), got {P.shape}")
        if P.shape[0] < 3:
            raise ValueError(f"Need at least 3 points to compute 3D rigid alignment, got {P.shape[0]}")
        
        # 1. Compute centroids
        p_centroid = np.mean(P, axis=0)
        q_centroid = np.mean(Q, axis=0)
        
        # 2. Center point clouds at origin
        P_centered = P - p_centroid
        Q_centered = Q - q_centroid
        
        # 3. Compute 3x3 Cross-Covariance Matrix H
        H = P_centered.T @ Q_centered
        
        # 4. SVD of H: H = U * Sigma * V^T
        U, S, Vt = np.linalg.svd(H)
        
        # 5. Extract rigid rotation (Discarding Sigma to enforce physical isometry)
        R = Vt.T @ U.T
        
        # 6. Reflection correction (ensure right-handed coordinate system in SO(3))
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
            
        # 7. Recover translation
        t = q_centroid - R @ p_centroid
        
        return R, t
