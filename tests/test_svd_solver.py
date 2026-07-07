import numpy as np
import pytest
from gnomie_library import SVDSolver


def test_linear_regression_svd():
    """Test that SVD linear regression accurately recovers slope and intercept."""
    np.random.seed(42)
    # y = 3 * x1 - 2 * x2 + 5
    X = np.random.rand(100, 2)
    true_beta = np.array([3.0, -2.0, 5.0])
    y = X @ true_beta[:2] + true_beta[2] + np.random.normal(0, 1e-10, size=100)
    
    beta_hat = SVDSolver.solve_linear_regression(X, y, add_bias=True)
    np.testing.assert_allclose(beta_hat, true_beta, atol=1e-5)


def test_icp_alignment_svd():
    """Test that SVD ICP alignment accurately recovers rigid 3D rotation and translation without scaling distortion."""
    np.random.seed(42)
    # Generate random 3D point cloud P
    P = np.random.rand(50, 3) * 10.0
    
    # Define rigid 90 degree rotation around Z axis and translation t
    theta = np.pi / 2
    R_true = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,             0,              1]
    ])
    t_true = np.array([2.5, -1.3, 4.2])
    
    # Target cloud Q = R * P + t
    Q = (R_true @ P.T).T + t_true
    
    R_hat, t_hat = SVDSolver.compute_icp_alignment(P, Q)
    
    # Assert orthogonal R in SO(3)
    np.testing.assert_allclose(R_hat.T @ R_hat, np.eye(3), atol=1e-6)
    np.testing.assert_almost_equal(np.linalg.det(R_hat), 1.0)
    
    # Assert exact rotation and translation recovery
    np.testing.assert_allclose(R_hat, R_true, atol=1e-5)
    np.testing.assert_allclose(t_hat, t_true, atol=1e-5)
