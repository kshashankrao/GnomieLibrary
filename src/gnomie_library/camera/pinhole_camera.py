import numpy as np
import cv2
from .base_camera import BaseCamera

class PinholeCamera(BaseCamera):
    """
    Pinhole camera model implementing project and unproject methods.
    Supports radial and tangential distortion coefficients.
    """

    def __init__(self,
                 intrinsic_matrix: np.ndarray,
                 distortion_coefficients: np.ndarray = None,
                 rotation: np.ndarray = None,
                 translation: np.ndarray = None):
        """
        Initializes the pinhole camera.

        Args:
            intrinsic_matrix (np.ndarray): 3x3 intrinsic matrix K.
            distortion_coefficients (np.ndarray, optional): Array of distortion coefficients
                [k1, k2, p1, p2, k3, ...]. Defaults to None (no distortion).
            rotation (np.ndarray, optional): 3x3 rotation matrix. Defaults to Identity.
            translation (np.ndarray, optional): 3x1 translation vector. Defaults to Zeros.
        """
        super().__init__(rotation, translation)
        self.intrinsic_matrix = intrinsic_matrix  # Use setter for validation
        self.distortion_coefficients = distortion_coefficients  # Use setter for validation

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        """Get the 3x3 intrinsic matrix."""
        return self._intrinsic_matrix

    @intrinsic_matrix.setter
    def intrinsic_matrix(self, K: np.ndarray) -> None:
        """Set the 3x3 intrinsic matrix."""
        K = np.asarray(K, dtype=np.float64)
        if K.shape != (3, 3):
            raise ValueError(f"Intrinsic matrix must have shape (3, 3), got {K.shape}")
        self._intrinsic_matrix = K

    @property
    def distortion_coefficients(self) -> np.ndarray:
        """Get the distortion coefficients array."""
        return self._distortion_coefficients

    @distortion_coefficients.setter
    def distortion_coefficients(self, dist_coeffs: np.ndarray) -> None:
        """Set the distortion coefficients array."""
        if dist_coeffs is None:
            self._distortion_coefficients = None
        else:
            self._distortion_coefficients = np.asarray(dist_coeffs, dtype=np.float64).flatten()

    @property
    def focal_lengths(self) -> tuple:
        """Returns the focal lengths (fx, fy)."""
        return float(self._intrinsic_matrix[0, 0]), float(self._intrinsic_matrix[1, 1])

    @property
    def principal_point(self) -> tuple:
        """Returns the principal point (cx, cy)."""
        return float(self._intrinsic_matrix[0, 2]), float(self._intrinsic_matrix[1, 2])

    def project(self, points_3d: np.ndarray, in_world_coords: bool = True) -> np.ndarray:
        """
        Projects 3D points to 2D image plane coordinates.

        Args:
            points_3d (np.ndarray): Array of shape (N, 3) or (3,) containing 3D points.
            in_world_coords (bool, optional): Whether points are in world coords (True)
                                              or camera coords (False). Defaults to True.

        Returns:
            np.ndarray: Array of shape (N, 2) or (2,) containing 2D image coordinates (u, v).
        """
        pts_3d = np.asarray(points_3d, dtype=np.float64)
        is_single = pts_3d.ndim == 1
        if is_single:
            if pts_3d.shape != (3,):
                raise ValueError(f"Single 3D point must have shape (3,), got {pts_3d.shape}")
            pts_3d = pts_3d[np.newaxis, :]
        elif pts_3d.ndim != 2 or pts_3d.shape[1] != 3:
            raise ValueError(f"3D points must have shape (N, 3), got {pts_3d.shape}")

        if in_world_coords:
            pts_camera = self.world_to_camera(pts_3d)
        else:
            pts_camera = pts_3d

        rvec = np.zeros(3, dtype=np.float64)
        tvec = np.zeros(3, dtype=np.float64)

        # cv2.projectPoints takes points as (N, 1, 3) or (N, 3)
        image_points, _ = cv2.projectPoints(
            pts_camera,
            rvec,
            tvec,
            self._intrinsic_matrix,
            self._distortion_coefficients
        )
        projected = image_points.reshape(-1, 2)

        return projected[0] if is_single else projected

    def unproject(self, points_2d: np.ndarray, depths: np.ndarray, in_world_coords: bool = True) -> np.ndarray:
        """
        Unprojects 2D image coordinates and depths back to 3D.

        Args:
            points_2d (np.ndarray): Array of shape (N, 2) or (2,) containing 2D image coordinates (u, v).
            depths (np.ndarray): Array of shape (N,) or a scalar depth value.
            in_world_coords (bool, optional): Whether to return 3D points in world coords (True)
                                              or camera coords (False). Defaults to True.

        Returns:
            np.ndarray: Array of shape (N, 3) or (3,) containing 3D points.
        """
        pts_2d = np.asarray(points_2d, dtype=np.float64)
        is_single_pt = pts_2d.ndim == 1
        if is_single_pt:
            if pts_2d.shape != (2,):
                raise ValueError(f"Single 2D point must have shape (2,), got {pts_2d.shape}")
            pts_2d = pts_2d[np.newaxis, :]
        elif pts_2d.ndim != 2 or pts_2d.shape[1] != 2:
            raise ValueError(f"2D points must have shape (N, 2), got {pts_2d.shape}")

        # Handle depths shape
        if np.isscalar(depths):
            depths_arr = np.full(len(pts_2d), depths, dtype=np.float64)
            is_single_depth = True
        else:
            depths_arr = np.asarray(depths, dtype=np.float64).flatten()
            if len(depths_arr) != len(pts_2d):
                raise ValueError(f"Depths size ({len(depths_arr)}) must match points count ({len(pts_2d)})")
            is_single_depth = False

        # Undistort points to normalized image coordinates (x', y')
        if self._distortion_coefficients is not None and len(self._distortion_coefficients) > 0:
            pts_2d_reshaped = pts_2d[:, np.newaxis, :]
            undistorted_norm = cv2.undistortPoints(
                pts_2d_reshaped,
                self._intrinsic_matrix,
                self._distortion_coefficients
            )
            norm_x = undistorted_norm[:, 0, 0]
            norm_y = undistorted_norm[:, 0, 1]
        else:
            # Linear back-projection
            fx, fy = self.focal_lengths
            cx, cy = self.principal_point
            norm_x = (pts_2d[:, 0] - cx) / fx
            norm_y = (pts_2d[:, 1] - cy) / fy

        # Build 3D points in camera coordinates
        pts_camera = np.zeros((len(pts_2d), 3), dtype=np.float64)
        pts_camera[:, 0] = norm_x * depths_arr
        pts_camera[:, 1] = norm_y * depths_arr
        pts_camera[:, 2] = depths_arr

        # Transform to world coordinates if requested
        if in_world_coords:
            pts_3d = self.camera_to_world(pts_camera)
        else:
            pts_3d = pts_camera

        if is_single_pt and (is_single_depth or np.isscalar(depths)):
            return pts_3d[0]
        return pts_3d
