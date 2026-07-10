from abc import ABC, abstractmethod
import numpy as np

class BaseCamera(ABC):
    """
    Abstract base class representing a camera model.
    Handles camera extrinsic parameters (rotation, translation) and coordinate transformations.
    """

    def __init__(self, rotation: np.ndarray = None, translation: np.ndarray = None):
        """
        Initializes a camera.

        Args:
            rotation (np.ndarray, optional): 3x3 rotation matrix. Defaults to Identity.
            translation (np.ndarray, optional): 3x1 (or 3-element) translation vector. Defaults to Zeros.
        """
        if rotation is None:
            self._rotation = np.eye(3, dtype=np.float64)
        else:
            self.rotation = rotation  # Use setter for validation

        if translation is None:
            self._translation = np.zeros((3, 1), dtype=np.float64)
        else:
            self.translation = translation  # Use setter for validation

    @property
    def rotation(self) -> np.ndarray:
        """Get the 3x3 rotation matrix."""
        return self._rotation

    @rotation.setter
    def rotation(self, R: np.ndarray) -> None:
        """Set the 3x3 rotation matrix."""
        R = np.asarray(R, dtype=np.float64)
        if R.shape != (3, 3):
            raise ValueError(f"Rotation matrix must have shape (3, 3), got {R.shape}")
        self._rotation = R

    @property
    def translation(self) -> np.ndarray:
        """Get the 3x1 translation vector."""
        return self._translation

    @translation.setter
    def translation(self, t: np.ndarray) -> None:
        """Set the 3x1 or 3-element translation vector."""
        t = np.asarray(t, dtype=np.float64)
        if t.size != 3:
            raise ValueError(f"Translation vector must have 3 elements, got {t.size}")
        self._translation = t.reshape(3, 1)

    @property
    def extrinsic_matrix(self) -> np.ndarray:
        """Get the 4x4 extrinsic transformation matrix (world-to-camera, [R | t])."""
        ext = np.eye(4, dtype=np.float64)
        ext[:3, :3] = self._rotation
        ext[:3, 3:4] = self._translation
        return ext

    @extrinsic_matrix.setter
    def extrinsic_matrix(self, T: np.ndarray) -> None:
        """Set the rotation and translation from a 4x4 extrinsic matrix."""
        T = np.asarray(T, dtype=np.float64)
        if T.shape != (4, 4):
            raise ValueError(f"Extrinsic matrix must have shape (4, 4), got {T.shape}")
        self.rotation = T[:3, :3]
        self.translation = T[:3, 3]

    def world_to_camera(self, points_world: np.ndarray) -> np.ndarray:
        """
        Transforms 3D points from world coordinates to camera coordinates:
        X_c = R * X_w + t

        Args:
            points_world (np.ndarray): Array of shape (N, 3) or (3,) containing 3D points.

        Returns:
            np.ndarray: Array of shape (N, 3) or (3,) containing camera-frame 3D points.
        """
        points_world = np.asarray(points_world, dtype=np.float64)
        is_single = points_world.ndim == 1
        if is_single:
            if points_world.shape != (3,):
                raise ValueError(f"Single point must have shape (3,), got {points_world.shape}")
            points_world = points_world[np.newaxis, :]
        elif points_world.ndim != 2 or points_world.shape[1] != 3:
            raise ValueError(f"Points must have shape (N, 3), got {points_world.shape}")

        points_camera = points_world @ self._rotation.T + self._translation.T

        return points_camera[0] if is_single else points_camera

    def camera_to_world(self, points_camera: np.ndarray) -> np.ndarray:
        """
        Transforms 3D points from camera coordinates to world coordinates:
        X_w = R^T * (X_c - t)

        Args:
            points_camera (np.ndarray): Array of shape (N, 3) or (3,) containing camera-frame 3D points.

        Returns:
            np.ndarray: Array of shape (N, 3) or (3,) containing world-frame 3D points.
        """
        points_camera = np.asarray(points_camera, dtype=np.float64)
        is_single = points_camera.ndim == 1
        if is_single:
            if points_camera.shape != (3,):
                raise ValueError(f"Single point must have shape (3,), got {points_camera.shape}")
            points_camera = points_camera[np.newaxis, :]
        elif points_camera.ndim != 2 or points_camera.shape[1] != 3:
            raise ValueError(f"Points must have shape (N, 3), got {points_camera.shape}")

        points_world = (points_camera - self._translation.T) @ self._rotation

        return points_world[0] if is_single else points_world

    @abstractmethod
    def project(self, points_3d: np.ndarray, in_world_coords: bool = True) -> np.ndarray:
        """
        Projects 3D points to 2D image plane coordinates.

        Args:
            points_3d (np.ndarray): Array of shape (N, 3) or (3,) containing 3D points.
            in_world_coords (bool, optional): Whether the points are in the world coordinates
                                              or camera coordinates. Defaults to True.

        Returns:
            np.ndarray: Array of shape (N, 2) or (2,) containing 2D image coordinates (u, v).
        """
        pass

    @abstractmethod
    def unproject(self, points_2d: np.ndarray, depths: np.ndarray, in_world_coords: bool = True) -> np.ndarray:
        """
        Unprojects 2D image coordinates and depths back to 3D.

        Args:
            points_2d (np.ndarray): Array of shape (N, 2) or (2,) containing 2D image coordinates (u, v).
            depths (np.ndarray): Array of shape (N,) or float containing depth values.
            in_world_coords (bool, optional): Whether to return the 3D points in world coordinates
                                              or camera coordinates. Defaults to True.

        Returns:
            np.ndarray: Array of shape (N, 3) or (3,) containing 3D points.
        """
        pass
