import unittest
import numpy as np
from gnomie_library import BaseCamera, PinholeCamera

class MockCamera(BaseCamera):
    """A minimal mock implementation of BaseCamera for testing base class functionality."""
    def project(self, points_3d: np.ndarray, in_world_coords: bool = True) -> np.ndarray:
        # Dummy projection returning x and y coordinates from 3D
        pts_camera = self.world_to_camera(points_3d) if in_world_coords else points_3d
        return pts_camera[..., :2]

    def unproject(self, points_2d: np.ndarray, depths: np.ndarray, in_world_coords: bool = True) -> np.ndarray:
        # Dummy unprojection
        pts_camera = np.hstack([points_2d, depths.reshape(-1, 1)])
        return self.camera_to_world(pts_camera) if in_world_coords else pts_camera


class TestBaseCameraExtrinsics(unittest.TestCase):
    def test_default_initialization(self):
        cam = MockCamera()
        np.testing.assert_array_equal(cam.rotation, np.eye(3))
        np.testing.assert_array_equal(cam.translation, np.zeros((3, 1)))

        expected_extrinsic = np.eye(4)
        np.testing.assert_array_equal(cam.extrinsic_matrix, expected_extrinsic)

    def test_custom_initialization(self):
        # 90 degrees rotation around Z axis
        R = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        t = np.array([1.0, 2.0, 3.0])
        cam = MockCamera(rotation=R, translation=t)

        np.testing.assert_array_equal(cam.rotation, R)
        np.testing.assert_array_equal(cam.translation, t.reshape(3, 1))

        # Check extrinsic matrix
        ext = cam.extrinsic_matrix
        np.testing.assert_array_equal(ext[:3, :3], R)
        np.testing.assert_array_equal(ext[:3, 3], t)

    def test_setters_and_validation(self):
        cam = MockCamera()

        # Valid setting
        R_new = np.diag([1.0, -1.0, 1.0])
        t_new = np.array([5.0, 6.0, 7.0])
        cam.rotation = R_new
        cam.translation = t_new

        np.testing.assert_array_equal(cam.rotation, R_new)
        np.testing.assert_array_equal(cam.translation, t_new.reshape(3, 1))

        # Invalid rotation shape
        with self.assertRaises(ValueError):
            cam.rotation = np.array([1, 2, 3])

        # Invalid translation shape
        with self.assertRaises(ValueError):
            cam.translation = np.eye(3)

    def test_extrinsic_matrix_setter(self):
        cam = MockCamera()
        T = np.eye(4)
        T[:3, :3] = np.diag([-1.0, 1.0, -1.0])
        T[:3, 3] = [10.0, 20.0, 30.0]

        cam.extrinsic_matrix = T
        np.testing.assert_array_equal(cam.rotation, T[:3, :3])
        np.testing.assert_array_equal(cam.translation, T[:3, 3].reshape(3, 1))

        # Invalid extrinsic shape
        with self.assertRaises(ValueError):
            cam.extrinsic_matrix = np.eye(3)

    def test_coordinate_transforms(self):
        # Rotate 90 deg around Z, translate by [1, 2, 3]
        # X_c = R * X_w + t
        R = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        t = np.array([1.0, 2.0, 3.0])
        cam = MockCamera(rotation=R, translation=t)

        pt_world = np.array([2.0, 1.0, 0.0])
        # R * [2, 1, 0]^T + t = [-1, 2, 0]^T + [1, 2, 3]^T = [0, 4, 3]^T
        expected_cam = np.array([0.0, 4.0, 3.0])

        # Single point
        pt_cam_res = cam.world_to_camera(pt_world)
        np.testing.assert_allclose(pt_cam_res, expected_cam)

        pt_world_res = cam.camera_to_world(pt_cam_res)
        np.testing.assert_allclose(pt_world_res, pt_world)

        # Batch points
        pts_world = np.array([
            [2.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [-1.0, -2.0, -3.0]
        ])
        pts_cam_res = cam.world_to_camera(pts_world)
        self.assertEqual(pts_cam_res.shape, (3, 3))

        # Round trip
        pts_world_res = cam.camera_to_world(pts_cam_res)
        np.testing.assert_allclose(pts_world_res, pts_world)


class TestPinholeCamera(unittest.TestCase):
    def setUp(self):
        # K matrix: fx=500, fy=500, cx=320, cy=240
        self.K = np.array([
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0]
        ])
        # Identity extrinsics
        self.cam = PinholeCamera(intrinsic_matrix=self.K)

    def test_pinhole_properties(self):
        self.assertEqual(self.cam.focal_lengths, (500.0, 500.0))
        self.assertEqual(self.cam.principal_point, (320.0, 240.0))

    def test_projection_without_distortion_single(self):
        # Point in camera coordinates
        pt_3d = np.array([1.0, -0.5, 2.0]) # depth = 2.0
        # Expected: x_norm = 0.5, y_norm = -0.25
        # u = 500 * 0.5 + 320 = 570
        # v = 500 * (-0.25) + 240 = 115
        expected_2d = np.array([570.0, 115.0])

        # Project (in_world_coords=False since camera frame)
        pt_2d = self.cam.project(pt_3d, in_world_coords=False)
        np.testing.assert_allclose(pt_2d, expected_2d)

        # Unproject back to camera coordinates
        pt_3d_unproj = self.cam.unproject(pt_2d, depths=2.0, in_world_coords=False)
        np.testing.assert_allclose(pt_3d_unproj, pt_3d)

    def test_projection_without_distortion_batch(self):
        pts_3d = np.array([
            [1.0, -0.5, 2.0],
            [0.0, 0.0, 5.0],
            [-2.0, 1.5, 10.0]
        ])
        depths = pts_3d[:, 2]

        pts_2d = self.cam.project(pts_3d, in_world_coords=False)
        self.assertEqual(pts_2d.shape, (3, 2))

        # Check round trip unproject
        pts_3d_unproj = self.cam.unproject(pts_2d, depths=depths, in_world_coords=False)
        np.testing.assert_allclose(pts_3d_unproj, pts_3d, rtol=1e-5, atol=1e-5)

    def test_projection_with_distortion_roundtrip(self):
        # Set radial and tangential distortion coefficients
        # [k1, k2, p1, p2, k3]
        dist_coeffs = np.array([0.1, -0.05, 0.001, 0.002, 0.01])
        cam_dist = PinholeCamera(intrinsic_matrix=self.K, distortion_coefficients=dist_coeffs)

        pts_3d = np.array([
            [0.5, -0.3, 1.5],
            [0.0, 0.0, 2.0],
            [-0.8, 0.6, 3.0]
        ])
        depths = pts_3d[:, 2]

        # Project points to 2D
        pts_2d = cam_dist.project(pts_3d, in_world_coords=False)
        self.assertEqual(pts_2d.shape, (3, 2))

        # Unproject points back to 3D.
        # Note: undistortion in unproject utilizes numerical solver under the hood,
        # so we assert check with a small tolerance.
        pts_3d_unproj = cam_dist.unproject(pts_2d, depths=depths, in_world_coords=False)
        np.testing.assert_allclose(pts_3d_unproj, pts_3d, rtol=1e-5, atol=1e-5)

    def test_invalid_inputs(self):
        # Invalid intrinsic shape
        with self.assertRaises(ValueError):
            PinholeCamera(intrinsic_matrix=np.eye(4))

        # Invalid 3D points shape in project
        with self.assertRaises(ValueError):
            self.cam.project(np.array([1, 2]))

        # Invalid 2D points shape in unproject
        with self.assertRaises(ValueError):
            self.cam.unproject(np.array([1, 2, 3]), depths=1.0)

        # Depth array shape mismatch in unproject
        with self.assertRaises(ValueError):
            self.cam.unproject(np.array([[1, 2], [3, 4]]), depths=np.array([1.0]))

if __name__ == '__main__':
    unittest.main()
