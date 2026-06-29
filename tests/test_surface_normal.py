import unittest
import torch
from gnomie_library import DepthMapSurfaceNormalCalculator

class TestDepthMapSurfaceNormalCalculator(unittest.TestCase):
    def setUp(self):
        self.intrinsics = (500.0, 500.0, 320.0, 240.0) # fx, fy, cx, cy
        self.calculator = DepthMapSurfaceNormalCalculator(camera_intrinsics=self.intrinsics)

    def test_depth_map_to_point_cloud_shape(self):
        depth_map = torch.ones((1, 480, 640)) # Batch size 1
        points = self.calculator.depth_map_to_point_cloud(depth_map)
        # Output should be (1, 480, 640, 3)
        self.assertEqual(points.shape, (1, 480, 640, 3))

    def test_calculate_surface_normals_shape(self):
        depth_map = torch.ones((1, 480, 640))
        normals = self.calculator.calculate_surface_normals(depth_map)
        self.assertEqual(normals.shape, (1, 480, 640, 3))

    def test_flat_surface_normal_direction(self):
        # A perfectly flat depth map should have normals pointing in Z direction (0, 0, 1) or (0, 0, -1) depending on convention
        depth_map = torch.ones((480, 640))
        normals = self.calculator.calculate_surface_normals(depth_map)
        
        # Central pixel normal
        n_center = normals[240, 320]
        # X and Y components should be 0, Z component should be 1
        self.assertAlmostEqual(n_center[0].item(), 0.0, places=5)
        self.assertAlmostEqual(n_center[1].item(), 0.0, places=5)
        # It could be negative depending on cross product order, but the magnitude is 1
        self.assertAlmostEqual(abs(n_center[2].item()), 1.0, places=5)

if __name__ == '__main__':
    unittest.main()
