import torch
import numpy as np
import matplotlib.pyplot as plt

class DepthMapSurfaceNormalCalculator:
    def __init__(self, camera_intrinsics, device="cpu", neighborhood_size=1):
        """
        Initializes the DepthMapSurfaceNormalCalculator.

        Args:
            camera_intrinsics (tuple): (fx, fy, cx, cy) camera intrinsics.
            device (str): Device to use ("cpu" or "cuda").
        """
        self.fx, self.fy, self.cx, self.cy = camera_intrinsics
        self.device = torch.device(device)
        self.neighborhood_size = neighborhood_size


    def depth_map_to_point_cloud(self, depth_map):
        """Converts a depth map to a 3D point cloud."""
        height, width = depth_map.shape[-2:]
        u = torch.arange(width, device=self.device).float().repeat(height, 1)
        v = torch.arange(height, device=self.device).float().repeat(width, 1).t()
        z = depth_map
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        points = torch.stack((x, y, z), dim=-1)
        return points

    def calculate_surface_normals(self, depth_map):
        """Calculates surface normals from a depth map (vectorized)."""
        points = self.depth_map_to_point_cloud(depth_map)
        height, width, _ = points.shape[-3:]
        normals = torch.zeros_like(points)
        n = self.neighborhood_size

        # Create shifted point clouds
        # points_up = torch.roll(points, shifts=-n, dims=-3)
        # points_down = torch.roll(points, shifts=n, dims=-3)
        # points_left = torch.roll(points, shifts=n, dims=-2)
        # points_right = torch.roll(points, shifts=-n, dims=-2)
        # points_up_left = torch.roll(torch.roll(points, shifts=-n, dims=-3), shifts=n, dims=-2)
        # points_up_right = torch.roll(torch.roll(points, shifts=-n, dims=-3), shifts=-n, dims=-2)
        # points_down_left = torch.roll(torch.roll(points, shifts=n, dims=-3), shifts=n, dims=-2)
        # points_down_right = torch.roll(torch.roll(points, shifts=n, dims=-3), shifts=-n, dims=-2)
        # Calculate tangent vectors (central differences)
        # tangent_horizontal = points_right - points_left
        # tangent_vertical = points_up - points_down
        # tangent_diag1 = points_up_right - points_down_left
        # tangent_diag2 = points_up_left - points_down_right
        # Average tangent vectors
        #tangent_avg1 = torch.stack([tangent_horizontal, tangent_vertical, tangent_diag1, tangent_diag2], dim=0).mean(dim=0)
        # Create a second average tangent that is more orthogonal.
        #tangent_avg2 = torch.stack([tangent_vertical, tangent_diag1, tangent_diag2, tangent_horizontal], dim=0).mean(dim=0)
        # Cross product (using averaged tangents)
        # normals = torch.cross(tangent_avg1, tangent_avg2, dim=-1)

        
        # Create shifted point clouds
        points_u_plus = torch.roll(points, shifts=-1, dims=-2)
        points_u_minus = torch.roll(points, shifts=1, dims=-2)
        points_v_plus = torch.roll(points, shifts=-1, dims=-3)
        points_v_minus = torch.roll(points, shifts=1, dims=-3)

        # Calculate the tangent with the neighbors
        tangent_u = points_u_plus - points_u_minus
        tangent_v = points_v_plus - points_v_minus

        # Cross product to get the perpendicular vector
        normals = torch.cross(tangent_u, tangent_v, dim=-1)
        
        # Normalize
        norms = torch.norm(normals, dim=-1, keepdim=True)
        normals = torch.where(norms > 0, normals / norms, torch.zeros_like(normals))

        return normals