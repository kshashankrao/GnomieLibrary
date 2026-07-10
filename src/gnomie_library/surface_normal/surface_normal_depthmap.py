import numpy as np

class DepthMapSurfaceNormalCalculator:
    def __init__(self, camera_intrinsics, device="cpu", neighborhood_size=1):
        """
        Initializes the DepthMapSurfaceNormalCalculator.

        Args:
            camera_intrinsics (tuple): (fx, fy, cx, cy) camera intrinsics.
            device (str): Ignored parameter, kept for backward compatibility.
            neighborhood_size (int): Size of the neighborhood to calculate tangents.
        """
        if len(camera_intrinsics) != 4:
            raise ValueError(f"camera_intrinsics must be a tuple/list of length 4 (fx, fy, cx, cy), got {camera_intrinsics}")
        self.fx, self.fy, self.cx, self.cy = camera_intrinsics
        self.neighborhood_size = neighborhood_size

    def depth_map_to_point_cloud(self, depth_map):
        """Converts a depth map to a 3D point cloud."""
        if isinstance(depth_map, (str, bytes)):
            raise TypeError(f"depth_map must be a numeric array-like object, got {type(depth_map)}")
        depth_map = np.asarray(depth_map, dtype=np.float64)
        if depth_map.ndim < 2:
            raise ValueError(f"depth_map must have at least 2 dimensions, got shape {depth_map.shape}")

        height, width = depth_map.shape[-2:]
        u = np.arange(width, dtype=np.float64)
        v = np.arange(height, dtype=np.float64)
        uu, vv = np.meshgrid(u, v)
        
        z = depth_map
        x = (uu - self.cx) * z / self.fx
        y = (vv - self.cy) * z / self.fy
        points = np.stack((x, y, z), axis=-1)
        return points

    def calculate_surface_normals(self, depth_map):
        """Calculates surface normals from a depth map (vectorized)."""
        if isinstance(depth_map, (str, bytes)):
            raise TypeError(f"depth_map must be a numeric array-like object, got {type(depth_map)}")
        depth_map = np.asarray(depth_map, dtype=np.float64)
        if depth_map.ndim < 2:
            raise ValueError(f"depth_map must have at least 2 dimensions, got shape {depth_map.shape}")

        points = self.depth_map_to_point_cloud(depth_map)
        
        # Shifted point clouds along width (axis=-2) and height (axis=-3)
        points_u_plus = np.roll(points, shift=-1, axis=-2)
        points_u_minus = np.roll(points, shift=1, axis=-2)
        points_v_plus = np.roll(points, shift=-1, axis=-3)
        points_v_minus = np.roll(points, shift=1, axis=-3)

        # Calculate the tangent with the neighbors
        tangent_u = points_u_plus - points_u_minus
        tangent_v = points_v_plus - points_v_minus

        # Cross product to get the perpendicular vector
        normals = np.cross(tangent_u, tangent_v, axis=-1)
        
        # Normalize
        norms = np.linalg.norm(normals, axis=-1, keepdims=True)
        normals = np.where(norms > 0, normals / norms, np.zeros_like(normals))
        
        return normals