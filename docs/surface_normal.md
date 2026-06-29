# Surface Normals from Depthmap

Calculate surface normals directly from depth maps using vectorized operations.

## Example Usage

```python
import torch
from gnomie_library import DepthMapSurfaceNormalCalculator

# Camera intrinsics (fx, fy, cx, cy)
intrinsics = (500.0, 500.0, 320.0, 240.0)

# Initialize calculator
calculator = DepthMapSurfaceNormalCalculator(camera_intrinsics=intrinsics)

# Dummy depth map (1 x Height x Width)
depth_map = torch.ones((1, 480, 640))

# Calculate normals
normals = calculator.calculate_surface_normals(depth_map)

print("Normals Shape:", normals.shape)
```
