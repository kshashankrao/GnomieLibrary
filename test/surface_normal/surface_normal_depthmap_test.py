import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
from data.test_data import create_dog_depth_map

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from surface_normal import DepthMapSurfaceNormalCalculator


device = "cuda" if torch.cuda.is_available() else "cpu"
# Generate a complex depth map
height, width = 256, 512
depth_map_np = create_dog_depth_map(height, width)
depth_map_torch = torch.from_numpy(depth_map_np).float().unsqueeze(0).to(device)

camera_intrinsics = (500, 500, width / 2, height / 2)
normal_calculator = DepthMapSurfaceNormalCalculator(camera_intrinsics, device=device)
normals = normal_calculator.calculate_surface_normals(depth_map_torch)

# Visualization
normals_np = normals.squeeze(0).cpu().numpy()
depth_map_np = depth_map_torch.squeeze(0).cpu().numpy()

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.imshow(depth_map_np, cmap='viridis')
plt.title("Depth Map")
plt.colorbar()

plt.subplot(1, 2, 2)
normal_rgb = (normals_np + 1) / 2
plt.imshow(normal_rgb)
plt.title("Surface Normals (RGB)")

plt.show()