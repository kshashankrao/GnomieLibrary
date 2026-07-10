# Camera Models & Projection

Implement and simulate camera projection and unprojection models with support for extrinsics transformations and lens distortions.

---

## Mathematical Equations

### 1. Extrinsic Coordinate Transformation (World to Camera)
To transform a 3D point in the world frame $\mathbf{X}_w = [X_w, Y_w, Z_w]^T$ to the camera reference frame $\mathbf{X}_c = [X_c, Y_c, Z_c]^T$, we apply the extrinsic rotation matrix $R \in \text{SO}(3)$ and translation vector $t \in \mathbb{R}^3$:

$$\mathbf{X}_c = R \mathbf{X}_w + t$$

In homogeneous coordinates:

$$\begin{bmatrix} X_c \\ Y_c \\ Z_c \\ 1 \end{bmatrix} = \begin{bmatrix} R & t \\ \mathbf{0}^T & 1 \end{bmatrix} \begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}$$

### 2. Perspective Projection (Camera to Normalized Coordinates)
Under the pinhole camera model, 3D points in camera coordinates are projected onto a normalized image plane at depth $Z_c = 1$:

$$x_n = \frac{X_c}{Z_c}, \quad y_n = \frac{Y_c}{Z_c}$$

### 3. Lens Distortion Model
Physical camera lenses introduce distortion. The library handles radial distortion (barrel/pincushion effects) and tangential distortion (misaligned lens elements).

Given normalized coordinates $[x_n, y_n]^T$ and radial distance squared $r^2 = x_n^2 + y_n^2$:
- **Radial distortion coefficients**: $k_1, k_2, k_3$
- **Tangential distortion coefficients**: $p_1, p_2$

The distorted normalized coordinates $[x_d, y_d]^T$ are:

$$x_d = x_n (1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + 2 p_1 x_n y_n + p_2 (r^2 + 2 x_n^2)$$

$$y_d = y_n (1 + k_1 r^2 + k_2 r^4 + k_3 r^6) + p_1 (r^2 + 2 y_n^2) + 2 p_2 x_n y_n$$

### 4. Intrinsic Coordinate Transformation (Pixel Mapping)
Finally, pixel coordinates $[u, v]^T$ are computed using focal lengths $f_x, f_y$ and principal point offset $c_x, c_y$:

$$u = f_x x_d + c_x$$

$$v = f_y y_d + c_y$$

In matrix form (for undistorted points):

$$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x_n \\ y_n \\ 1 \end{bmatrix} = K \begin{bmatrix} x_n \\ y_n \\ 1 \end{bmatrix}$$

---

## API & Classes

The camera module consists of two primary classes:
1. `BaseCamera` (Abstract): Manages extrinsics properties, pose inversion, and world-to-camera coordinate transitions.
2. `PinholeCamera` (Concrete): Implements 2D-3D projection and unprojection (using OpenCV's robust undistortion algorithms).

---

## Example Usage

```python
import numpy as np
from gnomie_library import PinholeCamera

# Define camera intrinsics (fx, fy, cx, cy)
K = np.array([
    [800.0,   0.0, 320.0],
    [  0.0, 800.0, 240.0],
    [  0.0,   0.0,   1.0]
])

# Radial and tangential distortion coefficients: [k1, k2, p1, p2, k3]
dist_coeffs = np.array([-0.2, 0.1, 0.001, -0.002, 0.0])

# Camera pose: Extrinsic Rotation (R) and Translation (t)
R = np.eye(3)
t = np.array([0.0, 0.0, -1.0])  # Camera translated 1 unit backwards

# Initialize camera
camera = PinholeCamera(K=K, R=R, t=t, dist_coeffs=dist_coeffs)

# 1. Project 3D World Points to 2D Image Pixels
world_pts = np.array([
    [0.0, 0.0, 2.0],  # Directly in front of the camera at Z=2 in world coordinates
    [0.5, -0.2, 3.0]
])
pixel_coords = camera.project(world_pts)
print("Projected Pixel Coordinates:\n", pixel_coords)

# 2. Unproject 2D Pixels back to 3D World Points given Depth
pixels = np.array([
    [320.0, 240.0],
    [400.0, 200.0]
])
depths = np.array([3.0, 4.0])

world_pts_recon = camera.unproject(pixels, depths)
print("Reconstructed 3D Points:\n", world_pts_recon)
```
