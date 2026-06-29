# Coordinate Transformations

This module provides the `CoordinateTransformer3D` class for easy conversions between Cartesian, Cylindrical, and Spherical coordinate systems.

## Example Usage

```python
from gnomie_library import CoordinateTransformer3D

# Cartesian to Cylindrical
x, y, z = 1.0, 1.0, 2.0
rho, phi, z_cyl = CoordinateTransformer3D.cartesian_to_cylindrical(x, y, z)
print(f"Cylindrical: rho={rho}, phi={phi}, z={z_cyl}")

# Cartesian to Spherical
r, theta, phi_sph = CoordinateTransformer3D.cartesian_to_spherical(x, y, z)
print(f"Spherical: r={r}, theta={theta}, phi={phi_sph}")
```
