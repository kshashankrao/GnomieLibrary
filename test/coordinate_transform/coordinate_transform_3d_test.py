import numpy as np
from coordinate_transform import CoordinateTransformer3D
# Example Cartesian point
x, y, z = 1.0, 1.0, 2.0

# Cartesian to Cylindrical
rho, phi, z_cylindrical = CoordinateTransformer3D.cartesian_to_cylindrical(x, y, z)
print(f"Cartesian ({x}, {y}, {z}) to Cylindrical ({rho:.4f}, {phi:.4f}, {z_cylindrical:.4f})")

# Cylindrical to Cartesian
x_cartesian, y_cartesian, z_cartesian = CoordinateTransformer3D.cylindrical_to_cartesian(rho, phi, z_cylindrical)
print(f"Cylindrical ({rho:.4f}, {phi:.4f}, {z_cylindrical:.4f}) to Cartesian ({x_cartesian:.4f}, {y_cartesian:.4f}, {z_cartesian:.4f})")

# Cartesian to Spherical
r, theta, phi_spherical = CoordinateTransformer3D.cartesian_to_spherical(x, y, z)
print(f"Cartesian ({x}, {y}, {z}) to Spherical ({r:.4f}, {theta:.4f}, {phi_spherical:.4f})")

# Spherical to Cartesian
x_cartesian, y_cartesian, z_cartesian = CoordinateTransformer3D.spherical_to_cartesian(r, theta, phi_spherical)
print(f"Spherical ({r:.4f}, {theta:.4f}, {phi_spherical:.4f}) to Cartesian ({x_cartesian:.4f}, {y_cartesian:.4f}, {z_cartesian:.4f})")

# Cylindrical to Spherical
r_spherical, theta_spherical, phi_spherical2 = CoordinateTransformer3D.cylindrical_to_spherical(rho, phi, z_cylindrical)
print(f"Cylindrical ({rho:.4f}, {phi:.4f}, {z_cylindrical:.4f}) to Spherical ({r_spherical:.4f}, {theta_spherical:.4f}, {phi_spherical2:.4f})")

# Spherical to Cylindrical
rho_cylindrical, phi_cylindrical2, z_cylindrical2 = CoordinateTransformer3D.spherical_to_cylindrical(r, theta, phi_spherical)
print(f"Spherical ({r:.4f}, {theta:.4f}, {phi_spherical:.4f}) to Cylindrical ({rho_cylindrical:.4f}, {phi_cylindrical2:.4f}, {z_cylindrical2:.4f})")