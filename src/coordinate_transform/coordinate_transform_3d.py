import numpy as np

class CoordinateTransformer3D:
    """A class for transforming 3D points between Cartesian, cylindrical, and spherical coordinate systems."""

    @staticmethod
    def cartesian_to_cylindrical(x, y, z):
        """Converts Cartesian coordinates to cylindrical coordinates."""
        rho = np.sqrt(x**2 + y**2)
        phi = np.arctan2(y, x)
        return rho, phi, z

    @staticmethod
    def cylindrical_to_cartesian(rho, phi, z):
        """Converts cylindrical coordinates to Cartesian coordinates."""
        x = rho * np.cos(phi)
        y = rho * np.sin(phi)
        return x, y, z

    @staticmethod
    def cartesian_to_spherical(x, y, z):
        """Converts Cartesian coordinates to spherical coordinates."""
        r = np.sqrt(x**2 + y**2 + z**2)
        theta = np.arctan2(y, x)
        phi = np.arccos(z / r) if r != 0 else 0  # Handle r=0
        return r, theta, phi

    @staticmethod
    def spherical_to_cartesian(r, theta, phi):
        """Converts spherical coordinates to Cartesian coordinates."""
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        return x, y, z

    @staticmethod
    def cylindrical_to_spherical(rho, phi, z):
        """Converts cylindrical coordinates to spherical coordinates."""
        r = np.sqrt(rho**2 + z**2)
        theta = phi
        phi_spherical = np.arccos(z / r) if r != 0 else 0  # Handle r=0
        return r, theta, phi_spherical

    @staticmethod
    def spherical_to_cylindrical(r, theta, phi):
        """Converts spherical coordinates to cylindrical coordinates."""
        rho = r * np.sin(phi)
        phi_cylindrical = theta
        z = r * np.cos(phi)
        return rho, phi_cylindrical, z
