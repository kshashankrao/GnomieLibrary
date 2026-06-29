import unittest
import numpy as np
from gnomie_library import CoordinateTransformer3D

class TestCoordinateTransformer3D(unittest.TestCase):
    def test_cartesian_to_cylindrical(self):
        x, y, z = 1.0, 1.0, 2.0
        rho, phi, z_cyl = CoordinateTransformer3D.cartesian_to_cylindrical(x, y, z)
        self.assertAlmostEqual(rho, np.sqrt(2.0))
        self.assertAlmostEqual(phi, np.pi / 4)
        self.assertAlmostEqual(z_cyl, 2.0)

    def test_cylindrical_to_cartesian(self):
        rho, phi, z = np.sqrt(2.0), np.pi / 4, 2.0
        x, y, z_cart = CoordinateTransformer3D.cylindrical_to_cartesian(rho, phi, z)
        self.assertAlmostEqual(x, 1.0)
        self.assertAlmostEqual(y, 1.0)
        self.assertAlmostEqual(z_cart, 2.0)

    def test_cartesian_to_spherical(self):
        x, y, z = 1.0, 1.0, 2.0
        r, theta, phi = CoordinateTransformer3D.cartesian_to_spherical(x, y, z)
        self.assertAlmostEqual(r, np.sqrt(6.0))
        self.assertAlmostEqual(theta, np.pi / 4)
        self.assertAlmostEqual(phi, np.arccos(2.0 / np.sqrt(6.0)))

    def test_spherical_to_cartesian(self):
        r = np.sqrt(6.0)
        theta = np.pi / 4
        phi = np.arccos(2.0 / np.sqrt(6.0))
        x, y, z = CoordinateTransformer3D.spherical_to_cartesian(r, theta, phi)
        self.assertAlmostEqual(x, 1.0)
        self.assertAlmostEqual(y, 1.0)
        self.assertAlmostEqual(z, 2.0)

if __name__ == '__main__':
    unittest.main()
