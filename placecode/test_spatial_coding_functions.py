import unittest
import numpy as np
from spatial_coding_functions import vector_sum


class TestVectorSum(unittest.TestCase):

    def test_vectors_zero_sum(self):
        """The vector sum of (1, 0), (0, 1), (-1, 0), (0, -1) should be the zero vector.
        """
        angles = np.array([0., 0.5*np.pi, np.pi, 1.5*np.pi])
        radii = np.array([1.0, 1.0, 1.0, 1.0])
        radius, angle = vector_sum(radii, angles)
        self.assertAlmostEqual(radius, 0.)
        # The angle can take any value

    def test_vectors_aligned(self):
        """The vector sum of two identical vectors should have same angle as original vectors, and twice the length
        """
        angle = 0.5124526246
        radius = 1.3524626
        angles = np.array([angle, angle])
        radii = np.array([radius, radius])
        sum_radius, sum_angle = vector_sum(radii, angles)
        self.assertAlmostEqual(angle, sum_angle)
        self.assertAlmostEqual(sum_radius, 2*radius)

    def test_empty_vectors(self):
        """The vector sum of 0 vectors should raise ValueError
        """
        angles = np.array([], dtype=np.float64)
        radii = np.array([], dtype=np.float64)
        self.assertRaises(ValueError, vector_sum, radii, angles)

    def test_angles_radii_different_length(self):
        """Input radii and angles of different length should raise ValueError
        """
        angles = np.array([0.5, 0.6], dtype=np.float64)
        radii = np.array([1.0, 1.2, 1.3], dtype=np.float64)
        self.assertRaises(ValueError, vector_sum, radii, angles)
        angles = np.array([0.5, 0.6, 0.7], dtype=np.float64)
        radii = np.array([1.0, 1.2], dtype=np.float64)
        self.assertRaises(ValueError, vector_sum, radii, angles)

    def test_proper_vector_sum(self):
        """Test vector_sum() with some correct input arrays
        """
        angles = np.array([0.5, 0.6])
        radii = np.array([0.12, 0.53])
        radius, angle = vector_sum(radii, angles)
        expected_radius = 0.6495109928425884
        expected_angle = 0.5815542931355522
        self.assertAlmostEqual(radius, expected_radius)
        self.assertAlmostEqual(angle, expected_angle)


if __name__ == '__main__':
    unittest.main()
