import unittest
from Black_Scholes_Model import * 

class TestSimulateTrajectory(unittest.TestCase):
    def test_abbas(self):

        S0, S, M, N = trajectory()

        

        # Test shape of array
        self.assertEqual(S.shape, (M, N + 1))
        # Test initial values
        self.assertTrue(np.allclose(S[:, 0], S0))
        # Test values are finite
        self.assertTrue(np.all(np.isfinite(S)))
        # Test all values are positive
        self.assertTrue(np.all(S > 0))

if __name__ == "__main__":
    unittest.main()