import unittest
import numpy as np
from scipy.stats import shapiro
from Black_Scholes_Model import trajectory


def is_normal(data, alpha=0.05):
    stat, p = shapiro(data)
    return p > alpha  # True if we cannot reject normality


class TestBlackScholesModel(unittest.TestCase):

    def setUp(self):
        self.S0, self.S, self.M, self.N = trajectory()

    def test_shape_and_values(self):
        # Check correct shape
        self.assertEqual(self.S.shape, (self.M, self.N + 1))

        # Check initial values
        self.assertTrue(np.allclose(self.S[:, 0], self.S0))

        # Check for finite and positive values
        self.assertTrue(np.all(np.isfinite(self.S)))
        self.assertTrue(np.all(self.S > 0))

    def test_log_returns_are_normal(self):
        log_returns = np.log(self.S[:, 1:] / self.S[:, :-1]).flatten()
        self.assertTrue(is_normal(log_returns), "Log-returns are not normally distributed.")

    def test_log_final_prices_are_normal(self):
        log_final_prices = np.log(self.S[:, -1])
        self.assertTrue(is_normal(log_final_prices), "Log of final prices is not normally distributed.")

    def test_final_prices_not_normal(self):
        final_prices = self.S[:, -1]
        self.assertFalse(is_normal(final_prices), "Final prices appear normally distributed (unexpected).")


if __name__ == "__main__":
    unittest.main()
