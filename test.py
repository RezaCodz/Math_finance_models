import unittest
from scipy.stats import shapiro
from Black_Scholes_Model import * 

def is_normal(data, alpha=0.05):
    stat, p = shapiro(data)
    return p > alpha  # True if we can't reject normality

class TestSimulateTrajectory(unittest.TestCase):

    def test_prmary(self):

        S0, S, M, N = trajectory()

        # Test shape of array
        self.assertEqual(S.shape, (M, N + 1))
        # Test initial values
        self.assertTrue(np.allclose(S[:, 0], S0))
        # Test values are finite
        self.assertTrue(np.all(np.isfinite(S)))
        # Test all values are positive
        self.assertTrue(np.all(S > 0))

    def test_secondary(self):
        
        S0, S, M, N = trajectory()
        self.assertEqual(S.shape, (M, N + 1))

        # Check distribution of final prices
        final_prices = S[:, -1]
        log_final_prices = np.log(final_prices)
        log_returns = np.log(S[:, 1:] / S[:, :-1]).flatten()

        print("\nTesting distribution of simulated data:")

        if is_normal(final_prices):
            print("Final stock prices appear normally distributed.")
        else:
            print("Final stock prices are NOT normally distributed (as expected).")

        if is_normal(log_final_prices):
            print("Log of final stock prices appears normally distributed ⇒ prices are log-normal ✔")
        else:
            print("Log of final stock prices are NOT normally distributed ⇒ prices may not be log-normal ❌")

        if is_normal(log_returns):
            print("Log-returns appear normally distributed ✔")
        else:
            print("Log-returns are NOT normally distributed ❌")        

if __name__ == "__main__":
    unittest.main()