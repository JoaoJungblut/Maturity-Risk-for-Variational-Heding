from OptimalHedging.Simulator import BaseSimulator
import numpy as np

class GBMSimulator(BaseSimulator):
    """
    Simulator for Geometric Brownian Motion (GBM).
    """

    def simulate_S(self) -> np.ndarray:
        """
        Simulate Geometric Brownian Motion (GBM) paths using a fully vectorized Euler–Maruyama scheme.
            dS = mu * S * dt + sqrt(v) * S * dW

        Returns
        -------
        S : ndarray, shape (M, N+1)
            Simulated GBM asset price paths.
        """

        dW = np.random.normal(loc=0.0, 
                                   scale=np.sqrt(self.dt), 
                                   size=(self.M, self.N - 1))           # Generate Brownian increments (M x N-1) as N(0,1) 
        factors = 1 + self.mu * self.dt + self.sigma * dW               # Compute multiplicative factors for all steps
        factors = np.hstack((np.ones((self.M, 1)), factors))            # Insert a column of ones for the initial asset value
        S_path = self.S0 * np.cumprod(factors, axis=1)                  # Cumulative product along time to build paths

        self.S_path = S_path
        self.dW = dW

        return S_path
    
    
    def simulate_H(self, K: float) -> np.ndarray:
        """
        Backward estimation of H_t and its derivatives using a Longstaff–Schwartz style regression.

        This method uses the stored asset paths self.S_paths, which must be generated
        previously by calling simulate_S().

        Parameters
        ----------
        K : float
            Strike price for the terminal payoff.

        Returns
        -------
        H : ndarray, shape (M, N)
            Backward estimated values of H_t.
        dH_dS : ndarray, shape (M, N)
            Estimated ∂H/∂S.
        d2H_dSS : ndarray, shape (M, N)
            Estimated ∂²H/∂S².
        dH_dt : ndarray, shape (M, N)
            Estimated ∂H/∂t.
        """
        S_path = self.S_path
        M, N = S_path.shape

        # Initialize arrays
        H = np.zeros((M, N))
        dH_dS = np.zeros((M, N))
        d2H_dSS = np.zeros((M, N))
        dH_dt = np.zeros((M, N))

        # Terminal condition
        H[:, -1] = np.maximum(S_path[:, -1] - K, 0)

        # Backward iteration with polynomial regression
        for n in reversed(range(N - 1)):
            S_n = S_path[:, n]
            Y = H[:, n + 1]

            # Regression basis [1, S, S^2]
            X = np.column_stack((np.ones(M), S_n, S_n**2))

            # Solve least squares for coefficients
            coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
            a0, a1, a2 = coeffs

            # Fitted H_n from regression
            H_fit = a0 + a1 * S_n + a2 * (S_n**2)
            H[:, n] = np.maximum(H_fit, 0)

            # Derivatives from regression coefficients
            dH_dS[:, n] = a1 + 2 * a2 * S_n
            d2H_dSS[:, n] = 2 * a2

            # Time derivative using Itô's formula
            dH_dt[:, n] = ((H[:, n + 1] - H[:, n]) / self.dt
                        - self.mu * S_n * dH_dS[:, n]
                        - 0.5 * self.sigma**2 * S_n**2 * d2H_dSS[:, n])
            
            self.H = H
            self.dh_dS = dH_dS
            self.d2H_dSS = d2H_dSS
            self.dH_dt = dH_dt

        return H, dH_dS, d2H_dSS, dH_dt
    
    
    def simulate_L(self, h0: float) -> np.ndarray:
        return super().simulate_L()
