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

        self.dW = np.random.normal(loc=0.0, 
                                   scale=np.sqrt(self.dt), 
                                   size=(self.M, self.N - 1))           # Generate Brownian increments (M x N-1) as N(0,1) 
        factors = 1 + self.mu * self.dt + self.sigma * dW               # Compute multiplicative factors for all steps
        factors = np.hstack((np.ones((self.M, 1)), factors))            # Insert a column of ones for the initial asset value
        self.S_path = self.S0 * np.cumprod(factors, axis=1)             # Cumulative product along time to build paths
        return self.S_path
    
    def simulate_H(self, K: float) -> np.ndarray:
        """
        Backward simulation of H_t using BSDE regression scheme.

        This method uses self.S_paths and self.dW, which must have been generated 
        previously by calling simulate_S().

        Parameters
        ----------
        K : float
            Strike price for the European call option.

        Returns
        -------
        H : ndarray, shape (M, N)
            Backward simulated values of H_t.
        dH_dS : ndarray, shape (M, N)
            Estimated first derivative with respect to S.
        d2H_dSS : ndarray, shape (M, N)
            Estimated second derivative with respect to S.
        dH_dt : ndarray, shape (M, N)
            Estimated time derivative.
        """
        S_paths = self.S_paths
        dW = self.dW

        M, N = S_paths.shape
        H = np.zeros((M, N))
        dH_dS = np.zeros((M, N))
        d2H_dSS = np.zeros((M, N))
        dH_dt = np.zeros((M, N))

        # Terminal condition
        H[:, -1] = np.maximum(S_paths[:, -1] - K, 0)

        # Backward iteration over time steps
        for n in reversed(range(N - 1)):
            S_n = S_paths[:, n]
            dW_n = dW[:, n]

            # Linear regression to estimate alpha and beta
            X = np.column_stack((np.full(M, self.dt), dW_n))
            Y = H[:, n+1]
            coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
            alpha_n, beta_n = coeffs

            # Update H_n for all paths
            H[:, n] = H[:, n+1] - alpha_n * self.dt - beta_n * dW_n

            # Compute derivatives
            dH_dS[:, n] = beta_n / (self.sigma * S_n)
            d2H_dSS[:, n] = 2 * (alpha_n - self.mu * S_n * dH_dS[:, n]) / (self.sigma**2 * S_n**2)
            dH_dt[:, n] = ((H[:, n+1] - H[:, n]) / self.dt
                        - self.mu * S_n * dH_dS[:, n]
                        - 0.5 * self.sigma**2 * S_n**2 * d2H_dSS[:, n])
        
        self.H = H
        self.dH_dS = dH_dS
        self.d2H_dSS = d2H_dSS
        self.dH_dt = dH_dt

        return self.H, self.dH_dS, self.d2H_dSS, self.dH_dt
