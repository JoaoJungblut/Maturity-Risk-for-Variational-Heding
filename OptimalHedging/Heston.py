from OptimalHedging.Simulator import BaseSimulator
import numpy as np

class HestonSimulator(BaseSimulator):
    """
    Simulator for the Heston stochastic volatility model.
    """

    def __init__(self,
                 v0: float,
                 kappa: float,
                 theta: float,
                 sigma_v: float,
                 rho: float,
                 **kwargs):
        """
        Additional parameters for the Heston model:
        - v0      : initial variance
        - kappa   : rate of mean reversion of variance
        - theta   : long-term variance level
        - sigma_v : volatility of variance
        - rho     : correlation between dW1 and dW2
        """
        super().__init__(**kwargs)
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho

    def simulate_S(self) -> np.ndarray:
        """
        Simulate Heston model using Euler–Maruyama with correlated Brownian motions:
            dS = mu * S * dt + sqrt(v) * S * dW1
            dv = kappa * (theta - v) * dt + sigma_v * sqrt(v) * dW2
            Corr(dW1, dW2) = rho 

        Returns
        -------
        S : ndarray, shape (M, N+1)
            Simulated asset price paths.
        v : ndarray, shape (M, N+1)
            Simulated variance paths.
        """

        # --- Generate correlated Brownian increments ---
        cov = np.array([[1.0, self.rho],
                        [self.rho, 1.0]])
        dW = np.random.multivariate_normal(mean=[0.0, 0.0], cov=cov, size=(self.M, self.N - 1))
        dW *= np.sqrt(self.dt)
        self.dW1 = dW[:, :, 0]
        self.dW2 = dW[:, :, 1]

        # --- Initialize paths ---
        S = np.zeros((self.M, self.N))
        v = np.zeros((self.M, self.N))
        S[:, 0] = self.S0
        v[:, 0] = self.v0

        # --- Alfonsi updates for v_t ---
        for n in range(1, self.N):
            v_prev = v[:, n-1]
            v[:, n] = (v_prev + self.kappa * self.theta * self.dt
                    + self.sigma_v * np.sqrt(v_prev) * self.dW2[:, n-1]
                    + 0.25 * self.sigma_v**2 * self.dt) / (1 + self.kappa * self.dt)

        # --- Euler additive for S_t ---
        increments = 1 + self.mu * self.dt + np.sqrt(v[:, :-1]) * self.dW1
        S[:, 1:] = self.S0 * np.cumprod(increments, axis=1)

        self.v_path = v
        self.S_path = S
        return self.S_path, self.v_path