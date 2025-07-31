from OptimalHedging.Simulator import BaseSimulator
import numpy as np

class JumpDiffusionSimulator(BaseSimulator):
    """
    Simulator for the Merton jump-diffusion model with Poisson jumps.
    """

    def __init__(self,
                 lam: float,
                 mu_J: float,
                 sigma_J: float,
                 **kwargs):
        """
        Additional parameters for jump-diffusion:
        - lam     : jump intensity (lambda) for the Poisson process
        - mu_J    : mean of log(J) for the lognormal jump size
        - sigma_J : volatility of log(J) for the lognormal jump size
        """
        super().__init__(**kwargs)
        self.lam = lam
        self.mu_J = mu_J
        self.sigma_J = sigma_J

    def simulate_S(self) -> np.ndarray:
        """
        Simulate Merton Jump-Diffusion process using the Euler additive scheme:
            dS = mu * S * dt + sigma * S * dW + S * (J - 1) * dN
        where:
            - dW ∼ Normal(0, dt)
            - dN ∼ Poisson(lam * dt)
            - log(J) ∼ Normal(mu_J, sigma_J^2)

        Returns
        -------
        S : ndarray, shape (M, N)
            Simulated asset price paths including jumps.
        """
        
        self.dW = np.random.normal(0.0, np.sqrt(self.dt), size=(self.M, self.N - 1))            # Generate Brownian increments
        self.dN = np.random.poisson(self.lam * self.dt, size=(self.M, self.N - 1))              # Generate Poisson increments
        self.J = np.exp(np.random.normal(self.mu_J, self.sigma_J, size=(self.M, self.N - 1)))   # Generate lognormal jump sizes 
        S = np.zeros((self.M, self.N))                                                          # Initialize asset price paths
        S[:, 0] = self.S0
        increments = 1 + self.mu * self.dt + self.sigma * self.dW + (self.J - 1) * self.dN      # Build Euler additive increments
        S[:, 1:] = self.S0 * np.cumprod(increments, axis=1)                                     # Compute paths by cumulative product
        self.S_path = S
        return self.S_path