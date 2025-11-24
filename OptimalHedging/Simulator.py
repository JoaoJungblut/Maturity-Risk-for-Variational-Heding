import numpy as np
from abc import ABC, abstractmethod

class BaseSimulator(ABC):
    """
    Abstract base class for stochastic process simulators.

    Responsibilities:
      - Construct time grid
      - Initialize random seed
      - Store core parameters (S0, mu, sigma, etc.)

    Subclasses must implement the simulate() method.
    """

    def __init__(self,
                 S0: float,
                 mu: float,
                 sigma: float,
                 K: float,
                 t0: float = 0.0,
                 T: float = 1.0,
                 N: int = 252,
                 M: int = 1000,
                 seed: int = 123):
        """
        Initialize the base simulator.

        Parameters
        ----------
        S0 : float
            Initial asset price.
        mu : float
            Drift coefficient.
        sigma : float
            Diffusion volatility (standard deviation).
        t0 : float, default=0.0
            Initial time.
        T : float, default=1.0
            Final time.
        N : int, default=252
            Number of time steps.
        M : int, default=1000
            Number of simulated paths.
        seed : int, default=123
            Random seed for reproducibility.
        """
        # store parameters
        self.S0 = S0
        self.mu = mu 
        self.sigma = sigma
        self.K = K

        self.t0 = t0
        self.T = T
        self.N = N
        self.M = M

        self.seed = seed

        # compute time increment and grid
        self.dt = T / N
        self.time_grid = np.linspace(t0, T, N)

        # initialize random number generator if seed is given
        if seed is not None:
            np.random.seed(seed)

    @abstractmethod
    def simulate_S(self):
        """
        Simulate the stochastic process.

        Must be overridden by each subclass.

        Returns
        -------
        ndarray, shape (M, N+1)
            Array of simulated paths.
        """
        pass

    @abstractmethod
    def simulate_H(self):
        """
        Backward simulation of H_t.
        To be implemented by subclasses (or a generic version in BaseSimulator).
        """
        pass

    @abstractmethod
    def simulate_L(self, h0: float):
        """
        Simulation of L_t.
        To be implemented by subclasses (or a generic version in BaseSimulator).
        """
        pass

    @abstractmethod
    def generate_adj(self):
        """
        Generation of adjoint process p, q, r.
        To be implemented by subclasses (or a generic version in BaseSimulator).
        """
        pass

    @abstractmethod
    def update_control(self):
        """
        Control Update and Convergence Criterion
        To be implemented by subclasses (or a generic version in BaseSimulator).
        """
        pass

    @abstractmethod
    def update_L(self):
        """
        Forward simulation of the profit process L_t using the optimal hedge h_opt.
        To be implemented by subclasses (or a generic version in BaseSimulator).
        """
        pass