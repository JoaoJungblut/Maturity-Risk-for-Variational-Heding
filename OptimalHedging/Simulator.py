import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple


class BaseSimulator(ABC):
    """
    Abstract base class for stochastic process simulators and optimal hedging.

    Responsibilities:
      - Store core model parameters and time grid
      - Define the interface for:
          * underlying simulation (S)
          * derivative pricing/greeks (H, dH/dS, ...)
          * P&L dynamics
          * adjoint BSDE (p, q)
          * optimality condition and control update
          * main optimization loop
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
        K : float
            Strike price (used for the derivative).
        t0 : float, default=0.0
            Initial time.
        T : float, default=1.0
            Final time.
        N : int, default=252
            Number of time steps (used to compute dt).
        M : int, default=1000
            Number of simulated paths.
        seed : int, default=123
            Random seed for reproducibility.
        """
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.K = K

        self.t0 = t0
        self.T = T
        self.N = N
        self.M = M

        self.seed = seed

        # time step
        self.dt = (T - t0) / N

        if seed is not None:
            np.random.seed(seed)

        # placeholders to be filled by subclasses
        self.S_path: np.ndarray | None = None
        self.dW: np.ndarray | None = None

        self.S_grid: np.ndarray | None = None
        self.H_grid: np.ndarray | None = None
        self.Delta_grid: np.ndarray | None = None
        self.Gamma_grid: np.ndarray | None = None
        self.dHdt_grid: np.ndarray | None = None
        self.d2H_dt_dS_grid: np.ndarray | None = None
        self.d3H_dS3_grid: np.ndarray | None = None

        self.H: np.ndarray | None = None
        self.dH_dS: np.ndarray | None = None
        self.d2H_dSS: np.ndarray | None = None
        self.dH_dt: np.ndarray | None = None
        self.d2H_dt_dS: np.ndarray | None = None
        self.d3H_dS3: np.ndarray | None = None

    # ============================================================
    # 0. Underlying S and derivative H
    # ============================================================

    @abstractmethod
    def simulate_S(self) -> np.ndarray:
        """
        Simulate the underlying asset paths S_t.

        Returns
        -------
        S_path : ndarray, shape (M, steps)
            Simulated underlying paths.
        """
        raise NotImplementedError

    @abstractmethod
    def simulate_H(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                  np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute (or approximate) the derivative price H_t and its greeks
        along the simulated S_path.

        Returns
        -------
        H : ndarray, shape (M, steps)
        dH_dS : ndarray, shape (M, steps)
        d2H_dSS : ndarray, shape (M, steps)
        dH_dt : ndarray, shape (M, steps)
        d2H_dt_dS : ndarray, shape (M, steps)
        d3H_dS3 : ndarray, shape (M, steps)
        """
        raise NotImplementedError

    # ============================================================
    # 1. Control initialization
    # ============================================================

    @abstractmethod
    def init_control(self, kind: str = "delta") -> np.ndarray:
        """
        Initialize the control h.

        Parameters
        ----------
        kind : {"delta", "zero"}
            Initialization rule.

        Returns
        -------
        h : ndarray, shape (M, steps-1)
            Initial control on each time interval [t_n, t_{n+1}).
        """
        raise NotImplementedError

    # ============================================================
    # 2. Forward P&L
    # ============================================================

    @abstractmethod
    def forward_PL(self, h: np.ndarray, L0: float = 0.0) -> np.ndarray:
        """
        Simulate the P&L L_t given the control h.

        Parameters
        ----------
        h : ndarray, shape (M, steps-1)
            Control on each interval [t_n, t_{n+1}).
        L0 : float, default=0.0
            Initial P&L.

        Returns
        -------
        L : ndarray, shape (M, steps)
            Simulated P&L paths.
        """
        raise NotImplementedError

    # ============================================================
    # 3. Risk functional rho_u(L_T)
    # ============================================================

    @staticmethod
    @abstractmethod
    def risk_function(LT: np.ndarray, risk_type: str, **kwargs) -> float:
        """
        Compute a sample-based value of the composed risk functional rho_u(L_T).

        Parameters
        ----------
        LT : ndarray, shape (M,)
            Terminal P&L samples L_T^h.
        risk_type : str
            One of:
                'ele'  -> Expected Loss Exponential
                'elw'  -> Expected Loss Weibull
                'entl' -> Entropic Linear
                'ente' -> Entropic Exponential
                'entw' -> Entropic Weibull
                'es'   -> Expected Shortfall (linear utility)
        **kwargs :
            Parameters depending on risk_type.

        Returns
        -------
        rho : float
            Estimated risk value.
        """
        raise NotImplementedError

    # ============================================================
    # 4. Terminal adjoint p_T = Upsilon(L_T)
    # ============================================================

    @staticmethod
    @abstractmethod
    def terminal_adjoint(LT: np.ndarray, risk_type: str, **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        Compute the adjoint terminal condition p_T = Upsilon(L_T)
        for the composed risk measures.

        Parameters
        ----------
        LT : ndarray, shape (M,)
            Terminal P&L samples L_T^h.
        risk_type : str
            One of {"ele", "elw", "entl", "ente", "entw", "es"}.
        **kwargs :
            Parameters depending on risk_type.

        Returns
        -------
        pT : ndarray, shape (M,)
            Terminal adjoint samples.
        info : dict
            Extra information (e.g. normalizing constants, alpha_star for ES).
        """
        raise NotImplementedError

    # ============================================================
    # 5. Backward adjoint (p_t, q_t)
    # ============================================================

    @abstractmethod
    def backward_adjoint(self, pT: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve (approximately) the backward SDE for (p_t, q_t),
        given a terminal condition p_T = pT.

        Parameters
        ----------
        pT : ndarray, shape (M,)
            Terminal condition p_T.

        Returns
        -------
        p : ndarray, shape (M, steps)
        q : ndarray, shape (M, steps-1)
        """
        raise NotImplementedError

    # ============================================================
    # 6. Optimality condition / gradient
    # ============================================================

    @abstractmethod
    def compute_gradient(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Compute the violation of the local optimality condition
        (e.g. G_n = μ S_n p_n + σ S_n q_n for GBM).

        Parameters
        ----------
        p : ndarray, shape (M, steps)
        q : ndarray, shape (M, steps-1)

        Returns
        -------
        G : ndarray, shape (M, steps-1)
            Gradient-like quantity used to update the control.
        """
        raise NotImplementedError

    # ============================================================
    # 7. Control update
    # ============================================================

    @staticmethod
    @abstractmethod
    def update_control(h: np.ndarray, G: np.ndarray, alpha: float) -> np.ndarray:
        """
        Update the control using G (e.g. a gradient step).

        Parameters
        ----------
        h : ndarray, shape (M, steps-1)
            Current control.
        G : ndarray, shape (M, steps-1)
            Gradient-like term.
        alpha : float
            Step size.

        Returns
        -------
        h_new : ndarray, shape (M, steps-1)
            Updated control.
        """
        raise NotImplementedError

    # ============================================================
    # 8. Main optimization loop
    # ============================================================

    @abstractmethod
    def optimize_hedge(self,
                       risk_type: str,
                       risk_kwargs: Dict,
                       max_iter: int = 20,
                       tol: float = 1e-4,
                       alpha: float = 1e-3,
                       verbose: bool = True) -> Tuple[np.ndarray, List[Dict]]:
        """
        Main optimization loop to compute an approximate optimal hedge h.

        Parameters
        ----------
        risk_type : str
            One of {"ele", "elw", "entl", "ente", "entw", "es"}.
        risk_kwargs : dict
            Parameters required for the chosen risk_type.
        max_iter : int
            Maximum number of iterations.
        tol : float
            Tolerance on E[G^2] (stopping criterion).
        alpha : float
            Step size for control updates.
        verbose : bool
            If True, print iteration logs.

        Returns
        -------
        h_opt : ndarray, shape (M, steps-1)
            Approximate optimal control.
        history : list of dict
            Iteration history (risk value, gradient norm, etc.).
        """
        raise NotImplementedError
