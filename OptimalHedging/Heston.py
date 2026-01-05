from OptimalHedging.Simulator import BaseSimulator
import numpy as np
from typing import Dict, List, Tuple


class HestonSimulator(BaseSimulator):
    """
    Heston ('Reston') + máquina de hedge igual ao GBM,
    mas com gradiente ∂_h H próprio do modelo.
    """

    def __init__(self,
                 kappa: float,
                 theta: float,
                 sigma_v: float,
                 corr: float,
                 **base_kwargs):
        """
        Initialize the Heston simulator.

        Parameters
        ----------
        kappa : float
            Mean-reversion speed of the variance process.
        theta : float
            Long-run mean of the variance process.
        sigma_v : float
            Volatility of variance (vol-of-vol).
        corr : float
            Correlation between dW1 (asset) and dW2 (variance).
        base_kwargs : dict
            Same arguments as BaseSimulator/GBMSimulator:
            S0, mu, sigma, K, t0, T, N, M, seed.
        """
        super().__init__(**base_kwargs)

        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.corr = corr


    # ============================================================
    # 0. Underlying S,v and derivative H
    # ============================================================

    def simulate_S(self) -> np.ndarray:
        """
        Simulate Heston model paths using a vectorized (over paths) Euler–Maruyama scheme.

            dS = mu * S * dt + sqrt(v) * S * dW1
            dv = kappa*(theta - v)*dt + sigma_v*sqrt(v)*dW2

        with corr(dW1, dW2) = corr.

        Returns
        -------
        S : ndarray, shape (M, steps)
            Simulated Heston asset price paths.
        v: ndarray, shape (M, steps)
            Simulated Heston stochastic variance.    
        """
        steps = int(np.round((self.T - self.t0) / self.dt))

        # 2x2 correlation (covariance) matrix for (dW1, dW2)
        Sigma = np.array([
            [1.0,       self.corr],
            [self.corr, 1.0]
        ])

        # correlated Brownian increments: shape (M, steps-1, 2)
        # each increment has covariance Sigma * dt
        dW = np.random.multivariate_normal(
            mean=np.zeros(2),
            cov=Sigma * self.dt,
            size=(self.M, steps - 1)
        )

        dW1 = dW[:, :, 0]  # (M, steps-1): Brownian increments driving S
        dW2 = dW[:, :, 1]  # (M, steps-1): Brownian increments driving v

        S = np.zeros((self.M, steps))
        v = np.zeros((self.M, steps))

        S[:, 0] = self.S0
        v[:, 0] = self.sigma

        eps = 1e-8  # floor to keep variance non-negative numerically

        # time loop only (vectorized across paths)
        for n in range(steps - 1):
            Sn = S[:, n]
            vn = np.maximum(v[:, n], eps)

            # Euler step for S
            S[:, n + 1] = Sn * (1.0 + self.mu * self.dt + np.sqrt(vn) * dW1[:, n])

            # Euler step for v (with truncation)
            v_next = vn + self.kappa * (self.theta - vn) * self.dt + self.sigma_v * np.sqrt(vn) * dW2[:, n]
            v[:, n + 1] = np.maximum(v_next, eps)

        self.S_Heston = S
        self.v_Heston = v
        self.dW1_Heston = dW1
        self.dW2_Heston = dW2

        return S, v
 

    def simulate_H(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                  np.ndarray, np.ndarray, np.ndarray]:
        """
        Mesmo esquema do GBM, mas os inner paths são Heston.
        H(t,S) (dependência em v é "integrada" via a simulação).
        """
        S_path = self.S_path
        v_path = self.v_path
        M, N_time = S_path.shape
        K_strike = self.K
        mu = self.mu
        dt = self.dt

        S_min = S_path.min()
        S_max = S_path.max()
        margin = 0.1 * (S_max - S_min)
        S_min -= margin
        S_max += margin

        n_S_grid = 50
        n_inner = 200

        S_grid = np.linspace(S_min, S_max, n_S_grid)

        H_grid = np.zeros((N_time, n_S_grid))
        Delta_grid = np.zeros((N_time, n_S_grid))
        Gamma_grid = np.zeros((N_time, n_S_grid))
        dHdt_grid = np.zeros((N_time, n_S_grid))
        d2H_dt_dS_grid = np.zeros((N_time, n_S_grid))
        d3H_dS3_grid = np.zeros((N_time, n_S_grid))

        # terminal
        H_grid[-1, :] = np.maximum(S_grid - K_strike, 0.0)

        # backward em t – nested MC Heston
        for n in reversed(range(N_time - 1)):
            # v médio na data (para iniciar inner paths)
            v_mean_n = float(np.mean(v_path[:, n]))
            v_mean_n = max(v_mean_n, 1e-8)

            for k, s0 in enumerate(S_grid):
                S_inner = np.full(n_inner, s0, float)
                v_inner = np.full(n_inner, v_mean_n, float)

                for j in range(n, N_time - 1):
                    Z1 = np.random.normal(0.0, 1.0, size=n_inner)
                    Z2 = np.random.normal(0.0, 1.0, size=n_inner)
                    dW1_inner = np.sqrt(dt) * Z1
                    dW2_ind_inner = np.sqrt(dt) * Z2
                    dW2_inner = self.rho * dW1_inner + np.sqrt(1.0 - self.rho**2) * dW2_ind_inner

                    v_inner = np.maximum(v_inner, 1e-8)
                    S_inner = S_inner + mu * S_inner * dt + np.sqrt(v_inner) * S_inner * dW1_inner
                    v_inner = (
                        v_inner
                        + self.kappa * (self.theta - v_inner) * dt
                        + self.sigma_v * np.sqrt(v_inner) * dW2_inner
                    )
                    v_inner = np.maximum(v_inner, 1e-8)

                payoff_inner = np.maximum(S_inner - K_strike, 0.0)
                H_grid[n, k] = payoff_inner.mean()

        # derivadas espaciais em S (como no GBM)
        dS = S_grid[1] - S_grid[0]

        for n in range(N_time):
            for k in range(1, n_S_grid - 1):
                Delta_grid[n, k] = (H_grid[n, k + 1] - H_grid[n, k - 1]) / (2.0 * dS)
                Gamma_grid[n, k] = (H_grid[n, k + 1] - 2.0 * H_grid[n, k] + H_grid[n, k - 1]) / (dS ** 2)

            Delta_grid[n, 0] = Delta_grid[n, 1]
            Delta_grid[n, -1] = Delta_grid[n, -2]
            Gamma_grid[n, 0] = Gamma_grid[n, 1]
            Gamma_grid[n, -1] = Gamma_grid[n, -2]

            for k in range(1, n_S_grid - 1):
                d3H_dS3_grid[n, k] = (Gamma_grid[n, k + 1] - Gamma_grid[n, k - 1]) / (2.0 * dS)

            d3H_dS3_grid[n, 0] = d3H_dS3_grid[n, 1]
            d3H_dS3_grid[n, -1] = d3H_dS3_grid[n, -2]

        # derivadas em t
        for n in range(1, N_time - 1):
            dHdt_grid[n, :] = (H_grid[n + 1, :] - H_grid[n - 1, :]) / (2.0 * dt)
        dHdt_grid[0, :] = (H_grid[1, :] - H_grid[0, :]) / dt
        dHdt_grid[-1, :] = (H_grid[-1, :] - H_grid[-2, :]) / dt

        # mista t,S
        for n in range(N_time):
            for k in range(1, n_S_grid - 1):
                d2H_dt_dS_grid[n, k] = (dHdt_grid[n, k + 1] - dHdt_grid[n, k - 1]) / (2.0 * dS)
            d2H_dt_dS_grid[n, 0] = d2H_dt_dS_grid[n, 1]
            d2H_dt_dS_grid[n, -1] = d2H_dt_dS_grid[n, -2]

        # interpola para as paths
        H = np.zeros((M, N_time))
        dH_dS = np.zeros((M, N_time))
        d2H_dSS = np.zeros((M, N_time))
        dH_dt = np.zeros((M, N_time))
        d2H_dt_dS = np.zeros((M, N_time))
        d3H_dS3 = np.zeros((M, N_time))

        for n in range(N_time):
            S_n = S_path[:, n]
            H[:, n]          = np.interp(S_n, S_grid, H_grid[n, :])
            dH_dS[:, n]      = np.interp(S_n, S_grid, Delta_grid[n, :])
            d2H_dSS[:, n]    = np.interp(S_n, S_grid, Gamma_grid[n, :])
            dH_dt[:, n]      = np.interp(S_n, S_grid, dHdt_grid[n, :])
            d2H_dt_dS[:, n]  = np.interp(S_n, S_grid, d2H_dt_dS_grid[n, :])
            d3H_dS3[:, n]    = np.interp(S_n, S_grid, d3H_dS3_grid[n, :])

        self.S_grid = S_grid
        self.H_grid = H_grid
        self.Delta_grid = Delta_grid
        self.Gamma_grid = Gamma_grid
        self.dHdt_grid = dHdt_grid
        self.d2H_dt_dS_grid = d2H_dt_dS_grid
        self.d3H_dS3_grid = d3H_dS3_grid

        dH_dt = dH_dt / 100.0
        d2H_dt_dS = d2H_dt_dS / 100.0

        self.H = H
        self.dH_dS = dH_dS
        self.d2H_dSS = d2H_dSS
        self.dH_dt = dH_dt
        self.d2H_dt_dS = d2H_dt_dS
        self.d3H_dS3 = d3H_dS3

        return H, dH_dS, d2H_dSS, dH_dt, d2H_dt_dS, d3H_dS3

    # ============================================================
    # 1. Control initialization
    # ============================================================

    def init_control(self, kind: str = "delta") -> np.ndarray:
        S_path = self.S_path
        dH_dS = self.dH_dS
        M, steps = S_path.shape

        if kind == "delta":
            h0 = dH_dS[:, :-1].copy()
        elif kind == "zero":
            h0 = np.zeros((M, steps - 1))
        else:
            raise ValueError("Unknown control initialization kind")

        return h0

    # ============================================================
    # 2. Forward P&L
    # ============================================================

    def forward_PL(self, h: np.ndarray, L0: float = 0.0) -> np.ndarray:
        S_path = self.S_path
        H      = self.H

        M, steps = S_path.shape
        assert h.shape == (M, steps - 1)

        L = np.zeros((M, steps))
        L[:, 0] = L0

        for n in range(steps - 1):
            dS = S_path[:, n + 1] - S_path[:, n]
            dH = H[:, n + 1]      - H[:, n]
            L[:, n + 1] = L[:, n] + h[:, n] * dS - dH

        return L

    # ============================================================
    # 3. Risk functional  (mesmo do GBM)
    # ============================================================

    @staticmethod
    def risk_function(LT: np.ndarray, risk_type: str, **kwargs) -> float:
        LT = np.asarray(LT)

        if risk_type == "ele":
            a = kwargs.get("a")
            if a is None:
                raise ValueError("Parameter 'a' is required for 'ele'.")
            rho = np.mean(np.exp(-a * LT))

        elif risk_type == "elw":
            k = kwargs.get("k")
            if k is None:
                raise ValueError("Parameter 'k' is required for 'elw'.")
            loss = -np.minimum(LT, 0.0)
            rho = np.mean(np.exp(loss**k))

        elif risk_type == "entl":
            gamma = kwargs.get("gamma")
            if gamma is None:
                raise ValueError("Parameter 'gamma' is required for 'entl'.")
            w = np.exp(-gamma * LT)
            den = np.mean(w)
            rho = (1.0 / gamma) * np.log(den)

        elif risk_type == "ente":
            gamma = kwargs.get("gamma")
            a = kwargs.get("a")
            if gamma is None or a is None:
                raise ValueError("Parameters 'gamma' and 'a' are required for 'ente'.")
            v = np.exp(-a * LT)
            w = np.exp(gamma * v)
            den = np.mean(w)
            rho = (1.0 / gamma) * np.log(den)

        elif risk_type == "entw":
            gamma = kwargs.get("gamma")
            k     = kwargs.get("k")
            scale = kwargs.get("scale", 20.0)
            if gamma is None or k is None:
                raise ValueError("Parameters 'gamma' and 'k' are required for 'entw'.")
            LT_scaled = LT / scale
            loss = -np.minimum(LT_scaled, 0.0)
            g = np.clip(loss**k, 0.0, 10.0)
            v = np.exp(g)
            z = np.clip(gamma * v, -20.0, 20.0)
            w = np.exp(z)
            den = np.mean(w)
            rho = (1.0 / gamma) * np.log(den)

        elif risk_type == "es":
            beta = kwargs.get("beta")
            if beta is None:
                raise ValueError("Parameter 'beta' is required for 'es'.")
            alpha = kwargs.get("alpha")
            if alpha is None:
                alpha = np.quantile(LT, beta)
            excess = np.maximum(LT - alpha, 0.0)
            rho = alpha + (1.0 / (1.0 - beta)) * np.mean(excess)

        else:
            raise ValueError(f"Unknown risk_type '{risk_type}'.")

        return rho

    # ============================================================
    # 4. Terminal adjoint (mesmo do GBM)
    # ============================================================

    @staticmethod
    def terminal_adjoint(LT: np.ndarray, risk_type: str, **kwargs) -> Tuple[np.ndarray, Dict]:
        LT = np.asarray(LT)

        if risk_type == "ele":
            a = kwargs.get("a")
            if a is None:
                raise ValueError("Parameter 'a' is required for 'ele'.")
            pT = -a * np.exp(-a * LT)
            info = {}

        elif risk_type == "elw":
            k = kwargs.get("k")
            if k is None:
                raise ValueError("Parameter 'k' is required for 'elw'.")
            loss = -np.minimum(LT, 0.0)
            pT = -k * (loss**(k - 1.0)) * np.exp(loss**k)
            info = {}

        elif risk_type == "entl":
            gamma = kwargs.get("gamma")
            if gamma is None:
                raise ValueError("Parameter 'gamma' is required for 'entl'.")
            w = np.exp(-gamma * LT)
            den = np.mean(w)
            if den <= 0.0:
                raise ValueError("Non-positive denominator in entropic-linear.")
            pT = -w / den
            info = {"denominator": den}

        elif risk_type == "ente":
            gamma = kwargs.get("gamma")
            a = kwargs.get("a")
            if gamma is None or a is None:
                raise ValueError("Parameters 'gamma' and 'a' are required for 'ente'.")
            v = np.exp(-a * LT)
            w = np.exp(gamma * v)
            den = np.mean(w)
            if den <= 0.0:
                raise ValueError("Non-positive denominator in entropic-exponential.")
            pT = -a * v * w / den
            info = {"denominator": den}

        elif risk_type == "entw":
            gamma = kwargs.get("gamma")
            k     = kwargs.get("k")
            scale = kwargs.get("scale", 20.0)
            if gamma is None or k is None:
                raise ValueError("Parameters 'gamma' and 'k' are required for 'entw'.")
            LT_scaled = LT / scale
            loss = -np.minimum(LT_scaled, 0.0)
            g = np.clip(loss**k, 0.0, 10.0)
            v = np.exp(g)
            z = np.clip(gamma * v, -20.0, 20.0)
            w = np.exp(z)
            den = np.mean(w)
            if den <= 0.0:
                raise ValueError("Non-positive denominator in entropic-Weibull.")
            v_prime = -k * (loss**(k - 1.0)) * v
            p_base = v_prime * w / den
            pT = (1.0 / scale) * p_base
            info = {"denominator": den}

        elif risk_type == "es":
            beta = kwargs.get("beta")
            if beta is None:
                raise ValueError("Parameter 'beta' is required for 'es'.")
            alpha = kwargs.get("alpha")
            if alpha is None:
                alpha = np.quantile(LT, beta)
            indicator = (LT >= alpha).astype(float)
            pT = indicator / (1.0 - beta)
            info = {"alpha_star": alpha}

        else:
            raise ValueError(f"Unknown risk_type '{risk_type}'.")

        return pT, info

    # ============================================================
    # 5. Backward adjoint (driver tipo GBM, mas com v_t em σ_t)
    # ============================================================

    def backward_adjoint(self, pT):
        S_path    = self.S_path
        v_path    = self.v_path
        dW        = self.dW
        d2H_dt_dS = self.d2H_dt_dS
        d2H_dSS   = self.d2H_dSS
        d3H_dS3   = self.d3H_dS3
        dt = self.dt

        M, steps = S_path.shape
        p = np.zeros((M, steps))
        q = np.zeros((M, steps-1))
        p[:, -1] = pT

        for n in reversed(range(steps-1)):
            Sn = S_path[:, n]
            vn = np.maximum(v_path[:, n], 1e-8)
            sigma_n = np.sqrt(vn)
            dWn = dW[:, n]

            d2H_dt_dS_n = 100.0 * d2H_dt_dS[:, n]
            d2H_dSS_n   = d2H_dSS[:, n]
            d3H_dS3_n   = d3H_dS3[:, n]

            A_n = d2H_dt_dS_n + 0.5 * (sigma_n**2) * (
                2.0 * Sn * d2H_dSS_n + (Sn**2) * d3H_dS3_n
            )

            p_next = p[:, n+1]
            p_n = p_next - A_n * p_next * dt
            p[:, n] = p_n

            denom = dWn.copy()
            denom[np.abs(denom) < 1e-8] = 1e-8
            q[:, n] = (p_next - p_n - A_n * p_n * dt) / denom

        return p, q

    # ============================================================
    # 6. Gradient (aqui entra o ∂_h H correto do Heston sem q^ν)
    # ============================================================

    def compute_gradient(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        G_n ≈ (μ S_n + κ(θ - v_n)) p_n + sqrt(v_n) S_n q_n.
        (termo em σ_v sqrt(v_n) q^ν é ignorado porque o código
        não tem o segundo componente de q.)
        """
        S_path = self.S_path
        v_path = self.v_path
        mu = self.mu
        kappa = self.kappa
        theta = self.theta

        M, steps = S_path.shape
        assert p.shape == (M, steps)
        assert q.shape == (M, steps - 1)

        S_trunc = S_path[:, :-1]
        v_trunc = v_path[:, :-1]
        sqrt_v  = np.sqrt(np.maximum(v_trunc, 1e-8))

        G = (mu * S_trunc + kappa * (theta - v_trunc)) * p[:, :-1] \
            + sqrt_v * S_trunc * q
        return G

    # ============================================================
    # 7. Control update
    # ============================================================

    @staticmethod
    def update_control(
        h: np.ndarray,
        G: np.ndarray,
        alpha: float,
        max_G: float = 0.1,
    ) -> np.ndarray:
        assert h.shape == G.shape
        G_clipped = np.clip(G, -max_G, max_G)
        return h - alpha * G_clipped

    # ============================================================
    # 8. Main optimization loop
    # ============================================================

    def optimize_hedge(self,
                       risk_type: str,
                       risk_kwargs: Dict,
                       max_iter: int = 50,
                       tol: float = 1e-4,
                       alpha: float = 1e-3,
                       verbose: bool = True) -> Tuple[np.ndarray, List[Dict]]:

        h = self.init_control(kind="delta")
        history: List[Dict] = []

        for k in range(max_iter):
            L = self.forward_PL(h, L0=0.0)
            LT = L[:, -1]

            rho = self.risk_function(LT, risk_type, **risk_kwargs)
            pT, info = self.terminal_adjoint(LT, risk_type, **risk_kwargs)
            p, q = self.backward_adjoint(pT)

            G = self.compute_gradient(p, q)
            grad_norm = np.mean(G**2)

            history.append({"iter": k, "rho": rho, "grad_norm": grad_norm})

            if verbose:
                print(f"iter {k:3d} | rho={rho:.6f} | grad_norm={grad_norm:.6e}")

            if grad_norm < tol:
                break

            h = self.update_control(h, G, alpha)

        return h, history
