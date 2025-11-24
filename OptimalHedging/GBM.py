from OptimalHedging.Simulator import BaseSimulator
import numpy as np
from typing import Dict, List, Tuple


class GBMSimulator(BaseSimulator):
    """
    Simulator for Geometric Brownian Motion (GBM) + optimal hedging machinery.
    """

    # ============================================================
    # 0. Underlying S and derivative H (your original code)
    # ============================================================

    def simulate_S(self) -> np.ndarray:
        """
        Simulate Geometric Brownian Motion (GBM) paths using a fully vectorized Euler–Maruyama scheme.

            dS = mu * S * dt + sigma * S * dW

        Returns
        -------
        S : ndarray, shape (M, steps)
            Simulated GBM asset price paths.
        """
        steps = int(np.round((self.T - self.t0) / self.dt))
        dW = np.random.normal(
            loc=0.0,
            scale=np.sqrt(self.dt),
            size=(self.M, steps - 1)
        )  # (M, steps-1)

        # multiplicative Euler step: S_{n+1} = S_n * (1 + mu*dt + sigma*dW)
        factors = 1 + self.mu * self.dt + self.sigma * dW
        factors = np.hstack((np.ones((self.M, 1)), factors))  # insert initial factor = 1
        S_path = self.S0 * np.cumprod(factors, axis=1)        # (M, steps)

        self.S_path = S_path
        self.dW = dW

        return S_path

    def simulate_H(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                  np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward estimation of H_t and its derivatives on a space grid, using
        nested Monte Carlo and finite differences.

        Uses
        ----
        self.S_path : ndarray, shape (M, N_time)
            Simulated GBM paths.

        Produces (pathwise, via interpolation on the S_grid)
        -----------------------------------------------
        H        : (M, N_time)
        dH_dS    : (M, N_time)
        d2H_dSS  : (M, N_time)
        dH_dt    : (M, N_time)
        d2H_dt_dS: (M, N_time)
        d3H_dS3  : (M, N_time)
        """
        S_path = self.S_path
        M, N_time = S_path.shape
        K_strike = self.K
        mu = self.mu
        sigma = self.sigma
        dt = self.dt

        # ------------------------------------------------------------
        # 1) Build a space grid S_grid covering the simulated paths
        # ------------------------------------------------------------
        S_min = S_path.min()
        S_max = S_path.max()
        margin = 0.1 * (S_max - S_min)
        S_min -= margin
        S_max += margin

        n_S_grid = 50    # number of space points (tunable)
        n_inner = 200    # inner MC paths per (t_n, s_k) (tunable)

        S_grid = np.linspace(S_min, S_max, n_S_grid)  # (n_S_grid,)

        # Tables on the space grid:
        H_grid = np.zeros((N_time, n_S_grid))
        Delta_grid = np.zeros((N_time, n_S_grid))
        Gamma_grid = np.zeros((N_time, n_S_grid))
        dHdt_grid = np.zeros((N_time, n_S_grid))
        d2H_dt_dS_grid = np.zeros((N_time, n_S_grid))
        d3H_dS3_grid = np.zeros((N_time, n_S_grid))

        # ------------------------------------------------------------
        # 2) Terminal condition at T: H(T,s) = max(s - K, 0)
        # ------------------------------------------------------------
        H_grid[-1, :] = np.maximum(S_grid - K_strike, 0.0)

        # ------------------------------------------------------------
        # 3) Backward in time on the grid via nested Monte Carlo
        # ------------------------------------------------------------
        for n in reversed(range(N_time - 1)):  # n = N_time-2, ..., 0
            for k, s0 in enumerate(S_grid):
                # inner paths starting at S(t_n) = s0
                S_inner = np.full(n_inner, s0, dtype=float)

                # simulate from t_n to T using Euler–Maruyama
                for j in range(n, N_time - 1):
                    dW_inner = np.random.normal(0.0, np.sqrt(dt), size=n_inner)
                    S_inner = S_inner + mu * S_inner * dt + sigma * S_inner * dW_inner

                payoff_inner = np.maximum(S_inner - K_strike, 0.0)
                H_grid[n, k] = payoff_inner.mean()

        # ------------------------------------------------------------
        # 4) Spatial derivatives via finite differences
        # ------------------------------------------------------------
        dS = S_grid[1] - S_grid[0]  # uniform grid

        for n in range(N_time):
            # First and second derivative in S (interior)
            for k in range(1, n_S_grid - 1):
                Delta_grid[n, k] = (H_grid[n, k + 1] - H_grid[n, k - 1]) / (2.0 * dS)
                Gamma_grid[n, k] = (H_grid[n, k + 1] - 2.0 * H_grid[n, k] + H_grid[n, k - 1]) / (dS ** 2)

            # simple boundary handling: copy neighbors
            Delta_grid[n, 0] = Delta_grid[n, 1]
            Delta_grid[n, -1] = Delta_grid[n, -2]
            Gamma_grid[n, 0] = Gamma_grid[n, 1]
            Gamma_grid[n, -1] = Gamma_grid[n, -2]

            # Third derivative in S: derivative of Gamma_grid in S
            for k in range(1, n_S_grid - 1):
                d3H_dS3_grid[n, k] = (Gamma_grid[n, k + 1] - Gamma_grid[n, k - 1]) / (2.0 * dS)

            d3H_dS3_grid[n, 0] = d3H_dS3_grid[n, 1]
            d3H_dS3_grid[n, -1] = d3H_dS3_grid[n, -2]

        # ------------------------------------------------------------
        # 5) Time derivative via finite differences in t
        # ------------------------------------------------------------
        for n in range(1, N_time - 1):
            dHdt_grid[n, :] = (H_grid[n + 1, :] - H_grid[n - 1, :]) / (2.0 * dt)

        # forward difference at initial time
        dHdt_grid[0, :] = (H_grid[1, :] - H_grid[0, :]) / dt
        # backward difference at terminal time
        dHdt_grid[-1, :] = (H_grid[-1, :] - H_grid[-2, :]) / dt

        # ------------------------------------------------------------
        # 6) Mixed derivative in t and S
        # ------------------------------------------------------------
        for n in range(N_time):
            for k in range(1, n_S_grid - 1):
                d2H_dt_dS_grid[n, k] = (dHdt_grid[n, k + 1] - dHdt_grid[n, k - 1]) / (2.0 * dS)

            d2H_dt_dS_grid[n, 0] = d2H_dt_dS_grid[n, 1]
            d2H_dt_dS_grid[n, -1] = d2H_dt_dS_grid[n, -2]

        # ------------------------------------------------------------
        # 7) Interpolate everything to the simulated paths S_path
        # ------------------------------------------------------------
        H = np.zeros((M, N_time))
        dH_dS = np.zeros((M, N_time))
        d2H_dSS = np.zeros((M, N_time))
        dH_dt = np.zeros((M, N_time))
        d2H_dt_dS = np.zeros((M, N_time))
        d3H_dS3 = np.zeros((M, N_time))

        for n in range(N_time):
            S_n = S_path[:, n]  # (M,)

            H[:, n] = np.interp(S_n, S_grid, H_grid[n, :])
            dH_dS[:, n] = np.interp(S_n, S_grid, Delta_grid[n, :])
            d2H_dSS[:, n] = np.interp(S_n, S_grid, Gamma_grid[n, :])
            dH_dt[:, n] = np.interp(S_n, S_grid, dHdt_grid[n, :])
            d2H_dt_dS[:, n] = np.interp(S_n, S_grid, d2H_dt_dS_grid[n, :])
            d3H_dS3[:, n] = np.interp(S_n, S_grid, d3H_dS3_grid[n, :])

        # ------------------------------------------------------------
        # 8) Store everything on self
        # ------------------------------------------------------------
        self.S_grid = S_grid
        self.H_grid = H_grid
        self.Delta_grid = Delta_grid
        self.Gamma_grid = Gamma_grid
        self.dHdt_grid = dHdt_grid
        self.d2H_dt_dS_grid = d2H_dt_dS_grid
        self.d3H_dS3_grid = d3H_dS3_grid

        # rescaling in time-derivatives (numerical stabilization)
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
        """
        Simulate P&L L_t using portfolio value:

            V_t^h = h_t S_t - H_t
            ΔL_n = V_{n+1}^h - V_n^h
                = h_n (S_{n+1} - S_n) - (H_{n+1} - H_n)

        Parameters
        ----------
        h : ndarray, shape (M, steps-1)
            Hedge ratio on each interval [t_n, t_{n+1}).
        L0 : float, default=0.0
            Initial P&L (relative).

        Returns
        -------
        L : ndarray, shape (M, steps)
            P&L paths over time.
        """
        S_path = self.S_path   # (M, steps)
        H      = self.H        # (M, steps)

        M, steps = S_path.shape
        assert h.shape == (M, steps - 1)

        L = np.zeros((M, steps))
        L[:, 0] = L0

        for n in range(steps - 1):
            dS = S_path[:, n + 1] - S_path[:, n]
            dH = H[:, n + 1] - H[:, n]

            # incremental P&L: h_n * ΔS - ΔH
            L[:, n + 1] = L[:, n] + h[:, n] * dS - dH

        return L


    # ============================================================
    # 3. Risk functional
    # ============================================================

    @staticmethod
    def risk_function(LT: np.ndarray, risk_type: str, **kwargs) -> float:
        """
        Compute rho_u(L_T) for different risk specifications.

        risk_type in {"ele","elw","entl","ente","entw","es"}.
        """
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
            rho = (1.0 / gamma) * np.log(np.mean(w))

        elif risk_type == "ente":
            gamma = kwargs.get("gamma")
            a = kwargs.get("a")
            if gamma is None or a is None:
                raise ValueError("Parameters 'gamma' and 'a' are required for 'ente'.")
            v = np.exp(-a * LT)
            w = np.exp(gamma * v)
            rho = (1.0 / gamma) * np.log(np.mean(w))

        elif risk_type == "entw":
            # Entropic with Weibull-type utility:
            # v(X)   = exp( ( -min(X/scale, 0) )^k )
            # rho(X) = (1/gamma) log E[ exp( gamma * v(X) ) ]
            gamma = kwargs.get("gamma")
            k     = kwargs.get("k")
            scale = kwargs.get("scale", 20.0)  # internal scaling to stabilize numerics
            if gamma is None or k is None:
                raise ValueError("Parameters 'gamma' and 'k' are required for risk_type='entw'.")

            # Scale the P&L to reduce loss magnitude inside exponentials
            LT_scaled = LT / scale

            # Loss = -min(LT_scaled, 0)  >= 0
            loss = -np.minimum(LT_scaled, 0.0)

            # Inner power: g = loss^k
            g = loss**k
            # Clip g to avoid huge exponents in exp(g)
            g = np.clip(g, 0.0, 10.0)  # exp(10) ~ 2.2e4 -> still safe

            # v = exp(g)
            v = np.exp(g)

            # Outer exponent: z = gamma * v
            z = gamma * v
            # Clip again before exp(z)
            z = np.clip(z, -20.0, 20.0)  # exp(20) ~ 4.8e8 -> safe

            w = np.exp(z)  # exp(gamma * v)
            rho = (1.0 / gamma) * np.log(np.mean(w))


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
    # 4. Terminal adjoint
    # ============================================================

    @staticmethod
    def terminal_adjoint(LT: np.ndarray, risk_type: str, **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        Compute p_T = Upsilon(L_T) for the given risk_type.
        """
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
            # Terminal adjoint for entropic-Weibull with internal scaling:
            gamma = kwargs.get("gamma")
            k     = kwargs.get("k")
            scale = kwargs.get("scale", 20.0)
            if gamma is None or k is None:
                raise ValueError("Parameters 'gamma' and 'k' are required for risk_type='entw'.")

            LT_scaled = LT / scale
            loss = -np.minimum(LT_scaled, 0.0)  # >= 0

            # Inner power and its exp
            g = loss**k
            g = np.clip(g, 0.0, 10.0)
            v = np.exp(g)                       # v(L/scale)

            # Outer exponent
            z = gamma * v
            z = np.clip(z, -20.0, 20.0)
            w = np.exp(z)                       # exp(gamma * v)

            den = np.mean(w)
            if den <= 0.0:
                raise ValueError("Non-positive denominator in entropic-Weibull Upsilon.")

            # v'(y) where y = LT_scaled:
            # loss = -min(y,0), so for y<0: loss = -y -> d(loss)/dy = -1
            # g = loss^k -> g' = k * loss^(k-1) * d(loss)/dy = -k * loss^(k-1)
            # v = exp(g) -> v' = v * g' = -k * loss^(k-1) * v
            v_prime = -k * (loss**(k - 1.0)) * v

            # Base adjoint w.r.t. LT_scaled:
            p_base = v_prime * w / den

            # Chain rule: LT_scaled = LT / scale -> d/dLT = (1/scale) d/d(LT_scaled)
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
    # 5. Backward adjoint (simplified driver)
    # ============================================================

    def backward_adjoint(self, pT):
        S_path    = self.S_path
        dW        = self.dW
        d2H_dt_dS = self.d2H_dt_dS
        d2H_dSS   = self.d2H_dSS
        d3H_dS3   = self.d3H_dS3
        mu, sigma, dt = self.mu, self.sigma, self.dt

        M, steps = S_path.shape
        p = np.zeros((M, steps))
        q = np.zeros((M, steps-1))
        p[:, -1] = pT

        for n in reversed(range(steps-1)):
            Sn = S_path[:, n]
            dWn = dW[:, n]

            d2H_dt_dS_n = 100.0 * d2H_dt_dS[:, n]
            d2H_dSS_n   = d2H_dSS[:, n]
            d3H_dS3_n   = d3H_dS3[:, n]

            A_n = d2H_dt_dS_n + 0.5 * sigma**2 * (
                2.0 * Sn * d2H_dSS_n + (Sn**2) * d3H_dS3_n
            )

            # aqui uso um esquema explícito simples para p_n
            p_next = p[:, n+1]
            # por ex.: p_n = p_next - A_n p_next dt  (Euler backward aproximado)
            p_n = p_next - A_n * p_next * dt
            p[:, n] = p_n

            # e isolamos q_n a partir da variação:
            q[:, n] = (p_next - p_n - A_n * p_n * dt) / dWn

        return p, q


    # ============================================================
    # 6. Gradient (optimality condition)
    # ============================================================

    def compute_gradient(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """
        Compute G_n = μ S_n p_n + σ S_n q_n.
        """
        S_path = self.S_path
        mu = self.mu
        sigma = self.sigma

        M, steps = S_path.shape
        assert p.shape == (M, steps)
        assert q.shape == (M, steps - 1)

        S_trunc = S_path[:, :-1]
        G = mu * S_trunc * p[:, :-1] + sigma * S_trunc * q
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
        """
        Simple gradient step with G clipped to a small interval.

        h_new = h - alpha * G_clipped
        with G_clipped in [-max_G, max_G].
        """
        assert h.shape == G.shape

        # keep G small
        G_clipped = np.clip(G, -max_G, max_G)

        # standard update
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
        """
        Main shooting loop in control space.
        """
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
