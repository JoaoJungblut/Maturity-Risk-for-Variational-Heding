from OptimalHedging.Simulator import BaseSimulator
import numpy as np
from typing import Dict, List, Tuple


class JumpDiffusionSimulator(BaseSimulator):
    """
    Merton Jump Diffusion + máquina de hedge.
    """

    def __init__(self,
                 lam: float,
                 meanJ: float,
                 stdJ: float,
                 **base_kwargs):
        """
        Initialize the jump-diffusion simulator.

        Parameters
        ----------
        lam : float
            Jump intensity (Poisson rate).
        meanJ : float
            Mean of jump size distribution.
        stdJ : float
            Standard deviation of jump size distribution.
        base_kwargs : dict
            Same arguments as BaseSimulator/GBMSimulator.
        """
        super().__init__(**base_kwargs)

        self.lam = lam
        self.meanJ = meanJ
        self.stdJ = stdJ


    # ============================================================
    # 0. Underlying S and derivative H
    # ============================================================

    def simulate_S(self) -> np.ndarray:
        """
        Merton: dS = mu*S*dt + sigma*S*dW + S*(J-1)*dN.
        """
        steps = int(np.round((self.T - self.t0) / self.dt))

        dW = np.random.normal(0.0, np.sqrt(self.dt), size=(self.M, steps - 1))
        dN = np.random.poisson(self.lam * self.dt, size=(self.M, steps - 1))
        ZJ = np.random.normal(0.0, 1.0, size=(self.M, steps - 1))
        J  = np.exp(self.mJ + self.sJ * ZJ)

        factors = 1 + self.mu * self.dt + self.sigma * dW + (J - 1.0) * dN
        factors = np.hstack((np.ones((self.M, 1)), factors))
        S = self.S0 * np.cumprod(factors, axis=1)

        self.S_Jump = S
        self.dW_Jump = dW
        self.dN_Jump = dN
        self.ZJ_Jump = ZJ
        self.J_Jump  = J

        return S, J

    def simulate_H(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                                  np.ndarray, np.ndarray, np.ndarray]:
        """
        Mesmo esquema do GBM, mas inner paths seguem Merton.
        """
        S_path = self.S_path
        M, N_time = S_path.shape
        K_strike = self.K
        mu = self.mu
        sigma = self.sigma
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

        # backward – nested MC com Merton
        for n in reversed(range(N_time - 1)):
            for k, s0 in enumerate(S_grid):
                S_inner = np.full(n_inner, s0, float)

                for j in range(n, N_time - 1):
                    dW_inner = np.random.normal(0.0, np.sqrt(dt), size=n_inner)
                    dN_inner = np.random.poisson(self.lam * dt, size=n_inner)
                    ZJ_inner = np.random.normal(0.0, 1.0, size=n_inner)
                    J_inner  = np.exp(self.mJ + self.sJ * ZJ_inner)

                    S_inner = (
                        S_inner
                        + mu * S_inner * dt
                        + sigma * S_inner * dW_inner
                        + S_inner * (J_inner - 1.0) * dN_inner
                    )

                payoff_inner = np.maximum(S_inner - K_strike, 0.0)
                H_grid[n, k] = payoff_inner.mean()

        # derivadas espaciais
        dS = S_grid[1] - S_grid[0]

        for n in range(N_time):
            for k in range(1, n_S_grid - 1):
                Delta_grid[n, k] = (H_grid[n, k + 1] - H_grid[n, k - 1]) / (2.0 * dS)
                Gamma_grid[n, k] = (
                    H_grid[n, k + 1] - 2.0 * H_grid[n, k] + H_grid[n, k - 1]
                ) / (dS ** 2)

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

        # interp paths
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
    # 5. Backward adjoint: agora com r_t (salto)
    # ============================================================

    def backward_adjoint(self, pT):
        S_path    = self.S_path
        dW        = self.dW
        dN        = self.dN
        d2H_dt_dS = self.d2H_dt_dS
        d2H_dSS   = self.d2H_dSS
        d3H_dS3   = self.d3H_dS3
        mu, sigma, dt = self.mu, self.sigma, self.dt

        M, steps = S_path.shape
        p = np.zeros((M, steps))
        q = np.zeros((M, steps-1))
        r = np.zeros((M, steps-1))
        p[:, -1] = pT

        for n in reversed(range(steps-1)):
            Sn  = S_path[:, n]
            dWn = dW[:, n]
            dNn = dN[:, n]

            d2H_dt_dS_n = 100.0 * d2H_dt_dS[:, n]
            d2H_dSS_n   = d2H_dSS[:, n]
            d3H_dS3_n   = d3H_dS3[:, n]

            # A_n igual ao GBM (driver difusivo)
            A_n = d2H_dt_dS_n + 0.5 * sigma**2 * (
                2.0 * Sn * d2H_dSS_n + (Sn**2) * d3H_dS3_n
            )

            p_next = p[:, n+1]
            p_n = p_next - A_n * p_next * dt
            p[:, n] = p_n

            # parte martingale: q dW + r dN
            mart = p_next - p_n - A_n * p_n * dt

            # estima q usando apenas caminhos sem salto
            q_n = np.zeros(M)
            mask_nojump = (dNn == 0) & (np.abs(dWn) > 1e-8)
            q_n[mask_nojump] = mart[mask_nojump] / dWn[mask_nojump]

            # para o resto, reaproveita q do passo seguinte ou zera
            if n < steps - 2:
                q_n[~mask_nojump] = q[:, n+1][~mask_nojump]
            q[:, n] = q_n

            # agora estima r para caminhos que saltaram
            r_n = np.zeros(M)
            mask_jump = (dNn > 0)
            if np.any(mask_jump):
                r_n[mask_jump] = (mart[mask_jump] - q_n[mask_jump] * dWn[mask_jump]) / dNn[mask_jump]
            r[:, n] = r_n

        return p, q, r

    # ============================================================
    # 6. Gradient: ∂_h H = μ S p + σ S q + S(J-1) r
    # ============================================================

    def compute_gradient(self, p: np.ndarray, q: np.ndarray, r: np.ndarray) -> np.ndarray:
        S_path = self.S_path
        J      = self.J
        mu = self.mu
        sigma = self.sigma

        M, steps = S_path.shape
        assert p.shape == (M, steps)
        assert q.shape == (M, steps - 1)
        assert r.shape == (M, steps - 1)

        S_trunc = S_path[:, :-1]
        J_trunc = J  # (M, steps-1)

        G = mu * S_trunc * p[:, :-1] \
            + sigma * S_trunc * q \
            + S_trunc * (J_trunc - 1.0) * r

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

            p, q, r = self.backward_adjoint(pT)
            G = self.compute_gradient(p, q, r)
            grad_norm = np.mean(G**2)

            history.append({"iter": k, "rho": rho, "grad_norm": grad_norm})

            if verbose:
                print(f"iter {k:3d} | rho={rho:.6f} | grad_norm={grad_norm:.6e}")

            if grad_norm < tol:
                break

            h = self.update_control(h, G, alpha)

        return h, history
