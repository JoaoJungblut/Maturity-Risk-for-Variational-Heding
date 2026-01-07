from OptimalHedging.Simulator import BaseSimulator
import OptimalHedging.tools as tools
import numpy as np
from typing import Dict, List, Tuple


class JumpDiffusionSimulator(BaseSimulator):
    """
    Simulator for Merton jump diffusion process with optimal hedging machinery.

    This class implements all model dependent components required to solve the
    optimal hedging problem under a jump diffusion dynamics. All methods below
    are specific to the jump diffusion model and are therefore implemented
    directly in this class. Only the terminal risk functional and the control
    update rule are inherited from BaseSimulator.

    Implemented methods
    -------------------
    - init :
        Initializes the simulator and forwards all parameters to BaseSimulator.

    - simulate_S :
        Simulates jump diffusion paths for the underlying asset and stores the
        asset paths together with the Brownian increments and jump information.

    - simulate_H :
        Computes the contingent claim price H t S and its time and space
        derivatives using a single pass LSMC regression on a Chebyshev basis.

    - init_control :
        Initializes the hedging control typically using the delta of the
        contingent claim as a smooth initial guess.

    - forward_PL :
        Simulates the forward profit and loss process associated with a given
        hedging strategy under the jump diffusion dynamics.

    - backward_adjoint :
        Solves the backward adjoint equation associated with the jump diffusion
        optimal hedging problem using regression based conditional expectations.

    - compute_gradient :
        Computes the jump diffusion specific optimality condition gradient used
        to update the control process.

    - optimize_hedge :
        Runs the full iterative optimization loop to compute an approximate
        optimal hedge under the chosen risk functional.

    - compute_MR :
        Computes the maturity risk by comparing optimal risks on sub horizons
        and on the full time horizon.

    Inherited from BaseSimulator
    ----------------------------
    - risk_function :
        Evaluation of the terminal composed risk functional.

    - terminal_adjoint :
        Computation of the terminal adjoint associated with the risk functional.

    - update_control :
        Generic control update rule based on a normalized gradient step.
    """

    def __init__(self,
                 lam: float,
                 meanJ: float,
                 stdJ: float,
                 **base_kwargs):
        """
        Initialize the jump diffusion simulator.

        Parameters
        ----------
        lam : float
            Jump intensity of the Poisson process.
        meanJ : float
            Mean of the jump size distribution.
        stdJ : float
            Standard deviation of the jump size distribution.
        base_kwargs : dict
            Same arguments as BaseSimulator
            S0 mu sigma K t0 T N M seed.
        """
        super().__init__(**base_kwargs)

        self.lam = float(lam)
        self.meanJ = float(meanJ)
        self.stdJ = float(stdJ)



    # ============================================================
    # 0. Underlying S and derivative H
    # ============================================================
    def simulate_S(self) -> np.ndarray:
        """
        Simulate Merton jump diffusion paths using a vectorized Euler scheme.

            dS = mu * S * dt + sigma * S * dW + S * (J - 1) * dN

        where dN is a Poisson process with intensity lam and J denotes the
        random jump size.

        Returns
        -------
        S : ndarray, shape (M, steps)
            Simulated underlying asset price paths.
        """
        dW = np.random.normal(loc=0.0,
                            scale=np.sqrt(self.dt),
                            size=(self.M, self.steps - 1))

        dN = np.random.poisson(lam=self.lam * self.dt,
                            size=(self.M, self.steps - 1))

        ZJ = np.random.normal(loc=0.0,
                            scale=1.0,
                            size=(self.M, self.steps - 1))

        J = np.exp(self.meanJ + self.stdJ * ZJ)

        # multiplicative Euler step with jumps
        factors = 1.0 + self.mu * self.dt + self.sigma * dW + (J - 1.0) * dN
        factors = np.hstack((np.ones((self.M, 1)), factors))

        S = self.S0 * np.cumprod(factors, axis=1)

        self.S_Jump = S
        self.dW_Jump = dW
        self.dN_Jump = dN
        self.ZJ_Jump = ZJ
        self.J_Jump = J

        return S

    def simulate_H(self,
                p_t: int = 1,
                p_x: int = 2,
                lam_ridge: float = 1e-2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate H(t,S) = E[(S_T - K)^+ | S_t = S] and derivatives using a single pass
        LSMC regression on the already simulated jump diffusion paths.

        Approach
        --------
        - Regress payoff Y = (S_T - K)^+ on a tensor Chebyshev basis in (t, x),
        where x = log(S), using ridge regularization.
        - Compute derivatives analytically from the fitted basis in (t, x).
        - Convert x derivatives to S derivatives via chain rule.
        - Additionally evaluate H and dH dS at the post jump state S_jump = J * S
        using the same fitted coefficients.

        Parameters
        ----------
        p_t : int, default=1
            Chebyshev degree in time
        p_x : int, default=2
            Chebyshev degree in x = log(S)
        lam_ridge : float, default=1e-2
            Ridge regularization

        Returns
        -------
        H : ndarray, shape (M, steps)
            Simulated contingent claim price paths.
        dH_dS : ndarray, shape (M, steps)
            First order derivative against underlying asset.
        d2H_dSS : ndarray, shape (M, steps)
            Second order derivative against underlying asset.
        dH_dt : ndarray, shape (M, steps)
            First order derivative against time.
        d2H_dt_dS : ndarray, shape (M, steps)
            Second order mixed derivative against time and underlying asset.
        d3H_dS3 : ndarray, shape (M, steps)
            Third order derivative against underlying asset.
        H_jump : ndarray, shape (M, steps)
            Price evaluated at the post jump state J times S.
        dH_dS_jump : ndarray, shape (M, steps)
            First order derivative evaluated at the post jump state J times S.
        """
        if getattr(self, "S_Jump", None) is None:
            raise ValueError("self.S_Jump is None. Run simulate_S() first.")
        if getattr(self, "J_Jump", None) is None:
            raise ValueError("self.J_Jump is None. Run simulate_S() first.")

        # ----------------------------
        # 1) Targets and state
        # ----------------------------
        Y = np.maximum(self.S_Jump[:, -1] - self.K, 0.0)

        S_path = np.maximum(self.S_Jump, 1e-300)
        J_path = np.maximum(self.J_Jump, 1e-300)

        X_path = np.log(S_path)                  # log(S)
        X_jump_path = np.log(S_path * J_path)    # log(J*S) = log(S) + log(J)

        t_grid = self.t0 + self.dt * np.arange(self.steps)

        # ----------------------------
        # 2) Scale (t, x) to [-1, 1]
        # ----------------------------
        x_min = float(min(X_path.min(), X_jump_path.min()))
        x_max = float(max(X_path.max(), X_jump_path.max()))
        if (not np.isfinite(x_min)) or (not np.isfinite(x_max)) or (x_max <= x_min):
            raise ValueError("Invalid X_path range for scaling.")

        zt = tools._minmax_scale(t_grid, self.t0, self.T)                # (steps,)
        zx = tools._minmax_scale(X_path.reshape(-1), x_min, x_max)       # (M*steps,)
        zx_jump = tools._minmax_scale(X_jump_path.reshape(-1), x_min, x_max)

        dz_dt = 2.0 / (self.T - self.t0)
        dz_dx = 2.0 / (x_max - x_min)

        # ----------------------------
        # 3) Chebyshev basis and derivatives in (t, x)
        # ----------------------------
        Tt = tools._cheb_eval_all(zt, p_t, deriv=0)
        dTt_dz = tools._cheb_eval_all(zt, p_t, deriv=1)
        dTt_dt = dTt_dz * dz_dt

        Tx = tools._cheb_eval_all(zx, p_x, deriv=0)
        dTx_dz = tools._cheb_eval_all(zx, p_x, deriv=1)
        d2Tx_dz2 = tools._cheb_eval_all(zx, p_x, deriv=2)
        d3Tx_dz3 = tools._cheb_eval_all(zx, p_x, deriv=3)

        dTx_dx = dTx_dz * dz_dx
        d2Tx_dx2 = d2Tx_dz2 * (dz_dx ** 2)
        d3Tx_dx3 = d3Tx_dz3 * (dz_dx ** 3)

        Tx_jump = tools._cheb_eval_all(zx_jump, p_x, deriv=0)
        dTx_jump_dz = tools._cheb_eval_all(zx_jump, p_x, deriv=1)
        dTx_jump_dx = dTx_jump_dz * dz_dx

        # ----------------------------
        # 4) Build design matrix Phi for tensor basis in (t, x)
        # ----------------------------
        N_obs = self.M * self.steps
        P = (p_t + 1) * (p_x + 1)
        Phi = np.empty((N_obs, P), dtype=float)

        col = 0
        for i in range(p_t + 1):
            tt_rep = np.tile(Tt[:, i], self.M)
            for j in range(p_x + 1):
                Phi[:, col] = tt_rep * Tx[:, j]
                col += 1

        y = np.repeat(Y, self.steps)

        # ----------------------------
        # 5) Fit ridge regression
        # ----------------------------
        beta = tools._ridge_solve(Phi, y, lam=lam_ridge)
        B = beta.reshape(p_t + 1, p_x + 1)

        # ----------------------------
        # 6) Evaluate H and derivatives in (t, x)
        # ----------------------------
        H_flat = np.zeros(N_obs, dtype=float)
        Ht_flat = np.zeros(N_obs, dtype=float)
        Hx_flat = np.zeros(N_obs, dtype=float)
        Hxx_flat = np.zeros(N_obs, dtype=float)
        Hxxx_flat = np.zeros(N_obs, dtype=float)
        Htx_flat = np.zeros(N_obs, dtype=float)

        H_jump_flat = np.zeros(N_obs, dtype=float)
        Hx_jump_flat = np.zeros(N_obs, dtype=float)

        for i in range(p_t + 1):
            tt0 = np.tile(Tt[:, i], self.M)
            tt1 = np.tile(dTt_dt[:, i], self.M)

            for j in range(p_x + 1):
                Bij = B[i, j]
                if Bij == 0.0:
                    continue

                x0 = Tx[:, j]
                x1 = dTx_dx[:, j]
                x2 = d2Tx_dx2[:, j]
                x3 = d3Tx_dx3[:, j]

                H_flat += Bij * (tt0 * x0)
                Ht_flat += Bij * (tt1 * x0)

                Hx_flat += Bij * (tt0 * x1)
                Hxx_flat += Bij * (tt0 * x2)
                Hxxx_flat += Bij * (tt0 * x3)

                Htx_flat += Bij * (tt1 * x1)

                x0j = Tx_jump[:, j]
                x1j = dTx_jump_dx[:, j]

                H_jump_flat += Bij * (tt0 * x0j)
                Hx_jump_flat += Bij * (tt0 * x1j)

        H = H_flat.reshape(self.M, self.steps)
        H = np.maximum(H, 0.0)

        H_t = Ht_flat.reshape(self.M, self.steps)

        H_x = Hx_flat.reshape(self.M, self.steps)
        H_xx = Hxx_flat.reshape(self.M, self.steps)
        H_xxx = Hxxx_flat.reshape(self.M, self.steps)

        H_tx = Htx_flat.reshape(self.M, self.steps)

        H_jump = H_jump_flat.reshape(self.M, self.steps)
        H_jump = np.maximum(H_jump, 0.0)

        H_x_jump = Hx_jump_flat.reshape(self.M, self.steps)

        # ----------------------------
        # 7) Convert x=log(S) derivatives to S derivatives
        # ----------------------------
        S = np.maximum(self.S_Jump, 1e-300)

        dH_dS = (1.0 / S) * H_x
        d2H_dSS = (1.0 / (S ** 2)) * (H_xx - H_x)
        d3H_dS3 = (1.0 / (S ** 3)) * (H_xxx - 3.0 * H_xx + 2.0 * H_x)

        dH_dt = H_t
        d2H_dtdS = (1.0 / S) * H_tx

        S_jump = np.maximum(self.S_Jump * self.J_Jump, 1e-300)
        dH_dS_jump = (1.0 / S_jump) * H_x_jump

        # ----------------------------
        # 8) Store
        # ----------------------------
        self.H_Jump = H

        self.dH_dS_Jump = dH_dS
        self.d2H_dSS_Jump = d2H_dSS
        self.dH_dt_Jump = dH_dt
        self.d2H_dtdS_Jump = d2H_dtdS
        self.d3H_dS3_Jump = d3H_dS3

        self.H_jump_Jump = H_jump
        self.dH_dS_jump_Jump = dH_dS_jump

        self._H_fit_Jump = {
            "p_t": p_t,
            "p_x": p_x,
            "lam_ridge": lam_ridge,
            "x_min": x_min,
            "x_max": x_max,
            "beta": beta,
        }

        return H, dH_dS, d2H_dSS, dH_dt, d2H_dtdS, d3H_dS3, H_jump, dH_dS_jump
   


    # ============================================================
    # 1. Control initialization
    # ============================================================
    def init_control(self, kind: str = "Delta") -> np.ndarray:
        """
        Initialize the control h.

        Parameters
        ----------
        kind : {"Delta", "MinVar", "zero"}
            Initialization rule.

            Delta  : h = dH_dS
            MinVar : scalar h that minimizes the instantaneous variance combining
                    diffusion and jump components.
            zero   : h = 0

        Returns
        -------
        h : ndarray, shape (M, steps-1)
            Initial control on each time interval [t_n, t_{n+1}).
        """
        if kind == "Delta":
            if getattr(self, "dH_dS_Jump", None) is None:
                raise ValueError("Missing dH_dS_Jump. Run simulate_H() first.")
            h0 = self.dH_dS_Jump[:, :-1].copy()

        elif kind == "MinVar":
            if getattr(self, "dH_dS_Jump", None) is None or getattr(self, "H_Jump", None) is None or getattr(self, "H_jump_Jump", None) is None:
                raise ValueError("Missing derivatives. Run simulate_H() first.")

            S = np.maximum(self.S_Jump[:, :-1], 1e-12)

            Delta = self.dH_dS_Jump[:, :-1]
            H0 = self.H_Jump[:, :-1]
            HJ = self.H_jump_Jump[:, :-1]

            dH_jump = HJ - H0

            Jm1 = np.maximum(self.J_Jump[:, :-1] - 1.0, -1e12)

            num = (self.sigma ** 2) * (S ** 2) * Delta + self.lam * (dH_jump * S * Jm1)
            den = (self.sigma ** 2) * (S ** 2) + self.lam * (S ** 2) * (Jm1 ** 2)

            h0 = num / np.maximum(den, 1e-16)
            h0 = h0.copy()

        elif kind == "zero":
            h0 = np.zeros((self.M, self.steps - 1), dtype=float)

        else:
            raise ValueError("Unknown control initialization kind")

        return h0


    # ============================================================
    # 2. Forward P&L
    # ============================================================
    def forward_PL(self,
                h: np.ndarray,
                L0: float = 0.0,
                t_start: int = 0) -> np.ndarray:
        """
        Vectorized Profit and Loss L_t using portfolio value.

        If start_idx > 0, the profit and loss is forced to be zero for all times
        before start_idx, so that L[:, -1] represents the residual profit and loss
        from t_start to T.

        Parameters
        ----------
        h : ndarray, shape (M, steps-1)
            Hedge ratio on each interval [t_n, t_{n+1}).
        L0 : float, default=0.0
            Initial profit and loss level.
        t_start : float, default=0
            Time index at which profit and loss accumulation starts.

        Returns
        -------
        L : ndarray, shape (M, steps)
            Profit and loss paths over time.
        """
        assert h.shape == (self.M, self.steps - 1)
        assert self.t0 <= t_start <= self.T
        start_idx = tools._time_to_index(N=self.N, t0=self.t0, T=self.T, t_start=t_start)

        # increments occur at times 1,...,steps-1
        dS = np.diff(self.S_Jump, axis=1)          # (M, steps-1)
        dH = np.diff(self.H_Jump, axis=1)          # (M, steps-1)
        dL = h * dS - dH                           # (M, steps-1)

        # cut past
        if start_idx > 0:
            dL = dL.copy()
            dL[:, :start_idx] = 0.0

        # cumulative profit and loss
        L = np.zeros((self.M, self.steps), dtype=float)
        L[:, 0] = L0
        L[:, 1:] = L0 + np.cumsum(dL, axis=1)

        return L



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
