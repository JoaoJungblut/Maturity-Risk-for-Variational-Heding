from OptimalHedging.Simulator import BaseSimulator
import OptimalHedging.tools as tools
import numpy as np
from typing import Dict, List, Tuple



class HestonSimulator(BaseSimulator):
    """
    Simulator for Heston process with optimal hedging machinery.

    This class implements all model-dependent components required to solve the
    optimal hedging problem under a Stochastic Volatility dynamics. All
    methods below are specific to the GBM model and are therefore implemented
    directly in this class. Only the terminal risk functional and the control
    update rule are inherited from BaseSimulator.

    Implemented methods
    -------------------
    - __init__ :
        Initializes the simulator and forwards all parameters to BaseSimulator.

    - simulate_S :
        Simulates Heston paths for the underlying asset and stores both S_Heston, v_Heston and
        the corresponding Brownian increments dW1_Heston and dW2_Heston.

    - simulate_H :
        Computes the contingent claim price H(t, S) and its time and space
        derivatives using a single-pass LSMC regression on a Chebyshev basis.

    - init_control :
        Initializes the hedging control, typically using the Delta of the
        contingent claim as a smooth initial guess.

    - forward_PL :
        Simulates the forward Profit and Loss process associated with a given
        hedging strategy under the Heston dynamics.

    - backward_adjoint :
        Solves the backward adjoint equation associated with the Heston optimal
        hedging problem using regression-based conditional expectations.

    - compute_gradient :
        Computes the Heston-specific optimality condition (gradient) used to update
        the control process.

    - optimize_hedge :
        Runs the full iterative optimization loop to compute an approximate
        optimal hedge under the chosen risk functional.

    - compute_MR :
        Computes the maturity risk by comparing optimal risks on sub-horizons
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
            Mean reversion speed of the variance process.
        theta : float
            Long run mean of the variance process.
        sigma_v : float
            Volatility of variance vol of vol.
        corr : float
            Correlation between the asset Brownian motion and the variance Brownian motion.
            Must satisfy -1 <= corr <= 1.
        base_kwargs : dict
            Same arguments as BaseSimulator:
            S0, mu, sigma, K, t0, T, N, M, seed.
        """
        super().__init__(**base_kwargs)

        if not (-1.0 <= corr <= 1.0):
            raise ValueError("corr must be in [-1, 1].")

        self.kappa = float(kappa)
        self.theta = float(theta)
        self.sigma_v = float(sigma_v)
        self.corr = float(corr)



    # ============================================================
    # 0. Underlying S,v and derivative H
    # ============================================================
    def simulate_S(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate Heston model paths using a vectorized (over paths) Euler scheme.

            dS = mu * S * dt + sqrt(v) * S * dW1
            dv = kappa * (theta - v) * dt + sigma_v * sqrt(v) * dW2

        with corr(dW1, dW2) = corr.

        Returns
        -------
        S : ndarray, shape (M, steps)
            Simulated underlying asset price paths.
        v : ndarray, shape (M, steps)
            Simulated stochastic variance paths.
        """
        Sigma = np.array([
            [1.0,       self.corr],
            [self.corr, 1.0]
        ], dtype=float)

        dW = np.random.multivariate_normal(
            mean=np.zeros(2),
            cov=Sigma * self.dt,
            size=(self.M, self.steps - 1)
        )

        dW1 = dW[:, :, 0]
        dW2 = dW[:, :, 1]

        S = np.zeros((self.M, self.steps), dtype=float)
        v = np.zeros((self.M, self.steps), dtype=float)

        S[:, 0] = self.S0
        v[:, 0] = self.sigma  # treated as v0 by design

        eps = 1e-12

        for n in range(self.steps - 1):
            Sn = S[:, n]
            vn = v[:, n]

            v_pos = np.maximum(vn, 0.0)

            # Euler step for S using full truncation in the diffusion term
            S[:, n + 1] = Sn * (1.0 + self.mu * self.dt + np.sqrt(v_pos + eps) * dW1[:, n])

            # Full truncation Euler for v
            v_next = vn + self.kappa * (self.theta - v_pos) * self.dt + self.sigma_v * np.sqrt(v_pos + eps) * dW2[:, n]
            v[:, n + 1] = np.maximum(v_next, 0.0)

        self.S_Heston = S
        self.v_Heston = v
        self.dW1_Heston = dW1
        self.dW2_Heston = dW2

        return S, v

    def simulate_H(self,
                   p_t: int = 1,
                   p_x: int = 2,
                   p_v: int = 2,
                   lam_ridge: float = 1e-2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                     np.ndarray, np.ndarray, np.ndarray,
                                                     np.ndarray, np.ndarray, np.ndarray,
                                                     np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate H(t,S,v) = E[(S_T - K)^+ | S_t = S, v_t = v] and derivatives using a single-pass
        LSMC regression on the already-simulated Heston paths.

        Approach
        --------
        - Regress payoff Y = (S_T - K)^+ on a tensor Chebyshev basis in (t, x, v),
        where x = log(S), using ridge regularization.
        - Compute derivatives analytically from the fitted basis in (t, x, v).
        - Convert x-derivatives to S-derivatives via chain rule.

        Parameters
        ----------
        p_t : int, default=1
            Chebyshev degree in time.
        p_x : int, default=2
            Chebyshev degree in x = log(S).
        p_v : int, default=2
            Chebyshev degree in v.
        lam_ridge : float, default=1e-2
            Ridge regularization.

        Returns
        -------
        H : ndarray, shape (M, steps)
        dH_dS : ndarray, shape (M, steps)
        d2H_dSS : ndarray, shape (M, steps)
        dH_dt : ndarray, shape (M, steps)
        d2H_dtdS : ndarray, shape (M, steps)
        d3H_dS3 : ndarray, shape (M, steps)
        dH_dv : ndarray, shape (M, steps)
        d2H_dvv : ndarray, shape (M, steps)
        d2H_dSdv : ndarray, shape (M, steps)
        d3H_dS2dv : ndarray, shape (M, steps)
        d3H_dSdvv : ndarray, shape (M, steps)
        d3H_dv3 : ndarray, shape (M, steps)
        d2H_dt_dv : ndarray, shape (M, steps)
        """
        if getattr(self, "S_Heston", None) is None or getattr(self, "v_Heston", None) is None:
            raise ValueError("self.S_Heston or self.v_Heston is None. Run simulate_S() first.")

        # ----------------------------
        # 1) Targets and state
        # ----------------------------
        Y = np.maximum(self.S_Heston[:, -1] - self.K, 0.0)  # (M,)

        X_path = np.log(np.maximum(self.S_Heston, 1e-300))  # (M, steps)
        V_path = np.maximum(self.v_Heston, 1e-300)          # (M, steps)

        t_grid = self.t0 + self.dt * np.arange(self.steps)  # (steps,)

        # ----------------------------
        # 2) Scale (t, x, v) to [-1, 1]
        # ----------------------------
        x_min = float(X_path.min())
        x_max = float(X_path.max())
        v_min = float(V_path.min())
        v_max = float(V_path.max())

        if (not np.isfinite(x_min)) or (not np.isfinite(x_max)) or (x_max <= x_min):
            raise ValueError("Invalid X_path range for scaling.")
        if (not np.isfinite(v_min)) or (not np.isfinite(v_max)) or (v_max <= v_min):
            raise ValueError("Invalid V_path range for scaling.")

        zt = tools._minmax_scale(t_grid, self.t0, self.T)            # (steps,)
        zx = tools._minmax_scale(X_path.reshape(-1), x_min, x_max)   # (M*steps,)
        zv = tools._minmax_scale(V_path.reshape(-1), v_min, v_max)   # (M*steps,)

        dz_dt = 2.0 / (self.T - self.t0)
        dz_dx = 2.0 / (x_max - x_min)
        dz_dv = 2.0 / (v_max - v_min)

        # ----------------------------
        # 3) Chebyshev basis and derivatives
        # ----------------------------
        # time basis
        Tt = tools._cheb_eval_all(zt, p_t, deriv=0)
        dTt_dz = tools._cheb_eval_all(zt, p_t, deriv=1)
        dTt_dt = dTt_dz * dz_dt

        # x basis (flattened observations)
        Tx = tools._cheb_eval_all(zx, p_x, deriv=0)
        dTx_dz = tools._cheb_eval_all(zx, p_x, deriv=1)
        d2Tx_dz2 = tools._cheb_eval_all(zx, p_x, deriv=2)
        d3Tx_dz3 = tools._cheb_eval_all(zx, p_x, deriv=3)

        dTx_dx = dTx_dz * dz_dx
        d2Tx_dx2 = d2Tx_dz2 * (dz_dx ** 2)
        d3Tx_dx3 = d3Tx_dz3 * (dz_dx ** 3)

        # v basis (flattened observations)
        Tv = tools._cheb_eval_all(zv, p_v, deriv=0)
        dTv_dz = tools._cheb_eval_all(zv, p_v, deriv=1)
        d2Tv_dz2 = tools._cheb_eval_all(zv, p_v, deriv=2)
        d3Tv_dz3 = tools._cheb_eval_all(zv, p_v, deriv=3)

        dTv_dv = dTv_dz * dz_dv
        d2Tv_dv2 = d2Tv_dz2 * (dz_dv ** 2)
        d3Tv_dv3 = d3Tv_dz3 * (dz_dv ** 3)

        # ----------------------------
        # 4) Build design matrix Phi for tensor basis in (t, x, v)
        # ----------------------------
        N_obs = self.M * self.steps
        P = (p_t + 1) * (p_x + 1) * (p_v + 1)
        Phi = np.empty((N_obs, P), dtype=float)

        col = 0
        for i in range(p_t + 1):
            tt_rep = np.tile(Tt[:, i], self.M)  # (N_obs,)
            for j in range(p_x + 1):
                xj = Tx[:, j]
                for k in range(p_v + 1):
                    Phi[:, col] = tt_rep * xj * Tv[:, k]
                    col += 1

        y = np.repeat(Y, self.steps)  # (N_obs,)

        # ----------------------------
        # 5) Fit ridge regression
        # ----------------------------
        beta = tools._ridge_solve(Phi, y, lam=lam_ridge)  # (P,)
        B = beta.reshape(p_t + 1, p_x + 1, p_v + 1)

        # ----------------------------
        # 6) Evaluate H and derivatives in (t, x, v)
        # ----------------------------
        H_flat = np.zeros(N_obs, dtype=float)

        Ht_flat = np.zeros(N_obs, dtype=float)

        Hx_flat = np.zeros(N_obs, dtype=float)
        Hxx_flat = np.zeros(N_obs, dtype=float)
        Hxxx_flat = np.zeros(N_obs, dtype=float)

        Hv_flat = np.zeros(N_obs, dtype=float)
        Hvv_flat = np.zeros(N_obs, dtype=float)
        Hvvv_flat = np.zeros(N_obs, dtype=float)

        Htx_flat = np.zeros(N_obs, dtype=float)
        Htv_flat = np.zeros(N_obs, dtype=float)
        Hxv_flat = np.zeros(N_obs, dtype=float)

        Hxxv_flat = np.zeros(N_obs, dtype=float)
        Hxvv_flat = np.zeros(N_obs, dtype=float)

        for i in range(p_t + 1):
            tt0 = np.tile(Tt[:, i], self.M)      # (N_obs,)
            tt1 = np.tile(dTt_dt[:, i], self.M)  # (N_obs,)

            for j in range(p_x + 1):
                x0 = Tx[:, j]
                x1 = dTx_dx[:, j]
                x2 = d2Tx_dx2[:, j]
                x3 = d3Tx_dx3[:, j]

                for k in range(p_v + 1):
                    Bik = B[i, j, k]
                    if Bik == 0.0:
                        continue

                    v0 = Tv[:, k]
                    v1 = dTv_dv[:, k]
                    v2 = d2Tv_dv2[:, k]
                    v3 = d3Tv_dv3[:, k]

                    H_flat += Bik * (tt0 * x0 * v0)

                    Ht_flat += Bik * (tt1 * x0 * v0)

                    Hx_flat += Bik * (tt0 * x1 * v0)
                    Hxx_flat += Bik * (tt0 * x2 * v0)
                    Hxxx_flat += Bik * (tt0 * x3 * v0)

                    Hv_flat += Bik * (tt0 * x0 * v1)
                    Hvv_flat += Bik * (tt0 * x0 * v2)
                    Hvvv_flat += Bik * (tt0 * x0 * v3)

                    Htx_flat += Bik * (tt1 * x1 * v0)
                    Htv_flat += Bik * (tt1 * x0 * v1)
                    Hxv_flat += Bik * (tt0 * x1 * v1)

                    Hxxv_flat += Bik * (tt0 * x2 * v1)
                    Hxvv_flat += Bik * (tt0 * x1 * v2)

        H = H_flat.reshape(self.M, self.steps)
        H = np.maximum(H, 0.0)

        H_t = Ht_flat.reshape(self.M, self.steps)

        H_x = Hx_flat.reshape(self.M, self.steps)
        H_xx = Hxx_flat.reshape(self.M, self.steps)
        H_xxx = Hxxx_flat.reshape(self.M, self.steps)

        H_v = Hv_flat.reshape(self.M, self.steps)
        H_vv = Hvv_flat.reshape(self.M, self.steps)
        H_vvv = Hvvv_flat.reshape(self.M, self.steps)

        H_tx = Htx_flat.reshape(self.M, self.steps)
        H_tv = Htv_flat.reshape(self.M, self.steps)
        H_xv = Hxv_flat.reshape(self.M, self.steps)

        H_xxv = Hxxv_flat.reshape(self.M, self.steps)
        H_xvv = Hxvv_flat.reshape(self.M, self.steps)

        # ----------------------------
        # 7) Convert x=log(S) derivatives to S-derivatives
        # ----------------------------
        S = np.maximum(self.S_Heston, 1e-300)

        dH_dS = (1.0 / S) * H_x
        d2H_dSS = (1.0 / (S ** 2)) * (H_xx - H_x)
        d3H_dS3 = (1.0 / (S ** 3)) * (H_xxx - 3.0 * H_xx + 2.0 * H_x)

        dH_dt = H_t
        d2H_dtdS = (1.0 / S) * H_tx

        dH_dv = H_v
        d2H_dvv = H_vv
        d2H_dSdv = (1.0 / S) * H_xv

        d3H_dS2dv = (1.0 / (S ** 2)) * (H_xxv - H_xv)
        d3H_dSdvv = (1.0 / S) * H_xvv
        d3H_dv3 = H_vvv

        d2H_dt_dv = H_tv

        # ----------------------------
        # 8) Store
        # ----------------------------
        self.H_Heston = H

        self.dH_dS_Heston = dH_dS
        self.d2H_dSS_Heston = d2H_dSS
        self.d3H_dS3_Heston = d3H_dS3

        self.dH_dt_Heston = dH_dt
        self.d2H_dtdS_Heston = d2H_dtdS

        self.dH_dv_Heston = dH_dv
        self.d2H_dvv_Heston = d2H_dvv
        self.d2H_dSdv_Heston = d2H_dSdv

        self.d3H_dS2dv_Heston = d3H_dS2dv
        self.d3H_dSdvv_Heston = d3H_dSdvv
        self.d3H_dv3_Heston = d3H_dv3

        self.d2H_dtdv_Heston = d2H_dt_dv

        self._H_fit_Heston = {
            "p_t": p_t,
            "p_x": p_x,
            "p_v": p_v,
            "lam_ridge": lam_ridge,
            "x_min": x_min,
            "x_max": x_max,
            "v_min": v_min,
            "v_max": v_max,
            "beta": beta,
        }

        return (H,
                dH_dS, d2H_dSS, dH_dt, d2H_dtdS, d3H_dS3,
                dH_dv, d2H_dvv, d2H_dSdv, d3H_dS2dv, d3H_dSdvv, d3H_dv3,
                d2H_dt_dv)

    

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
            MinVar : scalar h that minimizes the instantaneous diffusion variance
                    implied by the two Brownian drivers under correlation.
            zero   : h = 0

        Returns
        -------
        h : ndarray, shape (M, steps-1)
            Initial control on each time interval [t_n, t_{n+1}).
        """
        if kind == "Delta":
            if getattr(self, "dH_dS_Heston", None) is None:
                raise ValueError("Missing dH_dS_Heston. Run simulate_H() first.")
            h0 = self.dH_dS_Heston[:, :-1].copy()

        elif kind == "MinVar":
            if getattr(self, "dH_dS_Heston", None) is None or getattr(self, "dH_dv_Heston", None) is None:
                raise ValueError("Missing derivatives. Run simulate_H() first.")

            S = np.maximum(self.S_Heston[:, :-1], 1e-12)

            Delta = self.dH_dS_Heston[:, :-1]
            Vega  = self.dH_dv_Heston[:, :-1]

            # Minimum variance projection of dv-risk onto dS
            h0 = Delta + self.corr * self.sigma_v * Vega / S

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
                   t_start: float = 0) -> np.ndarray:
        """
        Vectorized Profit and Loss L_t using portfolio value.

        If start_idx > 0, the P&L is forced to be zero for all times < start_idx,
        so that L[:, -1] represents the residual P&L L_{t,T}.

        Parameters
        ----------
        h : ndarray, shape (M, steps-1)
            Hedge ratio on each interval [t_n, t_{n+1}).
        L0 : float, default=0.0
            Initial Profit and Loss level.
        t_start : float, default=0
            Time index t at which Profit and Loss accumulation starts.

        Returns
        -------
        L : ndarray, shape (M, steps)
            Profit and Loss paths over time.
        """
        assert h.shape == (self.M, self.steps - 1)
        assert self.t0 <= t_start <= self.T

        start_idx = tools._time_to_index(
            N=self.N, t0=self.t0, T=self.T, t_start=t_start
        )

        # increments occur at times 1,...,steps-1
        dS = np.diff(self.S_Heston, axis=1)   # (M, steps-1)
        dH = np.diff(self.H_Heston, axis=1)   # (M, steps-1)
        dL = h * dS - dH                      # (M, steps-1)

        # cut past
        if start_idx > 0:
            dL = dL.copy()
            dL[:, :start_idx] = 0.0

        # cumulative P&L
        L = np.zeros((self.M, self.steps), dtype=float)
        L[:, 0] = L0
        L[:, 1:] = L0 + np.cumsum(dL, axis=1)

        return L



    # ============================================================
    # 5. Backward adjoint (driver tipo GBM, mas com v_t em σ_t)
    # ============================================================
    def backward_adjoint(self,
                         pT: np.ndarray,
                         p_x: int = 2,
                         p_v: int = 1,
                         lam_ridge: float = 1e-2,
                         lam_time: float = 1e-2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve the backward adjoint BSDE for the Heston model by a discrete-time backward scheme with regression.

        - Two Brownian drivers imply two martingale components: q1 and q2.
        - Conditional expectations are approximated via a tensor Chebyshev basis in (x, v),
        where x = log(S).
        - Ridge regularization and temporal smoothing are applied to regression coefficients.

        Parameters
        ----------
        pT : ndarray, shape (M,)
            Terminal adjoint values.
        p_x : int, default=2
            Chebyshev degree in x = log(S).
        p_v : int, default=1
            Chebyshev degree in v.
        lam_ridge : float, default=1e-2
            Ridge regularization.
        lam_time : float, default=1e-2
            Temporal smoothing.

        Returns
        -------
        p : ndarray, shape (M, steps)
            Adjoint process.
        q1 : ndarray, shape (M, steps-1)
            Martingale component for dW1.
        q2 : ndarray, shape (M, steps-1)
            Martingale component for dW2.
        """
        if getattr(self, "S_Heston", None) is None or getattr(self, "v_Heston", None) is None:
            raise ValueError("Missing Heston paths. Run simulate_S() first.")
        if getattr(self, "dW1_Heston", None) is None or getattr(self, "dW2_Heston", None) is None:
            raise ValueError("Missing Brownian increments. Run simulate_S() first.")
        if getattr(self, "H_Heston", None) is None:
            raise ValueError("Missing H. Run simulate_H() first.")

        # required H derivatives for the driver
        req = [
            "dH_dS_Heston", "d2H_dSS_Heston", "d2H_dSdv_Heston",
            "d2H_dtdS_Heston", "d3H_dS3_Heston", "d3H_dS2dv_Heston", "d3H_dSdvv_Heston"
        ]
        for name in req:
            if getattr(self, name, None) is None:
                raise ValueError(f"Missing {name}. Run simulate_H() first.")

        p = np.zeros((self.M, self.steps), dtype=float)
        q1 = np.zeros((self.M, self.steps - 1), dtype=float)
        q2 = np.zeros((self.M, self.steps - 1), dtype=float)
        p[:, -1] = np.asarray(pT, dtype=float)

        # regressors: x = log(S), v = v
        S_all = np.maximum(self.S_Heston, 1e-300)
        X_all = np.log(S_all)
        V_all = np.maximum(self.v_Heston, 0.0)

        x_min = float(X_all.min())
        x_max = float(X_all.max())
        v_min = float(V_all.min())
        v_max = float(V_all.max())

        if (not np.isfinite(x_min)) or (not np.isfinite(x_max)) or (x_max <= x_min):
            raise ValueError("Invalid scaling range for log(S).")
        if (not np.isfinite(v_min)) or (not np.isfinite(v_max)) or (v_max <= v_min):
            raise ValueError("Invalid scaling range for v.")

        Px = p_x + 1
        Pv = p_v + 1
        P = Px * Pv

        beta_p_next = np.zeros(P)
        beta_q1_next = np.zeros(P)
        beta_q2_next = np.zeros(P)

        eps = 1e-12

        for n in range(self.steps - 2, -1, -1):
            Sn = self.S_Heston[:, n]
            vn = np.maximum(self.v_Heston[:, n], 0.0)
            sqrt_vn = np.sqrt(vn + eps)

            dW1n = self.dW1_Heston[:, n]
            dW2n = self.dW2_Heston[:, n]

            # build tensor Chebyshev features Phi(x_n, v_n)
            x_n = X_all[:, n]
            v_n = vn

            zx = tools._minmax_scale(x_n, x_min, x_max)
            zv = tools._minmax_scale(v_n, v_min, v_max)

            Tx = tools._cheb_eval_all(zx, p_x, deriv=0)  # (M, Px)
            Tv = tools._cheb_eval_all(zv, p_v, deriv=0)  # (M, Pv)

            Phi = np.empty((self.M, P), dtype=float)
            col = 0
            for i in range(Px):
                for j in range(Pv):
                    Phi[:, col] = Tx[:, i] * Tv[:, j]
                    col += 1

            p_next = p[:, n + 1]

            # q1_n and q2_n via martingale representation
            y_q1 = (p_next * dW1n) / self.dt
            beta_q1 = tools._solve_smoothed(Phi, y_q1, beta_q1_next, lam_ridge, lam_time)
            q1[:, n] = Phi @ beta_q1

            y_q2 = (p_next * dW2n) / self.dt
            beta_q2 = tools._solve_smoothed(Phi, y_q2, beta_q2_next, lam_ridge, lam_time)
            q2[:, n] = Phi @ beta_q2

            # driver terms (at time n)
            H_S   = self.dH_dS_Heston[:, n]
            H_SS  = self.d2H_dSS_Heston[:, n]
            H_Sv  = self.d2H_dSdv_Heston[:, n]

            H_tS    = self.d2H_dtdS_Heston[:, n]
            H_SSS   = self.d3H_dS3_Heston[:, n]
            H_SSv   = self.d3H_dS2dv_Heston[:, n]
            H_Svv   = self.d3H_dSdvv_Heston[:, n]

            # main drift coefficient multiplying p
            C_n = (
                H_tS
                + self.mu * Sn * H_SS
                + self.kappa * (self.theta - vn) * H_Sv
                + 0.5 * vn * (Sn ** 2) * H_SSS
                + 0.5 * (self.sigma_v ** 2) * vn * H_Svv
                + self.corr * self.sigma_v * vn * Sn * H_SSv
            )

            # extra drift terms involving q1 and q2
            term_q1 = -sqrt_vn * (H_S + Sn * H_SS) * q1[:, n]
            term_q2 = (self.sigma_v * sqrt_vn * H_Sv) * q2[:, n]

            drift_p = C_n * p_next + term_q1 + term_q2

            y_p = p_next - drift_p * self.dt
            beta_p = tools._solve_smoothed(Phi, y_p, beta_p_next, lam_ridge, lam_time)
            p[:, n] = Phi @ beta_p

            beta_q1_next = beta_q1
            beta_q2_next = beta_q2
            beta_p_next = beta_p

        self._backward_fit_Heston = {
            "p_x": p_x,
            "p_v": p_v,
            "lam_ridge": lam_ridge,
            "lam_time": lam_time,
            "x_min": x_min,
            "x_max": x_max,
            "v_min": v_min,
            "v_max": v_max,
        }

        self.p_Heston = p
        self.q1_Heston = q1
        self.q2_Heston = q2

        return p, q1, q2



    # ============================================================
    # 6. Gradient (optimality condition)
    # ============================================================
    def compute_gradient(self,
                         p: np.ndarray,
                         q1: np.ndarray,
                         q2: np.ndarray,
                         eps: float = 1e-12) -> np.ndarray:
        """
        Compute the violation of the local optimality condition for Heston.

        Parameters
        ----------
        p : ndarray, shape (M, steps)
            Adjoint process p.
        q1 : ndarray, shape (M, steps-1)
            Martingale component for dW1.
        q2 : ndarray, shape (M, steps-1)
            Martingale component for dW2.
        eps : float, default=1e-12
            Small constant for numerical stability in sqrt(v).

        Returns
        -------
        G : ndarray, shape (M, steps-1)
            Gradient like quantity used to update the control.
        """
        assert p.shape == (self.M, self.steps)
        assert q1.shape == (self.M, self.steps - 1)
        assert q2.shape == (self.M, self.steps - 1)

        S_trunc = self.S_Heston[:, :-1]
        v_trunc = np.maximum(self.v_Heston[:, :-1], 0.0)
        sqrt_v = np.sqrt(v_trunc + eps)

        G = (self.mu * S_trunc + self.kappa * (self.theta - v_trunc)) * p[:, :-1] \
            + sqrt_v * S_trunc * q1 \
            + self.sigma_v * sqrt_v * q2

        return G



    # ============================================================
    # 8. Main optimization loop
    # ============================================================
    def optimize_hedge(self,
                       risk_type: str,
                       risk_kwargs: Dict,
                       t_idx: float = 0,
                       kind: str = "Delta",
                       max_iter: int = 20,
                       tol: float = 1e-4,
                       alpha: float = 1e-3,
                       verbose: bool = True) -> Tuple[np.ndarray, List[Dict]]:
        """
        Main optimization loop to compute an approximate optimal hedge h.
        - Keep the last two accepted controls: h_curr (best/current), h_prev (previous).
        - At each iteration, compute grad_norm at h_curr.
        - If the first trial step from h_curr with current alpha does NOT improve grad_norm,
            rollback immediately to h_prev, shrink alpha, and search from there.
        - During the search, the anchor stays fixed; alpha is halved until improvement is found.
        - If improvement cannot be found (alpha too small), stop.

        Parameters
        ----------
        risk_type : str
            One of {"ele", "elw", "entl", "ente", "entw", "es"}.
        risk_kwargs : dict
            Parameters required for the chosen risk_type.
        t_idx : float, default=0
            Time index t at which P&L accumulation starts.
        kind : {"Delta", "MinVar", "zero"}
            Initialization rule.

            Delta  : h = dH_dS
            MinVar : scalar h that minimizes the instantaneous diffusion variance
                    implied by the two Brownian drivers under correlation.
            zero   : h = 0
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
        h_curr = self.init_control(kind=kind)
        h_prev = None

        history: List[Dict] = []

        shrink = 0.5
        bt_max = 20
        alpha_min = 1e-12

        for k in range(max_iter):
            # ----- evaluate gradient norm at current accepted control -----
            L = self.forward_PL(h_curr, L0=0.0, t_start=t_idx)
            LT = L[:, -1]
            pT = self.terminal_adjoint(LT, risk_type, **risk_kwargs)

            p, q1, q2 = self.backward_adjoint(pT)
            G_curr = self.compute_gradient(p, q1, q2)
            g_curr = np.linalg.norm(G_curr) / np.sqrt(G_curr.size)

            history.append({
                "iter": k,
                "phase": "base",
                "alpha": alpha,
                "grad_norm": g_curr,
                "accepted": True
            })

            if verbose:
                print(f"iter {k:3d} | grad_norm={g_curr:.6e} | alpha={alpha:.2e}")

            if g_curr < tol:
                break

            # ----- FIRST TRIAL from current point -----
            alpha_try = alpha
            h_try = self.update_control(h_curr, G_curr, alpha_try)

            L_try = self.forward_PL(h_try, L0=0.0, t_start=t_idx)
            LT_try = L_try[:, -1]
            pT_try = self.terminal_adjoint(LT_try, risk_type, **risk_kwargs)

            p_try, q1_try, q2_try = self.backward_adjoint(pT_try)
            G_try = self.compute_gradient(p_try, q1_try, q2_try)
            g_try = np.linalg.norm(G_try) / np.sqrt(G_try.size)

            first_accept = (g_try < g_curr)

            history.append({
                "iter": k,
                "phase": "trial",
                "alpha": alpha_try,
                "grad_norm": g_try,
                "accepted": first_accept
            })

            if verbose:
                tag = "ACCEPT" if first_accept else "reject"
                print(f"    trial    | alpha={alpha_try:.2e} | grad_norm={g_try:.6e} | {tag}")

            if first_accept:
                h_prev = h_curr
                h_curr = h_try
                continue

            if k < 2:
                if verbose:
                    print("rollback disabled (k < 2)")
                continue

            # ----- ROLLBACK / BACKTRACKING ANCHOR -----
            anchor_h = h_prev if (h_prev is not None) else h_curr
            alpha_try = alpha * shrink

            # recompute gradient at anchor (required)
            L_a = self.forward_PL(anchor_h, L0=0.0, t_start=t_idx)
            LT_a = L_a[:, -1]
            pT_a = self.terminal_adjoint(LT_a, risk_type, **risk_kwargs)

            p_a, q1_a, q2_a = self.backward_adjoint(pT_a)
            G_a = self.compute_gradient(p_a, q1_a, q2_a)
            g_a = np.linalg.norm(G_a) / np.sqrt(G_a.size)

            history.append({
                "iter": k,
                "phase": "rollback_anchor" if (h_prev is not None) else "bt_anchor_curr",
                "alpha": alpha_try,
                "grad_norm": g_a,
                "accepted": True
            })

            if verbose:
                anc = "prev" if (h_prev is not None) else "curr"
                print(f"    rollback | anchor={anc} | grad_norm={g_a:.6e} | alpha={alpha_try:.2e}")

            accepted = False
            for j in range(bt_max):
                if alpha_try < alpha_min:
                    break

                h_new = self.update_control(anchor_h, G_a, alpha_try)

                L_new = self.forward_PL(h_new, L0=0.0, t_start=t_idx)
                LT_new = L_new[:, -1]
                pT_new = self.terminal_adjoint(LT_new, risk_type, **risk_kwargs)

                p_new, q1_new, q2_new = self.backward_adjoint(pT_new)
                G_new = self.compute_gradient(p_new, q1_new, q2_new)
                g_new = np.linalg.norm(G_new) / np.sqrt(G_new.size)

                is_accept = (g_new < g_a)

                history.append({
                    "iter": k,
                    "phase": "search_from_prev" if (h_prev is not None) else "search_from_curr",
                    "search_step": j,
                    "alpha": alpha_try,
                    "grad_norm": g_new,
                    "accepted": is_accept
                })

                if verbose:
                    tag = "ACCEPT" if is_accept else "reject"
                    print(f"    search {j:2d} | alpha={alpha_try:.2e} | grad_norm={g_new:.6e} | {tag}")

                if is_accept:
                    h_prev = anchor_h
                    h_curr = h_new
                    alpha = alpha_try
                    accepted = True
                    break

                alpha_try *= shrink

            if not accepted:
                if verbose:
                    print("no improving step found after rollback search (stopping).")
                break

        return h_curr, history
    


    # ============================================================
    # 9. Maturity Risk computation
    # ============================================================
    def compute_MR(self,
                   risk_type: str,
                   risk_kwargs: Dict,
                   t_idx: float = 0,
                   kind: str = "Delta",
                   max_iter: int = 20,
                   tol: float = 1e-4,
                   alpha: float = 1e-3,
                   verbose: bool = True) -> Tuple[float, Dict]:
        """
        Compute the maturity risk MR(t) via two optimal hedging problems.

        The maturity risk is defined as the difference between:
            - the optimal risk on the sub-horizon [t, T], and
            - the optimal risk on the full horizon [0, T].

            MR(t) = rho_t - rho_T

        Parameters
        ----------
        t_idx : float
            Time index t at which the maturity risk is evaluated.
            Must satisfy 0 <= t_idx <= self.steps.
        risk_type : str
            Risk functional identifier (e.g. "ele", "elw", "entl", "ente", "entw", "esl").
        risk_kwargs : dict
            Parameters required by the chosen risk functional.
        kind : {"Delta", "MinVar", "zero"}
            Initialization rule.

            Delta  : h = dH_dS
            MinVar : scalar h that minimizes the instantaneous diffusion variance
                    implied by the two Brownian drivers under correlation.
            zero   : h = 0
        max_iter : int, default=20
            Maximum number of iterations in the hedge optimization.
        tol : float, default=1e-4
            Tolerance on the gradient norm (stopping criterion).
        alpha : float, default=1e-3
            Step size used in the control update.
        verbose : bool, default=True
            If True, prints optimization diagnostics.

        Returns
        -------
        MR : float
            Maturity risk at time index t_idx, defined as rho_t - rho_T.
        info : dict
            Dictionary containing:
                - "h_T"   : optimal hedge on [0, T]
                - "h_t"   : optimal hedge on [t, T]
                - "rho_T" : optimal risk on [0, T]
                - "rho_t" : optimal risk on [t, T]
        """
        assert 0 <= t_idx <= self.T

        # --- at maturity [, T] ---
        h_T, _ = self.optimize_hedge(
            risk_type=risk_type,
            risk_kwargs=risk_kwargs,
            t_idx=self.T,
            kind=kind,
            max_iter=max_iter,
            tol=tol,
            alpha=alpha,
            verbose=verbose,
        )
        L_T = self.forward_PL(h_T, L0=0.0, t_start=self.T)
        rho_T = self.risk_function(L_T[:, -1], risk_type, **risk_kwargs)

        # --- sub-horizon [t, T] ---
        h_t, _ = self.optimize_hedge(
            risk_type=risk_type,
            risk_kwargs=risk_kwargs,
            t_idx=t_idx,
            kind=kind,
            max_iter=max_iter,
            tol=tol,
            alpha=alpha,
            verbose=verbose
        )
        L_t = self.forward_PL(h_t, L0=0.0, t_start=t_idx)
        rho_t = self.risk_function(L_t[:, -1], risk_type, **risk_kwargs)

        MR = rho_t - rho_T

        return MR, {"h_T": h_T, "h_t": h_t, "rho_T": rho_T, "rho_t": rho_t}