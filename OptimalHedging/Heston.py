from OptimalHedging.Simulator import BaseSimulator
import OptimalHedging.tools as tools
import numpy as np
from typing import Dict, List, Tuple


class HestonSimulator(BaseSimulator):
    """
    Simulator for the Heston model with optimal hedging machinery.

    This class implements the model-specific components associated with the
    Heston stochastic volatility dynamics. It follows the same overall hedging
    workflow used in GBMSimulator, but replaces the underlying simulation and
    any model-dependent quantities accordingly.
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
        # 7b) Enforce terminal condition exactly (payoff does not depend on v)
        # ----------------------------
        payoff_T = np.maximum(self.S_Heston[:, -1] - self.K, 0.0)
        H[:, -1] = payoff_T

        dH_dS[:, -1] = (self.S_Heston[:, -1] > self.K).astype(float)
        d2H_dSS[:, -1] = 0.0
        d3H_dS3[:, -1] = 0.0

        dH_dv[:, -1] = 0.0
        d2H_dvv[:, -1] = 0.0
        d3H_dv3[:, -1] = 0.0

        d2H_dSdv[:, -1] = 0.0
        d3H_dS2dv[:, -1] = 0.0
        d3H_dSdvv[:, -1] = 0.0
        d2H_dt_dv[:, -1] = 0.0
        d2H_dtdS[:, -1] = 0.0
        dH_dt[:, -1] = 0.0

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
    def init_control(self, kind: str = "MinVar") -> np.ndarray:
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

            Delta = self.dH_dS_Heston[:, :-1]
            Vega  = self.dH_dv_Heston[:, :-1]

            a = float(self.sigma_v)
            r = float(self.corr)

            denom = 1.0 + a * a + 2.0 * r * a
            if abs(denom) < 1e-12:
                raise ValueError("Invalid parameters: denominator too small in MinVar initialization.")

            # h = argmin_h ( (h-Delta)^2 + (a*h - Vega)^2 + 2*r*(h-Delta)*(a*h - Vega) )
            h0 = ((1.0 + r * a) * Delta + (a + r) * Vega) / denom
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
