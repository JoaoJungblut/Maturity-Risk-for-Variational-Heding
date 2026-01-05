from OptimalHedging.Simulator import BaseSimulator
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
from typing import Dict, List, Tuple
import OptimalHedging.tools as tools


class GBMSimulator(BaseSimulator):
    """
    Simulator for Geometric Brownian Motion (GBM) + optimal hedging machinery.
    """

    def __init__(self, 
                 **base_kwargs):
        """
        Initialize the GBM simulator.

        This class does not introduce new parameters and simply forwards
        all arguments to the BaseSimulator initializer.
        """
        super().__init__(**base_kwargs)

    # ============================================================
    # 0. Simulation of underlying asset S and derivative H 
    # ============================================================

    def simulate_S(self) -> np.ndarray:
        """
        Simulate Geometric Brownian Motion (GBM) paths using a fully vectorized Euler–Maruyama scheme.

            dS = mu * S * dt + sigma * S * dW

        Returns
        -------
        S : ndarray, shape (M, steps)
            Simulated GBM underlying asset price paths.
        """
        dW = np.random.normal(loc=0.0,
                              scale=np.sqrt(self.dt),
                              size=(self.M, self.steps - 1))    # (M, steps-1)

        # multiplicative Euler step: S_{n+1} = S_n * (1 + mu*dt + sigma*dW)
        factors = 1 + self.mu * self.dt + self.sigma * dW
        factors = np.hstack((np.ones((self.M, 1)), factors))    # insert initial factor = 1
        S = self.S0 * np.cumprod(factors, axis=1)               # (M, steps)
        
        self.S_GBM = S
        self.dW_GBM = dW

        return S

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
    

    def simulate_H(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate H(t,S) = E[(S_T - K)^+ | S_t = S] and derivatives using a single-pass 
        LSMC (least-squares Monte Carlo) regression on the already-simulated paths.

        Approach
        --------
        - Regress payoff Y = (S_T - K)^+ on a tensor Chebyshev basis in (t, x), where x = log(S) using ridge regularization.
        - Compute derivatives analytically from the fitted basis in (t, x).
        - Convert x-derivatives to S-derivatives via chain rule.

        Returns
        -------
        H : ndarray, shape (M, steps)
            Simulated contigent claim price path.
        dH_dS : ndarray, shape (M, steps)
            First-order derivative against underlying asset.
        d2H_dSS : ndarray, shape (M, steps)
            Second-order derivative against underlying asset. 
        dH_dt : ndarray, shape (M, steps)
            First-order derivative against time-step.
        d2H_dt_dS : ndarray, shape (M, steps)
            Second-order derivative against time-step and underlying asset.
        d3H_dS3 : ndarray, shape (M, steps)
            Third-order derivative against underlying asset.
        """

        if getattr(self, "S_GBM", None) is None:
            raise ValueError("self.S_GBM is None. Run simulate_S() first.")

        # ----------------------------
        # Hyperparameters (tunable)
        # ----------------------------
        p_t = 1              # Chebyshev degree in time
        p_x = 2              # Chebyshev degree in x = log(S)
        lam_ridge = 1e-2     # ridge regularization (increase if derivatives are noisy)

        # ----------------------------
        # 1) Targets and state
        # ----------------------------
        # payoff per path (same target used across all times for that path)
        Y = np.maximum(self.S_GBM[:, -1] - self.K, 0.0) 

        # x = log(S) (work in x for numerical stability)
        X_path = np.log(np.maximum(self.S_GBM, 1e-300))                             # (M, self.steps)

        # time grid
        t_grid = self.t0 + self.dt * np.arange(self.steps)                          # (self.steps, )

        # ----------------------------
        # 2) Scale (t, x) to [-1, 1]
        # ----------------------------
        x_min = float(X_path.min())
        x_max = float(X_path.max())
        if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
            raise ValueError("Invalid X_path range for scaling.")

        zt = tools._minmax_scale(t_grid, self.t0, self.T)                           # (self.steps, )
        zx = tools._minmax_scale(X_path.reshape(-1), x_min, x_max)                  # (M*self.steps, )

        # scaling factors for derivatives:
        # z = a*u + b => dz/du = a
        dz_dt = 2.0 / (self.T - self.t0)
        dz_dx = 2.0 / (x_max - x_min)

        # ----------------------------
        # 3) Chebyshev basis and derivatives in (t, zt) and (x, zx)
        # ----------------------------
        # time basis: (N_time, p_t+1)
        Tt = tools._cheb_eval_all(zt, p_t, deriv=0)
        dTt_dz = tools._cheb_eval_all(zt, p_t, deriv=1)
        dTt_dt = dTt_dz * dz_dt                                                    # chain rule

        # x basis on all observations (flattened): (M*N_time, p_x+1)
        Tx = tools._cheb_eval_all(zx, p_x, deriv=0)
        dTx_dz = tools._cheb_eval_all(zx, p_x, deriv=1)
        d2Tx_dz2 = tools._cheb_eval_all(zx, p_x, deriv=2)
        d3Tx_dz3 = tools._cheb_eval_all(zx, p_x, deriv=3)

        # convert z-derivatives to x-derivatives (x = log S):
        dTx_dx   = dTx_dz   * (dz_dx)
        d2Tx_dx2 = d2Tx_dz2 * (dz_dx ** 2)
        d3Tx_dx3 = d3Tx_dz3 * (dz_dx ** 3)

        # ----------------------------
        # 4) Build design matrix Phi for tensor basis in (t, x)
        #    Phi row corresponds to one observation (m,n).
        # ----------------------------
        N_obs = self.M * self.steps
        P = (p_t + 1) * (p_x + 1)
        Phi = np.empty((N_obs, P), dtype=float)

        # Repeat time basis rows M times to align with flattened X_path ordering
        # Flatten convention used above: X_path.reshape(-1) is row-major -> time index changes fastest.
        # That means order is: (m=0,n=0..), (m=1,n=0..), ...
        # So time basis for each m is the same t_grid, repeated per path.
        col = 0
        for i in range(p_t + 1):
            tt_rep = np.tile(Tt[:, i], self.M)                                  # (N_obs, )
            for j in range(p_x + 1):
                Phi[:, col] = tt_rep * Tx[:, j]
                col += 1

        # Targets replicated per time within each path:
        y = np.repeat(Y, self.steps)                                            # (N_obs, )

        # ----------------------------
        # 5) Fit ridge regression
        # ----------------------------
        beta = tools._ridge_solve(Phi, y, lam=lam_ridge)                        # (P, )

        # reshape coefficients into (p_t+1, p_x+1) for clearer evaluation
        B = beta.reshape(p_t + 1, p_x + 1)

        # ----------------------------
        # 6) Evaluate H and derivatives in (t, x)
        # ----------------------------
        # We evaluate on the same observation grid (m,n) using flattened arrays for x-basis,
        # then reshape back to (M, N_time).
        H_flat    = np.zeros(N_obs, dtype=float)
        Ht_flat   = np.zeros(N_obs, dtype=float)
        Hx_flat   = np.zeros(N_obs, dtype=float)
        Hxx_flat  = np.zeros(N_obs, dtype=float)
        Hxxx_flat = np.zeros(N_obs, dtype=float)
        Htx_flat  = np.zeros(N_obs, dtype=float)

        for i in range(p_t + 1):
            tt0 = np.tile(Tt[:, i], self.M)                                      # (N_obs, )
            tt1 = np.tile(dTt_dt[:, i], self.M)                                  # (N_obs, )
            for j in range(p_x + 1):
                Bij = B[i, j]
                if Bij == 0.0:
                    continue
                base = tt0 * Tx[:, j]
                H_flat += Bij * base

                # time derivative
                Ht_flat += Bij * (tt1 * Tx[:, j])

                # x-derivatives
                Hx_flat   += Bij * (tt0 * dTx_dx[:, j])
                Hxx_flat  += Bij * (tt0 * d2Tx_dx2[:, j])
                Hxxx_flat += Bij * (tt0 * d3Tx_dx3[:, j])

                # mixed derivative (t,x)
                Htx_flat  += Bij * (tt1 * dTx_dx[:, j])

        H = H_flat.reshape(self.M, self.steps)
        H = np.maximum(H, 0)
        H_t = Ht_flat.reshape(self.M, self.steps)
        H_x = Hx_flat.reshape(self.M, self.steps)
        H_xx = Hxx_flat.reshape(self.M, self.steps)
        H_xxx = Hxxx_flat.reshape(self.M, self.steps)
        H_tx = Htx_flat.reshape(self.M, self.steps)

        # ----------------------------
        # 7) Convert x=log(S) derivatives to S-derivatives
        # ----------------------------
        S = np.maximum(self.S_GBM, 1e-300)
        dH_dS = (1.0 / S) * H_x
        d2H_dSS = (1.0 / (S ** 2)) * (H_xx - H_x)
        d3H_dS3 = (1.0 / (S ** 3)) * (H_xxx - 3.0 * H_xx + 2.0 * H_x)
        dH_dt = H_t
        d2H_dtdS = (1.0 / S) * H_tx

        # ----------------------------
        # 8) Store 
        # ----------------------------
        self.H_GBM = H
        self.dH_dS_GBM = dH_dS
        self.d2H_dSS_GBM = d2H_dSS
        self.dH_dt_GBM = dH_dt
        self.d2H_dtdS_GBM = d2H_dtdS
        self.d3H_dS3_GBM = d3H_dS3

        # Keep regression metadata if useful for debugging/reuse
        self._H_fit_GBM = {
            "p_t": p_t,
            "p_x": p_x,
            "lam_ridge": lam_ridge,
            "x_min": x_min,
            "x_max": x_max,
            "beta": beta,
        }

        return H, dH_dS, d2H_dSS, dH_dt, d2H_dtdS, d3H_dS3



    # ============================================================
    # 1. Control initialization
    # ============================================================

    def init_control(self, kind: str = "Delta") -> np.ndarray:
        """
        Initialize the control h.

        Parameters
        ----------
        kind : {"Delta", "zero"}
            Initialization rule.

        Returns
        -------
        h : ndarray, shape (M, steps-1)
            Initial control on each time interval [t_n, t_{n+1}).
        """
        if kind == "Delta":
            h0 = 0.7 * self.dH_dS_GBM[:, :-1].copy()                            # smooth initial guess
        elif kind == "zero":
            h0 = np.zeros((self.M, self.steps - 1))
        else:
            raise ValueError("Unknown control initialization kind")

        return h0



    # ============================================================
    # 2. Forward Profit and Loss
    # ============================================================
    
    def forward_PL(self, h: np.ndarray, L0: float = 0.0) -> np.ndarray:
        """
        Vectorized Profit and Loss L_t using portfolio value:

            V_t^h = h_t S_t - H_t
            Delta L_n = V_{n+1}^h - V_n^h
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
            Profit and Loss paths over time.
        """
        assert h.shape == (self.M, self.steps - 1)

        L = np.zeros((self.M, self.steps))
        L[:, 0] = L0

        # increments
        dS = np.diff(self.S_GBM, axis=1)         
        dH = np.diff(self.H_GBM, axis=1)          
        dL = h * dS - dH

        L = np.empty((self.M, self.steps), dtype=float)
        L[:, 0] = L0
        L[:, 1:] = L0 + np.cumsum(dL, axis=1)

        return L


    # ============================================================
    # 3. Risk functional
    # ============================================================

    @staticmethod
    def risk_function(LT: np.ndarray, risk_type: str, **kwargs) -> float:
        """
        Compute the composed risk functional rho_u(L_T) from terminal Profit and Loss simulation.

        Parameters
        ----------
        LT : ndarray, shape (M,)
            Terminal Profit and Loss samples L_T^h generated by a given hedging strategy.
        risk_type : str
            Type of composed risk measure to be evaluated. Supported values are:
                - "ele"  : Expected loss with exponential utility
                - "elw"  : Expected loss with Weibull-type utility
                - "entl" : Entropic risk with linear utility
                - "ente" : Entropic risk with exponential utility
                - "entw" : Entropic risk with Weibull-type utility
                - "esl"   : Expected shortfall linear utility
        **kwargs :
            Parameters required by the chosen risk_type:
                - "a"     : risk aversion parameter (exponential utility)
                - "k"     : shape parameter (Weibull utility)
                - "gamma" : entropic risk aversion parameter
                - "beta"  : confidence level for expected shortfall

        Returns
        -------
        rho : float
            Estimation of the composed risk functional rho_u(L_T).
        """
        LT = np.asarray(LT)

        # ----------------------------
        # ele
        # ----------------------------
        if risk_type == "ele":
            a = kwargs.get("a")
            if a is None:
                raise ValueError("Parameter 'a' is required for 'ele'.")
            rho = np.mean(np.exp(-a * LT))

        # ----------------------------
        # elw
        # ----------------------------
        elif risk_type == "elw":
            k = kwargs.get("k")
            if k is None:
                raise ValueError("Parameter 'k' is required for 'elw'.")
            rho = np.mean(np.exp((-np.minimum(LT, 0.0))**k))

        # ----------------------------
        # entl
        # ----------------------------
        elif risk_type == "entl":
            gamma = kwargs.get("gamma")
            if gamma is None:
                raise ValueError("Parameter 'gamma' is required for 'entl'.")
            rho = (1.0 / gamma) * np.log(np.mean(np.exp(-gamma * LT)))

        # ----------------------------
        # ente
        # ----------------------------
        elif risk_type == "ente":
            gamma = kwargs.get("gamma")
            a = kwargs.get("a")
            if gamma is None or a is None:
                raise ValueError("Parameters 'gamma' and 'a' are required for 'ente'.")
            rho = (1.0 / gamma) * np.log(np.mean(np.exp(gamma * np.exp(-a * LT))))

        # ----------------------------
        # entw
        # ----------------------------
        elif risk_type == "entw":
            gamma = kwargs.get("gamma")
            k     = kwargs.get("k")
            if gamma is None or k is None:
                raise ValueError("Parameters 'gamma' and 'k' are required for risk_type='entw'.")
            rho = (1.0 / gamma) * np.log(np.mean(np.exp(gamma * np.exp((-np.minimum(LT, 0.0))**k)) ))

        # ----------------------------
        # esl
        # ----------------------------
        elif risk_type == "esl":
            beta = kwargs.get("beta")
            if beta is None:
                raise ValueError("Parameter 'beta' is required for risk_type='esl'.")
            LT = -LT
            alpha = np.quantile(LT, beta)
            rho = alpha + (1.0 / (1.0 - beta)) * np.mean(np.maximum(LT - alpha, 0.0))

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
