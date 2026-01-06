from OptimalHedging.Simulator import BaseSimulator
import numpy as np
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

    def simulate_H(self, 
                   p_t : int = 1, 
                   p_x : int = 2, 
                   lam_ridge : int = 1e-2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate H(t,S) = E[(S_T - K)^+ | S_t = S] and derivatives using a single-pass 
        LSMC (least-squares Monte Carlo) regression on the already-simulated paths.

        Approach
        --------
        - Regress payoff Y = (S_T - K)^+ on a tensor Chebyshev basis in (t, x), where x = log(S) using ridge regularization.
        - Compute derivatives analytically from the fitted basis in (t, x).
        - Convert x-derivatives to S-derivatives via chain rule.

        Parameters
        ----------
        p_t : int, default=1
            Chebyshev degree in time
        p_x : int, default=2
            Chebyshev degree in x = log(S)
        lam_ridge: int, default=1e-2 
            Ridge regularization
        
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

    def init_control(self, 
                     kind: str = "Delta") -> np.ndarray:
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
            h0 = 1 * self.dH_dS_GBM[:, :-1].copy()                            # smooth initial guess
        elif kind == "zero":
            h0 = np.zeros((self.M, self.steps - 1))
        else:
            raise ValueError("Unknown control initialization kind")

        return h0



    # ============================================================
    # 2. Forward Profit and Loss
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
        start_idx = tools._time_to_index(N=self.N, t0=self.t0, T=self.T, t_start=t_start)


        # increments occur at times 1,...,steps-1
        dS = np.diff(self.S_GBM, axis=1)        # (M, steps-1)
        dH = np.diff(self.H_GBM, axis=1)        # (M, steps-1)
        dL = h * dS - dH                        # (M, steps-1)

        # cumulative P&L
        L = np.zeros((self.M, self.steps))
        L[:, 0] = L0
        L[:, 1:] = L0 + np.cumsum(dL, axis=1)

        # cut past
        if start_idx > 0:
            L[:, :start_idx] = 0.0

        return L



    # ============================================================
    # 3. Risk functional
    # ============================================================
    @staticmethod
    def risk_function(LT: np.ndarray, 
                      risk_type: str, 
                      **kwargs) -> float:
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
    def terminal_adjoint(LT: np.ndarray, 
                         risk_type: str, 
                         **kwargs) -> np.ndarray:
        """
        Compute the terminal adjoint p_T = Upsilon(L_T) for the composed risk functional.

        Parameters
        ----------
        LT : ndarray, shape (M,)
            Terminal Profit and Loss samples L_T^h generated by a given hedging strategy.
        risk_type : str
            Type of composed risk measure. Supported values are:
                - "ele"  : Expected loss with exponential utility
                - "elw"  : Expected loss with Weibull-type utility
                - "entl" : Entropic risk with linear utility
                - "ente" : Entropic risk with exponential utility
                - "entw" : Entropic risk with Weibull-type utility
                - "esl"  : Expected shortfall with linear utility 
        **kwargs :
            Parameters required by the chosen risk_type:
                - "a"     : risk aversion parameter (exponential utility)
                - "k"     : shape parameter (Weibull utility)
                - "gamma" : entropic risk aversion parameter
                - "beta"  : confidence level for expected shortfall

        Returns
        -------
        pT : ndarray, shape (M,)
            Terminal adjoint samples p_T = Upsilon(L_T).
        """
        LT = np.asarray(LT)

        # ----------------------------
        # ele
        # ----------------------------
        if risk_type == "ele":
            a = kwargs.get("a")
            if a is None:
                raise ValueError("Parameter 'a' is required for 'ele'.")
            # Gamma(X) = -a e^{-a X}
            pT = -a * np.exp(-a * LT)

        # ----------------------------
        # elw
        # ----------------------------
        elif risk_type == "elw":
            k = kwargs.get("k")
            U = kwargs.get("U", 5.0)   
            if k is None:
                raise ValueError("Parameter 'k' is required for 'elw'.")
            u = np.minimum((-np.minimum(LT, 0.0))**k,U)
            pT = -k * (-np.minimum(LT, 0.0)) **(k - 1.0) * np.exp(u)

        # ----------------------------
        # entl
        # ----------------------------
        elif risk_type == "entl":
            gamma = kwargs.get("gamma")
            U = kwargs.get("U", 5.0)   
            if gamma is None:
                raise ValueError("Parameter 'gamma' is required for 'entl'.")
            u = np.clip(-gamma * LT, -U, U)
            pT = -np.exp(u) / np.mean(np.exp(u))

        # ----------------------------
        # ente
        # ----------------------------
        elif risk_type == "ente":
            gamma = kwargs.get("gamma")
            a = kwargs.get("a")
            U = kwargs.get("U", 5.0)   
            if gamma is None or a is None:
                raise ValueError("Parameters 'gamma' and 'a' are required for 'ente'.")
            u1 = np.clip(-a * LT, -U, U)
            u2 = np.minimum(gamma * np.exp(u1), U)
            pT = -a * np.exp(u1) * np.exp(u2)  / np.mean(np.exp(u2))

        # ----------------------------
        # entw
        # ----------------------------
        elif risk_type == "entw":
            gamma = kwargs.get("gamma")
            k = kwargs.get("k")
            U = kwargs.get("U", 5.0)
            if gamma is None or k is None:
                raise ValueError("Parameters 'gamma' and 'k' are required for risk_type='entw'.")
            u1 = np.minimum((-np.minimum(LT, 0.0) )**k, U)     
            u2 = np.minimum(gamma * np.exp(u1), U)  
            pT = -k * (-np.minimum(LT, 0.0))**(k - 1.0) * np.exp(u1) * np.exp(u2) / np.mean(np.exp(u2))

        # ----------------------------
        # esl
        # ----------------------------
        elif risk_type == "esl":
            beta = kwargs.get("beta")
            if beta is None:
                raise ValueError("Parameter 'beta' is required for risk_type='esl'.")
            # ES is applied to losses: X = -L_T
            LT = -LT
            alpha = np.quantile(LT, beta) 
            pT = (LT >= alpha).astype(float) / (1.0 - beta)

        else:
            raise ValueError(f"Unknown risk_type '{risk_type}'.")

        return pT
    


    # ============================================================
    # 5. Backward adjoint (simplified driver)
    # ============================================================
    def backward_adjoint(self,
                         pT: float,
                         p_x: int = 2,
                         lam_ridge: float = 1e-2,
                         lam_time: float = 1e-2) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve the backward adjoint BSDE associated with the optimal hedging problem 
        by a discrete-time backward scheme combined with regression.

        Numerical scheme
        ----------------
        - Time is discretized on the same grid used for the forward simulation.
        - The backward equation is solved iteratively for n = N-1, ..., 0.
        - For each n:
            * q_n is obtained from the martingale representation q_n = E[p_{n+1} Delta W_n | S_n] / Delta t.
            * p_n is obtained from the drift part of the adjoint dynamics,
            using a first-order Euler discretization that includes the
            second- and third-order sensitivities of H(t,S).

        Regression method
        -----------------
        - The conditional expectations are approximated using a Chebyshev polynomial
        basis in the regressor x = log(S).
        - A ridge-regularized least-squares fit is performed at each time step.
        - Intertemporal smoothing of the regression coefficients is applied to
        stabilize the backward propagation.

        Parameters
        ----------
        pT : ndarray, shape (M,)
            Terminal adjoint values p_T, obtained from the derivative of the
            terminal risk functional with respect to the terminal P&L.
        p_x : int, default=2
            Degree of the Chebyshev polynomial basis used in the regression.
        lam_ridge : float, default=1e-2
            Ridge regularization parameter for the cross-sectional regressions.
        lam_time : float, default=1e-2
            Temporal smoothing parameter for regression coefficients across time.

        Returns
        -------
        p : ndarray, shape (M, steps)
            Backward adjoint process p_t along each simulated path.
        q : ndarray, shape (M, steps-1)
            Martingale component q_t of the adjoint process.
        """
        p = np.zeros((self.M, self.steps), dtype=float)
        q = np.zeros((self.M, self.steps - 1), dtype=float)
        p[:, -1] = pT

        # global regressor: x = log(S)
        S_all = np.maximum(self.S_GBM, 1e-300)
        X_all = np.log(S_all)

        x_min = float(X_all.min())
        x_max = float(X_all.max())
        if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
            raise ValueError("Invalid scaling range for log(S).")

        P = p_x + 1
        beta_p_next = np.zeros(P)
        beta_q_next = np.zeros(P)

        for n in range(self.steps - 2, -1, -1):
            Sn = self.S_GBM[:, n]
            dWn = self.dW_GBM[:, n]

            # drift coefficient A_n
            d2H_dtdS_n = self.d2H_dtdS_GBM[:, n]
            d2H_dSS_n  = self.d2H_dSS_GBM[:, n]
            d3H_dS3_n  = self.d3H_dS3_GBM[:, n]

            A_n = d2H_dtdS_n + 0.5 * (self.sigma**2) * (2.0 * Sn * d2H_dSS_n + (Sn**2) * d3H_dS3_n)

            p_next = p[:, n + 1]

            # regressors: phi(x_n)
            x_n = X_all[:, n]
            z = tools._minmax_scale(x_n, x_min, x_max)
            Phi = tools._cheb_eval_all(z, p_x, deriv=0)

            # q_n
            y_q = (p_next * dWn) / self.dt
            beta_q = tools._solve_smoothed(Phi, y_q, beta_q_next, lam_ridge, lam_time)
            q[:, n] = Phi @ beta_q

            # p_n
            y_p = p_next - A_n * p_next * self.dt
            beta_p = tools._solve_smoothed(Phi, y_p, beta_p_next, lam_ridge, lam_time)
            p[:, n] = Phi @ beta_p

            beta_q_next = beta_q
            beta_p_next = beta_p

        # Keep regression metadata if useful for debugging/reuse
        self._backward_fit_GBM = {
            "p_x": p_x,
            "lam_ridge": lam_ridge,
            "lam_time": lam_time,
            "x_min": x_min,
            "x_max": x_max,
        }

        self.p_GBM = p
        self.q_GBM = q
        return p, q



    # ============================================================
    # 6. Gradient (optimality condition)
    # ============================================================
    def compute_gradient(self, 
                         p: np.ndarray, 
                         q: np.ndarray) -> np.ndarray:
        """
        Compute the violation of the local optimality condition
        (e.g. G_n = mu S_n p_n + sigma S_n q_n for GBM).

        Parameters
        ----------
        p : ndarray, shape (M, steps)
        q : ndarray, shape (M, steps-1)

        Returns
        -------
        G : ndarray, shape (M, steps-1)
            Gradient-like quantity used to update the control.
        """
        assert p.shape == (self.M, self.steps)
        assert q.shape == (self.M, self.steps - 1)

        S_trunc = self.S_GBM[:, :-1]
        G = self.mu * S_trunc * p[:, :-1] + self.sigma * S_trunc * q
        return G




    # ============================================================
    # 7. Control update
    # ============================================================
    @staticmethod
    def update_control(h: np.ndarray,
                       G: np.ndarray,
                       alpha: float = 1,
                       eps: float = 1e-12) -> np.ndarray:
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
        eps : float, optional
            Small constant to avoid division by zero.

        Returns
        -------
        h_new : ndarray, shape (M, steps-1)
            Updated control.
        """
        assert h.shape == G.shape
        norm_G = np.linalg.norm(G) / np.sqrt(G.size) + eps
        G_normalized = G / norm_G
        h_new = h - alpha * G_normalized

        return h_new



    # ============================================================
    # 8. Main optimization loop
    # ============================================================
    def optimize_hedge(self,
                    risk_type: str,
                    risk_kwargs: Dict,
                    t_idx: int = 0,
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
        t_idx : int, default=0
            Time index t at which P&L accumulation starts.
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
        h_curr = self.init_control(kind="Delta")
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
            p, q = self.backward_adjoint(pT)
            G_curr = self.compute_gradient(p, q)
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
            p_try, q_try = self.backward_adjoint(pT_try)
            G_try = self.compute_gradient(p_try, q_try)
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
                if verbose: print("rollback disabled (k < 2)")
                continue
            
            # ----- ROLLBACK / BACKTRACKING ANCHOR -----
            # If no previous accepted control exists (e.g. k=0), anchor at current control (do not stop).
            anchor_h = h_prev if (h_prev is not None) else h_curr
            alpha_try = alpha * shrink

            # recompute gradient at anchor (required)
            L_a = self.forward_PL(anchor_h, L0=0.0, t_start=t_idx)
            LT_a = L_a[:, -1]
            pT_a = self.terminal_adjoint(LT_a, risk_type, **risk_kwargs)
            p_a, q_a = self.backward_adjoint(pT_a)
            G_a = self.compute_gradient(p_a, q_a)
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
                p_new, q_new = self.backward_adjoint(pT_new)
                G_new = self.compute_gradient(p_new, q_new)
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
                    # accept: shift chain and set current alpha to the accepted one
                    h_prev = anchor_h  # will be h_curr if we anchored at curr
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
                   t_idx: int,
                   risk_type: str,
                   risk_kwargs: Dict,
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
        t_idx : int
            Time index t at which the maturity risk is evaluated.
            Must satisfy 0 <= t_idx <= self.steps.
        risk_type : str
            Risk functional identifier (e.g. "ele", "elw", "entl", "ente", "entw", "esl").
        risk_kwargs : dict
            Parameters required by the chosen risk functional.
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

        # --- full horizon [0, T] ---
        h_T, _ = self.optimize_hedge(
            risk_type=risk_type,
            risk_kwargs=risk_kwargs,
            t_idx=self.T,
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
            max_iter=max_iter,
            tol=tol,
            alpha=alpha,
            verbose=verbose
        )
        L_t = self.forward_PL(h_t, L0=0.0, t_start=t_idx)
        rho_t = self.risk_function(L_t[:, -1], risk_type, **risk_kwargs)

        MR = rho_t - rho_T

        return MR, {"h_T": h_T, "h_t": h_t, "rho_T": rho_T, "rho_t": rho_t}

    
