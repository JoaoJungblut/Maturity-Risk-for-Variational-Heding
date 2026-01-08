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
        self.J_Jump = np.hstack((np.ones((self.M, 1)), J))  # (M, steps)

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
                L0: float = 0,
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
    def backward_adjoint(self,
                        pT: np.ndarray,
                        p_x: int = 2,
                        lam_ridge: float = 1e-2,
                        lam_time: float = 1e-2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve the backward adjoint BSDE for the Merton jump diffusion model
        by a discrete-time backward scheme combined with regression.

        Numerical scheme
        ----------------
        - Time is discretized on the same grid used for the forward simulation.
        - The backward equation is solved iteratively for n = N-1, ..., 0.
        - Three backward processes are estimated:
            * p_n : adjoint process
            * q_n : martingale component for the Brownian motion
            * r_n : martingale component for the Poisson jump process
        - For each n:
            * q_n is obtained from the martingale representation
            q_n = E[p_{n+1} Delta W_n | S_n] / Delta t.
            * r_n is obtained from the martingale representation
            r_n = E[p_{n+1} Delta N_n | S_n] / Delta t.
            * p_n is obtained from the drift part of the adjoint dynamics,
            using a first-order Euler discretization that includes the
            second- and third-order sensitivities of H(t,S) and the
            jump sensitivity term involving H_S(t, J S).

        Parameters
        ----------
        pT : ndarray, shape (M,)
            Terminal adjoint values p_T, obtained from the derivative of the
            terminal risk functional with respect to the terminal P and L.
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
            Martingale component q_t of the adjoint process (Brownian).
        r : ndarray, shape (M, steps-1)
            Martingale component r_t of the adjoint process (Poisson jumps).
        """
        if getattr(self, "S_Jump", None) is None:
            raise ValueError("Missing S_Jump. Run simulate_S() first.")
        if getattr(self, "dW_Jump", None) is None:
            raise ValueError("Missing dW_Jump. Run simulate_S() first.")
        if getattr(self, "dN_Jump", None) is None:
            raise ValueError("Missing dN_Jump. Run simulate_S() first.")
        if getattr(self, "J_Jump", None) is None:
            raise ValueError("Missing J_Jump. Run simulate_S() first.")

        req = [
            "dH_dS_Jump",
            "d2H_dSS_Jump",
            "d2H_dtdS_Jump",
            "d3H_dS3_Jump",
            "dH_dS_jump_Jump",
        ]
        for name in req:
            if getattr(self, name, None) is None:
                raise ValueError(f"Missing {name}. Run simulate_H() first.")

        # ------------------------------------------------------------
        # 1) Allocate backward processes
        # ------------------------------------------------------------
        p = np.zeros((self.M, self.steps), dtype=float)
        q = np.zeros((self.M, self.steps - 1), dtype=float)
        r = np.zeros((self.M, self.steps - 1), dtype=float)
        p[:, -1] = np.asarray(pT, dtype=float)

        # ------------------------------------------------------------
        # 2) Global regressor: x = log(S)
        # ------------------------------------------------------------
        S_all = np.maximum(self.S_Jump, 1e-300)
        X_all = np.log(S_all)

        x_min = float(X_all.min())
        x_max = float(X_all.max())
        if (not np.isfinite(x_min)) or (not np.isfinite(x_max)) or (x_max <= x_min):
            raise ValueError("Invalid scaling range for log(S).")

        P = p_x + 1
        beta_p_next = np.zeros(P)
        beta_q_next = np.zeros(P)
        beta_r_next = np.zeros(P)

        # ------------------------------------------------------------
        # 3) Backward loop
        # ------------------------------------------------------------
        for n in range(self.steps - 2, -1, -1):
            Sn = self.S_Jump[:, n]
            dWn = self.dW_Jump[:, n]
            dNn = self.dN_Jump[:, n]

            # jump size aligned to time index n
            # - if J_Jump is stored per interval, use J_Jump[:, n]
            # - if J_Jump is stored on the full grid, use J_Jump[:, n]
            Jbuf = self.J_Jump
            if Jbuf.ndim != 2 or Jbuf.shape[0] != self.M:
                raise ValueError("Invalid shape for J_Jump.")
            if Jbuf.shape[1] == self.steps - 1:
                Jn = Jbuf[:, n]
            elif Jbuf.shape[1] == self.steps:
                Jn = Jbuf[:, n]
            else:
                raise ValueError("J_Jump must have shape (M, steps-1) or (M, steps).")

            # --------------------------------------------------------
            # 3.1) Regressors: phi(x_n)
            # --------------------------------------------------------
            x_n = X_all[:, n]
            z = tools._minmax_scale(x_n, x_min, x_max)
            Phi = tools._cheb_eval_all(z, p_x, deriv=0)

            p_next = p[:, n + 1]

            # --------------------------------------------------------
            # 3.2) q_n via martingale representation
            # --------------------------------------------------------
            y_q = (p_next * dWn) / self.dt
            beta_q = tools._solve_smoothed(Phi, y_q, beta_q_next, lam_ridge, lam_time)
            q[:, n] = Phi @ beta_q

            # --------------------------------------------------------
            # 3.3) r_n via martingale representation
            # --------------------------------------------------------
            y_r = (p_next * dNn) / self.dt
            beta_r = tools._solve_smoothed(Phi, y_r, beta_r_next, lam_ridge, lam_time)
            r[:, n] = Phi @ beta_r

            # --------------------------------------------------------
            # 3.4) Driver terms (paper equation for jump diffusion adjoint)
            # --------------------------------------------------------
            H_S = self.dH_dS_Jump[:, n]
            H_SS = self.d2H_dSS_Jump[:, n]
            H_tS = self.d2H_dtdS_Jump[:, n]
            H_SSS = self.d3H_dS3_Jump[:, n]
            H_S_jump = self.dH_dS_jump_Jump[:, n]

            # alpha_n multiplies p
            alpha_n = (
                H_tS
                + self.mu * Sn * H_SS
                + 0.5 * (self.sigma ** 2) * (Sn ** 2) * H_SSS
                + (self.sigma ** 2) * Sn * H_SS
            )

            # beta_n multiplies q with a minus sign
            beta_n = self.sigma * (H_S + Sn * H_SS)

            # gamma_n multiplies r with a minus sign
            gamma_n = (Jn - 1.0) - (Jn * H_S_jump - H_S)

            # full drift for p at time n (evaluated using p_next, q_n, r_n)
            drift_p = alpha_n * p_next - beta_n * q[:, n] - gamma_n * r[:, n]

            # --------------------------------------------------------
            # 3.5) p_n via Euler drift and denoising regression
            # --------------------------------------------------------
            y_p = p_next - drift_p * self.dt
            beta_p = tools._solve_smoothed(Phi, y_p, beta_p_next, lam_ridge, lam_time)
            p[:, n] = Phi @ beta_p

            # update smoothed coefficient states
            beta_q_next = beta_q
            beta_r_next = beta_r
            beta_p_next = beta_p

        # ------------------------------------------------------------
        # 4) Store regression metadata
        # ------------------------------------------------------------
        self._backward_fit_Jump = {
            "p_x": p_x,
            "lam_ridge": lam_ridge,
            "lam_time": lam_time,
            "x_min": x_min,
            "x_max": x_max,
        }

        self.p_Jump = p
        self.q_Jump = q
        self.r_Jump = r

        return p, q, r



    # ============================================================
    # 6. Gradient (optimality condition)
    # ============================================================
    def compute_gradient(self,
                        p: np.ndarray,
                        q: np.ndarray,
                        r: np.ndarray) -> np.ndarray:
        """
        Compute the violation of the local optimality condition for Merton jump diffusion.

        Paper first order condition:
            0 = mu S_t p_t + sigma S_t q_t + S_{t-} (J - 1) r_t

        Parameters
        ----------
        p : ndarray, shape (M, steps)
            Adjoint process p.
        q : ndarray, shape (M, steps-1)
            Martingale component for dW.
        r : ndarray, shape (M, steps-1)
            Martingale component for dN.

        Returns
        -------
        G : ndarray, shape (M, steps-1)
            Gradient like quantity used to update the control.
        """
        assert p.shape == (self.M, self.steps)
        assert q.shape == (self.M, self.steps - 1)
        assert r.shape == (self.M, self.steps - 1)

        S_trunc = self.S_Jump[:, :-1]

        # J aligned to increments on [t_n, t_{n+1})
        Jbuf = self.J_Jump
        if Jbuf.shape[1] == self.steps - 1:
            J_trunc = Jbuf
        elif Jbuf.shape[1] == self.steps:
            J_trunc = Jbuf[:, :-1]
        else:
            raise ValueError("J_Jump must have shape (M, steps-1) or (M, steps).")

        G = (self.mu * S_trunc) * p[:, :-1] \
            + (self.sigma * S_trunc) * q \
            + (S_trunc * (J_trunc - 1.0)) * r

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
        Main optimization loop to compute an approximate optimal hedge h for jump diffusion.

        Same structure as GBM:
        - Keep the last two accepted controls: h_curr (best/current), h_prev (previous).
        - At each iteration, compute grad_norm at h_curr.
        - If the first trial step from h_curr with current alpha does NOT improve grad_norm,
        rollback immediately to h_prev, shrink alpha, and search from there.
        - During the search, the anchor stays fixed; alpha is halved until improvement is found.
        - If improvement cannot be found (alpha too small), stop.

        Parameters
        ----------
        risk_type : str
            One of {"ele", "elw", "entl", "ente", "entw", "esl"}.
        risk_kwargs : dict
            Parameters required for the chosen risk_type.
        t_idx : float, default=0
            Time index t at which P and L accumulation starts.
        kind : {"Delta", "MinVar", "zero"}
            Initialization rule.
        max_iter : int
            Maximum number of iterations.
        tol : float
            Tolerance on mean square gradient (stopping criterion).
        alpha : float
            Step size for control updates.
        verbose : bool
            If True, print iteration logs.

        Returns
        -------
        h_opt : ndarray, shape (M, steps-1)
            Approximate optimal control.
        history : list of dict
            Iteration history (gradient norm, etc.).
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

            p, q, r = self.backward_adjoint(pT)
            G_curr = self.compute_gradient(p, q, r)
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

            p_try, q_try, r_try = self.backward_adjoint(pT_try)
            G_try = self.compute_gradient(p_try, q_try, r_try)
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

            p_a, q_a, r_a = self.backward_adjoint(pT_a)
            G_a = self.compute_gradient(p_a, q_a, r_a)
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

                p_new, q_new, r_new = self.backward_adjoint(pT_new)
                G_new = self.compute_gradient(p_new, q_new, r_new)
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
                t_idx: float =0,
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
            verbose=verbose,
        )
        L_t = self.forward_PL(h_t, L0=0.0, t_start=t_idx)
        rho_t = self.risk_function(L_t[:, -1], risk_type, **risk_kwargs)

        MR = rho_t - rho_T

        return MR, {"h_T": h_T, "h_t": h_t, "rho_T": rho_T, "rho_t": rho_t}


