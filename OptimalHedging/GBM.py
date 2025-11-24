from OptimalHedging.Simulator import BaseSimulator
import numpy as np

class GBMSimulator(BaseSimulator):
    """
    Simulator for Geometric Brownian Motion (GBM).
    """

    def simulate_S(self) -> np.ndarray:
        """
        Simulate Geometric Brownian Motion (GBM) paths using a fully vectorized Euler–Maruyama scheme.
            dS = mu * S * dt + sqrt(v) * S * dW

        Returns
        -------
        S : ndarray, shape (M, N+1)
            Simulated GBM asset price paths.
        """
        steps = int(np.round((self.T - self.t0) / self.dt))
        dW = np.random.normal(loc=0.0, 
                                   scale=np.sqrt(self.dt), 
                                   size=(self.M, steps - 1))    # Generate Brownian increments (M x N-1) as N(0,1) 
        factors = 1 + self.mu * self.dt + self.sigma * dW       # Compute multiplicative factors for all steps
        factors = np.hstack((np.ones((self.M, 1)), factors))    # Insert a column of ones for the initial asset value
        S_path = self.S0 * np.cumprod(factors, axis=1)          # Cumulative product along time to build paths

        self.S_path = S_path
        self.dW = dW

        return S_path
    
    
    def simulate_H(self):
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
        n_inner  = 200   # inner MC paths per (t_n, s_k) (tunable)

        S_grid = np.linspace(S_min, S_max, n_S_grid)  # (n_S_grid,)

        # Tables on the space grid:
        # H_grid(t_n, s_k), its S-derivatives and t-derivatives
        H_grid          = np.zeros((N_time, n_S_grid))
        Delta_grid      = np.zeros((N_time, n_S_grid))  
        Gamma_grid      = np.zeros((N_time, n_S_grid))  
        dHdt_grid       = np.zeros((N_time, n_S_grid))  
        d2H_dt_dS_grid  = np.zeros((N_time, n_S_grid))  
        d3H_dS3_grid    = np.zeros((N_time, n_S_grid))  

        # ------------------------------------------------------------
        # 2) Terminal condition at T: H(T,s) = max(s - K, 0)
        # ------------------------------------------------------------
        H_grid[-1, :] = np.maximum(S_grid - K_strike, 0.0)

        # ------------------------------------------------------------
        # 3) Backward in time on the grid via nested Monte Carlo
        #    For each (t_n, s_k), approximate E[ payoff | S_n = s_k ]
        # ------------------------------------------------------------
        for n in reversed(range(N_time - 1)):  # n = N_time-2, ..., 0
            for k, s0 in enumerate(S_grid):
                # inner paths starting at S(t_n) = s0
                S_inner = np.full(n_inner, s0, dtype=float)

                # simulate from t_n to T using Euler–Maruyama
                # number of steps: N_time-1 - n
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
            # First and second derivative in S
            for k in range(1, n_S_grid - 1):
                Delta_grid[n, k] = (H_grid[n, k + 1] - H_grid[n, k - 1]) / (2.0 * dS)
                Gamma_grid[n, k] = (H_grid[n, k + 1] - 2.0 * H_grid[n, k] + H_grid[n, k - 1]) / (dS ** 2)

            # simple boundary handling: copy neighbors
            Delta_grid[n, 0]  = Delta_grid[n, 1]
            Delta_grid[n, -1] = Delta_grid[n, -2]
            Gamma_grid[n, 0]  = Gamma_grid[n, 1]
            Gamma_grid[n, -1] = Gamma_grid[n, -2]

            # Third derivative in S: derivative of Gamma_grid in S
            for k in range(1, n_S_grid - 1):
                d3H_dS3_grid[n, k] = (Gamma_grid[n, k + 1] - Gamma_grid[n, k - 1]) / (2.0 * dS)

            d3H_dS3_grid[n, 0]  = d3H_dS3_grid[n, 1]
            d3H_dS3_grid[n, -1] = d3H_dS3_grid[n, -2]

        # ------------------------------------------------------------
        # 5) Time derivative via finite differences in t
        # ------------------------------------------------------------
        # interior: central difference; boundaries: one-sided
        for n in range(1, N_time - 1):
            dHdt_grid[n, :] = (H_grid[n + 1, :] - H_grid[n - 1, :]) / (2.0 * dt)

        # forward difference at initial time
        dHdt_grid[0, :] = (H_grid[1, :] - H_grid[0, :]) / dt
        # backward difference at terminal time
        dHdt_grid[-1, :] = (H_grid[-1, :] - H_grid[-2, :]) / dt

        # ------------------------------------------------------------
        # 6) Mixed derivative
        # ------------------------------------------------------------
        for n in range(N_time):
            for k in range(1, n_S_grid - 1):
                d2H_dt_dS_grid[n, k] = (dHdt_grid[n, k + 1] - dHdt_grid[n, k - 1]) / (2.0 * dS)

            d2H_dt_dS_grid[n, 0]  = d2H_dt_dS_grid[n, 1]
            d2H_dt_dS_grid[n, -1] = d2H_dt_dS_grid[n, -2]

        # ------------------------------------------------------------
        # 7) Interpolate everything to the simulated paths S_path
        # ------------------------------------------------------------
        H        = np.zeros((M, N_time))
        dH_dS    = np.zeros((M, N_time))
        d2H_dSS  = np.zeros((M, N_time))
        dH_dt    = np.zeros((M, N_time))
        d2H_dt_dS = np.zeros((M, N_time))
        d3H_dS3  = np.zeros((M, N_time))

        for n in range(N_time):
            S_n = S_path[:, n]  # (M,)

            H[:, n]         = np.interp(S_n, S_grid, H_grid[n, :])
            dH_dS[:, n]     = np.interp(S_n, S_grid, Delta_grid[n, :])
            d2H_dSS[:, n]   = np.interp(S_n, S_grid, Gamma_grid[n, :])
            dH_dt[:, n]     = np.interp(S_n, S_grid, dHdt_grid[n, :])
            d2H_dt_dS[:, n] = np.interp(S_n, S_grid, d2H_dt_dS_grid[n, :])
            d3H_dS3[:, n]   = np.interp(S_n, S_grid, d3H_dS3_grid[n, :])

        # ------------------------------------------------------------
        # 8) Store everything on self for use in L and adjoint dynamics
        # ------------------------------------------------------------
        self.S_grid          = S_grid
        self.H_grid          = H_grid
        self.Delta_grid      = Delta_grid
        self.Gamma_grid      = Gamma_grid
        self.dHdt_grid       = dHdt_grid
        self.d2H_dt_dS_grid  = d2H_dt_dS_grid
        self.d3H_dS3_grid    = d3H_dS3_grid

        dH_dt = dH_dt / 100
        d2H_dt_dS = d2H_dt_dS / 100

        self.H        = H
        self.dH_dS    = dH_dS
        self.d2H_dSS  = d2H_dSS
        self.dH_dt    = dH_dt 
        self.d2H_dt_dS = d2H_dt_dS
        self.d3H_dS3  = d3H_dS3

        # Return all pathwise derivatives 
        return H, dH_dS, d2H_dSS, dH_dt, d2H_dt_dS, d3H_dS3

    

    def simulate_L(self, h0=0.5) -> np.ndarray:
        """
        Forward simulation of the profit process L_t.

        Parameters
        ----------
        S_path : ndarray (M, N)
        h : ndarray (M, N)
        dH_dS, d2H_dSS, dH_dt : ndarray (M, N)
        mu, sigma : floats
        dW : ndarray (M, N-1)
        dt : float

        Returns
        -------
        L : ndarray (M, N)
        """
        M, N = self.S_path.shape

        S_n = self.S_path[:, :-1]
        h_n = np.full_like(S_n, h0)
        dH_dS = self.dH_dS[:, :-1]
        dH_SS = self.d2H_dSS[:, :-1]
        dH_dt = self.dH_dt[:, :-1]

        # Drift and diffusion for profit and loss process
        b_n = (h_n - dH_dS) * self.mu * S_n - dH_dt - 0.5 * self.sigma**2 * (S_n**2) * dH_SS
        eta_n = (h_n - dH_dS) * self.sigma * S_n

        # Increments
        dL = b_n * self.dt + eta_n * self.dW

        # Cumulative process
        L = np.zeros((M, N))
        L[:, 1:] = np.cumsum(dL, axis=1)

        self.L = L

        return L
    
    
    
    def generate_adj(self, risk_type='ell', **kwargs) -> np.ndarray:
            """
            Backward solution of the BSDE for the adjoint process p_t^(k) and q_t^(k).

            Parameters
            ----------
            risk_type : str
                Type of risk measure:
                'ell'  -> Expected Loss Linear
                'ele'  -> Expected Loss Exponential
                'elw'  -> Expected Loss Weibull
                'entl' -> Entropic Linear
                'ente' -> Entropic Exponential
                'entw' -> Entropic Weibull
                'es'   -> Expected Shortfall
            **kwargs :
                Optional parameters depending on risk_type:
                    a     : float -> risk aversion coefficient (entropic cases)
                    alpha : float -> threshold (expected shortfall)
                    beta  : float -> confidence level (expected shortfall)

            Returns
            -------
            p : ndarray (M, N)
                Backward adjoint process p_t^(k,m).
            q : ndarray (M, N-1)
                Auxiliary process q_n^(k,m).
            """

            # Extract optional arguments
            a = kwargs.get('a', None)
            alpha = kwargs.get('alpha', None)
            beta = kwargs.get('beta', None)

            S = self.S_path
            L = self.L
            dW = self.dW
            dt = self.dt
            M, N = S.shape

            p = np.zeros((M, N))
            q = np.zeros((M, N-1))

            # === Terminal conditions and drivers ===
            if risk_type == 'ell':  # Expected Loss Linear
                terminal_grad = lambda L: np.zeros_like(L)
                f_func = lambda t, S, L, p: np.zeros_like(L)

            elif risk_type == 'ele':  # Expected Loss Exponential
                terminal_grad = lambda L: np.ones_like(L)
                f_func = lambda t, S, L, p: np.zeros_like(L)

            elif risk_type == 'elw':  # Expected Loss Weibull
                # Usa self.K que já está definido
                def terminal_grad(L):
                    pT = np.zeros_like(L)
                    mask = (L <= 0)
                    pT[mask] = -self.K * (-L[mask])**(self.K - 1) * np.exp(-(-L[mask])**self.K)
                    return pT
                f_func = lambda t, S, L, p: np.zeros_like(L)

            elif risk_type == 'entl':  # Entropic Risk Linear
                if a is None:
                    raise ValueError("Parameter 'a' must be provided for entropic risks.")
                terminal_grad = lambda L: a * np.exp(-a * L)
                f_func = lambda t, S, L, p: -a * p

            elif risk_type == 'ente':  # Entropic Risk Exponential
                if a is None:
                    raise ValueError("Parameter 'a' must be provided for entropic risks.")
                terminal_grad = lambda L: a * np.exp(-a * np.exp(-a * L))
                f_func = lambda t, S, L, p: -a * p

            elif risk_type == 'entw':  # Entropic Risk Weibull
                if a is None:
                    raise ValueError("Parameter 'a' must be provided for entropic risks.")
                terminal_grad = lambda L: np.exp(-np.minimum(L, 0)**self.K)
                f_func = lambda t, S, L, p: -a * p

            elif risk_type == 'es':  # Expected Shortfall
                if alpha is None or beta is None:
                    raise ValueError("Parameters 'alpha' and 'beta' must be provided for Expected Shortfall.")
                def terminal_grad(L):
                    pT = np.zeros_like(L)
                    pT[L > alpha] = 1 / (1 - beta)
                    return pT
                f_func = lambda t, S, L, p: np.zeros_like(L)

            else:
                raise ValueError(f"Unknown risk_type: {risk_type}")

            # === Step 1: Terminal condition ===
            p[:, -1] = terminal_grad(L[:, -1])

            # === Step 2: Backward iteration with regression ===
            for n in reversed(range(N-1)):
                t_next = (n+1) * dt
                p_next = p[:, n+1]
                f_val = f_func(t_next, S[:, n+1], L[:, n+1], p_next)
                Y = p_next + f_val * dt

                X = np.column_stack((np.ones(M), S[:, n], L[:, n]))
                coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
                p[:, n] = coeffs[0] + coeffs[1] * S[:, n] + coeffs[2] * L[:, n]

            # === Step 3: Estimate q_n ===
            for n in range(N-1):
                delta_p = p[:, n+1] - p[:, n]
                q[:, n] = np.mean(delta_p * dW[:, n]) / dt

            # set terminal values to 0
            q[:, -1] = 0.0

            self.p = p
            self.q = q
            return p, q
    

    
    def update_control(self, eps=1e-4, max_iter=50) -> np.ndarray:
        """
        Control Update and Convergence Criterion

        Parameters
        ----------
        eps : float
            Convergence tolerance for ||h^{(k+1)} - h^{(k)}||.
        max_iter : int
            Maximum number of iterations.
        r : ndarray (M, N), optional
            Auxiliary process r_n^(k,m). If None, it is set to zeros.

        Returns
        -------
        h_opt : ndarray (M, N)
            Optimal control h_t^* after convergence.
        """

        M, N = self.S_path.shape
        dH_dS = self.dH_dS
        S = self.S_path
        p = self.p
        q = self.q

        # === Initialize hedge with the classical delta ===
        h_old = np.copy(dH_dS)

        # === Phi: correction term based on adjoint variables ===
        def Phi(Sn, pn, qn):
            """
            Correction term Phi(S, p, q, r) based on adjoints.
            Generic implementation:
                Phi = (mu / (sigma^2 * S)) * p
            """
            return (self.mu / (self.sigma**2 * Sn)) * pn

        # === Fixed-point iteration for control update ===
        for k in range(max_iter):
            h_new = np.zeros_like(h_old)

            for n in range(N):
                for m in range(M):
                    Sn = S[m, n]
                    pn = p[m, n]
                    qn = q[m, n-1] if n > 0 else 0
                    h_new[m, n] = dH_dS[m, n] + Phi(Sn, pn, qn)

            # === Check convergence criterion ===
            diff = np.max(np.abs(h_new - h_old))
            if diff < eps:
                print(f"[OK] Convergence achieved in {k+1} iterations (diff={diff:.2e})")
                self.h_opt = h_new
                return h_new

            # Update for the next iteration
            h_old = h_new.copy()

        print("[Warning] Control update reached max_iter without full convergence.")
        self.h_opt = h_new
        return h_new


    def update_L(self) -> np.ndarray:
        """
        Forward simulation of the profit process L_t under the Heston model
        using the optimal hedge h_opt.

        The dynamics follow:
            dL_t = b_t^Heston dt + η1_t^Heston dW_t^S + η2_t^Heston dW_t^v,

        Parameters
        ----------
        None (uses self.h_opt and stored paths)

        Returns
        -------
        L_opt : ndarray (M, N)
            Simulated profit process under the optimal hedge.
        """
        # === Retrieve simulated quantities ===
        M, N = self.S_path.shape
        h_opt = self.h_opt
        S_n = self.S_path[:, :-1]
        v_n = self.v_path[:, :-1]
        dH_S = self.dH_dS[:, :-1]
        d2H_SS = self.d2H_dSS[:, :-1]
        dH_v = self.dH_dv[:, :-1]
        d2H_vv = self.d2H_dvv[:, :-1]
        d2H_Sv = self.d2H_dSv[:, :-1]
        dH_t = self.dH_dt[:, :-1]
        dW1 = self.dW1
        dW2 = self.dW2

        # === Drift term b_t^Heston ===
        b_n = (
            h_opt[:, :-1] * self.mu * S_n
            - (dH_t
            + self.mu * S_n * dH_S
            + self.kappa * (self.theta - v_n) * dH_v
            + 0.5 * v_n * S_n**2 * d2H_SS
            + 0.5 * (self.sigma_v**2) * v_n * d2H_vv
            + self.rho * self.sigma_v * v_n * S_n * d2H_Sv)
        )

        # === Diffusion terms ===
        eta1_n = (h_opt[:, :-1] - dH_S) * np.sqrt(v_n) * S_n
        eta2_n = (h_opt[:, :-1] - dH_v) * self.sigma_v * np.sqrt(v_n)

        # === Increments of L_t ===
        dL = b_n * self.dt + eta1_n * dW1 + eta2_n * dW2

        # === Build cumulative profit process ===
        L_opt = np.zeros((M, N))
        L_opt[:, 1:] = np.cumsum(dL, axis=1)

        # === Store and return ===
        self.L_opt = L_opt
        return L_opt
    





        

    def simulate_L(self, h):
        """
        Forward simulation of the P&L process L_t for a given hedge h.

        Parameters
        ----------
        h : ndarray, shape (N_time-1,) or (N_time,)
            Hedge position at each time step t_n.
            If length N_time, the last entry h[-1] is ignored.

        Sets
        ----
        self.L : ndarray, shape (M, N_time)
            Simulated P&L paths.
        """
        S = self.S_path              # (M, N_time)
        dW = self.dW                 # (M, N_time-1)
        H = self.H                   # (M, N_time)  (not directly needed here)
        dH_dS = self.dH_dS           # (M, N_time)
        d2H_dSS = self.d2H_dSS       # (M, N_time)
        dH_dt = self.dH_dt           # (M, N_time)

        M, N_time = S.shape
        dt = self.dt
        mu = self.mu
        sigma = self.sigma

        # ensure h has length N_time-1
        h = np.asarray(h, dtype=float)
        if h.shape[0] == N_time:
            h = h[:-1]
        assert h.shape[0] == N_time - 1, "h must have length N_time-1 or N_time"

        L = np.zeros((M, N_time))

        for n in range(N_time - 1):
            S_n = S[:, n]
            delta_n = dH_dS[:, n]
            gamma_n = d2H_dSS[:, n]
            dHt_n = dH_dt[:, n]
            h_n = h[n]

            # drift and diffusion of L according to your GBM formula
            b_n = (h_n - delta_n) * mu * S_n \
                - dHt_n \
                - 0.5 * sigma**2 * (S_n**2) * gamma_n

            eta_n = (h_n - delta_n) * sigma * S_n

            L[:, n+1] = L[:, n] + b_n * dt + eta_n * dW[:, n]

        self.h = h
        self.L = L
        return L
    

    def generate_adj(self, risk_type: str = 'ele', **kwargs) -> np.ndarray:
        """
        Backward solution of the BSDE for the adjoint processes p_t^(k) and q_t^(k).

        Parameters
        ----------
        risk_type : str
            Type of risk measure:
            'ele'  -> Expected Loss Exponential
            'elw'  -> Expected Loss Weibull
            'entl' -> Entropic Linear
            'ente' -> Entropic Exponential
            'entw' -> Entropic Weibull
            'es'   -> Expected Shortfall
        **kwargs :
            Optional parameters depending on risk_type:
                a       : float -> risk aversion coefficient (exponential utility)
                k       : float -> Weibull shape parameter
                gamma   : float -> entropic risk parameter
                beta    : float -> confidence level (expected shortfall, in (0,1))
                alpha   : float -> ES threshold (if not given, use empirical VaR_beta)
                max_log : float -> clipping threshold for exponentials (default 50.0)

        Returns
        -------
        p : ndarray (M, N_time)
            Backward adjoint process p_t^(k,m).
        q : ndarray (M, N_time-1)
            Auxiliary process q_n^(k,m).
        """
        S   = self.S_path              # (M, N_time)
        dW  = self.dW                  # (M, N_time-1)
        L   = self.L                   # (M, N_time)

        d2H_dSS    = self.d2H_dSS      # (M, N_time)
        d2H_dt_dS  = self.d2H_dt_dS    # (M, N_time)
        d3H_dS3    = self.d3H_dS3      # (M, N_time)

        mu    = self.mu
        sigma = self.sigma
        dt    = self.dt

        M, N_time = S.shape
        L_T = L[:, -1]                 # terminal P&L

        max_log = kwargs.get('max_log', 50.0)  # clipping to avoid overflow
        risk_type = risk_type.lower()

        # ------------------------------------------------------------------
        # 1) Terminal condition p_T = Upsilon(L_T) for each risk_type
        # ------------------------------------------------------------------
        if risk_type == 'ele':
            # Expected loss + exponential utility: ρ_u(X) = E[exp(-a X)]
            a = kwargs.get('a', 1.0)
            x = -a * L_T
            x_clipped = np.clip(x, -max_log, max_log)
            p_T = -a * np.exp(x_clipped)

        elif risk_type == 'elw':
            # Expected loss + Weibull utility: ρ_u(X) = E[exp((-min(X,0))^k)]
            k = kwargs.get('k', 2.0)
            neg_part = -np.minimum(L_T, 0.0)         # (-min(X,0)) >= 0
            g = neg_part**k
            g_clipped = np.clip(g, -max_log, max_log)
            e_g = np.exp(g_clipped)
            # Upsilon(X) = -k (-min(X,0))^{k-1} exp((-min(X,0))^k)
            p_T = np.zeros_like(L_T)
            mask = (L_T <= 0.0)
            p_T[mask] = -k * (neg_part[mask]**(k-1)) * e_g[mask]

        elif risk_type == 'entl':
            # Entropic + linear: ρ_u(X) = (1/γ) log E[exp(-γ X)]
            gamma = kwargs.get('gamma', 1.0)
            x = -gamma * L_T
            x_shift = x - x.max()              # log-sum-exp stabilization
            w = np.exp(x_shift)
            denom = np.mean(w)
            p_T = -w / denom                   # exp(x.max()) cancels

        elif risk_type == 'ente':
            # Entropic + exponential utility:
            # ρ_u(X) = (1/γ) log E[exp(γ exp(-a X))]
            a     = kwargs.get('a', 1.0)
            gamma = kwargs.get('gamma', 1.0)

            y = -a * L_T
            y_clipped = np.clip(y, -max_log, max_log)
            e_inner = np.exp(y_clipped)        # ~ exp(-a X)

            z = gamma * e_inner
            z_clipped = np.clip(z, -max_log, max_log)
            w = np.exp(z_clipped)

            denom = np.mean(w)
            # Upsilon(X) = -a exp(-a X) exp(γ exp(-a X)) / E[exp(γ exp(-a X))]
            p_T = -a * e_inner * w / denom

        elif risk_type == 'entw':
            # Entropic + Weibull utility:
            # ρ_u(X) = (1/γ) log E[exp(γ exp((-min(X,0))^k))]
            gamma = kwargs.get('gamma', 1.0)
            k     = kwargs.get('k', 2.0)

            neg_part = -np.minimum(L_T, 0.0)
            g = neg_part**k
            g_clipped = np.clip(g, -max_log, max_log)
            e_g = np.exp(g_clipped)

            z = gamma * e_g
            z_clipped = np.clip(z, -max_log, max_log)
            w = np.exp(z_clipped)

            denom = np.mean(w)

            g_prime = np.zeros_like(L_T)
            mask = (L_T <= 0.0)
            g_prime[mask] = -k * (neg_part[mask]**(k-1))

            # Upsilon(X) = e^g * g'(X) * exp(γ e^g) / E[exp(γ e^g)]
            p_T = e_g * g_prime * w / denom

        elif risk_type == 'es':
            # Expected shortfall with linear utility:
            # ρ_u(X) = ES_β(X), Upsilon(X) = 1/(1-β) 1_{X >= α*}
            beta = kwargs.get('beta', 0.975)
            alpha = kwargs.get('alpha', None)
            if alpha is None:
                alpha = np.quantile(L_T, beta)
            indicator = (L_T >= alpha).astype(float)
            p_T = indicator / (1.0 - beta)

        else:
            raise ValueError(f"Unknown risk_type '{risk_type}'")

        # optional: clip p_T itself to avoid insane values
        p_T = np.clip(p_T, -np.exp(max_log), np.exp(max_log))

        # ------------------------------------------------------------------
        # 2) Backward recursion for p and q using linear SDE structure
        #    dp_t = A_t p_t dt - (μ/σ) p_t dW_t,  q_t = -(μ/σ) p_t
        #    use multiplicative (log-Euler) scheme for stability.
        # ------------------------------------------------------------------
        p = np.zeros((M, N_time))
        q = np.zeros((M, N_time - 1))

        p[:, -1] = p_T

        b  = - mu / sigma
        b2 = b * b

        for n in reversed(range(N_time - 1)):
            S_n          = S[:, n]
            d2H_dt_dS_n  = d2H_dt_dS[:, n]
            d2H_dSS_n    = d2H_dSS[:, n]
            d3H_dS3_n    = d3H_dS3[:, n]

            # A_n = ∂²H/(∂t∂S) + 0.5 σ² (2 S_t ∂²H/∂S² + S_t² ∂³H/∂S³)
            A_n = d2H_dt_dS_n + 0.5 * sigma**2 * (
                2.0 * S_n * d2H_dSS_n + (S_n**2) * d3H_dS3_n
            )

            # backward multiplicative update for linear SDE
            # p_n = p_{n+1} * exp( - (A_n - 0.5 b^2) dt - b dW_n )
            expo = - (A_n - 0.5 * b2) * dt - b * dW[:, n]
            expo = np.clip(expo, -max_log, max_log)  # safety clip
            p[:, n] = p[:, n+1] * np.exp(expo)

            q[:, n] = - (mu / sigma) * p[:, n]

        self.p = p
        self.q = q
        return p, q



    def update_control(self, alpha: float):
        """
        Gradient step for h using pathwise derivative of L_T wrt h_n:

            dρ/dh_n ≈ E[ p_T * (μ S_n dt + σ S_n dW_n) ].

        Uses self.p[:, -1] as p_T = Upsilon(L_T).
        """
        S   = self.S_path
        dW  = self.dW
        L   = self.L
        U_T = self.p[:, -1]      # p_T = Υ(L_T)

        dt    = self.dt
        mu    = self.mu
        sigma = self.sigma

        M, N_time = S.shape
        h = self.h
        assert h.shape[0] == N_time - 1

        grad = np.zeros(N_time - 1)
        for n in range(N_time - 1):
            term_n = mu * S[:, n] * dt + sigma * S[:, n] * dW[:, n]
            grad[n] = np.mean(U_T * term_n)

        h_new = h - alpha * grad
        self.h = h_new
        return h_new



    def update_L(self) -> np.ndarray:
        """
        Forward simulation of the profit process L_t under the Heston model
        using the optimal hedge h_opt.

        The dynamics follow:
            dL_t = b_t^Heston dt + η1_t^Heston dW_t^S + η2_t^Heston dW_t^v,

        Parameters
        ----------
        None (uses self.h_opt and stored paths)

        Returns
        -------
        L_opt : ndarray (M, N)
            Simulated profit process under the optimal hedge.
        """
        # === Retrieve simulated quantities ===
        M, N = self.S_path.shape
        h_opt = self.h_opt
        S_n = self.S_path[:, :-1]
        v_n = self.v_path[:, :-1]
        dH_S = self.dH_dS[:, :-1]
        d2H_SS = self.d2H_dSS[:, :-1]
        dH_v = self.dH_dv[:, :-1]
        d2H_vv = self.d2H_dvv[:, :-1]
        d2H_Sv = self.d2H_dSv[:, :-1]
        dH_t = self.dH_dt[:, :-1]
        dW1 = self.dW1
        dW2 = self.dW2

        # === Drift term b_t^Heston ===
        b_n = (
            h_opt[:, :-1] * self.mu * S_n
            - (dH_t
            + self.mu * S_n * dH_S
            + self.kappa * (self.theta - v_n) * dH_v
            + 0.5 * v_n * S_n**2 * d2H_SS
            + 0.5 * (self.sigma_v**2) * v_n * d2H_vv
            + self.rho * self.sigma_v * v_n * S_n * d2H_Sv)
        )

        # === Diffusion terms ===
        eta1_n = (h_opt[:, :-1] - dH_S) * np.sqrt(v_n) * S_n
        eta2_n = (h_opt[:, :-1] - dH_v) * self.sigma_v * np.sqrt(v_n)

        # === Increments of L_t ===
        dL = b_n * self.dt + eta1_n * dW1 + eta2_n * dW2

        # === Build cumulative profit process ===
        L_opt = np.zeros((M, N))
        L_opt[:, 1:] = np.cumsum(dL, axis=1)

        # === Store and return ===
        self.L_opt = L_opt
        return L_opt



    def run_picard(self, risk_type='ele', alpha=1e-3, n_iter=20, h_init=None, **kwargs):
        """
        Run Picard / gradient iterations to approximate the optimal hedge h*.

        Parameters
        ----------
        risk_type : str
            Same codes as in generate_adj().
        alpha : float
            Step size for control update.
        n_iter : int
            Number of Picard iterations.
        h_init : ndarray or None
            Initial hedge (N_time-1,). If None, starts from zero.
        **kwargs :
            Passed to generate_adj (risk parameters).

        Returns
        -------
        h_opt : ndarray, shape (N_time-1,)
        """
        if not hasattr(self, "S_path"):
            self.simulate_S()
        if not hasattr(self, "H"):
            self.simulate_H()

        _, N_time = self.S_path.shape

        if h_init is None:
            h = np.zeros(N_time - 1)
        else:
            h = np.asarray(h_init, float)
            if h.shape[0] == N_time:
                h = h[:-1]
        self.h = h

        for k in range(n_iter):
            self.simulate_L(self.h)
            self.generate_adj(risk_type=risk_type, **kwargs)
            self.update_control(alpha=alpha, risk_type=risk_type, **kwargs)

        self.h_opt = self.h
        return self.h_opt

    def risk_functional(L_T, risk_type='ele', **kwargs):
        """
        Monte Carlo evaluation of ρ_u(L_T) for one of the risk types.

        Parameters
        ----------
        L_T : ndarray, shape (M,)
            Terminal P&L samples.
        risk_type : str
            Same codes as in generate_adj().
        **kwargs :
            Risk parameters (a, k, gamma, alpha, beta, ...).

        Returns
        -------
        rho : float
            Estimated risk measure ρ_u(L_T).
        """
        L_T = np.asarray(L_T, dtype=float)
        M = L_T.shape[0]
        rt = risk_type.lower()

        if rt == 'ele':
            a = kwargs.get('a', 1.0)
            return np.mean(np.exp(-a * L_T))

        elif rt == 'elw':
            k = kwargs.get('k', 2.0)
            neg_part = -np.minimum(L_T, 0.0)
            return np.mean(np.exp(neg_part**k))

        elif rt == 'entl':
            gamma = kwargs.get('gamma', 1.0)
            return (1.0 / gamma) * np.log(np.mean(np.exp(-gamma * L_T)))

        elif rt == 'ente':
            a     = kwargs.get('a', 1.0)
            gamma = kwargs.get('gamma', 1.0)
            e_inner = np.exp(-a * L_T)
            return (1.0 / gamma) * np.log(np.mean(np.exp(gamma * e_inner)))

        elif rt == 'entw':
            gamma = kwargs.get('gamma', 1.0)
            k     = kwargs.get('k', 2.0)
            neg_part = -np.minimum(L_T, 0.0)
            g = neg_part**k
            return (1.0 / gamma) * np.log(np.mean(np.exp(gamma * np.exp(g))))

        elif rt == 'es':
            beta = kwargs.get('beta', 0.975)
            alpha = kwargs.get('alpha', None)
            if alpha is None:
                alpha = np.quantile(L_T, beta)
            tail = L_T[L_T >= alpha]
            if tail.size == 0:
                return alpha
            return np.mean(tail)

        else:
            raise ValueError(f"Unknown risk_type '{risk_type}'")
