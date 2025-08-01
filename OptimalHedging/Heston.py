from OptimalHedging.Simulator import BaseSimulator
import numpy as np

class HestonSimulator(BaseSimulator):
    """
    Simulator for the Heston stochastic volatility model.
    """

    def __init__(self,
                 v0: float,
                 kappa: float,
                 theta: float,
                 sigma_v: float,
                 rho: float,
                 **kwargs):
        """
        Additional parameters for the Heston model:
        - v0      : initial variance
        - kappa   : rate of mean reversion of variance
        - theta   : long-term variance level
        - sigma_v : volatility of variance
        - rho     : correlation between dW1 and dW2
        """
        super().__init__(**kwargs)
        self.v0 = v0
        self.kappa = kappa
        self.theta = theta
        self.sigma_v = sigma_v
        self.rho = rho

    def simulate_S(self) -> np.ndarray:
        """
        Simulate Heston model using Euler–Maruyama with correlated Brownian motions:
            dS = mu * S * dt + sqrt(v) * S * dW1
            dv = kappa * (theta - v) * dt + sigma_v * sqrt(v) * dW2
            Corr(dW1, dW2) = rho 

        Returns
        -------
        S : ndarray, shape (M, N+1)
            Simulated asset price paths.
        v : ndarray, shape (M, N+1)
            Simulated variance paths.
        """

        # --- Generate correlated Brownian increments ---
        cov = np.array([[1.0, self.rho],
                        [self.rho, 1.0]])
        dW = np.random.multivariate_normal(mean=[0.0, 0.0], cov=cov, size=(self.M, self.N - 1))
        dW *= np.sqrt(self.dt)
        self.dW1 = dW[:, :, 0]
        self.dW2 = dW[:, :, 1]

        # --- Initialize paths ---
        S = np.zeros((self.M, self.N))
        v = np.zeros((self.M, self.N))
        S[:, 0] = self.S0
        v[:, 0] = self.v0

        # --- Alfonsi updates for v_t ---
        for n in range(1, self.N):
            v_prev = v[:, n-1]
            v[:, n] = (v_prev + self.kappa * self.theta * self.dt
                    + self.sigma_v * np.sqrt(v_prev) * self.dW2[:, n-1]
                    + 0.25 * self.sigma_v**2 * self.dt) / (1 + self.kappa * self.dt)

        # --- Euler additive for S_t ---
        increments = 1 + self.mu * self.dt + np.sqrt(v[:, :-1]) * self.dW1
        S[:, 1:] = self.S0 * np.cumprod(increments, axis=1)

        self.v_path = v
        self.S_path = S
        return self.S_path, self.v_path
    

    def simulate_H(self) -> tuple:
        """
        Backward estimation of H_t and its derivatives for the Heston model.

        Uses regression to estimate H(S, v, t) and its spatial derivatives.

        Parameters
        ----------
        K : float
            Strike price.

        Returns
        -------
        H : ndarray (M, N)
            Estimated H_t.
        dH_dS : ndarray (M, N)
            ∂H/∂S
        d2H_dSS : ndarray (M, N)
            ∂²H/∂S²
        dH_dt : ndarray (M, N)
            ∂H/∂t (estimated via Itô)
        dH_dv : ndarray (M, N)
            ∂H/∂v
        d2H_dvv : ndarray (M, N)
            ∂²H/∂v²
        d2H_dSv : ndarray (M, N)
            ∂²H/∂S∂v
        """

        S = self.S_path
        v = self.v_path
        K = self.K
        M, N = S.shape

        H = np.zeros((M, N))
        dH_dS = np.zeros((M, N))
        d2H_dSS = np.zeros((M, N))
        dH_dv = np.zeros((M, N))
        d2H_dvv = np.zeros((M, N))
        d2H_dSv = np.zeros((M, N))
        dH_dt = np.zeros((M, N))

        # Terminal payoff
        H[:, -1] = np.maximum(S[:, -1] - K, 0)

        # Backward induction
        for n in reversed(range(N - 1)):
            S_n = S[:, n]
            v_n = v[:, n]
            Y = H[:, n + 1]

            # Basis: 1, S, S², v, v², S*v
            X = np.column_stack((
                np.ones(M),
                S_n,
                S_n**2,
                v_n,
                v_n**2,
                S_n * v_n
            ))

            coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
            a0, a1, a2, a3, a4, a5 = coeffs

            # Fit H
            H_fit = (a0 + a1 * S_n + a2 * S_n**2
                        + a3 * v_n + a4 * v_n**2
                        + a5 * S_n * v_n)
            H[:, n] = np.maximum(H_fit, 0)

            # Derivatives
            dH_dS[:, n] = a1 + 2 * a2 * S_n + a5 * v_n
            d2H_dSS[:, n] = 2 * a2
            dH_dv[:, n] = a3 + 2 * a4 * v_n + a5 * S_n
            d2H_dvv[:, n] = 2 * a4
            d2H_dSv[:, n] = a5

            # Time derivative from full Itô formula (drift only)
            dH_dt[:, n] = (
                (H[:, n+1] - H[:, n]) / self.dt
                - self.mu * S_n * dH_dS[:, n]
                - self.kappa * (self.theta - v_n) * dH_dv[:, n]
                - 0.5 * v_n * S_n**2 * d2H_dSS[:, n]
                - 0.5 * self.sigma_v**2 * v_n * d2H_dvv[:, n]
                - self.rho * self.sigma_v * S_n * v_n * d2H_dSv[:, n]
            )

        # Save in object
        self.H = H
        self.dH_dS = dH_dS
        self.d2H_dSS = d2H_dSS
        self.dH_dv = dH_dv
        self.d2H_dvv = d2H_dvv
        self.d2H_dSv = d2H_dSv
        self.dH_dt = dH_dt

        return H, dH_dS, d2H_dSS, dH_dt, dH_dv, d2H_dvv, d2H_dSv


    def simulate_L(self, h0=0.5) -> np.ndarray:
        """
        Forward simulation of the profit process L_t under the Heston model.

        Parameters
        ----------
        h0 : float
            Initial guess for control.

        Returns
        -------
        L : ndarray (M, N)
            Simulated profit process.
        """
        M, N = self.S_path.shape

        # Extract stored quantities
        S_n = self.S_path[:, :-1]
        v_n = self.v_path[:, :-1]
        dH_S = self.dH_dS[:, :-1]
        d2H_SS = self.d2H_dSS[:, :-1]
        dH_v = self.dH_dv[:, :-1]
        d2H_vv = self.d2H_dvv[:, :-1]
        d2H_Sv = self.d2H_dSv[:, :-1]
        dH_t = self.dH_dt[:, :-1]

        # Initial control guess
        h_n = np.full_like(S_n, h0)

        # Brownian increments
        dW1 = self.dW1
        dW2 = self.dW2

        # === Drift term b_t^{Heston} ===
        b_n = (h_n * self.mu * S_n
            - (dH_t
                + self.mu * S_n * dH_S
                + self.kappa * (self.theta - v_n) * dH_v
                + 0.5 * v_n * S_n**2 * d2H_SS
                + 0.5 * self.sigma_v**2 * v_n * d2H_vv
                + self.rho * self.sigma_v * S_n * v_n * d2H_Sv))

        # === Diffusion terms eta_1 and eta_2 ===
        eta1_n = h_n * np.sqrt(v_n) * S_n - np.sqrt(v_n) * S_n * dH_S
        eta2_n = h_n * self.sigma_v * np.sqrt(v_n) - self.sigma_v * np.sqrt(v_n) * dH_v

        # === Increment dL ===
        dL = b_n * self.dt + eta1_n * dW1 + eta2_n * dW2

        # === Accumulate L ===
        L = np.zeros((M, N))
        L[:, 1:] = np.cumsum(dL, axis=1)

        self.L = L
        return L

    
    def generate_adj(self, risk_type='ell', **kwargs):
        """
        Backward solution of the BSDE for the adjoint process p_t^(k),
        including q_t^(k) and r_t^(k) for the Heston model.

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
            Auxiliary process q_n^(k,m) associated with dW^S.
        r : ndarray (M, N-1)
            Auxiliary process r_n^(k,m) associated with dW^v.
        """
        # === Load paths ===
        S = self.S_path
        L = self.L
        v = self.v_path
        dW1 = self.dW1
        dW2 = self.dW2
        dt = self.dt
        M, N = S.shape

        # === Initialize storage ===
        p = np.zeros((M, N))
        q = np.zeros((M, N-1))
        r = np.zeros((M, N-1))

        # === Define terminal gradient and driver f(t,S,v,L,p) ===
        if risk_type == 'ell':  # Expected Loss Linear
            terminal_grad = lambda L: np.zeros_like(L)
            f_func = lambda t, S, v, L, p: np.zeros_like(L)

        elif risk_type == 'ele':  # Expected Loss Exponential
            terminal_grad = lambda L: np.ones_like(L)
            f_func = lambda t, S, v, L, p: np.zeros_like(L)

        elif risk_type == 'elw':  # Expected Loss Weibull
            k = kwargs.get('k', 2.0)
            def terminal_grad(L):
                pT = np.zeros_like(L)
                mask = (L <= 0)
                pT[mask] = -k * (-L[mask])**(k-1) * np.exp(-(-L[mask])**k)
                return pT
            f_func = lambda t, S, v, L, p: np.zeros_like(L)

        elif risk_type == 'entl':  # Entropic Linear
            a = kwargs.get('a', 1.0)
            terminal_grad = lambda L: a * np.exp(-a * L)
            f_func = lambda t, S, v, L, p: -a * p

        elif risk_type == 'ente':  # Entropic Exponential
            a = kwargs.get('a', 1.0)
            terminal_grad = lambda L: a * np.exp(-a * np.exp(-a * L))
            f_func = lambda t, S, v, L, p: -a * p

        elif risk_type == 'entw':  # Entropic Weibull
            a = kwargs.get('a', 1.0)
            k = kwargs.get('k', 2.0)
            terminal_grad = lambda L: np.exp(-np.minimum(L, 0)**k)
            f_func = lambda t, S, v, L, p: -a * p

        elif risk_type == 'es':  # Expected Shortfall
            alpha = kwargs.get('alpha', 0.05)
            beta = kwargs.get('beta', 0.95)
            def terminal_grad(L):
                pT = np.zeros_like(L)
                pT[L > alpha] = 1 / (1 - beta)
                return pT
            f_func = lambda t, S, v, L, p: np.zeros_like(L)

        else:
            raise ValueError("Unknown risk_type")

        # === Step 1: Terminal condition ===
        p[:, -1] = terminal_grad(L[:, -1])

        # === Step 2: Backward iteration using regression ===
        for n in reversed(range(N-1)):
            t_next = (n+1) * dt
            S_next = S[:, n+1]
            v_next = v[:, n+1]
            L_next = L[:, n+1]
            p_next = p[:, n+1]

            f_val = f_func(t_next, S_next, v_next, L_next, p_next)
            Y = p_next + f_val * dt

            # Basis for Monte Carlo regression [1, S_n, v_n, L_n]
            X = np.column_stack((np.ones(M), S[:, n], v[:, n], L[:, n]))
            coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
            p[:, n] = coeffs[0] + coeffs[1] * S[:, n] + coeffs[2] * v[:, n] + coeffs[3] * L[:, n]

        # === Step 3: Estimate q_n and r_n via martingale representation ===
        for n in range(N-1):
            delta_p = p[:, n+1] - p[:, n]
            q[:, n] = np.mean(delta_p * dW1[:, n]) / dt  # noise from dW^S
            r[:, n] = np.mean(delta_p * dW2[:, n]) / dt  # noise from dW^v

        # set terminal values to 0
        q[:, -1] = 0.0
        r[:, -1] = 0.0

        # === Store results ===
        self.p = p
        self.q = q
        self.r = r
        return p, q, r

    
    def update_control(self, eps=1e-4, max_iter=50) -> np.ndarray:
        """
        Control Update and Convergence Criterion for the Heston model.

        Parameters
        ----------
        eps : float
            Convergence tolerance for ||h^{(k+1)} - h^{(k)}||.
        max_iter : int
            Maximum number of iterations.

        Returns
        -------
        h_opt : ndarray (M, N)
            Optimal control h_t^* after convergence.
        """

        M, N = self.S_path.shape
        dH_dS = self.dH_dS
        S = self.S_path
        v = self.v_path        # ⬅️ needed for Heston
        p = self.p
        q = self.q
        r = getattr(self, "r", np.zeros((M, N)))  # ⬅️ optional r term

        # === Initialize hedge with the classical delta ===
        h_old = np.copy(dH_dS)

        # === Phi: correction term for Heston based on adjoints ===
        def Phi(Sn, vn, pn, qn, rn):
            """
            Correction term Phi(S, v, p, q, r) based on adjoints for the Heston model.
            For Heston, optimality condition: 
                h* = ∂H/∂S + (μ / (v S)) * p
            """
            return (self.mu / (vn * Sn)) * pn   # ⬅️ modified denominator

        # === Fixed-point iteration for control update ===
        for k in range(max_iter):
            h_new = np.zeros_like(h_old)

            for n in range(N):
                for m in range(M):
                    Sn = S[m, n]
                    vn = v[m, n]
                    pn = p[m, n]
                    qn = q[m, n-1] if n > 0 else 0
                    rn = r[m, n-1] if n > 0 else 0
                    h_new[m, n] = dH_dS[m, n] + Phi(Sn, vn, pn, qn, rn)

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
