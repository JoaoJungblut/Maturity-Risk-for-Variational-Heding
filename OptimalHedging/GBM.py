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

        dW = np.random.normal(loc=0.0, 
                                   scale=np.sqrt(self.dt), 
                                   size=(self.M, self.N - 1))           # Generate Brownian increments (M x N-1) as N(0,1) 
        factors = 1 + self.mu * self.dt + self.sigma * dW               # Compute multiplicative factors for all steps
        factors = np.hstack((np.ones((self.M, 1)), factors))            # Insert a column of ones for the initial asset value
        S_path = self.S0 * np.cumprod(factors, axis=1)                  # Cumulative product along time to build paths

        self.S_path = S_path
        self.dW = dW

        return S_path
    
    
    def simulate_H(self) -> np.ndarray:
        """
        Backward estimation of H_t and its derivatives using a Longstaff–Schwartz style regression.

        This method uses the stored asset paths self.S_paths, which must be generated
        previously by calling simulate_S().

        Returns
        -------
        H : ndarray, shape (M, N)
            Backward estimated values of H_t.
        dH_dS : ndarray, shape (M, N)
            Estimated ∂H/∂S.
        d2H_dSS : ndarray, shape (M, N)
            Estimated ∂²H/∂S².
        dH_dt : ndarray, shape (M, N)
            Estimated ∂H/∂t.
        """
        S_path = self.S_path
        M, N = S_path.shape
        K = self.K

        # Initialize arrays
        H = np.zeros((M, N))
        dH_dS = np.zeros((M, N))
        d2H_dSS = np.zeros((M, N))
        dH_dt = np.zeros((M, N))

        # Terminal condition
        H[:, -1] = np.maximum(S_path[:, -1] - K, 0)

        # Backward iteration with polynomial regression
        for n in reversed(range(N - 1)):
            S_n = S_path[:, n]
            Y = H[:, n + 1]

            # Regression basis [1, S, S^2]
            X = np.column_stack((np.ones(M), S_n, S_n**2))

            # Solve least squares for coefficients
            coeffs, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
            a0, a1, a2 = coeffs

            # Fitted H_n from regression
            H_fit = a0 + a1 * S_n + a2 * (S_n**2)
            H[:, n] = np.maximum(H_fit, 0)

            # Derivatives from regression coefficients
            dH_dS[:, n] = a1 + 2 * a2 * S_n
            d2H_dSS[:, n] = 2 * a2

            # Time derivative using Itô's formula
            dH_dt[:, n] = ((H[:, n + 1] - H[:, n]) / (2*self.dt) # central difference
                        - self.mu * S_n * dH_dS[:, n]
                        - 0.5 * self.sigma**2 * S_n**2 * d2H_dSS[:, n])
        
            self.H = H
            self.dH_dS = dH_dS
            self.d2H_dSS = d2H_dSS
            self.dH_dt = dH_dt

        return H, dH_dS, d2H_dSS, dH_dt
    

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
