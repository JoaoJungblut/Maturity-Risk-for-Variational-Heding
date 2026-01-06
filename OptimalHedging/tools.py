import numpy as np
from numpy.polynomial.chebyshev import Chebyshev

def _time_to_index(N, t0, T, t_start):
    assert t0 <= t_start <= T
    dt = (T - t0) / N
    return int(np.floor((t_start - t0) / dt))


def _minmax_scale(u: np.ndarray, umin: float, umax: float) -> np.ndarray:
    """Affine map u in [umin, umax] -> z in [-1, 1]."""
    return 2.0 * (u - umin) / (umax - umin) - 1.0


def _cheb_eval_all(z: np.ndarray, deg: int, deriv: int = 0) -> np.ndarray:
    """
    Evaluate Chebyshev basis {T_0,...,T_deg} and its 'deriv'-th derivative w.r.t z.
    Returns array shape (len(z), deg+1).
    """
    out = np.empty((z.size, deg + 1), dtype=float)
    for k in range(deg + 1):
        out[:, k] = Chebyshev.basis(k).deriv(deriv)(z)
    return out


def _ridge_solve(Phi: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Solve ridge: (Phi^T Phi + lam I) beta = Phi^T y."""
    P = Phi.shape[1]
    A = Phi.T @ Phi
    A.flat[:: P + 1] += lam  # add lam to diagonal
    b = Phi.T @ y
    return np.linalg.solve(A, b)


def _solve_smoothed(Phi: np.ndarray,
                    y: np.ndarray,
                    beta_next: np.ndarray,
                    lam_ridge: float,
                    lam_time: float) -> np.ndarray:
    """
    Solve smoothed ridge:
        min ||Phi beta - y||^2
            + lam_ridge ||beta||^2
            + lam_time  ||beta - beta_next||^2
    """
    P = Phi.shape[1]

    A = Phi.T @ Phi
    A.flat[:: P + 1] += (lam_ridge + lam_time)

    b = Phi.T @ y + lam_time * beta_next

    return np.linalg.solve(A, b)

