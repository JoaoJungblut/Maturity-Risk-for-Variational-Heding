import numpy as np
from numpy.polynomial.chebyshev import Chebyshev


def _time_to_index(N, t0, T, t_start):
    """
    Convert a continuous start time t_start into the corresponding discrete
    time index on a uniform grid.

    The time grid is assumed to be uniform with N time steps over [t0, T].

    Parameters
    ----------
    N : int
        Number of time intervals.
    t0 : float
        Initial time.
    T : float
        Terminal time.
    t_start : float
        Continuous time to be mapped to a grid index.

    Returns
    -------
    idx : int
        Time index such that t_idx <= t_start < t_{idx+1}.
    """
    assert t0 <= t_start <= T
    dt = (T - t0) / N
    return int(np.floor((t_start - t0) / dt))


def _minmax_scale(u: np.ndarray, umin: float, umax: float) -> np.ndarray:
    """
    Affine scaling of a variable u from the interval [umin, umax] to [-1, 1].

    This scaling is typically used to map state variables to the canonical
    domain of Chebyshev polynomials.

    Parameters
    ----------
    u : ndarray
        Input array.
    umin : float
        Lower bound of u.
    umax : float
        Upper bound of u.

    Returns
    -------
    z : ndarray
        Scaled array with values in [-1, 1].
    """
    return 2.0 * (u - umin) / (umax - umin) - 1.0


def _cheb_eval_all(z: np.ndarray, deg: int, deriv: int = 0) -> np.ndarray:
    """
    Evaluate the Chebyshev polynomial basis and its derivatives.

    This function evaluates the Chebyshev polynomials
        {T_0(z), T_1(z), ..., T_deg(z)}
    or their 'deriv'-th derivatives with respect to z.

    Parameters
    ----------
    z : ndarray
        Evaluation points in [-1, 1].
    deg : int
        Maximum polynomial degree.
    deriv : int, default=0
        Order of the derivative with respect to z.

    Returns
    -------
    Phi : ndarray, shape (len(z), deg+1)
        Matrix containing the evaluated Chebyshev basis (or derivatives),
        where column k corresponds to T_k^{(deriv)}(z).
    """
    out = np.empty((z.size, deg + 1), dtype=float)
    for k in range(deg + 1):
        out[:, k] = Chebyshev.basis(k).deriv(deriv)(z)
    return out


def _ridge_solve(Phi: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """
    Solve a ridge-regularized least squares problem.

    The solution beta satisfies:
        (Phi^T Phi + lam I) beta = Phi^T y

    Parameters
    ----------
    Phi : ndarray, shape (N, P)
        Design matrix.
    y : ndarray, shape (N,)
        Target vector.
    lam : float
        Ridge regularization parameter.

    Returns
    -------
    beta : ndarray, shape (P,)
        Ridge regression coefficients.
    """
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
    Solve a temporally smoothed ridge regression problem.

    The regression coefficients are obtained by minimizing:
        ||Phi beta - y||^2
        + lam_ridge ||beta||^2
        + lam_time  ||beta - beta_next||^2

    The last term enforces temporal smoothness by penalizing deviations from
    the coefficients at the next time step.

    Parameters
    ----------
    Phi : ndarray, shape (N, P)
        Design matrix.
    y : ndarray, shape (N,)
        Target vector.
    beta_next : ndarray, shape (P,)
        Regression coefficients at the next time step.
    lam_ridge : float
        Ridge regularization parameter.
    lam_time : float
        Temporal smoothing parameter.

    Returns
    -------
    beta : ndarray, shape (P,)
        Smoothed regression coefficients.
    """
    P = Phi.shape[1]

    A = Phi.T @ Phi
    A.flat[:: P + 1] += (lam_ridge + lam_time)

    b = Phi.T @ y + lam_time * beta_next

    return np.linalg.solve(A, b)
