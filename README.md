# Maturity Risk for Variational Hedge — Code

## Files

### Simulator.py
Defines `BaseSimulator` (abstract base class).

Contains:
- model/time-grid parameters (`S0, mu, sigma, K, t0, T, N, M, dt`)
- abstract interfaces:
  - `simulate_S`
  - `simulate_H` (returns price and derivatives)
  - `init_control`
  - `forward_PL`
  - `backward_adjoint`
  - `compute_gradient`
  - `optimize_hedge`
- static methods:
  - `risk_function(LT, risk_type, **kwargs)` for: `ele`, `elw`, `entl`, `ente`, `entw`, `esl`
  - `terminal_adjoint(LT, risk_type, **kwargs)` for the same set of risk types
  - `update_control(h, G, alpha, eps)` (normalized gradient step)
- declares `compute_MR(...)` in the base class docstring/signature, but leaves it unimplemented.

### tools.py
Utility functions used by the simulators:
- `_time_to_index` (maps continuous start time to grid index)
- `_minmax_scale` (maps to `[-1, 1]`)
- `_cheb_eval_all` (Chebyshev basis and derivatives)
- `_ridge_solve` (ridge regression solve)
- `_solve_smoothed` (ridge + temporal smoothing toward next-step coefficients)

### GBM.py
Defines `GBMSimulator(BaseSimulator)`.

Implements:
- `simulate_S` (GBM Euler-type multiplicative scheme; stores `S_GBM`, `dW_GBM`)
- `simulate_H` (single-pass LSMC via Chebyshev tensor basis in `(t, x=log S)`; stores `H_GBM` and derivatives)
- `init_control` (`Delta` or `zero`)
- `forward_PL` (P&L increments `dL = h dS - dH`, supports `t_start`)
- `backward_adjoint` (regression-based conditional expectation with Chebyshev basis + smoothing)
- `compute_gradient` (uses `mu`, `sigma`, `S`, `p`, `q`)
- `optimize_hedge` (iterative update with rollback/backtracking logic based on gradient norm)
- `compute_MR` (runs two optimizations and returns `MR = rho_t - rho_T` with `h_T, h_t, rho_T, rho_t`)

### Heston.py
Defines `HestonSimulator(BaseSimulator)`.

Implements:
- `simulate_S` (two correlated Brownian drivers; stores `S_Heston`, `v_Heston`, `dW1_Heston`, `dW2_Heston`)
- `simulate_H` (single-pass LSMC via Chebyshev tensor basis in `(t, x=log S, v)`; stores `H_Heston` and derivatives)
- `init_control` supports `Delta`, `MinVar`, `zero`
- `forward_PL`
- `backward_adjoint` (two martingale terms `q1`, `q2`, with smoothed regressions)
- `compute_gradient` (uses `p, q1, q2`, plus model parameters)
- `optimize_hedge` (same rollback/backtracking structure)
- `compute_MR` (same definition `MR = rho_t - rho_T`)

### JumpDiff.py
Defines `JumpDiffusionSimulator(BaseSimulator)`.

Implements:
- `simulate_S` (Brownian + Poisson jumps; stores `S_Jump`, `dW_Jump`, `dN_Jump`, `J_Jump`)
- `simulate_H` (LSMC in `(t, x=log S)`; also evaluates `H_jump` and `dH_dS_jump` at post-jump state)
- `init_control` supports `Delta`, `MinVar`, `zero`
- `forward_PL`
- `backward_adjoint` (includes jump martingale component `r`)
- `compute_gradient` (uses `p, q, r` and jump term)
- `optimize_hedge` (same rollback/backtracking structure)
- `compute_MR` (same definition `MR = rho_t - rho_T`)
