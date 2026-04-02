"""
Parameter identification for the CPSS model.

Two-stage pipeline:
  1. **fit_cpss_params** — Least-squares on finite-difference derivatives
     for states x1 … x7.  Physical / cyber γ values are hyper-parameters.
  2. **fit_cost_weights** — Random search over cost weights evaluated by a
     closed-loop actor–critic score (fear reduction + infrastructure).

All results are reproducible (fixed seeds) and logged.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares  # type: ignore

from cpss_model import (
    CPSSParams, STATE_DIM,
    IX1, IX2, IX3, IX4, IX5, IX6, IX7, IX8, IX9, IX10,
    psi, psi_c, softmax_weights,
    gamma_mixing,
    compute_PS, compute_Cplus,
    compute_x1hat, compute_x2hat, compute_x3hat,
    compute_eq4, compute_eq5, compute_eq6, compute_eq7,
    drift, compute_d_traj,
)
from utils import finite_differences, save_json

logger = logging.getLogger("cpss_experiment")

# ═══════════════════════════════════════════════════════════════════════
#  Pack / unpack parameter vector  (for scipy optimiser)
# ═══════════════════════════════════════════════════════════════════════
# Fitted parameters (35 total):
#   [0]    kappa
#   [1]    zeta
#   [2]    alpha1
#   [3]    eta1
#   [4:9]  iota1 (5)
#   [9]    alpha2
#   [10]   eta2
#   [11:15] iota2 (4)
#   [15]   alpha3
#   [16]   eta3
#   [17:20] iota3 (3)
#   [20]   alpha4
#   [21:24] w4 (3)
#   [24]   alpha5
#   [25:28] w5 (3)
#   [28]   alpha6
#   [29:33] w6 (4)
#   [33]   alpha7
#   [34:37] w7 (3)
# Total: 37

_N_FIT = 37


def _pack(p: CPSSParams) -> np.ndarray:
    """Flatten fitted parameters into a vector of length 37."""
    return np.concatenate([
        [p.kappa, p.zeta],
        [p.alpha1, p.eta1], p.iota1,
        [p.alpha2, p.eta2], p.iota2,
        [p.alpha3, p.eta3], p.iota3,
        [p.alpha4], p.w4,
        [p.alpha5], p.w5,
        [p.alpha6], p.w6,
        [p.alpha7], p.w7,
    ])


def _unpack(theta: np.ndarray, base: CPSSParams) -> CPSSParams:
    """Write a flat vector back into a CPSSParams (non-fitted fields unchanged)."""
    p = CPSSParams()
    # Copy non-fitted fields from base
    p.gamma8 = base.gamma8
    p.gamma9 = base.gamma9
    p.gamma10 = base.gamma10
    p.beta1 = base.beta1
    p.beta8 = base.beta8
    p.beta9 = base.beta9
    p.beta10 = base.beta10
    p.ubar = base.ubar.copy()
    p.n_substeps = base.n_substeps

    idx = 0
    p.kappa = float(theta[idx]); idx += 1
    p.zeta = float(theta[idx]); idx += 1

    p.alpha1 = float(theta[idx]); idx += 1
    p.eta1 = float(theta[idx]); idx += 1
    p.iota1 = theta[idx: idx + 5].copy(); idx += 5

    p.alpha2 = float(theta[idx]); idx += 1
    p.eta2 = float(theta[idx]); idx += 1
    p.iota2 = theta[idx: idx + 4].copy(); idx += 4

    p.alpha3 = float(theta[idx]); idx += 1
    p.eta3 = float(theta[idx]); idx += 1
    p.iota3 = theta[idx: idx + 3].copy(); idx += 3

    p.alpha4 = float(theta[idx]); idx += 1
    p.w4 = theta[idx: idx + 3].copy(); idx += 3

    p.alpha5 = float(theta[idx]); idx += 1
    p.w5 = theta[idx: idx + 3].copy(); idx += 3

    p.alpha6 = float(theta[idx]); idx += 1
    p.w6 = theta[idx: idx + 4].copy(); idx += 4

    p.alpha7 = float(theta[idx]); idx += 1
    p.w7 = theta[idx: idx + 3].copy(); idx += 3

    assert idx == _N_FIT
    return p


# ═══════════════════════════════════════════════════════════════════════
#  Residual function for least-squares
# ═══════════════════════════════════════════════════════════════════════

def _residuals(
    theta: np.ndarray,
    x_data: np.ndarray,
    dxdt: np.ndarray,
    base: CPSSParams,
    reg_lambda: float,
    theta0: np.ndarray,
) -> np.ndarray:
    """Vector of residuals  r = [ f_i(x[k]; θ) − ẋ_i[k] ]  for states 0…6.

    Plus Tikhonov regularisation  √λ · (θ − θ₀).
    """
    params = _unpack(theta, base)
    K_minus1 = dxdt.shape[0]  # 17

    residuals = []
    for k in range(K_minus1):
        x_k = x_data[k]
        # Use x_data as d for drift (x8-x10 terms not in objective but
        # drift still reads d[7:9] for those states — pass x_data row).
        f_k = drift(x_k, x_data[k], params)
        for i in range(7):  # states x1 … x7 only
            residuals.append(f_k[i] - dxdt[k, i])

    # Tikhonov regularisation
    reg = np.sqrt(reg_lambda) * (theta - theta0)
    return np.concatenate([np.array(residuals), reg])


# ═══════════════════════════════════════════════════════════════════════
#  Stage 1: fit CPSS parameters
# ═══════════════════════════════════════════════════════════════════════

def fit_cpss_params(
    x_data: np.ndarray,
    dt: float,
    base: Optional[CPSSParams] = None,
    reg_lambda: float = 0.01,
    seed: int = 0,
    verbose: bool = True,
) -> CPSSParams:
    """Fit unknown model parameters to the observed trajectory.

    Uses ``scipy.optimize.least_squares`` on finite-difference derivatives
    of states x1 … x7.  Physical/cyber γ values are fixed hyper-parameters.

    Parameters
    ----------
    x_data     : (18, 10) observed data.
    dt         : time-step (1.0).
    base       : starting CPSSParams (defaults used if None).
    reg_lambda : Tikhonov regularisation weight.
    seed       : for reproducibility (jitters initial guess).
    verbose    : log progress.

    Returns
    -------
    CPSSParams with fitted values.
    """
    np.random.seed(seed)
    if base is None:
        base = CPSSParams()

    dxdt = finite_differences(x_data, dt)

    theta0 = _pack(base)
    # Small random jitter for robustness
    theta_init = theta0 + 0.05 * np.random.randn(_N_FIT)

    # ── bounds ───────────────────────────────────────────────────────
    lb = np.full(_N_FIT, -5.0)
    ub = np.full(_N_FIT, 5.0)
    # kappa ∈ [1, 20]
    lb[0] = 1.0;  ub[0] = 20.0
    # zeta ∈ [0.1, 0.9]
    lb[1] = 0.1;  ub[1] = 0.9
    # alpha's ∈ [0.05, 10]
    alpha_idxs = [2, 9, 15, 20, 24, 28, 33]
    for ai in alpha_idxs:
        lb[ai] = 0.05; ub[ai] = 10.0
    # eta's ∈ [0.01, 0.99]
    eta_idxs = [3, 10, 16]
    for ei in eta_idxs:
        lb[ei] = 0.01; ub[ei] = 0.99

    # Clip init to bounds
    theta_init = np.clip(theta_init, lb + 1e-6, ub - 1e-6)

    if verbose:
        logger.info("Fitting CPSS params (%d parameters, %d residuals) …",
                     _N_FIT, dxdt.shape[0] * 7 + _N_FIT)

    result = least_squares(
        _residuals,
        theta_init,
        args=(x_data, dxdt, base, reg_lambda, theta0),
        bounds=(lb, ub),
        method="trf",
        loss="soft_l1",      # robust to outliers in noisy data
        max_nfev=5000,
        verbose=0,
    )

    fitted = _unpack(result.x, base)

    if verbose:
        logger.info("  optimality = %.4e,  cost = %.4e,  nfev = %d",
                     result.optimality, result.cost, result.nfev)

    return fitted


# ═══════════════════════════════════════════════════════════════════════
#  Stage 2: fit / search cost weights
# ═══════════════════════════════════════════════════════════════════════

def _score_controlled(
    cost_kw: Dict[str, float],
    x_data: np.ndarray,
    dt: float,
    cpss_params: CPSSParams,
) -> float:
    """Run a short actor–critic episode and return a scalar score.

    Lower is better.
    Score = mean(x1) + 0.5·max(x1)
          + 2·mean(1−x8) + 2·mean(1−x9) + mean(1−x4)
          + 0.1·mean(Σ u²)
    """
    # Lazy imports to avoid circular at module level
    from actor_critic_game import (
        CostConfig, LearningConfig, run_actor_critic,
    )

    cc = CostConfig(**cost_kw)
    lc = LearningConfig()
    d_traj = compute_d_traj(x_data, dt, cpss_params)
    T = x_data.shape[0] - 1

    res = run_actor_critic(
        x0=x_data[0],
        d_traj=d_traj,
        T_steps=T,
        cpss_params=cpss_params,
        cost_config=cc,
        learn_config=lc,
        dt=dt,
        seed=42,
    )

    x1 = res.x_traj[:, IX1]
    x4 = res.x_traj[:, IX4]
    x8 = res.x_traj[:, IX8]
    x9 = res.x_traj[:, IX9]
    u2 = np.sum(res.u_traj ** 2, axis=1)

    score = (
        np.mean(x1) + 0.5 * np.max(x1)
        + 2.0 * np.mean(1.0 - x8)
        + 2.0 * np.mean(1.0 - x9)
        + np.mean(1.0 - x4)
        + 0.1 * np.mean(u2)
    )
    return float(score)


def fit_cost_weights(
    x_data: np.ndarray,
    dt: float,
    cpss_params: CPSSParams,
    n_trials: int = 40,
    seed: int = 123,
    verbose: bool = True,
) -> Dict[str, float]:
    """Random search around default cost weights.

    Returns the best ``CostConfig`` keyword dict found.
    """
    np.random.seed(seed)

    # Default cost-weight ranges (log-uniform)
    keys = ["q11", "q1_10", "q21", "q2_8", "q31", "q3_4", "q3_9"]
    lo, hi = 0.1, 5.0
    R_lo, R_hi = 0.3, 3.0

    best_score = float("inf")
    best_kw: Dict[str, float] = {}

    # Evaluate the default first
    default_kw: Dict[str, Any] = {
        "q11": 1.0, "q1_10": 0.5, "q21": 0.5,
        "q2_8": 1.0, "q31": 0.5, "q3_4": 0.5, "q3_9": 1.0,
    }
    default_score = _score_controlled(default_kw, x_data, dt, cpss_params)
    best_score = default_score
    best_kw = dict(default_kw)
    if verbose:
        logger.info("  default cost score = %.4f", default_score)

    for trial in range(n_trials):
        kw: Dict[str, Any] = {}
        for k in keys:
            kw[k] = float(np.exp(np.random.uniform(np.log(lo), np.log(hi))))
        # R diagonal — search own-control penalties
        R_vals = [float(np.exp(np.random.uniform(np.log(R_lo), np.log(R_hi))))
                  for _ in range(3)]
        kw["R"] = np.diag(R_vals)

        try:
            sc = _score_controlled(kw, x_data, dt, cpss_params)
        except Exception:
            continue

        if sc < best_score:
            best_score = sc
            best_kw = kw
            if verbose:
                logger.info("  trial %d/%d  score = %.4f (new best)",
                             trial + 1, n_trials, sc)

    if verbose:
        logger.info("  best cost score = %.4f", best_score)

    # Convert R matrix to list for JSON serialisation
    if "R" in best_kw and isinstance(best_kw["R"], np.ndarray):
        best_kw["R"] = best_kw["R"].tolist()

    return best_kw


