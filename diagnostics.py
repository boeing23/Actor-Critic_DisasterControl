#!/usr/bin/env python3
"""
Post-hoc diagnostics for the actor–critic NZS game solution.

Computes:
  1. Near-Nash / exploitability test  (unilateral deviation)
  2. Persistence-of-excitation (PE) proxy eigenvalues
  3. Control-saturation and state-projection activation frequency
  4. Sensitivity check: alternative ODE for x8/x9/x10
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple

from features import phi_quadratic, grad_phi_quadratic, quadratic_basis_dim
from cpss_model import (
    CPSSParams, STATE_DIM, NUM_PLAYERS,
    drift, g_vectors, compute_PS, compute_Cplus, compute_d_traj,
    IX1, IX4, IX8, IX9, IX10,
)
from actor_critic_game import (
    CostConfig, LearningConfig, SimulationResult,
    run_actor_critic, simulate_baseline,
    compute_Q, compute_policy,
)
from utils import saturate, project_controlled_states, probing_signal


# ═══════════════════════════════════════════════════════════════════════
#  1.  Near-Nash / Exploitability Test
# ═══════════════════════════════════════════════════════════════════════

def _rollout_cost(
    player: int,
    x0: np.ndarray,
    d_traj: np.ndarray,
    T_steps: int,
    Wa_all: List[np.ndarray],
    cpss_params: CPSSParams,
    cost_config: CostConfig,
    dt: float = 1.0,
    discount: float = 1.0,
) -> float:
    """Rollout cost J_i for one player under fixed policies for all players.

    Policies are deterministic: u_i(x) = -1/(2R_ii) g_i^T grad_phi W_{a,i},
    saturated to [0, ubar_i].
    """
    g_vecs = g_vectors(cpss_params)
    n_sub = cpss_params.n_substeps
    dt_sub = dt / n_sub
    n = STATE_DIM

    x = x0.copy()
    total_cost = 0.0
    gamma_t = 1.0

    for k in range(T_steps):
        d_idx = min(k, d_traj.shape[0] - 1)
        d_k = d_traj[d_idx]

        # Compute all policies
        u_exec = np.zeros(NUM_PLAYERS)
        for j in range(NUM_PLAYERS):
            R_jj = float(cost_config.R[j, j])
            raw = compute_policy(x, Wa_all[j], g_vecs[j], R_jj)
            u_exec[j] = saturate(raw, 0.0, float(cpss_params.ubar[j]))

        # Stage cost for player i
        Q_i = compute_Q(player, x, cost_config)
        u_pen = sum(cost_config.R[player, j] * u_exec[j] ** 2
                     for j in range(NUM_PLAYERS))
        total_cost += gamma_t * (Q_i + u_pen) * dt
        gamma_t *= discount

        # Propagate state
        x_sub = x.copy()
        for _s in range(n_sub):
            f_sub = drift(x_sub, d_k, cpss_params)
            dyn = f_sub.copy()
            for j in range(NUM_PLAYERS):
                dyn += g_vecs[j] * u_exec[j]
            x_sub = x_sub + dyn * dt_sub
        x = project_controlled_states(x_sub)

    return total_cost


def compute_nash_gap(
    x0: np.ndarray,
    d_traj: np.ndarray,
    T_steps: int,
    Wa_converged: List[np.ndarray],
    cpss_params: CPSSParams,
    cost_config: CostConfig,
    dt: float = 1.0,
    n_deviations: int = 50,
    deviation_scales: Tuple[float, ...] = (0.05, 0.10, 0.20),
    seed: int = 99,
) -> Dict[str, float]:
    """Unilateral deviation test for near-Nash verification.

    For each player i and each perturbation scale ε:
      1. Compute J_i under the converged joint policy (baseline cost).
      2. Generate n_deviations random perturbations of scale ε·‖W_{a,i}‖
         to W_{a,i} only, keeping other players' weights fixed.
      3. Report best improvement found (if any).

    The 'exploitability' is max_i (max improvement_i) at the smallest scale.
    Near-Nash means exploitability ≈ 0.

    Returns dict with per-player baseline costs, best deviating costs,
    improvement percentages, and overall exploitability.
    """
    rng = np.random.RandomState(seed)
    p = Wa_converged[0].shape[0]

    results: Dict[str, float] = {}

    for i in range(NUM_PLAYERS):
        # Baseline cost for player i
        J_base = _rollout_cost(
            i, x0, d_traj, T_steps, Wa_converged,
            cpss_params, cost_config, dt,
        )
        results[f"J{i+1}_baseline"] = J_base

        w_norm = max(np.linalg.norm(Wa_converged[i]), 1e-8)

        for scale in deviation_scales:
            best_J = J_base
            for _ in range(n_deviations):
                Wa_perturbed = [w.copy() for w in Wa_converged]
                perturbation = rng.randn(p) * scale * w_norm
                Wa_perturbed[i] = Wa_converged[i] + perturbation

                J_dev = _rollout_cost(
                    i, x0, d_traj, T_steps, Wa_perturbed,
                    cpss_params, cost_config, dt,
                )
                if J_dev < best_J:
                    best_J = J_dev

            improvement = max(0.0, (J_base - best_J) / max(abs(J_base), 1e-8) * 100)
            scale_tag = f"{int(scale*100)}pct"
            results[f"player{i+1}_improv_{scale_tag}"] = improvement

    # Overall exploitability at each scale
    for scale in deviation_scales:
        tag = f"{int(scale*100)}pct"
        vals = [results.get(f"player{i+1}_improv_{tag}", 0.0) for i in range(NUM_PLAYERS)]
        results[f"exploitability_{tag}"] = max(vals)

    return results


# ═══════════════════════════════════════════════════════════════════════
#  2.  PE Proxy: minimum eigenvalue of running covariance
# ═══════════════════════════════════════════════════════════════════════

def compute_pe_diagnostics(
    ctrl_result: SimulationResult,
    d_traj: np.ndarray,
    cpss_params: CPSSParams,
    cost_config: CostConfig,
    learn_config: LearningConfig,
    dt: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Compute PE proxy: minimum eigenvalue of the running covariance
    of the normalised regressor sigma_i / (sigma_i^T sigma_i + 1).

    Returns
    -------
    dict with keys:
        'min_eig_running'  : (T, 3) min eigenvalue of cumulative covariance
        'min_eig_window'   : (T, 3) min eigenvalue over sliding window
    """
    x_traj = ctrl_result.x_traj
    u_traj = ctrl_result.u_traj
    T = u_traj.shape[0]
    n = STATE_DIM
    p = quadratic_basis_dim(n)
    g_vecs = g_vectors(cpss_params)

    min_eig_running = np.zeros((T, NUM_PLAYERS))
    min_eig_window = np.zeros((T, NUM_PLAYERS))

    # We compute a single sigma per timestep (shared across players in this
    # implementation since sigma depends on the common state and executed controls)
    # but PE condition is per-player, so we track per-player regressors.
    # Actually in Vamvoudakis-Lewis, sigma_i uses the same dynamics but
    # with the player's own W_c in the Bellman residual. The regressor
    # sigma itself is shared. We track it as a single quantity.

    window_len = max(learn_config.window_length, 4)
    cov_running = np.zeros((p, p))

    sigma_normd_history = np.zeros((T, p))

    for k in range(T):
        x_k = x_traj[k]
        u_exec = u_traj[k]

        d_idx = min(k, d_traj.shape[0] - 1)
        d_k = d_traj[d_idx]

        f_val = drift(x_k, d_k, cpss_params)
        dynamics = f_val.copy()
        for j in range(NUM_PLAYERS):
            dynamics += g_vecs[j] * u_exec[j]

        gp = grad_phi_quadratic(x_k)   # (n, p)
        sigma = gp.T @ dynamics          # (p,)

        # Normalised regressor
        norm_sq = float(sigma @ sigma)
        sigma_normd = sigma / (norm_sq + 1.0)
        sigma_normd_history[k] = sigma_normd

        # Running covariance
        outer = np.outer(sigma_normd, sigma_normd)
        cov_running += outer

        # Min eigenvalue of running average covariance
        cov_avg = cov_running / (k + 1)
        eig_min = np.real(np.linalg.eigvalsh(cov_avg)[0])

        for i in range(NUM_PLAYERS):
            min_eig_running[k, i] = eig_min

        # Windowed covariance
        if k >= window_len - 1:
            cov_win = np.zeros((p, p))
            for kk in range(k - window_len + 1, k + 1):
                s = sigma_normd_history[kk]
                cov_win += np.outer(s, s)
            cov_win /= window_len
            eig_win = np.real(np.linalg.eigvalsh(cov_win)[0])
        else:
            eig_win = eig_min

        for i in range(NUM_PLAYERS):
            min_eig_window[k, i] = eig_win

    # Also compute rank / effective dimension at each step
    rank_running = np.zeros(T)
    for k in range(T):
        # Recompute running covariance for rank
        if k == 0:
            cov_k = np.outer(sigma_normd_history[0], sigma_normd_history[0])
        else:
            cov_k = np.zeros((p, p))
            for kk in range(k + 1):
                s = sigma_normd_history[kk]
                cov_k += np.outer(s, s)
            cov_k /= (k + 1)
        eigs = np.real(np.linalg.eigvalsh(cov_k))
        rank_running[k] = np.sum(eigs > 1e-10)

    return {
        "min_eig_running": min_eig_running,
        "min_eig_window": min_eig_window,
        "rank_running": rank_running,
        "basis_dim": p,
    }


# ═══════════════════════════════════════════════════════════════════════
#  3.  Saturation & Projection Activation Frequency
# ═══════════════════════════════════════════════════════════════════════

def compute_saturation_stats(
    ctrl_result: SimulationResult,
    cpss_params: CPSSParams,
    tol: float = 1e-6,
) -> Dict[str, float]:
    """Compute activation frequency of control saturation and state projection.

    Returns dict with:
        sat_pct_player{i}  : % timesteps u_i hits [0, ubar]
        proj_pct_x{k}      : % timesteps x_k is clipped to [0,1]
    """
    u_traj = ctrl_result.u_traj
    x_traj = ctrl_result.x_traj
    T = u_traj.shape[0]

    results: Dict[str, float] = {}

    # Control saturation
    for i in range(NUM_PLAYERS):
        ubar = float(cpss_params.ubar[i])
        n_sat = np.sum(
            (u_traj[:, i] <= tol) | (np.abs(u_traj[:, i] - ubar) <= tol)
        )
        results[f"sat_pct_player{i+1}"] = float(n_sat / T * 100)

    # Lower saturation only (hitting 0)
    for i in range(NUM_PLAYERS):
        n_lo = np.sum(u_traj[:, i] <= tol)
        results[f"sat_lo_pct_player{i+1}"] = float(n_lo / T * 100)

    # Upper saturation (hitting ubar)
    for i in range(NUM_PLAYERS):
        ubar = float(cpss_params.ubar[i])
        n_hi = np.sum(np.abs(u_traj[:, i] - ubar) <= tol)
        results[f"sat_hi_pct_player{i+1}"] = float(n_hi / T * 100)

    # State projection for controlled states (x1, x8, x9, x10)
    controlled_indices = {0: 'x1', 7: 'x8', 8: 'x9', 9: 'x10'}
    for idx, name in controlled_indices.items():
        # Check if state is at boundary (would have been clipped)
        n_clip = np.sum(
            (x_traj[1:, idx] <= tol) | (np.abs(x_traj[1:, idx] - 1.0) <= tol)
        )
        results[f"proj_pct_{name}"] = float(n_clip / T * 100)

    return results


# ═══════════════════════════════════════════════════════════════════════
#  4.  Sensitivity Check: Alternative ODE for x8/x9/x10
# ═══════════════════════════════════════════════════════════════════════

def _drift_saturating(
    x: np.ndarray,
    d: np.ndarray,
    params: CPSSParams,
    PS_override=None,
    Cplus_override=None,
) -> np.ndarray:
    """Alternative drift with saturating recovery for x8, x9, x10.

    Instead of linear relaxation  dx/dt = gamma*(d - x),
    use a saturating form:  dx/dt = gamma * tanh(K*(d - x))
    where K controls the saturation sharpness.

    Accepts the same signature as cpss_model.drift for monkey-patching.
    """
    from cpss_model import (
        STATE_DIM, compute_PS, compute_Cplus,
        compute_x1hat, compute_x2hat, compute_x3hat,
        compute_eq4, compute_eq5, compute_eq6, compute_eq7,
        gamma_mixing as gm,
        IX1, IX2, IX3, IX4, IX5, IX6, IX7, IX8, IX9, IX10,
    )

    f = np.zeros(STATE_DIM)

    PS = compute_PS(x, PS_override)
    Cplus = compute_Cplus(x, Cplus_override)

    # Social states (identical to original)
    x1hat = compute_x1hat(x, PS, Cplus, params)
    x2hat = compute_x2hat(x, PS, Cplus, params)
    x3hat = compute_x3hat(x, PS, Cplus, params)

    f[IX1] = params.alpha1 * (gm(x1hat, x[IX1], x[IX5], params.eta1) - x[IX1])
    f[IX2] = params.alpha2 * (gm(x2hat, x[IX2], x[IX5], params.eta2) - x[IX2])
    f[IX3] = params.alpha3 * (gm(x3hat, x[IX3], x[IX5], params.eta3) - x[IX3])

    f[IX4] = params.alpha4 * (compute_eq4(x, params) - x[IX4])
    f[IX5] = params.alpha5 * (compute_eq5(x, params) - x[IX5])
    f[IX6] = params.alpha6 * (compute_eq6(x, params) - x[IX6])
    f[IX7] = params.alpha7 * (compute_eq7(x, params) - x[IX7])

    # Physical/cyber: saturating recovery instead of linear
    K_sat = 3.0
    for state_idx, gamma_val in [(IX8, params.gamma8), (IX9, params.gamma9), (IX10, params.gamma10)]:
        gap = d[state_idx] - x[state_idx]
        f[state_idx] = gamma_val * np.tanh(K_sat * gap)

    return f


def run_sensitivity_check(
    x_data: np.ndarray,
    d_traj: np.ndarray,
    cpss_params: CPSSParams,
    cost_config: CostConfig,
    learn_config: LearningConfig,
    dt: float = 1.0,
) -> Dict[str, float]:
    """Run actor-critic with the alternative saturating-recovery ODE and
    compare key metrics against the baseline linear-relaxation run.

    Returns comparison metrics.
    """
    import cpss_model
    import actor_critic_game as acg

    T = x_data.shape[0] - 1

    # --- Standard run ---
    res_std = run_actor_critic(
        x0=x_data[0], d_traj=d_traj, T_steps=T,
        cpss_params=cpss_params, cost_config=cost_config,
        learn_config=learn_config, dt=dt, seed=42,
    )

    # --- Alternative run with patched drift ---
    # Monkey-patch drift in ALL modules that import it by name
    original_drift_cpss = cpss_model.drift
    original_drift_acg = acg.drift

    cpss_model.drift = _drift_saturating
    acg.drift = _drift_saturating

    try:
        res_alt = run_actor_critic(
            x0=x_data[0], d_traj=d_traj, T_steps=T,
            cpss_params=cpss_params, cost_config=cost_config,
            learn_config=learn_config, dt=dt, seed=42,
        )
    finally:
        cpss_model.drift = original_drift_cpss
        acg.drift = original_drift_acg

    # Compare
    fear_std = float(np.mean(res_std.x_traj[:T+1, IX1]))
    fear_alt = float(np.mean(res_alt.x_traj[:T+1, IX1]))

    pow_def_std = float(np.mean(1.0 - res_std.x_traj[:T+1, IX8]))
    pow_def_alt = float(np.mean(1.0 - res_alt.x_traj[:T+1, IX8]))

    ems_def_std = float(np.mean(1.0 - res_std.x_traj[:T+1, IX9]))
    ems_def_alt = float(np.mean(1.0 - res_alt.x_traj[:T+1, IX9]))

    effort_std = float(np.sum(res_std.u_traj ** 2))
    effort_alt = float(np.sum(res_alt.u_traj ** 2))

    return {
        "fear_mean_linear": fear_std,
        "fear_mean_saturating": fear_alt,
        "fear_diff_pct": abs(fear_std - fear_alt) / max(fear_std, 1e-8) * 100,
        "pow_deficit_linear": pow_def_std,
        "pow_deficit_saturating": pow_def_alt,
        "ems_deficit_linear": ems_def_std,
        "ems_deficit_saturating": ems_def_alt,
        "effort_linear": effort_std,
        "effort_saturating": effort_alt,
    }

