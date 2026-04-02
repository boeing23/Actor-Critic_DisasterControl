#!/usr/bin/env python3
"""
═════════════════════════════════════════════════════════════════════════
  CPSS Actor–Critic 3-Player Game  —  Main Experiment Script
═════════════════════════════════════════════════════════════════════════

HOW TO RUN
----------
    python run_experiment.py

WHAT IT DOES
------------
1.  Loads the 10-dimensional CPSS disaster time-series  (data.py).
2.  Fits unknown model/control parameters via least-squares on
    finite-difference derivatives                        (fit_params.py).
3.  Optionally tunes cost weights via random search.
4.  Runs a *baseline* (u = 0) simulation from x[0].
5.  Runs online actor–critic learning with piecewise-stationary
    windowing (Algorithm A5) for 3 players.
6.  Validates closed-loop control vs. the open-loop baseline.
7.  Writes all outputs to  ``./artifacts/``.

ARTIFACTS
---------
    artifacts/
      fitted_params.json    — fitted CPSS + cost parameters
      learned_weights.npz   — final actor/critic weight vectors
      metrics.json           — quantitative comparison
      plots/
        states.png           — 10 states: observed vs baseline vs controlled
        controls.png         — u1, u2, u3  over time
        bellman.png          — Bellman residuals ε_i
        weights.png          — ‖W_c‖, ‖W_a‖  norms

DEPENDENCIES
------------
    numpy, scipy, matplotlib   (no deep-learning frameworks)
═════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt

from data import load_data, STATE_NAMES, PLAYER_NAMES
from utils import (
    ensure_dir, save_json, setup_logging, finite_differences,
)
from cpss_model import (
    CPSSParams, STATE_DIM, NUM_PLAYERS, compute_d_traj,
    IX1, IX4, IX8, IX9,
)
from actor_critic_game import (
    CostConfig, LearningConfig, SimulationResult,
    run_actor_critic, simulate_baseline,
)
from fit_params import fit_cpss_params, fit_cost_weights
from diagnostics import (
    compute_nash_gap, compute_pe_diagnostics,
    compute_saturation_stats, run_sensitivity_check,
)

# ── paths ────────────────────────────────────────────────────────────────────
ARTIFACTS = os.path.join(os.path.dirname(__file__) or ".", "artifacts")
PLOTS_DIR = os.path.join(ARTIFACTS, "plots")


# ═══════════════════════════════════════════════════════════════════════
#  Metric computation
# ═══════════════════════════════════════════════════════════════════════

def compute_metrics(
    x_data: np.ndarray,
    x_base: np.ndarray,
    ctrl_res: SimulationResult,
) -> dict:
    """Compute all validation metrics."""
    T = x_data.shape[0]
    x_ctrl = ctrl_res.x_traj[:T]

    # Tracking error (baseline vs observed)
    track_err = float(np.sqrt(np.mean((x_base[:T] - x_data) ** 2)))

    # Fear
    fear_base_mean = float(np.mean(x_base[:T, IX1]))
    fear_ctrl_mean = float(np.mean(x_ctrl[:, IX1]))
    fear_base_max = float(np.max(x_base[:T, IX1]))
    fear_ctrl_max = float(np.max(x_ctrl[:, IX1]))

    # Deficits
    def deficit(arr: np.ndarray, idx: int) -> float:
        return float(np.mean(1.0 - arr[:, idx]))

    deficit_x8_base = deficit(x_base[:T], IX8)
    deficit_x9_base = deficit(x_base[:T], IX9)
    deficit_x4_base = deficit(x_base[:T], IX4)

    deficit_x8_ctrl = deficit(x_ctrl, IX8)
    deficit_x9_ctrl = deficit(x_ctrl, IX9)
    deficit_x4_ctrl = deficit(x_ctrl, IX4)

    # Control effort
    ctrl_effort = float(np.sum(ctrl_res.u_traj ** 2))

    return {
        "baseline_tracking_RMSE": track_err,
        "fear_mean_baseline": fear_base_mean,
        "fear_mean_controlled": fear_ctrl_mean,
        "fear_max_baseline": fear_base_max,
        "fear_max_controlled": fear_ctrl_max,
        "fear_reduction_mean_pct": (fear_base_mean - fear_ctrl_mean)
                                   / max(fear_base_mean, 1e-8) * 100,
        "deficit_x8_baseline": deficit_x8_base,
        "deficit_x8_controlled": deficit_x8_ctrl,
        "deficit_x9_baseline": deficit_x9_base,
        "deficit_x9_controlled": deficit_x9_ctrl,
        "deficit_x4_baseline": deficit_x4_base,
        "deficit_x4_controlled": deficit_x4_ctrl,
        "control_effort_total": ctrl_effort,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Plotting
# ═══════════════════════════════════════════════════════════════════════

def _time_axis(T: int, dt: float) -> np.ndarray:
    return np.arange(T) * dt


def plot_states(
    x_data: np.ndarray,
    x_base: np.ndarray,
    x_ctrl: np.ndarray,
    dt: float,
    path: str,
) -> None:
    """10-panel figure: observed vs baseline vs controlled."""
    T = x_data.shape[0]
    t_obs = _time_axis(T, dt)
    t_sim = _time_axis(x_base.shape[0], dt)
    t_ctrl = _time_axis(x_ctrl.shape[0], dt)

    fig, axes = plt.subplots(5, 2, figsize=(14, 16), sharex=True)
    axes = axes.flatten()

    for i in range(STATE_DIM):
        ax = axes[i]
        ax.plot(t_obs, x_data[:, i], "ko-", ms=3, label="Observed")
        ax.plot(t_sim, x_base[:, i], "b--", lw=1.2, label="Baseline (u=0)")
        ax.plot(t_ctrl, x_ctrl[:, i], "r-", lw=1.4, label="Controlled")
        ax.set_ylabel(STATE_NAMES[i], fontsize=8)
        ax.set_ylim(-0.05, 1.15)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")

    for ax in axes[-2:]:
        ax.set_xlabel("Time step")

    fig.suptitle("CPSS States: Observed vs Baseline vs Controlled", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_controls(
    u_traj: np.ndarray, dt: float, path: str,
) -> None:
    """3-panel figure of u1, u2, u3."""
    T = u_traj.shape[0]
    t = _time_axis(T, dt)

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for i, ax in enumerate(axes):
        ax.step(t, u_traj[:, i], where="mid", lw=1.4)
        ax.set_ylabel(PLAYER_NAMES[i])
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time step")
    fig.suptitle("Control Inputs", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_bellman(
    bell: np.ndarray, dt: float, path: str,
) -> None:
    """Bellman residuals ε_i over time."""
    T = bell.shape[0]
    t = _time_axis(T, dt)

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, bell[:, i], lw=1.2)
        ax.set_ylabel(f"ε_{i+1}")
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time step")
    fig.suptitle("Bellman Residuals", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_weights(
    Wc_norms: np.ndarray,
    Wa_norms: np.ndarray,
    dt: float,
    path: str,
) -> None:
    """‖W_c‖ and ‖W_a‖ for each player."""
    T = Wc_norms.shape[0]
    t = _time_axis(T, dt)

    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    for i, ax in enumerate(axes):
        ax.plot(t, Wc_norms[:, i], label=f"‖Wc_{i+1}‖", lw=1.2)
        ax.plot(t, Wa_norms[:, i], "--", label=f"‖Wa_{i+1}‖", lw=1.2)
        ax.set_ylabel(PLAYER_NAMES[i])
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time step")
    fig.suptitle("Weight Norms", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
#  PE eigenvalue plot
# ═══════════════════════════════════════════════════════════════════════

def plot_pe_eigenvalues(
    pe_data: dict, dt: float, T_explore: int, path: str,
) -> None:
    """Plot minimum eigenvalue of running and windowed PE covariance proxy."""
    min_eig_run = pe_data["min_eig_running"]
    min_eig_win = pe_data["min_eig_window"]
    T = min_eig_run.shape[0]
    t = _time_axis(T, dt)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax = axes[0]
    ax.plot(t, min_eig_run[:, 0], lw=1.2, label="Running avg")
    ax.axvline(T_explore * dt, color='gray', ls='--', lw=0.8, label=f"T_ex={T_explore}")
    ax.set_ylabel("λ_min (running)")
    ax.set_title("PE Proxy: Min Eigenvalue of Normalised Regressor Covariance")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t, min_eig_win[:, 0], lw=1.2, color='C1', label="Windowed")
    ax.axvline(T_explore * dt, color='gray', ls='--', lw=0.8)
    ax.set_ylabel("λ_min (windowed)")
    ax.set_xlabel("Time step")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════
#  Extended rollout (optional longer horizon)
# ═══════════════════════════════════════════════════════════════════════

def extend_d_traj(
    d_traj: np.ndarray, extra: int,
) -> np.ndarray:
    """Extrapolate d_traj by holding last row constant."""
    if extra <= 0:
        return d_traj
    tail = np.tile(d_traj[-1:], (extra, 1))
    return np.vstack([d_traj, tail])


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    log = setup_logging()
    ensure_dir(PLOTS_DIR)

    # ── 1. Load data ─────────────────────────────────────────────────
    log.info("Loading data …")
    x_data, dt = load_data()
    T = x_data.shape[0] - 1        # 17 simulation steps
    log.info("  %d samples, dt = %.1f, T_steps = %d", x_data.shape[0], dt, T)

    # ── 2. Fit CPSS parameters ───────────────────────────────────────
    log.info("═══ Stage 1: fitting CPSS model parameters ═══")
    cpss_params = fit_cpss_params(x_data, dt, verbose=True)
    log.info("  Fitted params saved.")

    # Derive exogenous d trajectory
    d_traj = compute_d_traj(x_data, dt, cpss_params)

    # ── 3. Baseline simulation ───────────────────────────────────────
    log.info("Running baseline (u = 0) simulation …")
    x_base = simulate_baseline(x_data[0], d_traj, T, cpss_params, dt)
    base_rmse = float(np.sqrt(np.mean((x_base[:T+1] - x_data) ** 2)))
    log.info("  Baseline RMSE (all states) = %.4f", base_rmse)

    # ── 4. Fit / search cost weights ─────────────────────────────────
    log.info("═══ Stage 2: searching cost weights ═══")
    best_cost_kw = fit_cost_weights(x_data, dt, cpss_params, n_trials=40,
                                     verbose=True)
    log.info("  Best cost config: %s",
             {k: (round(v, 3) if isinstance(v, float) else v)
              for k, v in best_cost_kw.items()})

    # Rebuild CostConfig from best keywords
    cost_cfg = CostConfig()
    for k, v in best_cost_kw.items():
        if k == "R":
            cost_cfg.R = np.array(v)
        elif hasattr(cost_cfg, k):
            setattr(cost_cfg, k, v)

    learn_cfg = LearningConfig()

    # ── 5. Actor-critic controlled run  (data horizon) ───────────────
    log.info("═══ Stage 3: actor–critic controlled simulation ═══")
    ctrl_res = run_actor_critic(
        x0=x_data[0],
        d_traj=d_traj,
        T_steps=T,
        cpss_params=cpss_params,
        cost_config=cost_cfg,
        learn_config=learn_cfg,
        dt=dt,
        seed=42,
    )
    log.info("  Controlled run completed (%d steps).", T)

    # ── 5b. Extended rollout (optional: 2× horizon) ──────────────────
    T_ext = 2 * T
    d_ext = extend_d_traj(d_traj, T_ext - d_traj.shape[0] + 1)
    learn_ext = LearningConfig(T_explore=T)   # no extra exploration
    ctrl_ext = run_actor_critic(
        x0=x_data[0],
        d_traj=d_ext,
        T_steps=T_ext,
        cpss_params=cpss_params,
        cost_config=cost_cfg,
        learn_config=learn_ext,
        dt=dt,
        seed=42,
    )
    x_base_ext = simulate_baseline(x_data[0], d_ext, T_ext, cpss_params, dt)
    log.info("  Extended rollout (%d steps) completed.", T_ext)

    # ── 6. Metrics ───────────────────────────────────────────────────
    log.info("Computing metrics …")
    metrics = compute_metrics(x_data, x_base, ctrl_res)
    for k, v in metrics.items():
        log.info("  %-35s  %.4f", k, v)

    # ── 6b. Near-Nash / exploitability test ──────────────────────────
    log.info("═══ Stage 4: Near-Nash exploitability test ═══")
    nash_gap = compute_nash_gap(
        x0=x_data[0],
        d_traj=d_traj,
        T_steps=T,
        Wa_converged=ctrl_res.Wa_final,
        cpss_params=cpss_params,
        cost_config=cost_cfg,
        dt=dt,
        n_deviations=50,
        deviation_scales=(0.05, 0.10, 0.20),
        seed=99,
    )
    for k, v in nash_gap.items():
        log.info("  %-35s  %.4f", k, v)
    metrics.update(nash_gap)

    # ── 6c. PE diagnostics ───────────────────────────────────────────
    log.info("═══ Stage 5: PE diagnostics ═══")
    pe_data = compute_pe_diagnostics(
        ctrl_res, d_traj, cpss_params, cost_cfg, learn_cfg, dt,
    )
    pe_min_run_final = float(pe_data["min_eig_running"][-1, 0])
    pe_min_win_final = float(pe_data["min_eig_window"][-1, 0])
    pe_min_run_explore = float(np.min(pe_data["min_eig_running"][:learn_cfg.T_explore, 0]))
    pe_rank_final = float(pe_data["rank_running"][-1])
    pe_basis_dim = int(pe_data["basis_dim"])
    log.info("  PE min-eig (running, final)   = %.6f", pe_min_run_final)
    log.info("  PE min-eig (windowed, final)  = %.6f", pe_min_win_final)
    log.info("  PE min-eig (running, explore) = %.6f", pe_min_run_explore)
    log.info("  PE rank / basis dim           = %d / %d", pe_rank_final, pe_basis_dim)
    metrics["pe_min_eig_running_final"] = pe_min_run_final
    metrics["pe_min_eig_windowed_final"] = pe_min_win_final
    metrics["pe_min_eig_running_explore"] = pe_min_run_explore
    metrics["pe_rank_final"] = pe_rank_final
    metrics["pe_basis_dim"] = pe_basis_dim

    # ── 6d. Saturation & projection stats ────────────────────────────
    log.info("═══ Stage 6: saturation & projection stats ═══")
    sat_stats = compute_saturation_stats(ctrl_res, cpss_params)
    for k, v in sat_stats.items():
        log.info("  %-35s  %.1f%%", k, v)
    metrics.update(sat_stats)

    # ── 6e. Sensitivity: alternative ODE ─────────────────────────────
    log.info("═══ Stage 7: sensitivity check (saturating ODE) ═══")
    sens = run_sensitivity_check(
        x_data, d_traj, cpss_params, cost_cfg, learn_cfg, dt,
    )
    for k, v in sens.items():
        log.info("  %-35s  %.4f", k, v)
    metrics.update(sens)

    # ── 7. Save artifacts ────────────────────────────────────────────
    log.info("Saving artifacts to %s …", ARTIFACTS)

    # fitted params
    all_params: dict = {
        "cpss_params": cpss_params.to_dict(),
        "cost_config": cost_cfg.to_dict(),
        "learning_config": learn_cfg.to_dict(),
    }
    save_json(all_params, os.path.join(ARTIFACTS, "fitted_params.json"))

    # learned weights (include full final weights for reproducibility)
    np.savez(
        os.path.join(ARTIFACTS, "learned_weights.npz"),
        Wc_norms=ctrl_res.Wc_norms,
        Wa_norms=ctrl_res.Wa_norms,
        u_traj=ctrl_res.u_traj,
        bellman=ctrl_res.bellman_residuals,
        Wc_final_0=ctrl_res.Wc_final[0],
        Wc_final_1=ctrl_res.Wc_final[1],
        Wc_final_2=ctrl_res.Wc_final[2],
        Wa_final_0=ctrl_res.Wa_final[0],
        Wa_final_1=ctrl_res.Wa_final[1],
        Wa_final_2=ctrl_res.Wa_final[2],
    )

    # metrics
    save_json(metrics, os.path.join(ARTIFACTS, "metrics.json"))

    # ── 8. Plots ─────────────────────────────────────────────────────
    log.info("Generating plots …")

    # a) states — data-horizon view
    plot_states(
        x_data, x_base, ctrl_res.x_traj, dt,
        os.path.join(PLOTS_DIR, "states.png"),
    )

    # b) states — extended rollout
    plot_states(
        x_data, x_base_ext, ctrl_ext.x_traj, dt,
        os.path.join(PLOTS_DIR, "states_extended.png"),
    )

    # c) controls
    plot_controls(ctrl_res.u_traj, dt, os.path.join(PLOTS_DIR, "controls.png"))
    plot_controls(ctrl_ext.u_traj, dt,
                  os.path.join(PLOTS_DIR, "controls_extended.png"))

    # d) Bellman residuals
    plot_bellman(ctrl_res.bellman_residuals, dt,
                 os.path.join(PLOTS_DIR, "bellman.png"))

    # e) weight norms
    plot_weights(ctrl_res.Wc_norms, ctrl_res.Wa_norms, dt,
                 os.path.join(PLOTS_DIR, "weights.png"))

    # f) PE eigenvalue diagnostics
    plot_pe_eigenvalues(
        pe_data, dt, learn_cfg.T_explore,
        os.path.join(PLOTS_DIR, "pe_eigenvalues.png"),
    )

    log.info("═══ Done.  All artifacts in %s ═══", ARTIFACTS)


if __name__ == "__main__":
    main()

