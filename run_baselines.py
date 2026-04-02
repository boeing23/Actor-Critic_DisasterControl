#!/usr/bin/env python3
"""
Competing baseline controllers for comparison against the NZS actor-critic.

Baselines:
  1. Open-loop (u=0)           — already in the paper
  2. Constant maximum (u=ubar) — aggressive but wasteful
  3. Proportional (greedy)     — u_i proportional to domain deficit
  4. Centralized single-agent  — one agent minimizes sum of all Q_i

All baselines use the same CPS dynamics and are evaluated on Harvey data.
"""

from __future__ import annotations
import os, json, copy
import numpy as np

from data import load_data
from cpss_model import (
    CPSSParams, STATE_DIM, NUM_PLAYERS, compute_d_traj,
    drift, g_vectors, compute_PS, compute_Cplus,
    IX1, IX4, IX8, IX9, IX10,
)
from actor_critic_game import (
    CostConfig, LearningConfig, SimulationResult,
    run_actor_critic, simulate_baseline,
    compute_Q,
)
from features import phi_quadratic, grad_phi_quadratic, quadratic_basis_dim
from fit_params import fit_cpss_params
from utils import saturate, project_controlled_states, probing_signal

ARTIFACTS = os.path.join(os.path.dirname(__file__) or ".", "artifacts")


# ═══════════════════════════════════════════════════════════════════════
#  Baseline 2: Constant maximum control
# ═══════════════════════════════════════════════════════════════════════

def simulate_constant_max(
    x0: np.ndarray,
    d_traj: np.ndarray,
    T_steps: int,
    cpss_params: CPSSParams,
    dt: float = 1.0,
) -> tuple:
    """All players apply u_j = ubar_j at every step."""
    n = STATE_DIM
    n_sub = cpss_params.n_substeps
    dt_sub = dt / n_sub
    g_vecs = g_vectors(cpss_params)

    x_traj = np.zeros((T_steps + 1, n))
    u_traj = np.zeros((T_steps, NUM_PLAYERS))
    x_traj[0] = x0.copy()

    for k in range(T_steps):
        d_idx = min(k, d_traj.shape[0] - 1)
        d_k = d_traj[d_idx]

        u_exec = np.array([float(cpss_params.ubar[j]) for j in range(NUM_PLAYERS)])
        u_traj[k] = u_exec

        x_sub = x_traj[k].copy()
        for _s in range(n_sub):
            f_sub = drift(x_sub, d_k, cpss_params)
            dyn = f_sub.copy()
            for j in range(NUM_PLAYERS):
                dyn += g_vecs[j] * u_exec[j]
            x_sub = x_sub + dyn * dt_sub

        x_traj[k + 1] = project_controlled_states(x_sub)

    return x_traj, u_traj


# ═══════════════════════════════════════════════════════════════════════
#  Baseline 3: Proportional (greedy) controller
# ═══════════════════════════════════════════════════════════════════════

def simulate_proportional(
    x0: np.ndarray,
    d_traj: np.ndarray,
    T_steps: int,
    cpss_params: CPSSParams,
    dt: float = 1.0,
    gains: tuple = (1.0, 1.0, 1.0),
) -> tuple:
    """Greedy proportional: u1 = K1*x1, u2 = K2*(1-x8), u3 = K3*(1-x9).

    Each player reacts to its own domain deficit, saturated to [0, ubar].
    """
    n = STATE_DIM
    n_sub = cpss_params.n_substeps
    dt_sub = dt / n_sub
    g_vecs = g_vectors(cpss_params)

    x_traj = np.zeros((T_steps + 1, n))
    u_traj = np.zeros((T_steps, NUM_PLAYERS))
    x_traj[0] = x0.copy()

    K1, K2, K3 = gains

    for k in range(T_steps):
        d_idx = min(k, d_traj.shape[0] - 1)
        d_k = d_traj[d_idx]
        x_k = x_traj[k]

        # Proportional to domain deficit
        u1_raw = K1 * x_k[IX1]              # fear level → counter-messaging
        u2_raw = K2 * (1.0 - x_k[IX8])      # power deficit → restoration
        u3_raw = K3 * (1.0 - x_k[IX9])      # EMS deficit → deployment

        u_exec = np.array([
            saturate(u1_raw, 0.0, float(cpss_params.ubar[0])),
            saturate(u2_raw, 0.0, float(cpss_params.ubar[1])),
            saturate(u3_raw, 0.0, float(cpss_params.ubar[2])),
        ])
        u_traj[k] = u_exec

        x_sub = x_k.copy()
        for _s in range(n_sub):
            f_sub = drift(x_sub, d_k, cpss_params)
            dyn = f_sub.copy()
            for j in range(NUM_PLAYERS):
                dyn += g_vecs[j] * u_exec[j]
            x_sub = x_sub + dyn * dt_sub

        x_traj[k + 1] = project_controlled_states(x_sub)

    return x_traj, u_traj


# ═══════════════════════════════════════════════════════════════════════
#  Baseline 4: Centralized single-agent controller
# ═══════════════════════════════════════════════════════════════════════

def run_centralized_actor_critic(
    x0: np.ndarray,
    d_traj: np.ndarray,
    T_steps: int,
    cpss_params: CPSSParams,
    cost_config: CostConfig,
    learn_config: LearningConfig,
    dt: float = 1.0,
    seed: int = 42,
) -> tuple:
    """Centralized actor-critic: one agent controls all three u_j,
    minimizing a combined cost Q_total = Q1 + Q2 + Q3 + sum R_jj u_j^2.

    Uses one critic with the same quadratic basis, and three actor outputs
    derived from a single value function gradient.
    """
    np.random.seed(seed)

    n = STATE_DIM
    p = quadratic_basis_dim(n)
    Delta = learn_config.window_length
    T_ex = learn_config.T_explore
    n_sub = cpss_params.n_substeps
    dt_sub = dt / n_sub

    Wc = np.zeros(p)   # single critic
    # Three actor weight vectors (one per control channel)
    Wa = [np.zeros(p) for _ in range(NUM_PLAYERS)]

    g_vecs = g_vectors(cpss_params)

    x_traj = np.zeros((T_steps + 1, n))
    u_traj = np.zeros((T_steps, NUM_PLAYERS))
    x_traj[0] = x0.copy()

    num_windows = (T_steps + Delta - 1) // Delta

    alpha_c = 0.5
    alpha_a = 0.1

    for w in range(num_windows):
        t_start = w * Delta
        t_end = min((w + 1) * Delta, T_steps)

        d_idx = min(t_start, d_traj.shape[0] - 1)
        d_frozen = d_traj[d_idx].copy()
        PS_frozen = compute_PS(d_traj[d_idx])
        Cp_frozen = compute_Cplus(d_traj[d_idx])

        for k in range(t_start, t_end):
            x_k = x_traj[k]
            gp = grad_phi_quadratic(x_k)  # (n, p)

            # Compute centralized policies: u_j = -1/(2 R_jj) g_j^T grad_phi Wa_j
            uhat = np.zeros(NUM_PLAYERS)
            for j in range(NUM_PLAYERS):
                R_jj = float(cost_config.R[j, j])
                nabla_V = gp @ Wa[j]
                uhat[j] = float(-1.0 / (2.0 * R_jj) * np.dot(g_vecs[j], nabla_V))

            # Exploration + saturation
            u_exec = np.zeros(NUM_PLAYERS)
            for j in range(NUM_PLAYERS):
                noise = 0.0
                if k < T_ex:
                    noise = probing_signal(float(k), j, 0.3)
                u_exec[j] = saturate(uhat[j] + noise, 0.0, float(cpss_params.ubar[j]))
            u_traj[k] = u_exec

            # Drift
            f_val = drift(x_k, d_frozen, cpss_params,
                         PS_override=PS_frozen, Cplus_override=Cp_frozen)
            dynamics = f_val.copy()
            for j in range(NUM_PLAYERS):
                dynamics += g_vecs[j] * u_exec[j]

            sigma = gp.T @ dynamics  # (p,)

            # Combined cost: Q1 + Q2 + Q3 + sum R_jj u_j^2
            Q_total = sum(compute_Q(i, x_k, cost_config) for i in range(NUM_PLAYERS))
            u_pen = sum(cost_config.R[j, j] * u_exec[j] ** 2 for j in range(NUM_PLAYERS))
            eps = Q_total + u_pen + float(Wc @ sigma)

            # Single critic update
            norm_sq_plus1 = float(sigma @ sigma) + 1.0
            Wc = Wc - alpha_c * (sigma / (norm_sq_plus1 ** 2)) * eps * dt

            # Actor updates: each Wa_j tracks the single Wc
            for j in range(NUM_PLAYERS):
                Wa[j] = Wa[j] - alpha_a * (Wa[j] - Wc) * dt

            # State propagation
            x_sub = x_k.copy()
            for _s in range(n_sub):
                f_sub = drift(x_sub, d_frozen, cpss_params,
                             PS_override=PS_frozen, Cplus_override=Cp_frozen)
                dyn_sub = f_sub.copy()
                for j in range(NUM_PLAYERS):
                    dyn_sub += g_vecs[j] * u_exec[j]
                x_sub = x_sub + dyn_sub * dt_sub

            x_traj[k + 1] = project_controlled_states(x_sub)

    return x_traj, u_traj


# ═══════════════════════════════════════════════════════════════════════
#  Metric computation
# ═══════════════════════════════════════════════════════════════════════

def compute_comparison_metrics(
    x_traj: np.ndarray,
    u_traj: np.ndarray,
    x_base: np.ndarray,
    T: int,
    cost_config: CostConfig,
) -> dict:
    """Compute standard metrics for any controller."""
    x = x_traj[:T+1]
    xb = x_base[:T+1]

    fear_base = float(np.mean(xb[:, IX1]))
    fear_ctrl = float(np.mean(x[:, IX1]))
    fear_red = (fear_base - fear_ctrl) / max(fear_base, 1e-8) * 100

    pow_def = float(np.mean(1.0 - x[:, IX8]))
    ems_def = float(np.mean(1.0 - x[:, IX9]))
    health_def = float(np.mean(1.0 - x[:, IX4]))
    effort = float(np.sum(u_traj[:T] ** 2))

    # Total integrated cost (sum of all players' costs)
    total_cost = 0.0
    for k in range(T):
        for i in range(NUM_PLAYERS):
            Q_i = compute_Q(i, x[k], cost_config)
            u_pen = sum(cost_config.R[i, j] * u_traj[k, j] ** 2
                       for j in range(NUM_PLAYERS))
            total_cost += Q_i + u_pen

    return {
        "fear_red_pct": round(fear_red, 1),
        "mean_fear": round(fear_ctrl, 3),
        "pow_deficit": round(pow_def, 3),
        "ems_deficit": round(ems_def, 3),
        "health_deficit": round(health_def, 3),
        "effort": round(effort, 3),
        "total_cost": round(total_cost, 2),
    }


def main():
    # Load and fit
    x_data, dt = load_data()
    T = x_data.shape[0] - 1
    cpss_params = fit_cpss_params(x_data, dt, verbose=False)
    d_traj = compute_d_traj(x_data, dt, cpss_params)
    cost_cfg = CostConfig()
    learn_cfg = LearningConfig()

    # ── 1. Open-loop baseline ────────────────────────────────────────
    print("Running: Open-loop (u=0) ...")
    x_base = simulate_baseline(x_data[0], d_traj, T, cpss_params, dt)
    u_zero = np.zeros((T, NUM_PLAYERS))
    m_open = compute_comparison_metrics(x_base, u_zero, x_base, T, cost_cfg)
    m_open["label"] = "Open-loop ($u{=}0$)"
    print(f"  Fear red: {m_open['fear_red_pct']}%  Total cost: {m_open['total_cost']}")

    # ── 2. Constant maximum ──────────────────────────────────────────
    print("Running: Constant maximum ...")
    x_const, u_const = simulate_constant_max(x_data[0], d_traj, T, cpss_params, dt)
    m_const = compute_comparison_metrics(x_const, u_const, x_base, T, cost_cfg)
    m_const["label"] = "Constant ($u{=}\\bar{u}$)"
    print(f"  Fear red: {m_const['fear_red_pct']}%  Total cost: {m_const['total_cost']}")

    # ── 3. Proportional ──────────────────────────────────────────────
    print("Running: Proportional (K=1) ...")
    x_prop, u_prop = simulate_proportional(x_data[0], d_traj, T, cpss_params, dt,
                                            gains=(1.0, 1.0, 1.0))
    m_prop = compute_comparison_metrics(x_prop, u_prop, x_base, T, cost_cfg)
    m_prop["label"] = "Proportional"
    print(f"  Fear red: {m_prop['fear_red_pct']}%  Total cost: {m_prop['total_cost']}")

    # ── 3b. Proportional (tuned) ─────────────────────────────────────
    # Try a few gain settings and pick the best total-cost one
    best_prop = None
    best_prop_cost = float('inf')
    for K1 in [0.5, 1.0, 2.0]:
        for K2 in [0.5, 1.0]:
            for K3 in [0.5, 1.0]:
                x_p, u_p = simulate_proportional(
                    x_data[0], d_traj, T, cpss_params, dt,
                    gains=(K1, K2, K3))
                m = compute_comparison_metrics(x_p, u_p, x_base, T, cost_cfg)
                if m["total_cost"] < best_prop_cost:
                    best_prop_cost = m["total_cost"]
                    best_prop = (x_p, u_p, m, (K1, K2, K3))

    x_prop_t, u_prop_t, m_prop_t, best_gains = best_prop
    m_prop_t["label"] = f"Prop.\\,tuned"
    print(f"Running: Proportional (tuned K={best_gains}) ...")
    print(f"  Fear red: {m_prop_t['fear_red_pct']}%  Total cost: {m_prop_t['total_cost']}")

    # ── 4. Centralized single-agent ──────────────────────────────────
    print("Running: Centralized AC ...")
    x_cent, u_cent = run_centralized_actor_critic(
        x_data[0], d_traj, T, cpss_params, cost_cfg, learn_cfg, dt, seed=42)
    m_cent = compute_comparison_metrics(x_cent, u_cent, x_base, T, cost_cfg)
    m_cent["label"] = "Centralized AC"
    print(f"  Fear red: {m_cent['fear_red_pct']}%  Total cost: {m_cent['total_cost']}")

    # ── 5. NZS Actor-Critic (our method) ─────────────────────────────
    print("Running: NZS Actor-Critic ...")
    ctrl_res = run_actor_critic(
        x0=x_data[0], d_traj=d_traj, T_steps=T,
        cpss_params=cpss_params, cost_config=cost_cfg,
        learn_config=learn_cfg, dt=dt, seed=42)
    m_nzs = compute_comparison_metrics(ctrl_res.x_traj, ctrl_res.u_traj, x_base, T, cost_cfg)
    m_nzs["label"] = "NZS AC (ours)"
    print(f"  Fear red: {m_nzs['fear_red_pct']}%  Total cost: {m_nzs['total_cost']}")

    # ── Summary ──────────────────────────────────────────────────────
    results = [m_open, m_const, m_prop, m_prop_t, m_cent, m_nzs]

    print("\n" + "=" * 80)
    print(f"{'Method':<25} {'Fear%':>7} {'MnFear':>7} {'Pow':>7} {'EMS':>7} "
          f"{'Effort':>7} {'TotCost':>8}")
    print("-" * 80)
    for r in results:
        print(f"{r['label']:<25} {r['fear_red_pct']:>7.1f} {r['mean_fear']:>7.3f} "
              f"{r['pow_deficit']:>7.3f} {r['ems_deficit']:>7.3f} "
              f"{r['effort']:>7.3f} {r['total_cost']:>8.2f}")
    print("=" * 80)

    # Save
    out_path = os.path.join(ARTIFACTS, "baseline_comparison.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
