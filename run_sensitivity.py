#!/usr/bin/env python3
"""
Sensitivity analysis: vary cost weights and control gains, report key metrics.

Generates a compact table for the paper showing robustness of the framework.
"""

from __future__ import annotations
import os, sys, json, copy
import numpy as np

from data import load_data
from cpss_model import CPSSParams, compute_d_traj, IX1, IX4, IX8, IX9
from actor_critic_game import (
    CostConfig, LearningConfig, SimulationResult,
    run_actor_critic, simulate_baseline,
)
from fit_params import fit_cpss_params
from diagnostics import compute_nash_gap

ARTIFACTS = os.path.join(os.path.dirname(__file__) or ".", "artifacts")


def run_one_config(
    label: str,
    cost_cfg: CostConfig,
    x_data: np.ndarray,
    d_traj: np.ndarray,
    x_base: np.ndarray,
    cpss_params: CPSSParams,
    learn_cfg: LearningConfig,
    dt: float,
    T: int,
) -> dict:
    """Run actor-critic with given cost config, return summary metrics."""
    res = run_actor_critic(
        x0=x_data[0], d_traj=d_traj, T_steps=T,
        cpss_params=cpss_params, cost_config=cost_cfg,
        learn_config=learn_cfg, dt=dt, seed=42,
    )
    x_ctrl = res.x_traj[:T+1]

    fear_base = float(np.mean(x_base[:T+1, IX1]))
    fear_ctrl = float(np.mean(x_ctrl[:, IX1]))
    fear_red = (fear_base - fear_ctrl) / max(fear_base, 1e-8) * 100

    pow_def_base = float(np.mean(1.0 - x_base[:T+1, IX8]))
    pow_def = float(np.mean(1.0 - x_ctrl[:, IX8]))
    ems_def_base = float(np.mean(1.0 - x_base[:T+1, IX9]))
    ems_def = float(np.mean(1.0 - x_ctrl[:, IX9]))
    effort = float(np.sum(res.u_traj ** 2))

    # Exploitability
    nash = compute_nash_gap(
        x0=x_data[0], d_traj=d_traj, T_steps=T,
        Wa_converged=res.Wa_final,
        cpss_params=cpss_params, cost_config=cost_cfg,
        dt=dt, n_deviations=30, deviation_scales=(0.05, 0.10, 0.20),
        seed=99,
    )

    return {
        "label": label,
        "q11": cost_cfg.q11,
        "q1_10": cost_cfg.q1_10,
        "q28": cost_cfg.q2_8,
        "q39": cost_cfg.q3_9,
        "fear_red_pct": round(fear_red, 1),
        "pow_deficit": round(pow_def, 3),
        "ems_deficit": round(ems_def, 3),
        "ems_deficit_base": round(ems_def_base, 3),
        "effort": round(effort, 3),
        "p1_exploit_5": round(nash.get("player1_improv_5pct", 0), 1),
        "p1_exploit_10": round(nash.get("player1_improv_10pct", 0), 1),
        "p1_exploit_20": round(nash.get("player1_improv_20pct", 0), 1),
        "J1": round(nash.get("J1_baseline", 0), 2),
        "J2": round(nash.get("J2_baseline", 0), 2),
        "J3": round(nash.get("J3_baseline", 0), 2),
    }


def run_with_beta(
    label: str,
    cost_cfg: CostConfig,
    x_data: np.ndarray,
    cpss_params_orig: CPSSParams,
    learn_cfg: LearningConfig,
    dt: float,
    T: int,
    beta_overrides: dict = None,
) -> dict:
    """Run with modified beta values."""
    params = copy.deepcopy(cpss_params_orig)
    if beta_overrides:
        for k, v in beta_overrides.items():
            setattr(params, k, v)

    d_traj = compute_d_traj(x_data, dt, params)
    x_base = simulate_baseline(x_data[0], d_traj, T, params, dt)

    res = run_actor_critic(
        x0=x_data[0], d_traj=d_traj, T_steps=T,
        cpss_params=params, cost_config=cost_cfg,
        learn_config=learn_cfg, dt=dt, seed=42,
    )
    x_ctrl = res.x_traj[:T+1]

    fear_base = float(np.mean(x_base[:T+1, IX1]))
    fear_ctrl = float(np.mean(x_ctrl[:, IX1]))
    fear_red = (fear_base - fear_ctrl) / max(fear_base, 1e-8) * 100

    pow_def = float(np.mean(1.0 - x_ctrl[:, IX8]))
    ems_def = float(np.mean(1.0 - x_ctrl[:, IX9]))
    ems_def_base = float(np.mean(1.0 - x_base[:T+1, IX9]))
    effort = float(np.sum(res.u_traj ** 2))

    nash = compute_nash_gap(
        x0=x_data[0], d_traj=d_traj, T_steps=T,
        Wa_converged=res.Wa_final,
        cpss_params=params, cost_config=cost_cfg,
        dt=dt, n_deviations=30, deviation_scales=(0.05, 0.10, 0.20),
        seed=99,
    )

    return {
        "label": label,
        "fear_red_pct": round(fear_red, 1),
        "pow_deficit": round(pow_def, 3),
        "ems_deficit": round(ems_def, 3),
        "ems_deficit_base": round(ems_def_base, 3),
        "effort": round(effort, 3),
        "p1_exploit_5": round(nash.get("player1_improv_5pct", 0), 1),
        "p1_exploit_10": round(nash.get("player1_improv_10pct", 0), 1),
        "p1_exploit_20": round(nash.get("player1_improv_20pct", 0), 1),
    }


def main():
    # Load and fit
    x_data, dt = load_data()
    T = x_data.shape[0] - 1
    cpss_params = fit_cpss_params(x_data, dt, verbose=False)
    d_traj = compute_d_traj(x_data, dt, cpss_params)
    x_base = simulate_baseline(x_data[0], d_traj, T, cpss_params, dt)
    learn_cfg = LearningConfig()

    # ── Part A: Cost weight sensitivity ──
    print("=" * 70)
    print("  PART A: Cost Weight Sensitivity")
    print("=" * 70)

    weight_configs = [
        # label, CostConfig
        ("Default",         CostConfig()),
        ("$2{\\times}q_{1,1}$",  CostConfig(q11=2.0)),
        ("$0.5{\\times}q_{1,1}$", CostConfig(q11=0.5)),
        ("$2{\\times}q_{2,8}$",  CostConfig(q2_8=2.0)),
        ("$0.5{\\times}q_{2,8}$", CostConfig(q2_8=0.5)),
        ("$2{\\times}q_{3,9}$",  CostConfig(q3_9=2.0)),
        ("$5{\\times}q_{3,9}$",  CostConfig(q3_9=5.0)),
    ]

    weight_results = []
    for label, cfg in weight_configs:
        print(f"\n  Running: {label} ...")
        r = run_one_config(label, cfg, x_data, d_traj, x_base,
                           cpss_params, learn_cfg, dt, T)
        weight_results.append(r)
        print(f"    Fear: {r['fear_red_pct']}%  Pow: {r['pow_deficit']}  "
              f"EMS: {r['ems_deficit']} (base: {r['ems_deficit_base']})  "
              f"P1@5%: {r['p1_exploit_5']}%")

    # ── Part B: Control gain (beta) sensitivity ──
    print("\n" + "=" * 70)
    print("  PART B: Control Gain (β) Sensitivity")
    print("=" * 70)

    beta_configs = [
        ("Default $\\beta$", CostConfig(), None),
        ("$2{\\times}\\beta_1$", CostConfig(), {"beta1": 1.0, "beta10": 0.6}),
        ("$0.5{\\times}\\beta_1$", CostConfig(), {"beta1": 0.25, "beta10": 0.15}),
        ("$2{\\times}\\beta_9$", CostConfig(), {"beta9": 1.0}),
    ]

    beta_results = []
    for label, cfg, betas in beta_configs:
        print(f"\n  Running: {label} ...")
        r = run_with_beta(label, cfg, x_data, cpss_params, learn_cfg, dt, T,
                          beta_overrides=betas)
        beta_results.append(r)
        print(f"    Fear: {r['fear_red_pct']}%  Pow: {r['pow_deficit']}  "
              f"EMS: {r['ems_deficit']} (base: {r['ems_deficit_base']})  "
              f"P1@5%: {r['p1_exploit_5']}%")

    # ── Part C: Learning rate sensitivity ──
    print("\n" + "=" * 70)
    print("  PART C: Learning Rate Sensitivity")
    print("=" * 70)

    lr_configs = [
        ("Default $\\alpha$", CostConfig(), LearningConfig()),
        ("$2{\\times}\\alpha_c$", CostConfig(),
         LearningConfig(alpha_c=np.array([1.0, 1.0, 1.0]))),
        ("$0.5{\\times}\\alpha_c$", CostConfig(),
         LearningConfig(alpha_c=np.array([0.25, 0.25, 0.25]))),
        ("$\\Delta{=}3$", CostConfig(),
         LearningConfig(window_length=3)),
        ("$\\Delta{=}12$", CostConfig(),
         LearningConfig(window_length=12)),
    ]

    lr_results = []
    for label, cfg, lcfg in lr_configs:
        print(f"\n  Running: {label} ...")
        res = run_actor_critic(
            x0=x_data[0], d_traj=d_traj, T_steps=T,
            cpss_params=cpss_params, cost_config=cfg,
            learn_config=lcfg, dt=dt, seed=42,
        )
        x_ctrl = res.x_traj[:T+1]
        fear_base = float(np.mean(x_base[:T+1, IX1]))
        fear_ctrl = float(np.mean(x_ctrl[:, IX1]))
        fear_red = (fear_base - fear_ctrl) / max(fear_base, 1e-8) * 100
        pow_def = float(np.mean(1.0 - x_ctrl[:, IX8]))
        ems_def = float(np.mean(1.0 - x_ctrl[:, IX9]))

        r = {"label": label, "fear_red_pct": round(fear_red, 1),
             "pow_deficit": round(pow_def, 3), "ems_deficit": round(ems_def, 3)}
        lr_results.append(r)
        print(f"    Fear: {r['fear_red_pct']}%  Pow: {r['pow_deficit']}  "
              f"EMS: {r['ems_deficit']}")

    # Save all results
    all_results = {
        "weight_sensitivity": weight_results,
        "beta_sensitivity": beta_results,
        "lr_sensitivity": lr_results,
    }
    out_path = os.path.join(ARTIFACTS, "sensitivity_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {out_path}")


if __name__ == "__main__":
    main()
