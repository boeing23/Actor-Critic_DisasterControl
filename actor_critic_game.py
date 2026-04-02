"""
3-player non-zero-sum differential game with online actor–critic learning.

Cost functionals (continuous-time):
    J_i = ∫ [ Q_i(x) + R_{ii} u_i² + Σ_{j≠i} R_{ij} u_j² ] dt

Value-function approximation:
    V̂_i(x) = W_{c,i}ᵀ φ(x)

Actor policy:
    û_i(x) = −(1 / 2R_{ii}) g_iᵀ ∇φ(x) W_{a,i}

Critic Bellman residual:
    ε_i = Q_i + Σ_j R_{ij} u_j² + W_{c,i}ᵀ σ
    σ = ∇φᵀ (f + Σ_j g_j u_{exec,j})

Critic update (exact normalisation):
    W_{c,i} ← W_{c,i} − α_c · (σ / (σᵀσ + 1)²) · ε_i · dt

Actor tracking update:
    W_{a,i} ← W_{a,i} − α_a · (W_{a,i} − W_{c,i}) · dt

Piecewise-stationary windowing (A5):
    d(t) is frozen within windows of length Δ;
    weights warm-start across window boundaries.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List

from features import phi_quadratic, grad_phi_quadratic, quadratic_basis_dim
from cpss_model import (
    CPSSParams, STATE_DIM, NUM_PLAYERS,
    drift, g_vectors, compute_PS, compute_Cplus,
    IX1, IX4, IX8, IX9, IX10,
)
from utils import saturate, project_controlled_states, probing_signal


# ═══════════════════════════════════════════════════════════════════════
#  Configuration dataclasses
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CostConfig:
    """Per-player cost weights.

    Q1 = q11·x1² + q1_10·x10²
    Q2 = q21·x1² + q2_8·(1−x8)²
    Q3 = q31·x1² + q3_4·(1−x4)² + q3_9·(1−x9)²

    R ∈ ℝ³ˣ³ : R[i,j] = penalty player i pays for player j's control.
    Default R is diagonal (non-zero-sum coupling togglable).
    """
    q11: float = 1.0
    q1_10: float = 0.5
    q21: float = 0.5
    q2_8: float = 1.0
    q31: float = 0.5
    q3_4: float = 0.5
    q3_9: float = 1.0
    R: np.ndarray = field(
        default_factory=lambda: np.diag([1.0, 1.0, 1.0]).copy()
    )

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items()}
        if isinstance(d.get("R"), np.ndarray):
            d["R"] = d["R"].tolist()
        return d


@dataclass
class LearningConfig:
    """Actor–critic hyper-parameters."""
    alpha_c: np.ndarray = field(
        default_factory=lambda: np.array([0.5, 0.5, 0.5])
    )
    alpha_a: np.ndarray = field(
        default_factory=lambda: np.array([0.1, 0.1, 0.1])
    )
    probe_amplitude: np.ndarray = field(
        default_factory=lambda: np.array([0.3, 0.3, 0.3])
    )
    T_explore: int = 12              # steps with probing on
    probe_enabled: np.ndarray = field(
        default_factory=lambda: np.array([True, True, True])
    )
    window_length: int = 6           # Δ (piecewise-stationary window)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        for k, v in self.__dict__.items():
            d[k] = v.tolist() if isinstance(v, np.ndarray) else v
        return d


# ═══════════════════════════════════════════════════════════════════════
#  Stage costs
# ═══════════════════════════════════════════════════════════════════════

def compute_Q(player: int, x: np.ndarray, cost: CostConfig) -> float:
    """State-dependent running cost Q_i(x)  (0-indexed player)."""
    if player == 0:
        return cost.q11 * x[IX1] ** 2 + cost.q1_10 * x[IX10] ** 2
    if player == 1:
        return cost.q21 * x[IX1] ** 2 + cost.q2_8 * (1.0 - x[IX8]) ** 2
    if player == 2:
        return (
            cost.q31 * x[IX1] ** 2
            + cost.q3_4 * (1.0 - x[IX4]) ** 2
            + cost.q3_9 * (1.0 - x[IX9]) ** 2
        )
    raise ValueError(f"Invalid player index: {player}")


def compute_stage_cost(
    player: int, x: np.ndarray, u_exec: np.ndarray, cost: CostConfig
) -> float:
    """Full instantaneous cost: Q_i + Σ_j R_{ij} u_j²."""
    Q = compute_Q(player, x, cost)
    return Q + sum(
        cost.R[player, j] * u_exec[j] ** 2 for j in range(NUM_PLAYERS)
    )


# ═══════════════════════════════════════════════════════════════════════
#  Actor policy
# ═══════════════════════════════════════════════════════════════════════

def compute_policy(
    x: np.ndarray,
    Wa_i: np.ndarray,
    g_i: np.ndarray,
    R_ii: float,
) -> float:
    """û_i(x) = −(1 / 2R_{ii}) g_iᵀ ∇φ(x) W_{a,i}."""
    gp = grad_phi_quadratic(x)            # (n, p)
    nabla_V = gp @ Wa_i                   # (n,)
    return float(-1.0 / (2.0 * R_ii) * np.dot(g_i, nabla_V))


# ═══════════════════════════════════════════════════════════════════════
#  Weight updates
# ═══════════════════════════════════════════════════════════════════════

def critic_update(
    Wc: np.ndarray,
    sigma: np.ndarray,
    eps: float,
    alpha_c: float,
    dt: float,
) -> np.ndarray:
    """Critic:  W_c ← W_c − α_c (σ / (σᵀσ + 1)²) ε dt."""
    norm_sq_plus1 = float(sigma @ sigma) + 1.0
    return Wc - alpha_c * (sigma / (norm_sq_plus1 ** 2)) * eps * dt


def actor_update(
    Wa: np.ndarray,
    Wc: np.ndarray,
    alpha_a: float,
    dt: float,
) -> np.ndarray:
    """Actor tracking:  W_a ← W_a − α_a (W_a − W_c) dt."""
    return Wa - alpha_a * (Wa - Wc) * dt


# ═══════════════════════════════════════════════════════════════════════
#  Simulation result container
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SimulationResult:
    """Trajectories and diagnostics from a controlled run."""
    x_traj: np.ndarray              # (T+1, n)
    u_traj: np.ndarray              # (T, 3)
    bellman_residuals: np.ndarray   # (T, 3)
    Wc_norms: np.ndarray            # (T+1, 3)
    Wa_norms: np.ndarray            # (T+1, 3)
    stage_costs: np.ndarray         # (T, 3)
    Wc_final: List[np.ndarray] = field(default_factory=list)  # converged critic weights
    Wa_final: List[np.ndarray] = field(default_factory=list)  # converged actor weights


# ═══════════════════════════════════════════════════════════════════════
#  Online actor–critic with windowing  (Algorithm A5)
# ═══════════════════════════════════════════════════════════════════════

def run_actor_critic(
    x0: np.ndarray,
    d_traj: np.ndarray,
    T_steps: int,
    cpss_params: CPSSParams,
    cost_config: CostConfig,
    learn_config: LearningConfig,
    dt: float = 1.0,
    seed: int = 42,
) -> SimulationResult:
    """Run online actor–critic with piecewise-stationary windowing (A5).

    1. Divide [0, T_steps) into windows of length Δ.
    2. Freeze d(t), PS, C⁺ at each window start.
    3. Within each window run AC updates (critic then actor).
    4. Warm-start weights at window boundaries.

    Initialisation (admissible):  W_{a,i}(0) = 0,  W_{c,i}(0) = 0.
    Executed controls:  u_exec = sat(û + probing,  0,  ū).

    Parameters
    ----------
    x0         : (n,) initial state.
    d_traj     : (≥T_steps, 10) exogenous baseline.
    T_steps    : simulation horizon.
    cpss_params, cost_config, learn_config : config objects.
    dt, seed   : time-step and RNG seed.

    Returns
    -------
    SimulationResult
    """
    np.random.seed(seed)

    n = STATE_DIM
    p = quadratic_basis_dim(n)
    Delta = learn_config.window_length
    T_ex = learn_config.T_explore
    n_sub = cpss_params.n_substeps
    dt_sub = dt / n_sub

    # ── admissible initialisation ────────────────────────────────────
    Wc: List[np.ndarray] = [np.zeros(p) for _ in range(NUM_PLAYERS)]
    Wa: List[np.ndarray] = [np.zeros(p) for _ in range(NUM_PLAYERS)]

    g_vecs = g_vectors(cpss_params)

    # ── storage ──────────────────────────────────────────────────────
    x_traj = np.zeros((T_steps + 1, n))
    u_traj = np.zeros((T_steps, NUM_PLAYERS))
    bell_res = np.zeros((T_steps, NUM_PLAYERS))
    Wc_norms = np.zeros((T_steps + 1, NUM_PLAYERS))
    Wa_norms = np.zeros((T_steps + 1, NUM_PLAYERS))
    costs_arr = np.zeros((T_steps, NUM_PLAYERS))

    x_traj[0] = x0.copy()
    for i in range(NUM_PLAYERS):
        Wc_norms[0, i] = np.linalg.norm(Wc[i])
        Wa_norms[0, i] = np.linalg.norm(Wa[i])

    num_windows = (T_steps + Delta - 1) // Delta

    for w in range(num_windows):
        t_start = w * Delta
        t_end = min((w + 1) * Delta, T_steps)

        # ── freeze exogenous at window start ─────────────────────────
        d_idx = min(t_start, d_traj.shape[0] - 1)
        d_frozen = d_traj[d_idx].copy()
        PS_frozen = compute_PS(d_traj[d_idx])
        Cp_frozen = compute_Cplus(d_traj[d_idx])

        for k in range(t_start, t_end):
            x_k = x_traj[k]

            # ── 1. compute actor policies ────────────────────────────
            uhat = np.zeros(NUM_PLAYERS)
            for i in range(NUM_PLAYERS):
                R_ii = float(cost_config.R[i, i])
                uhat[i] = compute_policy(x_k, Wa[i], g_vecs[i], R_ii)

            # ── 2. exploration + saturation → u_exec ─────────────────
            u_exec = np.zeros(NUM_PLAYERS)
            for i in range(NUM_PLAYERS):
                noise = 0.0
                if k < T_ex and learn_config.probe_enabled[i]:
                    noise = probing_signal(
                        float(k), i, float(learn_config.probe_amplitude[i])
                    )
                u_exec[i] = saturate(
                    uhat[i] + noise, 0.0, float(cpss_params.ubar[i])
                )
            u_traj[k] = u_exec

            # ── 3. drift & dynamics at x_k (for σ / ε) ──────────────
            f_val = drift(
                x_k, d_frozen, cpss_params,
                PS_override=PS_frozen, Cplus_override=Cp_frozen,
            )
            dynamics = f_val.copy()
            for j in range(NUM_PLAYERS):
                dynamics += g_vecs[j] * u_exec[j]

            # ── 4. σ = ∇φᵀ (f + Σ g_j u_j) ─────────────────────────
            gp = grad_phi_quadratic(x_k)       # (n, p)
            sigma = gp.T @ dynamics              # (p,)

            # ── 5. Bellman residual & weight updates ─────────────────
            #    compute all ε first, then update Wc, then Wa
            eps = np.zeros(NUM_PLAYERS)
            for i in range(NUM_PLAYERS):
                Q_i = compute_Q(i, x_k, cost_config)
                u_pen = sum(
                    cost_config.R[i, j] * u_exec[j] ** 2
                    for j in range(NUM_PLAYERS)
                )
                eps[i] = Q_i + u_pen + float(Wc[i] @ sigma)
                bell_res[k, i] = eps[i]
                costs_arr[k, i] = Q_i + u_pen

            for i in range(NUM_PLAYERS):
                Wc[i] = critic_update(
                    Wc[i], sigma, eps[i],
                    float(learn_config.alpha_c[i]), dt,
                )
            for i in range(NUM_PLAYERS):
                Wa[i] = actor_update(
                    Wa[i], Wc[i],
                    float(learn_config.alpha_a[i]), dt,
                )

            # ── 6. state propagation  (Euler sub-stepping) ───────────
            x_sub = x_k.copy()
            for _s in range(n_sub):
                f_sub = drift(
                    x_sub, d_frozen, cpss_params,
                    PS_override=PS_frozen, Cplus_override=Cp_frozen,
                )
                dyn_sub = f_sub.copy()
                for j in range(NUM_PLAYERS):
                    dyn_sub += g_vecs[j] * u_exec[j]
                x_sub = x_sub + dyn_sub * dt_sub

            # project controlled states to [0, 1]
            x_traj[k + 1] = project_controlled_states(x_sub)

            for i in range(NUM_PLAYERS):
                Wc_norms[k + 1, i] = np.linalg.norm(Wc[i])
                Wa_norms[k + 1, i] = np.linalg.norm(Wa[i])

    return SimulationResult(
        x_traj=x_traj,
        u_traj=u_traj,
        bellman_residuals=bell_res,
        Wc_norms=Wc_norms,
        Wa_norms=Wa_norms,
        stage_costs=costs_arr,
        Wc_final=[w.copy() for w in Wc],
        Wa_final=[w.copy() for w in Wa],
    )


# ═══════════════════════════════════════════════════════════════════════
#  Baseline (uncontrolled) simulation
# ═══════════════════════════════════════════════════════════════════════

def simulate_baseline(
    x0: np.ndarray,
    d_traj: np.ndarray,
    T_steps: int,
    cpss_params: CPSSParams,
    dt: float = 1.0,
) -> np.ndarray:
    """Forward simulate with u ≡ 0 (open-loop baseline).

    Returns
    -------
    x_traj : (T_steps+1, n)
    """
    n = STATE_DIM
    n_sub = cpss_params.n_substeps
    dt_sub = dt / n_sub

    x_traj = np.zeros((T_steps + 1, n))
    x_traj[0] = x0.copy()

    for k in range(T_steps):
        d_idx = min(k, d_traj.shape[0] - 1)
        d_k = d_traj[d_idx]

        x_sub = x_traj[k].copy()
        for _s in range(n_sub):
            f_sub = drift(x_sub, d_k, cpss_params)
            x_sub = x_sub + f_sub * dt_sub

        x_traj[k + 1] = np.clip(x_sub, 0.0, 1.0)

    return x_traj

