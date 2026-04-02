"""
CPSS (Cyber-Physical-Social System) continuous-time dynamics.

Implements the ValinejadMili-style control-affine model with logistic gating
for a 10-dimensional state vector.

State ordering (0-indexed):
    0: x1  – Fear                (diffusion)
    1: x2  – Information Seeking (diffusion)
    2: x3  – Flexibility         (diffusion)
    3: x4  – Physical Health     (gated relaxation)
    4: x5  – Risk Perception     (gated relaxation)
    5: x6  – Cooperation         (gated relaxation)
    6: x7  – Learning            (gated relaxation)
    7: x8  – Power Availability  (physical/cyber baseline)
    8: x9  – Emergency Services  (physical/cyber baseline)
    9: x10 – Fake News           (physical/cyber baseline)

Control-affine structure:
    ẋ = f(x, d) + g₁ u₁ + g₂ u₂ + g₃ u₃
    g₁ → (−β₁, 0, …, 0, −β₁₀)   communication reduces fear & fake news
    g₂ → (0, …, 0, +β₈, 0, 0)    power-supply effort
    g₃ → (0, …, 0, 0, +β₉, 0)    emergency-services effort

All exogenous signals (PS, C⁺, d₈, d₉, d₁₀) are configurable.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ── Dimensions ───────────────────────────────────────────────────────────────
STATE_DIM: int = 10
NUM_PLAYERS: int = 3

# ── Readable indices ─────────────────────────────────────────────────────────
IX1, IX2, IX3, IX4, IX5, IX6, IX7, IX8, IX9, IX10 = range(10)


# ═══════════════════════════════════════════════════════════════════════
#  Parameter container
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class CPSSParams:
    """All tuneable CPSS model parameters.

    Weight arrays (iota*, w*) are stored as *raw logits*; they are
    passed through ``softmax_weights`` inside the target functions so
    they always form proper convex combinations → targets ∈ [0, 1].
    """

    # ── logistic gating ψ(z) = 1/(1+exp(-κ(z−ζ))) ──────────────────
    kappa: float = 5.0
    zeta: float = 0.5

    # ── diffusion rates (x1, x2, x3) ────────────────────────────────
    alpha1: float = 1.0
    alpha2: float = 1.0
    alpha3: float = 1.0

    # ── nonlinearity for Γ mixing ────────────────────────────────────
    eta1: float = 0.5
    eta2: float = 0.5
    eta3: float = 0.5

    # ── x1hat target weights (raw logits, 5 drivers) ────────────────
    #    drivers: ψ(PS), ψ(x10), ψ_c(x8), ψ(x5), ψ_c(C⁺)
    iota1: np.ndarray = field(default_factory=lambda: np.zeros(5))

    # ── x2hat target weights (4 drivers) ─────────────────────────────
    #    drivers: ψ(x1), ψ(x5), ψ_c(x8), ψ_c(C⁺)
    iota2: np.ndarray = field(default_factory=lambda: np.zeros(4))

    # ── x3hat target weights (3 drivers) ─────────────────────────────
    #    drivers: ψ(x6), ψ(x7), ψ(x4)
    iota3: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # ── gated-relaxation rates (x4 … x7) ────────────────────────────
    alpha4: float = 1.0
    alpha5: float = 1.0
    alpha6: float = 1.0
    alpha7: float = 1.0

    # ── equilibrium target weights for x4…x7 ─────────────────────────
    w4: np.ndarray = field(default_factory=lambda: np.zeros(3))   # ψ(x9), ψ(x6), ψ_c(x1)
    w5: np.ndarray = field(default_factory=lambda: np.zeros(3))   # ψ(x1), ψ(x10), ψ_c(x8)
    w6: np.ndarray = field(default_factory=lambda: np.zeros(4))   # ψ(x3), ψ(x7), ψ_c(x1), ψ(x8)
    w7: np.ndarray = field(default_factory=lambda: np.zeros(3))   # ψ(x2), ψ(x6), ψ_c(x5)

    # ── physical / cyber relaxation ──────────────────────────────────
    gamma8: float = 5.0
    gamma9: float = 5.0
    gamma10: float = 5.0

    # ── control-affine coupling ──────────────────────────────────────
    beta1: float = 0.5      # u1 → x1 (reduce fear)
    beta8: float = 0.5      # u2 → x8 (increase power)
    beta9: float = 0.5      # u3 → x9 (increase EMS)
    beta10: float = 0.3     # u1 → x10 (reduce fake news)

    # ── control bounds ───────────────────────────────────────────────
    ubar: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0]))

    # ── integration sub-steps (Euler) ────────────────────────────────
    n_substeps: int = 10

    # ─── serialisation ───────────────────────────────────────────────
    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        for key, val in self.__dict__.items():
            d[key] = val.tolist() if isinstance(val, np.ndarray) else val
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CPSSParams":
        p = cls()
        for key, val in d.items():
            if hasattr(p, key):
                cur = getattr(p, key)
                if isinstance(cur, np.ndarray):
                    setattr(p, key, np.asarray(val, dtype=float))
                else:
                    setattr(p, key, type(cur)(val))
        return p


# ═══════════════════════════════════════════════════════════════════════
#  Logistic gating
# ═══════════════════════════════════════════════════════════════════════

def psi(z: np.ndarray | float, kappa: float, zeta: float) -> np.ndarray | float:
    """Logistic gate  ψ(z) = 1 / (1 + exp(−κ(z − ζ)))."""
    arg = np.clip(-kappa * (np.asarray(z, dtype=float) - zeta), -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(arg))


def psi_c(z: np.ndarray | float, kappa: float, zeta: float) -> np.ndarray | float:
    """Complement gate  ψ_c(z) = 1 − ψ(z)."""
    return 1.0 - psi(z, kappa, zeta)


# ═══════════════════════════════════════════════════════════════════════
#  Softmax weight normalisation
# ═══════════════════════════════════════════════════════════════════════

def softmax_weights(w: np.ndarray) -> np.ndarray:
    """Softmax on raw logits → non-negative weights summing to 1."""
    e = np.exp(w - np.max(w))
    return e / e.sum()


# ═══════════════════════════════════════════════════════════════════════
#  Exogenous proxy signals
# ═══════════════════════════════════════════════════════════════════════

def compute_PS(x: np.ndarray, override: Optional[float] = None) -> float:
    """Perceived-severity proxy  PS = 0.5·(1−x8) + 0.5·x10.

    Configurable: pass ``override`` to use a fixed value.
    """
    if override is not None:
        return float(override)
    return float(0.5 * (1.0 - x[IX8]) + 0.5 * x[IX10])


def compute_Cplus(x: np.ndarray, override: Optional[float] = None) -> float:
    """Positive-communication proxy  C⁺ = 1 − x10.

    Configurable: pass ``override`` to use a fixed value.
    """
    if override is not None:
        return float(override)
    return float(1.0 - x[IX10])


# ═══════════════════════════════════════════════════════════════════════
#  Diffusion targets  x̂₁, x̂₂, x̂₃
# ═══════════════════════════════════════════════════════════════════════

def compute_x1hat(
    x: np.ndarray, PS: float, Cplus: float, params: CPSSParams
) -> float:
    """Fear target  x̂₁  (ValinejadMili formula).

    x̂₁ = Σ_k  ω_k · driver_k        (convex combination of logistic transforms)

    Drivers:
        ψ(PS)       – perceived severity increases fear target
        ψ(x10)      – fake news increases fear target
        ψ_c(x8)     – low power increases fear target
        ψ(x5)       – risk perception increases fear target
        ψ_c(C⁺)     – lack of positive communication increases fear target

    NOTE: the weights ω_k are softmax(ι₁), guaranteeing x̂₁ ∈ [0, 1].
    """
    k, z = params.kappa, params.zeta
    drivers = np.array([
        psi(PS, k, z),
        psi(x[IX10], k, z),
        psi_c(x[IX8], k, z),
        psi(x[IX5], k, z),
        psi_c(Cplus, k, z),
    ])
    w = softmax_weights(params.iota1)
    return float(w @ drivers)


def compute_x2hat(
    x: np.ndarray, PS: float, Cplus: float, params: CPSSParams
) -> float:
    """Information-seeking target x̂₂  (learnable softmax-weighted logistic).

    Drivers: ψ(x1), ψ(x5), ψ_c(x8), ψ_c(C⁺).
    """
    k, z = params.kappa, params.zeta
    drivers = np.array([
        psi(x[IX1], k, z),
        psi(x[IX5], k, z),
        psi_c(x[IX8], k, z),
        psi_c(Cplus, k, z),
    ])
    w = softmax_weights(params.iota2)
    return float(w @ drivers)


def compute_x3hat(
    x: np.ndarray, PS: float, Cplus: float, params: CPSSParams
) -> float:
    """Flexibility target x̂₃  (learnable softmax-weighted logistic).

    Drivers: ψ(x6), ψ(x7), ψ(x4).
    """
    k, z = params.kappa, params.zeta
    drivers = np.array([
        psi(x[IX6], k, z),
        psi(x[IX7], k, z),
        psi(x[IX4], k, z),
    ])
    w = softmax_weights(params.iota3)
    return float(w @ drivers)


# ═══════════════════════════════════════════════════════════════════════
#  Diffusion mixing Γ
# ═══════════════════════════════════════════════════════════════════════

def gamma_mixing(xhat: float, x_val: float, x5: float, eta: float) -> float:
    """Γ(θ̂, θ) with risk-perception modulation.

    Γ = η [ x5·(1−(1−θ)(1−θ̂)) + (1−x5)·θ̂ θ ] + (1−η)·θ̂
    """
    term_risk = x5 * (1.0 - (1.0 - x_val) * (1.0 - xhat))
    term_norisk = (1.0 - x5) * xhat * x_val
    return eta * (term_risk + term_norisk) + (1.0 - eta) * xhat


# ═══════════════════════════════════════════════════════════════════════
#  Equilibrium targets for gated-relaxation states x4 … x7
# ═══════════════════════════════════════════════════════════════════════

def compute_eq4(x: np.ndarray, params: CPSSParams) -> float:
    """Physical-health equilibrium.  Drivers: ψ(x9), ψ(x6), ψ_c(x1)."""
    k, z = params.kappa, params.zeta
    d = np.array([psi(x[IX9], k, z), psi(x[IX6], k, z), psi_c(x[IX1], k, z)])
    return float(softmax_weights(params.w4) @ d)


def compute_eq5(x: np.ndarray, params: CPSSParams) -> float:
    """Risk-perception equilibrium.  Drivers: ψ(x1), ψ(x10), ψ_c(x8)."""
    k, z = params.kappa, params.zeta
    d = np.array([psi(x[IX1], k, z), psi(x[IX10], k, z), psi_c(x[IX8], k, z)])
    return float(softmax_weights(params.w5) @ d)


def compute_eq6(x: np.ndarray, params: CPSSParams) -> float:
    """Cooperation equilibrium.  Drivers: ψ(x3), ψ(x7), ψ_c(x1), ψ(x8)."""
    k, z = params.kappa, params.zeta
    d = np.array([
        psi(x[IX3], k, z), psi(x[IX7], k, z),
        psi_c(x[IX1], k, z), psi(x[IX8], k, z),
    ])
    return float(softmax_weights(params.w6) @ d)


def compute_eq7(x: np.ndarray, params: CPSSParams) -> float:
    """Learning equilibrium.  Drivers: ψ(x2), ψ(x6), ψ_c(x5)."""
    k, z = params.kappa, params.zeta
    d = np.array([psi(x[IX2], k, z), psi(x[IX6], k, z), psi_c(x[IX5], k, z)])
    return float(softmax_weights(params.w7) @ d)


# ═══════════════════════════════════════════════════════════════════════
#  Drift f(x, d)  (uncontrolled)
# ═══════════════════════════════════════════════════════════════════════

def drift(
    x: np.ndarray,
    d: np.ndarray,
    params: CPSSParams,
    PS_override: Optional[float] = None,
    Cplus_override: Optional[float] = None,
) -> np.ndarray:
    """Compute drift vector f(x) ∈ ℝ¹⁰ (no control).

    Parameters
    ----------
    x  : (10,) current state.
    d  : (10,) exogenous driving trajectory (only indices 7,8,9 used).
    params : CPSSParams.
    PS_override, Cplus_override :
        If given, override the proxy computation.

    Returns
    -------
    f : (10,) drift.
    """
    f = np.zeros(STATE_DIM)

    PS = compute_PS(x, PS_override)
    Cplus = compute_Cplus(x, Cplus_override)

    # ── diffusion states x1, x2, x3 ─────────────────────────────────
    x1hat = compute_x1hat(x, PS, Cplus, params)
    x2hat = compute_x2hat(x, PS, Cplus, params)
    x3hat = compute_x3hat(x, PS, Cplus, params)

    f[IX1] = params.alpha1 * (gamma_mixing(x1hat, x[IX1], x[IX5], params.eta1) - x[IX1])
    f[IX2] = params.alpha2 * (gamma_mixing(x2hat, x[IX2], x[IX5], params.eta2) - x[IX2])
    f[IX3] = params.alpha3 * (gamma_mixing(x3hat, x[IX3], x[IX5], params.eta3) - x[IX3])

    # ── gated relaxation x4 … x7 ────────────────────────────────────
    f[IX4] = params.alpha4 * (compute_eq4(x, params) - x[IX4])
    f[IX5] = params.alpha5 * (compute_eq5(x, params) - x[IX5])
    f[IX6] = params.alpha6 * (compute_eq6(x, params) - x[IX6])
    f[IX7] = params.alpha7 * (compute_eq7(x, params) - x[IX7])

    # ── physical / cyber baseline x8, x9, x10 ───────────────────────
    f[IX8] = params.gamma8 * (d[IX8] - x[IX8])
    f[IX9] = params.gamma9 * (d[IX9] - x[IX9])
    f[IX10] = params.gamma10 * (d[IX10] - x[IX10])

    return f


# ═══════════════════════════════════════════════════════════════════════
#  Control influence vectors  g_i
# ═══════════════════════════════════════════════════════════════════════

def g_vectors(params: CPSSParams) -> List[np.ndarray]:
    """Return g₁, g₂, g₃ ∈ ℝ¹⁰.

    g₁ (comm)  :  −β₁ on x1,   −β₁₀ on x10
    g₂ (power) :  +β₈ on x8
    g₃ (EMS)   :  +β₉ on x9
    """
    g1 = np.zeros(STATE_DIM)
    g1[IX1] = -params.beta1
    g1[IX10] = -params.beta10

    g2 = np.zeros(STATE_DIM)
    g2[IX8] = params.beta8

    g3 = np.zeros(STATE_DIM)
    g3[IX9] = params.beta9

    return [g1, g2, g3]


# ═══════════════════════════════════════════════════════════════════════
#  d-trajectory construction (from data)
# ═══════════════════════════════════════════════════════════════════════

def compute_d_traj(
    x_data: np.ndarray,
    dt: float,
    params: CPSSParams,
) -> np.ndarray:
    """Derive exogenous driver trajectory d(t) from observed data.

    For states x8, x9, x10 the model is  ẋ = γ(d − x).  Given measured
    x and ẋ ≈ Δx / dt we can back-solve:

        d_i[k] = x_i[k] + (Δx_i[k] / dt) / γ_i     for k < K−1
        d_i[K−1] = d_i[K−2]                           extrapolate last

    This guarantees that the baseline (u = 0) Euler simulation
    reconstructs the data *exactly* at the coarse time-step.

    For all other states (0–6) d is just x_data (unused by drift,
    but kept for index consistency).
    """
    K, n = x_data.shape
    d_traj = x_data.copy()
    dxdt = (x_data[1:] - x_data[:-1]) / dt            # (K−1, n)

    gammas = [params.gamma8, params.gamma9, params.gamma10]
    for local_i, state_idx in enumerate([IX8, IX9, IX10]):
        g = gammas[local_i]
        for k in range(K - 1):
            d_traj[k, state_idx] = x_data[k, state_idx] + dxdt[k, state_idx] / g
        d_traj[K - 1, state_idx] = d_traj[K - 2, state_idx]

    return d_traj


