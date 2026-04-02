"""
Utility functions for CPSS actor–critic experiment.

Provides: saturation, state projection, finite differences,
probing-noise generation, directory helpers, JSON I/O, logging.
"""

import json
import logging
import os
from typing import Optional

import numpy as np

# ── Saturation / projection ─────────────────────────────────────────────────

def saturate(u: float, lo: float, hi: float) -> float:
    """Saturate a scalar control to [lo, hi]."""
    return float(np.clip(u, lo, hi))


def project_controlled_states(x: np.ndarray) -> np.ndarray:
    """Clip *controlled* states (x1, x8, x9, x10) into [0, 1].

    Indices 0, 7, 8, 9 are the states directly affected by control
    (communication, power, EMS, fake-news).  Other states are left
    unchanged — they should stay in [0, 1] by model construction.
    """
    x_out = x.copy()
    for idx in (0, 7, 8, 9):
        x_out[idx] = np.clip(x_out[idx], 0.0, 1.0)
    return x_out


# ── Finite differences ──────────────────────────────────────────────────────

def finite_differences(x_data: np.ndarray, dt: float) -> np.ndarray:
    """Forward finite-difference approximation of dx/dt.

    Parameters
    ----------
    x_data : (K, n) state trajectory.
    dt     : time-step.

    Returns
    -------
    dxdt : (K-1, n)
    """
    return (x_data[1:] - x_data[:-1]) / dt


# ── Probing (exploration) noise ─────────────────────────────────────────────

# Primes used for incommensurate frequencies (≥ 10 values)
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]


def probing_signal(
    t: float,
    player_idx: int,
    amplitude: float = 0.3,
    num_freq: int = 3,
) -> float:
    """Bounded non-negative probing noise via sin² with incommensurate ω.

    Result ∈ [0, amplitude].

    Frequencies:  ω_k = π √(prime_{offset+k}),  where offset = player_idx * num_freq.
    Using √(prime) ensures pairwise incommensurability ⇒ persistent excitation.
    """
    offset = player_idx * num_freq
    val = 0.0
    for k in range(num_freq):
        omega = np.pi * np.sqrt(_PRIMES[offset + k])
        val += np.sin(omega * t) ** 2
    # Normalise to [0, amplitude]
    return amplitude * val / num_freq


# ── File / directory helpers ────────────────────────────────────────────────

def ensure_dir(path: str) -> None:
    """Create *path* (and parents) if it does not exist."""
    os.makedirs(path, exist_ok=True)


def save_json(data: dict, path: str) -> None:
    """Write a dict to a JSON file (auto-creates parent dirs)."""
    parent = os.path.dirname(path)
    if parent:
        ensure_dir(parent)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)


def _json_default(obj):
    """JSON fallback serialiser for NumPy scalars / arrays."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# ── Logging ─────────────────────────────────────────────────────────────────

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger named ``cpss_experiment``."""
    logger = logging.getLogger("cpss_experiment")
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


