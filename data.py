"""
Data module for CPSS time series — Hurricane Harvey.

Data sourced from:
  https://github.com/Jaber-Valinejad/Cyber-Physical-Social-systems-Data-Analytics-Package

Loads the 10-dimensional normalized [0,1] state trajectory (18 time samples).

State ordering (1-indexed in comments, 0-indexed in arrays):
  x1  = Fear
  x2  = Information Seeking
  x3  = Flexibility
  x4  = Physical Health
  x5  = Risk Perception
  x6  = Cooperation
  x7  = Learning
  x8  = Power Availability
  x9  = Emergency Services
  x10 = Fake News
"""

import numpy as np
from typing import Tuple

# ── State labels (for plots / logs) ──────────────────────────────────────────
STATE_NAMES: list[str] = [
    "Fear (x1)",
    "Info Seeking (x2)",
    "Flexibility (x3)",
    "Phys. Health (x4)",
    "Risk Percep. (x5)",
    "Cooperation (x6)",
    "Learning (x7)",
    "Power Avail. (x8)",
    "Emergency Svc (x9)",
    "Fake News (x10)",
]

PLAYER_NAMES: list[str] = [
    "P1 (Comm)",
    "P2 (Power)",
    "P3 (EMS)",
]


def load_data() -> Tuple[np.ndarray, float]:
    """Load CPSS state trajectory data.

    Returns
    -------
    x_data : np.ndarray, shape (18, 10)
        State trajectory x[k] for k = 0 … 17.  All values in [0, 1].
    dt : float
        Uniform time-step (= 1.0).
    """
    # Fear x1
    x1 = [0.62, 0.75, 0.87, 0.77, 0.4, 0.42, 0.5, 0.69, 0.75, 0.7,
          1.0, 0.9, 0.8, 0.7, 0.8, 0.81, 1.0, 0.85]
    # Information Seeking x2
    x2 = [0.59, 0.99, 0.51, 0.44, 0.45, 0.38, 0.30, 0.14, 0.16, 0.24,
          0.12, 0.13, 0.08, 0.06, 0.05, 0.04, 0.02, 0.01]
    # Flexibility x3
    x3 = [0.09, 0.01, 0.19, 0.51, 0.4, 0.43, 0.45, 0.47, 0.49, 0.5,
          0.7, 0.46, 0.55, 0.35, 0.3, 0.4, 0.3, 0.5]
    # Physical Health x4
    x4 = [0.7, 0.74, 0.47, 0.32, 0.6, 0.68, 0.65, 0.67, 0.75, 0.65,
          0.63, 0.56, 0.6, 0.9, 0.7, 0.8, 0.4, 0.3]
    # Risk Perception x5
    x5 = [1.0, 0.75, 0.55, 0.83, 0.7, 0.45, 0.3, 0.53, 0.55, 0.5,
          0.4, 0.29, 0.23, 0.55, 0.33, 0.52, 0.28, 0.53]
    # Cooperation x6
    x6 = [0.36, 0.58, 0.75, 0.9, 0.91, 1.0, 0.85, 0.6, 0.6, 0.39,
          0.2, 0.5, 0.42, 0.3, 0.4, 0.6, 0.45, 0.3]
    # Learning x7
    x7 = [0.58, 1.0, 0.57, 0.46, 0.48, 0.4, 0.1, 0.13, 0.28, 0.14,
          0.18, 0.03, 0.015, 0.012, 0.011, 0.12, 0.11, 0.0]
    # Power Availability x8
    x8 = [1.0, 0.9, 0.7, 0.74, 0.75, 0.75, 0.78, 0.82, 0.86, 0.87,
          0.89, 0.91, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98]
    # Emergency Services x9
    x9 = [0.0, 0.42, 0.65, 0.99, 0.28, 0.61, 0.93, 0.73, 0.84, 0.95,
          0.85, 0.74, 0.63, 0.53, 0.42, 0.31, 0.21, 0.10]
    # Fake News x10
    x10 = [0.0, 0.0, 0.0, 0.25, 0.25, 0.75, 0.75, 1.0, 0.0, 0.25,
           0.25, 0.75, 0.25, 0.25, 0.25, 0.25, 0.0, 0.75]

    x_data = np.column_stack([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10])
    dt = 1.0
    return x_data, dt


