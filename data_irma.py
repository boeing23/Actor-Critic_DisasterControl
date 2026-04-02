"""
Data module for CPSS time series — Hurricane Irma.

Data sourced from:
  https://github.com/Jaber-Valinejad/Cyber-Physical-Social-systems-Data-Analytics-Package

Loads the 10-dimensional normalized [0,1] state trajectory (13 time samples).

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


def load_irma_data() -> Tuple[np.ndarray, float]:
    """Load Hurricane Irma CPSS state trajectory data.

    Returns
    -------
    x_data : np.ndarray, shape (13, 10)
        State trajectory x[k] for k = 0 … 12.  All values in [0, 1].
    dt : float
        Uniform time-step (= 1.0).
    """
    # Fear x1
    x1 = [0.9, 0.0, 0.8, 0.9, 0.9, 1.0, 0.7, 0.8, 0.9, 1.0, 1.0, 0.8, 0.9]

    # Information Seeking x2
    x2 = [0.0, 0.0, 0.02, 0.18, 0.47, 0.7, 0.6, 0.64, 0.8, 1.0, 0.57, 0.2, 0.01]

    # Flexibility x3
    x3 = [0.1, 1.0, 0.57, 0.38, 0.43, 0.31, 0.18, 0.0, 0.2, 0.37, 0.35, 0.34, 0.6]

    # Physical Health x4
    x4 = [0.57, 0.58, 0.28, 0.1, 0.05, 0.05, 0.02, 0.13, 0.1, 0.04, 0.07, 0.1, 0.08]

    # Risk Perception x5
    x5 = [0.56, 0.1, 0.4, 0.18, 0.07, 0.1, 0.03, 0.1, 0.08, 0.06, 0.05, 0.06, 0.07]

    # Cooperation x6
    x6 = [0.19, 1.0, 0.5, 0.05, 0.02, 0.02, 0.0, 0.05, 0.04, 0.03, 0.02, 0.03, 0.03]

    # Learning x7
    x7 = [0.55, 0.6, 0.4, 0.25, 0.08, 0.15, 0.1, 0.04, 0.1, 0.05, 0.04, 0.057, 0.0]

    # Power Availability x8
    x8 = [1.0, 0.98, 1.0, 0.97, 0.96, 0.96, 0.95, 0.97, 0.96, 0.95, 0.4, 0.5, 0.65]

    # Emergency Services x9
    x9 = [0.0, 0.061378270736807954, 0.12275654147361591,
          0.18413481221042385, 0.2538730248920794, 0.32361123757373494,
          0.39334945025539053, 0.4958464314817678, 0.6134620081608769,
          0.731077584839986, 0.8491567283505864, 0.967235871861187,
          0.8663730995265174]

    # Fake News x10
    x10 = [0.0, 0.0, 0.0, 0.0, 0.16666666666666666, 0.0,
           0.3333333333333333, 0.5, 0.16666666666666666, 0.0,
           1.0, 0.3333333333333333, 0.16666666666666666]

    x_data = np.column_stack([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10])
    dt = 1.0
    return x_data, dt

