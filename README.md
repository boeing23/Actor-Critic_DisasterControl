# Alleviating Community Fear in Disasters via Multi-Agent Actor-Critic Reinforcement Learning

Code for the paper:

> **Alleviating Community Fear in Disasters via Multi-Agent Actor-Critic Reinforcement Learning**
> Yashodhan D Hakke, Almuatazbellah Boker, Lamine Mili, Michael R. von Spakovsky, Hoda Eldardiry

## Overview

This repository implements a **3-player non-zero-sum differential game** with **online actor-critic learning** to coordinate disaster-response agents (communication, power infrastructure, emergency services) and minimize community fear during hurricanes.

The framework models disasters as a 10-dimensional **Cyber-Physical-Social System (CPSS)** and learns near-Nash equilibrium control policies via piecewise-stationary actor-critic updates.

## Repository Structure

| File | Description |
|------|-------------|
| `cpss_model.py` | CPSS continuous-time dynamics (control-affine model with logistic gating) |
| `actor_critic_game.py` | 3-player actor-critic learning with critic Bellman residual updates |
| `features.py` | Quadratic monomial basis functions for value-function approximation |
| `fit_params.py` | Parameter identification via least-squares on finite-difference derivatives |
| `diagnostics.py` | Post-hoc diagnostics: Nash gap, PE eigenvalues, saturation analysis |
| `utils.py` | Utility functions (saturation, projection, probing noise) |
| `data.py` | Hurricane Harvey state trajectory (18 time steps) |
| `data_irma.py` | Hurricane Irma state trajectory (13 time steps) |
| `run_experiment.py` | Main experiment script (Harvey) |
| `run_baselines.py` | Baseline controllers (open-loop, constant, proportional, centralized) |
| `run_irma_test.py` | Cross-event validation on Hurricane Irma |
| `run_sensitivity.py` | Sensitivity analysis over cost weights and control gains |

## Data Source

Hurricane time-series data is sourced from:

> Jaber Valinejad, *Cyber-Physical-Social Systems Data Analytics Package*
> https://github.com/Jaber-Valinejad/Cyber-Physical-Social-systems-Data-Analytics-Package

## Requirements

- Python 3.10+
- NumPy
- SciPy
- Matplotlib

## Usage

```bash
# Run the main Harvey experiment
python run_experiment.py

# Run baseline comparisons
python run_baselines.py

# Run Irma cross-validation
python run_irma_test.py

# Run sensitivity analysis
python run_sensitivity.py
```

Results and plots are saved to `artifacts/` and `artifacts_irma/`.
