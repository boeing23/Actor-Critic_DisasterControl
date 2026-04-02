"""
Basis functions φ(x) and Jacobian ∇φ(x) for value-function approximation.

**Quadratic monomial basis** (required):
    φ(x) = [1,  x_1, …, x_n,  x_1², x_1 x_2, …, x_n²]
    Dimension:  p = 1 + n + n(n+1)/2.
    For n = 10:  p = 1 + 10 + 55 = 66.

Convention
----------
∇φ(x) has shape **(n, p)** so that
    ∇V = ∇φ @ W  ∈ ℝⁿ    (gradient of value function)
    σ  = ∇φᵀ @ f  ∈ ℝᵖ    (basis-projected dynamics)

**RBF basis** (optional, kept for experiments):
    φ_k(x) = exp(−‖x − c_k‖² / (2 σ_k²))
"""

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
#  Quadratic monomials
# ═══════════════════════════════════════════════════════════════════════

def quadratic_basis_dim(n: int) -> int:
    """Return dimensionality *p* of the quadratic monomial basis."""
    return 1 + n + n * (n + 1) // 2


def phi_quadratic(x: np.ndarray) -> np.ndarray:
    """Evaluate quadratic monomial basis.

    Parameters
    ----------
    x : (n,)

    Returns
    -------
    phi : (p,)
    """
    n = x.shape[0]
    p = quadratic_basis_dim(n)
    phi = np.empty(p)

    # --- constant ---
    phi[0] = 1.0

    # --- linear ---
    phi[1: n + 1] = x

    # --- quadratic x_i x_j  (i ≤ j) ---
    idx = n + 1
    for i in range(n):
        for j in range(i, n):
            phi[idx] = x[i] * x[j]
            idx += 1

    return phi


def grad_phi_quadratic(x: np.ndarray) -> np.ndarray:
    """Analytically compute the Jacobian ∇φ(x).

    Parameters
    ----------
    x : (n,)

    Returns
    -------
    grad : (n, p)   — entry [m, k] = ∂φ_k / ∂x_m
    """
    n = x.shape[0]
    p = quadratic_basis_dim(n)
    grad = np.zeros((n, p))

    # ∂(1)/∂x_m = 0  →  already zero

    # ∂(x_i)/∂x_m = δ_{im}
    for m in range(n):
        grad[m, 1 + m] = 1.0

    # ∂(x_i x_j)/∂x_m
    idx = n + 1
    for i in range(n):
        for j in range(i, n):
            if i == j:
                grad[i, idx] = 2.0 * x[i]
            else:
                grad[i, idx] = x[j]
                grad[j, idx] = x[i]
            idx += 1

    return grad


# ═══════════════════════════════════════════════════════════════════════
#  Radial-basis functions (optional)
# ═══════════════════════════════════════════════════════════════════════

def phi_rbf(
    x: np.ndarray,
    centers: np.ndarray,
    widths: np.ndarray,
) -> np.ndarray:
    """RBF basis.  ``centers`` (p, n), ``widths`` (p,)  →  φ (p,)."""
    diff = x[np.newaxis, :] - centers          # (p, n)
    return np.exp(-np.sum(diff ** 2, axis=1) / (2.0 * widths ** 2))


def grad_phi_rbf(
    x: np.ndarray,
    centers: np.ndarray,
    widths: np.ndarray,
) -> np.ndarray:
    """Jacobian of RBF basis.  Shape (n, p)."""
    phi_val = phi_rbf(x, centers, widths)      # (p,)
    diff = x[np.newaxis, :] - centers           # (p, n)
    # ∂φ_k/∂x_m = φ_k · (c_{km} − x_m) / σ_k²
    #            = −φ_k · diff[k, m] / σ_k²
    grad_pxn = -phi_val[:, np.newaxis] * diff / (widths[:, np.newaxis] ** 2)
    return grad_pxn.T  # (n, p)


