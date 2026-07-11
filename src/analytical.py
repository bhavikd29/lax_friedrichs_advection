"""Exact analytical solution of the 1D linear advection system.

The coupled system

    d(rho)/dt + d(u)/dx = 0
    d(u)/dt   + d(rho)/dx = 0

decouples under the Riemann invariants

    w1 = rho + u ,   w2 = rho - u .

Adding and subtracting the two PDEs gives two independent linear advection
equations

    d(w1)/dt + d(w1)/dx = 0   ->   wave speed c = +1
    d(w2)/dt - d(w2)/dx = 0   ->   wave speed c = -1

so each invariant is simply transported at constant speed without changing
shape:

    w1(x, t) = w1_0(x - t) ,   w2(x, t) = w2_0(x + t).

For the Riemann initial data (a jump at x = 0) each invariant is piecewise
constant, and rho, u are recovered by inverting the transform:

    rho = 0.5 (w1 + w2) ,   u = 0.5 (w1 - w2).

This exact solution is what the Lax-Friedrichs scheme is validated against.
"""

from __future__ import annotations

import numpy as np

# Left/right constant states of the Riemann problem (must match solver.py).
RHO_LEFT, RHO_RIGHT = 0.1, 10.0
U_LEFT, U_RIGHT = 2.0, 1.0


def analytical_solution(x: np.ndarray, t_final: float) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the exact ``(rho, u)`` on grid ``x`` at time ``t_final``.

    Parameters
    ----------
    x:
        Grid node positions to evaluate the solution on.  Passed in explicitly
        so the function has no dependence on any global grid.
    t_final:
        Time at which to evaluate the exact solution.

    Returns
    -------
    rho_exact, u_exact:
        Arrays of the same shape as ``x``.
    """
    x = np.asarray(x, dtype=float)

    # Initial invariant states on the left/right of the discontinuity.
    w1_left, w1_right = RHO_LEFT + U_LEFT, RHO_RIGHT + U_RIGHT
    w2_left, w2_right = RHO_LEFT - U_LEFT, RHO_RIGHT - U_RIGHT

    # w1 travels right at c = +1  ->  value at (x, t) is set by the sign of x - t.
    # w2 travels left  at c = -1  ->  value at (x, t) is set by the sign of x + t.
    w1 = np.where(x - t_final <= 0.0, w1_left, w1_right)
    w2 = np.where(x + t_final <= 0.0, w2_left, w2_right)

    # Recover the primitive variables from the invariants.
    rho_exact = 0.5 * (w1 + w2)
    u_exact = 0.5 * (w1 - w2)
    return rho_exact, u_exact
