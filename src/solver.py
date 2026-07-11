"""Lax-Friedrichs solver for the 1D linear advection system.

The governing equations are the coupled, linear, hyperbolic system

    d(rho)/dt + d(u)/dx = 0     (continuity)
    d(u)/dt   + d(rho)/dx = 0   (momentum)

on the domain x in [x_min, x_max], with a Riemann (discontinuous) initial
condition and Neumann (zero-gradient) boundary conditions at both ends.

Writing U = (rho, u) and the flux F(U) = (u, rho), the system reads
U_t + F(U)_x = 0.  The Lax-Friedrichs update for an interior node i is

    U_i^{n+1} = 0.5 * (U_{i+1}^n + U_{i-1}^n)
                - dt / (2 dx) * (F_{i+1}^n - F_{i-1}^n).

Component-wise this becomes

    rho_i^{n+1} = 0.5 (rho_{i+1} + rho_{i-1}) - a (u_{i+1}   - u_{i-1})
    u_i^{n+1}   = 0.5 (u_{i+1}   + u_{i-1})   - a (rho_{i+1} - rho_{i-1})

with coefficient a = dt / (2 dx).  This module implements exactly that scheme
using vectorised array slicing (mathematically identical to an averaging-matrix
formulation, but clearer and faster).

Note on grid handling: unlike the original notebook, nothing here relies on a
module-level global grid.  The resolution ``nx`` is a parameter and every
function builds (or is handed) its own grid, which is what makes the
grid-refinement convergence study possible.
"""

from __future__ import annotations

from typing import NamedTuple, Union

import numpy as np


class LaxFriedrichsResult(NamedTuple):
    """Result container returned by :func:`lax_friedrichs`.

    Attributes
    ----------
    x:
        Grid node positions, shape ``(nx + 2,)``.
    dx:
        Uniform grid spacing.
    dt:
        Time step used (``cfl * dx``).
    times:
        With ``store_history=True`` an array of the ``nt + 1`` sampled times;
        with ``store_history=False`` the single reached final time (a float).
    rho, u:
        Density and velocity fields.  Shape ``(nx + 2, nt + 1)`` (space x time)
        when the history is stored, or ``(nx + 2,)`` (final state only) when it
        is not.
    """

    x: np.ndarray
    dx: float
    dt: float
    times: Union[np.ndarray, float]
    rho: np.ndarray
    u: np.ndarray

# Physical domain.  Fixed for this problem, but exposed as defaults so callers
# may override them if desired.
X_MIN = -1.0
X_MAX = 1.0

# Riemann initial condition: left/right constant states separated by a jump at
# x = 0.
RHO_LEFT, RHO_RIGHT = 0.1, 10.0
U_LEFT, U_RIGHT = 2.0, 1.0


def build_grid(nx: int, x_min: float = X_MIN, x_max: float = X_MAX) -> tuple[np.ndarray, float]:
    """Build the spatial grid used by the solver.

    The grid has ``nx`` interior nodes plus two boundary (ghost) nodes, i.e.
    ``nx + 2`` points in total, matching the interior/boundary split used by the
    Neumann boundary conditions.

    Parameters
    ----------
    nx:
        Number of interior grid nodes.
    x_min, x_max:
        Domain endpoints.

    Returns
    -------
    x:
        Array of ``nx + 2`` node positions.
    dx:
        Uniform grid spacing ``x[1] - x[0]``.
    """
    x = np.linspace(x_min, x_max, nx + 2)
    dx = float(x[1] - x[0])
    return x, dx


def initial_condition(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the Riemann initial states ``(rho0, u0)`` sampled on ``x``.

    The initial data is a single discontinuity at ``x = 0``:
    ``rho = 0.1, u = 2.0`` for ``x <= 0`` and ``rho = 10.0, u = 1.0`` for
    ``x > 0``.
    """
    rho0 = np.where(x <= 0.0, RHO_LEFT, RHO_RIGHT)
    u0 = np.where(x <= 0.0, U_LEFT, U_RIGHT)
    return rho0, u0


def lax_friedrichs(
    t_final: float,
    nx: int = 1000,
    cfl: float = 0.4,
    store_history: bool = True,
) -> LaxFriedrichsResult:
    """Integrate the advection system to ``t_final`` with the Lax-Friedrichs scheme.

    The time step is chosen as ``dt = cfl * dx``.  Because both characteristic
    wave speeds satisfy ``|c| = 1`` for this system, the Courant number is
    ``C = |c| * dt / dx = cfl``; the scheme is stable for ``C <= 1`` (see the
    notebook / README for the discussion of the CFL condition).

    Parameters
    ----------
    t_final:
        Physical end time of the simulation.
    nx:
        Number of interior grid nodes (grid resolution).
    cfl:
        Courant number ``C = |c| dt / dx``.  With ``|c| = 1`` this equals
        ``dt / dx``.
    store_history:
        If ``True`` (default) the full space-time history is retained, which is
        needed for the animation and multi-time comparison plots.  If ``False``
        only the final state is kept, which is far cheaper in memory and is all
        the convergence study needs.

    Returns
    -------
    LaxFriedrichsResult
        Named tuple with fields ``x``, ``dx``, ``dt``, ``times``, ``rho`` and
        ``u``.  When ``store_history`` is ``True`` the ``rho``/``u`` arrays have
        shape ``(nx + 2, nt + 1)`` (space x time) and ``times`` has length
        ``nt + 1``; when ``False`` they have shape ``(nx + 2,)`` (final state
        only) and ``times`` is the single reached time.
    """
    x, dx = build_grid(nx)
    dt = cfl * dx
    nt = int(t_final / dt)
    a = dt / (2.0 * dx)

    rho0, u0 = initial_condition(x)

    if store_history:
        rho = np.zeros((nx + 2, nt + 1))
        u = np.zeros((nx + 2, nt + 1))
        rho[:, 0] = rho0
        u[:, 0] = u0
        for n in range(nt):
            r, uu = rho[:, n], u[:, n]
            rho[1:-1, n + 1] = 0.5 * (r[2:] + r[:-2]) - a * (uu[2:] - uu[:-2])
            u[1:-1, n + 1] = 0.5 * (uu[2:] + uu[:-2]) - a * (r[2:] - r[:-2])
            _apply_neumann_bc(rho[:, n + 1])
            _apply_neumann_bc(u[:, n + 1])
        times = np.arange(nt + 1) * dt
        return LaxFriedrichsResult(x=x, dx=dx, dt=dt, times=times, rho=rho, u=u)

    # Memory-light path: keep only the current state and march it forward.
    rho = rho0.astype(float).copy()
    u = u0.astype(float).copy()
    for _ in range(nt):
        rho_new = rho.copy()
        u_new = u.copy()
        rho_new[1:-1] = 0.5 * (rho[2:] + rho[:-2]) - a * (u[2:] - u[:-2])
        u_new[1:-1] = 0.5 * (u[2:] + u[:-2]) - a * (rho[2:] - rho[:-2])
        _apply_neumann_bc(rho_new)
        _apply_neumann_bc(u_new)
        rho, u = rho_new, u_new
    return LaxFriedrichsResult(x=x, dx=dx, dt=dt, times=nt * dt, rho=rho, u=u)


def _apply_neumann_bc(field: np.ndarray) -> None:
    """Apply zero-gradient (Neumann) boundary conditions in place.

    Copies the first/last interior node value onto the corresponding boundary
    node, so that the one-sided difference at the domain edge sees a zero
    gradient.
    """
    field[0] = field[1]
    field[-1] = field[-2]
