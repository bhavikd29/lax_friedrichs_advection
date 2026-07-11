"""Quantified convergence (grid-refinement) study for the Lax-Friedrichs scheme.

The scheme is run at increasing spatial resolutions, all to the same fixed final
time and with the same fixed Courant number.  At each resolution the error
against the exact solution is measured in discrete, grid-normalised L1 and L2
norms, and the observed order of accuracy is estimated from the slope of a
log-log fit of error versus grid spacing dx.

Discrete norms (normalised by the grid so values are comparable across
resolutions):

    ||e||_1 = dx * sum_i |e_i|
    ||e||_2 = sqrt( dx * sum_i e_i^2 )

Lax-Friedrichs is a first-order scheme, so for a *smooth* solution we would
expect slope ~ 1.  This problem, however, is a Riemann problem whose solution
contains discontinuities; convergence near a shock is typically degraded
(observed order below 1), which the study reports honestly rather than hiding.

Run as a script to reproduce the figure::

    python src/convergence.py
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Sequence

import numpy as np

# Support both "python src/convergence.py" (src/ on sys.path) and
# "from src.convergence import ..." style imports.
try:  # pragma: no cover - trivial import shim
    from solver import lax_friedrichs
    from analytical import analytical_solution
except ImportError:  # pragma: no cover
    from src.solver import lax_friedrichs
    from src.analytical import analytical_solution

# Default resolutions and run parameters for the study.
DEFAULT_NX = (100, 200, 400, 800, 1600)
DEFAULT_T_FINAL = 0.4
DEFAULT_CFL = 0.8


def l1_norm(error: np.ndarray, dx: float) -> float:
    """Grid-normalised discrete L1 norm ``dx * sum |error|``."""
    return float(dx * np.sum(np.abs(error)))


def l2_norm(error: np.ndarray, dx: float) -> float:
    """Grid-normalised discrete L2 norm ``sqrt(dx * sum error**2)``."""
    return float(np.sqrt(dx * np.sum(error ** 2)))


class ConvergenceResult(NamedTuple):
    """Tabulated output of :func:`run_convergence`.

    All error arrays are ordered to match ``nx`` / ``dx``.
    """

    nx: np.ndarray
    dx: np.ndarray
    rho_l1: np.ndarray
    rho_l2: np.ndarray
    u_l1: np.ndarray
    u_l2: np.ndarray


def run_convergence(
    nx_list: Sequence[int] = DEFAULT_NX,
    t_final: float = DEFAULT_T_FINAL,
    cfl: float = DEFAULT_CFL,
) -> ConvergenceResult:
    """Run the solver at each resolution and measure errors against the exact solution.

    For each ``nx`` the solver is advanced to (approximately) ``t_final``.  The
    exact solution is then evaluated at the *actual* reached time so the two are
    compared at an identical instant, removing a spurious error contribution
    from the ``nt = floor(t_final / dt)`` time rounding.

    Parameters
    ----------
    nx_list:
        Interior-node counts to test, e.g. ``(100, 200, 400, 800, 1600)``.
    t_final:
        Target final time (same for every resolution).
    cfl:
        Fixed Courant number (same for every resolution).

    Returns
    -------
    ConvergenceResult
        Arrays of ``nx``, ``dx`` and the L1/L2 errors of ``rho`` and ``u``.
    """
    nx_arr, dx_arr = [], []
    rho_l1, rho_l2, u_l1, u_l2 = [], [], [], []

    for nx in nx_list:
        result = lax_friedrichs(t_final, nx=nx, cfl=cfl, store_history=False)
        # result.times is the actual reached time when store_history is False.
        rho_exact, u_exact = analytical_solution(result.x, result.times)

        rho_err = result.rho - rho_exact
        u_err = result.u - u_exact

        nx_arr.append(nx)
        dx_arr.append(result.dx)
        rho_l1.append(l1_norm(rho_err, result.dx))
        rho_l2.append(l2_norm(rho_err, result.dx))
        u_l1.append(l1_norm(u_err, result.dx))
        u_l2.append(l2_norm(u_err, result.dx))

    return ConvergenceResult(
        nx=np.array(nx_arr),
        dx=np.array(dx_arr),
        rho_l1=np.array(rho_l1),
        rho_l2=np.array(rho_l2),
        u_l1=np.array(u_l1),
        u_l2=np.array(u_l2),
    )


def estimate_order(dx: np.ndarray, error: np.ndarray) -> float:
    """Estimate the order of accuracy as the slope of log(error) vs log(dx).

    A least-squares line is fit to ``(log dx, log error)``; its slope is the
    observed convergence order ``p`` in ``error ~ dx**p``.
    """
    slope, _ = np.polyfit(np.log(dx), np.log(error), 1)
    return float(slope)


def plot_convergence(
    result: ConvergenceResult,
    save_path: str | Path | None = None,
    show: bool = False,
):
    """Log-log plot of error vs dx with a reference slope-1 line.

    Plots the L1 and L2 errors of both ``rho`` and ``u``, overlays a first-order
    (slope-1) reference line, and annotates each series with its fitted order.

    Parameters
    ----------
    result:
        Output of :func:`run_convergence`.
    save_path:
        If given, the figure is saved there (parent directories are created).
    show:
        If ``True``, call ``plt.show()`` (useful when run as a script).

    Returns
    -------
    (fig, ax):
        The Matplotlib figure and axes, so a notebook can display them.
    """
    import matplotlib.pyplot as plt

    dx = result.dx
    series = [
        ("rho  L1", result.rho_l1, "o", "tab:blue"),
        ("rho  L2", result.rho_l2, "s", "tab:cyan"),
        ("u    L1", result.u_l1, "^", "tab:red"),
        ("u    L2", result.u_l2, "d", "tab:orange"),
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, err, marker, color in series:
        p = estimate_order(dx, err)
        ax.loglog(dx, err, marker=marker, color=color, lw=1.5,
                  label=f"{label}  (order = {p:.2f})")

    # Reference slope-1 (first-order) line, anchored to the finest-grid L2 point.
    ref = result.rho_l2[-1] * (dx / dx[-1]) ** 1.0
    ax.loglog(dx, ref, "k--", lw=1.2, label="reference slope 1")

    ax.set_xlabel(r"grid spacing $\Delta x$")
    ax.set_ylabel(r"error  ($\|\cdot\|_1$, $\|\cdot\|_2$)")
    ax.set_title("Lax-Friedrichs convergence: error vs grid spacing")
    ax.grid(True, which="both", ls="--", lw=0.6, alpha=0.7)
    ax.legend()
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    return fig, ax


def main() -> None:
    """Run the default study, print the table + fitted orders, and save the figure."""
    result = run_convergence()

    print(f"Convergence study: t_final = {DEFAULT_T_FINAL}, CFL = {DEFAULT_CFL}\n")
    header = f"{'nx':>6} {'dx':>10} {'rho L1':>12} {'rho L2':>12} {'u L1':>12} {'u L2':>12}"
    print(header)
    print("-" * len(header))
    for i in range(len(result.nx)):
        print(f"{result.nx[i]:>6} {result.dx[i]:>10.5f} "
              f"{result.rho_l1[i]:>12.4e} {result.rho_l2[i]:>12.4e} "
              f"{result.u_l1[i]:>12.4e} {result.u_l2[i]:>12.4e}")

    print("\nEstimated order of accuracy (fitted log-log slope):")
    print(f"  rho:  L1 = {estimate_order(result.dx, result.rho_l1):.3f}, "
          f"L2 = {estimate_order(result.dx, result.rho_l2):.3f}")
    print(f"  u:    L1 = {estimate_order(result.dx, result.u_l1):.3f}, "
          f"L2 = {estimate_order(result.dx, result.u_l2):.3f}")

    figures_dir = Path(__file__).resolve().parent.parent / "figures"
    save_path = figures_dir / "convergence.png"
    plot_convergence(result, save_path=save_path)
    print(f"\nSaved convergence plot to: {save_path}")


if __name__ == "__main__":
    main()
