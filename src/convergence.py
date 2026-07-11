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

Run as a script to print the error table and show the plot::

    python src/convergence.py
"""

import numpy as np

from solver import lax_friedrichs
from analytical import analytical_solution

# Default resolutions and run parameters for the study.
DEFAULT_NX = (100, 200, 400, 800, 1600)
DEFAULT_T_FINAL = 0.4
DEFAULT_CFL = 0.8


def l1_norm(error, dx):
    """Grid-normalised discrete L1 norm ``dx * sum |error|``."""
    return dx * np.sum(np.abs(error))


def l2_norm(error, dx):
    """Grid-normalised discrete L2 norm ``sqrt(dx * sum error**2)``."""
    return np.sqrt(dx * np.sum(error ** 2))


def run_convergence(nx_list=DEFAULT_NX, t_final=DEFAULT_T_FINAL, cfl=DEFAULT_CFL):
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
    A plain tuple of numpy arrays ``(nx, dx, rho_l1, rho_l2, u_l1, u_l2)``,
    all ordered to match one another.  Unpack it at the call site.
    """
    nx_arr, dx_arr = [], []
    rho_l1, rho_l2, u_l1, u_l2 = [], [], [], []

    for nx in nx_list:
        # store_history=False keeps only the final state (all this study needs).
        # times here is the actual reached time, so compare the exact solution
        # at that same instant.
        x, dx, dt, times, rho, u = lax_friedrichs(t_final, nx=nx, cfl=cfl,
                                                  store_history=False)
        rho_exact, u_exact = analytical_solution(x, times)

        rho_err = rho - rho_exact
        u_err = u - u_exact

        nx_arr.append(nx)
        dx_arr.append(dx)
        rho_l1.append(l1_norm(rho_err, dx))
        rho_l2.append(l2_norm(rho_err, dx))
        u_l1.append(l1_norm(u_err, dx))
        u_l2.append(l2_norm(u_err, dx))

    return (np.array(nx_arr), np.array(dx_arr),
            np.array(rho_l1), np.array(rho_l2),
            np.array(u_l1), np.array(u_l2))


def estimate_order(dx, error):
    """Estimate the order of accuracy as the slope of log(error) vs log(dx).

    A least-squares line is fit to ``(log dx, log error)``; its slope is the
    observed convergence order ``p`` in ``error ~ dx**p``.
    """
    slope, intercept = np.polyfit(np.log(dx), np.log(error), 1)
    return slope


def plot_convergence(dx, rho_l1, rho_l2, u_l1, u_l2):
    """Log-log plot of error vs dx with a reference slope-1 line.

    Plots the L1 and L2 errors of both ``rho`` and ``u``, overlays a first-order
    (slope-1) reference line, and annotates each series with its fitted order.
    The arrays are the ones returned by :func:`run_convergence`.
    """
    import matplotlib.pyplot as plt

    series = [
        ("rho  L1", rho_l1, "o", "tab:blue"),
        ("rho  L2", rho_l2, "s", "tab:cyan"),
        ("u    L1", u_l1, "^", "tab:red"),
        ("u    L2", u_l2, "d", "tab:orange"),
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, err, marker, color in series:
        p = estimate_order(dx, err)
        ax.loglog(dx, err, marker=marker, color=color, lw=1.5,
                  label=f"{label}  (order = {p:.2f})")

    # Reference slope-1 (first-order) line, anchored to the finest-grid L2 point.
    ref = rho_l2[-1] * (dx / dx[-1]) ** 1.0
    ax.loglog(dx, ref, "k--", lw=1.2, label="reference slope 1")

    ax.set_xlabel(r"grid spacing $\Delta x$")
    ax.set_ylabel(r"error  ($\|\cdot\|_1$, $\|\cdot\|_2$)")
    ax.set_title("Lax-Friedrichs convergence: error vs grid spacing")
    ax.grid(True, which="both", ls="--", lw=0.6, alpha=0.7)
    ax.legend()
    fig.tight_layout()
    plt.show()

    # The figure is only displayed here, not written to disk.  A saved copy of
    # this plot is kept in the repository at figures/convergence.png.


def main():
    """Run the default study, print the table + fitted orders, and show the plot."""
    nx, dx, rho_l1, rho_l2, u_l1, u_l2 = run_convergence()

    print(f"Convergence study: t_final = {DEFAULT_T_FINAL}, CFL = {DEFAULT_CFL}\n")
    header = f"{'nx':>6} {'dx':>10} {'rho L1':>12} {'rho L2':>12} {'u L1':>12} {'u L2':>12}"
    print(header)
    print("-" * len(header))
    for i in range(len(nx)):
        print(f"{nx[i]:>6} {dx[i]:>10.5f} "
              f"{rho_l1[i]:>12.4e} {rho_l2[i]:>12.4e} "
              f"{u_l1[i]:>12.4e} {u_l2[i]:>12.4e}")

    print("\nEstimated order of accuracy (fitted log-log slope):")
    print(f"  rho:  L1 = {estimate_order(dx, rho_l1):.3f}, "
          f"L2 = {estimate_order(dx, rho_l2):.3f}")
    print(f"  u:    L1 = {estimate_order(dx, u_l1):.3f}, "
          f"L2 = {estimate_order(dx, u_l2):.3f}")

    plot_convergence(dx, rho_l1, rho_l2, u_l1, u_l2)


if __name__ == "__main__":
    main()
