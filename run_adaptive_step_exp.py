import argparse
import numpy as np
import matplotlib.pyplot as plt
from vs_ns_periodic_mrSAV_solver import vs_mrSAV_Vorticity_Stream_Periodic_Solver as vs_mrSAV_solver

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.family": "Times New Roman",
    "font.size": 12,
    "xtick.direction": "in",
    "ytick.direction": "in",
})

np.random.seed(1)

def force_term(X, Y, t, m):
    return m * np.cos(m * Y)

def initial_streamfunction(x, y, nu, m, eps):
    base_flow = np.zeros_like(x)
    k_max = 10
    k1_vals = np.arange(-k_max, k_max + 1)
    k2_vals = np.arange(-k_max, k_max + 1)
    k1_grid, k2_grid = np.meshgrid(k1_vals, k2_vals, indexing="ij")
    k_mod = np.sqrt(k1_grid**2 + k2_grid**2)
    mask = k_mod <= 10
    k1_valid = k1_grid[mask]
    k2_valid = k2_grid[mask]
    k_mod_valid = k_mod[mask]
    perturbation = np.zeros_like(x, dtype=np.float64)
    for k1, k2, k_abs in zip(k1_valid, k2_valid, k_mod_valid):
        if k_abs < 1e-10:
            continue
        term = (1 / k_abs**3) * (
            np.cos(k1 * x) * np.cos(k2 * y)
            + np.sin(k1 * x) * np.cos(k2 * y)
            + np.cos(k1 * x) * np.sin(k2 * y)
            + np.sin(k1 * x) * np.sin(k2 * y)
        )
        perturbation += term
    return base_flow + eps * perturbation


def run(nu, m, eps, t_ini, gam, s_domain, discrete_num, t_period):
    xn = np.linspace(s_domain[0], s_domain[2], discrete_num[0] + 1)
    yn = np.linspace(s_domain[1], s_domain[3], discrete_num[1] + 1)
    X, Y = np.meshgrid(xn, yn)

    _force = lambda X, Y, t: force_term(X, Y, t, m)

    # --- warm-up to t_ini ---
    initial_phi = initial_streamfunction(X[:-1, :-1], Y[:-1, :-1], nu, m, eps)
    solver_init = vs_mrSAV_solver(nu, gam, s_domain, discrete_num, initial_phi, _force, "ETD_mrGSAV_MS2_b")
    solver_init.Omega0 = np.pad(solver_init.stream2velocity(initial_phi)[0], ((0, 1), (0, 1)))[:-1, :-1]
    solver_init.solve_fix_step((0, t_ini), 0.0025)
    initial_vorticity = np.pad(solver_init.Omega[-1], ((0, 1), (0, 1)))

    # --- adaptive solver ---
    solver_2h = vs_mrSAV_solver(nu, 1000, s_domain, discrete_num, initial_vorticity, _force, "ETD_mrGSAV_MS2_b")
    solver_2h.solve_adaptive_step(
        t_period, 1e-5, 1e-2,
        snapshot=np.linspace(0, t_period[1], 651),
        compute_ref_err=False, rho=0.95, rtol=1e-4, rtol_q=1e-4, r=1/2,
    )

    solver_2h_cpu_time = solver_2h.cpu_time
    solver_2h.solve_adaptive_step(
        t_period, 1e-5, 1e-2,
        snapshot=np.linspace(0, t_period[1], 651),
        compute_ref_err=True, rho=0.95, rtol=1e-4, rtol_q=1e-4, r=1/2,
    )

    # --- fixed-step solvers ---
    solver_fix1 = vs_mrSAV_solver(nu, gam, s_domain, discrete_num, initial_vorticity, _force, "ETD_mrGSAV_MS2_b")
    solver_fix1.solve_fix_step(t_period, 0.005)

    solver_fix2 = vs_mrSAV_solver(nu, gam, s_domain, discrete_num, initial_vorticity, _force, "ETD_mrGSAV_MS2_b")
    solver_fix2.solve_fix_step(t_period, 0.002)

    solver_fix3 = vs_mrSAV_solver(nu, gam, s_domain, discrete_num, initial_vorticity, _force, "ETD_mrGSAV_MS2_b")
    solver_fix3.solve_fix_step(t_period, 0.00125)

    solver_fix4 = vs_mrSAV_solver(nu, gam, s_domain, discrete_num, initial_vorticity, _force, "ETDRK4")
    solver_fix4.solve_fix_step(t_period, 0.00125)

    solver_fix5 = vs_mrSAV_solver(nu, gam, s_domain, discrete_num, initial_vorticity, _force, "ETDRK4")
    solver_fix5.solve_fix_step(t_period, 0.001)

    # --- reference errors ---
    def rel_err(a, b):
        out = np.zeros(len(a))
        for i in range(len(a)):
            out[i] = np.sqrt(
                solver_2h.inner_product(a[i] - b[i], a[i] - b[i])
                / solver_2h.inner_product(b[i], b[i])
            )
        return out

    ref_error_fix  = rel_err(solver_fix1.Omega, solver_fix4.Omega[::4]) * 2
    ref_error_fix2 = rel_err(solver_fix2.Omega, solver_fix5.Omega[::2]) * 2
    ref_error_fix3 = rel_err(solver_fix3.Omega, solver_fix4.Omega) * 2

    # --- plot ---
    t_end = t_period[1]
    xticks = np.linspace(0, t_end, 9)

    fig = plt.figure(figsize=(12, 10))

    ax1 = plt.subplot2grid((6, 6), (0, 0), colspan=2, rowspan=2)
    ax1.plot(solver_2h.tn, solver_2h.tau, color="C0", marker='o', markersize=5, markeredgewidth=0,      markevery=200, linewidth=1, label=r"$\tau =$ Ada. steps")
    ax1.plot(np.linspace(0, t_end, t_end+1), np.ones(t_end+1)*0.005,   color='C1', marker="*", markeredgewidth=0, markersize=5, linewidth=1, label=r"$\tau = 0.005$")
    ax1.plot(np.linspace(0, t_end, t_end+1), np.ones(t_end+1)*0.002,  color='C2', marker="d", markeredgewidth=0, markersize=4, linewidth=1, label=r"$\tau = 0.002$")
    ax1.plot(np.linspace(0, t_end, t_end+1), np.ones(t_end+1)*0.00125, color='C3', marker="s", markeredgewidth=0, markersize=4, linewidth=1, label=r"$\tau = 0.00125$")
    ax1.hlines(1e-2, 0, t_end, colors='k', linestyles='dashed', alpha=0.5, linewidth=1)
    ax1.hlines(1e-5, 0, t_end, colors='k', linestyles='dashed', alpha=0.5, linewidth=1)
    ax1.set_yscale("log"); ax1.set_xlim(0, t_end); ax1.set_xticks(xticks)
    ax1.set_xlabel(r"$t$", fontsize=15); ax1.set_ylabel(r"Step sizes", fontsize=12)
    ax1.grid(alpha=0.5, linestyle='dashed', linewidth=0.5)

    ax2 = plt.subplot2grid((6, 6), (0, 2), colspan=2, rowspan=2)
    ax2.plot(solver_2h.tn,    solver_2h_cpu_time - solver_2h_cpu_time[1],         color="C0", marker='o', markersize=5, markeredgewidth=0, markevery=300, linewidth=1)
    ax2.plot(solver_fix1.tn,  solver_fix1.cpu_time - solver_fix1.cpu_time[1],     color="C1", marker="*", markeredgewidth=0, markersize=5, markevery=400, linewidth=1)
    ax2.plot(solver_fix2.tn,  solver_fix2.cpu_time - solver_fix2.cpu_time[1],     color="C2", marker="d", markeredgewidth=0, markersize=5, markevery=800, linewidth=1)
    ax2.plot(solver_fix3.tn,  solver_fix3.cpu_time - solver_fix3.cpu_time[1],     color="C3", marker="s", markeredgewidth=0, markersize=5, markevery=800, linewidth=1)
    _cpu_max = max(
        (solver_2h_cpu_time - solver_2h_cpu_time[1]).max(),
        (solver_fix1.cpu_time - solver_fix1.cpu_time[1]).max(),
        (solver_fix2.cpu_time - solver_fix2.cpu_time[1]).max(),
        (solver_fix3.cpu_time - solver_fix3.cpu_time[1]).max(),
    )
    ax2.set_xticks(xticks); ax2.set_xlim(0, t_end); ax2.set_ylim(0, _cpu_max * 1.12)
    ax2.set_xlabel(r"$t$", fontsize=15); ax2.set_ylabel(r"CPU time (s)", fontsize=12)
    ax2.grid(alpha=0.5, linestyle='dashed', linewidth=0.5)

    ax6 = plt.subplot2grid((6, 6), (0, 4), colspan=2, rowspan=2)
    ax6.plot(solver_2h.tn[4:], solver_2h.B_n[4:], linewidth=1, color="C0", marker='o', markersize=5, markeredgewidth=0, markevery=300)
    ax6.set_xticks(xticks); ax6.set_xlim(0, t_end); ax6.set_ylim(1e-4, 1e0)
    ax6.set_xlabel(r"$t$", fontsize=15); ax6.set_ylabel(r"$\Vert B^n \Vert$", fontsize=12)
    ax6.set_yscale("log"); ax6.grid(alpha=0.5, linestyle='dashed', linewidth=0.5)

    ax3 = plt.subplot2grid((6, 6), (2, 0), colspan=2, rowspan=2)
    ax3.plot(solver_2h.tn,   solver_2h.ref_err_p,              marker='o', markersize=5, markeredgewidth=0, markevery=200, linewidth=1)
    ax3.plot(solver_fix1.tn, np.abs(solver_fix1.q - 1), color="C1", marker="*", markeredgewidth=0, markersize=5, markevery=400, linewidth=1)
    ax3.plot(solver_fix2.tn, np.abs(solver_fix2.q - 1), color="C2", marker="d", markeredgewidth=0, markersize=5, markevery=600, linewidth=1)
    ax3.plot(solver_fix3.tn, np.abs(solver_fix3.q - 1), color="C3", marker="s", markeredgewidth=0, markersize=5, markevery=600, linewidth=1)
    ax3.set_ylim(1e-16, 1e0); ax3.hlines(1e-4, 0, t_end, colors='k', linestyles='dashed', alpha=0.5, linewidth=1)
    ax3.set_yscale("log"); ax3.set_xlim(0, t_end)
    ax3.set_xlabel(r"$t$", fontsize=15); ax3.set_ylabel(r"$| r |$", fontsize=12)
    ax3.grid(alpha=0.5, linestyle='dashed', linewidth=0.5)

    ax4 = plt.subplot2grid((6, 6), (2, 2), colspan=2, rowspan=2)
    ax4.plot(solver_2h.tn, solver_2h.rel_err, marker='o', markersize=5, markeredgewidth=0, markevery=300, linewidth=1)
    ax4.set_yscale("log"); ax4.set_xlim(0, t_end)
    ax4.hlines(1e-4, 0, t_end, colors='k', linestyles='dashed', alpha=0.5, linewidth=1)
    ax4.set_ylim(1e-11, 1e-2); ax4.set_yticks([1e-10, 1e-8, 1e-6, 1e-4, 1e-2])
    ax4.set_ylabel(r"Relative $L^2$ error of Vorticity", fontsize=12)
    ax4.set_xlabel(r"$t$", fontsize=15); ax4.grid(alpha=0.5, linestyle='dashed', linewidth=0.5)

    ax5 = plt.subplot2grid((6, 6), (2, 4), colspan=2, rowspan=2)
    ax5.plot(solver_2h.tn,   solver_2h.ref_err,  linewidth=1,  color="C0", marker='o', markersize=5, markeredgewidth=0, markevery=300, linestyle='-')
    ax5.plot(solver_fix1.tn, ref_error_fix,       linewidth=1, color="C1", marker="*", markeredgewidth=0, markersize=5, markevery=400, linestyle='-.')
    ax5.plot(solver_fix2.tn, ref_error_fix2,      linewidth=1, color="C2", marker="d", markeredgewidth=0, markersize=5, markevery=800, linestyle='--')
    ax5.plot(solver_fix3.tn, ref_error_fix3,      linewidth=1, color="C3", marker="s", markeredgewidth=0, markersize=5, markevery=800, linestyle=':')
    ax5.set_yscale("log"); ax5.set_xlim(0, t_end); ax5.set_ylim(1e-11, 1e1)
    ax5.hlines(1e-4, 0, t_end, colors='k', linestyles='dashed', alpha=0.5, linewidth=1)
    ax5.hlines(1e-3, 0, t_end, colors='k', linestyles='dashed', alpha=0.5, linewidth=1)
    ax5.set_yticks([1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e0])
    ax5.set_ylabel(r"Reference relative $L^2$ error of Vorticity", fontsize=12)
    ax5.set_xlabel(r"$t$", fontsize=15); ax5.grid(alpha=0.5, linestyle='dashed', linewidth=0.5)

    for label, ax_sub in zip(['(a)', '(b)', '(c)', '(d)', '(e)'], [ax1, ax2, ax3, ax4, ax5]):
        ax_sub.text(0.5, -0.21, label, transform=ax_sub.transAxes,
                    fontsize=18, fontweight='bold', ha='center', va='center')

    plt.figlegend(loc="lower center", ncol=4, fontsize=15, bbox_to_anchor=[0.5, 0.23])
    plt.tight_layout()
    output=f"adaptive_step_{int(1/nu)}_{m}_{eps}_{t_ini}_2.png"
    plt.savefig(output, bbox_inches='tight', dpi=300)
    print(f"Saved: {output}")


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Adaptive step experiment for mrSAV NS solver")
    # parser.add_argument("--nu",           type=float, default=1/20)
    # parser.add_argument("--m",            type=int,   default=2)
    # parser.add_argument("--eps",          type=float, default=0.1)
    # parser.add_argument("--t_ini",        type=float, default=20.0)
    # parser.add_argument("--gam",          type=float, default=1000.0)
    # parser.add_argument("--s_domain",     type=float, nargs=4, default=[0, 0, 2*np.pi, 2*np.pi], metavar=("x0","y0","x1","y1"))
    # parser.add_argument("--discrete_num", type=int,   nargs=2, default=[64, 64], metavar=("Nx","Ny"))
    # parser.add_argument("--t_period",     type=float, nargs=2, default=[0, 20],  metavar=("t0","t1"))
    # parser.add_argument("--output",       type=str,   default="adaptive_step_result.pdf")
    # args = parser.parse_args()

    eps = [2.5]
    t_inis = [1]
    for t_ini in t_inis:
        for e in eps:
            run(nu=1/20, m=2, eps=e, t_ini=t_ini, gam=1000.0, 
            s_domain=[0, 0, 2*np.pi, 2*np.pi], 
            discrete_num=[128, 128], t_period=(0, 20))
