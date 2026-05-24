import h5py
import numpy as np
import argparse

np.random.seed(1)

from vs_ns_periodic_mrSAV_solver import vs_mrSAV_Vorticity_Stream_Periodic_Solver as sav_vs_solver

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, choices=["fix", "adaptive"], default="fix",
                    help="solver mode: fix (fixed step) or adaptive (variable step)")
parser.add_argument("--Re", type=float, default=50, help="Reynolds number")
parser.add_argument("--m", type=int, default=2, help="m parameter")
parser.add_argument("--M", type=str, default="ETD_mrGSAV_MS2_b", help="solver method")
parser.add_argument("--eps", type=float, default=2.5, help="perturbation strength")
parser.add_argument("--t_split", type=float, default=1000, help="Time interval size for splitting simulation")
parser.add_argument("--gamma", type=float, default=1000)

# fix mode
parser.add_argument("--tau", type=float, default=0.001, help="fixed time step size (fix mode)")

# adaptive mode
parser.add_argument("--tau-min", type=float, default=1e-5, help="min time step (adaptive mode)")
parser.add_argument("--tau-max", type=float, default=1e-2, help="max time step (adaptive mode)")
parser.add_argument("--rtol", type=float, default=1e-4, help="relative tolerance (adaptive mode)")
parser.add_argument("--rtol-q", type=float, default=1e-4, help="relative tolerance for q (adaptive mode)")

args = parser.parse_args()

mode = args.mode
Re = args.Re
m = args.m
M = args.M
eps = args.eps
t_split = args.t_split
ga = args.gamma

force_term = lambda X, Y, t: m * np.cos(m * Y)


def initial_streamfunction(x: np.ndarray, y: np.ndarray, nu: float, m: float, eps: float) -> np.ndarray:
    base_flow = (1.0 / (nu * m**3)) * np.cos(m * y)

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
        term = (1 / (k_abs**3)) * (
            1 * np.cos(k1 * x) * np.cos(k2 * y)
            + 1 * np.sin(k1 * x) * np.cos(k2 * y)
            + 1 * np.cos(k1 * x) * np.sin(k2 * y)
            + 1 * np.sin(k1 * x) * np.sin(k2 * y)
        )
        perturbation += term

    phi = 0 * base_flow + eps * perturbation
    return phi


s_domain = (0, 0, 2 * np.pi, 2 * np.pi)
discrete_num = [128, 128]
xn = np.linspace(s_domain[0], s_domain[2], discrete_num[0] + 1)
yn = np.linspace(s_domain[1], s_domain[3], discrete_num[1] + 1)
X, Y = np.meshgrid(xn, yn)

t_period = (0, 1000)
nu = 1 / Re

# ---- warmup: build initial vorticity ----
initial_phi = initial_streamfunction(X[:-1, :-1], Y[:-1, :-1], nu, m, eps)
solver_init = sav_vs_solver(nu, ga, s_domain, discrete_num, initial_phi, force_term, "ETD_mrGSAV_MS2_b")
initial_vorticity = np.pad((solver_init.stream2velocity(initial_phi))[0], ((0, 1), (0, 1)))
solver_init.Omega0 = initial_vorticity[:-1, :-1]
solver_init.solve_fix_step((0, 1), 0.0025)
initial_vorticity = np.pad(solver_init.Omega[-1], ((0, 1), (0, 1)))

etdms_solver = sav_vs_solver(nu, ga, s_domain, discrete_num, initial_vorticity, force_term, M)

# ---- output path ----
if mode == "fix":
    gamma_suffix = f"_g_{int(ga)}" if ga != 1000 else ""
    h5_path = f"./data/ns_{M}_bursting_{Re}_{m}_{eps}_{args.tau}{gamma_suffix}.h5"
else:
    h5_path = f"./data/ns_{M}_bursting_{Re}_{m}_{eps}_vs.h5"

# ---- main loop ----
if mode == "fix":
    # split into equal-length intervals
    t_intervals = []
    t = t_period[0]
    while t < t_period[1]:
        t_next = min(t + t_split, t_period[1])
        t_intervals.append((t, t_next))
        t = t_next
else:
    # adaptive mode: dynamic intervals based on actual end time
    t_intervals = None

if mode == "fix":
    for idx, (t_start, t_end) in enumerate(t_intervals):
        snapshots = np.linspace(t_start, t_end, int((t_end - t_start) * 10) + 1)

        print(f"\n=== Interval {idx + 1}/{len(t_intervals)}: t=[{t_start}, {t_end}] ===")

        etdms_solver.solve_fix_step((t_start, t_end), args.tau, snapshot=snapshots)
        final_omega = etdms_solver.Omega[-1].copy()
        final_q = etdms_solver.q[-1]

        group_name = f"t_{t_start}_{t_end}"
        file_mode = "w" if idx == 0 else "a"

        with h5py.File(h5_path, file_mode) as f:
            group = f.create_group(group_name)
            group["Omega"] = etdms_solver.Omega
            group["tn_s"] = etdms_solver.tn_s
            group["q"] = etdms_solver.q
            group["tn"] = etdms_solver.tn
            group["Mx"] = etdms_solver.Mx
            group["Energy"] = etdms_solver.Energy
            group["Enstrophy"] = etdms_solver.Enstrophy
            group["Palinstrophy"] = etdms_solver.Palinstrophy
            group["CPU_time"] = etdms_solver.cpu_time

        print(f"  -> saved to group '{group_name}'")

        etdms_solver.Omega0 = final_omega
        etdms_solver.q0 = final_q

else:
    t_current = t_period[0]
    idx = 0
    while t_current < t_period[1]:
        t_target = min(t_current + args.t_split, t_period[1])

        snapshots = np.linspace(t_current, t_target, int((t_target - t_current) * 10) + 1)
        snapshots = np.unique(np.concatenate([[t_current], snapshots, [t_target]]))

        print(f"\n=== Interval {idx + 1}: t=[{t_current}, {t_target}] ===")

        etdms_solver.solve_adaptive_step(
            (t_current, t_target), args.tau_min, args.tau_max, snapshots,
            rtol=args.rtol, rtol_q=args.rtol_q,
        )

        t_actual_end = etdms_solver.tn[-1]
        final_omega = etdms_solver.Omega[-1].copy()
        final_q = etdms_solver.q[-1]

        group_name = f"t_{t_current}_{t_actual_end}"
        file_mode = "w" if idx == 0 else "a"

        with h5py.File(h5_path, file_mode) as f:
            group = f.create_group(group_name)
            group["Omega"] = etdms_solver.Omega
            group["tn_s"] = etdms_solver.tn_s
            group["q"] = etdms_solver.q
            group["tn"] = etdms_solver.tn
            group["tau"] = etdms_solver.tau
            group["Mx"] = etdms_solver.Mx
            group["Energy"] = etdms_solver.Energy
            group["Enstrophy"] = etdms_solver.Enstrophy
            group["Palinstrophy"] = etdms_solver.Palinstrophy
            group["CPU_time"] = etdms_solver.cpu_time

        print(f"  -> saved to group '{group_name}', actual end: {t_actual_end:.6f}")

        t_current = t_actual_end
        idx += 1
