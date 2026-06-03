import h5py
import numpy as np
import argparse

np.random.seed(1)

from vs_ns_periodic_mrSAV_solver import vs_mrSAV_Vorticity_Stream_Periodic_Solver as sav_vs_solver

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, choices=["fix", "adaptive"], default="fix",
                    help="solver mode: fix (fixed step) or adaptive (variable step)")
parser.add_argument("--Re", type=float, default=40,              help="Reynolds number")
parser.add_argument("--m", type=int, default=4,                  help="m parameter")
parser.add_argument("--M", type=str, default="ETD_mrGSAV_MS2_b", help="solver method")
parser.add_argument("--eps", type=float, default=3,              help="perturbation strength")
parser.add_argument("--gamma", type=float, default=1000)

# fix mode
parser.add_argument("--tau", type=float, default=0.001,  help="fixed time step size (fix mode)")

# adaptive mode
parser.add_argument("--tau-min", type=float, default=1e-5, help="min time step (adaptive mode)")
parser.add_argument("--tau-max", type=float, default=1e-2, help="max time step (adaptive mode)")
parser.add_argument("--rtol",    type=float, default=1e-4,    help="relative tolerance (adaptive mode)")
parser.add_argument("--rtol-q",  type=float, default=1e-4,  help="relative tolerance for q (adaptive mode)")

args = parser.parse_args()

mode = args.mode
Re = args.Re
m = args.m
M = args.M
eps = args.eps
ga = args.gamma
re_tag = f"{Re:g}"
eps_tag = f"{eps:g}"

Primitive_force = lambda X, Y, t: ( -np.sin(m * Y), 0)
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
discrete_num = [64, 64]
xn = np.linspace(s_domain[0], s_domain[2], discrete_num[0] + 1)
yn = np.linspace(s_domain[1], s_domain[3], discrete_num[1] + 1)
X, Y = np.meshgrid(xn, yn)

t_period = (0, 10000)
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
    h5_path = f"./data/ns_{M}_bursting_{re_tag}_{m}_{eps_tag}_{args.tau:g}.h5"
else:
    h5_path = f"./data/ns_{M}_bursting_{re_tag}_{m}_{eps_tag}_vs.h5"

# ---- main solve ----
if mode == "fix":
    total_steps = int(round((t_period[1] - t_period[0]) / args.tau))
    snapshot_stride = max(1, int(round(0.1 / args.tau)))
    snapshot_indices = np.arange(0, total_steps + 1, snapshot_stride, dtype=np.int64)
    if snapshot_indices[-1] != total_steps:
        snapshot_indices = np.append(snapshot_indices, total_steps)
    snapshots = t_period[0] + args.tau * snapshot_indices
else:
    snapshots = np.linspace(t_period[0], t_period[1], int((t_period[1] - t_period[0]) * 10) + 1)

print(f"\n=== Running {mode} solve: t={t_period} ===")

if mode == "fix":
    etdms_solver.solve_fix_step(t_period, args.tau, snapshot=snapshots)
else:
    etdms_solver.solve_adaptive_step(
        t_period, args.tau_min, args.tau_max, snapshots,
        rtol=args.rtol, rtol_q=args.rtol_q,
    )

with h5py.File(h5_path, "w") as f:
    f.attrs["mode"] = mode
    f.attrs["Re"] = Re
    f.attrs["m"] = m
    f.attrs["method"] = M
    f.attrs["eps"] = eps
    f.attrs["gamma"] = ga
    f.attrs["nu"] = nu
    f.attrs["s_domain"] = s_domain
    f.attrs["discrete_num"] = discrete_num
    f.attrs["t_period"] = t_period

    f["Omega"] = etdms_solver.Omega
    f["tn_s"] = etdms_solver.tn_s
    f["q"] = etdms_solver.q
    f["tn"] = etdms_solver.tn
    f["Mx"] = etdms_solver.Mx
    f["Energy"] = etdms_solver.Energy
    f["Energy_rate"] = etdms_solver.Energy_rate
    f["Enstrophy"] = etdms_solver.Enstrophy
    f["Enstrophy_rate"] = etdms_solver.Enstrophy_rate
    f["Palinstrophy"] = etdms_solver.Palinstrophy
    f["CPU_time"] = etdms_solver.cpu_time

    if mode == "fix":
        f.attrs["tau"] = args.tau
    else:
        f.attrs["tau_min"] = args.tau_min
        f.attrs["tau_max"] = args.tau_max
        f.attrs["rtol"] = args.rtol
        f.attrs["rtol_q"] = args.rtol_q
        f["tau"] = etdms_solver.tau

print(f"  -> saved to '{h5_path}'")
