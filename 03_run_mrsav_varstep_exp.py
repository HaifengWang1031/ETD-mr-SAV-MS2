from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from vs_ns_periodic_mrSAV_solver import (
    vs_mrSAV_Vorticity_Stream_Periodic_Solver as vs_mrSAV_solver,
)


def make_force_term(m: float):
    def force_term(X, Y, t):
        return m * np.cos(m * Y)

    return force_term


def initial_streamfunction(x: np.ndarray, y: np.ndarray, nu: float, m: float, eps: float) -> np.ndarray:
    base_flow = -(1.0 / (nu * m**3)) * np.cos(m * y)

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
            np.cos(k1 * x) * np.cos(k2 * y)
            + np.sin(k1 * x) * np.cos(k2 * y)
            + np.cos(k1 * x) * np.sin(k2 * y)
            + np.sin(k1 * x) * np.sin(k2 * y)
        )
        perturbation += term

    return 0 * base_flow + eps * perturbation


def tol_key(rtol, rtol_q):
    return (float(f"{rtol:.12g}"), float(f"{rtol_q:.12g}"))


def default_tolerance_lists():
    tol_grid_list = [
        (5.5e-4, 5.5e-4),
        # (5e-4, 1e-4),
        # (5e-4, 1e-3),
        # (1e-4, 1e-4),
        # (1e-4, 5e-4),
        # (1e-4, 1e-3),
        # (1e-3, 1e-4),
        # (1e-3, 5e-4),
        # (1e-3, 1e-3),
    ]

    tol_perturb_list = [
        (5.5e-4*1.1, 5.5e-4*1.1),
        (5.5e-4*1.1, 5.5e-4    ),
        (5.5e-4*1.1, 5.5e-4*0.9),
        (5.5e-4,     5.5e-4*1.1),
        (5.5e-4,     5.5e-4    ),
        (5.5e-4,     5.5e-4*0.9),
        (5.5e-4*0.9, 5.5e-4*1.1),
        (5.5e-4*0.9, 5.5e-4    ),
        (5.5e-4*0.9, 5.5e-4*0.9),
    ]

    # Keep one computed copy of the shared (1e-4, 1e-4) case.
    tol_list = tol_perturb_list + tol_grid_list[1:]
    return tol_grid_list, tol_perturb_list, tol_list


def default_fixed_configs():
    return [
        {
            "key": "tau_0p004",
            "tau": 0.004,
        },
        {
            "key": "tau_0p002",
            "tau": 0.002,
        },
        {
            "key": "tau_0p0015",
            "tau": 0.0015,
        },
        {
            "key": "tau_0p001",
            "tau": 0.001,
        },
        {
            "key": "tau_0p0005",
            "tau": 0.0005,
        },
    ]


def fixed_end_time(t_period, tau):
    n_steps = int(np.ceil((t_period[1] - t_period[0]) / tau))
    return t_period[0] + n_steps * tau


def compute_reference_error(metric_solver, omega, omega_t, ref_omega, ref_t):
    ref_dt = ref_t[1] - ref_t[0]
    ref_idx = np.rint((omega_t - ref_t[0]) / ref_dt).astype(int)

    if np.any(ref_idx < 0) or np.any(ref_idx >= len(ref_t)):
        raise ValueError("Fixed-step time grid is outside the reference grid.")

    if not np.allclose(ref_t[ref_idx], omega_t, atol=1e-10, rtol=1e-10):
        raise ValueError("Fixed-step time grid is not aligned with reference grid.")

    ref_error = np.zeros(len(omega))
    for i, j in enumerate(ref_idx):
        diff = omega[i] - ref_omega[j]
        ref_norm = metric_solver.inner_product(ref_omega[j], ref_omega[j])
        ref_error[i] = np.sqrt(metric_solver.inner_product(diff, diff) / ref_norm)
    return 2 * ref_error


def format_param_for_filename(value: float) -> str:
    return f"{value:g}"


def add_parameter_suffix(path: Path, eps: float, t_ini: float) -> Path:
    suffix = f"_{format_param_for_filename(eps)}_{format_param_for_filename(t_ini)}"
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def make_initial_vorticity(nu, gam, m, eps, t_ini, s_domain, discrete_num, force_term):
    xn = np.linspace(s_domain[0], s_domain[2], discrete_num[0] + 1)
    yn = np.linspace(s_domain[1], s_domain[3], discrete_num[1] + 1)
    X, Y = np.meshgrid(xn, yn)

    initial_phi = initial_streamfunction(X[:-1, :-1], Y[:-1, :-1], nu, m, eps)
    solver_init = vs_mrSAV_solver(
        nu,
        gam,
        s_domain,
        discrete_num,
        initial_phi,
        force_term,
        "ETDRK4",
    )
    initial_vorticity = np.pad(solver_init.stream2velocity(initial_phi)[0], ((0, 1), (0, 1)))
    solver_init.Omega0 = initial_vorticity[:-1, :-1]
    if t_ini <= 0:
        return initial_vorticity
    solver_init.solve_fix_step((0, t_ini), 0.0025)
    return np.pad(solver_init.Omega[-1], ((0, 1), (0, 1)))


def run_experiment(
    output_path: Path,
    force: bool = False,
    *,
    nu: float = 1 / 40,
    m: float = 4,
    eps: float = 3,
    t_ini: float = 0,
    gam: float = 1000,
):
    output_path = add_parameter_suffix(output_path, eps, t_ini)

    if output_path.exists() and not force:
        logging.info("Output already exists: %s", output_path)
        logging.info("Use --force to recompute and overwrite it.")
        return

    np.random.seed(1)

    s_domain = (0.0, 0.0, 2 * np.pi, 2 * np.pi)
    discrete_num = [128, 128]
    t_period = (0.0, 20.0)
    adaptive_cache_version = 2
    fixed_cache_version = 3
    ref_tau = 0.00025
    force_term = make_force_term(m)

    tol_grid_list, tol_perturb_list, tol_list = default_tolerance_lists()
    fixed_configs = default_fixed_configs()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(
        "Parameters: nu=%g, m=%g, eps=%g, t_ini=%g, gam=%g",
        nu,
        m,
        eps,
        t_ini,
        gam,
    )
    logging.info("Building initial vorticity")
    t0 = time.perf_counter()
    initial_vorticity = make_initial_vorticity(
        nu,
        gam,
        m,
        eps,
        t_ini,
        s_domain,
        discrete_num,
        force_term,
    )
    logging.info("Initial vorticity ready in %.2f s", time.perf_counter() - t0)

    all_adaptive_results = []
    for idx, (rtol, rtol_q) in enumerate(tol_list, start=1):
        logging.info(
            "Adaptive run %d/%d: rtol=%.3e, rtol_q=%.3e",
            idx,
            len(tol_list),
            rtol,
            rtol_q,
        )
        t0 = time.perf_counter()
        solver = vs_mrSAV_solver(
            nu,
            gam,
            s_domain,
            discrete_num,
            initial_vorticity,
            force_term,
            "ETD_mrGSAV_MS2_b",
        )
        solver.solve_adaptive_step(
            t_period,
            1e-5,
            1e-2,
            snapshot=np.linspace(0, t_period[1], 201),
            compute_ref_err=True,
            rho   = 0.9,
            rtol  = rtol,
            rtol_q= rtol_q,
            r     = 1/2,
            ref_substeps = 4
        )
        all_adaptive_results.append(
            {
                "rtol": rtol,
                "rtol_q": rtol_q,
                "cpu_time": solver.cpu_time.copy(),
                "step_count": len(solver.tn),
                "tn": solver.tn.copy(),
                "tau": solver.tau.copy(),
                "ref_err": solver.ref_err.copy(),
                "rel_err": solver.rel_err.copy(),
                "ref_err_p": solver.ref_err_p.copy(),
            }
        )
        logging.info(
            "Adaptive run %d finished in %.2f s with %d steps",
            idx,
            time.perf_counter() - t0,
            len(solver.tn),
        )

    ref_t_period = (
        t_period[0],
        max(fixed_end_time(t_period, cfg["tau"]) for cfg in fixed_configs),
    )
    for cfg in fixed_configs:
        ratio = cfg["tau"] / ref_tau
        if not np.isclose(ratio, np.rint(ratio), rtol=0.0, atol=1e-10):
            raise ValueError(f"tau={cfg['tau']} is not aligned with ref_tau={ref_tau}")

    ref_step_count = (ref_t_period[1] - ref_t_period[0]) / ref_tau
    if not np.isclose(ref_step_count, np.rint(ref_step_count), rtol=0.0, atol=1e-10):
        raise ValueError("Reference end time is not aligned with ref_tau.")

    logging.info("Fixed-step reference run: tau=%.4g, t_period=%s", ref_tau, ref_t_period)
    t0 = time.perf_counter()
    ref_solver = vs_mrSAV_solver(
        nu,
        gam,
        s_domain,
        discrete_num,
        initial_vorticity,
        force_term,
        "ETDRK4",
    )
    ref_solver.solve_fix_step(ref_t_period, ref_tau)
    logging.info("Reference run finished in %.2f s", time.perf_counter() - t0)

    metric_solver = vs_mrSAV_solver(
        nu,
        gam,
        s_domain,
        discrete_num,
        initial_vorticity,
        force_term,
        "ETD_mrGSAV_MS2_b",
    )

    fixed_results = []
    for idx, cfg in enumerate(fixed_configs, start=1):
        logging.info("Fixed run %d/%d: tau=%.4g", idx, len(fixed_configs), cfg["tau"])
        t0 = time.perf_counter()
        solver = vs_mrSAV_solver(
            nu,
            gam,
            s_domain,
            discrete_num,
            initial_vorticity,
            force_term,
            "ETD_mrGSAV_MS2_b",
        )
        solver.solve_fix_step(t_period, cfg["tau"])
        ref_error = compute_reference_error(
            metric_solver,
            solver.Omega,
            solver.tn,
            ref_solver.Omega,
            ref_solver.tn,
        )
        fixed_results.append(
            {
                **cfg,
                "tn": solver.tn.copy(),
                "tau_values": solver.tau.copy(),
                "q": solver.q.copy(),
                "cpu_time": solver.cpu_time.copy(),
                "ref_error": ref_error,
            }
        )
        logging.info(
            "Fixed run %d finished in %.2f s with %d steps",
            idx,
            time.perf_counter() - t0,
            len(solver.tn),
        )

    save_data = {
        "adaptive_cache_version": np.array(adaptive_cache_version),
        "fixed_cache_version": np.array(fixed_cache_version),
        "nu": np.array(nu),
        "m": np.array(m),
        "eps": np.array(eps),
        "t_ini": np.array(t_ini),
        "gam": np.array(gam),
        "s_domain": np.array(s_domain),
        "discrete_num": np.array(discrete_num),
        "t_period": np.array(t_period),
        "ref_tau": np.array(ref_tau),
        "tol_grid_rtol": np.array([tol[0] for tol in tol_grid_list]),
        "tol_grid_rtol_q": np.array([tol[1] for tol in tol_grid_list]),
        "tol_perturb_rtol": np.array([tol[0] for tol in tol_perturb_list]),
        "tol_perturb_rtol_q": np.array([tol[1] for tol in tol_perturb_list]),
        "adaptive_count": np.array(len(all_adaptive_results)),
        "fixed_count": np.array(len(fixed_results)),
        "ref_tn": ref_solver.tn.copy(),
    }

    for i, res in enumerate(all_adaptive_results):
        save_data.update(
            {
                f"adaptive_{i}_rtol": np.array(res["rtol"]),
                f"adaptive_{i}_rtol_q": np.array(res["rtol_q"]),
                f"adaptive_{i}_step_count": np.array(res["step_count"]),
                f"adaptive_{i}_cpu_time": res["cpu_time"],
                f"adaptive_{i}_tn": res["tn"],
                f"adaptive_{i}_tau": res["tau"],
                f"adaptive_{i}_ref_err": res["ref_err"],
                f"adaptive_{i}_rel_err": res["rel_err"],
                f"adaptive_{i}_ref_err_p": res["ref_err_p"],
            }
        )

    for i, res in enumerate(fixed_results):
        save_data.update(
            {
                f"fixed_{i}_key": np.array(res["key"]),
                f"fixed_{i}_tau": np.array(res["tau"]),
                f"fixed_{i}_tn": res["tn"],
                f"fixed_{i}_tau_values": res["tau_values"],
                f"fixed_{i}_q": res["q"],
                f"fixed_{i}_cpu_time": res["cpu_time"],
                f"fixed_{i}_ref_error": res["ref_error"],
            }
        )

    np.savez(output_path, **save_data)
    logging.info("Saved data to %s", output_path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the 03_mrSAV-VarStep-Exp simulations and save plotting data."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/03_mrSAV_varstep_exp_data.npz"),
        help="Output NPZ path.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if the output file already exists.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("logs/03_mrSAV_varstep_exp.log"),
        help="Log file path.",
    )
    parser.add_argument(
        "--nu",
        type=float,
        default=1/40,
        help="Viscosity parameter.",
    )
    parser.add_argument(
        "--m",
        type=float,
        default=4,
        help="Mode parameter used in the initial condition and forcing.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=3,
        help="Perturbation amplitude in the initial condition.",
    )
    parser.add_argument(
        "--t-ini",
        "--t_ini",
        dest="t_ini",
        type=float,
        default=0,
        help="Initial spin-up time used to build the starting vorticity.",
    )
    parser.add_argument(
        "--gam",
        "--gamma",
        dest="gam",
        type=float,
        default=1000,
        help="Mean-reverting gamma parameter.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.log.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(args.log, mode="a"),
        ],
    )
    run_experiment(
        args.output,
        force=args.force,
        nu=args.nu,
        m=args.m,
        eps=args.eps,
        t_ini=args.t_ini,
        gam=args.gam,
    )


if __name__ == "__main__":
    main()
