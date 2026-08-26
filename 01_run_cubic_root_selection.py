from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from vs_ns_periodic_mrSAV_solver import (
    vs_mrSAV_Vorticity_Stream_Periodic_Solver as vs_mrSAV_solver,
)


def force_term(x: np.ndarray, y: np.ndarray, t: float) -> np.ndarray:
    del y, t
    return np.cos(x)


def initial_streamfunction(
    x: np.ndarray,
    y: np.ndarray,
    eps: float,
) -> np.ndarray:
    k_max = 10
    k1_vals = np.arange(-k_max, k_max + 1)
    k2_vals = np.arange(-k_max, k_max + 1)
    k1_grid, k2_grid = np.meshgrid(k1_vals, k2_vals, indexing="ij")
    k_mod = np.sqrt(k1_grid**2 + k2_grid**2)

    mask = k_mod <= 10
    perturbation = np.zeros_like(x, dtype=np.float64)
    for k1, k2, k_abs in zip(k1_grid[mask], k2_grid[mask], k_mod[mask]):
        if k_abs < 1e-10:
            continue
        perturbation += (1 / k_abs**3) * (
            np.cos(k1 * x) * np.cos(k2 * y)
            + np.sin(k1 * x) * np.cos(k2 * y)
            + np.cos(k1 * x) * np.sin(k2 * y)
            + np.sin(k1 * x) * np.sin(k2 * y)
        )
    return eps * perturbation


def make_initial_vorticity(
    nu: float,
    gamma: float,
    domain: tuple[float, float, float, float],
    grid: tuple[int, int],
    init_time: float,
    init_tau: float,
) -> np.ndarray:
    xn = np.linspace(domain[0], domain[2], grid[0] + 1)
    yn = np.linspace(domain[1], domain[3], grid[1] + 1)
    x_grid, y_grid = np.meshgrid(xn, yn)
    initial_phi = initial_streamfunction(
        x_grid[:-1, :-1],
        y_grid[:-1, :-1],
        eps=2.5,
    )
    solver = vs_mrSAV_solver(
        nu,
        gamma,
        domain,
        grid,
        initial_phi,
        force_term,
        "ETD_mrGSAV_MS2_b",
        root_selection="legacy",
    )
    initial_vorticity = np.pad(
        solver.stream2velocity(initial_phi)[0],
        ((0, 1), (0, 1)),
    )
    solver.Omega0 = initial_vorticity[:-1, :-1]
    if init_time > 0:
        with open(os.devnull, "w") as sink, redirect_stdout(sink):
            solver.solve_fix_step((0.0, init_time), init_tau, snapshot=[init_time])
        initial_vorticity = np.pad(solver.Omega[-1], ((0, 1), (0, 1)))
    return initial_vorticity


def root_history_arrays(history: list[dict[str, object]]) -> dict[str, np.ndarray]:
    count = len(history)
    roots_all = np.full((count, 3), np.nan + 1j * np.nan, dtype=np.complex128)
    real_roots = np.full((count, 3), np.nan, dtype=np.float64)
    for index, item in enumerate(history):
        all_values = np.asarray(item["roots_all"], dtype=np.complex128)
        real_values = np.asarray(item["real_roots"], dtype=np.float64)
        roots_all[index, : len(all_values)] = all_values
        real_roots[index, : len(real_values)] = real_values

    return {
        "root_time": np.array([item["time"] for item in history]),
        "root_tau": np.array([item["tau"] for item in history]),
        "alpha": np.array([item["alpha"] for item in history]),
        "beta": np.array([item["beta"] for item in history]),
        "target": np.array([item["target"] for item in history]),
        "discriminant": np.array([item["discriminant"] for item in history]),
        "root_case": np.array([item["root_case"] for item in history]),
        "real_root_count": np.array(
            [item["real_root_count"] for item in history], dtype=np.int64
        ),
        "distinct_real_root_count": np.array(
            [item["distinct_real_root_count"] for item in history], dtype=np.int64
        ),
        "roots_all": roots_all,
        "real_roots": real_roots,
        "selected_root": np.array([item["selected_root"] for item in history]),
        "selected_distance": np.array(
            [item["selected_distance"] for item in history]
        ),
        "selected_derivative": np.array(
            [item["selected_derivative"] for item in history]
        ),
        "selected_residual": np.array(
            [item["selected_residual"] for item in history]
        ),
        "selected_q_positive": np.array(
            [item["selected_q_positive"] for item in history], dtype=bool
        ),
    }


def summarize_roots(history: list[dict[str, object]]) -> dict[str, object]:
    case_counts = Counter(item["root_case"] for item in history)
    multiple_steps = sum(
        int(item["distinct_real_root_count"] > 1) for item in history
    )
    return {
        "evaluated_steps": len(history),
        "root_case_counts": dict(case_counts),
        "steps_with_multiple_distinct_real_roots": multiple_steps,
        "maximum_distinct_real_root_count": max(
            (int(item["distinct_real_root_count"]) for item in history),
            default=0,
        ),
        "maximum_selected_distance_from_target": max(
            (float(item["selected_distance"]) for item in history),
            default=0.0,
        ),
        "minimum_absolute_selected_derivative": min(
            (abs(float(item["selected_derivative"])) for item in history),
            default=np.nan,
        ),
        "maximum_selected_residual": max(
            (float(item["selected_residual"]) for item in history),
            default=0.0,
        ),
        "steps_with_nonpositive_selected_q": sum(
            int(not item["selected_q_positive"]) for item in history
        ),
    }


def case_key(tau: float, selection: str) -> str:
    return f"tau_{tau:.8g}_{selection}".replace(".", "p")


def run_case(
    initial_vorticity: np.ndarray,
    tau: float,
    selection: str,
    final_time: float,
    nu: float,
    gamma: float,
    domain: tuple[float, float, float, float],
    grid: tuple[int, int],
) -> dict[str, object]:
    solver = vs_mrSAV_solver(
        nu,
        gamma,
        domain,
        grid,
        initial_vorticity,
        force_term,
        "ETD_mrGSAV_MS2_b",
        root_selection=selection,
    )
    try:
        with open(os.devnull, "w") as sink, redirect_stdout(sink):
            solver.solve_fix_step((0.0, final_time), tau, snapshot=[final_time])
    except Exception as error:
        return {
            "status": "failed",
            "error": f"{type(error).__name__}: {error}",
            "solver": solver,
            "root_summary": summarize_roots(solver.cubic_root_history),
            "root_arrays": root_history_arrays(solver.cubic_root_history),
        }

    final_omega = solver.Omega[-1].copy()
    final_psi = solver.vorticity2stream(final_omega)
    final_u, final_v = solver.stream2velocity(final_psi)
    return {
        "status": "complete",
        "solver": solver,
        "final_omega": final_omega,
        "final_u": final_u,
        "final_v": final_v,
        "root_summary": summarize_roots(solver.cubic_root_history),
        "root_arrays": root_history_arrays(solver.cubic_root_history),
    }


def compare_cases(nearest: dict[str, object], farthest: dict[str, object]) -> dict[str, object]:
    if nearest["status"] != "complete" or farthest["status"] != "complete":
        return {
            "status": "incomplete",
            "nearest_status": nearest["status"],
            "farthest_status": farthest["status"],
        }

    nearest_solver = nearest["solver"]
    farthest_solver = farthest["solver"]
    omega_difference = farthest["final_omega"] - nearest["final_omega"]
    velocity_difference_sq = nearest_solver.inner_product(
        farthest["final_u"] - nearest["final_u"],
        farthest["final_u"] - nearest["final_u"],
    ) + nearest_solver.inner_product(
        farthest["final_v"] - nearest["final_v"],
        farthest["final_v"] - nearest["final_v"],
    )
    nearest_velocity_sq = nearest_solver.inner_product(
        nearest["final_u"], nearest["final_u"]
    ) + nearest_solver.inner_product(nearest["final_v"], nearest["final_v"])
    omega_norm_sq = nearest_solver.inner_product(
        nearest["final_omega"], nearest["final_omega"]
    )

    nearest_roots = nearest["root_arrays"]
    farthest_roots = farthest["root_arrays"]
    shared_root_steps = min(
        len(nearest_roots["selected_root"]),
        len(farthest_roots["selected_root"]),
    )
    selected_difference = np.abs(
        nearest_roots["selected_root"][:shared_root_steps]
        - farthest_roots["selected_root"][:shared_root_steps]
    )
    differing = selected_difference > 1e-10
    first_difference_time = (
        float(nearest_roots["root_time"][:shared_root_steps][differing][0])
        if np.any(differing)
        else None
    )

    return {
        "status": "complete",
        "relative_final_vorticity_difference": float(
            np.sqrt(nearest_solver.inner_product(omega_difference, omega_difference))
            / max(np.sqrt(omega_norm_sq), np.finfo(float).tiny)
        ),
        "relative_final_velocity_difference": float(
            np.sqrt(velocity_difference_sq)
            / max(np.sqrt(nearest_velocity_sq), np.finfo(float).tiny)
        ),
        "absolute_final_q_difference": float(
            abs(farthest_solver.q[-1] - nearest_solver.q[-1])
        ),
        "absolute_final_energy_difference": float(
            abs(farthest_solver.Energy[-1] - nearest_solver.Energy[-1])
        ),
        "absolute_final_enstrophy_difference": float(
            abs(farthest_solver.Enstrophy[-1] - nearest_solver.Enstrophy[-1])
        ),
        "maximum_energy_history_difference": float(
            np.max(np.abs(farthest_solver.Energy - nearest_solver.Energy))
        ),
        "maximum_enstrophy_history_difference": float(
            np.max(np.abs(farthest_solver.Enstrophy - nearest_solver.Enstrophy))
        ),
        "maximum_q_history_difference": float(
            np.max(np.abs(farthest_solver.q - nearest_solver.q))
        ),
        "steps_with_different_selected_roots": int(np.count_nonzero(differing)),
        "first_selected_root_difference_time": first_difference_time,
        "maximum_selected_root_difference": float(
            np.max(selected_difference) if selected_difference.size else 0.0
        ),
    }


def run_experiment(args: argparse.Namespace) -> dict[str, object]:
    np.random.seed(1)
    nu = 1 / 50
    gamma = 1000.0
    domain = (0.0, 0.0, 2 * np.pi, 2 * np.pi)
    grid = (args.grid, args.grid)

    for tau in args.taus:
        step_count = args.final_time / tau
        if not np.isclose(step_count, round(step_count), rtol=0.0, atol=1e-10):
            raise ValueError(f"final_time={args.final_time} is not divisible by tau={tau}")
    init_step_count = args.init_time / args.init_tau if args.init_time > 0 else 0
    if args.init_time > 0 and not np.isclose(
        init_step_count, round(init_step_count), rtol=0.0, atol=1e-10
    ):
        raise ValueError(
            f"init_time={args.init_time} is not divisible by init_tau={args.init_tau}"
        )

    print("Building the notebook-matched initial vorticity...", flush=True)
    initial_vorticity = make_initial_vorticity(
        nu,
        gamma,
        domain,
        grid,
        args.init_time,
        args.init_tau,
    )

    summary: dict[str, object] = {
        "configuration": {
            "source_notebook": "01_convergence_test.ipynb",
            "nu": nu,
            "m": 1,
            "gamma": gamma,
            "domain": domain,
            "grid": grid,
            "initial_perturbation": 2.5,
            "initialization_time": args.init_time,
            "initialization_tau": args.init_tau,
            "experiment_time": args.final_time,
            "taus": args.taus,
            "theoretical_root_target": "C = exp(-gamma*tau) * p_n",
        },
        "runs": {},
        "comparisons": {},
    }
    save_data: dict[str, np.ndarray] = {
        "initial_vorticity": initial_vorticity,
        "taus": np.asarray(args.taus),
        "grid": np.asarray(grid),
        "final_time": np.array(args.final_time),
        "init_time": np.array(args.init_time),
        "init_tau": np.array(args.init_tau),
    }

    for tau in args.taus:
        runs = {}
        for selection in ("nearest", "farthest"):
            print(f"Running tau={tau:g}, selection={selection}...", flush=True)
            result = run_case(
                initial_vorticity,
                tau,
                selection,
                args.final_time,
                nu,
                gamma,
                domain,
                grid,
            )
            runs[selection] = result
            key = case_key(tau, selection)
            run_summary = {
                "status": result["status"],
                "root_summary": result["root_summary"],
            }
            if result["status"] == "complete":
                solver = result["solver"]
                run_summary.update(
                    {
                        "final_q": float(solver.q[-1]),
                        "final_energy": float(solver.Energy[-1]),
                        "final_enstrophy": float(solver.Enstrophy[-1]),
                    }
                )
                save_data[f"{key}_final_omega"] = result["final_omega"]
                save_data[f"{key}_q"] = solver.q
                save_data[f"{key}_energy"] = solver.Energy
                save_data[f"{key}_enstrophy"] = solver.Enstrophy
            else:
                run_summary["error"] = result["error"]

            for name, values in result["root_arrays"].items():
                save_data[f"{key}_{name}"] = values
            summary["runs"][key] = run_summary

        comparison = compare_cases(runs["nearest"], runs["farthest"])
        summary["comparisons"][f"tau_{tau:.8g}".replace(".", "p")] = comparison
        print(json.dumps(comparison, indent=2), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **save_data)
    summary_path = args.output.with_suffix(".json")
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved arrays to {args.output}")
    print(f"Saved summary to {summary_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare nearest and farthest real-root choices for ETD-mrGSAV-MS2."
    )
    parser.add_argument(
        "--taus",
        type=float,
        nargs="+",
        default=[0.01, 0.005, 0.0025],
    )
    parser.add_argument("--grid", type=int, default=128)
    parser.add_argument("--final-time", type=float, default=1.0)
    parser.add_argument("--init-time", type=float, default=1.0)
    parser.add_argument("--init-tau", type=float, default=0.0025)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/cubic_root_selection/root_selection_comparison.npz"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
