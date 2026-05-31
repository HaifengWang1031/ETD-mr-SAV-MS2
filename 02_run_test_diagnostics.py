from __future__ import annotations

from datetime import datetime
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np

from vs_ns_periodic_mrSAV_solver import (
    vs_mrSAV_Vorticity_Stream_Periodic_Solver as sav_vs_solver,
)

DEFAULT_OUTPUT = Path("data/test_bursting_diagnostics.h5")
DEFAULT_LOG = Path("logs/run_test_diagnostics.log")
DIAGNOSTIC_KEYS = ("q", "Energy", "Enstrophy", "Enstrophy_rate", "Palinstrophy", "Mx", "CPU_time")
ERROR_KEYS = (
    "l2_error_gamma_1000",
    "l2_error_gamma_0",
    "rel_l2_gamma_1000",
    "rel_l2_gamma_0",
)


def initial_streamfunction(
    x: np.ndarray,
    y: np.ndarray,
    nu: float,
    m: float,
    eps: float,
) -> np.ndarray:
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
            np.cos(k1 * x) * np.cos(k2 * y)
            + np.sin(k1 * x) * np.cos(k2 * y)
            + np.cos(k1 * x) * np.sin(k2 * y)
            + np.sin(k1 * x) * np.sin(k2 * y)
        )
        perturbation += term

    return 0 * base_flow + eps * perturbation


def make_initial_vorticity(
    nu: float,
    gamma: float,
    m: int,
    eps: float,
    s_domain: tuple[float, float, float, float],
    discrete_num: tuple[int, int],
):
    xn = np.linspace(s_domain[0], s_domain[2], discrete_num[0] + 1)
    yn = np.linspace(s_domain[1], s_domain[3], discrete_num[1] + 1)
    x_grid, y_grid = np.meshgrid(xn, yn)
    force_term = lambda x, y, t: m * np.cos(m * y)

    initial_phi = initial_streamfunction(
        x_grid[:-1, :-1],
        y_grid[:-1, :-1],
        nu,
        m,
        eps,
    )
    solver_init = sav_vs_solver(
        nu,
        gamma,
        s_domain,
        discrete_num,
        initial_phi,
        force_term,
        "ETDRK4",
    )
    initial_vorticity = np.pad( solver_init.stream2velocity(initial_phi)[0], ((0, 1), (0, 1)))
    # solver_init.Omega0 = initial_vorticity[:-1, :-1]
    # solver_init.solve_fix_step((0, 1), 0.0025)
    # initial_vorticity = np.pad(solver_init.Omega[-1], ((0, 1), (0, 1)))
    return initial_vorticity, force_term


def make_solver(
    nu: float,
    gamma: float,
    s_domain: tuple[float, float, float, float],
    discrete_num: tuple[int, int],
    initial_vorticity: np.ndarray,
    force_term,
    method: str,
):
    solver = sav_vs_solver(
        nu,
        gamma,
        s_domain,
        discrete_num,
        initial_vorticity,
        force_term,
        method,
    )
    solver._fN_cache = None
    return solver


def empty_diagnostics(n_steps: int) -> dict[str, np.ndarray]:
    return {key: np.empty(n_steps + 1, dtype=np.float64) for key in DIAGNOSTIC_KEYS}


def record_diagnostics(
    solver,
    omega: np.ndarray,
    q: float,
    t: float,
    cpu_time: float,
    diagnostics: dict[str, np.ndarray],
    index: int,
) -> None:
    energy, enstrophy, palinstrophy = solver.vorticity_energy(omega)
    diagnostics["q"][index] = q
    diagnostics["Energy"][index] = energy
    diagnostics["Enstrophy"][index] = enstrophy
    diagnostics["Enstrophy_rate"][index] = solver.enstrophy_rate(omega, t)
    diagnostics["Palinstrophy"][index] = palinstrophy
    diagnostics["Mx"][index] = np.max(omega)
    diagnostics["CPU_time"][index] = cpu_time


def init_state(solver, tau: float):
    if getattr(solver.step, "__name__", "") == "ETDRK4":
        solver._prepare_ETDRK4_coefficients(tau)
    return {
        "solver": solver,
        "tau": tau,
        "t": 0.0,
        "omega_hist": [solver.Omega0.copy()],
        "q_hist": [solver.q0],
        "cpu_time": 0.0,
    }


def advance_one_step(state) -> tuple[np.ndarray, float]:
    solver = state["solver"]
    tau = state["tau"]
    omega_hist = state["omega_hist"]
    q_hist = state["q_hist"]

    start = perf_counter()
    if len(omega_hist) < solver.setup_step:
        omega_new, q_new = solver.ETD(
            np.asarray(omega_hist[-1:]),
            np.asarray(q_hist[-1:]),
            state["t"],
            np.asarray([tau], dtype=np.float64),
        )
        if len(omega_hist) + 1 == solver.setup_step:
            solver._fN_cache = None
    else:
        omega_new, q_new = solver.step(
            np.asarray(omega_hist[-solver.setup_step :]),
            np.asarray(q_hist[-solver.setup_step :]),
            state["t"],
            np.full(solver.setup_step, tau, dtype=np.float64),
        )
    state["cpu_time"] += perf_counter() - start

    state["t"] += tau
    omega_hist.append(omega_new)
    q_hist.append(q_new)
    del omega_hist[:-solver.setup_step]
    del q_hist[:-solver.setup_step]

    return omega_new, q_new


def log_message(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S} {message}"
    print(line, flush=True)
    with log_path.open("a", buffering=1) as f:
        f.write(line + "\n")


def initialize_diagnostics_file(
    output_path: Path,
    t: np.ndarray,
    attrs: dict[str, object],
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(output_path, "w")
    f.create_group("time").create_dataset("t", data=t)

    n_steps = len(t) - 1
    diag_chunk = (min(10_000, n_steps + 1),)
    error_chunk = (min(10_000, max(1, n_steps)),)

    for group_name in ("gamma_1000", "gamma_0", "etdrk4_ref"):
        group = f.create_group(group_name)
        for key in DIAGNOSTIC_KEYS:
            group.create_dataset(
                key,
                shape=(n_steps + 1,),
                dtype=np.float64,
                chunks=diag_chunk,
                fillvalue=np.nan,
            )

    error_group = f.create_group("errors")
    for key in ERROR_KEYS:
        error_group.create_dataset(
            key,
            shape=(n_steps,),
            dtype=np.float64,
            chunks=error_chunk,
            fillvalue=np.nan,
        )

    for key, value in attrs.items():
        f.attrs[key] = value
    f.attrs["completed_step"] = 0
    f.attrs["completed_time"] = t[0]
    f.attrs["complete"] = False
    return f


def write_checkpoint(
    f,
    start: int,
    end: int,
    gamma_1000: dict[str, np.ndarray],
    gamma_0: dict[str, np.ndarray],
    ref: dict[str, np.ndarray],
    errors: dict[str, np.ndarray],
    t: np.ndarray,
) -> None:
    if start <= end:
        target = slice(start, end + 1)
        for group_name, data in (
            ("gamma_1000", gamma_1000),
            ("gamma_0", gamma_0),
            ("etdrk4_ref", ref),
        ):
            group = f[group_name]
            for key in DIAGNOSTIC_KEYS:
                group[key][target] = data[key][target]

    if end > 0:
        error_start = max(start - 1, 0)
        error_target = slice(error_start, end)
        for key, values in errors.items():
            f["errors"][key][error_target] = values[error_target]

    f.attrs["completed_step"] = end
    f.attrs["completed_time"] = t[end]
    f.flush()


def run_diagnostics(
    output_path: Path | str = DEFAULT_OUTPUT,
    log_path: Path | str = DEFAULT_LOG,
    T: float = 100.0,
    tau: float = 0.001,
    tau_ref: float = 0.0005,
    Re: float = 40.0,
    m: int = 4,
    eps: float = 0.25,
    gamma: float = 1000.0,
    discrete_num: tuple[int, int] = (128, 128),
    s_domain: tuple[float, float, float, float] = (0.0, 0.0, 2 * np.pi, 2 * np.pi),
    checkpoint_every: int = 1000,
    progress_every: int = 100,
) -> Path:
    output_path = Path(output_path)
    log_path = Path(log_path)
    log_message(log_path, f"Starting diagnostics: output={output_path}")

    n_steps = int(round(T / tau))
    if not np.isclose(n_steps * tau, T, rtol=0.0, atol=max(1e-12, tau * 1e-8)):
        raise ValueError("T must be an integer multiple of tau")

    ref_steps_per_step = int(round(tau / tau_ref))
    if not np.isclose(
        ref_steps_per_step * tau_ref,
        tau,
        rtol=0.0,
        atol=max(1e-12, tau_ref * 1e-8),
    ):
        raise ValueError("tau must be an integer multiple of tau_ref")

    np.random.seed(1)
    nu = 1 / Re
    log_message(
        log_path,
        (
            f"Parameters: T={T}, tau={tau}, tau_ref={tau_ref}, "
            f"steps={n_steps}, Re={Re}, m={m}, eps={eps}, gamma={gamma}"
        ),
    )
    initial_vorticity, force_term = make_initial_vorticity(
        nu,
        gamma,
        m,
        eps,
        s_domain,
        discrete_num,
    )

    gamma_solver = make_solver(
        nu,
        gamma,
        s_domain,
        discrete_num,
        initial_vorticity,
        force_term,
        "ETD_mrGSAV_MS2_b",
    )
    gamma0_solver = make_solver(
        nu,
        0.0,
        s_domain,
        discrete_num,
        initial_vorticity,
        force_term,
        "ETD_mrGSAV_MS2_b",
    )
    ref_solver = make_solver(
        nu,
        gamma,
        s_domain,
        discrete_num,
        initial_vorticity,
        force_term,
        "ETDRK4",
    )

    gamma_state = init_state(gamma_solver, tau)
    gamma0_state = init_state(gamma0_solver, tau)
    ref_state = init_state(ref_solver, tau_ref)

    t = tau * np.arange(n_steps + 1, dtype=np.float64)
    gamma_data = empty_diagnostics(n_steps)
    gamma0_data = empty_diagnostics(n_steps)
    ref_data = empty_diagnostics(n_steps)
    l2_error_gamma_1000 = np.empty(n_steps, dtype=np.float64)
    l2_error_gamma_0 = np.empty(n_steps, dtype=np.float64)
    rel_l2_gamma_1000 = np.empty(n_steps, dtype=np.float64)
    rel_l2_gamma_0 = np.empty(n_steps, dtype=np.float64)

    omega_gamma = gamma_state["omega_hist"][-1]
    omega_gamma0 = gamma0_state["omega_hist"][-1]
    omega_ref = ref_state["omega_hist"][-1]
    record_diagnostics(gamma_solver, omega_gamma, gamma_solver.q0, t[0], 0.0, gamma_data, 0)
    record_diagnostics(gamma0_solver, omega_gamma0, gamma0_solver.q0, t[0], 0.0, gamma0_data, 0)
    record_diagnostics(ref_solver, omega_ref, ref_solver.q0, t[0], 0.0, ref_data, 0)

    scale = gamma_solver.hx * gamma_solver.hy
    progress_every = max(1, int(progress_every))
    checkpoint_every = max(1, min(int(checkpoint_every), n_steps))
    last_written_step = -1
    last_completed_step = 0

    errors = {
        "l2_error_gamma_1000": l2_error_gamma_1000,
        "l2_error_gamma_0": l2_error_gamma_0,
        "rel_l2_gamma_1000": rel_l2_gamma_1000,
        "rel_l2_gamma_0": rel_l2_gamma_0,
    }
    attrs = {
        "Re": Re,
        "m": m,
        "eps": eps,
        "T": T,
        "tau": tau,
        "tau_ref": tau_ref,
        "gamma": gamma,
        "discrete_num": discrete_num,
        "s_domain": s_domain,
        "checkpoint_every": checkpoint_every,
        "progress_every": progress_every,
    }

    with initialize_diagnostics_file(output_path, t, attrs) as f:
        write_checkpoint(
            f,
            0,
            0,
            gamma_data,
            gamma0_data,
            ref_data,
            errors,
            t,
        )
        last_written_step = 0
        log_message(log_path, "Wrote checkpoint at step 0")

        try:
            for step in range(1, n_steps + 1):
                omega_gamma, q_gamma = advance_one_step(gamma_state)
                omega_gamma0, q_gamma0 = advance_one_step(gamma0_state)

                q_ref = ref_state["q_hist"][-1]
                for _ in range(ref_steps_per_step):
                    omega_ref, q_ref = advance_one_step(ref_state)

                record_diagnostics(
                    gamma_solver,
                    omega_gamma,
                    q_gamma,
                    t[step],
                    gamma_state["cpu_time"],
                    gamma_data,
                    step,
                )
                record_diagnostics(
                    gamma0_solver,
                    omega_gamma0,
                    q_gamma0,
                    t[step],
                    gamma0_state["cpu_time"],
                    gamma0_data,
                    step,
                )
                record_diagnostics(
                    ref_solver,
                    omega_ref,
                    q_ref,
                    t[step],
                    ref_state["cpu_time"],
                    ref_data,
                    step,
                )

                err_gamma = np.linalg.norm((omega_gamma - omega_ref) * scale)
                err_gamma0 = np.linalg.norm((omega_gamma0 - omega_ref) * scale)
                ref_norm = np.linalg.norm(omega_ref * scale)
                l2_error_gamma_1000[step - 1] = err_gamma
                l2_error_gamma_0[step - 1] = err_gamma0
                rel_l2_gamma_1000[step - 1] = err_gamma / ref_norm
                rel_l2_gamma_0[step - 1] = err_gamma0 / ref_norm
                last_completed_step = step

                if step % progress_every == 0 or step == n_steps:
                    log_message(
                        log_path,
                        f"Progress: {step}/{n_steps} t={t[step]:.6f}",
                    )

                if step % checkpoint_every == 0 or step == n_steps:
                    write_checkpoint(
                        f,
                        last_written_step + 1,
                        step,
                        gamma_data,
                        gamma0_data,
                        ref_data,
                        errors,
                        t,
                    )
                    last_written_step = step
                    log_message(
                        log_path,
                        f"Wrote checkpoint at step {step}/{n_steps}",
                    )

            f.attrs["complete"] = True
            f.flush()
        except Exception as exc:
            if last_completed_step > last_written_step:
                write_checkpoint(
                    f,
                    last_written_step + 1,
                    last_completed_step,
                    gamma_data,
                    gamma0_data,
                    ref_data,
                    errors,
                    t,
                )
                log_message(
                    log_path,
                    f"Wrote checkpoint after failure at step {last_completed_step}",
                )
            log_message(log_path, f"Failed: {type(exc).__name__}: {exc}")
            raise

    log_message(log_path, f"Saved diagnostics to {output_path}")
    return output_path


if __name__ == "__main__":
    run_diagnostics()
