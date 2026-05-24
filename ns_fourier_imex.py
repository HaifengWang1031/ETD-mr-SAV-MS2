"""
Fourier-spectral solver for 2D incompressible Navier-Stokes (periodic domain).

Velocity formulation with Leray projection:
  ∂_t u = -ν|k|² û + P_k[u×ω] + P_k[f̂]

Time-stepping methods are registered via @NSFourierSolver.register and can be
added externally without modifying this file.

Built-in methods: "IMEX", "ETDRK4"

Usage
-----
def kolmogorov_force(XX, YY, t):
    return np.sin(4 * YY), np.zeros_like(XX)

solver = NSFourierSolver(Nx, Ny, Lx, Ly, Re, dt, u0, v0,
                         force=kolmogorov_force, method="IMEX")
solver.solve(T, snapshot_every=100)
omega = solver.snapshots   # list of vorticity arrays
ke    = solver.ke          # list of kinetic energy values

Custom stepper
--------------
@NSFourierSolver.register("MY_METHOD")
def my_step(s, ux_hat, uy_hat, t, dt):
    # s is the solver instance — access s.KX, s.leray(), etc.
    # t is the current simulation time, dt is the step size
    ...
    return ux_new, uy_new
"""

import time
import numpy as np

try:
    import pyfftw
    import pyfftw.interfaces.numpy_fft as _fft
    pyfftw.interfaces.cache.enable()
    _HAS_PYFFTW = True
except ImportError:
    import scipy.fft as _sfft
    _HAS_PYFFTW = False


def _optimal_fftw_threads(n: int) -> int:
    if n <= 64:
        return 1
    elif n <= 128:
        return 4
    elif n <= 512:
        return 6
    else:
        return 8


def _fft2(f):
    return _fft.fft2(f) if _HAS_PYFFTW else _sfft.fft2(f)


def _ifft2(f):
    return _fft.ifft2(f) if _HAS_PYFFTW else _sfft.ifft2(f)


class NSFourierSolver:
    """
    2D incompressible Navier-Stokes, velocity formulation, periodic domain.

    Parameters
    ----------
    Nx, Ny       : grid resolution
    Lx, Ly       : domain size (default 2π × 2π)
    Re           : Reynolds number (ν = 1/Re)
    dt           : time step
    u0, v0       : initial velocity fields, shape (Nx, Ny), physical space
    force        : optional (fx, fy) tuple of physical-space forcing arrays
    method       : name of the registered time-stepping method
    """

    _steppers: dict = {}   # class-level registry: name -> fn

    @classmethod
    def register(cls, name: str):
        """Decorator to register a time-stepping function under `name`.

        Stepper signature: fn(solver, ux_hat, uy_hat, t, dt) -> (ux_new, uy_new)
        where t is the current simulation time and dt is the step size.
        """
        def decorator(fn):
            cls._steppers[name] = fn
            return fn
        return decorator

    def __init__(self, Nx, Ny, Lx, Ly, Re, dt, u0, v0,
                 force=None, method="IMEX"):

        if method not in self._steppers:
            raise ValueError(f"Unknown method '{method}'. "
                             f"Available: {list(self._steppers)}")

        if _HAS_PYFFTW:
            pyfftw.config.NUM_THREADS = _optimal_fftw_threads(max(Nx, Ny))

        self.Nx, self.Ny = Nx, Ny
        self.Lx, self.Ly = Lx, Ly
        self.Re = Re
        self.dt = dt
        self._step_fn = self._steppers[method]

        # ── Wavenumbers ───────────────────────────────────────────────────────
        kx = 2 * np.pi / Lx * np.fft.fftfreq(Nx, d=1.0 / Nx)
        ky = 2 * np.pi / Ly * np.fft.fftfreq(Ny, d=1.0 / Ny)
        self.KX, self.KY = np.meshgrid(kx, ky, indexing='ij')   # (Nx, Ny)
        self.K2 = self.KX**2 + self.KY**2
        self.K2[0, 0] = 1.0   # avoid /0; mean mode zeroed by Leray

        # 2/3-rule dealiasing mask
        kx_max = Nx // 3
        ky_max = Ny // 3
        self.DEALIAS = (
            (np.abs(self.KX) < 2 * np.pi / Lx * kx_max) &
            (np.abs(self.KY) < 2 * np.pi / Ly * ky_max)
        )

        # IMEX implicit denominator (precomputed; recompute if dt changes)
        self.denom = 1.0 + dt * self.K2 / Re

        # Linear operator L = ν|k|² (for ETD methods)
        self.L = self.K2 / Re

        # ── Physical-space grid (needed for time-dependent forcing) ──────────
        x = np.linspace(0, Lx, Nx, endpoint=False)
        y = np.linspace(0, Ly, Ny, endpoint=False)
        self.XX, self.YY = np.meshgrid(x, y, indexing='ij')

        # ── Forcing: callable f(XX, YY, t) -> (fx, fy) ───────────────────────
        self.f = force   # None means no forcing

        # ── Initial state ─────────────────────────────────────────────────────
        self.ux_hat = self.ft(u0); self.ux_hat[0, 0] = 0.0
        self.uy_hat = self.ft(v0); self.uy_hat[0, 0] = 0.0
        self.t = 0.0

    # ── FFT wrappers ──────────────────────────────────────────────────────────

    def ft(self, u):
        return _fft2(u)

    def ift(self, u):
        return _ifft2(u)

    def dealias(self, u_hat):
        return u_hat * self.DEALIAS

    # ── Spectral operators ────────────────────────────────────────────────────

    def leray(self, vx_hat, vy_hat):
        """Leray (Helmholtz) projection onto divergence-free fields."""
        kdotv = self.KX * vx_hat + self.KY * vy_hat
        px = vx_hat - self.KX * kdotv / self.K2
        py = vy_hat - self.KY * kdotv / self.K2
        px[0, 0] = 0.0
        py[0, 0] = 0.0
        return px, py

    def nonlinear(self, ux_hat, uy_hat):
        """Dealiased nonlinear term: u×ω in 2D = (ω·uy, -ω·ux)."""
        ux = self.ift(self.dealias(ux_hat)).real
        uy = self.ift(self.dealias(uy_hat)).real
        omega = self.ift(self.dealias(1j * self.KX * uy_hat
                                      - 1j * self.KY * ux_hat)).real
        nx_hat = self.dealias(self.ft( uy * omega))
        ny_hat = self.dealias(self.ft(-ux * omega))
        return nx_hat, ny_hat

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def vorticity(self):
        return self.ift(1j * self.KX * self.uy_hat
                        - 1j * self.KY * self.ux_hat).real

    def kinetic_energy(self):
        ux = self.ift(self.ux_hat).real
        uy = self.ift(self.uy_hat).real
        return float(0.5 * np.mean(ux**2 + uy**2))

    def enstrophy(self):
        omega = self.vorticity()
        return float(0.5 * np.mean(omega**2))

    # ── Solver ────────────────────────────────────────────────────────────────

    def solve(self, T, snapshot_every=1):
        """Advance the simulation by time T, recording snapshots.

        Results stored on the solver:
          self.snapshots  — list of vorticity arrays
          self.times      — list of simulation times at each snapshot
          self.ke         — list of kinetic energy values
          self.ens        — list of enstrophy values
        """
        nsteps = int(T / self.dt)
        self.snapshots = []
        self.times     = []
        self.ke        = []
        self.ens       = []

        t_wall = time.perf_counter()

        for n in range(nsteps):
            t0 = time.perf_counter()
            self.ux_hat, self.uy_hat = self._step_fn(
                self, self.ux_hat, self.uy_hat, self.t, self.dt
            )
            self.t += self.dt
            step_ms = (time.perf_counter() - t0) * 1e3

            if n % snapshot_every == 0:
                self.snapshots.append(self.vorticity())
                self.times.append(self.t)
                ke  = self.kinetic_energy()
                ens = self.enstrophy()
                self.ke.append(ke)
                self.ens.append(ens)

            print(f"\rt={self.t:.4f}/{self.t + (nsteps - n - 1)*self.dt:.4f}"
                    f"  step={step_ms:.2f}ms  KE={ke:.6f}  Ens={ens:.6f}",
                    end="", flush=True)

        total = time.perf_counter() - t_wall
        print(f"\nDone: {nsteps} steps in {total:.2f}s "
              f"({total/nsteps*1e3:.2f} ms/step)")


# ── Built-in steppers ─────────────────────────────────────────────────────────

@NSFourierSolver.register("IMEX")
def _imex_step(s, ux_hat, uy_hat, t, dt):
    """First-order IMEX Euler."""
    nx, ny = s.nonlinear(ux_hat, uy_hat)
    if s.f is not None:
        fx, fy = s.f(s.XX, s.YY, t)
        nx = nx + s.ft(fx)
        ny = ny + s.ft(fy)
    nx, ny = s.leray(nx, ny)
    denom = 1.0 + dt * s.K2 / s.Re
    return (ux_hat + dt * nx) / denom, (uy_hat + dt * ny) / denom


@NSFourierSolver.register("ETDRK4")
def _etdrk4_step(s, ux_hat, uy_hat, t, dt):
    """Fourth-order ETD Runge-Kutta (Cox-Matthews) with M=16 contour quadrature."""
    L  = s.L

    M   = 16
    r   = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    Lr  = dt * L[..., np.newaxis] + r

    phi10 = np.exp(-dt * L / 2)
    phi11 = np.mean((1 - np.exp(-Lr / 2)) / Lr, axis=-1).real

    phi20 = np.exp(-dt * L)
    phi31 = np.mean((-4 - Lr + np.exp(Lr) * (4 - 3*Lr + Lr**2)) / Lr**3, axis=-1).real
    phi32 = np.mean(( 2 + Lr + np.exp(Lr) * (-2 + Lr))           / Lr**3, axis=-1).real
    phi33 = np.mean((-4 - 3*Lr - Lr**2 + np.exp(Lr) * (4 - Lr))  / Lr**3, axis=-1).real

    def _force_hat(t_eval):
        if s.f is None:
            return None, None
        fx, fy = s.f(s.XX, s.YY, t_eval)
        return s.ft(fx), s.ft(fy)

    def N(ux_h, uy_h, t_eval):
        nx, ny = s.nonlinear(ux_h, uy_h)
        fx_hat, fy_hat = _force_hat(t_eval)
        if fx_hat is not None:
            nx = nx + fx_hat
            ny = ny + fy_hat
        return s.leray(nx, ny)

    Nx0, Ny0 = N(ux_hat, uy_hat, t)

    ux1 = phi10 * ux_hat + dt * phi11 * Nx0
    uy1 = phi10 * uy_hat + dt * phi11 * Ny0
    Nx1, Ny1 = N(ux1, uy1, t + dt / 2)

    ux2 = phi10 * ux_hat + dt * phi11 * Nx1
    uy2 = phi10 * uy_hat + dt * phi11 * Ny1
    Nx2, Ny2 = N(ux2, uy2, t + dt / 2)

    ux3 = phi10 * ux1 + dt * phi11 * (2 * Nx2 - Nx0)
    uy3 = phi10 * uy1 + dt * phi11 * (2 * Ny2 - Ny0)
    Nx3, Ny3 = N(ux3, uy3, t + dt)

    ux_new = phi20 * ux_hat + dt * (phi31 * Nx0 + 2*phi32 * (Nx1 + Nx2) + phi33 * Nx3)
    uy_new = phi20 * uy_hat + dt * (phi31 * Ny0 + 2*phi32 * (Ny1 + Ny2) + phi33 * Ny3)
    return ux_new, uy_new


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--N",      type=int,   default=256)
    parser.add_argument("--Re",     type=float, default=1000.0)
    parser.add_argument("--dt",     type=float, default=5e-4)
    parser.add_argument("--T",      type=float, default=5.0)
    parser.add_argument("--method", type=str,   default="IMEX")
    args = parser.parse_args()

    Nx = Ny = args.N
    Lx = Ly = 2 * np.pi

    x = np.linspace(0, Lx, Nx, endpoint=False)
    y = np.linspace(0, Ly, Ny, endpoint=False)
    XX, YY = np.meshgrid(x, y, indexing='ij')

    u0 =  np.sin(XX) * np.cos(YY)
    v0 = -np.cos(XX) * np.sin(YY)

    # Kolmogorov forcing example (time-independent, but the interface supports t)

    def kolmogorov_force(XX, YY, t=None):
        return np.sin(2 * YY), np.zeros_like(XX)

    def zero_force(XX, YY, t=None):
        return np.zeros_like(XX), np.zeros_like(XX)

    solver = NSFourierSolver(Nx, Ny, Lx, Ly, args.Re, args.dt, u0, v0,
                             force=kolmogorov_force, method=args.method)
    save_every = max(1, int(args.T / args.dt) // 200)
    solver.solve(args.T, snapshot_every=save_every)

    print(f"Snapshots: {len(solver.snapshots)}, "
          f"final KE={solver.ke[-1]:.6f}, Ens={solver.ens[-1]:.6f}")
