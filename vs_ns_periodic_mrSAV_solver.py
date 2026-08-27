from typing import Tuple
from time import perf_counter

import numpy as np
import pyfftw
import pyfftw.interfaces.numpy_fft as fft
pyfftw.interfaces.cache.enable()
from scipy.optimize import newton, brentq
import numpy.typing as tnp


def _optimal_fftw_threads(n: int) -> int:
    if n <= 64:
        return 1
    elif n <= 128:
        return 4
    elif n <= 512:
        return 6
    else:
        return 8

class vs_mrSAV_Vorticity_Stream_Periodic_Solve():
    """
    variable step mean reverting Vorticity-Streamfunction formulation Solver in periodic domain.
    """
    def __init__(
            self,
            nu: float,
            ga: float,
            s_domain: Tuple[float,float ,float, float],
            discrete_num: Tuple[int,int],
            initial_condition: tnp.NDArray,
            force_term,
            step_method: str,
            root_selection: str = "legacy"):

        pyfftw.config.NUM_THREADS = _optimal_fftw_threads(max(discrete_num))

        step_methods = {
            "IMEX" : (self.IMEX, 1),
            "ETD" : (self.ETD, 1),
            "ETDMS2": (self.ETDMS2, 2),
            "ETDRK4" : (self.ETDRK4, 1),
            "ETD_mrGSAV_MS2_b": (self.ETD_mrGSAV_MS2_b, 2),
            "mr_SAV_BDF2": (self.mr_SAV_BDF2, 2),
        }

        if step_method not in step_methods:
            raise ValueError(f"Not supported step method: {step_method}")
        if root_selection not in {"legacy", "nearest", "farthest"}:
            raise ValueError(
                "root_selection must be 'legacy', 'nearest', or 'farthest'"
            )
        
        self.step, self.setup_step = step_methods[step_method]
        self.root_selection = root_selection
        self.cubic_root_history = []

        # viscous coefficient
        self.nu = nu
        # free parameter gamma
        self.ga = ga

        self.xa, self.ya, self.xb, self.yb = s_domain
        self.Nx, self.Ny = discrete_num
        self.hx = (self.xb - self.xa) / self.Nx
        self.hy = (self.yb - self.ya) / self.Ny
        self.h = self.hx*self.hy

        self.xn = np.linspace(self.xa, self.xb, self.Nx + 1)
        self.yn = np.linspace(self.ya, self.yb, self.Ny + 1)
        self.X,self.Y = np.meshgrid(self.xn,self.yn)

        # presudo spectral method
        self.mu_x = 2 * np.pi / (self.xb - self.xa)
        self.mu_y = 2 * np.pi / (self.yb - self.ya)

        k_x = np.zeros(self.Nx); k_x[0:self.Nx//2] = np.arange(0,self.Nx//2); k_x[self.Nx//2+1:] = np.arange(-self.Nx//2+1,0)
        k_y = np.zeros(self.Ny); k_y[0:self.Ny//2] = np.arange(0,self.Ny//2); k_y[self.Ny//2+1:] = np.arange(-self.Ny//2+1,0)
        self.D_x = (1j*self.mu_x*k_x)[np.newaxis,:]
        self.D_y = (1j*self.mu_y*k_y)[:,np.newaxis]

        k_xx = np.zeros(self.Nx); k_xx[0:self.Nx//2] = np.arange(0,self.Nx//2); k_xx[self.Nx//2:] = np.arange(-self.Nx//2,0)
        k_yy = np.zeros(self.Ny); k_yy[0:self.Ny//2] = np.arange(0,self.Ny//2); k_yy[self.Ny//2:] = np.arange(-self.Ny//2,0)
        self.D_xx = ((1j*self.mu_x*k_xx)**2)[np.newaxis,:]
        self.D_yy = ((1j*self.mu_y*k_yy)**2)[:,np.newaxis]

        self.Lap = self.D_xx + self.D_yy
        mask = np.zeros_like(self.Lap); mask[0,0] = 1
        self.inv_Lap = 1/(self.Lap + mask); self.inv_Lap[0,0] = 0

        # 2/3-rule dealiasing mask: zero wavenumbers |k| > N/3
        self.dealias_mask = np.ones((self.Nx, self.Ny), dtype=bool)
        kx_max = self.Nx // 3
        ky_max = self.Ny // 3
        self.dealias_mask[kx_max+1 : self.Nx-kx_max, :] = False
        self.dealias_mask[:, ky_max+1 : self.Ny-ky_max] = False

        # Linear Operator 
        self.L = self.linear_operator()

        self.f = force_term

        self.Omega0 = initial_condition[:-1,:-1] - np.mean(initial_condition[:-1,:-1])
        self.q0 = 1.0

    def ft(self,u):
        return fft.fft2(u)

    def ift(self,u):
        return fft.ifft2(u)

    def dealias(self, u_hat):
        return u_hat * self.dealias_mask

    def velocity2vorticity(self,u,v):
        f_u = self.ft(u); f_u[0,0] = 0
        f_v = self.ft(v); f_v[0,0] = 0
        omega = self.ift(f_v*self.D_x - f_u*self.D_y).real
        return omega

    def vorticity2stream(self,omega):
        fomega = self.ft(omega)
        return self.ift(-fomega*self.inv_Lap).real

    def stream2velocity(self,psi):
        u =  self.ift(self.ft(psi)*self.D_y).real
        v = -self.ift(self.ft(psi)*self.D_x).real
        return u,v

    def N(self,omega):
        omega_hat = self.dealias(self.ft(omega))
        omega_d   = self.ift(omega_hat).real

        psi_hat = self.dealias(-omega_hat * self.inv_Lap)
        u = self.ift(psi_hat * self.D_y).real
        v = self.ift(-psi_hat * self.D_x).real

        omega_x = self.ift(self.D_x * omega_hat).real
        omega_y = self.ift(self.D_y * omega_hat).real

        u_omega_x = self.ift(self.D_x * self.dealias(self.ft(u * omega_d))).real
        v_omega_y = self.ift(self.D_y * self.dealias(self.ft(v * omega_d))).real

        return -(u*omega_x + v*omega_y + u_omega_x + v_omega_y)/2

    def linear_operator(self):
        return  -self.nu*self.Lap
    
    def vorticity_energy(self,omega):
        omega_x = self.ift(self.D_x*self.ft(omega)).real
        omega_y = self.ift(self.D_y*self.ft(omega)).real
        u,v = self.stream2velocity(self.vorticity2stream(omega))

        Energy = (self.inner_product(u,u) + self.inner_product(v,v))/2
        Enstrophy = self.inner_product(omega,omega)/2
        Palinstrophy = (self.inner_product(omega_x,omega_x) + self.inner_product(omega_y,omega_y))/2
        return Energy, Enstrophy, Palinstrophy

    def vorticity_rhs(self, omega, t):
        linear = self.ift(-self.L*self.ft(omega)).real
        nonlinear = self.N(omega)
        force = self.f(self.X[:-1,:-1], self.Y[:-1,:-1], t)
        return linear + nonlinear + force

    def enstrophy_rate(self, omega, t):
        return self.inner_product(self.vorticity_rhs(omega, t), omega)

    def energy_rate(self, omega, t, enstrophy=None):
        if enstrophy is None:
            _, enstrophy, _ = self.vorticity_energy(omega)
        psi = self.vorticity2stream(omega)
        force = self.f(self.X[:-1, :-1], self.Y[:-1, :-1], t)
        injection = self.inner_product(psi, force)
        dissipation = -2 * self.nu * enstrophy
        return dissipation + injection

    def inner_product(self,f,g):
        return self.h*np.sum(f*g)
    
    def inner_product_ft(self,f_hat,g_hat):
        return (self.h*np.sum(f_hat*np.conj(g_hat))/self.Nx/self.Ny).real

    def IMEX(self,Omega_s, q_s, tn, tau_s):
        tau_n = tau_s[-1]
        self.phi1 = 1/(1 + tau_n*self.L)

        omega_n = Omega_s[-1]
        fomega_n = self.ft(omega_n)

        N_1 = self.N(omega_n)
        f_n = self.f(self.X[:-1,:-1],self.Y[:-1,:-1],tn)

        fomega_n1 = self.phi1*(fomega_n) + tau_n*self.phi1*self.ft(N_1 + f_n)
        return self.ift(fomega_n1).real, 1

    def ETD(self, Omega_s, q_s, tn, tau_s):
        tau_n = tau_s[-1]

        M = 16; dim = self.L.ndim
        r   = np.expand_dims(np.exp( 1j*np.pi*(np.arange(1,M+1) - .5)/M ), axis = list(range(dim)) )
        Lr  = np.expand_dims(self.L,axis=-1) + r

        self.phi0_L = np.exp(-tau_n*self.L)
        self.phi1_L = np.mean((1-np.exp(-tau_n*Lr))/(tau_n*Lr),axis=-1).real 

        omega_n   = Omega_s[-1]
        fomega_n  = self.ft(omega_n)

        N_n = self.N(omega_n); fN_n = self.ft(N_n)

        q_n = q_s[-1]
        p_n = q_n - 1

        fn = self.f(self.X[:-1,:-1],self.Y[:-1,:-1],tn)
        f_fn = self.ft(fn)

        fomega_n1 = self.phi0_L*fomega_n + tau_n*self.phi1_L*(f_fn); fomega_n1[0,0] = 0.+0j
        return self.ift(fomega_n1).real, p_n + 1

    def ETDMS2(self, Omega_s, q_s, tn, tau_s):
        tau_n = tau_s[-1]
        tau_nm = tau_s[-2]

        M = 16; dim = self.L.ndim
        r   = np.expand_dims(np.exp( 1j*np.pi*(np.arange(1,M+1) - .5)/M ), axis = list(range(dim)) )
        Lr  = np.expand_dims(self.L,axis=-1) + r

        self.phi0_L = np.exp(-tau_n*self.L)
        self.phi1_L = np.mean((1-np.exp(-tau_n*Lr))/(tau_n*Lr),axis=-1).real 

        omega_n   = Omega_s[-1]
        fomega_n  = self.ft(omega_n)
        omega_nm  = Omega_s[-2]

        fN_n = self.ft(self.N(omega_n))
        fN_nm = self._fN_cache if self._fN_cache is not None else self.ft(self.N(omega_nm))
        self._fN_cache = fN_n
        f_N12 = (tau_n/2 + tau_nm)/tau_nm*fN_n - (tau_n/2)/tau_nm*fN_nm

        q_n = q_s[-1]
        p_n = q_n - 1

        fn = self.f(self.X[:-1,:-1],self.Y[:-1,:-1],tn+tau_n/2)
        f_fn = self.ft(fn)

        fomega_n1 = self.phi0_L*fomega_n + tau_n*self.phi1_L*(f_N12 + f_fn); fomega_n1[0,0] = 0.+0j
        return self.ift(fomega_n1).real, p_n + 1

    def BDF2(self, Omega_s, q_s, tn, tau_s):
        tau_n  = tau_s[-1]
        tau_nm = tau_s[-2]
        rho = tau_n / tau_nm

        # variable-step BDF2 coefficients
        a0 = (1 + 2*rho) / (1 + rho)
        c1 = (1 + rho)**2 / (1 + 2*rho)     # coeff for omega_n
        c2 = rho**2 / (1 + 2*rho)           # coeff for omega_nm (subtracted)
        dt = tau_n / a0                     # effective step size

        phi_L  = 1 / (1 + dt*self.L)

        omega_n  = Omega_s[-1]; omega_nm = Omega_s[-2]
        fomega_n  = self.ft(omega_n); fomega_nm = self.ft(omega_nm)

        q_n  = q_s[-1]

        f_n = self.f(self.X[:-1,:-1], self.Y[:-1,:-1], tn + tau_n)

        N_2 = self.N((1+rho)*omega_n - rho*omega_nm)

        fomega_n1 = phi_L*(c1*fomega_n - c2*fomega_nm + dt*self.ft(N_2 + f_n))

        return self.ift(fomega_n1).real, q_n

    def _prepare_ETDRK4_coefficients(self, tau):
        tau = float(tau)
        if getattr(self, "_etdrk4_tau", None) == tau:
            return
        M = 16
        dim = self.L.ndim
        r  = np.expand_dims(np.exp( 1j*np.pi*(np.arange(1,M+1) - .5)/M ),axis = list(range(dim)) )
        Lr = tau*np.expand_dims(-self.L,axis=-1) + r

        self.phi10 = np.exp(-tau*self.L/2)
        self.phi11 = np.mean((1-np.exp(Lr/2))/(-Lr),axis=-1).real

        self.phi30 = np.exp(-tau*self.L)
        self.phi31 = np.mean((- 4 - Lr + np.exp(Lr)*(4 - 3*Lr + Lr**2))/(Lr)**3,axis=-1).real
        self.phi32 = np.mean((  2 + Lr + np.exp(Lr)*(-2+Lr))           /(Lr)**3,axis=-1).real
        self.phi33 = np.mean((- 4 - 3*Lr - Lr**2 + np.exp(Lr)*(4-Lr))  /(Lr)**3,axis=-1).real
        self._etdrk4_tau = tau

    def ETDRK4(self, Omega_s, q_s, tn, tau_s):
        tau = tau_s[-1]
        omega_n = Omega_s[-1]
        self._prepare_ETDRK4_coefficients(tau)

        fomega_n = self.ft(omega_n)
        N0 = self.ft(self.N(omega_n) + self.f(self.X[:-1,:-1],self.Y[:-1,:-1],tn))

        fomega_n1 = self.phi10*fomega_n  + tau*self.phi11*N0
        omega_n1 = self.ift(fomega_n1).real
        N1 =  self.ft(self.N(omega_n1) + self.f(self.X[:-1,:-1],self.Y[:-1,:-1], tn+tau/2))

        fomega_n2 = self.phi10*fomega_n  + tau*self.phi11*N1
        omega_n2 = self.ift(fomega_n2).real
        N2 =  self.ft(self.N(omega_n2) + self.f(self.X[:-1,:-1],self.Y[:-1,:-1], tn+tau/2))

        fomega_n3 = self.phi10*fomega_n1 + tau*self.phi11*(2*N2-N0)
        omega_n3 = self.ift(fomega_n3).real
        N3 =  self.ft(self.N(omega_n3) + self.f(self.X[:-1,:-1],self.Y[:-1,:-1], tn+tau))

        fomega_n4 = self.phi30*fomega_n  + tau*(self.phi31*N0 + 2*self.phi32*(N1 + N2) + self.phi33*N3)
        return self.ift(fomega_n4).real, 1

    def _solve_mrgsav_cubic(self, alpha, beta, C, tn, tau_n):
        coefficients = np.array(
            [beta, -beta, 1.0 - alpha - beta, alpha + beta - C],
            dtype=np.float64,
        )
        coefficient_scale = max(1.0, np.max(np.abs(coefficients)))
        degree_tol = 100 * np.finfo(np.float64).eps * coefficient_scale

        if abs(beta) <= degree_tol:
            slope = 1.0 - alpha
            if abs(slope) <= degree_tol:
                raise RuntimeError("Degenerate mrGSAV scalar equation")
            roots_all = np.array([(C - alpha) / slope], dtype=np.complex128)
            real_root_count = 1
            root_case = "linear"
            discriminant = np.nan
        else:
            normalized = coefficients / coefficient_scale
            a, b, c, d = normalized
            discriminant = (
                18 * a * b * c * d
                - 4 * b**3 * d
                + b**2 * c**2
                - 4 * a * c**3
                - 27 * a**2 * d**2
            )
            discriminant_tol = 1000 * np.finfo(np.float64).eps
            if discriminant > discriminant_tol:
                real_root_count = 3
                root_case = "three_distinct_real"
            elif discriminant < -discriminant_tol:
                real_root_count = 1
                root_case = "one_real"
            else:
                real_root_count = 3
                root_case = "multiple_real"
            roots_all = np.roots(normalized)

        real_mask = np.abs(roots_all.imag) <= 1e-8 * (1.0 + np.abs(roots_all.real))
        real_roots = np.sort(roots_all.real[real_mask])
        if real_roots.size == 0:
            raise RuntimeError("No numerically real root found for mrGSAV scalar equation")

        distinct_real_roots = []
        for root in real_roots:
            if (
                not distinct_real_roots
                or abs(root - distinct_real_roots[-1])
                > 1e-7 * (1.0 + abs(root))
            ):
                distinct_real_roots.append(root)

        def scalar_equation(p):
            return (
                beta * p**3
                - beta * p**2
                + (1.0 - alpha - beta) * p
                + alpha + beta - C
            )

        distances = np.abs(real_roots)
        if self.root_selection == "nearest":
            selected_root = real_roots[np.argmin(distances)]
        else:
            selected_root = real_roots[np.argmax(distances)]

        residual_scale = max(
            1.0,
            abs(beta * selected_root**3),
            abs(beta * selected_root**2),
            abs((1.0 - alpha - beta) * selected_root),
            abs(alpha + beta - C),
        )
        residual = abs(scalar_equation(selected_root))
        if residual > 1e-9 * residual_scale:
            raise RuntimeError(
                f"Selected mrGSAV root has residual {residual:.3e}"
            )

        derivative = (
            3.0 * beta * selected_root**2
            - 2.0 * beta * selected_root
            + 1.0 - alpha - beta
        )
        self.cubic_root_history.append(
            {
                "time": float(tn + tau_n),
                "tau": float(tau_n),
                "alpha": float(alpha),
                "beta": float(beta),
                "target": 0.0,
                "predictor": float(C),
                "discriminant": float(discriminant),
                "root_case": root_case,
                "real_root_count": real_root_count,
                "distinct_real_root_count": len(distinct_real_roots),
                "roots_all": roots_all.copy(),
                "real_roots": real_roots.copy(),
                "selected_root": float(selected_root),
                "selected_distance": float(abs(selected_root)),
                "selected_derivative": float(derivative),
                "selected_residual": float(residual),
                "selected_q_positive": bool(selected_root > -1.0),
                "selection": self.root_selection,
            }
        )
        return selected_root

    def ETD_mrGSAV_MS2_b(self, Omega_s, q_s, tn, tau_s):
        tau_n = tau_s[-1]
        tau_nm = tau_s[-2]

        M = 16; dim = self.L.ndim
        r   = np.expand_dims(np.exp( 1j*np.pi*(np.arange(1,M+1) - .5)/M ), axis = list(range(dim)) )
        Lr  = np.expand_dims(self.L,axis=-1) + r
        gar = np.expand_dims(self.ga,axis=-1) + r

        self.phi0_L = np.exp(-tau_n*self.L)
        self.phi1_L = np.mean((1-np.exp(-tau_n*Lr))/(tau_n*Lr),axis=-1).real 

        self.phi0_ga = np.exp(-tau_n*self.ga)
        self.phi1_ga = np.mean((1-np.exp(-tau_n*gar))/(tau_n*gar),axis=-1).real 

        omega_n   = Omega_s[-1]
        fomega_n  = self.ft(omega_n)
        omega_nm  = Omega_s[-2]

        fN_n = self.ft(self.N(omega_n))
        fN_nm = self._fN_cache if self._fN_cache is not None else self.ft(self.N(omega_nm))
        self._fN_cache = fN_n
        f_N12 = (tau_n/2 + tau_nm)/tau_nm*fN_n - (tau_n/2)/tau_nm*fN_nm

        q_n = q_s[-1]
        p_n = q_n - 1

        fn = self.f(self.X[:-1,:-1],self.Y[:-1,:-1],tn+tau_n/2)
        f_fn = self.ft(fn)

        A = tau_n*self.inner_product_ft(self.phi1_L * f_N12, self.phi0_L*fomega_n + tau_n*self.phi1_L*f_fn)
        B = tau_n**2*self.inner_product_ft(self.phi1_L * f_N12, self.phi1_L*f_N12)
        C =  self.phi0_ga*p_n
        
        Tgam = 0.1
        if self.root_selection == "legacy":
            f = lambda p: p + (1-p)*A*Tgam + (p**3 - p**2 - p + 1)*B*Tgam - C
            try:
                p_n1 = newton(f, 0.)
            except RuntimeError:
                lo, hi = -10., 10.
                if f(lo) * f(hi) > 0:
                    lo, hi = -100., 100.
                p_n1 = brentq(f, lo, hi)
        else:
            p_n1 = self._solve_mrgsav_cubic(
                Tgam * A,
                Tgam * B,
                C,
                tn,
                tau_n,
            )

        fomega_n1 = self.phi0_L*fomega_n + tau_n*self.phi1_L*((1 - p_n1**2)*f_N12 + f_fn); fomega_n1[0,0] = 0.+0j
        return self.ift(fomega_n1).real, p_n1 + 1
    
    def mr_SAV_BDF2(self, Omega_s, q_s, tn, tau_s):
        tau_n  = tau_s[-1]
        tau_nm = tau_s[-2]
        rho = tau_n / tau_nm

        # variable-step BDF2 coefficients
        a0 = (1 + 2*rho) / (1 + rho)
        c1 = (1 + rho)**2 / (1 + 2*rho)     # coeff for omega_n
        c2 = rho**2 / (1 + 2*rho)           # coeff for omega_nm (subtracted)
        dt = tau_n / a0                     # effective step size

        phi_L  = 1 / (1 + dt*self.L)
        phi_ga = 1 / (1 + dt*self.ga)

        omega_n  = Omega_s[-1]
        omega_nm = Omega_s[-2]
        fomega_n  = self.ft(omega_n)
        fomega_nm = self.ft(omega_nm)

        q_n  = q_s[-1]
        q_nm = q_s[-2]

        f_n = self.f(self.X[:-1,:-1], self.Y[:-1,:-1], tn + tau_n)

        N_2 = self.N((1+rho)*omega_n - rho*omega_nm)

        LHS = 1 + dt**2*phi_ga*self.inner_product(N_2, self.ift(phi_L*self.ft(N_2)).real)
        RHS = phi_ga*(c1*q_n - c2*q_nm) \
            + dt*phi_ga*self.ga \
            - dt*phi_ga*self.inner_product(N_2, self.ift(phi_L*(c1*fomega_n - c2*fomega_nm + dt*self.ft(f_n))).real)

        q_n1 = RHS / LHS
        fomega_n1 = phi_L*(c1*fomega_n - c2*fomega_nm + dt*self.ft(q_n1*N_2 + f_n))

        return self.ift(fomega_n1).real, q_n1

    def ETD_mrGSAV_MS12_b(self, Omega_s, q_s, tn, tau_s, fN_n=None, fN_nm=None):
        tau_n = tau_s[-1]
        tau_nm = tau_s[-2]

        M = 16; dim = self.L.ndim
        r   = np.expand_dims(np.exp( 1j*np.pi*(np.arange(1,M+1) - .5)/M ), axis = list(range(dim)) )
        Lr  = np.expand_dims(self.L,axis=-1) + r
        gar = np.expand_dims(self.ga,axis=-1) + r

        self.phi0_L = np.exp(-tau_n*self.L)
        self.phi1 = 1/(1 + tau_n*self.L)
        self.phi1_L = np.mean((1-np.exp(-tau_n*Lr))/(tau_n*Lr),axis=-1).real

        self.phi0_ga = np.exp(-tau_n*self.ga)
        self.phi1_ga = np.mean((1-np.exp(-tau_n*gar))/(tau_n*gar),axis=-1).real

        omega_n   = Omega_s[-1]
        fomega_n  = self.ft(omega_n)
        omega_nm  = Omega_s[-2]

        if fN_n is None:
            fN_n = self.ft(self.N(omega_n))
        if fN_nm is None:
            fN_nm = self.ft(self.N(omega_nm))
        f_N12 = (tau_n/2 + tau_nm)/tau_nm*fN_n - (tau_n/2)/tau_nm*fN_nm

        q_n = q_s[-1]
        p_n = q_n - 1

        fn = self.f(self.X[:-1,:-1],self.Y[:-1,:-1],tn+tau_n/2)
        f_fn = self.ft(fn)

        A = tau_n*self.inner_product_ft(self.phi1_L * f_N12, self.phi0_L*fomega_n + tau_n*self.phi1_L*f_fn)
        B = tau_n**2*self.inner_product_ft(self.phi1_L * f_N12, self.phi1_L*f_N12)
        C =  self.phi0_ga*p_n
        
        Tgam = 0.1
        f = lambda p: p + (1-p)*A*Tgam + (p**3 - p**2 - p + 1)*B*Tgam - C
        try:
            p_n1 = newton(f, 0.)
        except RuntimeError:
            lo, hi = -10., 10.
            if f(lo) * f(hi) > 0:
                lo, hi = -100., 100.
            p_n1 = brentq(f, lo, hi)

        fac_1 = 1 - p_n1**2
        fac_2 = 1 + p_n1
        # fac_3 = 1 - p_n2**2

        fomega_n1 = self.phi0_L*fomega_n + tau_n*self.phi1_L*(fac_1*f_N12 + f_fn); fomega_n1[0,0] = 0.+0j
        # fomega_n2 = self.phi0_L*(fomega_n) + tau_n*self.phi1_L*(fac_2*f_N12 + f_fn); fomega_n2[0,0] = 0.+0j
        fomega_n2 = self.phi1*(fomega_n) + tau_n*self.phi1*(fac_2*f_N12 + f_fn); fomega_n2[0,0] = 0.+0j
        # fomega_n2 = self.phi0_L*(fomega_n) + tau_n*self.phi1_L*(fac_3*f_N12 + f_fn); fomega_n2[0,0] = 0.+0j
        return self.ift(fomega_n1).real, self.ift(fomega_n2).real, p_n1 + 1, A, B

    def init_record(self, M_max):
        self.Energy = np.empty(M_max + 1, dtype=np.float64)
        self.Energy_rate = np.empty(M_max + 1, dtype=np.float64)
        self.Enstrophy = np.empty(M_max + 1, dtype=np.float64)
        self.Enstrophy_rate = np.empty(M_max + 1, dtype=np.float64)
        self.Palinstrophy =  np.empty(M_max + 1, dtype=np.float64)
        self.Mx = np.empty(M_max + 1, dtype=np.float64) 
        self.Energy[0], self.Enstrophy[0], self.Palinstrophy[0] = self.vorticity_energy(self.Omega0)
        self.Energy_rate[0] = self.energy_rate(self.Omega0, getattr(self, "T0", 0.0), self.Enstrophy[0])
        self.Enstrophy_rate[0] = self.enstrophy_rate(self.Omega0, getattr(self, "T0", 0.0))
        self.Mx[0] = np.max(self.Omega0)

    def result_record(self,i,Omega,q):
        self.Energy[i+1], self.Enstrophy[i+1], self.Palinstrophy[i+1] = self.vorticity_energy(Omega)
        self.Energy_rate[i+1] = self.energy_rate(Omega, self.tn[i+1], self.Enstrophy[i+1])
        self.Enstrophy_rate[i+1] = self.enstrophy_rate(Omega, self.tn[i+1])
        self.Mx[i+1] = np.max(Omega)
        msg = f"Vorticity Energy:{self.Energy[i+1]:.4f}, Energy rate:{self.Energy_rate[i+1]:.4e}, Enstrophy:{self.Enstrophy[i+1]:.4f}, Enstrophy rate:{self.Enstrophy_rate[i+1]:.4e}, Palinstrophy:{self.Palinstrophy[i+1]:.4f}, Maximum:{self.Mx[i+1]:.2f}, |q-1|:{np.abs(q - 1):.4e}"
        return msg
    
    def extend_array(self):
        # 扩展数组容量
        new_M_max = int(len(self.tn) * 1.5)  # 扩展为原来的1.5倍
        
        # 扩展 tau 数组
        new_tau = np.empty(new_M_max, dtype=np.float64)
        new_tau[:len(self.tau)] = self.tau
        self.tau = new_tau
        
        # 扩展 tn, q 数组
        new_tn = np.empty(new_M_max + 1, dtype=np.float64)
        new_tn[:len(self.tn)] = self.tn
        self.tn = new_tn
        
        new_q = np.empty(new_M_max + 1, dtype=np.float64)
        new_q[:len(self.q)] = self.q
        self.q = new_q

        if hasattr(self, 'ref_err'):
            new_ref_err = np.empty(new_M_max, dtype=np.float64)
            new_ref_err[:len(self.ref_err)] = self.ref_err
            self.ref_err = new_ref_err

            new_rel_err = np.empty(new_M_max, dtype=np.float64)
            new_rel_err[:len(self.rel_err)] = self.rel_err
            self.rel_err = new_rel_err

            new_ref_err_p = np.empty(new_M_max, dtype=np.float64)
            new_ref_err_p[:len(self.ref_err_p)] = self.ref_err_p
            self.ref_err_p = new_ref_err_p

            new_ref_err_b = np.empty(new_M_max, dtype=np.float64)
            new_ref_err_b[:len(self.ref_err_b)] = self.ref_err_b
            self.ref_err_b = new_ref_err_b

            new_A_n = np.empty(new_M_max + 1, dtype=np.float64)
            new_A_n[:len(self.A_n)] = self.A_n
            self.A_n = new_A_n

            new_B_n = np.empty(new_M_max + 1, dtype=np.float64)
            new_B_n[:len(self.B_n)] = self.B_n
            self.B_n = new_B_n
        
        # 扩展 Omega 数组
        if hasattr(self, 'Omega'):
            new_Omega = np.empty([new_M_max + 1] + list(self.Omega0.shape), dtype=np.float64)
            new_Omega[:len(self.Omega)] = self.Omega
            self.Omega = new_Omega
        
        # 扩展记录数组
        new_Energy = np.empty(new_M_max + 1, dtype=np.float64)
        new_Energy[:len(self.Energy)] = self.Energy
        self.Energy = new_Energy

        new_Energy_rate = np.empty(new_M_max + 1, dtype=np.float64)
        new_Energy_rate[:len(self.Energy_rate)] = self.Energy_rate
        self.Energy_rate = new_Energy_rate
        
        new_Enstrophy = np.empty(new_M_max + 1, dtype=np.float64)
        new_Enstrophy[:len(self.Enstrophy)] = self.Enstrophy
        self.Enstrophy = new_Enstrophy

        new_Enstrophy_rate = np.empty(new_M_max + 1, dtype=np.float64)
        new_Enstrophy_rate[:len(self.Enstrophy_rate)] = self.Enstrophy_rate
        self.Enstrophy_rate = new_Enstrophy_rate
        
        new_Palinstrophy = np.empty(new_M_max + 1, dtype=np.float64)
        new_Palinstrophy[:len(self.Palinstrophy)] = self.Palinstrophy
        self.Palinstrophy = new_Palinstrophy
        
        new_Mx = np.empty(new_M_max + 1, dtype=np.float64)
        new_Mx[:len(self.Mx)] = self.Mx
        self.Mx = new_Mx

        new_cpu_time = np.empty(new_M_max + 1, dtype=np.float64)
        new_cpu_time[:len(self.cpu_time)] = self.cpu_time
        self.cpu_time = new_cpu_time

    def solve_adaptive_step(self, t_span, tau_min, tau_max, snapshot=None, compute_ref_err=False, rho=0.9, rtol=1e-3, rtol_q=1e-3, r=1/2, ref_substeps=2):

        # 初始化自适应时间步长参数
        if not hasattr(self, 'rho'):
            self.rho = rho
        if not hasattr(self, 'rtol'):
            self.rtol = rtol
        if not hasattr(self, 'rtol_q'):
            self.rtol_q = rtol_q
        if not hasattr(self, 'r'):
            self.r = r

        if not isinstance(t_span, (list, tuple)) or len(t_span) != 2:
            raise ValueError("t_span 必须是长度为 2 的元组")
        if tau_min <= 0 or tau_max <= 0 or tau_min >= tau_max:
            raise ValueError("时间步长 tau_min, tau_max 必须大于 0, 且 tau_min < tau_max")
        if not isinstance(ref_substeps, int) or ref_substeps < 1:
            raise ValueError("ref_substeps 必须是 >= 1 的整数")

        self.T0, self.T = t_span

        snapshot_mode = snapshot is not None
        if snapshot_mode:
            snapshot = np.sort(np.array(snapshot))
            snapshot_index = 0
            has_snapshots = len(snapshot) > 0
            snapshot_atol = 1e-12

        M_max = int(np.ceil((self.T - self.T0)/tau_max)) + 200
        self.tau     = np.empty(M_max, dtype=np.float64)
        self.tn      = np.empty(M_max + 1, dtype=np.float64)
        self.q       = np.empty(M_max + 1, dtype=np.float64)
        self.ref_err = np.zeros(M_max + 1, dtype=np.float64)
        self.rel_err = np.zeros(M_max + 1, dtype=np.float64)
        self.ref_err_p = np.zeros(M_max + 1, dtype=np.float64)
        self.ref_err_b = np.zeros(M_max + 1, dtype=np.float64)
        self.A_n = np.zeros(M_max + 1, dtype=np.float64)
        self.B_n = np.zeros(M_max + 1, dtype=np.float64)
        self.cpu_time  = np.empty(M_max + 1, dtype=np.float64)

        # Omega_temp_2 在两种模式下都分配，用于计算参考误差
        self.Omega_temp_2 = np.empty([self.setup_step+1] + list(self.Omega0.shape), dtype=np.float64)

        if snapshot_mode:
            if hasattr(self, 'Omega'):
                del self.Omega
            self.Omega_temp = np.empty([self.setup_step+1] + list(self.Omega0.shape), dtype=np.float64)
            if has_snapshots:
                self.snapshot_Omega = np.empty([len(snapshot)] + list(self.Omega0.shape), dtype=np.float64)
                self.snapshot_tn    = np.empty(len(snapshot), dtype=np.float64)
            else:
                self.snapshot_Omega = np.array([])
                self.snapshot_tn    = np.array([])
        else:
            self.Omega = np.zeros([M_max + 1] + list(self.Omega0.shape), dtype=np.float64)

        # 初始化记录
        self.init_record(M_max)

        # 初始步设置
        self.tn[0]       = self.T0
        self.q[0]        = self.q0
        self.cpu_time[0] = 0.0
        if snapshot_mode:
            self.Omega_temp[0]   = self.Omega0
            self.Omega_temp_2[0] = self.Omega0
        else:
            self.Omega[0]        = self.Omega0
            self.Omega_temp_2[0] = self.Omega0
        
        
        if snapshot_mode:
            def save_initial_snapshots():
                nonlocal snapshot_index
                while has_snapshots and snapshot_index < len(snapshot) and snapshot[snapshot_index] <= self.T0 + snapshot_atol:
                    self.snapshot_Omega[snapshot_index] = self.Omega0
                    self.snapshot_tn[snapshot_index] = self.T0
                    snapshot_index += 1

            def save_closest_snapshots(t_prev, omega_prev, t_curr, omega_curr):
                nonlocal snapshot_index
                while has_snapshots and snapshot_index < len(snapshot) and snapshot[snapshot_index] <= t_curr + snapshot_atol:
                    target = snapshot[snapshot_index]
                    if abs(t_prev - target) <= abs(t_curr - target):
                        self.snapshot_Omega[snapshot_index] = omega_prev
                        self.snapshot_tn[snapshot_index] = t_prev
                    else:
                        self.snapshot_Omega[snapshot_index] = omega_curr
                        self.snapshot_tn[snapshot_index] = t_curr
                    snapshot_index += 1

            save_initial_snapshots()

        # 使用 ETDRK4 计算初始步，确保有足够的历史数据用于多步方法。
        for i in range(self.setup_step-1):
            self.tau[i] = 1e-3/2
            if snapshot_mode:
                self.Omega_temp[i+1], self.q[i+1] = self.ETDRK4(
                    self.Omega_temp[i:i+1], self.q[i:i+1], self.tn[i], self.tau[i:i+1]
                )
                self.Omega_temp_2[i+1] = self.Omega_temp[i+1]
            else:
                self.Omega[i+1], self.q[i+1] = self.ETDRK4(
                    self.Omega[i:i+1], self.q[i:i+1], self.tn[i], self.tau[i:i+1]
                )
                self.Omega_temp_2[i+1] = self.Omega[i+1]
            self.tn[i+1] = self.tn[i] + self.tau[i]
            self.cpu_time[i+1] = 0.0
            self.result_record(i, self.Omega_temp[i+1] if snapshot_mode else self.Omega[i+1], self.q[i+1])
            if snapshot_mode:
                save_closest_snapshots(self.tn[i], self.Omega_temp[i], self.tn[i+1], self.Omega_temp[i+1])

        self.tau[self.setup_step-1] = 1e-3/2

        # 主时间循环
        index = self.setup_step
        prev_fN_n = None
        while self.tn[index-1] < self.T:
            if index >= len(self.tn) - 10:
                self.extend_array()

            omega_hist = self.Omega_temp[:-1] if snapshot_mode else self.Omega[index-self.setup_step:index]
            omega_last = self.Omega_temp[:-1] if snapshot_mode else self.Omega[index-1:index]

            start_time = perf_counter()  # 包含 N 预计算和所有被拒绝步的计算时间，与 solve_fix_step 计时口径一致
            # 预计算 N(omega_n) 和 N(omega_nm) 的傅里叶变换，避免步长被拒绝时重复计算
            fN_nm = self.ft(self.N(omega_hist[-2])) if prev_fN_n is None else prev_fN_n
            fN_n  = self.ft(self.N(omega_hist[-1]))
            while True:
                Omega_2, Omega_1, q_2, A_n, B_n = self.ETD_mrGSAV_MS12_b(
                    omega_hist,
                    self.q[index-self.setup_step:index],
                    self.tn[index-1],
                    self.tau[index-self.setup_step:index],
                    fN_n=fN_n,
                    fN_nm=fN_nm,
                )

                # 计算误差
                Error_u = np.sqrt(self.inner_product(Omega_2 - Omega_1, Omega_2 - Omega_1)/self.inner_product(Omega_2, Omega_2)) + 1e-16
                Error_q = np.abs(q_2 - 1) + 1e-16

                # 计算新的时间步长
                tau = self.rho * min( (self.rtol / Error_u) ** (self.r) , self.rtol_q / Error_q) * self.tau[index-1]
                tau = max(tau_min, min(tau, tau_max))

                if (Error_u <= self.rtol and Error_q <= self.rtol_q) or tau <= tau_min:
                    # 接受当前时间步
                    if snapshot_mode:
                        self.Omega_temp[-1] = Omega_2
                    else:
                        self.Omega[index] = Omega_2
                    break
                else:
                    self.tau[index-1] = tau

            end_time = perf_counter()
            cpu_time = end_time - start_time
            self.cpu_time[index] = self.cpu_time[index-1] + cpu_time

            self.q[index]         = q_2
            self.tau[index]       = tau
            self.tn[index]        = self.tn[index-1] + self.tau[index-1]
            self.rel_err[index]   = Error_u
            self.ref_err_p[index] = Error_q
            self.A_n[index]       = A_n
            self.B_n[index]       = B_n

            if snapshot_mode and compute_ref_err:
                diff_b  = self.N(Omega_2) - self.N(Omega_1)
                Error_b = np.sqrt(self.inner_product(diff_b, diff_b)/self.inner_product(self.N(Omega_2), self.N(Omega_2)))
            else:
                Error_b = 0.0
            self.ref_err_b[index] = Error_b

            if compute_ref_err:
                ref_tau = self.tau[index-1] / ref_substeps
                ref_Omega = self.Omega_temp_2[-2]
                ref_q = self.q[index-1]
                ref_t = self.tn[index-1]
                for _ in range(ref_substeps):
                    ref_Omega, ref_q = self.ETDRK4(
                        np.array([ref_Omega]),
                        np.array([ref_q]),
                        ref_t,
                        np.array([ref_tau])
                    )
                    ref_t += ref_tau
                ref_err = np.sqrt(self.inner_product(Omega_2 - ref_Omega, Omega_2 - ref_Omega))
                self.ref_err[index] = ref_err / np.sqrt(self.inner_product(Omega_2, Omega_2)) + 1e-16
                self.Omega_temp_2[-1] = ref_Omega
            else:
                self.ref_err[index] = 0.0

            if snapshot_mode:
                save_closest_snapshots(self.tn[index-1], self.Omega_temp[-2], self.tn[index], self.Omega_temp[-1])

            current_Omega = self.Omega_temp[-1] if snapshot_mode else self.Omega[index]
            msg = self.result_record(index-1, current_Omega, self.q[index])
            print(f"\r {self.tn[index-1]:.6f}\\{self.T}, tau = {self.tau[index-1]:.6e}, Elapse: {cpu_time:.3f} s, " + msg + " "*5, end="")

            # 滚动更新
            if snapshot_mode:
                self.Omega_temp[:-1] = self.Omega_temp[1:]
            self.Omega_temp_2[:-1] = self.Omega_temp_2[1:]

            prev_fN_n = fN_n
            index += 1

        print("")
        # 结果整理 - 裁剪数组到实际使用的大小
        self.tau          = self.tau[:index]
        self.tn           = self.tn[:index]
        self.q            = self.q[:index]
        self.Energy       = self.Energy[:index]
        self.Energy_rate  = self.Energy_rate[:index]
        self.Enstrophy    = self.Enstrophy[:index]
        self.Enstrophy_rate = self.Enstrophy_rate[:index]
        self.Palinstrophy = self.Palinstrophy[:index]
        self.Mx           = self.Mx[:index]
        self.ref_err      = self.ref_err[:index]
        self.rel_err      = self.rel_err[:index]
        self.ref_err_p    = self.ref_err_p[:index]
        self.ref_err_b    = self.ref_err_b[:index]
        self.A_n          = self.A_n[:index]
        self.B_n          = self.B_n[:index]
        self.cpu_time     = self.cpu_time[:index]

        if snapshot_mode:
            self.Omega = self.snapshot_Omega
            self.tn_s  = self.snapshot_tn
        else:
            self.Omega = self.Omega[:index]

    def solve_fix_step(self, t_span, tau, snapshot=None):

        if not isinstance(t_span, tuple) or len(t_span) != 2:
            raise ValueError("t_span 必须是长度为 2 的元组")
        if tau <= 0:
            raise ValueError("时间步长 tau 必须大于 0")

        self.T0, self.T = t_span
        self.cubic_root_history = []
        M_float = (self.T - self.T0) / tau
        M = int(np.ceil(M_float))
        if M <= 0:
            raise ValueError("固定步长模式要求 T > T0")
        self.tau = np.full(M, tau, dtype=np.float64)
        self.tn = self.T0 + tau * np.arange(M + 1, dtype=np.float64)

        snapshot_mode = snapshot is not None
        if snapshot_mode:
            snapshot = np.sort(np.asarray(snapshot, dtype=np.float64))
            snapshot_indices = np.rint((snapshot - self.T0) / tau).astype(np.int64)
            grid_atol = max(1e-12, abs(tau) * 1e-8)
            if (
                np.any(snapshot_indices < 0)
                or np.any(snapshot_indices > M)
                or not np.allclose(self.tn[snapshot_indices], snapshot, rtol=0.0, atol=grid_atol)
            ):
                raise ValueError("snapshot 必须落在固定步长时间网格上")
            snapshot_cursor = 0
            self.snapshot_Omega = np.empty([len(snapshot)] + list(self.Omega0.shape), dtype=np.float64)
            self.snapshot_tn = self.tn[snapshot_indices].copy()
            self.Omega_temp = np.empty([self.setup_step + 1] + list(self.Omega0.shape), dtype=np.float64)
        else:
            snapshot_indices = np.array([], dtype=np.int64)
            snapshot_cursor = 0
            self.Omega = np.zeros([M + 1] + list(self.Omega0.shape), dtype=np.float64)

        self.q        = np.zeros(M + 1, dtype=np.float64)
        self.cpu_time = np.zeros(M + 1, dtype=np.float64)

        # 初始化记录
        self.init_record(M)

        def save_snapshot(index, omega):
            nonlocal snapshot_cursor
            while snapshot_cursor < len(snapshot_indices) and snapshot_indices[snapshot_cursor] == index:
                self.snapshot_Omega[snapshot_cursor] = omega
                snapshot_cursor += 1

        # 初始步设置
        self.q[0] = self.q0
        self.cpu_time[0] = 0.0
        if snapshot_mode:
            self.Omega_temp[0] = self.Omega0
        else:
            self.Omega[0] = self.Omega0
        save_snapshot(0, self.Omega0)

        # 使用 ETDRK4 启动多步法。
        for i in range(1, self.setup_step):
            omega_prev = self.Omega_temp[i-1:i] if snapshot_mode else self.Omega[i-1:i]
            omega_new, self.q[i] = self.ETDRK4(
                omega_prev,
                self.q[i-1:i],
                self.tn[i-1],
                self.tau[i-1:i]
            )

            if snapshot_mode:
                self.Omega_temp[i] = omega_new
            else:
                self.Omega[i] = omega_new
            self.cpu_time[i] = 0.0
            self.result_record(i-1, omega_new, self.q[i])
            save_snapshot(i, omega_new)

        # 主时间循环
        self._fN_cache = None
        if getattr(self.step, "__name__", "") == "ETDRK4":
            self._prepare_ETDRK4_coefficients(tau)

        for index in range(self.setup_step, M + 1):
            start_time = perf_counter()
            omega_hist = self.Omega_temp[:-1] if snapshot_mode else self.Omega[index-self.setup_step:index]
            omega_new, self.q[index] = self.step(
                omega_hist,
                self.q[index-self.setup_step:index],
                self.tn[index-1],
                self.tau[index-self.setup_step:index]
            )
            if snapshot_mode:
                self.Omega_temp[-1] = omega_new
            else:
                self.Omega[index] = omega_new
            end_time = perf_counter()

            cpu_time = end_time - start_time
            self.cpu_time[index] = self.cpu_time[index-1] + cpu_time

            msg = self.result_record(index-1, omega_new, self.q[index])
            print(f"\r {self.tn[index]:.6f}\\{self.T}, Elapse: {cpu_time:.3f} s, " + msg + " "*5, end="")
            save_snapshot(index, omega_new)

            if snapshot_mode:
                self.Omega_temp[:-1] = self.Omega_temp[1:]

        print("")
        if snapshot_mode:
            self.Omega = self.snapshot_Omega
            self.tn_s  = self.snapshot_tn

    def solve_random_step(self, t_span, M):

        if not isinstance(t_span, tuple) or len(t_span) != 2:
            raise ValueError("t_span 必须是长度为 2 的元组")
        if M <= 0:
            raise ValueError("时间步数 M 必须大于 0")

        self.T0, self.T = t_span
        eps        = np.random.uniform(0., 1., M)
        self.tau   = (self.T - self.T0)* eps / np.sum(eps)
        self.tn    = np.zeros(M + 1, dtype=np.float64)
        self.tn[0] = self.T0; self.tn[1:] = np.cumsum(self.tau) + self.T0

        self.q     = np.empty(M + 1, dtype=np.float64)
        self.Omega = np.zeros([M + 1] + list(self.Omega0.shape), dtype=np.float64)

        # 初始化记录
        self.init_record(M)

        # 初始步设置
        self.q[0]   = self.q0
        self.Omega[0] = self.Omega0

        # 使用ETDRK4计算初始步，确保有足够的历史数据用于高阶方法
        for i in range(1, self.setup_step):
            # 对于每个初始步，使用固定的tau_min
            # 调用ETDRK4计算下一步
            self.Omega[i], self.q[i] = self.ETDRK4(
                self.Omega[i-1:i],   # 传递最近1个Omega值（ETDRK4需要）
                self.q[i-1:i],       # 传递最近1个q值（ETDRK4需要）
                self.tn[i-1],        # 当前时间点
                self.tau[i-1:i]                  # 时间步长
            )
            # 记录结果
            self.result_record(i-1, self.Omega[i], self.q[i])
        
        # 主时间循环
        self._fN_cache = None
        index = self.setup_step
        for index in range(self.setup_step, M + 1):

            start_time = perf_counter()
            self.Omega[index], self.q[index] = self.step(
                self.Omega[index-self.setup_step:index],
                self.q[index-self.setup_step:index],
                self.tn[index-1],
                self.tau[index-self.setup_step:index]
            )

            end_time = perf_counter()
            cpu_time = end_time - start_time

            # 更新显示信息
            msg = self.result_record(index-1, self.Omega[index], self.q[index])
            print(f"\r {self.tn[index]:.6f}\\{self.T}, tau = {self.tau[index-1]:.6e}, Elapse: {cpu_time:.3f} s, " + msg + " "*5, end="")

        print("")

    def solve_given_tau(self, t_span, tau):

        if not isinstance(t_span, tuple) or len(t_span) != 2:
            raise ValueError("t_span 必须是长度为 2 的元组")

        self.tau = tau
        self.T0, self.T = t_span
        self.T = self.T0 + np.sum(self.tau)
        M = len(tau)

        self.tn = np.zeros(M + 1, dtype=np.float64)
        self.tn[0] = self.T0; self.tn[1:] = np.cumsum(self.tau) + self.T0

        self.q   = np.empty(M + 1, dtype=np.float64)
        self.Omega = np.zeros([M + 1] + list(self.Omega0.shape), dtype=np.float64)

        # 初始化记录
        self.init_record(M)

        # 初始步设置
        self.q[0]   = self.q0
        self.Omega[0] = self.Omega0

        # 使用ETDRK4计算初始步，确保有足够的历史数据用于高阶方法
        for i in range(self.setup_step):
            # 对于每个初始步，使用固定的tau_min
            # 调用ETDRK4计算下一步
            self.Omega[i+1], self.q[i+1] = self.ETDRK4(
                self.Omega[i:i+1],   # 传递最近1个Omega值（ETDRK4需要）
                self.q[i:i+1],       # 传递最近1个q值（ETDRK4需要）
                self.tn[i],        # 当前时间点
                self.tau[i:i+1]                  # 时间步长
            )
            # 记录结果
            self.result_record(i-1, self.Omega[i], self.q[i])
        
        # 主时间循环
        self._fN_cache = None
        for index in range(self.setup_step+1, M + 1):

            start_time = perf_counter()
            self.Omega[index], self.q[index] = self.step(
                self.Omega[index-self.setup_step:index],
                self.q[index-self.setup_step:index],
                self.tn[index-1],
                self.tau[index-self.setup_step:index]
            )

            end_time = perf_counter()
            cpu_time = end_time - start_time

            # 更新显示信息
            msg = self.result_record(index-1, self.Omega[index], self.q[index])
            print(f"\r {self.tn[index]:.6f}\\{self.T}, tau = {self.tau[index-1]:.6f}, Elapse: {cpu_time:.3f} s, " + msg + " "*5, end="")

        print("")

vs_mrSAV_Vorticity_Stream_Periodic_Solver = vs_mrSAV_Vorticity_Stream_Periodic_Solve

if __name__ == "__main__":
    # 测试代码
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description="Test the periodic mrSAV solver.")
    parser.add_argument("--nu", type=float, default=1/100, help="Reynolds number")
    parser.add_argument("--gam", type=float, default=1000, help="Gamma parameter")
    parser.add_argument("--M", type=str, default="ETD_mrGSAV_MS2_b", help="Step method to use")
    # parser.add_argument("--tau", type=float, default=0.001, help="Time step")

    args = parser.parse_args()

    # test_code
    nu = args.nu
    gam = args.gam

    s_domain = (0,0, 2*np.pi, 2*np.pi)
    discrete_num = [128, 128]

    xn = np.linspace(s_domain[0],s_domain[2],discrete_num[0]+1)
    yn = np.linspace(s_domain[1],s_domain[3],discrete_num[1]+1)
    X,Y = np.meshgrid(xn,yn)

    t_period = (0, 1)

    def periodic_f(t, T1=0.5, T2=0.5, l=4):
        """
        实现以 T = T1 + T2 为周期的周期函数f(t)
        参数:
            t: 输入的自变量
            T1, T2: 分段区间参数，周期 T = T1 + T2
            l: 函数中的参数
        返回:
            对应t的周期函数值
        """
        T = T1 + T2
        # 处理数组输入（向量化运算）
        if isinstance(t, np.ndarray):
            t_mod = t % T  # 映射到[0, T)周期内
            res = np.zeros_like(t_mod)
            # 分段赋值：[0, T1] → 0；[T1, T] → sin²(...)
            mask = (t_mod >= T1) & (t_mod < T)
            res[mask] = np.sin( (2 * np.pi * l * (t_mod[mask] - T1)) / T2 ) ** 2
            return res
        # 处理单个数值输入
        else:
            t_mod = t % T
            if 0 <= t_mod <= T1:
                return 0
            elif T1 < t_mod < T:
                return np.sin( (2 * np.pi * l * (t_mod - T1)) / T2 ) ** 2
            else:
                return 0  # 取余后t_mod必然在[0,T)，此分支实际不会触发

    m = 4
    def force_term(X, Y, t):
        f = -m*np.cos(m*Y)*periodic_f(t)
        return f

    def force_term(X, Y, t):
        f = np.zeros_like(X)
        return f

    omega = lambda X, Y: - 1/(nu*m) * np.cos(m*Y) + 0.001* np.cos(m*Y)* np.sin(m*X)
    initial_vorticity = omega(X,Y)

    solver_ref = vs_mrSAV_Vorticity_Stream_Periodic_Solver(nu,gam,s_domain, discrete_num, initial_vorticity, force_term, args.M)
    solver_ref.solve_adaptive_step((0, 2), 1e-5, 1e-2, [0.5, 1.0, 1.5, 2.0])

    print(solver_ref.snapshot_Omega.shape)
    print(solver_ref.snapshot_tn.shape)
