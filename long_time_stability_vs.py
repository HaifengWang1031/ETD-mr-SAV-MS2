import h5py, gc
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import numpy as np
from vs_ns_periodic_mrSAV_solver import vs_mrSAV_Vorticity_Stream_Periodic_Solver as vs_ns_solver

np.random.seed(0)

import argparse
parse = argparse.ArgumentParser()
parse.add_argument("--Re", type=float, default=20)
parse.add_argument("--m", type=float, default=2)
parse.add_argument("--ga", type=float, default=1000)
parse.add_argument("--eps", type=float, default=0.1)
parse.add_argument("--M", type=str, default="ETD_mrGSAV_MS2_b")
arg = parse.parse_args()

Re = arg.Re
nu = 1/arg.Re
ga = arg.ga
m  = arg.m
eps = arg.eps
M  = arg.M


def process_task(args):
    """多进程任务函数（必须定义在顶层作用域）"""
    force_term = lambda X,Y,t: -m*np.cos(m*Y)
    group_name, nu, ga, s_domain, discrete_num, initial_vorticity, t_period, M = args

    solver = vs_ns_solver(nu, ga, s_domain, discrete_num, initial_vorticity, force_term, M)
    solver.solve_adaptive_step(t_period, 1e-5, 1e-2, snapshot = np.linspace(t_period[0], t_period[1], int(t_period[1] - t_period[0]) + 1), rtol=1e-4, rtol_q=1e-4)

    # 返回计算结果（不直接写入文件，避免进程间文件操作冲突）
    return {
        'group_name': group_name,
        'Omega': solver.Omega,
        'q': solver.q,
        'tn': solver.tn,
        'tn_s': solver.tn_s,
        'Energy': solver.Energy,
        'Enstrophy': solver.Enstrophy,
        "Palinstrophy": solver.Palinstrophy,
        "cpu_time": solver.cpu_time
    }

if __name__ == '__main__':
    s_domain = (0,0, 2*np.pi, 2*np.pi)
    discrete_num = [64, 64]
    xn = np.linspace(s_domain[0],s_domain[2],discrete_num[0]+1)
    yn = np.linspace(s_domain[1],s_domain[3],discrete_num[1]+1)
    X,Y = np.meshgrid(xn,yn)

    def initial_streamfunction(x: np.ndarray, y: np.ndarray, nu: float, m: float, eps: float) -> np.ndarray:
        """
        计算初始流函数 φ(0)
        参数:
            x, y: 网格坐标数组 (可以是任意维度，会自动广播)
            nu: 运动粘度 ν
            m: 基波波数 m
            eps: 扰动强度 ε
        返回:
            phi: 流函数场，形状与 x, y 一致
        """
        # 1. 基流项: -1/(ν m³) * cos(m y)
        base_flow = - (1.0 / (nu * m**3)) * np.cos(m * y)
        
        # 2. 生成所有满足 |k| ≤ 10 的二维整数波矢 (k1, k2)
        k_max = 10
        k1_vals = np.arange(-k_max, k_max + 1)
        k2_vals = np.arange(-k_max, k_max + 1)
        k1_grid, k2_grid = np.meshgrid(k1_vals, k2_vals, indexing="ij")
        
        # 计算波矢模长 |k| = sqrt(k1² + k2²)
        k_mod = np.sqrt(k1_grid**2 + k2_grid**2)
        
        # 过滤 |k| ≤ 10 的波矢（排除模长>10的点）
        mask = k_mod <= 10
        k1_valid = k1_grid[mask]
        k2_valid = k2_grid[mask]
        k_mod_valid = k_mod[mask]
        
        # 3. 计算扰动项求和
        perturbation = np.zeros_like(x, dtype=np.float64)
        for k1, k2, k_abs in zip(k1_valid, k2_valid, k_mod_valid):
            if k_abs < 1e-10: # 避免 |k|=0 时分母为0
                continue
            # 1/|k|^(5/3) * cos(k1 x) * cos(k2 y)
            term = (1.0 / (k_abs ** (3))) *( 1 * np.cos(k1 * x) * np.cos(k2 * y) 
                                           + 1 * np.sin(k1 * x) * np.cos(k2 * y) 
                                           + 1 * np.cos(k1 * x) * np.sin(k2 * y) 
                                           + 1 * np.sin(k1 * x) * np.sin(k2 * y))
            perturbation += term
        
        # 4. 总流函数 = 基流 + ε*扰动
        phi = base_flow + eps * perturbation
        return phi


    ini_sf = initial_streamfunction(X, Y, nu, m, eps)
    etdms_solver = vs_ns_solver(nu, ga, s_domain, discrete_num, ini_sf, lambda x: x**2, "ETD_mrGSAV_MS2_b")
    # initial_vorticity = np.pad(etdms_solver.velocity2vorticity(*etdms_solver.stream2velocity(ini_sf[:-1, :-1])), ((0, 1), (0, 1)))
    initial_vorticity = np.pad((etdms_solver.stream2velocity(ini_sf[:-1, :-1]))[0], ((0, 1), (0, 1)))

    t_period = [0, 1000]

    # 准备任务参数
    tasks = [ ("adaptive", nu, ga, s_domain, discrete_num, initial_vorticity, t_period, M) ]

    # 使用进程池执行任务
    with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
        results = list(executor.map(process_task, tasks))

    # 主进程统一写入结果（避免多进程文件冲突）
    with h5py.File(f"./data/vs_ns_mrsav_{M}_longTimeStability_{m}_{Re}_{eps}.h5","w") as f:
        for result in results:
            group = f.create_group(result['group_name'])
            group["Omega"] = result['Omega']
            group["q"]     = result['q']
            group["tn"]    = result['tn']
            group["Energy"]    = result['Energy']
            group["Enstrophy"] = result['Enstrophy']
            group["CPU_time"] = result['cpu_time']
            group["Palinstrophy"] = result['Palinstrophy']

    print("All tasks completed with multiprocessing.")
