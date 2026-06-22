import numpy as np
from numba import njit
from dataclasses import dataclass
from typing import Tuple

@dataclass
class SimParams:
    L: float = 17.3
    N: int = 10
    dt: float = 0.001
    relax_steps: int = 300_000
    measure_steps: int = 300_000
    v0_mean: float = 1.24
    v0_std: float = 0.05
    tau: float = 0.61
    a: float = 0.36
    b: float = 0.56
    e: float = 0.07
    f: float = 2.0
    seed: int = 42


@njit(cache=True)
def random_positions_jit(N: int, L: float, a: float, seed: int) -> np.ndarray:
    np.random.seed(seed)
    free_space = L - N * a
    if free_space < 0:
        raise ValueError(
            f"Not enough space to place {N} particles with minimum spacing {a} in length {L}."
        )
    
    y = np.empty(N)
    for i in range(N):
        y[i] = np.random.uniform(0, free_space)
    y.sort()
    
    x = np.empty(N)
    for i in range(N):
        x[i] = y[i] + i * a
        
    shift = np.random.uniform(0, L)
    for i in range(N):
        x[i] = (x[i] + shift) % L
        
    x.sort()
    return x

@njit(cache=True)
def simulate_hard_body_jit(
    N: int, L: float, dt: float, relax_steps: int, measure_steps: int, # Fixed spelling here
    v0_mean: float, v0_std: float, tau: float, a: float, b: float, seed: int
) -> Tuple[float, np.ndarray]:
    """
    Hard body without remote action.
    Predict-Correct / Rollback
    """
    np.random.seed(seed)
    
    x = random_positions_jit(N, L, a, seed)
    v = np.zeros(N)
    
    v0 = np.empty(N)
    for i in range(N):
        v_rand = np.random.normal(v0_mean, v0_std)
        v0[i] = max(v_rand, 0.05)
        
    x_pred = np.empty(N)
    v_pred = np.empty(N)
    gaps = np.empty(N)
    
    vel_rec = np.empty(measure_steps)
    vel_idx = 0
    total_steps = relax_steps + measure_steps
    
    for step in range(total_steps):
        idx = np.argsort(x)
        xs = x[idx]
        vs = v[idx]
        v0s = v0[idx]
        
        # 1. Dự đoán bằng vòng lặp explicit (Numba chạy cực nhanh)
        for i in range(N):
            acc = (v0s[i] - vs[i]) / tau
            vp = vs[i] + acc * dt
            if vp < 0.0: vp = 0.0
            if vp > v0s[i]: vp = v0s[i]
            v_pred[i] = vp
            x_pred[i] = xs[i] + vp * dt
            
        # 2. Xử lý va chạm
        changed = True
        while changed:
            changed = False
            
            for i in range(N - 1):
                gaps[i] = x_pred[i+1] - x_pred[i]
            gaps[N-1] = L - x_pred[N-1] + x_pred[0]
            
            for i in range(N):
                d_req = a + b * v_pred[i]
                if gaps[i] <= d_req:
                    if x_pred[i] != xs[i] or v_pred[i] > 0.0:
                        x_pred[i] = xs[i]
                        v_pred[i] = 0.0
                        changed = True
                        
        # 3. Commit state
        for i in range(N):
            x[idx[i]] = x_pred[i] % L
            v[idx[i]] = v_pred[i]
            
        if step >= relax_steps:
            sum_v = 0.0
            for i in range(N):
                sum_v += v_pred[i]
            vel_rec[vel_idx] = sum_v / N
            vel_idx += 1
            
    sum_rec = 0.0
    for i in range(measure_steps):
        sum_rec += vel_rec[i]
        
    return sum_rec / measure_steps, vel_rec

@njit(cache=True)
def simulate_remote_action_jit(
    N: int, L: float, dt: float, relax_steps: int, measure_steps: int,
    v0_mean: float, v0_std: float, tau: float, a: float, b: float, e: float, f: float, seed: int
) -> Tuple[float, np.ndarray]:
    """Mô hình 2: Lực tác dụng từ xa (Explicit Euler)"""
    np.random.seed(seed)
    
    x = random_positions_jit(N, L, a, seed)
    v = np.zeros(N)
    
    v0 = np.empty(N)
    for i in range(N):
        v_rand = np.random.normal(v0_mean, v0_std)
        v0[i] = max(v_rand, 0.05)
        
    xs_new = np.empty(N)
    vs_new = np.empty(N)
    gaps = np.empty(N)
    
    vel_rec = np.empty(measure_steps)
    vel_idx = 0
    total_steps = relax_steps + measure_steps
    
    for step in range(total_steps):
        idx = np.argsort(x)
        xs = x[idx]
        vs = v[idx]
        v0s = v0[idx]
        
        for i in range(N - 1):
            gaps[i] = xs[i+1] - xs[i]
        gaps[N-1] = L - xs[N-1] + xs[0]
        
        for i in range(N):
            d_req = a + b * vs[i]
            eff = gaps[i] - d_req
            if eff < 1e-4: 
                eff = 1e-4
                
            G = (v0s[i] - vs[i]) / tau - e * (1.0 / eff) ** f
            
            if vs[i] > 0.0:
                F = G
            else:
                F = max(G, 0.0)
                
            vn = vs[i] + F * dt
            if vn < 0.0: 
                vn = 0.0
                
            vs_new[i] = vn
            xs_new[i] = (xs[i] + vn * dt) % L
            
        for i in range(N):
            x[idx[i]] = xs_new[i]
            v[idx[i]] = vs_new[i]
            
        if step >= relax_steps:
            sum_v = 0.0
            for i in range(N):
                sum_v += vs_new[i]
            vel_rec[vel_idx] = sum_v / N
            vel_idx += 1
            
    sum_rec = 0.0
    for i in range(measure_steps):
        sum_rec += vel_rec[i]
        
    return sum_rec / measure_steps, vel_rec


@njit(cache=True)
def simulate_remote_action_trajectory(
    N, L, dt,
    relax_steps,
    measure_steps,
    v0_mean, v0_std,
    tau, a, b, e, f, seed
):
    """Mô hình 2: Lực tác dụng từ xa (Explicit Euler)"""
    np.random.seed(seed)
    
    x = random_positions_jit(N, L, a, seed)
    v = np.zeros(N)
    
    v0 = np.empty(N)
    for i in range(N):
        v_rand = np.random.normal(v0_mean, v0_std)
        v0[i] = max(v_rand, 0.05)
        
    xs_new = np.empty(N)
    vs_new = np.empty(N)
    gaps = np.empty(N)
    
    vel_rec = np.empty(measure_steps)
    vel_idx = 0
    total_steps = relax_steps + measure_steps
    
    save_stride = 10

    trajectory = np.empty(
        (measure_steps // save_stride + 1, N)
    )

    for step in range(total_steps):
        idx = np.argsort(x)
        xs = x[idx]
        vs = v[idx]
        v0s = v0[idx]
        
        for i in range(N - 1):
            gaps[i] = xs[i+1] - xs[i]
        gaps[N-1] = L - xs[N-1] + xs[0]
        
        for i in range(N):
            d_req = a + b * vs[i]
            eff = gaps[i] - d_req
            if eff < 1e-4: 
                eff = 1e-4
                
            G = (v0s[i] - vs[i]) / tau - e * (1.0 / eff) ** f
            
            if vs[i] > 0.0:
                F = G
            else:
                F = max(G, 0.0)
                
            vn = vs[i] + F * dt
            if vn < 0.0: 
                vn = 0.0
                
            vs_new[i] = vn
            xs_new[i] = (xs[i] + vn * dt) % L
            
        for i in range(N):
            x[idx[i]] = xs_new[i]
            v[idx[i]] = vs_new[i]
            
        if step >= relax_steps:

            sum_v = 0.0
            for i in range(N):
                sum_v += vs_new[i]

            vel_rec[vel_idx] = sum_v / N

            if vel_idx % save_stride == 0:

                row = vel_idx // save_stride

                idx2 = np.argsort(x)

                for i in range(N):
                    trajectory[row, i] = x[idx2[i]]

            vel_idx += 1
                    
    sum_rec = 0.0
    for i in range(measure_steps):
        sum_rec += vel_rec[i]
        
    return (
        sum_rec / measure_steps,
        vel_rec,
        trajectory
    )


def fundamental_diagram(model="hard_body", density_values=None, base_params=None) -> Tuple[np.ndarray, np.ndarray]:
    """Hàm wrapper để bóc tách SimParams và gọi Numba."""
    if density_values is None:
        density_values = np.linspace(0.2, 2.5, 15)
    if base_params is None:
        base_params = SimParams()
        
    velocities = []
    
    # Ép kiểu Numba biên dịch 1 lần mồi (Warm-up)
    print(f"  [Đang biên dịch JIT cho {model} ...]")
    
    for rho in density_values:
        N = max(2, int(round(rho * base_params.L)))
        p = base_params
        
        if model == "hard_body":
            v_mean, _ = simulate_hard_body_jit(
                N, p.L, p.dt, p.relax_steps, p.measure_steps,
                p.v0_mean, p.v0_std, p.tau, p.a, p.b, p.seed
            )
        else:
            v_mean, _ = simulate_remote_action_jit(
                N, p.L, p.dt, p.relax_steps, p.measure_steps,
                p.v0_mean, p.v0_std, p.tau, p.a, p.b, p.e, p.f, p.seed
            )
            
        velocities.append(v_mean)
        print(f"  [rho ρ={rho:.2f}] v_mean = {v_mean:.3f} m/s")
        
    return density_values, np.array(velocities)

if __name__ == "__main__":
    print("--- ĐANG CHẠY MÔ PHỎNG HARD BODY ---")
    rhos, vels = fundamental_diagram(model="hard_body")