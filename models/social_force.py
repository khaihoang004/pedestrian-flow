import numpy as np
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


def random_positions(N: int, L: float, a: float, rng: np.random.Generator) -> np.ndarray:
    free_space = L - N * a
    if free_space < 0:
        raise ValueError(
            f"Not enough space to place {N} particles with minimum spacing {a} in length {L}."
        )
    
    y = rng.uniform(0, free_space, N)
    y.sort()
    
    x = y + np.arange(N) * a
    
    shift = rng.uniform(0, L)
    x = (x + shift) % L
    
    return np.sort(x)


def simulate_hard_body(params: SimParams) -> Tuple[float, np.ndarray]:
    """
    Hard body without remote action.
    Predict-Correct / Rollback
    """
    rng = np.random.default_rng(params.seed)
    N, L, dt = params.N, params.L, params.dt
    a, b, tau = params.a, params.b, params.tau

    x = random_positions(N, L, a, rng)
    v = np.zeros(N)
    v0 = rng.normal(params.v0_mean, params.v0_std, N).clip(0.05)

    vel_rec = []
    total_steps = params.relax_steps + params.measure_steps

    for step in range(total_steps):
        idx = np.argsort(x)
        xs = x[idx]
        vs = v[idx]
        v0s = v0[idx]

        acc = (v0s - vs) / tau
        v_pred = np.clip(vs + acc * dt, 0, v0s)
        
        x_pred = xs + v_pred * dt

        changed = True
        while changed:
            changed = False
            
            gaps = np.empty(N)
            gaps[:-1] = x_pred[1:] - x_pred[:-1]
            gaps[-1] = L - x_pred[-1] + x_pred[0]

            d_req = a + b * v_pred
            collision = gaps <= d_req
            coll_idx = np.where(collision)[0]

            if len(coll_idx) > 0:
                for i in coll_idx:
                    if x_pred[i] != xs[i] or v_pred[i] > 0:
                        x_pred[i] = xs[i]
                        v_pred[i] = 0.0
                        changed = True

        x[idx] = x_pred % L
        v[idx] = v_pred

        if step >= params.relax_steps:
            vel_rec.append(np.mean(v_pred))

    return float(np.mean(vel_rec)), np.array(vel_rec)


def simulate_remote_action(params: SimParams) -> Tuple[float, np.ndarray]:
    """
    Hard bodies with remote action
    """
    rng = np.random.default_rng(params.seed)
    N, L, dt = params.N, params.L, params.dt
    a, b, tau = params.a, params.b, params.tau
    e, f = params.e, params.f

    x = random_positions(N, L, a, rng)
    v = np.zeros(N)
    v0 = rng.normal(params.v0_mean, params.v0_std, N).clip(0.05)

    vel_rec = []
    total_steps = params.relax_steps + params.measure_steps

    for step in range(total_steps):
        idx = np.argsort(x)
        xs, vs, v0s = x[idx], v[idx], v0[idx]

        gaps = np.empty(N)
        gaps[:-1] = xs[1:] - xs[:-1]
        gaps[-1] = L - xs[-1] + xs[0]
        
        d_req = a + b * vs
        eff = np.maximum(gaps - d_req, 1e-4)
        
        G = (v0s - vs) / tau - e * (1.0 / eff) ** f
        
        F = np.where(vs > 0, G, np.maximum(G, 0.0))

        vs_new = np.maximum(vs + F * dt, 0.0)
        xs_new = (xs + vs_new * dt) % L

        x[idx] = xs_new
        v[idx] = vs_new

        if step >= params.relax_steps:
            vel_rec.append(np.mean(vs_new))

    return float(np.mean(vel_rec)), np.array(vel_rec)


def fundamental_diagram(
    model="hard_body",
    density_values=None,
    base_params=None,
    L=17.3,
) -> Tuple[np.ndarray, np.ndarray]:
    if density_values is None:
        density_values = np.linspace(0.2, 2.5, 15)
    if base_params is None:
        base_params = SimParams()
        
    velocities = []
    for rho in density_values:
        N = max(2, int(round(rho * L)))
        
        p = SimParams(
            L=L, N=N, dt=base_params.dt,
            relax_steps=base_params.relax_steps,
            measure_steps=base_params.measure_steps,
            v0_mean=base_params.v0_mean, v0_std=base_params.v0_std,
            tau=base_params.tau, a=base_params.a, b=base_params.b,
            e=base_params.e, f=base_params.f, seed=base_params.seed,
        )
        
        fn = simulate_hard_body if model == "hard_body" else simulate_remote_action
        v_mean, _ = fn(p)
        velocities.append(v_mean)
        print(f"Mật độ rho = {rho:.2f} (N={N}) -> Vận tốc trung bình v = {v_mean:.3f} m/s")
        
    return density_values, np.array(velocities)

if __name__ == "__main__":
    print("--- ĐANG CHẠY MÔ PHỎNG HARD BODY ---")
    rhos, vels = fundamental_diagram(model="hard_body")