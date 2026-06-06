import numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.social_force import SimParams, fundamental_diagram, simulate_hard_body, simulate_remote_action
from models.empirical import weidmann_velocity

# Coarse grid – fast
DENSITY_GRID = np.array([0.3, 0.5, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 2.9])
BASE = SimParams(relax_steps=8_000, measure_steps=8_000)

def rmse(v, rho): return float(np.sqrt(np.mean((v - weidmann_velocity(rho))**2)))

def _fd(model, b=None, tau=None, a=None, e=None, f=None, L=17.3, seed=42, v0_std=0.05):
    kw = dict(relax_steps=BASE.relax_steps, measure_steps=BASE.measure_steps, seed=seed, v0_std=v0_std)
    if b is not None: kw['b'] = b
    if tau is not None: kw['tau'] = tau
    if a is not None: kw['a'] = a
    if e is not None: kw['e'] = e
    if f is not None: kw['f'] = f
    kw['L'] = L
    p = SimParams(**kw)
    return fundamental_diagram(model=model, density_values=DENSITY_GRID, base_params=p, L=L)

def sensitivity_b(model="hard_body"):
    out = {}
    for b in [0.0, 0.25, 0.56, 1.06]:
        rho, v = _fd(model, b=b)
        out[b] = (rho, v); print(f"  b={b}  RMSE={rmse(v,rho):.4f}")
    return out

def sensitivity_tau(model="hard_body"):
    out = {}
    for tau in [0.2, 0.4, 0.61, 1.0, 1.5]:
        rho, v = _fd(model, tau=tau)
        out[tau] = (rho, v); print(f"  tau={tau}  RMSE={rmse(v,rho):.4f}")
    return out

def sensitivity_a(model="hard_body"):
    out = {}
    for a in [0.20, 0.28, 0.36, 0.44, 0.52]:
        rho, v = _fd(model, a=a)
        out[a] = (rho, v); print(f"  a={a}  RMSE={rmse(v,rho):.4f}")
    return out

def sensitivity_remote_force(e_values=None, f_values=None):
    if e_values is None: e_values = [0.02, 0.05, 0.07, 0.12, 0.20]
    if f_values is None: f_values = [1.0, 2.0, 3.0]
    re, rf = {}, {}
    for e in e_values:
        rho,v = _fd("remote", b=0.56, e=e); re[e]=(rho,v); print(f"  e={e}  RMSE={rmse(v,rho):.4f}")
    for f in f_values:
        rho,v = _fd("remote", b=0.56, f=f); rf[f]=(rho,v); print(f"  f={f}  RMSE={rmse(v,rho):.4f}")
    return re, rf

def robustness_seeds(model="hard_body", n_seeds=4):
    seeds=[42,123,777,2024][:n_seeds]; all_v=[]
    for seed in seeds:
        rho,v = _fd(model, seed=seed); all_v.append(v); print(f"  seed={seed} mean={np.mean(v):.3f}")
    all_v=np.array(all_v)
    return {"densities":DENSITY_GRID,"mean":np.mean(all_v,0),"std":np.std(all_v,0),"all":all_v,"seeds":seeds}

def robustness_system_size(model="hard_body"):
    out={}
    for L in [17.3, 25.0, 50.0]:
        rho,v=_fd(model, L=L); out[L]=(rho,v); print(f"  L={L}  RMSE={rmse(v,rho):.4f}")
    return out

def robustness_v0_std(model="hard_body"):
    out={}
    for s in [0.02, 0.05, 0.10, 0.20]:
        rho,v=_fd(model, v0_std=s); out[s]=(rho,v); print(f"  σ={s}  RMSE={rmse(v,rho):.4f}")
    return out

def limitation_density_waves():
    snaps={}
    for rho in [1.16, 1.21]:
        N=max(2,int(round(rho*17.3)))
        p=SimParams(N=N,b=0.0,e=0.07,f=2.0,relax_steps=5000,measure_steps=15000)
        _,ts=simulate_remote_action(p); snaps[rho]=ts
        print(f"  ρ={rho}  std={np.std(ts):.4f}")
    return snaps

def limitation_model_vs_empirical_b():
    out={}
    for b in [0.56, 1.06]:
        rho,v=_fd("hard_body",b=b); r=rmse(v,rho); out[b]=(rho,v,r)
        print(f"  b={b}  RMSE={r:.4f}")
    return out
