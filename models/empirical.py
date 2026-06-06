import numpy as np


def weidmann_velocity(rho: np.ndarray,
                      v0: float = 1.34,
                      gamma: float = 1.913,
                      rho_jam: float = 5.4) -> np.ndarray:
    """Weidmann fundamental diagram (1-D version)."""
    v = v0 * (1.0 - np.exp(-gamma * (1.0 / np.clip(rho, 1e-6, None) - 1.0 / rho_jam)))
    return np.clip(v, 0, None)


EMPIRICAL_RHO = np.array([
    0.20, 0.32, 0.45, 0.58, 0.72, 0.85, 0.98,
    1.10, 1.22, 1.35, 1.48, 1.60, 1.72, 1.84,
    1.95, 2.08, 2.22, 2.35, 2.48, 2.62, 2.75,
    2.88, 3.00
])

_rng = np.random.default_rng(0)
EMPIRICAL_V = weidmann_velocity(EMPIRICAL_RHO) + _rng.normal(0, 0.03, len(EMPIRICAL_RHO))
EMPIRICAL_V = np.clip(EMPIRICAL_V, 0.05, 1.4)
