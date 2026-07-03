import numpy as np

OUT_DIR = "results"

DENSITIES = np.linspace(0.2, 2.5, 15)
B_VALUES = [0.0, 0.56, 1.06]

# ---------------------------------------------------------------------
# Remote-action model
# ---------------------------------------------------------------------
REMOTE_E = 0.07
REMOTE_F = 2.0

# ---------------------------------------------------------------------
# Sensitivity / robustness experiments
# ---------------------------------------------------------------------
SIGMA_VALUES = [0.05, 0.1, 0.15]
SYSTEM_SIZES = [17.3, 20.0, 50.0]
F_VALUES = [1.0, 2.0, 3.0]

# ---------------------------------------------------------------------
# Density-wave experiment
# ---------------------------------------------------------------------
DENSITY_WAVE_RHO = [1.16, 1.21]
DENSITY_WAVE_MARKER_INDEX = 11
DENSITY_WAVE_NUM_SNAPSHOTS = 30
DENSITY_WAVE_TIME_SCALE = 10

DENSITY_WAVE_PARAMS = {
    "b": 0.0,
    "e": 0.07,
    "f": 2.0,
    "a": 0.36,
    "relax_steps": 300000,
    "measure_steps": 10000,
    "seed": 10,
}

# ---------------------------------------------------------------------
# Plot styling
# ---------------------------------------------------------------------
PALETTE = {
    "empirical": "#535353",
    "b0": "#E74C3C",
    "b056": "#2ECC71",
    "b106": "#3498DB",
    "remote_b0": "#9B59B6",
    "remote_b056": "#F39C12",
    "weidmann": "#7F8C8D",
    "grid": "#ECEDEE",
}

B_COLORS = {
    0.0: PALETTE["b0"],
    0.56: PALETTE["b056"],
    1.06: PALETTE["b106"],
}

B_MARKERS = {
    0.0: "s",
    0.56: "o",
    1.06: "v",
}

SYSTEM_SIZE_COLORS = {
    17.3: "#E74C3C",
    20.0: "#2ECC71",
    50.0: "#3498DB",
}

SYSTEM_SIZE_MARKERS = {
    17.3: "s",
    20.0: "^",
    50.0: "D",
}

SIGMA_COLORS = {
    0.05: "#46A2FE",
    0.1: "#D35400",
    0.15: "#27AE60",
}

SIGMA_MARKERS = {
    0.05: "o",
    0.1: "s",
    0.15: "^",
}

F_COLORS = {
    1.0: PALETTE["b056"],
    2.0: PALETTE["remote_b0"],
    3.0: "#C0392B",
}

F_MARKERS = {
    1.0: "v",
    2.0: "s",
    3.0: "D",
}

MATPLOTLIB_RC = {
    "font.family": "DejaVu Serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": PALETTE["grid"],
    "grid.linewidth": 0.8,
    "figure.dpi": 150,
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
}
