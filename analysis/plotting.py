import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from models.social_force import (
    SimParams,
    fundamental_diagram,
    simulate_remote_action_trajectory,
)
from analysis.helpers import (
    save_figure,
    setup_fd_axes,
    plot_empirical,
    plot_result_curves,
)
from analysis.config import *

os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams.update(MATPLOTLIB_RC)


# Figure 1: Sensitivity to b
def plot_sensitivity_b(results_b: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    plot_empirical(ax)
    plot_result_curves(
        ax,
        results_b,
        colors=B_COLORS,
        markers=B_MARKERS,
        label_fmt=lambda b: f"Model b = {b} s",
    )

    setup_fd_axes(ax, "Velocity-density relation (Hard body without remote action)")
    ax.legend(fontsize=9)
    save_figure(fig, "sensitivity_b.png")


# Figure 2: Remote-action comparison
def plot_remote_comparison(res_hard_b056, res_remote_b0, res_remote_b056) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        *res_hard_b056,
        "-o",
        color=PALETTE["b056"],
        ms=5,
        lw=1.8,
        label="without remote action, b=0.56 s",
    )
    ax.plot(
        *res_remote_b0,
        "-s",
        color=PALETTE["remote_b0"],
        ms=5,
        lw=1.8,
        label="with remote action, b=0 s",
    )
    ax.plot(
        *res_remote_b056,
        "-^",
        color=PALETTE["remote_b056"],
        ms=5,
        lw=1.8,
        label="with remote action, b=0.56 s",
    )

    setup_fd_axes(ax, "Velocity-density relation (Hard bodies with remote action)")
    ax.legend(fontsize=9)
    save_figure(fig, "remote_comparison.png")

# Figure 3: Density waves
def plot_density_wave_subplot(ax, trajectory, dt: float, L: float, rho_value: float, marker_index: int = 0) -> None:
    num_steps, num_agents = trajectory.shape
    step_interval = max(1, num_steps // DENSITY_WAVE_NUM_SNAPSHOTS)

    traj_subset = trajectory[::step_interval]
    time_subset = np.arange(0, num_steps, step_interval) * dt * DENSITY_WAVE_TIME_SCALE

    for positions, t_val in zip(traj_subset, time_subset):
        x_vals = np.mod(positions, L)

        ax.scatter(
            x_vals,
            np.full(num_agents, t_val),
            facecolors="none",
            edgecolors="gray",
            s=18,
            linewidths=0.8,
        )

        if 0 <= marker_index < num_agents:
            ax.scatter(x_vals[marker_index], t_val, color="black", s=18)

    ax.set_xlim(0, L)
    ax.invert_yaxis()
    ax.grid(False)
    ax.set_title(f"ρ = {rho_value} [1/m]")
    ax.set_xlabel("L [m]")
    ax.set_yticks([])


def run_density_wave_experiment() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    for idx, rho_value in enumerate(DENSITY_WAVE_RHO):
        params = SimParams(**DENSITY_WAVE_PARAMS)

        num_agents = int(round(rho_value * params.L))

        _, _, trajectory = simulate_remote_action_trajectory(
            num_agents,
            params.L,
            params.dt,
            params.relax_steps,
            params.measure_steps,
            params.v0_mean,
            params.v0_std,
            params.tau,
            params.a,
            params.b,
            params.e,
            params.f,
            params.seed,
        )

        marker_index = min(DENSITY_WAVE_MARKER_INDEX, num_agents - 1)
        plot_density_wave_subplot(
            axes[idx],
            trajectory=trajectory,
            dt=params.dt,
            L=params.L,
            rho_value=rho_value,
            marker_index=marker_index,
        )

        if idx == 0:
            axes[idx].set_ylabel("← t")

    plt.tight_layout()
    save_figure(fig, "density_waves_combined.png")


# Figure 4: Robustness: system size
def plot_robustness_system_size(results_L: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    colors=SYSTEM_SIZE_COLORS
    markers=SYSTEM_SIZE_MARKERS

    plot_empirical(ax)
    plot_result_curves(
        ax,
        results_L,
        colors=colors,
        markers=markers,
        label_fmt=lambda L: f"L = {L} m",
    )

    setup_fd_axes(ax, "Robustness — System Size (Finite-Size Effects)")
    ax.legend(fontsize=9)
    save_figure(fig, "robustness_system_size.png")

# Figure 5: Robustness: Sensitivity analysis: sigma, f
def plot_sensitivity_sigma(results_sigma: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    colors=SIGMA_COLORS
    markers=SIGMA_MARKERS

    plot_empirical(ax)
    plot_result_curves(
        ax,
        results_sigma,
        colors=colors,
        markers=markers,
        label_fmt=lambda sigma: f"σ = {sigma} m/s",
    )

    setup_fd_axes(ax, "Sensitivity Analysis: Velocity Standard Deviation (σ)")
    ax.legend(fontsize=9)
    save_figure(fig, "sensitivity_sigma.png")


def plot_sensitivity_f(results_f: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    colors=F_COLORS
    markers=F_MARKERS

    plot_empirical(ax)
    plot_result_curves(
        ax,
        results_f,
        colors=colors,
        markers=markers,
        label_fmt=lambda f_val: f"f = {f_val}",
    )

    setup_fd_axes(ax, "Sensitivity Analysis: Remote Force Range (f)")
    ax.legend(fontsize=9)
    save_figure(fig, "sensitivity_f.png")


# Runners
def run_hard_body_experiment(densities) -> dict:
    results = {}
    for b in B_VALUES:
        print(f"Running hard-body model with b = {b} s")
        results[b] = fundamental_diagram(
            model="hard_body",
            density_values=densities,
            base_params=SimParams(b=b),
        )
    return results


def run_remote_action_experiment(densities) -> dict:
    results = {}
    for b in B_VALUES:
        print(f"Running remote-action model with b = {b} s")
        results[b] = fundamental_diagram(
            model="remote_action",
            density_values=densities,
            base_params=SimParams(b=b, e=REMOTE_E, f=REMOTE_F),
        )
    return results


def run_sigma_sensitivity(densities) -> dict:
    results = {}
    for sigma in SIGMA_VALUES:
        print(f"Running sigma sensitivity with σ = {sigma} m/s")
        results[sigma] = fundamental_diagram(
            model="hard_body",
            density_values=densities,
            base_params=SimParams(b=0.56, v0_std=sigma),
        )
    return results


def run_system_size_robustness(densities) -> dict:
    results = {}
    for L in SYSTEM_SIZES:
        print(f"Running system-size robustness with L = {L} m")
        results[L] = fundamental_diagram(
            model="hard_body",
            density_values=densities,
            base_params=SimParams(b=0.56, L=L),
        )
    return results


# Main
def main() -> None:
    results_hard = run_hard_body_experiment(DENSITIES)
    plot_sensitivity_b(results_hard)

    results_remote = run_remote_action_experiment(DENSITIES)
    plot_remote_comparison(
        results_hard[0.56],
        results_remote[0.0],
        results_remote[0.56],
    )

    print("Running density-wave experiment...")
    run_density_wave_experiment()

    results_sigma = run_sigma_sensitivity(DENSITIES)
    plot_sensitivity_sigma(results_sigma)

    results_L = run_system_size_robustness(DENSITIES)
    plot_robustness_system_size(results_L)

    print(f"Done. All figures were saved to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
