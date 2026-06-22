import os
import numpy as np
from dataclasses import dataclass
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from models.empirical import weidmann_velocity, EMPIRICAL_RHO, EMPIRICAL_V
from models.social_force import SimParams, fundamental_diagram, simulate_remote_action_trajectory

PALETTE = {
    "empirical":  "#535353",
    "b0":         "#E74C3C",
    "b056":       "#2ECC71",
    "b106":       "#3498DB",
    "remote_b0":  "#9B59B6",
    "remote_b056":"#F39C12",
    "weidmann":   "#7F8C8D",
    "grid":       "#ECEDEE",
}

plt.rcParams.update({
    "font.family":       "DejaVu Serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.color":        PALETTE["grid"],
    "grid.linewidth":    0.8,
    "figure.dpi":        150,
    "savefig.dpi":       180,
    "savefig.bbox":      "tight",
})

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# PLOTTING
def _emp_scatter(ax, zorder=2):
    ax.scatter(EMPIRICAL_RHO, EMPIRICAL_V, s=15, color=PALETTE["empirical"], 
               zorder=zorder, label="Empirical (Seyfried 2005)", marker="D", linewidths=0)

def _weid_line(ax):
    rho = np.linspace(0.1, 3.2, 200)
    ax.plot(rho, weidmann_velocity(rho), "--", color=PALETTE["weidmann"], 
            linewidth=1.4, label="Weidmann (1993)", zorder=5)

def plot_fig1_sensitivity_b(results_b: dict):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {0.0: PALETTE["b0"], 0.56: PALETTE["b056"], 1.06: PALETTE["b106"]}
    markers = {0.0: "s", 0.56: "o", 1.06: "v"}

    _emp_scatter(ax)
    for b, (rho, v) in sorted(results_b.items()):
        ax.plot(rho, v, "-", color=colors.get(b, "#000"), marker=markers.get(b, "x"), ms=5, lw=1.8, label=f"Model b = {b} s")

    ax.set(xlim=(0, 3.2), ylim=(0, 1.5), xlabel="Density ρ [1/m]", ylabel="Mean velocity v̄ [m/s]",
           title="Velocity-density relation (Hard body without remote action)")
    ax.legend(fontsize=9)
    fig.savefig(f"{OUT_DIR}/fig1_sensitivity_b.png")
    plt.close(fig)

def plot_fig2_remote_comparison(res_hard_b056, res_remote_b0, res_remote_b056):
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(*res_hard_b056, "-o", color=PALETTE["b056"], ms=5, lw=1.8, label="without remote action, b=0.56 s")
    ax.plot(*res_remote_b0, "-s", color=PALETTE["remote_b0"], ms=5, lw=1.8, label="with remote action, b=0 s")
    ax.plot(*res_remote_b056, "-^", color=PALETTE["remote_b056"], ms=5, lw=1.8, label="with remote action, b=0.56 s")

    ax.set(xlim=(0, 3.2), ylim=(0, 1.5), xlabel="Density ρ [1/m]", ylabel="Mean velocity v̄ [m/s]",
           title="Velocity-density relation (Hard bodies with remote action)")
    ax.legend(fontsize=9)
    fig.savefig(f"{OUT_DIR}/fig2_remote_comparison.png")
    plt.close(fig)

def plot_density_waves(trajectory, dt, L):
    fig, ax = plt.subplots(figsize=(7, 6))
    T, N = trajectory.shape
    
    num_snapshots = 45
    step_interval = max(1, T // num_snapshots)
    
    traj_subset = trajectory[::step_interval, :]
    time_subset = np.arange(0, T, step_interval) * dt

    for i in range(len(time_subset)):
        t_val = time_subset[i]
        x_vals = np.mod(traj_subset[i, :], wave_params.L)
        
        # Vẽ những người đi bộ bình thường (chấm rỗng viền xám)
        ax.scatter(x_vals[1:], np.full(N-1, t_val), 
                   facecolors='none', edgecolors='gray', s=18, linewidths=0.8)
        
        # TÔ ĐEN người đi bộ số 0 để làm cột mốc theo dõi giống hệt bài báo
        ax.scatter(x_vals[0], t_val, color='black', s=18)

    ax.set_xlim(0, L)
    
    ax.invert_yaxis()
    
    ax.set_xlabel("L [m]")
    ax.set_ylabel("← t")
    ax.set_title("Time-development of the positions")
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.savefig(f"{OUT_DIR}/fig3_density_waves.png")
    plt.close(fig)

def plot_robustness_system_size(results_L: dict):
    fig, ax = plt.subplots(figsize=(7, 5))
    _emp_scatter(ax)

    colors = {17.3: "#E74C3C", 20.0: "#2ECC71", 50.0: "#3498DB"}
    markers = {17.3: "s", 20.0: "^", 50.0: "D"}
    for L, (rho, v) in sorted(results_L.items()):
        ax.plot(rho, v, "-", color=colors.get(L, "#000"), marker=markers.get(L, "x"), ms=5, lw=1.8, label=f"L = {L} m")

    ax.set(xlim=(0, 3.2), ylim=(0, 1.5), xlabel="Density ρ [1/m]", ylabel="Mean velocity v̄ [m/s]",
           title="Robustness — System Size (Finite-Size Effects)")
    ax.legend(fontsize=9)
    fig.savefig(f"{OUT_DIR}/fig7_robustness_system_size.png")
    plt.close(fig)

def plot_summary_dashboard(res_hard, res_remote, rob_L):
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    axes = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

    # (a) Sensitivity b
    _emp_scatter(axes[0])
    for i, b in enumerate(sorted(res_hard.keys())):
        axes[0].plot(*res_hard[b], "-o", ms=3, lw=1.4, label=f"b={b}")
    axes[0].set_title("(a) Sensitivity: b", fontsize=10)
    axes[0].legend(fontsize=7)

    # (b) Remote comparison
    _emp_scatter(axes[1])
    axes[1].plot(*res_remote[0.0], "-s", color=PALETTE["remote_b0"], ms=3, lw=1.4, label="Remote b=0")
    axes[1].plot(*res_remote[0.56], "-^", color=PALETTE["remote_b056"], ms=3, lw=1.4, label="Remote b=0.56")
    axes[1].set_title("(b) Remote force effect", fontsize=10)
    axes[1].legend(fontsize=7)

    # (c) Robustness: Seeds
    _emp_scatter(axes[2])
    axes[2].set_title("(c) Robustness: seeds (Skipped in runtime)", fontsize=10)

    # (d) Robustness System Size
    _emp_scatter(axes[3])
    if rob_L:
        for L, (rho, v) in sorted(rob_L.items()):
            axes[3].plot(rho, v, "-", ms=3, lw=1.4, label=f"L={L}m")
        axes[3].legend(fontsize=7)
    axes[3].set_title("(d) Robustness: system size", fontsize=10)

    # (e) ACTUAL RMSE Heatmap
    b_vals = sorted(res_hard.keys())
    rmse_mat = np.zeros((2, len(b_vals)))
    for j, b in enumerate(b_vals):
        rmse_mat[0, j] = np.sqrt(np.mean((res_hard[b][1] - weidmann_velocity(res_hard[b][0]))**2))
        rmse_mat[1, j] = np.sqrt(np.mean((res_remote[b][1] - weidmann_velocity(res_remote[b][0]))**2))

    im = axes[4].imshow(rmse_mat, cmap="RdYlGn_r", aspect="auto")
    axes[4].set_xticks(range(len(b_vals)))
    axes[4].set_xticklabels([str(b) for b in b_vals])
    axes[4].set_yticks([0, 1])
    axes[4].set_yticklabels(["Hard body", "Remote"])
    axes[4].set_title("(e) Actual RMSE Heatmap (b × model)", fontsize=10)
    fig.colorbar(im, ax=axes[4], shrink=0.8, label="RMSE [m/s]")

    # (f) Limitation: b mismatch
    _emp_scatter(axes[5])
    for b in [0.56, 1.06]:
        if b in res_hard:
            rmse_val = np.sqrt(np.mean((res_hard[b][1] - weidmann_velocity(res_hard[b][0]))**2))
            axes[5].plot(*res_hard[b], "-o", ms=3, lw=1.6, label=f"b={b}  RMSE={rmse_val:.3f}")
    axes[5].set_title("(f) Limitation: b mismatch", fontsize=10)
    axes[5].legend(fontsize=7)

    for ax in [axes[0], axes[1], axes[2], axes[3], axes[5]]:
        ax.set(xlim=(0, 3.2), ylim=(0, 1.5), xlabel="ρ [1/m]", ylabel="v̄ [m/s]")

    fig.suptitle("Pedestrian Flow Model — Summary Dashboard", fontsize=14, fontweight="bold", y=1.01)
    fig.savefig(f"{OUT_DIR}/fig_dashboard.png")
    plt.close(fig)

def plot_sensitivity_sigma(results_sigma: dict):
    fig, ax = plt.subplots(figsize=(7, 5))
    _emp_scatter(ax)
    
    colors = {0.05: "#46A2FE", 0.1: "#D35400", 0.15: "#27AE60"}
    markers = {0.05: "o", 0.1: "s", 0.15: "^"}
    
    for sigma, (rho, v) in sorted(results_sigma.items()):
        ax.plot(rho, v, "-", color=colors.get(sigma, "#000"), 
                marker=markers.get(sigma, "x"), ms=5, lw=1.8, label=f"σ = {sigma} m/s")
        
    ax.set(xlim=(0, 3.2), ylim=(0, 1.5), xlabel="Density ρ [1/m]", ylabel="Mean velocity v̄ [m/s]",
           title="Sensitivity Analysis: Velocity Standard Deviation (σ)")
    ax.legend(fontsize=9)
    fig.savefig(f"{OUT_DIR}/fig_sensitivity_sigma.png")
    plt.close(fig)

def plot_sensitivity_f(results_f: dict):
    fig, ax = plt.subplots(figsize=(7, 5))
    _emp_scatter(ax)
    
    colors = {1.0: PALETTE["b056"], 2.0: PALETTE["remote_b0"], 3.0: "#C0392B"}
    markers = {1.0: "v", 2.0: "s", 3.0: "D"}
    
    for f_val, (rho, v) in sorted(results_f.items()):
        ax.plot(rho, v, "-", color=colors.get(f_val, "#000"), 
                marker=markers.get(f_val, "x"), ms=5, lw=1.8, label=f"f = {f_val}")
        
    ax.set(xlim=(0, 3.2), ylim=(0, 1.5), xlabel="Density ρ [1/m]", ylabel="Mean velocity v̄ [m/s]",
           title="Sensitivity Analysis: Remote Force Range (f)")
    ax.legend(fontsize=9)
    fig.savefig(f"{OUT_DIR}/fig_sensitivity_f.png")
    plt.close(fig)

# =============================================================================
# 4. CHU TRÌNH THỰC THI (EXECUTION PIPELINE)
# =============================================================================
if __name__ == "__main__":
    print("🚀 Bắt đầu mô phỏng tái hiện Seyfried et al. (2006)...")
    
    densities = np.linspace(0.2, 2.5, 15)

    # 1. Chạy Hard Body (Fig 1)
    results_hard = {}
    for b_val in [0.0, 0.56, 1.06]:
        print(f"\n--- Chạy Hard Body, b = {b_val} ---")
        results_hard[b_val] = fundamental_diagram(model="hard_body", density_values=densities, base_params=SimParams(b=b_val))
    plot_fig1_sensitivity_b(results_hard)

    # 2. Chạy Remote Action (Fig 2)
    results_remote = {}
    for b_val in [0.0, 0.56, 1.06]: 
        print(f"\n--- Chạy Remote Action, b = {b_val} ---")
        results_remote[b_val] = fundamental_diagram(model="remote_action", density_values=densities, base_params=SimParams(b=b_val, e=0.07, f=2.0))
    plot_fig2_remote_comparison(results_hard[0.56], results_remote[0.0], results_remote[0.56])

    # 3. Density Wave (Mô phỏng lại Fig 3 với 2 mức rho)
    print("\n--- Density wave experiment (Fig 3) ---")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    rho_waves = [1.16, 1.21]
    
    for idx, rho_wave in enumerate(rho_waves):
        wave_params = SimParams(
            b=0.0, e=0.07, f=2.0,
            a=0.36,
            relax_steps=300000, measure_steps=10000,
            seed=10
        )
        N_wave = int(round(rho_wave * wave_params.L))

        _, _, trajectory = simulate_remote_action_trajectory(
            N_wave, wave_params.L, wave_params.dt, wave_params.relax_steps, 
            wave_params.measure_steps, wave_params.v0_mean, wave_params.v0_std, 
            wave_params.tau, wave_params.a, wave_params.b, wave_params.e, wave_params.f, wave_params.seed
        )
        
        # --- Bắt đầu vẽ trực tiếp lên subplot ---
        ax = axes[idx]
        T, N = trajectory.shape
        num_snapshots = 30
        step_interval = max(1, T // num_snapshots)
        
        traj_subset = trajectory[::step_interval, :]
        time_subset = np.arange(0, T, step_interval) * wave_params.dt * 10

        for i in range(len(time_subset)):
            t_val = time_subset[i]
            x_vals = np.mod(traj_subset[i, :], wave_params.L)
            
            # Vẽ các hạt rỗng
            ax.scatter(x_vals, np.full(N, t_val), 
                       facecolors='none', edgecolors='gray', s=18, linewidths=0.8)
            # Vẽ 1 hạt đen làm mốc
            ax.scatter(x_vals[11], t_val, color='black', s=18)

        ax.set_xlim(0, wave_params.L)
        ax.invert_yaxis()
        ax.grid(False)
        ax.set_title(f"ρ = {rho_wave} [1/m]")
        ax.set_xlabel("L [m]")
        
        ax.set_yticks([])
        if idx == 0:
            ax.set_ylabel("<-  t")
            
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig3_density_waves_combined.png")
    plt.close(fig)

    # 4. Sensitivity Analysis: Thay đổi độ lệch chuẩn vận tốc (Sigma)
    print("\n--- Chạy Sensitivity Analysis: Sigma (σ) ---")
    results_sigma = {}
    for sigma_val in [0.05, 0.1, 0.15]:
        print(f"  > Đang chạy với σ = {sigma_val} m/s...")
        # Sử dụng model hard_body, b=0.56 để kiểm tra giống kịch bản tốt nhất của bài báo
        results_sigma[sigma_val] = fundamental_diagram(
            model="hard_body", 
            density_values=densities, 
            base_params=SimParams(b=0.56, v0_std=sigma_val)
        )
    plot_sensitivity_sigma(results_sigma)

    # 5. Sensitivity Analysis: Thay đổi phạm vi tác dụng lực từ xa (Tham số f)
    print("\n--- Chạy Sensitivity Analysis: Range f ---")
    results_f = {}
    for f_val in [1.0, 2.0, 3.0]:
        print(f"  > Đang chạy với f = {f_val}...")
        # Buộc phải dùng b=0.0 để "khoảng trống vận tốc" (gap) hiện rõ trên biểu đồ
        results_f[f_val] = fundamental_diagram(
            model="remote_action", 
            density_values=densities, 
            base_params=SimParams(b=0.0, e=0.07, f=f_val)
        )
    plot_sensitivity_f(results_f)
    
    # 6. Robustness Analysis: System Size (L)
    print("\n--- Chạy Robustness Analysis: System Size (L) ---")
    rob_L_res = {}
    
    for L_val in [17.3, 20.0, 50.0]:
        print(f"  > Đang chạy với L = {L_val} m...")
        rob_L_res[L_val] = fundamental_diagram(
            model="hard_body", 
            density_values=densities, 
            base_params=SimParams(b=0.56, L=L_val)
        )
        
    # Gọi hàm vẽ đồ thị robustness
    plot_robustness_system_size(rob_L_res)
    
    print(f"\n✅ Hoàn tất! Tất cả biểu đồ đã được lưu trong thư mục ./{OUT_DIR}/")