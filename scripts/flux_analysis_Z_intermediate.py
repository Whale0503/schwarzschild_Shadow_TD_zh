import os
import numpy as np
import matplotlib.pyplot as plt
import json
import re
from matplotlib.ticker import LogFormatterMathtext
def extract_chi(filename):
    match = re.search(r'intermediate_([0-9.]+)', filename)
    return float(match.group(1)) if match else float("nan")

def plot_flux_profiles(input_paths, output_dir, tolerance, r_in, theta0, kappaff, kappaK):
    color_cycle = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    font_size = 16
    for idx, input_path in enumerate(input_paths):
        data = np.load(input_path)
        b = data["b"]
        alpha = data["alpha"]
        F = data["F"]

        mask_alpha_0 = np.abs(alpha - (np.pi / 2)) < tolerance
        mask_alpha_pi = np.abs(alpha - (3* np.pi / 2)) < tolerance

        b_0 = b[mask_alpha_0]
        F_0 = F[mask_alpha_0] * 1e5

        b_pi = -b[mask_alpha_pi]  # 左侧镜像
        F_pi = F[mask_alpha_pi] * 1e5
        #合并后统一归一化
       # F_all = np.concatenate([F_0, F_pi])
       # max_F = F_all.max() if len(F_all) > 0 else 0
        #if max_F > 0:
        #    F_0 = F_0 / max_F
        #    F_pi = F_pi / max_F

        chi = extract_chi(os.path.basename(input_path))
        color = color_cycle[idx % len(color_cycle)]

        # 排序以避免折线跳动
        sorted_idx_0 = np.argsort(b_0)
        sorted_idx_pi = np.argsort(b_pi)
        # 第一条线（α ≈ 0）用于图例
        ax.plot(b_0[sorted_idx_0], F_0[sorted_idx_0], label=fr"$\chi = {chi:.2f}/M$", color=color, linestyle='-')

        # 第二条线（α ≈ π），同色但不加图例
        ax.plot(b_pi[sorted_idx_pi], F_pi[sorted_idx_pi], color=color, linestyle='-')

        print(f"[{os.path.basename(input_path)}] α ≈ π/2: {np.sum(mask_alpha_0)}, α ≈ 3π/2: {np.sum(mask_alpha_pi)}")
        

    ax.set_xlabel(r"$z'/M$",fontsize=font_size)
    ax.set_ylabel(r"Flux × $10^{-5}$",fontsize=font_size)
    #plt.title("Flux Profile for Z axis")
    ax.set_xlim(-15, 15)
    ax.set_ylim(-0.05,5) 
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    ax.legend()
   # plt.grid(True) 
    ax.ticklabel_format(style='plain', axis='y')  # y轴用普通数字
    ax.legend(loc='upper right', fontsize=font_size-2)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"combined_flux_Z_axis_optical_{opt_regime}_rin={r_in:.1f}_theta0={theta0:.1f}_kappaff={kappaff:.3f}_kappaK={kappaK:.3f}.png")
    plt.savefig(output_path, dpi=300)
    plt.show()
    plt.close()

    print(f"Plot saved to: {output_path}")

# 示例调用
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "..", "config", "config.json")  
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    r_in = config["r_in"]
    theta0 = config["theta0_deg"]
    kappaff = config["kappa_ff"]
    kappaK = config["kappa_K"]
    opt_regime = config["optical_regime"]
    dalpha = config["dalpha"]
    tolerance = dalpha / 2
    r_max = config["r_max"]

    # List of psi0_deg values to analyze
    psi0_deg_list = [5.0, 30.0, 60.0, 75.0, 90.0]

    output_dir = os.path.join(base_dir, "..", "output")
    input_npzs = []
    for psi0_deg in psi0_deg_list:
        filename = f"flux_rmax={r_max:.1f}_optical_{opt_regime}_psi0={psi0_deg:.1f}_rin={r_in:.1f}_theta0={theta0:.1f}_kappaff={kappaff:.3f}_kappaK={kappaK:.3f}.npz"
        file_path = os.path.join(output_dir, filename)
        if os.path.exists(file_path):
            input_npzs.append(file_path)
            print(f"Found file: {filename}")
        else:
            print(f"Warning: File not found, skipping: {filename}")

    if len(input_npzs) == 0:
        print("Error: No flux files found. Please run step2 first.")
        exit(1)

    plot_flux_profiles(input_npzs, output_dir, tolerance=tolerance, r_in=r_in, theta0=theta0, kappaff=kappaff, kappaK=kappaK)
