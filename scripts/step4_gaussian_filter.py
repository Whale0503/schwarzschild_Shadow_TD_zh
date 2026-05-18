import os
import json
import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree


def build_blurred_map(b_vals, alpha_vals, f_vals, step_b, xmax, ymax, max_distance=0.05):
    print("正在准备绘图数据...")

    x_vals = b_vals * np.cos(alpha_vals)
    y_vals = b_vals * np.sin(alpha_vals)

    mask = (x_vals >= -xmax) & (x_vals <= xmax) & (y_vals >= -ymax) & (y_vals <= ymax)
    x_vals = x_vals[mask]
    y_vals = y_vals[mask]
    f_vals = f_vals[mask]

    print("正在生成规则网格并查找最近邻...")

    grid_spacing = step_b
    grid_x, grid_y = np.mgrid[-xmax:xmax:grid_spacing, -ymax:ymax:grid_spacing]
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    data_points = np.column_stack((x_vals, y_vals))

    tree = cKDTree(data_points)
    dist, idx = tree.query(grid_points, distance_upper_bound=max_distance)

    safe_f = np.zeros(len(grid_points), dtype=np.float64)
    support_flat = np.zeros(len(grid_points), dtype=bool)
    valid_mask = np.isfinite(dist)
    if np.any(valid_mask):
        sampled_f = f_vals[idx[valid_mask]]
        finite_sample_mask = np.isfinite(sampled_f)
        valid_indices = np.flatnonzero(valid_mask)[finite_sample_mask]
        safe_f[valid_indices] = sampled_f[finite_sample_mask]
        support_flat[valid_indices] = True

    i_raw = safe_f.reshape(grid_x.shape)
    support_map = support_flat.reshape(grid_x.shape)
    support_fraction = float(np.mean(support_map))
    print(f"有限样本覆盖率: {support_fraction:.6f} ({int(np.count_nonzero(support_map))}/{support_map.size})")

    sigma_pixels = 0.50 / step_b
    i_blur = gaussian_filter(i_raw, sigma=sigma_pixels)

    return grid_x, grid_y, i_raw, i_blur, support_map, support_fraction, grid_spacing, sigma_pixels



def save_blur_png(grid_x, grid_y, i_blur, save_path, config):
    print("正在绘图...")

    vmax = np.nanpercentile(i_blur, 99.9)
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = np.nanmax(i_blur) if np.any(np.isfinite(i_blur)) else 1.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    font_size = 16
    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap",
        [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)],
    )

    plt.figure(figsize=(8, 8))
    plt.pcolormesh(grid_x, grid_y, i_blur, shading="auto", cmap=cmap, vmin=0, vmax=vmax)

    plt.text(
        x=4,
        y=-16,
        s=(
            rf"$\theta_0 = {config['theta0_deg']:.0f}^\circ\,,\ \psi_0 = {config['psi0_deg']:.0f}^\circ$" "\n"
            rf"$r_{{\rm in}} = {config['r_in']:.0f}M\,,\ r_{{\rm out}} = {config['r_max']:.0f}M$" "\n"
            rf"$\kappa_{{\rm ff}} = {config['kappa_ff']:.1f}\,,\ \kappa_{{\rm K}} = {config['kappa_K']:.1f}$"
        ),
        color="white",
        fontsize=font_size,
        ha="left",
        va="bottom",
    )

    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    plt.xlim([-config["shadow_xmax"], config["shadow_xmax"]])
    plt.ylim([-config["shadow_ymax"], config["shadow_ymax"]])
    plt.gca().set_aspect("equal", adjustable="box")

    plt.savefig(save_path, dpi=300, pad_inches=0, bbox_inches="tight")
    plt.close()
    print(f"图像已成功保存至：{save_path}")



def make_shadow(step_b, input_path, plot_path, map_path, config):
    start_time = time.time()

    data = np.load(input_path)
    b_vals = data["b"]
    alpha_vals = data["alpha"]
    f_vals = data["F"]

    grid_x, grid_y, i_raw, i_blur, support_map, support_fraction, grid_spacing, sigma_pixels = build_blurred_map(
        b_vals,
        alpha_vals,
        f_vals,
        step_b,
        config["shadow_xmax"],
        config["shadow_ymax"],
    )

    if not os.path.exists(map_path):
        np.savez_compressed(
            map_path,
            grid_x=grid_x.astype(np.float32),
            grid_y=grid_y.astype(np.float32),
            I_raw=i_raw.astype(np.float32),
            I_blur=i_blur.astype(np.float32),
            support_map=support_map,
            support_fraction=np.float64(support_fraction),
            grid_spacing=np.float64(grid_spacing),
            sigma_pixels=np.float64(sigma_pixels),
            xmax=np.float64(config["shadow_xmax"]),
            ymax=np.float64(config["shadow_ymax"]),
        )
        print(f"模糊数值图已成功保存至：{map_path}")
    else:
        print(f"跳过保存数值图，因为文件已存在: {map_path}")

    if not os.path.exists(plot_path):
        save_blur_png(grid_x, grid_y, i_blur, plot_path, config)
    else:
        print(f"跳过保存图像，因为文件已存在: {plot_path}")

    elapsed = time.time() - start_time
    print(f"代码执行时间: {elapsed:.2f} 秒.")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "..", "config", "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    step_b = config["step_b"]
    opt_regime = config["optical_regime"].lower()
    if opt_regime == "intermediate":
        chi = config["absorption_coefficient"]
        output_opt = f"{opt_regime}_{chi:.3f}"
    else:
        output_opt = opt_regime

    output_dir = os.path.join(base_dir, "..", "output")
    input_name = (
        f"flux_rmax={config['r_max']:.1f}_optical_{output_opt}"
        f"_psi0={config['psi0_deg']:.1f}_rin={config['r_in']:.1f}"
        f"_theta0={config['theta0_deg']:.1f}_kappaff={config['kappa_ff']:.3f}"
        f"_kappaK={config['kappa_K']:.3f}.npz"
    )
    input_path = os.path.join(output_dir, input_name)
    file_name = os.path.basename(input_path).replace(".npz", "")
    plot_path = os.path.join(output_dir, file_name + "_fliter.png")
    map_path = os.path.join(output_dir, file_name + "_fliter_map.npz")

    if os.path.exists(plot_path) and os.path.exists(map_path):
        print(f"跳过此步，因为文件已存在: {plot_path} 和 {map_path}")
    else:
        print("Running Step 4: Shadow Plotting")
        make_shadow(step_b, input_path, plot_path, map_path, config)
