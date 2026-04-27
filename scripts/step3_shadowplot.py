import os
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial import cKDTree
import json


def make_shadow(step_b, input_path):
    start_time = time.time()

    # 读取 npz 文件
    data = np.load(input_path)
    b_vals = data["b"]
    alpha_vals = data["alpha"]
    F_vals = data["F"]

    # 进入绘图函数
    plot_density(b_vals, alpha_vals, F_vals, step_b, save_path=plot_path)

    elapsed = time.time() - start_time
    print(f"第三步耗时: {elapsed:.2f} 秒.")


def plot_density(b_vals, alpha_vals, F_vals, step_b, save_path, max_distance=0.05):
    print("正在准备绘图数据...")

    # 计算坐标
    x_vals = b_vals * np.cos(alpha_vals)
    y_vals = b_vals * np.sin(alpha_vals)

    # 限制绘图范围
    mask = (x_vals >= -xmax) & (x_vals <= xmax) & (y_vals >= -ymax) & (y_vals <= ymax)
    x_vals = x_vals[mask]
    y_vals = y_vals[mask]
    F_vals = F_vals[mask]

    print("正在生成规则网格并查找最近邻...")

    # 创建规则网格
    grid_spacing = step_b / 1
    grid_x, grid_y = np.mgrid[-xmax:xmax:grid_spacing, -ymax:ymax:grid_spacing]
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    data_points = np.column_stack((x_vals, y_vals))

    # 使用 KDTree 查找最近邻
    tree = cKDTree(data_points)
    dist, idx = tree.query(grid_points, distance_upper_bound=max_distance)
    safe_F = np.zeros(len(grid_points))
    valid_mask = np.isfinite(dist)
    safe_F[valid_mask] = F_vals[idx[valid_mask]]
    grid_F = safe_F.reshape(grid_x.shape)
    vmin = 0
    vmax = np.percentile(grid_F, 99.9)

    # 自定义 colormap：黑 -> 红 -> 黄 -> 白
    cmap = LinearSegmentedColormap.from_list("custom_cmap", [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)])

    print("正在绘图...")

    plt.figure(figsize=(8, 8))
    plt.pcolormesh(grid_x, grid_y, grid_F, shading='auto', cmap=cmap, vmin=vmin, vmax=vmax)

    # 去掉坐标轴刻度与标签
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])
    plt.xlim([-xmax, xmax])
    plt.ylim([-ymax, ymax])
    plt.gca().set_aspect('equal', adjustable='box')

    # 保存图像
    plt.savefig(save_path, dpi=300, pad_inches=0, bbox_inches='tight')
    plt.close()
    print(f"图像已成功保存至：{save_path}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "..", "config", "config.json")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    step_b = config["step_b"]
    xmax = config["shadow_xmax"]
    ymax = config["shadow_ymax"]
    kappa_ff = config["kappa_ff"]
    kappa_K = config["kappa_K"]
    r_max = config["r_max"]
    r_in = config["r_in"]
    psi0_deg = config["psi0_deg"]
    theta0_deg = config["theta0_deg"]
    opt_regime = config["optical_regime"].lower()
    if opt_regime == "intermediate":
        chi = config["absorption_coefficient"]
        output_opt = f"{opt_regime}_{chi:.3f}"
    else:
        output_opt = opt_regime

    output_dir = os.path.join(base_dir, "..", "output")
    input_name = f"flux_rmax={r_max:.1f}_optical_{output_opt}_psi0={psi0_deg:.1f}_rin={r_in:.1f}_theta0={theta0_deg:.1f}_kappaff={kappa_ff:.3f}_kappaK={kappa_K:.3f}.npz"
    input_path = os.path.join(output_dir, input_name)
    file_name = os.path.basename(input_path).replace(".npz", "")
    plot_path = os.path.join(output_dir, file_name + ".png")
    if os.path.exists(plot_path):
       print(f"跳过此步，因为文件已存在: {plot_path}")
    else:
       print("Running Step 3: Shadow Plotting")
       make_shadow(step_b, input_path)
