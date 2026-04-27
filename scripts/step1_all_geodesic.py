import numpy as np
import matplotlib.pyplot as plt
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from tqdm import tqdm
import json

# === 史瓦西黑洞度规函数 ===
def B_func_schwarzschild(r, M):
    """史瓦西黑洞度规函数 B(r) = 1 - 2M/r"""
    return 1.0 - 2.0 * M / r


def dB_dr_schwarzschild(r, M):
    """史瓦西黑洞度规导数"""
    return 2.0 * M / (r**2)


# === 初始条件生成 ===
def get_initial_conditions(b, r0):
    tan_phi0 = (b / r0) / np.sqrt(1 - (b**2 / r0**2))  # 渐进平直区域观测
    mu0 = 1 / r0
    dmu_dphi0 = mu0 / tan_phi0
    return np.array([mu0, dmu_dphi0])


# === 微分方程 ===
def geodesic_eq(phi, y, M):  # 默认度规: 1/g_{rr} = -g_{tt} = B = 1 - 2M / r
    mu, dmu_dphi = y
    # 史瓦西度规的测地线方程可化简为：
    # d²μ/dφ² = 3Mμ² - μ
    d2mu_dphi2 = 3 * M * mu**2 - mu
    return np.array([dmu_dphi, d2mu_dphi2])


# === RK4 固定步长积分 ===
def rk4_fixed_step(phi, y, dphi, M, eq):
    k1 = eq(phi, y, M)
    k2 = eq(phi + dphi / 2, y + dphi / 2 * k1, M)
    k3 = eq(phi + dphi / 2, y + dphi / 2 * k2, M)
    k4 = eq(phi + dphi, y + dphi * k3, M)
    y_new = y + dphi / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return y_new


# === 单个轨迹计算函数 ===
def compute_one_geodesic(b, r0, dphi, r_max, phi_max, M, r_h, b_threshold=1.0, dr=0.01):
    if b >= b_threshold:
        return compute_by_phi_step(b, r0, dphi, r_max, phi_max, M, r_h)
    else:
        return compute_by_r_step(b, r0, dr, r_max, M, r_h)


# 用 phi 为步进变量（适合 b 较大）
def compute_by_phi_step(b, r0, dphi, r_max, phi_max, M, r_h):
    y = get_initial_conditions(b, r0)
    phi_vals = [0]
    mu_vals = [y[0]]
    r_prev = 1 / y[0]
    phi = 0

    while phi <= phi_max:
        y = rk4_fixed_step(phi, y, dphi, M, geodesic_eq)
        phi += dphi
        mu = y[0]
        r_current = 1 / mu
        if r_current <= r_h or (r_prev < r_max and r_current >= r_max):  # 掉入黑洞或者越过 r_max
            break
        phi_vals.append(phi)
        mu_vals.append(mu)
        r_prev = r_current

    phi_vals_np = np.array(phi_vals)
    r_vals = 1 / np.array(mu_vals)
    x_vals = r_vals * np.cos(phi_vals_np)
    y_vals_cart = r_vals * np.sin(phi_vals_np)
    dr_list = [r_vals[i] - r_vals[i - 1] for i in range(1, len(r_vals))]

    data = [
        [b, r, phi, dr] for r, phi, dr in zip(r_vals, phi_vals_np, dr_list)
        if r_h <= r <= r_max
    ]
    return (x_vals, y_vals_cart, 'blue', 0.2), data


# 用 r 为步进变量（适合 b 接近 0）
def dphi_dr(r, b, M, b_threshold=1.0):
    numer = b / r**2
    B = B_func_schwarzschild(r, M)  # 默认度规: 1/g_{rr} = -g_{tt} = B = 1 - 2M / r
    denom_sq = 1 - B * b**2 / r**2
    if denom_sq <= 0 or np.isnan(denom_sq):
        raise ValueError(
            f"[dphi/dr Error] Invalid sqrt denom at r = {r:.5f}, b = {b:.5f}. "
            f"Please decrease b_threshold (currently {b_threshold})."
        )
    return numer / np.sqrt(denom_sq)


# 用 r 为步进变量（适合 b 接近 0）
def compute_by_r_step(b, r_start, dr, r_max, M, r_h, b_threshold=1.0):
    r_vals = [r_start]
    phi_vals = [0]
    r = r_start
    phi = 0

    while r > r_h:  # 默认视界在 r_h = 2M 处
        # 动态步长：可按需自定义更复杂的规则
        if r > 10000:
            dr_now = 1000
        elif r > 1000:
            dr_now = 100
        elif r > 100:
            dr_now = 10
        elif r > r_max:
            dr_now = 1
        else:
            dr_now = 0.01

        try:
            # 使用 RK4，注意 r 方向为递减
            k1 = dr_now * dphi_dr(r, b, M, b_threshold)
            k2 = dr_now * dphi_dr(r - 0.5 * dr_now, b, M, b_threshold)
            k3 = dr_now * dphi_dr(r - 0.5 * dr_now, b, M, b_threshold)
            k4 = dr_now * dphi_dr(r - dr_now, b, M, b_threshold)
        except ValueError as e:
            print(f"[compute_by_r_step Warning] {e}")
            break  # 出现非法根号，终止积分

        dphi = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        phi += dphi
        r -= dr_now  # 光子向黑洞靠近

        phi_vals.append(phi)
        r_vals.append(r)

    phi_vals_np = np.array(phi_vals)
    r_vals_np = np.array(r_vals)
    x_vals = r_vals_np * np.cos(phi_vals_np)
    y_vals_cart = r_vals_np * np.sin(phi_vals_np)
    dr_list = [r_vals_np[i] - r_vals_np[i - 1] for i in range(1, len(r_vals_np))]

    data = [
        [b, r, phi, dr] for r, phi, dr in zip(r_vals_np, phi_vals_np, dr_list)
        if r_h <= r <= r_max
    ]
    return (x_vals, y_vals_cart, 'blue', 0.2), data


# === 主程序入口 ===
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "..", "config", "config.json")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    M = config["M"]
    r0 = config["r0"]
    step_b = config["step_b"]
    dphi = config["dphi"]
    r_max = config["r_max"]
    b_max = config["b_max"]
    xmax = config["xmax"]
    ymax = config["ymax"]
    phi_max = 6 * np.pi        # 最大积分角度
    r_h = 2 * M                # 史瓦西黑洞视界

    save_dir = os.path.join(base_dir, "..", "output")
    os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
    save_path_fig = os.path.join(save_dir, f"geodesics_rmax={r_max:.1f}_step_b={step_b:.3f}_dphi={dphi:.3f}_geothick.png")
    save_path_data = os.path.join(save_dir, f"geodesics_rmax={r_max:.1f}_step_b={step_b:.3f}_dphi={dphi:.3f}_geothick.npy")
    plot_xlim = (-xmax, xmax)      # 图像 x 轴范围
    plot_ylim = (-ymax, ymax)      # 图像 y 轴范围

    print(f"史瓦西黑洞外视界 r_h = {r_h:.6f} M")

    if os.path.exists(save_path_data):
        print(f"跳过此步，因为文件已存在: {save_path_data}")
    else:
        print("Running Step 1: Geodesics Calculation")
        start_time = time.time()
        b_range = np.arange(step_b, b_max, step_b)

        curves = []
        all_data = []

        with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
            futures = {executor.submit(compute_one_geodesic, b, r0, dphi, r_max, phi_max, M, r_h): b for b in b_range}
            for future in tqdm(as_completed(futures), total=len(futures), desc="计算中"):
                curve, data = future.result()
                curves.append(curve)
                all_data.extend(data)

        # === 绘图 ===
        fig, ax = plt.subplots(figsize=(10, 10))
        for x_vals, y_vals, color, thickness in curves:
            ax.plot(x_vals, y_vals, color=color, linewidth=thickness)
        circle = plt.Circle((0, 0), r_h, color='black', zorder=10)
        ax.add_patch(circle)
        ax.set_aspect('equal')
        ax.set_xlim(*plot_xlim)
        ax.set_ylim(*plot_ylim)
        ax.set_xlabel("x (M)")
        ax.set_ylabel("y (M)")
        ax.set_title("Light Geodesics near Schwarzschild Black Hole (Fixed Step RK4)")
        ax.grid(False)
        os.makedirs(os.path.dirname(save_path_fig), exist_ok=True)
        plt.savefig(save_path_fig, dpi=300, bbox_inches='tight')
        plt.close()

        # === 保存数据 ===
        all_data_sorted = sorted(all_data, key=lambda x: x[0])  # x[0] 是 b
        all_data_array = np.array(all_data_sorted, dtype=np.float64)
        os.makedirs(os.path.dirname(save_path_data), exist_ok=True)
        np.save(save_path_data, all_data_array)

        print(f"图像保存至：{save_path_fig}")
        print(f"数据保存至：{save_path_data}")
        print(f"数据量：{len(all_data)} 行")
        print(f"第一步耗时：{time.time() - start_time:.2f} 秒")
