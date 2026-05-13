import os
import json
import time

import numpy as np
from scipy.interpolate import RegularGridInterpolator


RING_SELECTOR_TAG = "gaussian_bc_peakfit_v2"
CRITICAL_B_SIGMA_FACTOR = 0.05


def load_blur_map(map_path):
    data = np.load(map_path)
    return data["grid_x"], data["grid_y"], data["I_blur"], float(data["grid_spacing"])



def format_step5_value(value):
    if value is None:
        return "full"
    return f"{float(value):.3f}"



def build_step5_output_paths(output_dir, file_name, dalpha, ring_b_min, ring_b_max, ring_peak_rel_threshold):
    dalpha_tag = format_step5_value(dalpha)
    bmin_tag = format_step5_value(ring_b_min)
    bmax_tag = format_step5_value(ring_b_max)
    thr_tag = format_step5_value(ring_peak_rel_threshold)
    polar_name = f"{file_name}_fliter_polar_dalpha={dalpha_tag}_bmax={bmax_tag}.npz"
    ring_name = (
        f"{file_name}_fliter_ring_{RING_SELECTOR_TAG}"
        f"_dalpha={dalpha_tag}_bmin={bmin_tag}_bmax={bmax_tag}_thr={thr_tag}.npz"
    )
    return os.path.join(output_dir, polar_name), os.path.join(output_dir, ring_name)



def build_polar_map(grid_x, grid_y, i_blur, dalpha, grid_spacing, b_max=None):
    x_axis = grid_x[:, 0].astype(np.float64)
    y_axis = grid_y[0, :].astype(np.float64)
    map_b_max = min(np.max(np.abs(x_axis)), np.max(np.abs(y_axis)))
    if b_max is None:
        b_max = map_b_max
    else:
        b_max = min(float(b_max), map_b_max)

    n_phi = max(1, int(round(2 * np.pi / dalpha)))
    # phi 是观测平面上的方位角，不是轨道积分中的几何 phi。
    phi_grid = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    b_grid = np.arange(0.0, b_max + 0.5 * grid_spacing, grid_spacing)

    interpolator = RegularGridInterpolator(
        (x_axis, y_axis),
        i_blur.astype(np.float64),
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )

    b_mesh, phi_mesh = np.meshgrid(b_grid, phi_grid, indexing="xy")
    x_samples = b_mesh * np.cos(phi_mesh)
    y_samples = b_mesh * np.sin(phi_mesh)
    sample_points = np.column_stack((x_samples.ravel(), y_samples.ravel()))
    i_polar = interpolator(sample_points).reshape(phi_mesh.shape)
    i_polar = np.clip(i_polar, 0.0, None)

    return b_grid, phi_grid, i_polar



def find_local_peak_indices(profile):
    peak_indices = []
    n_profile = len(profile)
    i = 0
    while i < n_profile:
        j = i
        while j + 1 < n_profile and profile[j + 1] == profile[i]:
            j += 1

        peak_value = profile[i]
        has_left_neighbor = i > 0
        has_right_neighbor = j < n_profile - 1
        higher_than_left = has_left_neighbor and profile[i - 1] < peak_value
        higher_than_right = has_right_neighbor and profile[j + 1] < peak_value

        if (
            (not has_left_neighbor or higher_than_left)
            and (not has_right_neighbor or higher_than_right)
            and (higher_than_left or higher_than_right)
        ):
            peak_indices.append(j)

        i = j + 1

    return np.array(peak_indices, dtype=np.int64)



def parabolic_peak_position(x_values, y_values, peak_index):
    if peak_index <= 0 or peak_index >= len(x_values) - 1:
        return float(x_values[peak_index])

    x1, x2, x3 = x_values[peak_index - 1], x_values[peak_index], x_values[peak_index + 1]
    y1, y2, y3 = y_values[peak_index - 1], y_values[peak_index], y_values[peak_index + 1]
    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    if denom == 0.0:
        return float(x2)

    quad_a = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    quad_b = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom
    if quad_a >= 0.0:
        return float(x2)

    x_vertex = -quad_b / (2.0 * quad_a)
    return float(np.clip(x_vertex, min(x1, x3), max(x1, x3)))



def extract_ring_from_polar(b_grid, phi_grid, i_polar, search_b_min, search_b_max, peak_rel_threshold, mass):
    search_mask = (b_grid >= search_b_min) & (b_grid <= search_b_max)
    if not np.any(search_mask):
        raise ValueError("The configured ring search interval does not overlap with b_grid.")

    search_b = b_grid[search_mask].astype(np.float64)
    search_profiles = i_polar[:, search_mask].astype(np.float64)
    if search_b.size == 0:
        raise ValueError("No radial samples remain inside the configured ring search interval.")

    critical_b = 3.0 * np.sqrt(3.0) * float(mass)
    sigma_b = CRITICAL_B_SIGMA_FACTOR * float(search_b_max - search_b_min)
    if sigma_b <= 0.0:
        sigma_b = max(float(search_b[-1] - search_b[0]), 1e-6)
    gaussian_weight = np.exp(-0.5 * ((search_b - critical_b) / sigma_b) ** 2)

    selector_score = search_profiles * gaussian_weight[None, :]
    peak_indices = np.argmax(selector_score, axis=1)
    b_ring = np.array(
        [parabolic_peak_position(search_b, selector_score[i], int(peak_indices[i])) for i in range(len(phi_grid))],
        dtype=np.float64,
    )
    i_ring = search_profiles[np.arange(len(phi_grid)), peak_indices]
    x_ring = b_ring * np.cos(phi_grid)
    y_ring = b_ring * np.sin(phi_grid)

    phi_extended = np.concatenate((phi_grid, [2.0 * np.pi]))
    b_ring_extended = np.concatenate((b_ring, [b_ring[0]]))
    diameter_x = float(np.interp(0.0, phi_extended, b_ring_extended) + np.interp(np.pi, phi_extended, b_ring_extended))
    diameter_y = float(
        np.interp(0.5 * np.pi, phi_extended, b_ring_extended)
        + np.interp(1.5 * np.pi, phi_extended, b_ring_extended)
    )

    return {
        "phi": phi_grid,
        "b_ring": b_ring,
        "I_ring": i_ring,
        "x_ring": x_ring,
        "y_ring": y_ring,
        "diameter_mean": 2.0 * float(np.mean(b_ring)),
        "diameter_std": 2.0 * float(np.std(b_ring)),
        "diameter_min": 2.0 * float(np.min(b_ring)),
        "diameter_max": 2.0 * float(np.max(b_ring)),
        "diameter_x": diameter_x,
        "diameter_y": diameter_y,
        "search_b_min": float(search_b_min),
        "search_b_max": float(search_b_max),
        "critical_b": float(critical_b),
        "selector_sigma_b": float(sigma_b),
        "selector_tag": RING_SELECTOR_TAG,
    }



def main():
    start_time = time.time()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "..", "config", "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    opt_regime = config["optical_regime"].lower()
    if opt_regime == "intermediate":
        chi = config["absorption_coefficient"]
        output_opt = f"{opt_regime}_{chi:.3f}"
    else:
        output_opt = opt_regime

    output_dir = os.path.join(base_dir, "..", "output")
    file_name = (
        f"flux_rmax={config['r_max']:.1f}_optical_{output_opt}"
        f"_psi0={config['psi0_deg']:.1f}_rin={config['r_in']:.1f}"
        f"_theta0={config['theta0_deg']:.1f}_kappaff={config['kappa_ff']:.3f}"
        f"_kappaK={config['kappa_K']:.3f}"
    )
    map_path = os.path.join(output_dir, file_name + "_fliter_map.npz")
    dalpha = float(config["dalpha"])
    search_b_min = float(config.get("ring_b_min", 0.0))
    raw_search_b_max = config.get("ring_b_max")
    ring_peak_rel_threshold = float(config.get("ring_peak_rel_threshold", 0.5))
    polar_path, ring_path = build_step5_output_paths(
        output_dir,
        file_name,
        dalpha,
        search_b_min,
        raw_search_b_max,
        ring_peak_rel_threshold,
    )

    if os.path.exists(polar_path) and os.path.exists(ring_path):
        print(f"跳过此步，因为文件已存在: {polar_path} 和 {ring_path}")
        return

    print("Running Step 5: Extract Ring From Blurred Shadow")

    grid_x, grid_y, i_blur, grid_spacing = load_blur_map(map_path)

    if os.path.exists(polar_path):
        polar_data = np.load(polar_path)
        b_grid = polar_data["b_grid"].astype(np.float64)
        phi_grid = polar_data["phi_grid"].astype(np.float64)
        i_polar = polar_data["I_polar"].astype(np.float64)
        print(f"跳过极坐标强度图重建，因为文件已存在: {polar_path}")
    else:
        b_grid, phi_grid, i_polar = build_polar_map(
            grid_x,
            grid_y,
            i_blur,
            dalpha=dalpha,
            grid_spacing=grid_spacing,
            b_max=raw_search_b_max,
        )
        np.savez_compressed(
            polar_path,
            b_grid=b_grid.astype(np.float32),
            phi_grid=phi_grid.astype(np.float64),
            I_polar=i_polar.astype(np.float32),
        )
        print(f"极坐标强度图已成功保存至：{polar_path}")

    if raw_search_b_max is None:
        search_b_max = float(b_grid[-1])
    else:
        search_b_max = min(float(raw_search_b_max), float(b_grid[-1]))

    if search_b_min > search_b_max:
        raise ValueError(
            f"Invalid ring search interval: ring_b_min={search_b_min} is larger than ring_b_max={search_b_max}."
        )

    print(f"亮环搜索区间: [{search_b_min:.6f}, {search_b_max:.6f}]")
    print(f"亮环相对峰值阈值: {ring_peak_rel_threshold:.6f}")
    ring_result = extract_ring_from_polar(
        b_grid,
        phi_grid,
        i_polar,
        search_b_min,
        search_b_max,
        ring_peak_rel_threshold,
        mass=float(config.get("M", 1.0)),
    )

    if not os.path.exists(ring_path):
        np.savez_compressed(
            ring_path,
            phi=ring_result["phi"].astype(np.float64),
            b_ring=ring_result["b_ring"].astype(np.float32),
            I_ring=ring_result["I_ring"].astype(np.float32),
            x_ring=ring_result["x_ring"].astype(np.float32),
            y_ring=ring_result["y_ring"].astype(np.float32),
            diameter_mean=np.float64(ring_result["diameter_mean"]),
            diameter_std=np.float64(ring_result["diameter_std"]),
            diameter_min=np.float64(ring_result["diameter_min"]),
            diameter_max=np.float64(ring_result["diameter_max"]),
            diameter_x=np.float64(ring_result["diameter_x"]),
            diameter_y=np.float64(ring_result["diameter_y"]),
            search_b_min=np.float64(ring_result["search_b_min"]),
            search_b_max=np.float64(ring_result["search_b_max"]),
            critical_b=np.float64(ring_result["critical_b"]),
            selector_sigma_b=np.float64(ring_result["selector_sigma_b"]),
            selector_tag=np.array(ring_result["selector_tag"]),
        )
        print(f"亮环结果已成功保存至：{ring_path}")
    else:
        print(f"跳过保存亮环结果，因为文件已存在: {ring_path}")

    print(f"亮环提取器: {RING_SELECTOR_TAG}, critical_b={ring_result['critical_b']:.6f}, sigma_b={ring_result['selector_sigma_b']:.6f}")
    print(
        "亮环直径统计: "
        f"mean={ring_result['diameter_mean']:.6f}, "
        f"std={ring_result['diameter_std']:.6f}, "
        f"x={ring_result['diameter_x']:.6f}, "
        f"y={ring_result['diameter_y']:.6f}"
    )

    elapsed = time.time() - start_time
    print(f"代码执行时间: {elapsed:.2f} 秒.")


if __name__ == "__main__":
    main()
