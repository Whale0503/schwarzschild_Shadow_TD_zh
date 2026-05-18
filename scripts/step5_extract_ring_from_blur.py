import os
import json
import time

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


DEFAULT_SELECTOR_TAG = "gaussian_bc_peakfit_v2"
BLIND_SELECTOR_TAG = "blind_annular_basin_v1"
CRITICAL_B_SIGMA_FACTOR = 0.05
BLIND_PROFILE_NAMES = ("mean", "median", "robust")
BLIND_PRIMARY_PROFILE_NAME = BLIND_PROFILE_NAMES[0]
BLIND_PEAK_SUPPORT_FRACTION = 0.99
BLIND_MIN_SUPPORT_HALF_WIDTH_BINS = 3


class BlindPeakResult:
    def __init__(
        self,
        annular_start_b,
        annular_start_method,
        peak_b,
        peak_value,
        peak_prominence,
        left_base_b,
        right_base_b,
        basin_width,
        smoothed_peak_b,
        smoothed_peak_value,
        support_left_b,
        support_right_b,
        support_width,
    ):
        self.annular_start_b = float(annular_start_b)
        self.annular_start_method = str(annular_start_method)
        self.peak_b = float(peak_b)
        self.peak_value = float(peak_value)
        self.peak_prominence = float(peak_prominence)
        self.left_base_b = float(left_base_b)
        self.right_base_b = float(right_base_b)
        self.basin_width = float(basin_width)
        self.smoothed_peak_b = float(smoothed_peak_b)
        self.smoothed_peak_value = float(smoothed_peak_value)
        self.support_left_b = float(support_left_b)
        self.support_right_b = float(support_right_b)
        self.support_width = float(support_width)



def load_blur_map(map_path):
    data = np.load(map_path)
    return data["grid_x"], data["grid_y"], data["I_blur"], float(data["grid_spacing"])



def load_image_maps(map_path):
    data = np.load(map_path)
    return data["grid_x"], data["grid_y"], data["I_raw"], data["I_blur"], float(data["grid_spacing"])



def format_step5_value(value):
    if value is None:
        return "full"
    return f"{float(value):.3f}"



def build_step5_output_paths(output_dir, file_name, dalpha, ring_b_min, ring_b_max, ring_peak_rel_threshold, selector_tag=DEFAULT_SELECTOR_TAG):
    dalpha_tag = format_step5_value(dalpha)
    bmin_tag = format_step5_value(ring_b_min)
    bmax_tag = format_step5_value(ring_b_max)
    thr_tag = format_step5_value(ring_peak_rel_threshold)
    polar_name = f"{file_name}_fliter_polar_dalpha={dalpha_tag}_bmax={bmax_tag}.npz"
    ring_name = (
        f"{file_name}_fliter_ring_{selector_tag}"
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



def robust_center(values, lower_percentile=10.0, upper_percentile=90.0):
    finite_samples = np.asarray(values, dtype=np.float64)
    finite_samples = finite_samples[np.isfinite(finite_samples)]
    if finite_samples.size == 0:
        return 0.0
    if finite_samples.size < 4:
        return float(np.mean(finite_samples))

    lower, upper = np.nanpercentile(finite_samples, [lower_percentile, upper_percentile])
    clipped = finite_samples[(finite_samples >= lower) & (finite_samples <= upper)]
    if clipped.size == 0:
        clipped = finite_samples
    return float(np.mean(clipped))



def compute_polar_radial_profiles(b_grid, i_polar):
    polar_values = np.asarray(i_polar, dtype=np.float64)
    robust = np.array([robust_center(polar_values[:, i]) for i in range(polar_values.shape[1])], dtype=np.float64)
    finite_polar_values = np.where(np.isfinite(polar_values), polar_values, 0.0)
    return {
        "b": b_grid.astype(np.float64),
        "mean": np.mean(finite_polar_values, axis=0).astype(np.float64),
        "median": np.median(finite_polar_values, axis=0).astype(np.float64),
        "robust": robust,
        "max": np.max(finite_polar_values, axis=0).astype(np.float64),
        "count": np.full_like(b_grid, finite_polar_values.shape[0], dtype=np.int64),
    }



def _nearest_minimum_indices(minima, peak_index, n_points):
    left_candidates = minima[minima < peak_index]
    right_candidates = minima[minima > peak_index]
    left_index = int(left_candidates[-1]) if left_candidates.size else 0
    right_index = int(right_candidates[0]) if right_candidates.size else n_points - 1
    return left_index, right_index



def find_dominant_annular_peak(b_grid, profile, smooth_sigma_bins=2.0):
    b_grid = np.asarray(b_grid, dtype=np.float64)
    profile = np.asarray(profile, dtype=np.float64)
    if b_grid.ndim != 1 or profile.ndim != 1 or len(b_grid) != len(profile):
        raise ValueError("Invalid radial profile arrays for blind Step 5 selector.")
    if len(b_grid) < 5:
        raise ValueError("Profile is too short for blind Step 5 selector.")

    finite_mask = np.isfinite(profile)
    if not np.any(finite_mask):
        raise ValueError("Profile has no finite values for blind Step 5 selector.")
    if not np.all(finite_mask):
        profile = np.where(finite_mask, profile, 0.0)

    db = float(np.median(np.diff(b_grid)))
    analysis_profile = gaussian_filter1d(profile, sigma=smooth_sigma_bins) if smooth_sigma_bins > 0 else profile.copy()

    minima, _ = find_peaks(-analysis_profile)
    positive_minima = minima[b_grid[minima] > max(2.0 * db, 1e-12)]
    if positive_minima.size:
        annular_start_index = int(positive_minima[0])
        annular_start_b = float(b_grid[annular_start_index])
        annular_start_method = "first radial minimum"
    else:
        annular_start_b = max(20.0 * db, 0.1)
        annular_start_index = int(np.searchsorted(b_grid, annular_start_b, side="left"))
        annular_start_method = "fallback central-core cutoff"

    prominence_floor = max(float(np.nanmax(analysis_profile)) * 1e-4, 1e-12)
    peak_indices, properties = find_peaks(analysis_profile, prominence=prominence_floor)

    candidate_indices = np.where(b_grid[peak_indices] > annular_start_b)[0]
    if candidate_indices.size:
        chosen_local = int(candidate_indices[np.argmax(properties["prominences"][candidate_indices])])
        smooth_peak_index = int(peak_indices[chosen_local])
        left_base_index = int(properties["left_bases"][chosen_local])
        right_base_index = int(properties["right_bases"][chosen_local])
        peak_prominence = float(properties["prominences"][chosen_local])
    else:
        search_start_index = min(max(annular_start_index, 0), len(b_grid) - 2)
        if search_start_index >= len(b_grid) - 1:
            raise ValueError("Could not find a non-zero-radius annular region in blind Step 5 selector.")
        smooth_peak_index = search_start_index + int(np.argmax(analysis_profile[search_start_index:]))
        left_base_index, right_base_index = _nearest_minimum_indices(minima, smooth_peak_index, len(b_grid))
        peak_prominence = float(analysis_profile[smooth_peak_index] - min(analysis_profile[left_base_index], analysis_profile[right_base_index]))

    basin_slice = slice(left_base_index, right_base_index + 1)
    original_segment = profile[basin_slice]
    refined_peak_index = left_base_index + int(np.argmax(original_segment))
    refined_peak_b = float(parabolic_peak_position(b_grid, profile, refined_peak_index))
    refined_peak_value = float(np.interp(refined_peak_b, b_grid, profile))

    return BlindPeakResult(
        annular_start_b=annular_start_b,
        annular_start_method=annular_start_method,
        peak_b=refined_peak_b,
        peak_value=refined_peak_value,
        peak_prominence=peak_prominence,
        left_base_b=float(b_grid[left_base_index]),
        right_base_b=float(b_grid[right_base_index]),
        basin_width=float(b_grid[right_base_index] - b_grid[left_base_index]),
        smoothed_peak_b=float(b_grid[smooth_peak_index]),
        smoothed_peak_value=float(analysis_profile[smooth_peak_index]),
        support_left_b=refined_peak_b,
        support_right_b=refined_peak_b,
        support_width=0.0,
    )



def find_blind_peak_support_window(b_grid, profile, peak_b, peak_value, support_fraction=BLIND_PEAK_SUPPORT_FRACTION):
    b_grid = np.asarray(b_grid, dtype=np.float64)
    profile = np.asarray(profile, dtype=np.float64)
    if b_grid.ndim != 1 or profile.ndim != 1 or len(b_grid) != len(profile):
        raise ValueError("Invalid radial profile arrays for blind Step 5 support window.")

    support_fraction = float(np.clip(support_fraction, 0.0, 1.0))
    peak_index = int(np.argmin(np.abs(b_grid - float(peak_b))))
    threshold_value = float(peak_value) * support_fraction

    left_index = peak_index
    while left_index > 0 and profile[left_index - 1] >= threshold_value:
        left_index -= 1
    right_index = peak_index
    while right_index < len(b_grid) - 1 and profile[right_index + 1] >= threshold_value:
        right_index += 1

    min_half_width_bins = max(int(BLIND_MIN_SUPPORT_HALF_WIDTH_BINS), 0)
    left_index = min(left_index, max(0, peak_index - min_half_width_bins))
    right_index = max(right_index, min(len(b_grid) - 1, peak_index + min_half_width_bins))
    return float(b_grid[left_index]), float(b_grid[right_index])



def find_blind_annular_basin(b_grid, i_polar):
    profiles = compute_polar_radial_profiles(b_grid, i_polar)
    peak_results = {
        name: find_dominant_annular_peak(profiles["b"], profiles[name])
        for name in BLIND_PROFILE_NAMES
    }
    primary_result = peak_results[BLIND_PRIMARY_PROFILE_NAME]
    support_left_b, support_right_b = find_blind_peak_support_window(
        profiles["b"],
        profiles[BLIND_PRIMARY_PROFILE_NAME],
        primary_result.peak_b,
        primary_result.peak_value,
    )
    primary_result.support_left_b = float(support_left_b)
    primary_result.support_right_b = float(support_right_b)
    primary_result.support_width = float(support_right_b - support_left_b)
    return profiles, peak_results, primary_result



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
        "selector_tag": DEFAULT_SELECTOR_TAG,
    }



def extract_ring_from_auto_basin(b_grid, phi_grid, i_polar, basin_result):
    search_b_min = float(basin_result.support_left_b)
    search_b_max = float(basin_result.support_right_b)
    anchor_b = float(np.clip(basin_result.peak_b, search_b_min, search_b_max))
    search_mask = (b_grid >= search_b_min) & (b_grid <= search_b_max)
    if not np.any(search_mask):
        raise ValueError("The blind Step 5 physical-anchor window does not overlap with b_grid.")

    search_b = b_grid[search_mask].astype(np.float64)
    search_profiles = np.asarray(i_polar[:, search_mask], dtype=np.float64)
    search_profiles = np.where(np.isfinite(search_profiles), search_profiles, 0.0)

    b_ring = np.zeros(len(phi_grid), dtype=np.float64)
    i_ring = np.zeros(len(phi_grid), dtype=np.float64)
    interior_peak_count = 0
    anchor_fallback_count = 0
    for i in range(len(phi_grid)):
        profile = search_profiles[i]
        peak_indices = find_local_peak_indices(profile)
        interior_peak_indices = peak_indices[(peak_indices > 0) & (peak_indices < len(search_b) - 1)]
        if interior_peak_indices.size:
            chosen_index = int(interior_peak_indices[np.argmin(np.abs(search_b[interior_peak_indices] - anchor_b))])
            refined_b = float(parabolic_peak_position(search_b, profile, chosen_index))
            interior_peak_count += 1
        else:
            refined_b = anchor_b
            anchor_fallback_count += 1
        b_ring[i] = refined_b
        i_ring[i] = float(np.interp(refined_b, search_b, profile))
    x_ring = b_ring * np.cos(phi_grid)
    y_ring = b_ring * np.sin(phi_grid)

    phi_extended = np.concatenate((phi_grid, [2.0 * np.pi]))
    b_ring_extended = np.concatenate((b_ring, [b_ring[0]]))
    diameter_x = float(np.interp(0.0, phi_extended, b_ring_extended) + np.interp(np.pi, phi_extended, b_ring_extended))
    diameter_y = float(
        np.interp(0.5 * np.pi, phi_extended, b_ring_extended)
        + np.interp(1.5 * np.pi, phi_extended, b_ring_extended)
    )

    total_phi = max(len(phi_grid), 1)
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
        "search_b_min": search_b_min,
        "search_b_max": search_b_max,
        "critical_b": np.float64(np.nan),
        "selector_sigma_b": np.float64(np.nan),
        "selector_tag": BLIND_SELECTOR_TAG,
        "annular_start_b": float(basin_result.annular_start_b),
        "annular_start_method": basin_result.annular_start_method,
        "left_base_b": float(basin_result.left_base_b),
        "right_base_b": float(basin_result.right_base_b),
        "support_left_b": float(basin_result.support_left_b),
        "support_right_b": float(basin_result.support_right_b),
        "support_width": float(basin_result.support_width),
        "anchor_b": anchor_b,
        "interior_peak_count": np.int64(interior_peak_count),
        "anchor_fallback_count": np.int64(anchor_fallback_count),
        "interior_peak_fraction": float(interior_peak_count / total_phi),
        "anchor_fallback_fraction": float(anchor_fallback_count / total_phi),
    }



def resolve_selector_tag(config):
    selector_tag = str(config.get("ring_selector", DEFAULT_SELECTOR_TAG)).strip() or DEFAULT_SELECTOR_TAG
    if selector_tag not in (DEFAULT_SELECTOR_TAG, BLIND_SELECTOR_TAG):
        raise ValueError(
            f"Unsupported ring_selector={selector_tag!r}. Expected {DEFAULT_SELECTOR_TAG!r} or {BLIND_SELECTOR_TAG!r}."
        )
    return selector_tag



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
    selector_tag = resolve_selector_tag(config)

    polar_b_max_for_path = None if selector_tag == BLIND_SELECTOR_TAG else raw_search_b_max
    ring_b_min_for_path = 0.0 if selector_tag == BLIND_SELECTOR_TAG else search_b_min
    polar_path, ring_path = build_step5_output_paths(
        output_dir,
        file_name,
        dalpha,
        ring_b_min_for_path,
        polar_b_max_for_path,
        ring_peak_rel_threshold,
        selector_tag=selector_tag,
    )

    if os.path.exists(polar_path) and os.path.exists(ring_path):
        print(f"跳过此步，因为文件已存在: {polar_path} 和 {ring_path}")
        return

    print("Running Step 5: Extract Ring From Blurred Shadow")
    print(f"亮环提取器: {selector_tag}")

    if selector_tag == BLIND_SELECTOR_TAG:
        grid_x, grid_y, i_raw, i_blur, grid_spacing = load_image_maps(map_path)
    else:
        grid_x, grid_y, i_blur, grid_spacing = load_blur_map(map_path)
        i_raw = None

    if os.path.exists(polar_path):
        polar_data = np.load(polar_path)
        b_grid = polar_data["b_grid"].astype(np.float64)
        phi_grid = polar_data["phi_grid"].astype(np.float64)
        i_polar = polar_data["I_polar"].astype(np.float64)
        print(f"跳过极坐标强度图重建，因为文件已存在: {polar_path}")
    else:
        build_b_max = None if selector_tag == BLIND_SELECTOR_TAG else raw_search_b_max
        b_grid, phi_grid, i_polar = build_polar_map(
            grid_x,
            grid_y,
            i_blur,
            dalpha=dalpha,
            grid_spacing=grid_spacing,
            b_max=build_b_max,
        )
        np.savez_compressed(
            polar_path,
            b_grid=b_grid.astype(np.float32),
            phi_grid=phi_grid.astype(np.float64),
            I_polar=i_polar.astype(np.float32),
        )
        print(f"极坐标强度图已成功保存至：{polar_path}")

    if selector_tag == BLIND_SELECTOR_TAG:
        raw_b_grid, raw_phi_grid, raw_i_polar = build_polar_map(
            grid_x,
            grid_y,
            i_raw,
            dalpha=dalpha,
            grid_spacing=grid_spacing,
            b_max=None,
        )
        _, raw_peak_results, raw_basin_result = find_blind_annular_basin(raw_b_grid, raw_i_polar)
        _, blur_peak_results, blur_basin_result = find_blind_annular_basin(b_grid, i_polar)
        ring_result = extract_ring_from_auto_basin(
            b_grid,
            phi_grid,
            i_polar,
            raw_basin_result,
        )
        peak_b_values = np.array([blur_peak_results[name].peak_b for name in BLIND_PROFILE_NAMES], dtype=np.float64)
        ring_result["peak_b_mean"] = float(blur_peak_results["mean"].peak_b)
        ring_result["peak_b_median"] = float(blur_peak_results["median"].peak_b)
        ring_result["peak_b_robust"] = float(blur_peak_results["robust"].peak_b)
        ring_result["peak_spread"] = float(np.max(peak_b_values) - np.min(peak_b_values))
        ring_result["physical_anchor_b"] = float(raw_peak_results["mean"].peak_b)
        ring_result["physical_anchor_median_b"] = float(raw_peak_results["median"].peak_b)
        ring_result["physical_anchor_robust_b"] = float(raw_peak_results["robust"].peak_b)
        print(
            "盲提峰 annular basin: "
            f"raw-start={raw_basin_result.annular_start_b:.6f} ({raw_basin_result.annular_start_method}), "
            f"raw-basin=[{raw_basin_result.left_base_b:.6f}, {raw_basin_result.right_base_b:.6f}], "
            f"raw-anchor={raw_peak_results['mean'].peak_b:.6f}; "
            f"blur-support=[{blur_basin_result.support_left_b:.6f}, {blur_basin_result.support_right_b:.6f}]"
        )
        print(
            "盲提峰峰位: "
            f"blur-mean={blur_peak_results['mean'].peak_b:.6f}, "
            f"blur-median={blur_peak_results['median'].peak_b:.6f}, "
            f"blur-robust={blur_peak_results['robust'].peak_b:.6f}, "
            f"blur-spread={ring_result['peak_spread']:.6e}"
        )
    else:
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
        save_payload = {
            "phi": ring_result["phi"].astype(np.float64),
            "b_ring": ring_result["b_ring"].astype(np.float32),
            "I_ring": ring_result["I_ring"].astype(np.float32),
            "x_ring": ring_result["x_ring"].astype(np.float32),
            "y_ring": ring_result["y_ring"].astype(np.float32),
            "diameter_mean": np.float64(ring_result["diameter_mean"]),
            "diameter_std": np.float64(ring_result["diameter_std"]),
            "diameter_min": np.float64(ring_result["diameter_min"]),
            "diameter_max": np.float64(ring_result["diameter_max"]),
            "diameter_x": np.float64(ring_result["diameter_x"]),
            "diameter_y": np.float64(ring_result["diameter_y"]),
            "search_b_min": np.float64(ring_result["search_b_min"]),
            "search_b_max": np.float64(ring_result["search_b_max"]),
            "critical_b": np.float64(ring_result["critical_b"]),
            "selector_sigma_b": np.float64(ring_result["selector_sigma_b"]),
            "selector_tag": np.array(ring_result["selector_tag"]),
        }
        if selector_tag == BLIND_SELECTOR_TAG:
            save_payload.update(
                annular_start_b=np.float64(ring_result["annular_start_b"]),
                annular_start_method=np.array(ring_result["annular_start_method"]),
                left_base_b=np.float64(ring_result["left_base_b"]),
                right_base_b=np.float64(ring_result["right_base_b"]),
                support_left_b=np.float64(ring_result["support_left_b"]),
                support_right_b=np.float64(ring_result["support_right_b"]),
                support_width=np.float64(ring_result["support_width"]),
                peak_b_mean=np.float64(ring_result["peak_b_mean"]),
                peak_b_median=np.float64(ring_result["peak_b_median"]),
                peak_b_robust=np.float64(ring_result["peak_b_robust"]),
                peak_spread=np.float64(ring_result["peak_spread"]),
                physical_anchor_b=np.float64(ring_result["physical_anchor_b"]),
                physical_anchor_median_b=np.float64(ring_result["physical_anchor_median_b"]),
                physical_anchor_robust_b=np.float64(ring_result["physical_anchor_robust_b"]),
                anchor_b=np.float64(ring_result["anchor_b"]),
                interior_peak_count=np.int64(ring_result["interior_peak_count"]),
                anchor_fallback_count=np.int64(ring_result["anchor_fallback_count"]),
                interior_peak_fraction=np.float64(ring_result["interior_peak_fraction"]),
                anchor_fallback_fraction=np.float64(ring_result["anchor_fallback_fraction"]),
            )
        np.savez_compressed(ring_path, **save_payload)
        print(f"亮环结果已成功保存至：{ring_path}")
    else:
        print(f"跳过保存亮环结果，因为文件已存在: {ring_path}")

    if selector_tag == BLIND_SELECTOR_TAG:
        print(
            "亮环盲提取统计: "
            f"mean={ring_result['diameter_mean']:.6f}, "
            f"std={ring_result['diameter_std']:.6f}, "
            f"x={ring_result['diameter_x']:.6f}, "
            f"y={ring_result['diameter_y']:.6f}"
        )
        print(
            "盲提取诊断: "
            f"anchor_b={ring_result['anchor_b']:.6f}, "
            f"interior_peak_fraction={ring_result['interior_peak_fraction']:.6f}, "
            f"anchor_fallback_fraction={ring_result['anchor_fallback_fraction']:.6f}"
        )
    else:
        print(
            f"亮环提取器: {DEFAULT_SELECTOR_TAG}, critical_b={ring_result['critical_b']:.6f}, sigma_b={ring_result['selector_sigma_b']:.6f}"
        )
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
