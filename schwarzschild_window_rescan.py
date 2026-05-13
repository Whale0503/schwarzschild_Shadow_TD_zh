import csv
import importlib.util
import json
import math
import os
import subprocess
import sys
from itertools import product

import numpy as np

PROJECT_DIR = r"C:/Users/a/Desktop/S&N1/schwarzschild_Shadow_TD"
CONFIG_PATH = os.path.join(PROJECT_DIR, "config", "config.json")
STEP5_PATH = os.path.join(PROJECT_DIR, "scripts", "step5_extract_ring_from_blur.py")
DESKTOP_OUTPUT = os.path.join(PROJECT_DIR, "schwarzschild_window_rescan_results.csv")
DESKTOP_SUMMARY = os.path.join(PROJECT_DIR, "schwarzschild_window_rescan_summary.md")
LOG_PATH = os.path.join(PROJECT_DIR, "schwarzschild_window_rescan.log")
D_REF = 6.0 * math.sqrt(3.0)
BOUNDARY_TOL = 0.5 * 0.005
LOCK_FRACTION_FLAG = 0.10
PEAK_THRESHOLD = 0.5

TARGET_POINTS = [
    {"label": "psi0-45", "absorption_coefficient": 2.0, "psi0_deg": 45.0, "theta0_deg": 60.0},
    {"label": "psi0-60", "absorption_coefficient": 2.0, "psi0_deg": 60.0, "theta0_deg": 60.0},
    {"label": "psi0-75", "absorption_coefficient": 2.0, "psi0_deg": 75.0, "theta0_deg": 60.0},
    {"label": "theta0-75", "absorption_coefficient": 2.0, "psi0_deg": 30.0, "theta0_deg": 75.0},
    {"label": "theta0-30", "absorption_coefficient": 2.0, "psi0_deg": 30.0, "theta0_deg": 30.0},
    {"label": "theta0-45", "absorption_coefficient": 2.0, "psi0_deg": 30.0, "theta0_deg": 45.0},
    {"label": "theta0-60", "absorption_coefficient": 2.0, "psi0_deg": 30.0, "theta0_deg": 60.0},
]

WINDOWS = [
    (4.4, 5.8),
    (4.5, 5.8),
    (4.6, 5.8),
    (4.5, 5.9),
    (4.4, 6.0),
]


def load_step5_module():
    spec = importlib.util.spec_from_file_location("step5_extract_ring_from_blur", STEP5_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def append_log(text):
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def build_output_prefix(config):
    opt_regime = config["optical_regime"].lower()
    if opt_regime == "intermediate":
        output_opt = f"{opt_regime}_{config['absorption_coefficient']:.3f}"
    else:
        output_opt = opt_regime
    return (
        f"flux_rmax={config['r_max']:.1f}_optical_{output_opt}"
        f"_psi0={config['psi0_deg']:.1f}_rin={config['r_in']:.1f}"
        f"_theta0={config['theta0_deg']:.1f}_kappaff={config['kappa_ff']:.3f}"
        f"_kappaK={config['kappa_K']:.3f}"
    )


def main():
    step5 = load_step5_module()
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        original_text = f.read()
    base_config = json.loads(original_text)
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)

    rows = []
    try:
        for point, (bmin, bmax) in product(TARGET_POINTS, WINDOWS):
            config = dict(base_config)
            config["optical_regime"] = "intermediate"
            config["absorption_coefficient"] = point["absorption_coefficient"]
            config["psi0_deg"] = point["psi0_deg"]
            config["theta0_deg"] = point["theta0_deg"]
            config["ring_b_min"] = bmin
            config["ring_b_max"] = bmax
            config["ring_peak_rel_threshold"] = PEAK_THRESHOLD

            prefix = build_output_prefix(config)
            map_path = os.path.join(PROJECT_DIR, "output", prefix + "_fliter_map.npz")
            if not os.path.exists(map_path):
                rows.append({
                    "label": point["label"],
                    "absorption_coefficient": point["absorption_coefficient"],
                    "psi0_deg": point["psi0_deg"],
                    "theta0_deg": point["theta0_deg"],
                    "ring_b_min": bmin,
                    "ring_b_max": bmax,
                    "threshold": PEAK_THRESHOLD,
                    "status": "missing_map",
                    "diameter_mean": np.nan,
                    "diameter_std": np.nan,
                    "diameter_x": np.nan,
                    "diameter_y": np.nan,
                    "lower_lock_fraction": np.nan,
                    "upper_lock_fraction": np.nan,
                    "abs_error_to_6sqrt3": np.nan,
                    "note": "missing-map",
                    "ring_path": "",
                })
                continue

            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=4)
                f.write("\n")

            cmd = [sys.executable, STEP5_PATH]
            proc = subprocess.run(
                cmd,
                cwd=PROJECT_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            append_log(f"===== {point['label']} bmin={bmin:.3f} bmax={bmax:.3f} =====\n{proc.stdout}\n")

            polar_path, ring_path = step5.build_step5_output_paths(
                os.path.join(PROJECT_DIR, "output"),
                prefix,
                float(config["dalpha"]),
                float(config["ring_b_min"]),
                config.get("ring_b_max"),
                float(config["ring_peak_rel_threshold"]),
            )

            if proc.returncode != 0 or not os.path.exists(ring_path):
                rows.append({
                    "label": point["label"],
                    "absorption_coefficient": point["absorption_coefficient"],
                    "psi0_deg": point["psi0_deg"],
                    "theta0_deg": point["theta0_deg"],
                    "ring_b_min": bmin,
                    "ring_b_max": bmax,
                    "threshold": PEAK_THRESHOLD,
                    "status": "failed",
                    "diameter_mean": np.nan,
                    "diameter_std": np.nan,
                    "diameter_x": np.nan,
                    "diameter_y": np.nan,
                    "lower_lock_fraction": np.nan,
                    "upper_lock_fraction": np.nan,
                    "abs_error_to_6sqrt3": np.nan,
                    "note": "step5-failed",
                    "ring_path": ring_path,
                })
                continue

            ring = np.load(ring_path)
            b_ring = ring["b_ring"].astype(np.float64)
            search_b_min = float(ring["search_b_min"])
            search_b_max = float(ring["search_b_max"])
            diameter_mean = float(ring["diameter_mean"])
            diameter_std = float(ring["diameter_std"])
            diameter_x = float(ring["diameter_x"])
            diameter_y = float(ring["diameter_y"])
            lower_lock_fraction = float(np.mean(np.abs(b_ring - search_b_min) <= BOUNDARY_TOL))
            upper_lock_fraction = float(np.mean(np.abs(b_ring - search_b_max) <= BOUNDARY_TOL))
            notes = []
            if lower_lock_fraction >= LOCK_FRACTION_FLAG:
                notes.append("lower-lock")
            if upper_lock_fraction >= LOCK_FRACTION_FLAG:
                notes.append("upper-lock")
            if diameter_std >= 0.30:
                notes.append("large-std")
            note = ", ".join(notes) if notes else "normal"

            rows.append({
                "label": point["label"],
                "absorption_coefficient": point["absorption_coefficient"],
                "psi0_deg": point["psi0_deg"],
                "theta0_deg": point["theta0_deg"],
                "ring_b_min": bmin,
                "ring_b_max": bmax,
                "threshold": PEAK_THRESHOLD,
                "status": "completed",
                "diameter_mean": diameter_mean,
                "diameter_std": diameter_std,
                "diameter_x": diameter_x,
                "diameter_y": diameter_y,
                "lower_lock_fraction": lower_lock_fraction,
                "upper_lock_fraction": upper_lock_fraction,
                "abs_error_to_6sqrt3": abs(diameter_mean - D_REF),
                "note": note,
                "ring_path": ring_path,
            })
    finally:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(original_text)

    fieldnames = [
        "label", "absorption_coefficient", "psi0_deg", "theta0_deg", "ring_b_min", "ring_b_max", "threshold",
        "status", "diameter_mean", "diameter_std", "diameter_x", "diameter_y", "lower_lock_fraction",
        "upper_lock_fraction", "abs_error_to_6sqrt3", "note", "ring_path",
    ]
    with open(DESKTOP_OUTPUT, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    grouped = {}
    for row in rows:
        grouped.setdefault(row["label"], []).append(row)

    with open(DESKTOP_SUMMARY, "w", encoding="utf-8") as f:
        f.write("# Schwarzschild focused Step 5 window rescan\n\n")
        f.write(f"- Target reference: `D_ref = 6sqrt(3) ≈ {D_REF:.6f}`\n")
        f.write(f"- Fixed threshold: `{PEAK_THRESHOLD}`\n")
        f.write(f"- Window candidates: {WINDOWS}\n\n")
        for label, items in grouped.items():
            items = sorted(items, key=lambda r: (r["ring_b_min"], r["ring_b_max"]))
            f.write(f"## {label}\n\n")
            f.write("| bmin | bmax | status | D_ring | D_std | abs_err | lower_lock | upper_lock | note |\n")
            f.write("|---:|---:|---|---:|---:|---:|---:|---:|---|\n")
            for row in items:
                dmean = "" if math.isnan(row["diameter_mean"]) else f"{row['diameter_mean']:.6f}"
                dstd = "" if math.isnan(row["diameter_std"]) else f"{row['diameter_std']:.6f}"
                derr = "" if math.isnan(row["abs_error_to_6sqrt3"]) else f"{row['abs_error_to_6sqrt3']:.6f}"
                ll = "" if math.isnan(row["lower_lock_fraction"]) else f"{row['lower_lock_fraction']:.3f}"
                ul = "" if math.isnan(row["upper_lock_fraction"]) else f"{row['upper_lock_fraction']:.3f}"
                f.write(f"| {row['ring_b_min']:.3f} | {row['ring_b_max']:.3f} | {row['status']} | {dmean} | {dstd} | {derr} | {ll} | {ul} | {row['note']} |\n")
            f.write("\n")


if __name__ == "__main__":
    main()
