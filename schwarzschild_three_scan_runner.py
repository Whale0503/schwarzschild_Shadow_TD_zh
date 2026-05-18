import csv
import importlib.util
import json
import math
import os
import subprocess
import sys
import traceback
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = r"C:/Users/a/Desktop/S&N1/schwarzschild_Shadow_TD"
CONFIG_PATH = os.path.join(PROJECT_DIR, "config", "config.json")
RUN_ALL_PATH = os.path.join(PROJECT_DIR, "run_all.py")
STEP4_PATH = os.path.join(PROJECT_DIR, "scripts", "step4_gaussian_filter.py")
STEP5_PATH = os.path.join(PROJECT_DIR, "scripts", "step5_extract_ring_from_blur.py")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
RESULTS_DIR = PROJECT_DIR
LOG_PATH = os.path.join(RESULTS_DIR, "schwarzschild_three_scan_run.log")
RESULTS_CSV = os.path.join(RESULTS_DIR, "schwarzschild_ring_scan_results.csv")
TEMPLATE_CSV = os.path.join(RESULTS_DIR, "schwarzschild_ring_scan_record_template.csv")
SUMMARY_MD = os.path.join(RESULTS_DIR, "schwarzschild_ring_scan_summary.md")
COMBINED_PLOT = os.path.join(RESULTS_DIR, "schwarzschild_three_ring_plots.png")
PLOT_PSI = os.path.join(RESULTS_DIR, "schwarzschild_Dring_vs_psi0.png")
PLOT_THETA = os.path.join(RESULTS_DIR, "schwarzschild_Dring_vs_theta0.png")
PLOT_CHI = os.path.join(RESULTS_DIR, "schwarzschild_Dring_vs_absorption.png")
D_REF = 6.0 * math.sqrt(3.0)
WINDOW_MIN = 4.6
WINDOW_MAX = 5.6
PEAK_THRESHOLD = 0.5
BOUNDARY_TOL = 0.5 * 0.005
LOCK_FRACTION_FLAG = 0.10

RUNS = [
    {"run_id": "B-1", "scan": "B", "scan_label": "D_ring vs psi0_deg", "x_name": "psi0_deg", "x_value": 5.0, "absorption_coefficient": 2.0, "psi0_deg": 5.0, "theta0_deg": 60.0},
    {"run_id": "B-2", "scan": "B", "scan_label": "D_ring vs psi0_deg", "x_name": "psi0_deg", "x_value": 15.0, "absorption_coefficient": 2.0, "psi0_deg": 15.0, "theta0_deg": 60.0},
    {"run_id": "B-3", "scan": "B", "scan_label": "D_ring vs psi0_deg", "x_name": "psi0_deg", "x_value": 30.0, "absorption_coefficient": 2.0, "psi0_deg": 30.0, "theta0_deg": 60.0},
    {"run_id": "B-4", "scan": "B", "scan_label": "D_ring vs psi0_deg", "x_name": "psi0_deg", "x_value": 45.0, "absorption_coefficient": 2.0, "psi0_deg": 45.0, "theta0_deg": 60.0},
    {"run_id": "B-5", "scan": "B", "scan_label": "D_ring vs psi0_deg", "x_name": "psi0_deg", "x_value": 60.0, "absorption_coefficient": 2.0, "psi0_deg": 60.0, "theta0_deg": 60.0},
    {"run_id": "B-6", "scan": "B", "scan_label": "D_ring vs psi0_deg", "x_name": "psi0_deg", "x_value": 75.0, "absorption_coefficient": 2.0, "psi0_deg": 75.0, "theta0_deg": 60.0},
    {"run_id": "C-1", "scan": "C", "scan_label": "D_ring vs theta0_deg", "x_name": "theta0_deg", "x_value": 10.0, "absorption_coefficient": 2.0, "psi0_deg": 30.0, "theta0_deg": 10.0},
    {"run_id": "C-2", "scan": "C", "scan_label": "D_ring vs theta0_deg", "x_name": "theta0_deg", "x_value": 20.0, "absorption_coefficient": 2.0, "psi0_deg": 30.0, "theta0_deg": 20.0},
    {"run_id": "C-3", "scan": "C", "scan_label": "D_ring vs theta0_deg", "x_name": "theta0_deg", "x_value": 30.0, "absorption_coefficient": 2.0, "psi0_deg": 30.0, "theta0_deg": 30.0},
    {"run_id": "C-4", "scan": "C", "scan_label": "D_ring vs theta0_deg", "x_name": "theta0_deg", "x_value": 45.0, "absorption_coefficient": 2.0, "psi0_deg": 30.0, "theta0_deg": 45.0},
    {"run_id": "C-5", "scan": "C", "scan_label": "D_ring vs theta0_deg", "x_name": "theta0_deg", "x_value": 60.0, "absorption_coefficient": 2.0, "psi0_deg": 30.0, "theta0_deg": 60.0},
    {"run_id": "C-6", "scan": "C", "scan_label": "D_ring vs theta0_deg", "x_name": "theta0_deg", "x_value": 75.0, "absorption_coefficient": 2.0, "psi0_deg": 30.0, "theta0_deg": 75.0},
    {"run_id": "A-1", "scan": "A", "scan_label": "D_ring vs absorption_coefficient", "x_name": "absorption_coefficient", "x_value": 0.5, "absorption_coefficient": 0.5, "psi0_deg": 30.0, "theta0_deg": 60.0},
    {"run_id": "A-2", "scan": "A", "scan_label": "D_ring vs absorption_coefficient", "x_name": "absorption_coefficient", "x_value": 1.0, "absorption_coefficient": 1.0, "psi0_deg": 30.0, "theta0_deg": 60.0},
    {"run_id": "A-3", "scan": "A", "scan_label": "D_ring vs absorption_coefficient", "x_name": "absorption_coefficient", "x_value": 2.0, "absorption_coefficient": 2.0, "psi0_deg": 30.0, "theta0_deg": 60.0},
    {"run_id": "A-4", "scan": "A", "scan_label": "D_ring vs absorption_coefficient", "x_name": "absorption_coefficient", "x_value": 5.0, "absorption_coefficient": 5.0, "psi0_deg": 30.0, "theta0_deg": 60.0},
    {"run_id": "A-5", "scan": "A", "scan_label": "D_ring vs absorption_coefficient", "x_name": "absorption_coefficient", "x_value": 10.0, "absorption_coefficient": 10.0, "psi0_deg": 30.0, "theta0_deg": 60.0},
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


def run_command(label, command):
    print(f"\n>>> {label}")
    append_log(f"\n===== {label} =====\n")
    proc = subprocess.run(
        command,
        cwd=PROJECT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    append_log(proc.stdout)
    tail_lines = [line for line in proc.stdout.strip().splitlines() if line.strip()]
    console_encoding = sys.stdout.encoding or "utf-8"
    for line in tail_lines[-8:]:
        safe_line = line.encode(console_encoding, errors="replace").decode(console_encoding, errors="replace")
        print(safe_line)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed for {label} with exit code {proc.returncode}")


def config_for_run(base_config, run):
    config = dict(base_config)
    config["optical_regime"] = "intermediate"
    config["absorption_coefficient"] = float(run["absorption_coefficient"])
    config["psi0_deg"] = float(run["psi0_deg"])
    config["theta0_deg"] = float(run["theta0_deg"])
    config["ring_b_min"] = WINDOW_MIN
    config["ring_b_max"] = WINDOW_MAX
    config["ring_peak_rel_threshold"] = PEAK_THRESHOLD
    config["ring_selector"] = "gaussian_bc_peakfit_v2"
    return config


def write_config(config):
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
        f.write("\n")


def output_prefix(config):
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


def get_step5_paths(step5, config):
    prefix = output_prefix(config)
    map_path = os.path.join(OUTPUT_DIR, prefix + "_fliter_map.npz")
    polar_path, ring_path = step5.build_step5_output_paths(
        OUTPUT_DIR,
        prefix,
        float(config["dalpha"]),
        float(config.get("ring_b_min", 0.0)),
        config.get("ring_b_max"),
        float(config.get("ring_peak_rel_threshold", 0.5)),
    )
    return prefix, map_path, polar_path, ring_path


def read_ring_result(step5, config):
    prefix, map_path, polar_path, ring_path = get_step5_paths(step5, config)
    if not os.path.exists(ring_path):
        raise FileNotFoundError(f"Ring output not found: {ring_path}")
    ring_data = np.load(ring_path)
    b_ring = ring_data["b_ring"].astype(np.float64)
    search_b_min = float(ring_data["search_b_min"])
    search_b_max = float(ring_data["search_b_max"])
    diameter_mean = float(ring_data["diameter_mean"])
    diameter_std = float(ring_data["diameter_std"])
    diameter_x = float(ring_data["diameter_x"])
    diameter_y = float(ring_data["diameter_y"])
    diameter_min = float(ring_data["diameter_min"])
    diameter_max = float(ring_data["diameter_max"])
    lower_lock_fraction = float(np.mean(np.abs(b_ring - search_b_min) <= BOUNDARY_TOL))
    upper_lock_fraction = float(np.mean(np.abs(b_ring - search_b_max) <= BOUNDARY_TOL))
    notes = []
    if lower_lock_fraction >= LOCK_FRACTION_FLAG:
        notes.append("lower-boundary-lock")
    if upper_lock_fraction >= LOCK_FRACTION_FLAG:
        notes.append("upper-boundary-lock")
    if diameter_std >= 0.30:
        notes.append("large-std")
    note = ", ".join(notes) if notes else "normal"
    return {
        "status": "completed",
        "error": "",
        "prefix": prefix,
        "map_path": map_path,
        "polar_path": polar_path,
        "ring_path": ring_path,
        "search_b_min": search_b_min,
        "search_b_max": search_b_max,
        "diameter_mean": diameter_mean,
        "diameter_std": diameter_std,
        "diameter_x": diameter_x,
        "diameter_y": diameter_y,
        "diameter_min": diameter_min,
        "diameter_max": diameter_max,
        "b_mean": 0.5 * diameter_mean,
        "b_std": 0.5 * diameter_std,
        "abs_error_to_6sqrt3": abs(diameter_mean - D_REF),
        "lower_lock_fraction": lower_lock_fraction,
        "upper_lock_fraction": upper_lock_fraction,
        "note": note,
    }


def failure_record(step5, config, error_text):
    prefix, map_path, polar_path, ring_path = get_step5_paths(step5, config)
    return {
        "status": "failed",
        "error": error_text,
        "prefix": prefix,
        "map_path": map_path,
        "polar_path": polar_path,
        "ring_path": ring_path,
        "search_b_min": float(config.get("ring_b_min", WINDOW_MIN)),
        "search_b_max": float(config.get("ring_b_max", WINDOW_MAX)),
        "diameter_mean": np.nan,
        "diameter_std": np.nan,
        "diameter_x": np.nan,
        "diameter_y": np.nan,
        "diameter_min": np.nan,
        "diameter_max": np.nan,
        "b_mean": np.nan,
        "b_std": np.nan,
        "abs_error_to_6sqrt3": np.nan,
        "lower_lock_fraction": np.nan,
        "upper_lock_fraction": np.nan,
        "note": "failed",
    }


def write_results_csv(results):
    fieldnames = [
        "run_id", "scan", "scan_label", "x_name", "x_value", "absorption_coefficient", "psi0_deg", "theta0_deg",
        "status", "error", "diameter_mean", "diameter_std", "diameter_x", "diameter_y", "diameter_min", "diameter_max",
        "b_mean", "b_std", "abs_error_to_6sqrt3", "search_b_min", "search_b_max", "lower_lock_fraction",
        "upper_lock_fraction", "note", "prefix", "map_path", "polar_path", "ring_path",
    ]
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_template_csv(results):
    fieldnames = [
        "run_id", "scan", "scan_label", "x_name", "x_value", "absorption_coefficient", "psi0_deg", "theta0_deg",
        "status", "diameter_mean", "diameter_std", "diameter_x", "diameter_y", "note", "error"
    ]
    with open(TEMPLATE_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def format_tick_label(value):
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:g}"



def plot_scan(ax, rows, x_key, title, xlabel, logx=False, fixed_text=None, xticks=None):
    if not rows:
        ax.text(0.5, 0.5, "No completed data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$D_{\rm ring} - 2b_c$")
        ax.grid(alpha=0.3)
        return
    rows = sorted(rows, key=lambda r: r[x_key])
    x = [r[x_key] for r in rows]
    y = [r["diameter_mean"] - D_REF for r in rows]
    ax.plot(x, y, marker="o", lw=1.5)
    ax.axhline(0.0, color="k", ls="--", lw=1, label=r"$D_{\rm ring} = 2b_c = 6\sqrt{3}\,M$")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$D_{\rm ring} - 2b_c$")
    ax.grid(alpha=0.3)
    if logx:
        ax.set_xscale("log")
    if xticks is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels([format_tick_label(v) for v in xticks])
    if fixed_text:
        ax.text(
            0.02,
            0.98,
            fixed_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="none", boxstyle="round,pad=0.25"),
        )
    ax.legend(fontsize=9)


def make_plots(results):
    completed_rows = [row for row in results if row["status"] == "completed"]
    by_scan = defaultdict(list)
    for row in completed_rows:
        by_scan[row["scan"]].append(row)

    fixed_text_map = {
        "B": r"fixed: $\chi=2$, $\theta_0=60^\circ$",
        "C": r"fixed: $\chi=2$, $\psi_0=30^\circ$",
        "A": r"fixed: $\psi_0=30^\circ$, $\theta_0=60^\circ$",
    }
    chi_xticks = sorted(row["absorption_coefficient"] for row in by_scan["A"])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plot_scan(
        axes[0],
        by_scan["B"],
        "psi0_deg",
        "Schwarzschild: D_ring vs psi0_deg",
        "psi0_deg",
        fixed_text=fixed_text_map["B"],
    )
    plot_scan(
        axes[1],
        by_scan["C"],
        "theta0_deg",
        "Schwarzschild: D_ring vs theta0_deg",
        "theta0_deg",
        fixed_text=fixed_text_map["C"],
    )
    plot_scan(
        axes[2],
        by_scan["A"],
        "absorption_coefficient",
        "Schwarzschild: D_ring vs absorption",
        "absorption_coefficient",
        logx=True,
        fixed_text=fixed_text_map["A"],
        xticks=chi_xticks,
    )
    fig.tight_layout()
    fig.savefig(COMBINED_PLOT, dpi=220)
    plt.close(fig)

    for scan, path, title, x_key, xlabel, logx in [
        ("B", PLOT_PSI, "Schwarzschild: D_ring vs psi0_deg", "psi0_deg", "psi0_deg", False),
        ("C", PLOT_THETA, "Schwarzschild: D_ring vs theta0_deg", "theta0_deg", "theta0_deg", False),
        ("A", PLOT_CHI, "Schwarzschild: D_ring vs absorption coefficient", "absorption_coefficient", "absorption_coefficient", True),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        plot_scan(
            ax,
            by_scan[scan],
            x_key,
            title,
            xlabel,
            logx=logx,
            fixed_text=fixed_text_map[scan],
            xticks=chi_xticks if scan == "A" else None,
        )
        fig.tight_layout()
        fig.savefig(path, dpi=220)
        plt.close(fig)


def write_summary(results):
    by_scan = defaultdict(list)
    for row in results:
        by_scan[row["scan"]].append(row)

    with open(SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write("# Schwarzschild three-scan ring-diameter summary\n\n")
        f.write(f"- Target ring scale: `b_c = 3\\sqrt{{3}} ≈ {0.5 * D_REF:.6f}` and `D_ref = 6\\sqrt{{3}} ≈ {D_REF:.6f}`\n")
        f.write(f"- Unified Step 5 window: `ring_b_min={WINDOW_MIN}`, `ring_b_max={WINDOW_MAX}`, `ring_peak_rel_threshold={PEAK_THRESHOLD}`\n")
        f.write(f"- Total run points: {len(results)}\n")
        f.write(f"- Completed points: {sum(row['status'] == 'completed' for row in results)}\n")
        f.write(f"- Failed points: {sum(row['status'] == 'failed' for row in results)}\n\n")
        for scan in ["B", "C", "A"]:
            rows = sorted(by_scan[scan], key=lambda r: r["x_value"])
            if not rows:
                continue
            f.write(f"## Scan {scan}: {rows[0]['scan_label']}\n\n")
            f.write("| run_id | x_value | chi | psi0 | theta0 | status | D_ring | D_std | D_x | D_y | note |\n")
            f.write("|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|\n")
            for row in rows:
                dmean = "" if math.isnan(row["diameter_mean"]) else f"{row['diameter_mean']:.6f}"
                dstd = "" if math.isnan(row["diameter_std"]) else f"{row['diameter_std']:.6f}"
                dx = "" if math.isnan(row["diameter_x"]) else f"{row['diameter_x']:.6f}"
                dy = "" if math.isnan(row["diameter_y"]) else f"{row['diameter_y']:.6f}"
                f.write(
                    f"| {row['run_id']} | {row['x_value']:.3f} | {row['absorption_coefficient']:.3f} | {row['psi0_deg']:.1f} | {row['theta0_deg']:.1f} | {row['status']} | {dmean} | {dstd} | {dx} | {dy} | {row['note']} |\n"
                )
            f.write("\n")
        failed_rows = [row for row in results if row["status"] == "failed"]
        if failed_rows:
            f.write("## Failed runs\n\n")
            for row in failed_rows:
                f.write(f"- `{row['run_id']}`: {row['error']}\n")
            f.write("\n")
        f.write("## Files\n\n")
        f.write(f"- Results CSV: `{RESULTS_CSV}`\n")
        f.write(f"- Record template CSV: `{TEMPLATE_CSV}`\n")
        f.write(f"- Combined plot: `{COMBINED_PLOT}`\n")
        f.write(f"- psi0 plot: `{PLOT_PSI}`\n")
        f.write(f"- theta0 plot: `{PLOT_THETA}`\n")
        f.write(f"- absorption plot: `{PLOT_CHI}`\n")
        f.write(f"- Run log: `{LOG_PATH}`\n")


def main():
    step5 = load_step5_module()
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        original_config_text = f.read()
    base_config = json.loads(original_config_text)
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)

    unique_runs = {}
    for run in RUNS:
        key = (run["absorption_coefficient"], run["psi0_deg"], run["theta0_deg"])
        unique_runs.setdefault(key, run)

    completed = {}
    failed = {}
    try:
        for idx, (key, representative_run) in enumerate(unique_runs.items(), start=1):
            chi, psi0, theta0 = key
            current = config_for_run(base_config, representative_run)
            write_config(current)
            print(f"\n===== Unique run {idx}/{len(unique_runs)}: chi={chi}, psi0={psi0}, theta0={theta0} =====")
            try:
                run_command(f"run_all chi={chi} psi0={psi0} theta0={theta0}", [sys.executable, RUN_ALL_PATH])
                run_command(f"step4 chi={chi} psi0={psi0} theta0={theta0}", [sys.executable, STEP4_PATH])
                run_command(f"step5 chi={chi} psi0={psi0} theta0={theta0}", [sys.executable, STEP5_PATH])
                ring_info = read_ring_result(step5, current)
                completed[key] = ring_info
                print(
                    f"Recorded D_ring={ring_info['diameter_mean']:.6f}, D_std={ring_info['diameter_std']:.6f}, note={ring_info['note']}"
                )
            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"
                append_log("\n" + traceback.format_exc() + "\n")
                failed[key] = failure_record(step5, current, error_text)
                print(f"Run failed for chi={chi}, psi0={psi0}, theta0={theta0}: {error_text}")

        results = []
        for run in RUNS:
            key = (run["absorption_coefficient"], run["psi0_deg"], run["theta0_deg"])
            row = dict(run)
            if key in completed:
                row.update(completed[key])
            else:
                row.update(failed[key])
            results.append(row)

        write_results_csv(results)
        write_template_csv(results)
        make_plots(results)
        write_summary(results)
        print("\nAll Schwarzschild scan outputs have been written to the project directory.")
        print(f"Results CSV: {RESULTS_CSV}")
        print(f"Record template: {TEMPLATE_CSV}")
        print(f"Combined plot: {COMBINED_PLOT}")
        print(f"Summary: {SUMMARY_MD}")
    finally:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(original_config_text)
        print("Original config restored.")


if __name__ == "__main__":
    main()
