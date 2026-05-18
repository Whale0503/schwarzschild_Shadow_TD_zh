import glob
import json
import os
import traceback

import schwarzschild_three_scan_runner as base


R_IN_CONTROL = 4.5
TAG = "rin4p5"

base.LOG_PATH = os.path.join(base.RESULTS_DIR, f"schwarzschild_three_scan_run_{TAG}.log")
base.RESULTS_CSV = os.path.join(base.RESULTS_DIR, f"schwarzschild_ring_scan_results_{TAG}.csv")
base.TEMPLATE_CSV = os.path.join(base.RESULTS_DIR, f"schwarzschild_ring_scan_record_template_{TAG}.csv")
base.SUMMARY_MD = os.path.join(base.RESULTS_DIR, f"schwarzschild_ring_scan_summary_{TAG}.md")
base.COMBINED_PLOT = os.path.join(base.RESULTS_DIR, f"schwarzschild_three_ring_plots_{TAG}.png")
base.PLOT_PSI = os.path.join(base.RESULTS_DIR, f"schwarzschild_Dring_vs_psi0_{TAG}.png")
base.PLOT_THETA = os.path.join(base.RESULTS_DIR, f"schwarzschild_Dring_vs_theta0_{TAG}.png")
base.PLOT_CHI = os.path.join(base.RESULTS_DIR, f"schwarzschild_Dring_vs_absorption_{TAG}.png")


def config_for_run(base_config, run):
    config = base.config_for_run(base_config, run)
    config["r_in"] = R_IN_CONTROL
    return config


def clear_cached_artifacts(step5, config):
    prefix, map_path, polar_path, ring_path = base.get_step5_paths(step5, config)
    explicit_paths = [
        os.path.join(base.OUTPUT_DIR, prefix + ".npz"),
        os.path.join(base.OUTPUT_DIR, prefix + ".png"),
        os.path.join(base.OUTPUT_DIR, prefix + "_fliter.png"),
        map_path,
        polar_path,
        ring_path,
    ]
    pattern_paths = []
    pattern_paths.extend(glob.glob(os.path.join(base.OUTPUT_DIR, prefix + "_fliter_polar_*.npz")))
    pattern_paths.extend(glob.glob(os.path.join(base.OUTPUT_DIR, prefix + "_fliter_ring_*.npz")))

    removed = []
    seen = set()
    for path in explicit_paths + pattern_paths:
        norm_path = os.path.normcase(os.path.abspath(path))
        if norm_path in seen:
            continue
        seen.add(norm_path)
        if os.path.isfile(path):
            os.remove(path)
            removed.append(path)
    base.append_log(f"Cleared {len(removed)} cached artifacts for {prefix}\n")
    for path in removed:
        base.append_log(f"  removed {path}\n")
    return removed


def add_control_note_to_summary():
    with open(base.SUMMARY_MD, "r", encoding="utf-8") as f:
        text = f.read()
    marker = "# Schwarzschild three-scan ring-diameter summary\n\n"
    note = (
        "# Schwarzschild three-scan ring-diameter summary (`r_in = 4.5M` control)\n\n"
        f"- Control scan inner radius: `r_in = {R_IN_CONTROL:.1f}M`\n"
        "- Step 4 blur setting: `sigma_pixels = 0.50 / step_b`\n"
        "- Step 5 selector: `gaussian_bc_peakfit_v2`, extracted from the blurred map `I_blur`\n"
    )
    if text.startswith(marker):
        text = note + text[len(marker):]
    else:
        text = note + "\n" + text
    with open(base.SUMMARY_MD, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    step5 = base.load_step5_module()
    with open(base.CONFIG_PATH, "r", encoding="utf-8") as f:
        original_config_text = f.read()
    base_config = json.loads(original_config_text)
    if os.path.exists(base.LOG_PATH):
        os.remove(base.LOG_PATH)

    unique_runs = {}
    for run in base.RUNS:
        key = (run["absorption_coefficient"], run["psi0_deg"], run["theta0_deg"])
        unique_runs.setdefault(key, run)

    completed = {}
    failed = {}
    try:
        for idx, (key, representative_run) in enumerate(unique_runs.items(), start=1):
            chi, psi0, theta0 = key
            current = config_for_run(base_config, representative_run)
            base.write_config(current)
            print(
                f"\n===== r_in={R_IN_CONTROL:.1f} unique run {idx}/{len(unique_runs)}: "
                f"chi={chi}, psi0={psi0}, theta0={theta0} ====="
            )
            try:
                removed = clear_cached_artifacts(step5, current)
                print(f"Cleared {len(removed)} cached r_in={R_IN_CONTROL:.1f} artifacts for this point.")
                base.run_command(f"run_all r_in={R_IN_CONTROL:.1f} chi={chi} psi0={psi0} theta0={theta0}", [base.sys.executable, base.RUN_ALL_PATH])
                base.run_command(f"step4 r_in={R_IN_CONTROL:.1f} chi={chi} psi0={psi0} theta0={theta0}", [base.sys.executable, base.STEP4_PATH])
                base.run_command(f"step5 r_in={R_IN_CONTROL:.1f} chi={chi} psi0={psi0} theta0={theta0}", [base.sys.executable, base.STEP5_PATH])
                ring_info = base.read_ring_result(step5, current)
                completed[key] = ring_info
                print(
                    f"Recorded D_ring={ring_info['diameter_mean']:.6f}, "
                    f"D_std={ring_info['diameter_std']:.6f}, note={ring_info['note']}"
                )
            except Exception as exc:
                error_text = f"{type(exc).__name__}: {exc}"
                base.append_log("\n" + traceback.format_exc() + "\n")
                failed[key] = base.failure_record(step5, current, error_text)
                print(f"Run failed for chi={chi}, psi0={psi0}, theta0={theta0}: {error_text}")

        results = []
        for run in base.RUNS:
            key = (run["absorption_coefficient"], run["psi0_deg"], run["theta0_deg"])
            row = dict(run)
            if key in completed:
                row.update(completed[key])
            else:
                row.update(failed[key])
            results.append(row)

        base.write_results_csv(results)
        base.write_template_csv(results)
        base.make_plots(results)
        base.write_summary(results)
        add_control_note_to_summary()
        print("\nr_in=4.5 Schwarzschild control scan outputs have been written to the project directory.")
        print(f"Results CSV: {base.RESULTS_CSV}")
        print(f"Record template: {base.TEMPLATE_CSV}")
        print(f"Combined plot: {base.COMBINED_PLOT}")
        print(f"Summary: {base.SUMMARY_MD}")
    finally:
        with open(base.CONFIG_PATH, "w", encoding="utf-8") as f:
            f.write(original_config_text)
        print("Original config restored.")


if __name__ == "__main__":
    main()
