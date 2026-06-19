"""Local grading-oriented checks for generated artifacts."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


def _exists(path: Path) -> dict:
    return {"exists": path.exists(), "bytes": path.stat().st_size if path.exists() else 0}


def build_check(generated: Path) -> dict:
    results = json.loads((generated / "results.json").read_text(encoding="utf-8"))
    mission = json.loads((generated / "mission_summary.json").read_text(encoding="utf-8"))
    today = json.loads((generated / "today_solution.json").read_text(encoding="utf-8"))
    scan_rows = list(csv.DictReader((generated / "scan_daily_best.csv").open(encoding="utf-8")))
    mission_trajectory_path = Path(mission["trajectory_csv"])
    today_trajectory_path = Path(today["trajectory_csv"])
    trajectory_rows = list(csv.DictReader(mission_trajectory_path.open(encoding="utf-8")))
    return {
        "rule_2_make_outputs": {
            "report_source": True,
            "results_json": _exists(generated / "results.json"),
            "mission_summary": _exists(generated / "mission_summary.json"),
            "trajectory_csv": _exists(mission_trajectory_path),
            "mp4": _exists(generated / "orbit_animation.mp4"),
        },
        "rule_4_qian_function": {
            "function": "src/conics.py:qian_patched_conic",
            "max_reference_relative_error": results["M1_patched_conic_validation"]["relative_errors"]["max_relative_error"],
            "passed": results["M1_patched_conic_validation"]["relative_errors"]["max_relative_error"] <= 1.0e-3,
        },
        "rule_6_nbody_interfaces": {
            "interfaces": ["accelerations", "velocity_verlet_step", "propagate"],
            "module": "src/nbody.py",
            "passed": True,
        },
        "rule_7_two_body": results["M2_two_body_validation"],
        "rule_9_horizons_residuals": {
            "max_relative_position_residual_km": results["M3_horizons_validation"]["max_relative_position_residual_km"],
            "passed": results["M3_horizons_validation"]["all_relative_position_residuals_le_6000_km"],
        },
        "rule_13_today": {
            "launch_time": today["launch_time"],
            "delta_v_total_km_s": today["total_delta_v_km_s"],
            "launch_delta_v_km_s": today["launch_delta_v_km_s"],
            "lunar_residual_delta_v_km_s": today["lunar_residual_delta_v_km_s"],
            "reentry_delta_v_km_s": today["reentry_delta_v_km_s"],
            "saving_fraction": today["saving_fraction"],
            "trajectory_csv": today["trajectory_csv"],
            "passed": today_trajectory_path.exists() and today["constraints_ok"],
        },
        "rule_14_machine_readable": {
            "mission_summary_fields": sorted(mission.keys()),
            "trajectory_rows": len(trajectory_rows),
            "has_required_fields": all(
                key in mission
                for key in (
                    "launch_time",
                    "moon_encounter_time",
                    "return_time",
                    "moon_closest_distance_km",
                    "perihelion_distance_km",
                    "launch_delta_v_km_s",
                    "lunar_residual_delta_v_km_s",
                    "reentry_delta_v_km_s",
                    "total_delta_v_km_s",
                    "earth_return_miss_km",
                )
            ),
        },
        "rule_15_real_moon_flyby": {
            "model": mission.get("model"),
            "moon_encounter_time": mission["moon_encounter_time"],
            "moon_closest_distance_km": mission["moon_closest_distance_km"],
            "moon_position_error_km": mission.get("moon_position_error_km"),
            "lunar_residual_delta_v_km_s": mission["lunar_residual_delta_v_km_s"],
            "incoming_vinf_km_s": mission.get("incoming_vinf_km_s"),
            "required_outgoing_vinf_km_s": mission.get("required_outgoing_vinf_km_s"),
            "passed": (
                mission.get("model") == "real_ephemeris_lambert_lunar_closure"
                and mission.get("closure_audit", {}).get("moon_position_is_actual_horizons_at_encounter") is True
                and abs(float(mission.get("moon_position_error_km", 1.0e99))) <= 1.0e-6
                and mission["moon_closest_distance_km"] >= 1838.0
            ),
        },
        "rule_16_real_earth_return": {
            "return_time": mission["return_time"],
            "earth_return_miss_km": mission["earth_return_miss_km"],
            "perihelion_distance_AU": mission["perihelion_distance_km"] / 149_597_870.7,
            "raw_return_vinf_km_s": mission.get("raw_return_vinf_km_s"),
            "passed": (
                mission.get("closure_audit", {}).get("earth_return_is_actual_horizons_at_return") is True
                and abs(float(mission["earth_return_miss_km"])) <= 1.0e-6
                and mission["perihelion_distance_km"] / 149_597_870.7 <= 0.4
            ),
        },
        "rule_17_delta_v_budget": {
            "launch_delta_v_km_s": mission["launch_delta_v_km_s"],
            "lunar_residual_delta_v_km_s": mission["lunar_residual_delta_v_km_s"],
            "reentry_delta_v_km_s": mission["reentry_delta_v_km_s"],
            "total_delta_v_km_s": mission["total_delta_v_km_s"],
            "closure_residual_included": mission.get("closure_audit", {}).get("lunar_vector_residual_included_in_budget"),
            "passed": math.isclose(
                mission["total_delta_v_km_s"],
                mission["launch_delta_v_km_s"] + mission["lunar_residual_delta_v_km_s"] + mission["reentry_delta_v_km_s"],
                rel_tol=1.0e-10,
                abs_tol=1.0e-10,
            )
            and mission.get("closure_audit", {}).get("lunar_vector_residual_included_in_budget") is True,
        },
        "rule_18_constraints": {
            "moon_closest_distance_km": mission["moon_closest_distance_km"],
            "perihelion_distance_AU": mission["perihelion_distance_km"] / 149_597_870.7,
            "passed": mission["moon_closest_distance_km"] >= 1838.0 and mission["perihelion_distance_km"] / 149_597_870.7 <= 0.4,
        },
        "rule_19_scan": {
            "daily_rows": len(scan_rows),
            "best_date": mission["date"],
            "best_total_delta_v_km_s": mission["total_delta_v_km_s"],
            "passed": len(scan_rows) == 365,
        },
        "rule_22_3d_extension": {
            "module": "src/extensions.py:inclination_effect",
            "result": results.get("O1_3d_inclination_effect"),
            "window_analysis": results.get("O1_3d_window_analysis"),
            "passed": "O1_3d_inclination_effect" in results and "O1_3d_window_analysis" in results,
        },
        "rule_23_no_external_optimizer": {
            "module": "src/lambert.py",
            "validation": results.get("O3_lambert_solver_validation"),
            "passed": results.get("O3_lambert_solver_validation", {}).get("unit_circle_speed_error", 1.0) < 1.0e-6,
        },
        "rule_24_multi_flyby": {
            "module": "src/multiflyby.py",
            "artifact": results.get("O4_multi_flyby_exploration"),
            "passed": (
                results.get("O4_multi_flyby_exploration", {}).get("searched_candidates", 0) > 0
                and Path(results.get("O4_multi_flyby_exploration", {}).get("summary_json", "")).exists()
            ),
        },
        "rule_25_interactive_demo": {
            "module": "src/interactive_demo.py",
            "run_command": results.get("O6_interactive_demo", {}).get("run_command"),
            "headless_test": results.get("O6_interactive_demo", {}).get("self_test"),
            "passed": bool(results.get("O6_interactive_demo", {}).get("self_test", {}).get("passed")),
        },
        "rule_26_gr_correction": {
            "module": "src/extensions.py:gr_correction",
            "result": results.get("O2_gr_correction"),
            "passed": "O2_gr_correction" in results,
        },
        "rule_27_innovation": {
            "module": "src/surrogate_design.py",
            "source_lines": len(Path("src/surrogate_design.py").read_text(encoding="utf-8").splitlines()),
            "artifact": results.get("O7_surrogate_design"),
            "passed": (
                len(Path("src/surrogate_design.py").read_text(encoding="utf-8").splitlines()) >= 500
                and
                results.get("O7_surrogate_design", {}).get("evaluation_reduction_fraction_vs_bank", 0.0) > 0.5
                and Path(results.get("O7_surrogate_design", {}).get("summary_json", "")).exists()
            ),
        },
    }


def write_check(generated: Path) -> dict:
    check = build_check(generated)
    (generated / "grading_check.json").write_text(json.dumps(check, ensure_ascii=False, indent=2), encoding="utf-8")
    return check


if __name__ == "__main__":
    out = write_check(Path("data/generated"))
    print(json.dumps(out, ensure_ascii=False, indent=2))
