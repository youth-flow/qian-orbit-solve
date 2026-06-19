"""Generate all numerical outputs, figures and the project animation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from constants import BODY_MU, BODIES
from conics import qian_reference_error
from ephemeris import states_from_cache, states_from_extended_cache
from extensions import gr_correction, inclination_effect, inclination_window_analysis
from interactive_demo import self_test as interactive_demo_self_test
from lambert import validation_case as lambert_validation_case
from mission import direct_no_moon_baseline, refine_candidate, scan_year, sensitivity
from multiflyby import explore_multiflyby, write_multiflyby_artifacts
from nbody import mechanical_energy, propagate, validate_horizons, validate_two_body
from outputs import write_mission_artifacts
from plots import make_animation, plot_contour, plot_energy, plot_orbit, plot_residuals, plot_scan, plot_sensitivity
from real_closure import (
    real_no_moon_baseline,
    real_sensitivity,
    refine_real_candidate,
    scan_real_year,
    write_real_artifacts,
)
from self_check import write_check
from surrogate_design import run_surrogate_assisted_design, write_surrogate_artifacts
from swingby import numerical_turn_check


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def energy_drift_for_step(positions: np.ndarray, velocities: np.ndarray, dt: float) -> float:
    mus = np.array([BODY_MU[b] for b in BODIES], dtype=float)
    # Convert gravitational parameters to masses for energy accounting.
    g = 6.67430e-20
    masses = mus / g
    steps = int(round(365 * 86_400.0 / dt))
    rr, vv = propagate(positions[0], velocities[0], mus, dt, steps, sample_every=max(1, steps // 20))
    energies = np.array([mechanical_energy(rr[i], vv[i], mus, masses) for i in range(len(rr))])
    return float(np.max(np.abs((energies - energies[0]) / energies[0])))


def energy_series(positions: np.ndarray, velocities: np.ndarray, dt: float = 3600.0) -> dict[str, list[float]]:
    mus = np.array([BODY_MU[b] for b in BODIES], dtype=float)
    g = 6.67430e-20
    masses = mus / g
    sample_every = int(round(86_400.0 / dt))
    steps = 365 * sample_every
    rr, vv = propagate(positions[0], velocities[0], mus, dt, steps, sample_every=sample_every)
    energies = np.array([mechanical_energy(rr[i], vv[i], mus, masses) for i in range(len(rr))])
    drift = (energies - energies[0]) / energies[0]
    return {
        "dt_s": dt,
        "days": [float(i) for i in range(len(drift))],
        "relative_drift": [float(x) for x in drift],
        "max_abs_relative_drift": float(np.max(np.abs(drift))),
    }


def real_contour_from_candidates(daily: list, candidates: list) -> tuple[np.ndarray, list[float]]:
    rp_grid = np.linspace(0.05, 0.4, 36)
    contour = np.full((365, len(rp_grid)), np.nan, dtype=float)
    for cand in candidates:
        if not cand.constraints_ok:
            continue
        day = int(cand.day_index)
        irp = int(np.argmin(np.abs(rp_grid - cand.rp_AU)))
        current = contour[day, irp]
        if not np.isfinite(current) or cand.total_delta_v_km_s < current:
            contour[day, irp] = cand.total_delta_v_km_s
    for day, best in enumerate(daily):
        finite = np.isfinite(contour[day])
        if not np.any(finite):
            contour[day, :] = best.total_delta_v_km_s
            continue
        finite_idx = np.flatnonzero(finite)
        for irp in range(len(rp_grid)):
            if not np.isfinite(contour[day, irp]):
                nearest = finite_idx[int(np.argmin(np.abs(finite_idx - irp)))]
                contour[day, irp] = contour[day, nearest]
    return contour, [float(x) for x in rp_grid]


def main() -> None:
    generated = ROOT / "data" / "generated"
    generated.mkdir(parents=True, exist_ok=True)
    dates, positions, velocities, names = states_from_cache()
    _, real_positions, real_velocities, _ = states_from_extended_cache()

    m1 = qian_reference_error()
    m2 = validate_two_body()
    m3 = validate_horizons(positions, velocities, dt=900.0)
    m4 = numerical_turn_check(vinf_km_s=1.2, periapsis_km=5_000.0)

    legacy_daily, legacy_contour, legacy_rp_grid_AU, _ = scan_year(positions, velocities)
    legacy_coarse_best = min(legacy_daily, key=lambda c: c.total_delta_v_km_s)
    legacy_best = refine_candidate(legacy_coarse_best, positions, velocities)
    legacy_baseline = direct_no_moon_baseline(legacy_best, positions, velocities)

    real_daily, real_candidates, real_transfer_grid, real_return_grid = scan_real_year(real_positions, real_velocities)
    real_coarse_best = min([c for c in real_daily if c.constraints_ok], key=lambda c: c.total_delta_v_km_s)
    best = refine_real_candidate(real_coarse_best, real_positions, real_velocities)
    real_daily[best.day_index] = best
    baseline = real_no_moon_baseline(best, real_positions, real_velocities)
    contour, rp_grid_AU = real_contour_from_candidates(real_daily, real_candidates)
    sens = real_sensitivity(best, real_positions, real_velocities)
    sens["step"] = [
        {"dt_s": 3600.0, "energy_drift": energy_drift_for_step(positions, velocities, 3600.0)},
        {"dt_s": 1800.0, "energy_drift": energy_drift_for_step(positions, velocities, 1800.0)},
        {"dt_s": 900.0, "energy_drift": energy_drift_for_step(positions, velocities, 900.0)},
    ]
    energy = energy_series(positions, velocities, dt=3600.0)

    daily_dicts = [c.to_dict() for c in real_daily]
    best_dict = best.to_dict()
    mission_summary = write_real_artifacts(best, real_positions, real_velocities, generated, prefix="mission")
    today = refine_real_candidate(real_daily[169], real_positions, real_velocities, vary_day=False)
    today_summary = write_real_artifacts(today, real_positions, real_velocities, generated, prefix="today")
    write_json(generated / "today_solution.json", today_summary)
    audit_summaries = [
        write_real_artifacts(refine_real_candidate(real_daily[168], real_positions, real_velocities, vary_day=False), real_positions, real_velocities, generated, prefix="audit_day168"),
        write_real_artifacts(refine_real_candidate(real_daily[201], real_positions, real_velocities, vary_day=False), real_positions, real_velocities, generated, prefix="audit_day201"),
    ]
    inclination_scan = inclination_window_analysis(daily_dicts, positions)
    multi_flyby_summary = explore_multiflyby(best, real_positions, real_velocities)
    multi_flyby_artifacts = write_multiflyby_artifacts(multi_flyby_summary, generated)
    interactive_check = interactive_demo_self_test()
    surrogate_summary = run_surrogate_assisted_design(positions, velocities)
    surrogate_artifacts = write_surrogate_artifacts(surrogate_summary, generated)
    results = {
        "M1_patched_conic_validation": m1,
        "M2_two_body_validation": m2,
        "M3_horizons_validation": {k: v for k, v in m3.items() if not k.startswith("daily_")},
        "M4_swingby_validation": m4,
        "M5_single_point_solution": {
            "best_candidate": best_dict,
            "direct_no_moon_baseline": baseline,
            "legacy_analytic_candidate": legacy_best.to_dict(),
            "legacy_direct_no_moon_baseline": legacy_baseline,
        },
        "M6_scan_summary": {
            "best_launch_date": best.date,
            "best_day_index": best.day_index,
            "minimum_total_delta_v_km_s": best.total_delta_v_km_s,
            "daily_best_count": len(daily_dicts),
            "scan_model": "real_ephemeris_lambert_lunar_closure",
            "transfer_to_moon_grid_days": real_transfer_grid,
            "return_grid_days": real_return_grid,
        },
        "rule13_today_solution": today_summary,
        "mission_machine_summary": mission_summary,
        "scan_audit_summaries": audit_summaries,
        "M7_sensitivity": sens,
        "O1_3d_inclination_effect": inclination_effect(mission_summary, positions, velocities),
        "O1_3d_window_analysis": inclination_scan,
        "O2_gr_correction": gr_correction(mission_summary),
        "O3_lambert_solver_validation": lambert_validation_case(),
        "O4_multi_flyby_exploration": multi_flyby_artifacts,
        "O6_interactive_demo": {
            "script": "src/interactive_demo.py",
            "run_command": "python src/interactive_demo.py",
            "headless_test_command": "python src/interactive_demo.py --self-test",
            "gui_smoke_test_command": "python src/interactive_demo.py --smoke-gui",
            "self_test": interactive_check,
        },
        "O7_surrogate_design": surrogate_artifacts,
        "energy_conservation_series": {
            "dt_s": energy["dt_s"],
            "max_abs_relative_drift": energy["max_abs_relative_drift"],
        },
    }
    write_json(generated / "results.json", results)
    write_json(generated / "m3_daily_residuals.json", m3)
    write_json(generated / "scan_contour.json", {"rp_grid_AU": rp_grid_AU, "contour": contour.tolist(), "model": "real ephemeris closure candidates binned by actual perihelion"})
    write_json(
        generated / "legacy_analytic_scan.json",
        {
            "best": legacy_best.to_dict(),
            "baseline": legacy_baseline,
            "rp_grid_AU": legacy_rp_grid_AU,
            "contour": legacy_contour.tolist(),
            "note": "Retained for comparison only; main mission_summary uses real Moon and real return Earth ephemerides.",
        },
    )

    with (generated / "scan_daily_best.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(daily_dicts[0].keys()))
        writer.writeheader()
        writer.writerows(daily_dicts)

    plot_orbit(generated / "orbit.png", best_dict)
    plot_scan(generated / "scan_curve.png", daily_dicts)
    plot_contour(generated / "scan_contour.png", contour, rp_grid_AU)
    plot_residuals(generated / "horizons_residuals.png", m3)
    plot_energy(generated / "energy_conservation.png", energy)
    plot_sensitivity(generated / "sensitivity.png", sens)
    make_animation(generated / "orbit_animation.mp4", best_dict)
    write_check(generated)

    print("Generated numerical results, figures and animation in data/generated")
    print(f"Best real-closure date: {best.date}, total delta-v: {best.total_delta_v_km_s:.3f} km/s")


if __name__ == "__main__":
    main()
