"""Exploratory Earth-Moon-Venus-Sun-Earth multi-flyby audit.

The required mission is a single lunar assist.  This module is deliberately
separate: it explores a more complex sequence and reports the residual budget
needed to make the patched conics close.  Earth and Moon states come from the
assignment Horizons cache.  Venus is read from a generated JPL Horizons DE441
offline cache when available; the low-precision mean-element model remains as
a fallback so the module can still run if that optional file is absent.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from constants import AU_KM, MOON_SOI, MU_SUN
from ephemeris import venus_states_from_cache
from lambert import lambert_universal
from mission import Candidate
from outputs import mission_vectors
from real_closure import RealClosureCandidate, _reconstruct_states
from swingby import turn_angle


MU_VENUS = 324_858.592
R_VENUS = 6_051.8
VENUS_FLYBY_ALTITUDE_KM = 300.0
J2000_TO_2026_JAN1_DAYS = 9_497.5


@dataclass(frozen=True)
class MultiFlybyCandidate:
    launch_date: str
    launch_day_index: int
    moon_exit_day: float
    venus_flyby_day: int
    earth_return_day: int
    venus_periapsis_km: float
    moon_patch_delta_v_km_s: float
    venus_speed_residual_km_s: float
    venus_turn_residual_km_s: float
    earth_return_vinf_km_s: float
    total_residual_budget_km_s: float
    incoming_venus_vinf_km_s: float
    outgoing_venus_vinf_km_s: float
    required_venus_turn_deg: float
    available_venus_turn_deg: float
    solar_perihelion_after_venus_AU: float
    notes: str

    def to_dict(self) -> dict:
        return asdict(self)


def _solve_kepler(mean_anomaly_rad: float, eccentricity: float, max_iter: int = 30) -> float:
    mean = (mean_anomaly_rad + math.pi) % (2.0 * math.pi) - math.pi
    ecc_anomaly = mean if eccentricity < 0.8 else math.pi
    for _ in range(max_iter):
        f = ecc_anomaly - eccentricity * math.sin(ecc_anomaly) - mean
        fp = 1.0 - eccentricity * math.cos(ecc_anomaly)
        step = f / fp
        ecc_anomaly -= step
        if abs(step) < 1.0e-13:
            break
    return ecc_anomaly


def _rotation_matrix(raan: float, inc: float, arg_peri: float) -> np.ndarray:
    cO, sO = math.cos(raan), math.sin(raan)
    ci, si = math.cos(inc), math.sin(inc)
    cw, sw = math.cos(arg_peri), math.sin(arg_peri)
    return np.array(
        [
            [cO * cw - sO * sw * ci, -cO * sw - sO * cw * ci, sO * si],
            [sO * cw + cO * sw * ci, -sO * sw + cO * cw * ci, -cO * si],
            [sw * si, cw * si, ci],
        ],
        dtype=float,
    )


def venus_state_low_precision(day_index: float) -> tuple[np.ndarray, np.ndarray]:
    """Return heliocentric ecliptic Venus state in km and km/s.

    Elements are the common JPL low-precision set for Venus at J2000 with
    linear century rates.  They are adequate for a qualitative multi-flyby
    screening audit and keep the project offline-reproducible.
    """

    t_century = (J2000_TO_2026_JAN1_DAYS + day_index) / 36_525.0
    a_au = 0.723_335_66 + 0.000_003_90 * t_century
    e = 0.006_776_72 - 0.000_041_07 * t_century
    inc_deg = 3.394_676_05 - 0.000_788_90 * t_century
    mean_long_deg = 181.979_099_50 + 58_517.815_387_29 * t_century
    long_peri_deg = 131.602_467_18 + 0.002_683_29 * t_century
    raan_deg = 76.679_842_55 - 0.277_694_18 * t_century

    a = a_au * AU_KM
    inc = math.radians(inc_deg)
    mean_long = math.radians(mean_long_deg % 360.0)
    long_peri = math.radians(long_peri_deg % 360.0)
    raan = math.radians(raan_deg % 360.0)
    arg_peri = long_peri - raan
    mean_anomaly = mean_long - long_peri
    ecc_anomaly = _solve_kepler(mean_anomaly, e)
    cos_e, sin_e = math.cos(ecc_anomaly), math.sin(ecc_anomaly)
    n = math.sqrt(MU_SUN / (a**3))
    perifocal_r = np.array([a * (cos_e - e), a * math.sqrt(1.0 - e * e) * sin_e, 0.0], dtype=float)
    denom = 1.0 - e * cos_e
    perifocal_v = np.array(
        [-a * n * sin_e / denom, a * n * math.sqrt(1.0 - e * e) * cos_e / denom, 0.0],
        dtype=float,
    )
    rot = _rotation_matrix(raan, inc, arg_peri)
    return rot @ perifocal_r, rot @ perifocal_v


def _interpolate_cache_state(day_index: float, body_index: int, positions: np.ndarray, velocities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if day_index <= 0.0:
        return positions[0, body_index].copy(), velocities[0, body_index].copy()
    if day_index >= positions.shape[0] - 1:
        return positions[-1, body_index].copy(), velocities[-1, body_index].copy()
    lo = int(math.floor(day_index))
    frac = day_index - lo
    r = (1.0 - frac) * positions[lo, body_index] + frac * positions[lo + 1, body_index]
    v = (1.0 - frac) * velocities[lo, body_index] + frac * velocities[lo + 1, body_index]
    return r, v


def _interpolate_series_state(day_index: float, positions: np.ndarray, velocities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if day_index <= 0.0:
        return positions[0].copy(), velocities[0].copy()
    if day_index >= positions.shape[0] - 1:
        return positions[-1].copy(), velocities[-1].copy()
    lo = int(math.floor(day_index))
    frac = day_index - lo
    r = (1.0 - frac) * positions[lo] + frac * positions[lo + 1]
    v = (1.0 - frac) * velocities[lo] + frac * velocities[lo + 1]
    return r, v


def _angle_between(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    na = float(np.linalg.norm(vec_a))
    nb = float(np.linalg.norm(vec_b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    cosang = float(np.dot(vec_a, vec_b) / (na * nb))
    cosang = max(-1.0, min(1.0, cosang))
    return math.acos(cosang)


def _solar_orbit_perihelion_au(r_vec: np.ndarray, v_vec: np.ndarray) -> float:
    r = np.asarray(r_vec, dtype=float)
    v = np.asarray(v_vec, dtype=float)
    rnorm = float(np.linalg.norm(r))
    v2 = float(np.dot(v, v))
    energy = 0.5 * v2 - MU_SUN / rnorm
    if energy >= 0.0:
        return float("inf")
    a = -MU_SUN / (2.0 * energy)
    h = np.cross(r, v)
    e_vec = np.cross(v, h) / MU_SUN - r / rnorm
    e = float(np.linalg.norm(e_vec))
    return a * (1.0 - e) / AU_KM


def _moon_exit_state(candidate: Candidate | RealClosureCandidate, positions: np.ndarray, velocities: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    if hasattr(candidate, "moon_encounter_day"):
        states = _reconstruct_states(candidate, positions, velocities)
        return (
            float(candidate.moon_encounter_day),
            np.asarray(states["moon_periapsis_r"], dtype=float),
            np.asarray(states["v_depart_moon_required"], dtype=float),
        )
    vec = mission_vectors(candidate, positions, velocities)
    moon_r = positions[candidate.day_index, 2].copy()
    target_v = np.array([vec["target_v"][0], vec["target_v"][1], 0.0], dtype=float)
    vinf_out = np.array([vec["vinf_out"][0], vec["vinf_out"][1], 0.0], dtype=float)
    exit_r = moon_r + MOON_SOI * vinf_out / max(float(np.linalg.norm(vinf_out)), 1.0e-12)
    exit_day = candidate.day_index + 3.5
    return exit_day, exit_r, target_v


def evaluate_sequence(
    candidate: Candidate | RealClosureCandidate,
    positions: np.ndarray,
    velocities: np.ndarray,
    venus_day: int,
    return_day: int,
    venus_positions: np.ndarray | None = None,
    venus_velocities: np.ndarray | None = None,
    venus_periapsis_km: float = R_VENUS + VENUS_FLYBY_ALTITUDE_KM,
) -> MultiFlybyCandidate | None:
    exit_day, exit_r, exit_v = _moon_exit_state(candidate, positions, velocities)
    if venus_day <= exit_day + 20.0 or return_day <= venus_day + 30:
        return None
    if return_day >= positions.shape[0]:
        return None
    if venus_positions is not None and venus_velocities is not None:
        venus_r, venus_v = _interpolate_series_state(float(venus_day), venus_positions, venus_velocities)
    else:
        venus_r, venus_v = venus_state_low_precision(float(venus_day))
    earth_r, earth_v = _interpolate_cache_state(float(return_day), 1, positions, velocities)
    try:
        v_moon_to_venus_0, v_arrive_venus = lambert_universal(
            exit_r,
            venus_r,
            (venus_day - exit_day) * 86_400.0,
            MU_SUN,
        )
        v_leave_venus, v_arrive_earth = lambert_universal(
            venus_r,
            earth_r,
            (return_day - venus_day) * 86_400.0,
            MU_SUN,
        )
    except (ValueError, OverflowError, FloatingPointError):
        return None

    vinf_in = v_arrive_venus - venus_v
    vinf_out = v_leave_venus - venus_v
    vinf_in_speed = float(np.linalg.norm(vinf_in))
    vinf_out_speed = float(np.linalg.norm(vinf_out))
    if vinf_in_speed <= 1.0e-9 or vinf_out_speed <= 1.0e-9:
        return None
    required_turn = _angle_between(vinf_in, vinf_out)
    available_turn = turn_angle(0.5 * (vinf_in_speed + vinf_out_speed), venus_periapsis_km, MU_VENUS)
    speed_residual = abs(vinf_out_speed - vinf_in_speed)
    turn_residual = max(0.0, required_turn - available_turn) * 0.5 * (vinf_in_speed + vinf_out_speed)
    moon_patch = float(np.linalg.norm(v_moon_to_venus_0 - exit_v))
    earth_vinf = float(np.linalg.norm(v_arrive_earth - earth_v))
    perihelion = _solar_orbit_perihelion_au(venus_r, v_leave_venus)
    total = moon_patch + speed_residual + turn_residual + earth_vinf
    return MultiFlybyCandidate(
        launch_date=candidate.date,
        launch_day_index=candidate.day_index,
        moon_exit_day=exit_day,
        venus_flyby_day=venus_day,
        earth_return_day=return_day,
        venus_periapsis_km=venus_periapsis_km,
        moon_patch_delta_v_km_s=moon_patch,
        venus_speed_residual_km_s=speed_residual,
        venus_turn_residual_km_s=turn_residual,
        earth_return_vinf_km_s=earth_vinf,
        total_residual_budget_km_s=total,
        incoming_venus_vinf_km_s=vinf_in_speed,
        outgoing_venus_vinf_km_s=vinf_out_speed,
        required_venus_turn_deg=math.degrees(required_turn),
        available_venus_turn_deg=math.degrees(available_turn),
        solar_perihelion_after_venus_AU=perihelion,
        notes="Exploratory patched-conic sequence using explicit Earth, Moon and Venus ephemeris states.",
    )


def explore_multiflyby(
    candidate: Candidate | RealClosureCandidate,
    positions: np.ndarray,
    velocities: np.ndarray,
    venus_step_days: int = 8,
    return_step_days: int = 8,
    keep: int = 12,
) -> dict:
    candidates: list[MultiFlybyCandidate] = []
    venus_source = "JPL low-precision mean elements propagated from J2000"
    venus_positions = None
    venus_velocities = None
    try:
        _, venus_positions, venus_velocities = venus_states_from_cache()
        venus_source = "data/venus_horizons_cache_2026.json, JPL Horizons DE441, CENTER='@10'"
    except FileNotFoundError:
        venus_positions = None
        venus_velocities = None
    start_venus = max(candidate.day_index + 45, 0)
    stop_venus = min(candidate.day_index + 145, positions.shape[0] - 45)
    if venus_positions is not None:
        stop_venus = min(stop_venus, venus_positions.shape[0] - 1)
    for venus_day in range(start_venus, stop_venus + 1, venus_step_days):
        start_return = venus_day + 55
        stop_return = min(positions.shape[0] - 1, venus_day + 150)
        for return_day in range(start_return, stop_return + 1, return_step_days):
            item = evaluate_sequence(
                candidate,
                positions,
                velocities,
                venus_day,
                return_day,
                venus_positions=venus_positions,
                venus_velocities=venus_velocities,
            )
            if item is not None and math.isfinite(item.total_residual_budget_km_s):
                candidates.append(item)
    candidates.sort(key=lambda item: item.total_residual_budget_km_s)
    top = candidates[:keep]
    best = top[0] if top else None
    return {
        "sequence": "Earth -> Moon -> Venus -> Sun -> Earth",
        "ephemeris": {
            "Earth": "assignment JPL Horizons cache, CENTER='@10'",
            "Moon": "assignment JPL Horizons cache, CENTER='@10'",
            "Venus": venus_source,
        },
        "searched_candidates": len(candidates),
        "best": None if best is None else best.to_dict(),
        "top_candidates": [item.to_dict() for item in top],
    }


def write_multiflyby_artifacts(summary: dict, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "multi_flyby_summary.json"
    csv_path = out_dir / "multi_flyby_candidates.csv"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = summary.get("top_candidates", [])
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return {
        "summary_json": f"data/generated/{json_path.name}",
        "candidates_csv": f"data/generated/{csv_path.name}",
        "best": summary.get("best"),
        "searched_candidates": summary.get("searched_candidates", 0),
    }
