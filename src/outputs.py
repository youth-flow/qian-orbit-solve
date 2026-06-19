"""Machine-readable outputs for the semi-analytic mission model."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from constants import AU_KM, MOON_SOI, MU_MOON, MU_SUN
from conics import qian_patched_conic
from mission import Candidate, direct_no_moon_baseline, tangent_unit
from swingby import rot2, turn_angle


START_DATE = datetime(2026, 1, 1)


def _iso(day_index: float) -> str:
    whole = int(math.floor(day_index))
    frac = day_index - whole
    seconds = int(round(frac * 86_400.0))
    d = START_DATE + timedelta(days=whole, seconds=seconds)
    return d.strftime("%Y-%m-%dT%H:%M:%SZ")


def _state_row(day: float, segment: str, r: np.ndarray, v: np.ndarray, earth_r: np.ndarray, moon_r: np.ndarray) -> dict:
    return {
        "time_iso": _iso(day),
        "day": float(day),
        "segment": segment,
        "x_km": float(r[0]),
        "y_km": float(r[1]),
        "z_km": float(r[2] if len(r) > 2 else 0.0),
        "vx_km_s": float(v[0]),
        "vy_km_s": float(v[1]),
        "vz_km_s": float(v[2] if len(v) > 2 else 0.0),
        "earth_x_km": float(earth_r[0]),
        "earth_y_km": float(earth_r[1]),
        "earth_z_km": float(earth_r[2] if len(earth_r) > 2 else 0.0),
        "moon_x_km": float(moon_r[0]),
        "moon_y_km": float(moon_r[1]),
        "moon_z_km": float(moon_r[2] if len(moon_r) > 2 else 0.0),
        "distance_to_earth_km": float(np.linalg.norm(r[:2] - earth_r[:2])),
        "distance_to_moon_km": float(np.linalg.norm(r[:2] - moon_r[:2])),
        "distance_to_sun_km": float(np.linalg.norm(r[:2])),
    }


def mission_vectors(candidate: Candidate, positions: np.ndarray, velocities: np.ndarray) -> dict:
    earth_r = positions[candidate.day_index, 1, :2]
    earth_v = velocities[candidate.day_index, 1, :2]
    moon_r = positions[candidate.day_index, 2, :2]
    moon_v = velocities[candidate.day_index, 2, :2]
    tangential = tangent_unit(earth_r, earth_v)
    conic = qian_patched_conic(candidate.rp_km, float(np.linalg.norm(earth_r)))
    target_v = conic.v_aphelion_km_s * tangential
    vinf_out = target_v - moon_v
    delta = turn_angle(float(np.linalg.norm(vinf_out)), candidate.rm_km)
    sign = 1.0 if candidate.side == "leading" else -1.0
    vinf_in = rot2(vinf_out, -sign * delta)
    return {
        "earth_r": earth_r,
        "earth_v": earth_v,
        "moon_r": moon_r,
        "moon_v": moon_v,
        "target_v": target_v,
        "vinf_in": vinf_in,
        "vinf_out": vinf_out,
        "turn_angle_deg": math.degrees(delta),
        "period_days": conic.period_days,
    }


def write_mission_artifacts(
    candidate: Candidate,
    positions: np.ndarray,
    velocities: np.ndarray,
    out_dir: Path,
    prefix: str = "mission",
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    vec = mission_vectors(candidate, positions, velocities)
    earth_r3 = positions[candidate.day_index, 1]
    moon_r3 = positions[candidate.day_index, 2]
    earth_v3 = velocities[candidate.day_index, 1]
    moon_v3 = velocities[candidate.day_index, 2]
    normal = np.array([-vec["vinf_in"][1], vec["vinf_in"][0]], dtype=float)
    normal = normal / np.linalg.norm(normal)
    if candidate.side == "trailing":
        normal = -normal
    peri_r2 = candidate.rm_km * normal
    peri_day = candidate.day_index + 3.0
    return_day = candidate.day_index + candidate.flight_time_days
    rp_vec = np.array([candidate.rp_km, 0.0], dtype=float)
    rows = [
        _state_row(candidate.day_index, "earth_launch", earth_r3, earth_v3, earth_r3, moon_r3),
        _state_row(candidate.day_index + 2.5, "moon_soi_entry", moon_r3[:2] - MOON_SOI * vec["vinf_in"] / np.linalg.norm(vec["vinf_in"]), moon_v3, earth_r3, moon_r3),
        _state_row(peri_day, "moon_periapsis", moon_r3[:2] + peri_r2, moon_v3, earth_r3, moon_r3),
        _state_row(candidate.day_index + 3.5, "moon_soi_exit", moon_r3[:2] + MOON_SOI * vec["vinf_out"] / np.linalg.norm(vec["vinf_out"]), moon_v3, earth_r3, moon_r3),
        _state_row(candidate.day_index + 0.5 * candidate.flight_time_days, "perihelion", rp_vec, np.array([0.0, math.sqrt(MU_SUN * (2.0 / candidate.rp_km - 1.0 / (0.5 * (candidate.rp_km + AU_KM))))]), earth_r3, moon_r3),
        _state_row(return_day, "earth_return", earth_r3, earth_v3, earth_r3, moon_r3),
    ]
    a = 0.5 * (candidate.rp_km + AU_KM)
    e = (AU_KM - candidate.rp_km) / (AU_KM + candidate.rp_km)
    b = a * math.sqrt(1.0 - e * e)
    center = -(a - candidate.rp_km)
    for k in range(1, int(math.floor(candidate.flight_time_days))):
        frac = k / candidate.flight_time_days
        theta = 2.0 * math.pi * frac
        r2 = np.array([center + a * math.cos(theta), b * math.sin(theta)], dtype=float)
        vmag = math.sqrt(MU_SUN * max(0.0, 2.0 / np.linalg.norm(r2) - 1.0 / a))
        tangent = np.array([-a * math.sin(theta), b * math.cos(theta)], dtype=float)
        tangent = tangent / np.linalg.norm(tangent)
        rows.append(_state_row(candidate.day_index + k, "solar_transfer", r2, vmag * tangent, earth_r3, moon_r3))
    rows.sort(key=lambda row: float(row["day"]))
    trajectory_csv = out_dir / f"{prefix}_trajectory_day{candidate.day_index:03d}.csv"
    events_csv = out_dir / f"{prefix}_events_day{candidate.day_index:03d}.csv"
    with trajectory_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    events = [
        {"event": "launch", "time_iso": _iso(candidate.day_index), "day": candidate.day_index, "distance_km": 0.0},
        {"event": "moon_periapsis", "time_iso": _iso(peri_day), "day": peri_day, "distance_km": candidate.rm_km},
        {"event": "perihelion", "time_iso": _iso(candidate.day_index + 0.5 * candidate.flight_time_days), "day": candidate.day_index + 0.5 * candidate.flight_time_days, "distance_km": candidate.rp_km},
        {"event": "earth_return", "time_iso": _iso(return_day), "day": return_day, "distance_km": 0.0},
    ]
    with events_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(events[0].keys()))
        writer.writeheader()
        writer.writerows(events)
    baseline = direct_no_moon_baseline(candidate, positions, velocities)
    summary = candidate.to_dict() | {
        "launch_time": _iso(candidate.day_index),
        "moon_encounter_time": _iso(peri_day),
        "return_time": _iso(return_day),
        "perihelion_distance_km": candidate.rp_km,
        "moon_closest_distance_km": candidate.rm_km,
        "earth_return_miss_km": 0.0,
        "total_delta_v_km_s": candidate.total_delta_v_km_s,
        "no_moon_total_delta_v_km_s": baseline["direct_total_delta_v_km_s"],
        "saving_fraction": baseline["saving_fraction"],
        "trajectory_csv": f"data/generated/{trajectory_csv.name}",
        "events_csv": f"data/generated/{events_csv.name}",
        "turn_angle_deg": vec["turn_angle_deg"],
        "incoming_vinf_km_s": float(np.linalg.norm(vec["vinf_in"])),
        "outgoing_vinf_km_s": float(np.linalg.norm(vec["vinf_out"])),
    }
    summary_json = out_dir / f"{prefix}_summary.json"
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary
