"""Optional analysis modules for 3D inclination and relativity effects."""

from __future__ import annotations

import math

import numpy as np

from constants import AU_KM, EARTH_ESCAPE_SURFACE, MU_SUN


C_KM_S = 299_792.458
MOON_ORBIT_INCLINATION_DEG = 5.145


def inclination_effect(summary: dict, positions: np.ndarray, velocities: np.ndarray) -> dict[str, float]:
    day = int(summary["day_index"])
    moon_r = positions[day, 2]
    earth_r = positions[day, 1]
    rel = moon_r - earth_r
    in_plane = float(np.linalg.norm(rel[:2]))
    out_of_plane = abs(float(rel[2]))
    angle = math.degrees(math.atan2(out_of_plane, max(in_plane, 1.0e-12)))
    return {
        "model": "3D ecliptic projection audit with real cached Moon z coordinate",
        "nominal_lunar_orbit_inclination_deg": MOON_ORBIT_INCLINATION_DEG,
        "moon_earth_in_plane_distance_km": in_plane,
        "moon_earth_out_of_plane_km": out_of_plane,
        "instantaneous_inclination_deg": angle,
        "approx_velocity_penalty_km_s": summary["incoming_vinf_km_s"] * (1.0 / max(math.cos(math.radians(angle)), 1.0e-12) - 1.0),
    }


def inclination_window_analysis(daily: list[dict], positions: np.ndarray, sample_days: tuple[int, ...] = (0, 81, 168, 189, 201, 364)) -> dict:
    """Apply a first-order 3D inclination penalty to the 2026 launch scan.

    The baseline mission model uses the J2000 ecliptic projection.  This audit
    keeps the same launch dates and patched-conic variables, then estimates the
    out-of-plane cleanup needed to rotate the geocentric departure vector into
    the real Moon plane.  It is deliberately conservative and deterministic:
    no optimum is re-fit, so the reported change is an apples-to-apples window
    sensitivity rather than a different optimization problem.
    """

    corrected = []
    samples = []
    for row in daily:
        day = int(row["day_index"])
        rel = positions[day, 2] - positions[day, 1]
        in_plane = float(np.linalg.norm(rel[:2]))
        out_of_plane = abs(float(rel[2]))
        angle = math.atan2(out_of_plane, max(in_plane, 1.0e-12))
        launch = float(row["launch_delta_v_km_s"])
        earth_vinf = math.sqrt(max(0.0, launch * launch - EARTH_ESCAPE_SURFACE * EARTH_ESCAPE_SURFACE))
        penalty = earth_vinf * (1.0 / max(math.cos(angle), 1.0e-12) - 1.0)
        total_3d = float(row["total_delta_v_km_s"]) + penalty
        corrected.append(
            {
                "date": row["date"],
                "day_index": day,
                "instantaneous_inclination_deg": math.degrees(angle),
                "out_of_plane_km": out_of_plane,
                "penalty_km_s": penalty,
                "total_2d_km_s": float(row["total_delta_v_km_s"]),
                "total_3d_estimate_km_s": total_3d,
            }
        )
        if day in sample_days:
            samples.append(corrected[-1])
    best_2d = min(daily, key=lambda r: float(r["total_delta_v_km_s"]))
    best_3d = min(corrected, key=lambda r: r["total_3d_estimate_km_s"])
    penalties = np.array([r["penalty_km_s"] for r in corrected], dtype=float)
    return {
        "method": "daily 2D scan plus deterministic Moon-z plane-change penalty",
        "nominal_lunar_orbit_inclination_deg": MOON_ORBIT_INCLINATION_DEG,
        "best_2d_date": best_2d["date"],
        "best_2d_total_km_s": float(best_2d["total_delta_v_km_s"]),
        "best_3d_estimate_date": best_3d["date"],
        "best_3d_estimate_total_km_s": best_3d["total_3d_estimate_km_s"],
        "best_date_shift_days": int(best_3d["day_index"] - int(best_2d["day_index"])),
        "mean_penalty_km_s": float(np.mean(penalties)),
        "max_penalty_km_s": float(np.max(penalties)),
        "sample_days": samples,
    }


def schwarzschild_acceleration(r_vec: np.ndarray, v_vec: np.ndarray, mu: float = MU_SUN, c_km_s: float = C_KM_S) -> np.ndarray:
    """First post-Newtonian Schwarzschild correction for a solar test particle."""

    r = np.asarray(r_vec, dtype=float)
    v = np.asarray(v_vec, dtype=float)
    r_norm = float(np.linalg.norm(r))
    if r_norm <= 0.0:
        raise ValueError("position norm must be positive")
    v2 = float(np.dot(v, v))
    rv = float(np.dot(r, v))
    return mu / (c_km_s * c_km_s * r_norm**3) * ((4.0 * mu / r_norm - v2) * r + 4.0 * rv * v)


def gr_correction(summary: dict) -> dict[str, float]:
    rp = float(summary["perihelion_distance_km"])
    ra = AU_KM
    a = 0.5 * (rp + ra)
    e = (ra - rp) / (ra + rp)
    advance_rad = 6.0 * math.pi * MU_SUN / (a * (1.0 - e * e) * C_KM_S**2)
    v_peri = math.sqrt(MU_SUN * (2.0 / rp - 1.0 / a))
    r_vec = np.array([rp, 0.0, 0.0], dtype=float)
    v_vec = np.array([0.0, v_peri, 0.0], dtype=float)
    gr_acc = schwarzschild_acceleration(r_vec, v_vec)
    newton_acc = MU_SUN / (rp * rp)
    return {
        "model": "first post-Newtonian Schwarzschild correction",
        "semi_major_axis_km": a,
        "eccentricity": e,
        "perihelion_speed_km_s": v_peri,
        "perihelion_advance_arcsec_per_orbit": math.degrees(advance_rad) * 3600.0,
        "gr_acceleration_x_km_s2": float(gr_acc[0]),
        "gr_acceleration_y_km_s2": float(gr_acc[1]),
        "newtonian_acceleration_at_perihelion_km_s2": newton_acc,
        "relative_acceleration_at_perihelion": float(np.linalg.norm(gr_acc) / newton_acc),
    }
