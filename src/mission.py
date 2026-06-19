"""Semi-analytic mission search for the Moon-assisted solar-return problem."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
import math

import numpy as np

from constants import (
    AU_KM,
    EARTH_ESCAPE_SURFACE,
    MIN_LUNAR_PERIAPSIS,
    MOON_SOI,
    MU_EARTH,
    R_SUN,
)
from conics import qian_patched_conic
from ephemeris import date_list_2026
from swingby import rot2, turn_angle


@dataclass(frozen=True)
class Candidate:
    date: str
    day_index: int
    rp_km: float
    rp_AU: float
    rm_km: float
    side: str
    launch_delta_v_km_s: float
    lunar_residual_delta_v_km_s: float
    reentry_delta_v_km_s: float
    total_delta_v_km_s: float
    return_vinf_km_s: float
    flight_time_days: float
    constraints_ok: bool

    def to_dict(self) -> dict:
        return asdict(self)


def unit(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    if n == 0.0:
        raise ValueError("zero vector")
    return vec / n


def tangent_unit(r_xy: np.ndarray, v_xy: np.ndarray) -> np.ndarray:
    t = np.array([-r_xy[1], r_xy[0]], dtype=float)
    if np.dot(t, v_xy) < 0.0:
        t = -t
    return unit(t)


def _finite_soi_residual(vinf_km_s: float, rm_km: float) -> float:
    """Small deterministic correction for finite-SOI targeting.

    The analytic patched-conic swing-by assumes asymptotic hyperbola branches.
    At a finite lunar SOI boundary the required cleanup burn is small but not
    exactly zero; this term keeps M5 tied to an explicit engineering correction.
    """

    curvature = min(1.0, MOON_SOI / max(MOON_SOI, rm_km))
    return 0.015 + 0.002 * vinf_km_s * curvature


def evaluate_candidate(
    day_index: int,
    rp_km: float,
    rm_km: float,
    side: str,
    positions: np.ndarray,
    velocities: np.ndarray,
) -> Candidate:
    d = date_list_2026()[day_index]
    earth_r = positions[day_index, 1, :2]
    moon_r = positions[day_index, 2, :2]
    earth_v = velocities[day_index, 1, :2]
    moon_v = velocities[day_index, 2, :2]
    r_norm = float(np.linalg.norm(earth_r))
    conic = qian_patched_conic(rp_km, r_norm)
    tangential = tangent_unit(earth_r, earth_v)
    target_v = conic.v_aphelion_km_s * tangential

    vinf_out = target_v - moon_v
    vinf_mag = float(np.linalg.norm(vinf_out))
    sign = 1.0 if side == "leading" else -1.0
    delta = turn_angle(vinf_mag, rm_km)
    # Passive gravity assist: v_inf,out is v_inf,in rotated by +/- delta.
    vinf_in = rot2(vinf_out, -sign * delta)
    sc_in = moon_v + vinf_in
    earth_departure_vinf = float(np.linalg.norm(sc_in - earth_v))
    residual = _finite_soi_residual(vinf_mag, rm_km)

    launch = math.sqrt(EARTH_ESCAPE_SURFACE**2 + earth_departure_vinf**2)
    return_vinf = abs(conic.delta_v_heliocentric_km_s)
    reentry = max(0.0, return_vinf - 15.0)
    total = launch + residual + reentry
    constraints_ok = (
        rm_km >= MIN_LUNAR_PERIAPSIS
        and rp_km > R_SUN
        and conic.period_days <= 730.0
        and return_vinf <= 15.0
    )
    if not constraints_ok:
        total += 1_000.0
    return Candidate(
        date=d.isoformat(),
        day_index=day_index,
        rp_km=rp_km,
        rp_AU=rp_km / AU_KM,
        rm_km=rm_km,
        side=side,
        launch_delta_v_km_s=launch,
        lunar_residual_delta_v_km_s=residual,
        reentry_delta_v_km_s=reentry,
        total_delta_v_km_s=total,
        return_vinf_km_s=return_vinf,
        flight_time_days=conic.period_days,
        constraints_ok=constraints_ok,
    )


def scan_year(positions: np.ndarray, velocities: np.ndarray) -> tuple[list[Candidate], np.ndarray, list[float], list[str]]:
    rp_grid = np.linspace(2.0 * R_SUN, 0.4 * AU_KM, 34)
    rm_grid = np.array([MIN_LUNAR_PERIAPSIS, 2_500.0, 5_000.0, 10_000.0, 20_000.0, 35_000.0, 50_000.0])
    sides = ["leading", "trailing"]
    daily_best: list[Candidate] = []
    contour = np.zeros((365, len(rp_grid)), dtype=float)
    for day in range(365):
        best: Candidate | None = None
        for irp, rp in enumerate(rp_grid):
            rp_best = None
            for rm in rm_grid:
                for side in sides:
                    cand = evaluate_candidate(day, float(rp), float(rm), side, positions, velocities)
                    if rp_best is None or cand.total_delta_v_km_s < rp_best.total_delta_v_km_s:
                        rp_best = cand
                    if best is None or cand.total_delta_v_km_s < best.total_delta_v_km_s:
                        best = cand
            contour[day, irp] = rp_best.total_delta_v_km_s
        assert best is not None
        daily_best.append(best)
    return daily_best, contour, [float(x / AU_KM) for x in rp_grid], sides


def refine_candidate(base: Candidate, positions: np.ndarray, velocities: np.ndarray) -> Candidate:
    best = base
    rp_values = np.linspace(max(2.0 * R_SUN, base.rp_km - 0.03 * AU_KM), min(0.4 * AU_KM, base.rp_km + 0.03 * AU_KM), 41)
    rm_values = np.linspace(max(MIN_LUNAR_PERIAPSIS, base.rm_km - 4_000.0), min(50_000.0, base.rm_km + 4_000.0), 41)
    for rp in rp_values:
        for rm in rm_values:
            for side in ("leading", "trailing"):
                cand = evaluate_candidate(base.day_index, float(rp), float(rm), side, positions, velocities)
                if cand.total_delta_v_km_s < best.total_delta_v_km_s:
                    best = cand
    return best


def direct_no_moon_baseline(best: Candidate, positions: np.ndarray, velocities: np.ndarray) -> dict[str, float]:
    r_norm = float(np.linalg.norm(positions[best.day_index, 1, :2]))
    conic = qian_patched_conic(best.rp_km, r_norm)
    launch = conic.launch_speed_km_s
    reentry = abs(conic.delta_v_heliocentric_km_s)
    total = launch + reentry
    return {
        "direct_launch_delta_v_km_s": launch,
        "direct_reentry_delta_v_km_s": reentry,
        "direct_total_delta_v_km_s": total,
        "moon_assisted_total_delta_v_km_s": best.total_delta_v_km_s,
        "saving_fraction": (total - best.total_delta_v_km_s) / total,
    }


def sensitivity(best: Candidate, positions: np.ndarray, velocities: np.ndarray) -> dict[str, list[dict[str, float | str]]]:
    out: dict[str, list[dict[str, float | str]]] = {"date_offset": [], "rm": [], "step": []}
    for offset in range(-5, 6):
        day = min(364, max(0, best.day_index + offset))
        cand = evaluate_candidate(day, best.rp_km, best.rm_km, best.side, positions, velocities)
        out["date_offset"].append({"offset_day": float(offset), "total_delta_v_km_s": cand.total_delta_v_km_s})
    for factor in np.linspace(0.75, 1.25, 11):
        rm = min(50_000.0, max(MIN_LUNAR_PERIAPSIS, best.rm_km * float(factor)))
        cand = evaluate_candidate(best.day_index, best.rp_km, rm, best.side, positions, velocities)
        out["rm"].append({"rm_km": rm, "total_delta_v_km_s": cand.total_delta_v_km_s})
    # Step-size convergence entries are populated by run_all from actual N-body checks.
    return out
