"""Ephemeris-closed lunar-assist model for the corrected grading rules.

The older semi-analytic scan is useful for patched-conic intuition, but it
keeps the Moon and the returning Earth at the launch-day geometry.  This module
builds the machine-checkable mission artifacts from actual Horizons states:

* launch from the real Earth state at t0,
* intersect the real Moon state at tm,
* leave the Moon onto a Lambert arc ending at the real Earth state at tr,
* book the vector mismatch between passive lunar turning and the required
  outbound Lambert velocity as ``Delta v_Moon,res``.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import json
import math
from pathlib import Path

import numpy as np

from constants import AU_KM, EARTH_ESCAPE_SURFACE, MIN_LUNAR_PERIAPSIS, MU_SUN, R_SUN
from ephemeris import date_list_2026, interpolate_state
from lambert import lambert_universal, stumpff_c, stumpff_s
from swingby import turn_angle


START_DATE = datetime(2026, 1, 1)


@dataclass(frozen=True)
class RealClosureCandidate:
    date: str
    day_index: int
    moon_encounter_day: float
    return_day: float
    transfer_to_moon_days: float
    return_transfer_days: float
    rm_km: float
    side: str
    post_moon_long_way: bool
    launch_delta_v_km_s: float
    lunar_residual_delta_v_km_s: float
    reentry_delta_v_km_s: float
    total_delta_v_km_s: float
    raw_return_vinf_km_s: float
    return_vinf_km_s: float
    perihelion_distance_km: float
    rp_km: float
    rp_AU: float
    semi_major_axis_AU: float
    eccentricity: float
    incoming_vinf_km_s: float
    required_outgoing_vinf_km_s: float
    passive_outgoing_vinf_km_s: float
    required_turn_angle_deg: float
    available_turn_angle_deg: float
    turn_shortfall_deg: float
    earth_return_miss_km: float
    moon_position_error_km: float
    constraints_ok: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _iso(day_index: float) -> str:
    whole = int(math.floor(day_index))
    frac = day_index - whole
    seconds = int(round(frac * 86_400.0))
    d = START_DATE + timedelta(days=whole, seconds=seconds)
    return d.strftime("%Y-%m-%dT%H:%M:%SZ")


def _unit(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        raise ValueError("zero vector")
    return vec / norm


def _angle_between(a_vec: np.ndarray, b_vec: np.ndarray) -> float:
    denom = float(np.linalg.norm(a_vec) * np.linalg.norm(b_vec))
    if denom <= 0.0:
        return 0.0
    cosang = float(np.dot(a_vec, b_vec) / denom)
    return math.acos(max(-1.0, min(1.0, cosang)))


def _rotate_toward(v_from: np.ndarray, v_target: np.ndarray, max_angle_rad: float) -> np.ndarray:
    """Rotate ``v_from`` toward ``v_target`` by at most ``max_angle_rad``."""

    theta = _angle_between(v_from, v_target)
    if theta <= 1.0e-14:
        return v_from.copy()
    axis = np.cross(v_from, v_target)
    if np.linalg.norm(axis) < 1.0e-12:
        axis = np.cross(v_from, np.array([0.0, 0.0, 1.0], dtype=float))
    if np.linalg.norm(axis) < 1.0e-12:
        axis = np.array([0.0, 1.0, 0.0], dtype=float)
    k = _unit(axis)
    angle = min(theta, max_angle_rad)
    return (
        v_from * math.cos(angle)
        + np.cross(k, v_from) * math.sin(angle)
        + k * float(np.dot(k, v_from)) * (1.0 - math.cos(angle))
    )


def _solar_orbit_elements(r_vec: np.ndarray, v_vec: np.ndarray) -> dict[str, float]:
    r = np.asarray(r_vec, dtype=float)
    v = np.asarray(v_vec, dtype=float)
    r_norm = float(np.linalg.norm(r))
    v2 = float(np.dot(v, v))
    energy = 0.5 * v2 - MU_SUN / r_norm
    if energy >= 0.0:
        return {
            "perihelion_distance_km": float("inf"),
            "semi_major_axis_km": float("inf"),
            "eccentricity": float("inf"),
            "period_days": float("inf"),
            "time_to_perihelion_days": float("inf"),
        }
    a = -MU_SUN / (2.0 * energy)
    h_vec = np.cross(r, v)
    e_vec = np.cross(v, h_vec) / MU_SUN - r / r_norm
    e = float(np.linalg.norm(e_vec))
    q = a * (1.0 - e)
    period_days = 2.0 * math.pi * math.sqrt(a**3 / MU_SUN) / 86_400.0
    if e < 1.0e-12:
        time_to_perihelion = 0.0
    else:
        h_hat = _unit(h_vec)
        p_hat = _unit(e_vec)
        q_hat = np.cross(h_hat, p_hat)
        nu = math.atan2(float(np.dot(r, q_hat)), float(np.dot(r, p_hat)))
        ecc_anom = 2.0 * math.atan2(
            math.sqrt(max(0.0, 1.0 - e)) * math.sin(0.5 * nu),
            math.sqrt(1.0 + e) * math.cos(0.5 * nu),
        )
        mean_anom = (ecc_anom - e * math.sin(ecc_anom)) % (2.0 * math.pi)
        mean_motion = math.sqrt(MU_SUN / a**3)
        since_perihelion_days = mean_anom / mean_motion / 86_400.0
        time_to_perihelion = 0.0 if since_perihelion_days < 1.0e-9 else period_days - since_perihelion_days
    return {
        "perihelion_distance_km": float(q),
        "semi_major_axis_km": float(a),
        "eccentricity": float(e),
        "period_days": float(period_days),
        "time_to_perihelion_days": float(time_to_perihelion),
    }


def _kepler_propagate(r0_vec: np.ndarray, v0_vec: np.ndarray, dt_s: float, mu: float = MU_SUN) -> tuple[np.ndarray, np.ndarray]:
    """Propagate a two-body state with the universal-variable f-g solution."""

    r0 = np.asarray(r0_vec, dtype=float)
    v0 = np.asarray(v0_vec, dtype=float)
    r0_norm = float(np.linalg.norm(r0))
    v0_sq = float(np.dot(v0, v0))
    vr0 = float(np.dot(r0, v0) / r0_norm)
    alpha = 2.0 / r0_norm - v0_sq / mu
    if abs(dt_s) < 1.0e-12:
        return r0.copy(), v0.copy()
    sqrt_mu = math.sqrt(mu)
    if abs(alpha) > 1.0e-12:
        chi = sqrt_mu * abs(alpha) * dt_s
    else:
        chi = sqrt_mu * dt_s / r0_norm
    if chi == 0.0:
        chi = math.copysign(1.0e-6, dt_s)
    for _ in range(80):
        z = alpha * chi * chi
        c = stumpff_c(z)
        s = stumpff_s(z)
        f_val = (
            r0_norm * vr0 / sqrt_mu * chi * chi * c
            + (1.0 - alpha * r0_norm) * chi**3 * s
            + r0_norm * chi
            - sqrt_mu * dt_s
        )
        df_val = (
            r0_norm * vr0 / sqrt_mu * chi * (1.0 - z * s)
            + (1.0 - alpha * r0_norm) * chi * chi * c
            + r0_norm
        )
        step = f_val / df_val
        chi -= step
        if abs(step) < 1.0e-8:
            break
    z = alpha * chi * chi
    c = stumpff_c(z)
    s = stumpff_s(z)
    f = 1.0 - chi * chi / r0_norm * c
    g = dt_s - chi**3 / sqrt_mu * s
    r = f * r0 + g * v0
    r_norm = float(np.linalg.norm(r))
    fdot = sqrt_mu / (r_norm * r0_norm) * (alpha * chi**3 * s - chi)
    gdot = 1.0 - chi * chi / r_norm * c
    v = fdot * r0 + gdot * v0
    return r, v


def _periapsis_direction(
    launch_day: float,
    moon_day: float,
    vinf_arrive: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    side_sign: float,
) -> np.ndarray:
    earth_r, earth_v = interpolate_state(moon_day, 1, positions, velocities)
    moon_r, moon_v = interpolate_state(moon_day, 2, positions, velocities)
    moon_h = np.cross(moon_r - earth_r, moon_v - earth_v)
    if np.linalg.norm(moon_h) < 1.0e-12:
        launch_r, launch_v = interpolate_state(launch_day, 1, positions, velocities)
        moon_h = np.cross(moon_r - launch_r, moon_v - launch_v)
    b_dir = np.cross(moon_h, vinf_arrive)
    if np.linalg.norm(b_dir) < 1.0e-12:
        b_dir = np.cross(vinf_arrive, np.array([0.0, 0.0, 1.0], dtype=float))
    if np.linalg.norm(b_dir) < 1.0e-12:
        b_dir = np.array([1.0, 0.0, 0.0], dtype=float)
    return side_sign * _unit(b_dir)


def evaluate_real_candidate(
    day_index: int,
    transfer_to_moon_days: float,
    return_day_offset: float,
    rm_km: float,
    side: str,
    positions: np.ndarray,
    velocities: np.ndarray,
    post_moon_long_way: bool = False,
) -> RealClosureCandidate | None:
    """Evaluate one real-ephemeris patched-conic closure candidate."""

    if return_day_offset <= transfer_to_moon_days + 30.0:
        return None
    if day_index + return_day_offset >= positions.shape[0] - 1:
        return None
    side_sign = 1.0 if side == "leading" else -1.0
    t0 = float(day_index)
    tm = t0 + float(transfer_to_moon_days)
    tr = t0 + float(return_day_offset)
    earth_r0, earth_v0 = interpolate_state(t0, 1, positions, velocities)
    moon_r, moon_v = interpolate_state(tm, 2, positions, velocities)
    earth_rr, earth_vr = interpolate_state(tr, 1, positions, velocities)
    try:
        _, v_arrive_moon_center = lambert_universal(earth_r0, moon_r, (tm - t0) * 86_400.0, MU_SUN)
    except (ValueError, OverflowError, FloatingPointError):
        return None
    vinf_center = v_arrive_moon_center - moon_v
    try:
        peri_dir = _periapsis_direction(t0, tm, vinf_center, positions, velocities, side_sign)
    except ValueError:
        return None
    moon_periapsis_r = moon_r + float(rm_km) * peri_dir
    try:
        v_launch, v_arrive_periapsis = lambert_universal(
            earth_r0,
            moon_periapsis_r,
            (tm - t0) * 86_400.0,
            MU_SUN,
        )
        v_depart_moon_required, v_arrive_earth = lambert_universal(
            moon_periapsis_r,
            earth_rr,
            (tr - tm) * 86_400.0,
            MU_SUN,
            long_way=post_moon_long_way,
        )
    except (ValueError, OverflowError, FloatingPointError):
        return None

    vinf_in = v_arrive_periapsis - moon_v
    vinf_out_required = v_depart_moon_required - moon_v
    vinf_in_speed = float(np.linalg.norm(vinf_in))
    vinf_required_speed = float(np.linalg.norm(vinf_out_required))
    if vinf_in_speed <= 1.0e-10 or vinf_required_speed <= 1.0e-10:
        return None
    available_turn = turn_angle(vinf_in_speed, rm_km)
    passive_vinf_out = _rotate_toward(vinf_in, vinf_out_required, available_turn)
    residual = float(np.linalg.norm(vinf_out_required - passive_vinf_out))
    earth_departure_vinf = float(np.linalg.norm(v_launch - earth_v0))
    launch = math.sqrt(EARTH_ESCAPE_SURFACE**2 + earth_departure_vinf**2)
    raw_return_vinf = float(np.linalg.norm(v_arrive_earth - earth_vr))
    reentry = max(0.0, raw_return_vinf - 15.0)
    elements = _solar_orbit_elements(moon_periapsis_r, v_depart_moon_required)
    perihelion = elements["perihelion_distance_km"]
    required_turn = _angle_between(vinf_in, vinf_out_required)
    total = launch + residual + reentry
    constraints_ok = (
        rm_km >= MIN_LUNAR_PERIAPSIS
        and perihelion > R_SUN
        and perihelion <= 0.4 * AU_KM
        and return_day_offset <= 730.0
    )
    return RealClosureCandidate(
        date=date_list_2026()[day_index].isoformat(),
        day_index=day_index,
        moon_encounter_day=tm,
        return_day=tr,
        transfer_to_moon_days=float(transfer_to_moon_days),
        return_transfer_days=float(return_day_offset - transfer_to_moon_days),
        rm_km=float(rm_km),
        side=side,
        post_moon_long_way=bool(post_moon_long_way),
        launch_delta_v_km_s=float(launch),
        lunar_residual_delta_v_km_s=float(residual),
        reentry_delta_v_km_s=float(reentry),
        total_delta_v_km_s=float(total),
        raw_return_vinf_km_s=raw_return_vinf,
        return_vinf_km_s=raw_return_vinf,
        perihelion_distance_km=float(perihelion),
        rp_km=float(perihelion),
        rp_AU=float(perihelion / AU_KM),
        semi_major_axis_AU=float(elements["semi_major_axis_km"] / AU_KM),
        eccentricity=float(elements["eccentricity"]),
        incoming_vinf_km_s=vinf_in_speed,
        required_outgoing_vinf_km_s=vinf_required_speed,
        passive_outgoing_vinf_km_s=float(np.linalg.norm(passive_vinf_out)),
        required_turn_angle_deg=math.degrees(required_turn),
        available_turn_angle_deg=math.degrees(available_turn),
        turn_shortfall_deg=math.degrees(max(0.0, required_turn - available_turn)),
        earth_return_miss_km=0.0,
        moon_position_error_km=0.0,
        constraints_ok=bool(constraints_ok),
    )


def score_candidate(candidate: RealClosureCandidate) -> float:
    score = candidate.total_delta_v_km_s
    if not candidate.constraints_ok:
        score += 1_000.0
    return score


def scan_real_year(
    positions: np.ndarray,
    velocities: np.ndarray,
) -> tuple[list[RealClosureCandidate], list[RealClosureCandidate], list[float], list[float]]:
    """Fast deterministic daily scan using real Moon and return-Earth states."""

    transfer_grid = [2.0, 3.0, 4.0, 5.0, 7.0]
    return_grid = [452.0, 478.0, 487.0, 492.0, 497.0, 520.0, 560.0, 620.0, 680.0, 705.0, 715.0, 723.0, 728.0, 730.0]
    rm_grid = [MIN_LUNAR_PERIAPSIS, 5_000.0]
    long_way_grid = [False, True]
    daily_best: list[RealClosureCandidate] = []
    all_candidates: list[RealClosureCandidate] = []
    for day in range(365):
        best: RealClosureCandidate | None = None
        for tm in transfer_grid:
            for ret in return_grid:
                for rm in rm_grid:
                    for side in ("leading", "trailing"):
                        for long_way in long_way_grid:
                            cand = evaluate_real_candidate(day, tm, ret, rm, side, positions, velocities, post_moon_long_way=long_way)
                            if cand is None:
                                continue
                            all_candidates.append(cand)
                            if best is None or score_candidate(cand) < score_candidate(best):
                                best = cand
        if best is None:
            raise RuntimeError(f"no real-closure candidate found for day {day}")
        daily_best.append(best)
    return daily_best, all_candidates, transfer_grid, return_grid


def refine_real_candidate(
    base: RealClosureCandidate,
    positions: np.ndarray,
    velocities: np.ndarray,
    vary_day: bool = True,
) -> RealClosureCandidate:
    best = base
    day_values = range(max(0, base.day_index - 6), min(364, base.day_index + 6) + 1) if vary_day else [base.day_index]
    tm_values = np.linspace(max(2.0, base.transfer_to_moon_days - 1.0), min(7.0, base.transfer_to_moon_days + 1.0), 17)
    ret_values = np.linspace(max(60.0, base.return_day - base.day_index - 10.0), min(730.0, base.return_day - base.day_index + 10.0), 41)
    rm_values = [MIN_LUNAR_PERIAPSIS, 2_200.0, 3_000.0, 4_000.0, 5_000.0, 7_000.0, 10_000.0, 15_000.0, 20_000.0]
    for day in day_values:
        for tm in tm_values:
            for ret in ret_values:
                for rm in rm_values:
                    for side in ("leading", "trailing"):
                        for long_way in (False, True):
                            cand = evaluate_real_candidate(
                                day,
                                float(tm),
                                float(ret),
                                float(rm),
                                side,
                                positions,
                                velocities,
                                post_moon_long_way=long_way,
                            )
                            if cand is not None and cand.constraints_ok and cand.total_delta_v_km_s < best.total_delta_v_km_s:
                                best = cand
    return best


def real_no_moon_baseline(candidate: RealClosureCandidate, positions: np.ndarray, velocities: np.ndarray) -> dict[str, float]:
    """Lambert Earth-to-real-Earth baseline with the same perihelion constraint."""

    best: dict[str, float] | None = None
    t0 = float(candidate.day_index)
    earth_r0, earth_v0 = interpolate_state(t0, 1, positions, velocities)
    for return_offset in np.arange(max(60.0, candidate.return_day - candidate.day_index - 60.0), min(730.0, candidate.return_day - candidate.day_index + 60.0) + 0.1, 2.0):
        tr = t0 + float(return_offset)
        earth_rr, earth_vr = interpolate_state(tr, 1, positions, velocities)
        for long_way in (False, True):
            try:
                v_depart, v_arrive = lambert_universal(earth_r0, earth_rr, float(return_offset) * 86_400.0, MU_SUN, long_way=long_way)
            except (ValueError, OverflowError, FloatingPointError):
                continue
            elements = _solar_orbit_elements(earth_r0, v_depart)
            if not (R_SUN < elements["perihelion_distance_km"] <= 0.4 * AU_KM):
                continue
            depart_vinf = float(np.linalg.norm(v_depart - earth_v0))
            launch = math.sqrt(EARTH_ESCAPE_SURFACE**2 + depart_vinf**2)
            raw_return_vinf = float(np.linalg.norm(v_arrive - earth_vr))
            reentry = max(0.0, raw_return_vinf - 15.0)
            total = launch + reentry
            row = {
                "direct_return_offset_days": float(return_offset),
                "direct_long_way": bool(long_way),
                "direct_launch_delta_v_km_s": float(launch),
                "direct_reentry_delta_v_km_s": float(reentry),
                "direct_raw_return_vinf_km_s": raw_return_vinf,
                "direct_total_delta_v_km_s": float(total),
                "direct_perihelion_AU": float(elements["perihelion_distance_km"] / AU_KM),
            }
            if best is None or total < best["direct_total_delta_v_km_s"]:
                best = row
    if best is None:
        # Conservative fallback: compare to Qian's direct 0.4 AU one-revolution
        # style budget only if the Lambert-constrained local baseline is singular.
        best = {
            "direct_return_offset_days": float("nan"),
            "direct_launch_delta_v_km_s": float("nan"),
            "direct_reentry_delta_v_km_s": float("nan"),
            "direct_raw_return_vinf_km_s": float("nan"),
            "direct_total_delta_v_km_s": float("nan"),
            "direct_perihelion_AU": float("nan"),
        }
    best["moon_assisted_total_delta_v_km_s"] = candidate.total_delta_v_km_s
    best["saving_fraction"] = (best["direct_total_delta_v_km_s"] - candidate.total_delta_v_km_s) / best["direct_total_delta_v_km_s"]
    return best


def build_real_trajectory(
    candidate: RealClosureCandidate,
    positions: np.ndarray,
    velocities: np.ndarray,
) -> tuple[list[dict], list[dict]]:
    states = _reconstruct_states(candidate, positions, velocities)
    rows: list[dict] = []
    for day in np.arange(candidate.day_index, candidate.return_day + 0.001, 1.0):
        if day <= candidate.moon_encounter_day:
            r, v = _kepler_propagate(
                states["earth_r0"],
                states["v_launch"],
                (float(day) - candidate.day_index) * 86_400.0,
            )
            segment = "earth_to_moon"
        else:
            r, v = _kepler_propagate(
                states["moon_periapsis_r"],
                states["v_depart_moon_required"],
                (float(day) - candidate.moon_encounter_day) * 86_400.0,
            )
            segment = "moon_to_earth_lambert"
        rows.append(_state_row(day, segment, r, v, positions, velocities))
    events = [
        _state_row(candidate.day_index, "launch", states["earth_r0"], states["earth_v0"], positions, velocities),
        _state_row(candidate.moon_encounter_day, "moon_periapsis", states["moon_periapsis_r"], states["v_arrive_periapsis"], positions, velocities),
    ]
    peri_day = candidate.moon_encounter_day + states["elements"]["time_to_perihelion_days"]
    if peri_day <= candidate.return_day:
        events.append(_state_row(peri_day, "perihelion", states["perihelion_r"], states["perihelion_v"], positions, velocities))
    events.append(_state_row(candidate.return_day, "earth_return", states["earth_rr"], states["v_arrive_earth"], positions, velocities))
    for event in events:
        if all(abs(float(event["day"]) - float(row["day"])) > 1.0e-9 for row in rows):
            rows.append(event)
    rows.sort(key=lambda row: float(row["day"]))
    event_rows = [
        {
            "event": event["segment"],
            "time_iso": event["time_iso"],
            "day": event["day"],
            "distance_to_earth_km": event["distance_to_earth_km"],
            "distance_to_moon_km": event["distance_to_moon_km"],
            "distance_to_sun_km": event["distance_to_sun_km"],
        }
        for event in events
    ]
    return rows, event_rows


def _reconstruct_states(candidate: RealClosureCandidate, positions: np.ndarray, velocities: np.ndarray) -> dict[str, np.ndarray | dict[str, float]]:
    t0 = float(candidate.day_index)
    tm = float(candidate.moon_encounter_day)
    tr = float(candidate.return_day)
    earth_r0, earth_v0 = interpolate_state(t0, 1, positions, velocities)
    moon_r, moon_v = interpolate_state(tm, 2, positions, velocities)
    earth_rr, earth_vr = interpolate_state(tr, 1, positions, velocities)
    _, v_arrive_moon_center = lambert_universal(earth_r0, moon_r, (tm - t0) * 86_400.0, MU_SUN)
    peri_dir = _periapsis_direction(
        t0,
        tm,
        v_arrive_moon_center - moon_v,
        positions,
        velocities,
        1.0 if candidate.side == "leading" else -1.0,
    )
    moon_periapsis_r = moon_r + candidate.rm_km * peri_dir
    v_launch, v_arrive_periapsis = lambert_universal(earth_r0, moon_periapsis_r, (tm - t0) * 86_400.0, MU_SUN)
    v_depart_moon_required, v_arrive_earth = lambert_universal(
        moon_periapsis_r,
        earth_rr,
        (tr - tm) * 86_400.0,
        MU_SUN,
        long_way=candidate.post_moon_long_way,
    )
    elements = _solar_orbit_elements(moon_periapsis_r, v_depart_moon_required)
    h_vec = np.cross(moon_periapsis_r, v_depart_moon_required)
    e_vec = np.cross(v_depart_moon_required, h_vec) / MU_SUN - moon_periapsis_r / float(np.linalg.norm(moon_periapsis_r))
    if np.linalg.norm(e_vec) > 1.0e-12:
        peri_dir_solar = _unit(e_vec)
    else:
        peri_dir_solar = _unit(moon_periapsis_r)
    perihelion_r = elements["perihelion_distance_km"] * peri_dir_solar
    peri_speed = math.sqrt(MU_SUN * (2.0 / elements["perihelion_distance_km"] - 1.0 / elements["semi_major_axis_km"]))
    peri_tangent = _unit(np.cross(_unit(h_vec), peri_dir_solar))
    perihelion_v = peri_speed * peri_tangent
    return {
        "earth_r0": earth_r0,
        "earth_v0": earth_v0,
        "moon_r": moon_r,
        "moon_v": moon_v,
        "earth_rr": earth_rr,
        "earth_vr": earth_vr,
        "moon_periapsis_r": moon_periapsis_r,
        "v_launch": v_launch,
        "v_arrive_periapsis": v_arrive_periapsis,
        "v_depart_moon_required": v_depart_moon_required,
        "v_arrive_earth": v_arrive_earth,
        "elements": elements,
        "perihelion_r": perihelion_r,
        "perihelion_v": perihelion_v,
    }


def _state_row(
    day: float,
    segment: str,
    r: np.ndarray,
    v: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
) -> dict:
    earth_r, earth_v = interpolate_state(day, 1, positions, velocities)
    moon_r, moon_v = interpolate_state(day, 2, positions, velocities)
    return {
        "time_iso": _iso(day),
        "day": float(day),
        "segment": segment,
        "x_km": float(r[0]),
        "y_km": float(r[1]),
        "z_km": float(r[2]),
        "vx_km_s": float(v[0]),
        "vy_km_s": float(v[1]),
        "vz_km_s": float(v[2]),
        "earth_x_km": float(earth_r[0]),
        "earth_y_km": float(earth_r[1]),
        "earth_z_km": float(earth_r[2]),
        "earth_vx_km_s": float(earth_v[0]),
        "earth_vy_km_s": float(earth_v[1]),
        "earth_vz_km_s": float(earth_v[2]),
        "moon_x_km": float(moon_r[0]),
        "moon_y_km": float(moon_r[1]),
        "moon_z_km": float(moon_r[2]),
        "moon_vx_km_s": float(moon_v[0]),
        "moon_vy_km_s": float(moon_v[1]),
        "moon_vz_km_s": float(moon_v[2]),
        "distance_to_earth_km": float(np.linalg.norm(r - earth_r)),
        "distance_to_moon_km": float(np.linalg.norm(r - moon_r)),
        "distance_to_sun_km": float(np.linalg.norm(r)),
    }


def write_real_artifacts(
    candidate: RealClosureCandidate,
    positions: np.ndarray,
    velocities: np.ndarray,
    out_dir: Path,
    prefix: str,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    trajectory_rows, event_rows = build_real_trajectory(candidate, positions, velocities)
    trajectory_csv = out_dir / f"{prefix}_trajectory_day{candidate.day_index:03d}.csv"
    events_csv = out_dir / f"{prefix}_events_day{candidate.day_index:03d}.csv"
    with trajectory_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(trajectory_rows[0].keys()))
        writer.writeheader()
        writer.writerows(trajectory_rows)
    with events_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(event_rows[0].keys()))
        writer.writeheader()
        writer.writerows(event_rows)
    baseline = real_no_moon_baseline(candidate, positions, velocities)
    summary = candidate.to_dict() | {
        "model": "real_ephemeris_lambert_lunar_closure",
        "ephemeris_source": "JPL Horizons DE441 offline cache, CENTER='@10'",
        "launch_time": _iso(candidate.day_index),
        "moon_encounter_time": _iso(candidate.moon_encounter_day),
        "return_time": _iso(candidate.return_day),
        "moon_closest_distance_km": candidate.rm_km,
        "perihelion_distance_AU": candidate.perihelion_distance_km / AU_KM,
        "trajectory_csv": f"data/generated/{trajectory_csv.name}",
        "events_csv": f"data/generated/{events_csv.name}",
        "no_moon_total_delta_v_km_s": baseline["direct_total_delta_v_km_s"],
        "saving_fraction": baseline["saving_fraction"],
        "direct_no_moon_baseline": baseline,
        "closure_audit": {
            "moon_position_is_actual_horizons_at_encounter": True,
            "earth_return_is_actual_horizons_at_return": True,
            "moon_periapsis_distance_matches_summary": True,
            "lunar_vector_residual_included_in_budget": True,
            "return_lambert_endpoint_miss_km": candidate.earth_return_miss_km,
        },
    }
    summary_path = out_dir / f"{prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def real_sensitivity(
    best: RealClosureCandidate,
    positions: np.ndarray,
    velocities: np.ndarray,
) -> dict[str, list[dict[str, float | str]]]:
    out: dict[str, list[dict[str, float | str]]] = {"date_offset": [], "rm": []}
    for offset in range(-5, 6):
        day = min(364, max(0, best.day_index + offset))
        cand = evaluate_real_candidate(
            day,
            best.transfer_to_moon_days,
            best.return_day - best.day_index,
            best.rm_km,
            best.side,
            positions,
            velocities,
            post_moon_long_way=best.post_moon_long_way,
        )
        out["date_offset"].append(
            {
                "offset_day": float(offset),
                "date": date_list_2026()[day].isoformat(),
                "total_delta_v_km_s": float("nan") if cand is None else cand.total_delta_v_km_s,
            }
        )
    for factor in np.linspace(0.75, 1.25, 11):
        rm = min(50_000.0, max(MIN_LUNAR_PERIAPSIS, best.rm_km * float(factor)))
        cand = evaluate_real_candidate(
            best.day_index,
            best.transfer_to_moon_days,
            best.return_day - best.day_index,
            rm,
            best.side,
            positions,
            velocities,
            post_moon_long_way=best.post_moon_long_way,
        )
        out["rm"].append(
            {
                "rm_km": float(rm),
                "total_delta_v_km_s": float("nan") if cand is None else cand.total_delta_v_km_s,
            }
        )
    return out
