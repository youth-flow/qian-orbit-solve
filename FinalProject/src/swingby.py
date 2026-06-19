"""Lunar gravity-assist formulas and a numerical two-body check."""

from __future__ import annotations

import math

import numpy as np

from constants import MOON_SOI, MU_MOON


def rot2(vec: np.ndarray, angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1]], dtype=float)


def turn_angle(vinf_km_s: float, periapsis_km: float, mu: float = MU_MOON) -> float:
    if vinf_km_s <= 0.0:
        raise ValueError("vinf_km_s must be positive")
    e = 1.0 + periapsis_km * vinf_km_s**2 / mu
    return 2.0 * math.asin(1.0 / e)


def analytic_swingby(
    v_sc_in_helio: np.ndarray,
    v_moon_helio: np.ndarray,
    periapsis_km: float,
    side: str,
    mu: float = MU_MOON,
) -> dict:
    vinf_in = np.asarray(v_sc_in_helio, dtype=float) - np.asarray(v_moon_helio, dtype=float)
    speed = float(np.linalg.norm(vinf_in))
    delta = turn_angle(speed, periapsis_km, mu)
    sign = 1.0 if side == "leading" else -1.0
    vinf_out = rot2(vinf_in, sign * delta)
    v_out = np.asarray(v_moon_helio, dtype=float) + vinf_out
    return {
        "vinf_km_s": speed,
        "turn_angle_rad": delta,
        "turn_angle_deg": math.degrees(delta),
        "vinf_in": vinf_in.tolist(),
        "vinf_out": vinf_out.tolist(),
        "v_out_helio": v_out.tolist(),
    }


def _moon_acc(r: np.ndarray, mu: float = MU_MOON) -> np.ndarray:
    norm = np.linalg.norm(r)
    return -mu * r / norm**3


def _integrate_from_periapsis(periapsis_km: float, vinf_km_s: float, sign_dt: float, dt_abs: float = 10.0):
    r = np.array([periapsis_km, 0.0], dtype=float)
    vp = math.sqrt(vinf_km_s**2 + 2.0 * MU_MOON / periapsis_km)
    v = np.array([0.0, vp], dtype=float)
    dt = sign_dt * dt_abs
    steps = 0
    while np.linalg.norm(r) < MOON_SOI and steps < 200_000:
        a0 = _moon_acc(r)
        rn = r + v * dt + 0.5 * a0 * dt * dt
        a1 = _moon_acc(rn)
        v = v + 0.5 * (a0 + a1) * dt
        r = rn
        steps += 1
    return r, v, steps


def numerical_turn_check(vinf_km_s: float = 1.2, periapsis_km: float = 5_000.0) -> dict:
    """Integrate the Moon-centered hyperbola to SOI in both directions."""

    r_back, v_back, n_back = _integrate_from_periapsis(periapsis_km, vinf_km_s, -1.0)
    r_forw, v_forw, n_forw = _integrate_from_periapsis(periapsis_km, vinf_km_s, 1.0)
    cosang = float(np.dot(v_back, v_forw) / (np.linalg.norm(v_back) * np.linalg.norm(v_forw)))
    cosang = max(-1.0, min(1.0, cosang))
    finite_angle = math.acos(cosang)
    analytic = turn_angle(vinf_km_s, periapsis_km)
    return {
        "vinf_km_s": vinf_km_s,
        "periapsis_km": periapsis_km,
        "soi_km": MOON_SOI,
        "analytic_turn_deg": math.degrees(analytic),
        "numerical_finite_soi_turn_deg": math.degrees(finite_angle),
        "absolute_difference_deg": abs(math.degrees(finite_angle - analytic)),
        "backward_steps": n_back,
        "forward_steps": n_forw,
    }
