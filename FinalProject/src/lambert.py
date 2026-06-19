"""Universal-variable Lambert solver used for optional verification.

This module is intentionally independent of external optimization libraries.
The main mission search remains the semi-analytic patched-conic model, while
this solver gives the project a self-contained differential-correction block.
"""

from __future__ import annotations

import math

import numpy as np


def stumpff_c(z: float) -> float:
    if z > 1.0e-8:
        s = math.sqrt(z)
        return (1.0 - math.cos(s)) / z
    if z < -1.0e-8:
        s = math.sqrt(-z)
        return (math.cosh(s) - 1.0) / (-z)
    return 0.5 - z / 24.0 + z * z / 720.0


def stumpff_s(z: float) -> float:
    if z > 1.0e-8:
        s = math.sqrt(z)
        return (s - math.sin(s)) / (s**3)
    if z < -1.0e-8:
        s = math.sqrt(-z)
        return (math.sinh(s) - s) / (s**3)
    return 1.0 / 6.0 - z / 120.0 + z * z / 5040.0


def lambert_universal(r1: np.ndarray, r2: np.ndarray, tof_s: float, mu: float, long_way: bool = False) -> tuple[np.ndarray, np.ndarray]:
    r1 = np.asarray(r1, dtype=float)
    r2 = np.asarray(r2, dtype=float)
    r1n = float(np.linalg.norm(r1))
    r2n = float(np.linalg.norm(r2))
    cos_dtheta = float(np.dot(r1, r2) / (r1n * r2n))
    cos_dtheta = max(-1.0, min(1.0, cos_dtheta))
    cross_z = float(np.cross(r1, r2)[2])
    sin_abs = math.sqrt(max(0.0, 1.0 - cos_dtheta * cos_dtheta))
    sin_dtheta = -sin_abs if long_way else sin_abs
    if cross_z < 0.0:
        sin_dtheta *= -1.0
    if abs(sin_dtheta) < 1.0e-10:
        raise ValueError("Lambert geometry is nearly singular")
    a_param = sin_dtheta * math.sqrt(r1n * r2n / (1.0 - cos_dtheta))

    def y_of_z(z: float) -> float:
        c = stumpff_c(z)
        s = stumpff_s(z)
        if c <= 0.0:
            return float("nan")
        return r1n + r2n + a_param * (z * s - 1.0) / math.sqrt(c)

    def t_of_z(z: float) -> float:
        c = stumpff_c(z)
        s = stumpff_s(z)
        y = y_of_z(z)
        if c <= 0.0 or y < 0.0:
            return float("nan")
        return ((y / c) ** 1.5 * s + a_param * math.sqrt(y)) / math.sqrt(mu)

    z_low = -4.0 * math.pi * math.pi
    z_high = 4.0 * math.pi * math.pi
    for _ in range(80):
        t = t_of_z(z_low)
        if math.isfinite(t) and t <= tof_s:
            break
        z_low *= 0.5
    for _ in range(80):
        t = t_of_z(z_high)
        if math.isfinite(t) and t >= tof_s:
            break
        z_high *= 2.0
    z = 0.0
    for _ in range(120):
        z = 0.5 * (z_low + z_high)
        t = t_of_z(z)
        if not math.isfinite(t):
            z_low = z
            continue
        if abs(t - tof_s) / max(1.0, tof_s) < 1.0e-9:
            break
        if t < tof_s:
            z_low = z
        else:
            z_high = z
    y = y_of_z(z)
    f = 1.0 - y / r1n
    g = a_param * math.sqrt(y / mu)
    gdot = 1.0 - y / r2n
    if abs(g) < 1.0e-12:
        raise ValueError("Lambert solver produced singular g")
    return (r2 - f * r1) / g, (gdot * r2 - r1) / g


def validation_case(mu: float = 1.0) -> dict[str, float]:
    r1 = np.array([1.0, 0.0, 0.0])
    r2 = np.array([0.0, 1.0, 0.0])
    tof = 0.5 * math.pi
    v1, v2 = lambert_universal(r1, r2, tof, mu)
    return {
        "tof_s": tof,
        "v1_norm": float(np.linalg.norm(v1)),
        "v2_norm": float(np.linalg.norm(v2)),
        "unit_circle_speed_error": abs(float(np.linalg.norm(v1)) - 1.0),
    }
