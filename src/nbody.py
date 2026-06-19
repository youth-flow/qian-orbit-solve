"""N-body acceleration and Velocity-Verlet propagation."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from constants import BODY_MU, BODIES


def accelerations(positions: np.ndarray, mus: np.ndarray, gravitating: np.ndarray | None = None) -> np.ndarray:
    """Pairwise Newtonian accelerations.

    `mus[j]` is the gravitational parameter of source body j. Bodies with
    zero source parameter act as massless test particles.
    """

    r = np.asarray(positions, dtype=float)
    mu = np.asarray(mus, dtype=float)
    source_mask = np.ones(len(mu), dtype=bool) if gravitating is None else np.asarray(gravitating, dtype=bool)
    acc = np.zeros_like(r, dtype=float)
    for i in range(len(mu)):
        for j in range(len(mu)):
            if i == j or not source_mask[j] or mu[j] == 0.0:
                continue
            d = r[j] - r[i]
            norm = np.linalg.norm(d)
            acc[i] += mu[j] * d / norm**3
    return acc


def velocity_verlet_step(
    positions: np.ndarray,
    velocities: np.ndarray,
    dt: float,
    mus: np.ndarray,
    accel_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] = accelerations,
) -> tuple[np.ndarray, np.ndarray]:
    a0 = accel_fn(positions, mus)
    r1 = positions + velocities * dt + 0.5 * a0 * dt * dt
    a1 = accel_fn(r1, mus)
    v1 = velocities + 0.5 * (a0 + a1) * dt
    return r1, v1


def propagate(
    positions: np.ndarray,
    velocities: np.ndarray,
    mus: np.ndarray,
    dt: float,
    steps: int,
    sample_every: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    samples_r = []
    samples_v = []
    r = np.array(positions, dtype=float)
    v = np.array(velocities, dtype=float)
    for step in range(steps + 1):
        if step % sample_every == 0:
            samples_r.append(r.copy())
            samples_v.append(v.copy())
        if step == steps:
            break
        r, v = velocity_verlet_step(r, v, dt, mus)
    return np.array(samples_r), np.array(samples_v)


def mechanical_energy(positions: np.ndarray, velocities: np.ndarray, mus: np.ndarray, masses: np.ndarray) -> float:
    """Total Newtonian mechanical energy for massive bodies."""

    r = np.asarray(positions, dtype=float)
    v = np.asarray(velocities, dtype=float)
    m = np.asarray(masses, dtype=float)
    mu = np.asarray(mus, dtype=float)
    kinetic = 0.5 * np.sum(m[:, None] * v * v)
    potential = 0.0
    g = 6.67430e-20
    for i in range(len(m)):
        for j in range(i + 1, len(m)):
            if m[i] == 0.0 or m[j] == 0.0:
                continue
            potential -= g * m[i] * m[j] / np.linalg.norm(r[j] - r[i])
    return kinetic + potential


def validate_two_body(dt_years: float = 1.0 / 4000.0) -> dict[str, float]:
    """Dimensionless circular-orbit benchmark with mu=4*pi^2."""

    mu = 4.0 * math.pi * math.pi
    r = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float)
    v = np.array([[0.0, 0.0], [0.0, 2.0 * math.pi]], dtype=float)
    mus = np.array([mu, 0.0], dtype=float)
    steps = round(1.0 / dt_years)
    rr, vv = propagate(r, v, mus, dt_years, steps, sample_every=steps)
    final = rr[-1, 1]
    error = np.linalg.norm(final - np.array([1.0, 0.0]))
    return {
        "dt_years": dt_years,
        "steps": float(steps),
        "final_x": float(final[0]),
        "final_y": float(final[1]),
        "position_relative_error": float(error),
        "requirement": 1.0e-4,
        "passed": bool(error <= 1.0e-4),
    }


def validate_horizons(positions: np.ndarray, velocities: np.ndarray, dt: float = 900.0) -> dict:
    """Propagate Sun-Earth-Moon and compare daily heliocentric residuals."""

    mus = np.array([BODY_MU[b] for b in BODIES], dtype=float)
    days = positions.shape[0] - 1
    sample_every = round(86_400.0 / dt)
    steps = days * sample_every
    rr, vv = propagate(positions[0], velocities[0], mus, dt, steps, sample_every=sample_every)

    # Horizons cache is heliocentric. The integrated Sun is allowed to move
    # under Earth/Moon gravity, so transform the numerical solution back to
    # the instantaneous simulated solar center before comparison.
    rr_helio = rr - rr[:, 0:1, :]
    vv_helio = vv - vv[:, 0:1, :]
    pos_res = np.linalg.norm(rr_helio - positions, axis=2)
    vel_res = np.linalg.norm(vv_helio - velocities, axis=2)
    pairs = {
        "Moon-Earth": (2, 1),
        "Moon-Sun": (2, 0),
        "Earth-Sun": (1, 0),
    }
    rel_pos_res = {}
    rel_vel_res = {}
    for name, (i, j) in pairs.items():
        num_r = rr_helio[:, i, :] - rr_helio[:, j, :]
        ref_r = positions[:, i, :] - positions[:, j, :]
        num_v = vv_helio[:, i, :] - vv_helio[:, j, :]
        ref_v = velocities[:, i, :] - velocities[:, j, :]
        rel_pos_res[name] = np.linalg.norm(num_r - ref_r, axis=1)
        rel_vel_res[name] = np.linalg.norm(num_v - ref_v, axis=1)
    return {
        "dt_s": dt,
        "days": days,
        "body_names": list(BODIES),
        "max_position_residual_km": {BODIES[i]: float(np.max(pos_res[:, i])) for i in range(len(BODIES))},
        "max_velocity_residual_km_s": {BODIES[i]: float(np.max(vel_res[:, i])) for i in range(len(BODIES))},
        "max_relative_position_residual_km": {k: float(np.max(v)) for k, v in rel_pos_res.items()},
        "max_relative_velocity_residual_km_s": {k: float(np.max(v)) for k, v in rel_vel_res.items()},
        "all_position_residuals_le_6000_km": bool(np.max(pos_res) <= 6000.0),
        "all_relative_position_residuals_le_6000_km": bool(max(float(np.max(v)) for v in rel_pos_res.values()) <= 6000.0),
        "daily_position_residuals_km": pos_res.tolist(),
        "daily_velocity_residuals_km_s": vel_res.tolist(),
        "daily_relative_position_residuals_km": {k: v.tolist() for k, v in rel_pos_res.items()},
        "daily_relative_velocity_residuals_km_s": {k: v.tolist() for k, v in rel_vel_res.items()},
    }
