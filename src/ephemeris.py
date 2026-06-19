"""Read the offline JPL Horizons cache supplied with the assignment."""

from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path

import numpy as np

from constants import BODIES


def cache_path(root: Path | None = None) -> Path:
    base = Path(__file__).resolve().parents[1] if root is None else Path(root)
    return base / "data" / "horizons_cache_2026.json"


def extended_cache_path(root: Path | None = None) -> Path:
    """Return the longer Sun-Earth-Moon cache used for return closure checks."""

    base = Path(__file__).resolve().parents[1] if root is None else Path(root)
    preferred = base / "data" / "horizons_cache_2026_2029.json"
    if preferred.exists():
        return preferred
    fallback = base / "data" / "horizons_cache_2026_2028.json"
    if fallback.exists():
        return fallback
    return cache_path(base)


def load_cache(path: str | Path | None = None) -> dict:
    p = cache_path() if path is None else Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def load_extended_cache(path: str | Path | None = None) -> dict:
    p = extended_cache_path() if path is None else Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def venus_cache_path(root: Path | None = None) -> Path:
    base = Path(__file__).resolve().parents[1] if root is None else Path(root)
    return base / "data" / "venus_horizons_cache_2026.json"


def venus_states_from_cache(path: str | Path | None = None) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return Venus Horizons states from the optional offline cache."""

    p = venus_cache_path() if path is None else Path(path)
    c = json.loads(p.read_text(encoding="utf-8"))
    epochs = c["epochs"]
    dates = [e["calendar"] for e in epochs]
    positions = np.array([[e["x_km"], e["y_km"], e["z_km"]] for e in epochs], dtype=float)
    velocities = np.array([[e["vx_km_s"], e["vy_km_s"], e["vz_km_s"]] for e in epochs], dtype=float)
    return dates, positions, velocities


def states_from_cache(cache: dict | None = None) -> tuple[list[str], np.ndarray, np.ndarray, list[str]]:
    """Return dates, positions and velocities.

    Positions have shape (n_days, 3 bodies, 3 coordinates), velocities the same.
    The cache uses heliocentric J2000 ecliptic vectors in km and km/s.
    """

    c = load_cache() if cache is None else cache
    epochs = c["bodies"]["Earth"]["epochs"]
    dates = [e["calendar"] for e in epochs]
    positions = []
    velocities = []
    for body in BODIES:
        body_epochs = c["bodies"][body]["epochs"]
        positions.append([[e["x_km"], e["y_km"], e["z_km"]] for e in body_epochs])
        velocities.append([[e["vx_km_s"], e["vy_km_s"], e["vz_km_s"]] for e in body_epochs])
    return dates, np.transpose(np.array(positions, dtype=float), (1, 0, 2)), np.transpose(
        np.array(velocities, dtype=float), (1, 0, 2)
    ), list(BODIES)


def states_from_extended_cache(path: str | Path | None = None) -> tuple[list[str], np.ndarray, np.ndarray, list[str]]:
    """Return states from the cache long enough to cover 2026 launches plus two years."""

    return states_from_cache(load_extended_cache(path))


def interpolate_state(
    day_index: float,
    body_index: int,
    positions: np.ndarray,
    velocities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate a cached state at a fractional day after 2026-01-01."""

    if day_index <= 0.0:
        return positions[0, body_index].copy(), velocities[0, body_index].copy()
    if day_index >= positions.shape[0] - 1:
        return positions[-1, body_index].copy(), velocities[-1, body_index].copy()
    lo = int(np.floor(day_index))
    frac = float(day_index - lo)
    r = (1.0 - frac) * positions[lo, body_index] + frac * positions[lo + 1, body_index]
    v = (1.0 - frac) * velocities[lo, body_index] + frac * velocities[lo + 1, body_index]
    return r, v


def date_list_2026() -> list[date]:
    start = date(2026, 1, 1)
    return [start + timedelta(days=i) for i in range(365)]
