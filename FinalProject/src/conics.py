"""Patched-conic formulas used in Qian Xuesen's solar-return example."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from constants import AU_KM, EARTH_ESCAPE_SURFACE, MU_SUN, YEAR_DAYS


@dataclass(frozen=True)
class ConicResult:
    rp_km: float
    r1_km: float
    a_km: float
    e: float
    p_km: float
    b_km: float
    v_circular_km_s: float
    v_aphelion_km_s: float
    v_perihelion_km_s: float
    delta_v_heliocentric_km_s: float
    launch_speed_km_s: float
    period_days: float
    half_period_days: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def qian_patched_conic(
    rp_km: float,
    r1_km: float = AU_KM,
    mu_sun: float = MU_SUN,
    earth_escape_km_s: float = EARTH_ESCAPE_SURFACE,
) -> ConicResult:
    """Return Qian's two-body solar-transfer quantities.

    The departure/return point is treated as the aphelion of an ellipse with
    aphelion radius r1 and perihelion radius rp.
    """

    if not (0.0 < rp_km < r1_km):
        raise ValueError("rp_km must be between 0 and r1_km")

    a = 0.5 * (r1_km + rp_km)
    e = (r1_km - rp_km) / (r1_km + rp_km)
    p = a * (1.0 - e * e)
    b = a * math.sqrt(1.0 - e * e)
    v_circ = math.sqrt(mu_sun / r1_km)
    v_aphelion = math.sqrt(mu_sun * (2.0 / r1_km - 1.0 / a))
    v_perihelion = math.sqrt(mu_sun * (2.0 / rp_km - 1.0 / a))
    delta_v = v_aphelion - v_circ
    launch_speed = math.sqrt(earth_escape_km_s**2 + abs(delta_v) ** 2)
    period_s = 2.0 * math.pi * math.sqrt(a**3 / mu_sun)
    period_days = period_s / 86_400.0
    return ConicResult(
        rp_km=rp_km,
        r1_km=r1_km,
        a_km=a,
        e=e,
        p_km=p,
        b_km=b,
        v_circular_km_s=v_circ,
        v_aphelion_km_s=v_aphelion,
        v_perihelion_km_s=v_perihelion,
        delta_v_heliocentric_km_s=delta_v,
        launch_speed_km_s=launch_speed,
        period_days=period_days,
        half_period_days=0.5 * period_days,
    )


def qian_reference_error() -> dict[str, float]:
    """Compare the r_p=0.2 AU case with the values in the starter report."""

    result = qian_patched_conic(0.2 * AU_KM)
    reference = {
        "a_AU": 0.6,
        "e": 0.6667,
        # The report table rounds this to 0.333 AU; the exact formula gives 1/3.
        "p_AU": 1.0 / 3.0,
        "v_aphelion_km_s": 17.21,
        "delta_v_heliocentric_km_s": -12.59,
        "launch_speed_km_s": 16.84,
        "period_days": 169.8,
        "half_period_days": 84.9,
    }
    computed = {
        "a_AU": result.a_km / AU_KM,
        "e": result.e,
        "p_AU": result.p_km / AU_KM,
        "v_aphelion_km_s": result.v_aphelion_km_s,
        "delta_v_heliocentric_km_s": result.delta_v_heliocentric_km_s,
        "launch_speed_km_s": result.launch_speed_km_s,
        "period_days": result.period_days,
        "half_period_days": result.half_period_days,
    }
    errors = {}
    for key, ref in reference.items():
        errors[key] = abs((computed[key] - ref) / ref) if ref else abs(computed[key])
    errors["max_relative_error"] = max(errors.values())
    return {
        "reference": reference,
        "computed": computed,
        "relative_errors": errors,
    }


def transfer_speed_at_earth(rp_km: float, r_km: float = AU_KM) -> float:
    """Aphelion speed for an inner solar ellipse with aphelion r_km."""

    return qian_patched_conic(rp_km, r_km).v_aphelion_km_s


def transfer_period_days(rp_km: float, r_km: float = AU_KM) -> float:
    return qian_patched_conic(rp_km, r_km).period_days
