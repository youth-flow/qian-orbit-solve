"""Physical constants used by the project.

Units are km, s and kg-derived gravitational parameters unless noted.
"""

from __future__ import annotations

AU_KM = 149_597_870.7
DAY_S = 86_400.0
YEAR_DAYS = 365.25

MU_SUN = 1.327_124_400_18e11
MU_EARTH = 398_600.4418
MU_MOON = 4_902.800066

R_EARTH = 6_378.1363
R_MOON = 1_737.4
R_SUN = 696_000.0
MIN_LUNAR_PERIAPSIS = 1_838.0

EARTH_ESCAPE_SURFACE = 11.18
MOON_SOI = 66_000.0

BODIES = ("Sun", "Earth", "Moon")
BODY_MU = {
    "Sun": MU_SUN,
    "Earth": MU_EARTH,
    "Moon": MU_MOON,
}
