"""
satellite.py

Defines the Satellite data structure and the default two-satellite
scenario used by the MVP.

A satellite's *state* is fully described by 4 numbers:
    x, y   -- position in km (measured from Earth's center)
    vx, vy -- velocity in km/s

This is called the "state vector" in orbital mechanics. Everything
about the satellite's future path is determined by this state plus
Earth's gravity (see physics.py).
"""

from dataclasses import dataclass

import numpy as np

from physics import MU_EARTH, EARTH_RADIUS


@dataclass
class Satellite:
    """
    A single satellite's identity and current state.

    Fields:
        name: human-readable label, used in the GUI and plot legend
        x, y: position in km
        vx, vy: velocity in km/s
    """
    name: str
    x: float
    y: float
    vx: float
    vy: float


def circular_orbit_speed(radius: float) -> float:
    """
    Compute the speed (km/s) needed for a perfectly circular orbit
    at the given radius (km from Earth's center).

    Derivation: for a circular orbit, gravity provides exactly the
    centripetal force needed to keep the satellite on its circular
    path:

        MU * m / r^2  =  m * v^2 / r      (gravity = centripetal force)

    Solving for v:

        v = sqrt(MU / r)

    This is a standard, well-known orbital mechanics formula --
    useful for constructing realistic starting scenarios.
    """
    if radius <= EARTH_RADIUS:
        raise ValueError(
            f"Orbit radius ({radius} km) must be greater than Earth's radius ({EARTH_RADIUS} km)."
        )
    return float(np.sqrt(MU_EARTH / radius))


def create_default_scenario() -> tuple[Satellite, Satellite]:
    """
    Build a deterministic close-approach scenario.

    Satellite A and B share the same circular orbital radius but
    start with a small angular separation and move in opposite
    directions. This creates a future close approach without
    placing the satellites at the same position at t=0.

    This is an educational simulation scenario, not a real
    spacecraft conjunction.
    """

    altitude = 500.0
    radius = EARTH_RADIUS + altitude
    speed = circular_orbit_speed(radius)

    # Satellite A starts at angle 0 degrees.
    satellite_a = Satellite(
        name="Satellite A",
        x=radius,
        y=0.0,
        vx=0.0,
        vy=speed,
    )

    # Satellite B starts 10 degrees ahead, moving in the
    # opposite direction.
    phase_angle = np.deg2rad(10.0)

    satellite_b = Satellite(
        name="Satellite B",
        x=radius * np.cos(phase_angle),
        y=radius * np.sin(phase_angle),
        vx=0.0,
        vy=-speed,
    )

    return satellite_a, satellite_b