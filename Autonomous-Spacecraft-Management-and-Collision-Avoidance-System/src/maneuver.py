"""
maneuver.py

Educational autonomous avoidance-maneuver planner.

The planner does not use machine learning. Instead, it evaluates a
small deterministic set of candidate delta-v maneuvers, propagates
each candidate orbit, measures the resulting closest approach, and
selects the maneuver that produces the greatest separation.

This is an educational decision-making demonstration, not an
operational spacecraft flight-control system.
"""

from dataclasses import dataclass, replace

import numpy as np

from satellite import Satellite
from physics import Trajectory, propagate_orbit
from collision import find_closest_approach


# Candidate delta-v magnitudes in km/s.
# 0.001 km/s = 1 m/s
# 0.002 km/s = 2 m/s
# 0.005 km/s = 5 m/s
# 0.010 km/s = 10 m/s
DEFAULT_DELTA_V_OPTIONS = (
    0.001,
    0.002,
    0.005,
    0.010,
)


@dataclass
class ManeuverResult:
    """
    Result of the autonomous maneuver search.
    """

    trajectory: Trajectory
    delta_v: float
    direction: str
    original_min_distance: float
    new_min_distance: float


def _direction_vectors(satellite: Satellite) -> dict[str, np.ndarray]:
    """
    Calculate the four candidate maneuver directions:

    radial
    anti-radial
    prograde
    retrograde
    """

    position = np.array([satellite.x, satellite.y], dtype=float)
    velocity = np.array([satellite.vx, satellite.vy], dtype=float)

    position_norm = np.linalg.norm(position)
    velocity_norm = np.linalg.norm(velocity)

    if position_norm < 1.0:
        raise ValueError(
            "Satellite distance from Earth center is unrealistically small."
        )

    if velocity_norm < 1e-12:
        raise ValueError(
            "Satellite velocity is too small to define a maneuver direction."
        )

    radial = position / position_norm
    prograde = velocity / velocity_norm

    return {
        "radial": radial,
        "anti-radial": -radial,
        "prograde": prograde,
        "retrograde": -prograde,
    }


def _apply_candidate(
    satellite: Satellite,
    direction: np.ndarray,
    delta_v: float,
) -> Satellite:
    """
    Create a new satellite state after applying a candidate delta-v.
    The original satellite object is not modified.
    """

    dv = direction * delta_v

    return replace(
        satellite,
        vx=satellite.vx + dv[0],
        vy=satellite.vy + dv[1],
    )


def find_best_avoidance_maneuver(
    satellite: Satellite,
    reference_trajectory: Trajectory,
    duration: float,
    delta_v_options: tuple[float, ...] = DEFAULT_DELTA_V_OPTIONS,
) -> ManeuverResult:
    """
    Search a deterministic set of candidate maneuvers and select
    the one that maximizes the resulting minimum separation.

    The reference trajectory is the trajectory of the other
    spacecraft, which remains unchanged during the search.
    """

    original_trajectory = propagate_orbit(satellite, duration)
    original_min_distance, _ = find_closest_approach(
        reference_trajectory,
        original_trajectory,
    )

    directions = _direction_vectors(satellite)

    best_trajectory = original_trajectory
    best_delta_v = 0.0
    best_direction = "none"
    best_min_distance = original_min_distance

    for direction_name, direction_vector in directions.items():
        for delta_v in delta_v_options:
            candidate_satellite = _apply_candidate(
                satellite,
                direction_vector,
                delta_v,
            )

            candidate_trajectory = propagate_orbit(
                candidate_satellite,
                duration,
            )

            candidate_min_distance, _ = find_closest_approach(
                reference_trajectory,
                candidate_trajectory,
            )

            if candidate_min_distance > best_min_distance:
                best_min_distance = candidate_min_distance
                best_trajectory = candidate_trajectory
                best_delta_v = delta_v
                best_direction = direction_name

    return ManeuverResult(
        trajectory=best_trajectory,
        delta_v=best_delta_v,
        direction=best_direction,
        original_min_distance=original_min_distance,
        new_min_distance=best_min_distance,
    )


def apply_avoidance_maneuver(
    satellite: Satellite,
    duration: float,
    delta_v_magnitude: float = 0.002,
) -> tuple[Trajectory, float]:
    """
    Backward-compatible simple maneuver function.

    This function is retained so existing code continues to work.
    The newer autonomous planner is find_best_avoidance_maneuver().
    """

    directions = _direction_vectors(satellite)

    # Use anti-radial direction for the simple fallback maneuver.
    direction = directions["anti-radial"]

    maneuvered_satellite = _apply_candidate(
        satellite,
        direction,
        delta_v_magnitude,
    )

    new_trajectory = propagate_orbit(
        maneuvered_satellite,
        duration,
    )

    return new_trajectory, delta_v_magnitude