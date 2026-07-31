"""
collision.py

Close-approach detection and risk classification between two
satellite trajectories.

IMPORTANT — EDUCATIONAL MODEL ONLY:
This module implements a deliberately simplified "risk" model:
it compares the minimum distance between two satellites against a
fixed safety threshold and labels the result HIGH or LOW.

This is NOT how real collision risk is assessed operationally.
Real conjunction assessment (e.g. by CARA, 18th Space Defense
Squadron, or ESA) uses:
    - full 3D covariance (position uncertainty) modeling
    - probability of collision (Pc) computed from relative
      position/velocity uncertainty ellipsoids
    - combined "hard body" cross-sectional area of both objects
    - much more precise force models (see physics.py limitations)

Here, "risk" is just a distance threshold check. It is useful for
learning the *shape* of the collision-avoidance pipeline, not for
any real operational decision.
"""

from dataclasses import dataclass

import numpy as np

from physics import Trajectory


@dataclass
class ClosestApproachResult:
    """
    Result of a closest-approach search between two trajectories.

    Fields:
        min_distance: the smallest separation found, in km
        time_of_closest_approach: the simulation time (s) at which
            that minimum distance occurred
        index: the index into the trajectory arrays where the
            minimum occurred (useful for plotting the exact point)
    """
    min_distance: float
    time_of_closest_approach: float
    index: int


def relative_distance(traj_a: Trajectory, traj_b: Trajectory) -> np.ndarray:
    """
    Compute the distance between Satellite A and Satellite B at
    every sampled time step.

    Both trajectories must be sampled at the same time points
    (physics.py guarantees this, since both are propagated with
    the same SAMPLE_DT starting from t=0).

    This is simple Euclidean distance:

        distance(t) = sqrt( (xa(t) - xb(t))^2 + (ya(t) - yb(t))^2 )

    Returns:
        1D array of distances (km), same length as the trajectories.
    """
    if len(traj_a.t) != len(traj_b.t):
        raise ValueError(
            "Trajectories must be sampled at the same time points "
            f"(got lengths {len(traj_a.t)} and {len(traj_b.t)}). "
            "Make sure both were propagated with the same duration."
        )

    dx = traj_a.x - traj_b.x
    dy = traj_a.y - traj_b.y
    return np.sqrt(dx**2 + dy**2)


def find_closest_approach(traj_a: Trajectory, traj_b: Trajectory) -> tuple[float, float]:
    """
    Find the minimum distance between the two satellites over the
    full simulated time span, and the time at which it occurs.

    Method: compute the distance at every sampled time step (using
    relative_distance) and take the minimum. Because SAMPLE_DT is
    small (1 second, see physics.py), this dense sampling finds the
    true minimum to within about SAMPLE_DT of accuracy -- plenty
    precise for this MVP.

    (A more precise approach would use scipy.optimize.minimize_scalar
    on the dense_output solution between the two closest sample
    points. That's a natural extension once you're comfortable with
    this simpler version.)

    Returns:
        (min_distance_km, time_of_closest_approach_seconds)
    """
    distances = relative_distance(traj_a, traj_b)

    min_index = int(np.argmin(distances))
    min_distance = float(distances[min_index])
    time_of_closest_approach = float(traj_a.t[min_index])

    return min_distance, time_of_closest_approach


def find_closest_approach_full(traj_a: Trajectory, traj_b: Trajectory) -> ClosestApproachResult:
    """
    Same search as find_closest_approach, but also returns the
    array index of the closest point -- useful for visualization.py
    when it needs to mark the exact (x, y) point on the plot.
    """
    distances = relative_distance(traj_a, traj_b)

    min_index = int(np.argmin(distances))

    return ClosestApproachResult(
        min_distance=float(distances[min_index]),
        time_of_closest_approach=float(traj_a.t[min_index]),
        index=min_index,
    )


def classify_risk(min_distance: float, safe_distance: float) -> str:
    """
    Classify collision risk using a simple threshold comparison.

    This is the simplified educational risk model described in the
    module docstring above: if the satellites ever get closer than
    `safe_distance`, risk is labeled HIGH; otherwise LOW.

    Args:
        min_distance: minimum separation found during simulation (km)
        safe_distance: configurable safety threshold (km)

    Returns:
        "HIGH" or "LOW"
    """
    return "HIGH" if min_distance < safe_distance else "LOW"