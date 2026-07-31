"""
physics.py

Core two-body orbital physics for the simulation.

Everything here is standard two-body (Keplerian) mechanics:
Earth is treated as a fixed point mass at the origin, and each
satellite is a massless test particle affected only by Earth's
gravity. Satellites do NOT gravitationally affect each other or
Earth's motion (Earth doesn't move in this model).

Units used throughout the whole project:
    distance : kilometers (km)
    time     : seconds (s)
    velocity : km/s
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

# Earth's standard gravitational parameter (MU = G * M_earth).
# Using MU directly (instead of G and M separately) is standard
# practice in orbital mechanics -- it's known to much higher
# precision than G and M individually.
MU_EARTH = 398600.4418  # km^3 / s^2

EARTH_RADIUS = 6371.0  # km

# Time resolution used when sampling the propagated orbit.
# 1 second gives good precision for finding closest approach
# without generating an unreasonably large array for typical
# durations (a few thousand seconds).
SAMPLE_DT = 1.0  # seconds


@dataclass
class Trajectory:
    """
    A satellite's simulated path over time, sampled at a fixed
    time step (SAMPLE_DT).

    All arrays have the same length (one entry per sampled time).
    """
    t: np.ndarray    # time values, seconds, shape (N,)
    x: np.ndarray    # x position, km, shape (N,)
    y: np.ndarray    # y position, km, shape (N,)
    vx: np.ndarray   # x velocity, km/s, shape (N,)
    vy: np.ndarray   # y velocity, km/s, shape (N,)


def gravitational_acceleration(x: float, y: float) -> tuple[float, float]:
    """
    Compute the gravitational acceleration on a satellite due to
    Earth, at position (x, y).

    This is Newton's law of gravitation for a point mass, split
    into x and y components:

        r  = sqrt(x^2 + y^2)          <- distance from Earth's center
        a  = -MU / r^2                <- magnitude, pointing at Earth
        ax = a * (x / r) = -MU * x / r^3
        ay = a * (y / r) = -MU * y / r^3

    The negative sign means the acceleration always points back
    toward the origin (Earth's center) -- gravity pulls inward.

    Returns:
        (ax, ay) in km/s^2
    """
    r = np.sqrt(x**2 + y**2)

    if r < 1.0:
        # Safety guard: avoid division by ~zero if a satellite's
        # position ever collapses to the origin (should not happen
        # in a normal orbit, but protects the integrator from NaNs).
        raise ValueError(f"Satellite distance from Earth center is unrealistically small: {r:.3f} km")

    ax = -MU_EARTH * x / r**3
    ay = -MU_EARTH * y / r**3
    return ax, ay


def _equations_of_motion(t: float, state: np.ndarray) -> np.ndarray:
    """
    The ODE right-hand side used by solve_ivp.

    state = [x, y, vx, vy]

    The derivative of position is velocity, and the derivative of
    velocity is acceleration -- this is just Newton's second law
    written as a first-order system, which is the form solve_ivp
    expects:

        dx/dt  = vx
        dy/dt  = vy
        dvx/dt = ax(x, y)
        dvy/dt = ay(x, y)
    """
    x, y, vx, vy = state
    ax, ay = gravitational_acceleration(x, y)
    return [vx, vy, ax, ay]


def propagate_orbit(satellite, duration: float) -> Trajectory:
    """
    Propagate a satellite's orbit forward in time under Earth's
    gravity alone, from t=0 to t=duration.

    Uses scipy.integrate.solve_ivp with dense_output=True, which
    fits a continuous interpolation to the solution. We then sample
    that interpolation on a uniform time grid (step = SAMPLE_DT) so
    that later modules (collision.py) can search for the closest
    approach at fine time resolution without re-running the solver.

    Args:
        satellite: a Satellite object with fields x, y, vx, vy
                    (see satellite.py)
        duration:  how many seconds forward to simulate

    Returns:
        Trajectory containing sampled t, x, y, vx, vy arrays.
    """
    if duration <= 0:
        raise ValueError("duration must be positive")

    initial_state = [satellite.x, satellite.y, satellite.vx, satellite.vy]

    solution = solve_ivp(
        fun=_equations_of_motion,
        t_span=(0.0, duration),
        y0=initial_state,
        method="RK45",
        dense_output=True,   # lets us evaluate the solution at any t afterward
        max_step=10.0,       # cap internal step size for numerical accuracy
        rtol=1e-9,
        atol=1e-9,
    )

    if not solution.success:
        raise RuntimeError(f"Orbit integration failed: {solution.message}")

    # Sample the continuous solution on a uniform time grid.
    t_samples = np.arange(0.0, duration, SAMPLE_DT)
    sampled = solution.sol(t_samples)  # shape (4, N): rows = x, y, vx, vy

    return Trajectory(
        t=t_samples,
        x=sampled[0],
        y=sampled[1],
        vx=sampled[2],
        vy=sampled[3],
    )