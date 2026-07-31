"""
visualization.py

Builds the Matplotlib figure showing Earth, both satellite orbits,
their positions, and the closest-approach point.

This module only builds and returns a matplotlib.figure.Figure --
it does not know about Tkinter. main.py is responsible for embedding
the returned Figure into the GUI window.
"""

from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Circle

from satellite import Satellite
from physics import Trajectory, EARTH_RADIUS
from collision import find_closest_approach_full


def build_figure(
    sat_a: Satellite,
    sat_b: Satellite,
    traj_a: Optional[Trajectory],
    traj_b: Optional[Trajectory],
    safe_distance: float,
    t_closest: Optional[float] = None,
) -> Figure:
    """
    Build the full visualization figure.

    Args:
        sat_a, sat_b: satellite objects (used for initial-position
            markers even before a simulation has been run)
        traj_a, traj_b: propagated trajectories, or None if no
            simulation has been run yet (e.g. right after Reset)
        safe_distance: current safety threshold in km, used to draw
            a reference circle around the closest-approach point
        t_closest: time of closest approach in seconds, if known
            (only used for the info text; the marker position is
            recomputed from the trajectories directly)

    Returns:
        A matplotlib Figure ready to be embedded in Tkinter.
    """
    fig = Figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(111)

    _draw_earth(ax)

    if traj_a is not None:
        _draw_orbit(ax, traj_a, color="tab:blue", label=sat_a.name)
    _draw_initial_marker(ax, sat_a, color="tab:blue")

    if traj_b is not None:
        _draw_orbit(ax, traj_b, color="tab:orange", label=sat_b.name)
    _draw_initial_marker(ax, sat_b, color="tab:orange")

    min_distance = None
    if traj_a is not None and traj_b is not None:
        min_distance = _draw_closest_approach(ax, traj_a, traj_b, safe_distance)

    _finalize_axes(ax)
    _add_info_text(fig, min_distance, t_closest, safe_distance)

    return fig


def _draw_earth(ax) -> None:
    """Draw Earth as a filled circle at the origin, to scale in km."""
    earth = Circle(
        (0, 0),
        EARTH_RADIUS,
        facecolor="#2b6cb0",
        edgecolor="black",
        linewidth=0.5,
        label="Earth",
        zorder=1,
    )
    ax.add_patch(earth)


def _draw_orbit(ax, traj: Trajectory, color: str, label: str) -> None:
    """Plot a satellite's full path as a line."""
    ax.plot(traj.x, traj.y, color=color, linewidth=1.2, label=f"{label} orbit", zorder=2)


def _draw_initial_marker(ax, satellite: Satellite, color: str) -> None:
    """Mark a satellite's initial (t=0) position with a dot."""
    ax.plot(
        satellite.x, satellite.y,
        marker="o", markersize=6,
        color=color, markeredgecolor="black", markeredgewidth=0.5,
        linestyle="none",
        label=f"{satellite.name} start",
        zorder=3,
    )


def _draw_closest_approach(ax, traj_a: Trajectory, traj_b: Trajectory, safe_distance: float) -> float:
    """
    Mark the closest-approach point (midpoint between the two
    satellites at the moment of minimum separation) and draw a
    dashed circle showing the safety-distance threshold around it.

    Returns the minimum distance found (km), so the caller can
    display it in the info text without recomputing it separately.
    """
    result = find_closest_approach_full(traj_a, traj_b)
    idx = result.index

    ax_pos = (traj_a.x[idx], traj_a.y[idx])
    bx_pos = (traj_b.x[idx], traj_b.y[idx])
    midpoint = ((ax_pos[0] + bx_pos[0]) / 2, (ax_pos[1] + bx_pos[1]) / 2)

    # Line connecting the two satellites at closest approach.
    ax.plot(
        [ax_pos[0], bx_pos[0]], [ax_pos[1], bx_pos[1]],
        color="red", linewidth=1.0, linestyle="--",
        label="Closest approach", zorder=4,
    )
    ax.plot(*midpoint, marker="x", markersize=8, color="red", zorder=5)

    # Safety-distance reference circle, centered on the midpoint.
    # This gives a visual sense of scale for the threshold relative
    # to the actual gap between the satellites.
    safety_circle = Circle(
        midpoint,
        safe_distance,
        facecolor="none",
        edgecolor="red",
        linestyle=":",
        linewidth=1.0,
        label=f"Safety threshold ({safe_distance:.0f} km)",
        zorder=4,
    )
    ax.add_patch(safety_circle)

    return result.min_distance


def _finalize_axes(ax) -> None:
    """Apply shared axis formatting: labels, title, grid, aspect ratio."""
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title("Satellite Orbits and Closest Approach (Educational Simulation)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(loc="upper right", fontsize=8)


def _add_info_text(fig: Figure, min_distance: Optional[float], t_closest: Optional[float], safe_distance: float) -> None:
    """
    Add a small text block to the figure with the key numeric
    results, so the plot is self-explanatory even as a standalone
    image (e.g. if screenshotted or saved).
    """
    if min_distance is None:
        info = "Run a simulation to compute closest approach."
    else:
        risk = "HIGH" if min_distance < safe_distance else "LOW"
        info = (
            f"Min separation: {min_distance:.2f} km\n"
            f"Time of closest approach: {t_closest:.1f} s\n"
            f"Safety threshold: {safe_distance:.0f} km\n"
            f"Risk (educational model): {risk}"
        )

    fig.text(
        0.02, 0.02, info,
        fontsize=8, family="monospace",
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, edgecolor="gray"),
    )