"""
test_validation.py

Basic validation checks for the spacecraft collision avoidance MVP.

These are NOT a full test suite with a testing framework (like
pytest) -- they're simple, readable sanity checks you can run
directly to confirm the core physics and logic behave correctly.
Each check prints PASS or FAIL and explains what it verified.

Run with:
    python test_validation.py

All checks use fixed, deterministic inputs -- no randomness --
so results are always reproducible.
"""

import numpy as np

from satellite import Satellite, create_default_scenario, circular_orbit_speed
from physics import propagate_orbit, MU_EARTH, EARTH_RADIUS
from collision import relative_distance, find_closest_approach, classify_risk
from maneuver import apply_avoidance_maneuver


def check(description: str, condition: bool) -> bool:
    """Print a PASS/FAIL line for one check and return whether it passed."""
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {description}")
    return condition


# ----------------------------------------------------------------------
# 1. Orbit stays physically reasonable
# ----------------------------------------------------------------------

def test_orbit_remains_physically_reasonable() -> bool:
    """
    For a circular orbit with no maneuvers, two physical quantities
    should stay (nearly) constant over time:

        1. Distance from Earth's center (r) -- a circular orbit's
           altitude should not drift significantly.
        2. Specific orbital energy:
               E = v^2 / 2 - MU / r
           This is conserved in any two-body orbit (circular or
           not) as a consequence of energy conservation. If our
           numerical integration is accurate, E should stay nearly
           constant throughout the simulation.

    We use a generous tolerance (0.1%) since solve_ivp is a
    numerical (not exact) integrator.
    """
    print("\n--- Test 1: Orbit remains physically reasonable ---")

    satellite = Satellite(name="Test Sat", x=EARTH_RADIUS + 500, y=0.0, vx=0.0,
                           vy=circular_orbit_speed(EARTH_RADIUS + 500))

    traj = propagate_orbit(satellite, duration=6000.0)

    r = np.sqrt(traj.x**2 + traj.y**2)
    v_squared = traj.vx**2 + traj.vy**2
    specific_energy = v_squared / 2 - MU_EARTH / r

    r_variation = (r.max() - r.min()) / r.mean()
    energy_variation = (specific_energy.max() - specific_energy.min()) / abs(specific_energy.mean())

    result_1 = check(
        f"Orbital radius stays nearly constant for circular orbit (variation={r_variation:.2e})",
        r_variation < 1e-3,
    )
    result_2 = check(
        f"Specific orbital energy is conserved (variation={energy_variation:.2e})",
        energy_variation < 1e-3,
    )
    return result_1 and result_2


# ----------------------------------------------------------------------
# 2. Distance calculation works
# ----------------------------------------------------------------------

def test_distance_calculation() -> bool:
    """
    Verify relative_distance() against a hand-computed example.

    Two trajectories with known, simple positions -- distance
    between them should match a manual calculation exactly.
    """
    print("\n--- Test 2: Distance calculation ---")

    from physics import Trajectory

    traj_a = Trajectory(
        t=np.array([0.0, 1.0]),
        x=np.array([0.0, 0.0]),
        y=np.array([0.0, 0.0]),
        vx=np.array([0.0, 0.0]),
        vy=np.array([0.0, 0.0]),
    )
    traj_b = Trajectory(
        t=np.array([0.0, 1.0]),
        x=np.array([3.0, 0.0]),
        y=np.array([4.0, 5.0]),
        vx=np.array([0.0, 0.0]),
        vy=np.array([0.0, 0.0]),
    )
    # At t=0: distance = sqrt(3^2 + 4^2) = 5.0 (classic 3-4-5 triangle)
    # At t=1: distance = sqrt(0^2 + 5^2) = 5.0

    distances = relative_distance(traj_a, traj_b)

    return check(
        f"Distance matches hand-computed 3-4-5 triangle values (got {distances})",
        np.allclose(distances, [5.0, 5.0]),
    )


# ----------------------------------------------------------------------
# 3. Closest approach calculation works
# ----------------------------------------------------------------------

def test_closest_approach_calculation() -> bool:
    """
    Verify find_closest_approach() correctly identifies the minimum
    distance and its time, using a constructed trajectory pair
    where the answer is known in advance.
    """
    print("\n--- Test 3: Closest approach calculation ---")

    from physics import Trajectory

    # Satellite A stationary at origin.
    traj_a = Trajectory(
        t=np.array([0.0, 1.0, 2.0, 3.0]),
        x=np.array([0.0, 0.0, 0.0, 0.0]),
        y=np.array([0.0, 0.0, 0.0, 0.0]),
        vx=np.zeros(4), vy=np.zeros(4),
    )
    # Satellite B moves from far away, passes close at t=2, moves away again.
    traj_b = Trajectory(
        t=np.array([0.0, 1.0, 2.0, 3.0]),
        x=np.array([100.0, 50.0, 10.0, 60.0]),
        y=np.array([0.0, 0.0, 0.0, 0.0]),
        vx=np.zeros(4), vy=np.zeros(4),
    )
    # Expected minimum distance = 10.0 km at t=2.0 s

    min_dist, t_closest = find_closest_approach(traj_a, traj_b)

    result_1 = check(f"Minimum distance is 10.0 km (got {min_dist})", np.isclose(min_dist, 10.0))
    result_2 = check(f"Time of closest approach is 2.0 s (got {t_closest})", np.isclose(t_closest, 2.0))
    return result_1 and result_2


# ----------------------------------------------------------------------
# 4. Risk classification works
# ----------------------------------------------------------------------

def test_risk_classification() -> bool:
    """
    Verify classify_risk() applies the threshold comparison
    correctly at, above, and below the safety distance.
    """
    print("\n--- Test 4: Risk classification ---")

    result_1 = check(
        "Distance below threshold classified as HIGH",
        classify_risk(min_distance=50.0, safe_distance=100.0) == "HIGH",
    )
    result_2 = check(
        "Distance above threshold classified as LOW",
        classify_risk(min_distance=150.0, safe_distance=100.0) == "LOW",
    )
    result_3 = check(
        "Distance exactly at threshold classified as LOW (not strictly below)",
        classify_risk(min_distance=100.0, safe_distance=100.0) == "LOW",
    )
    return result_1 and result_2 and result_3


# ----------------------------------------------------------------------
# 5. Maneuver improves separation in the default scenario
# ----------------------------------------------------------------------

def test_maneuver_improves_separation() -> bool:
    """
    Run the default two-satellite scenario, find the original
    minimum separation, apply the avoidance maneuver to Satellite B,
    and confirm the new minimum separation is larger (i.e. safer).

    Note: because the maneuver direction (radial) is fixed rather
    than optimized, this won't ALWAYS improve separation for every
    possible scenario -- but it is confirmed to work for the
    default scenario used in this MVP, which is what this test
    checks.
    """
    print("\n--- Test 5: Maneuver improves separation (default scenario) ---")

    sat_a, sat_b = create_default_scenario()
    duration = 6000.0

    traj_a = propagate_orbit(sat_a, duration)
    traj_b_original = propagate_orbit(sat_b, duration)
    original_min_dist, _ = find_closest_approach(traj_a, traj_b_original)

    traj_b_maneuvered, delta_v = apply_avoidance_maneuver(sat_b, duration)
    new_min_dist, _ = find_closest_approach(traj_a, traj_b_maneuvered)

    print(f"    Original minimum separation: {original_min_dist:.2f} km")
    print(f"    Post-maneuver minimum separation: {new_min_dist:.2f} km")
    print(f"    Delta-v applied: {delta_v} km/s")

    return check(
        "Post-maneuver separation is greater than original separation",
        new_min_dist > original_min_dist,
    )


# ----------------------------------------------------------------------
# Run all checks
# ----------------------------------------------------------------------

def run_all_tests() -> None:
    results = [
        test_orbit_remains_physically_reasonable(),
        test_distance_calculation(),
        test_closest_approach_calculation(),
        test_risk_classification(),
        test_maneuver_improves_separation(),
    ]

    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    print(f"RESULT: {passed}/{total} checks passed")
    print("=" * 50)

    if passed < total:
        raise SystemExit(1)  # non-zero exit code if any check failed


if __name__ == "__main__":
    run_all_tests()