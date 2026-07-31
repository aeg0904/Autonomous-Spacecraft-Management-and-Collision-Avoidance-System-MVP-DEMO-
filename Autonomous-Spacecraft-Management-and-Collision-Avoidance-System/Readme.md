# Autonomous Spacecraft Management and Collision Avoidance System (Educational MVP)

## 1. What this project does

This is a desktop Python application that simulates two satellites
orbiting Earth, predicts their future positions, detects how close
they come to each other, classifies a simplified "collision risk,"
and demonstrates a basic hypothetical avoidance maneuver — all
visualized in a Matplotlib plot embedded in a Tkinter GUI.

The pipeline implemented is:

**This system does not control a real spacecraft.** It is a simulation
only, intended for learning orbital mechanics and simulation-engineering
concepts.

## 2. Why this project exists

This project exists as a learning tool for understanding, hands-on, how
the building blocks of real spacecraft conjunction-assessment and
collision-avoidance systems fit together: numerical orbit propagation,
relative-distance tracking over time, threshold-based risk flagging, and
maneuver planning — without the complexity of a real operational system.

## 3. Simplifications and assumptions

To keep this an approachable MVP, several simplifications are made:

- **2D motion only** (x, y), not full 3D. Real orbits are 3D; this
  keeps the visualization and math easier to follow while learning.
- **Two-body physics only** — only Earth's gravity acts on each
  satellite. Satellites don't affect each other gravitationally
  (correct, since satellite mass is negligible compared to Earth's).
- **Point-mass Earth** — no atmosphere, no oblateness (J2), no other
  perturbing forces.
- **Instantaneous maneuvers** — the avoidance delta-v is applied all
  at once at t=0, not as a real finite-duration engine burn.
- **Threshold-based risk model** — "risk" just means "did the satellites
  get closer than X km," not a real probability calculation.

See section 11 ("Important limitations") for the full list.

## 4. Physics model

Each satellite is represented by a state vector `[x, y, vx, vy]`
(position in km, velocity in km/s). Earth sits at the origin.

Gravitational acceleration on a satellite at position `(x, y)`:

where `MU = 398600.4418 km^3/s^2` is Earth's standard gravitational
parameter.

This is combined into a first-order ODE system (`[x, y, vx, vy]` and
their derivatives `[vx, vy, ax, ay]`) and integrated forward in time
using `scipy.integrate.solve_ivp` (RK45 method, dense output, tight
tolerances). See `src/physics.py` for the exact implementation — all
the math is written out explicitly rather than hidden inside a library
call.

## 5. Project architecture


Each module has one clear responsibility. `main.py` contains no physics
or math — it only calls functions from the other modules and displays
results.

## 6. Installation

Requirements: Python 3.12+ (Tkinter is included with standard Python on
Windows).

1. Clone or copy this project folder.
2. Open it in VS Code.
3. (Recommended) Create a virtual environment: 
python -m venv venv
venv\Scripts\activate
4. Install dependencies:
pip install -r requirements.txt


## 7. How to run

From the project root, in VS Code's terminal:
cd src
python main.py
This opens the GUI with a default two-satellite scenario already
loaded. Click **Run Simulation** to propagate both orbits and see the
closest approach. Click **Run Avoidance Maneuver** to apply a
hypothetical delta-v to Satellite B and compare the result. Click
**Reset** to return to the default scenario.

To run the validation checks instead:
cd src
python test_validation.py


This prints PASS/FAIL for each check and does not open the GUI.

## 8. How to change satellite parameters

Satellite initial conditions are defined in `src/satellite.py`, in
`create_default_scenario()`. Each satellite is a `Satellite` object
with:

- `name` — label shown in the GUI/plot legend
- `x, y` — initial position in km
- `vx, vy` — initial velocity in km/s

You can change the default altitudes (`altitude_a`, `altitude_b`) or
directions of motion (sign of `vy`), or construct entirely custom
starting states (position and velocity don't have to describe a
circular orbit — any valid `[x, y, vx, vy]` state works).

The `circular_orbit_speed(radius)` helper function computes the speed
needed for a circular orbit at a given radius, using `v = sqrt(MU/r)`,
if you want to build new circular-orbit scenarios.

## 9. How collision risk is calculated

At every sampled time step (1-second resolution over the simulation
duration), the distance between Satellite A and Satellite B is
computed:
distance(t) = sqrt( (xa(t) - xb(t))^2 + (ya(t) - yb(t))^2 )

The minimum distance across all time steps, and the time at which it
occurs, are found (`collision.py`, `find_closest_approach()`). If that
minimum distance is below the configurable safety threshold (default
100 km, adjustable in the GUI), risk is classified as **HIGH**;
otherwise **LOW**.

**This is a simplified educational risk model** — a real operational
system would compute an actual collision probability from position
uncertainty (covariance), not just compare a single distance value
against a threshold.

## 10. How the simplified maneuver works

If you click **Run Avoidance Maneuver**, a small fixed delta-v
(0.01 km/s by default) is applied to Satellite B's velocity at t=0, in
the **radial direction** (along the line from Earth's center to the
satellite — i.e. nudging it slightly toward or away from Earth rather
than speeding it up or slowing it down along its path). Satellite B's
orbit is then re-propagated from this new state, and the closest
approach is recalculated so you can compare:
BEFORE MANEUVER: minimum separation = X km
AFTER MANEUVER: minimum separation = Y km

This is labeled throughout the code as an **"Educational simplified
avoidance maneuver."** It is not optimized, not fuel-aware, and not
guaranteed to improve separation in every possible scenario — it's
verified to work for this project's default scenario (see
`test_validation.py`, Test 5).

## 11. Important limitations

This MVP is a simplified educational tool. It does **NOT** model:

- Earth's atmosphere (atmospheric drag)
- J2 perturbation (Earth's oblateness)
- Solar radiation pressure
- Third-body gravity (Sun, Moon)
- Real TLE/SGP4 propagation (the industry-standard method for
  propagating real satellite orbits from tracking data)
- Uncertainty/covariance in position or velocity
- Real collision probability (Pc) calculations
- Spacecraft attitude (orientation)
- Real thrusters or propulsion constraints
- A real onboard computer (OBC) or flight software
- Real-time autonomous spacecraft control

It exists purely to demonstrate the **shape** of a collision-avoidance
pipeline — orbit propagation, close-approach detection, risk
classification, and maneuver comparison — using real two-body physics
math, not to produce operationally trustworthy results.

## 12. Possible future improvements

- Extend to full 3D orbital motion.
- Add J2 perturbation for more realistic long-term orbit behavior.
- Use `scipy.optimize.minimize_scalar` to refine the closest-approach
  time beyond the sampling resolution.
- Replace the fixed-threshold risk model with a real probability-of-
  collision calculation using position covariance.
- Optimize maneuver direction/magnitude (e.g. minimize delta-v subject
  to a minimum-separation constraint) instead of using a fixed nudge.
- Support more than two satellites simultaneously.
- Load real satellite orbital data via TLE + SGP4 instead of
  hand-specified circular orbits.