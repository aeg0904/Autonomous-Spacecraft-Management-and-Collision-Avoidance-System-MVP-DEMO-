"""
main.py

Entry point for the Autonomous Spacecraft Management and
Collision Avoidance System (educational MVP).
"""

import tkinter as tk
from tkinter import ttk, messagebox

from satellite import create_default_scenario
from physics import propagate_orbit
from collision import find_closest_approach, classify_risk
from maneuver import find_best_avoidance_maneuver
from visualization import build_figure

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class SpacecraftApp:
    """Main application window."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(
            "Autonomous Spacecraft Collision Avoidance (Educational MVP)"
        )

        self.sat_a = None
        self.sat_b = None

        self.traj_a = None
        self.traj_b = None

        self.original_min_dist = None
        self.maneuvered_min_dist = None
        self.delta_v_used = None
        self.maneuver_direction = None

        self._build_input_panel()
        self._build_results_panel()
        self._build_plot_area()

        self.reset_simulation()

    # --------------------------------------------------------------
    # GUI construction
    # --------------------------------------------------------------

    def _build_input_panel(self) -> None:
        """Build simulation controls."""

        frame = ttk.Frame(self.root, padding=10)
        frame.grid(row=0, column=0, sticky="n")

        ttk.Label(
            frame,
            text="Simulation duration (s):"
        ).grid(row=0, column=0, sticky="w")

        self.duration_var = tk.StringVar(value="6000")

        ttk.Entry(
            frame,
            textvariable=self.duration_var,
            width=12,
        ).grid(row=0, column=1)

        ttk.Label(
            frame,
            text="Safety distance (km):"
        ).grid(row=1, column=0, sticky="w")

        self.safe_dist_var = tk.StringVar(value="100")

        ttk.Entry(
            frame,
            textvariable=self.safe_dist_var,
            width=12,
        ).grid(row=1, column=1)

        ttk.Button(
            frame,
            text="Run Simulation",
            command=self.run_simulation,
        ).grid(
            row=2,
            column=0,
            columnspan=2,
            pady=(10, 2),
            sticky="ew",
        )

        ttk.Button(
            frame,
            text="Run Autonomous Avoidance",
            command=self.run_maneuver,
        ).grid(
            row=3,
            column=0,
            columnspan=2,
            pady=2,
            sticky="ew",
        )

        ttk.Button(
            frame,
            text="Reset",
            command=self.reset_simulation,
        ).grid(
            row=4,
            column=0,
            columnspan=2,
            pady=2,
            sticky="ew",
        )

    def _build_results_panel(self) -> None:
        """Build the results display."""

        frame = ttk.Frame(self.root, padding=10)
        frame.grid(row=1, column=0, sticky="n")

        self.results_text = tk.StringVar(
            value="Run a simulation to see results."
        )

        ttk.Label(
            frame,
            textvariable=self.results_text,
            justify="left",
            wraplength=300,
        ).grid(row=0, column=0, sticky="w")

    def _build_plot_area(self) -> None:
        """Build the Matplotlib container."""

        self.plot_frame = ttk.Frame(
            self.root,
            padding=10,
        )

        self.plot_frame.grid(
            row=0,
            column=1,
            rowspan=2,
            sticky="nsew",
        )

        self.canvas = None

        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

    # --------------------------------------------------------------
    # Simulation actions
    # --------------------------------------------------------------

    def reset_simulation(self) -> None:
        """Reset the application to the default scenario."""

        self.sat_a, self.sat_b = create_default_scenario()

        self.traj_a = None
        self.traj_b = None

        self.original_min_dist = None
        self.maneuvered_min_dist = None
        self.delta_v_used = None
        self.maneuver_direction = None

        self.results_text.set(
            "Scenario reset.\n"
            "Click 'Run Simulation'."
        )

        self._update_plot()

    def run_simulation(self) -> None:
        """Run the baseline two-satellite simulation."""

        try:
            duration = float(self.duration_var.get())
            safe_distance = float(self.safe_dist_var.get())

            if duration <= 0:
                raise ValueError

            if safe_distance <= 0:
                raise ValueError

        except ValueError:
            messagebox.showerror(
                "Invalid input",
                "Duration and safety distance must be positive numbers.",
            )
            return

        try:
            self.traj_a = propagate_orbit(
                self.sat_a,
                duration,
            )

            self.traj_b = propagate_orbit(
                self.sat_b,
                duration,
            )

            min_dist, t_closest = find_closest_approach(
                self.traj_a,
                self.traj_b,
            )

        except Exception as exc:
            messagebox.showerror(
                "Simulation error",
                str(exc),
            )
            return

        risk = classify_risk(
            min_dist,
            safe_distance,
        )

        self.original_min_dist = min_dist
        self.maneuvered_min_dist = None
        self.delta_v_used = None
        self.maneuver_direction = None

        self._show_results(
            min_dist=min_dist,
            t_closest=t_closest,
            risk=risk,
            maneuver_applied=False,
        )

        self._update_plot(
            safe_distance=safe_distance,
            t_closest=t_closest,
        )

    def run_maneuver(self) -> None:
        """
        Run the autonomous maneuver planner.

        The planner evaluates multiple candidate directions and
        delta-v magnitudes and selects the candidate that produces
        the greatest minimum separation.
        """

        if self.traj_a is None or self.traj_b is None:
            messagebox.showinfo(
                "Run simulation first",
                "Please click 'Run Simulation' before running "
                "the autonomous avoidance planner.",
            )
            return

        try:
            duration = float(self.duration_var.get())
            safe_distance = float(self.safe_dist_var.get())

            if duration <= 0:
                raise ValueError

            if safe_distance <= 0:
                raise ValueError

        except ValueError:
            messagebox.showerror(
                "Invalid input",
                "Duration and safety distance must be positive numbers.",
            )
            return

        try:
            result = find_best_avoidance_maneuver(
                satellite=self.sat_b,
                reference_trajectory=self.traj_a,
                duration=duration,
            )

        except Exception as exc:
            messagebox.showerror(
                "Maneuver planning error",
                str(exc),
            )
            return

        self.traj_b = result.trajectory

        self.original_min_dist = result.original_min_distance
        self.maneuvered_min_dist = result.new_min_distance
        self.delta_v_used = result.delta_v
        self.maneuver_direction = result.direction

        min_dist, t_closest = find_closest_approach(
            self.traj_a,
            self.traj_b,
        )

        risk = classify_risk(
            min_dist,
            safe_distance,
        )

        self._show_results(
            min_dist=min_dist,
            t_closest=t_closest,
            risk=risk,
            maneuver_applied=True,
            maneuver_direction=result.direction,
            original_min_dist=result.original_min_distance,
        )

        self._update_plot(
            safe_distance=safe_distance,
            t_closest=t_closest,
        )

    # --------------------------------------------------------------
    # Display helpers
    # --------------------------------------------------------------

    def _show_results(
        self,
        min_dist: float,
        t_closest: float,
        risk: str,
        maneuver_applied: bool,
        maneuver_direction: str = None,
        original_min_dist: float = None,
    ) -> None:
        """Update the numerical results panel."""

        lines = [
            f"Minimum separation: {min_dist:.2f} km",
            f"Time of closest approach: {t_closest:.1f} s",
            f"Collision risk (educational model): {risk}",
        ]

        if maneuver_applied:
            lines.append("")
            lines.append("AUTONOMOUS MANEUVER")

            lines.append(
                f"Direction: {maneuver_direction}"
            )

            lines.append(
                f"Delta-v: {self.delta_v_used * 1000:.2f} m/s"
            )

            if original_min_dist is not None:
                improvement = (
                    (min_dist - original_min_dist)
                    / original_min_dist
                    * 100.0
                )

                lines.append("")
                lines.append(
                    f"Before maneuver: {original_min_dist:.2f} km"
                )

                lines.append(
                    f"After maneuver:  {min_dist:.2f} km"
                )

                lines.append(
                    f"Separation improvement: {improvement:.1f}%"
                )

        else:
            lines.append("")
            lines.append(
                "Maneuver status: not applied"
            )

        self.results_text.set(
            "\n".join(lines)
        )

    def _update_plot(
        self,
        safe_distance: float = 100.0,
        t_closest: float = None,
    ) -> None:
        """Rebuild and embed the Matplotlib figure."""

        fig = build_figure(
            sat_a=self.sat_a,
            sat_b=self.sat_b,
            traj_a=self.traj_a,
            traj_b=self.traj_b,
            safe_distance=safe_distance,
            t_closest=t_closest,
        )

        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()

        self.canvas = FigureCanvasTkAgg(
            fig,
            master=self.plot_frame,
        )

        self.canvas.draw()

        self.canvas.get_tk_widget().pack(
            fill="both",
            expand=True,
        )


def main() -> None:
    """Start the application."""

    root = tk.Tk()
    SpacecraftApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()