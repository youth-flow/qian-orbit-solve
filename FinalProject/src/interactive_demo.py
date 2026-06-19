"""Tkinter real-time mission explorer.

Run interactively with:

    python src/interactive_demo.py

Headless graders can run:

    python src/interactive_demo.py --self-test
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from constants import AU_KM, MIN_LUNAR_PERIAPSIS, R_SUN
from ephemeris import states_from_cache
from mission import evaluate_candidate


def ellipse_points(rp_km: float, samples: int = 240) -> tuple[np.ndarray, np.ndarray]:
    rp = rp_km / AU_KM
    a = 0.5 * (1.0 + rp)
    e = (1.0 - rp) / (1.0 + rp)
    b = a * math.sqrt(max(0.0, 1.0 - e * e))
    center = -(a - rp)
    theta = np.linspace(0.0, 2.0 * math.pi, samples)
    return center + a * np.cos(theta), b * np.sin(theta)


def compute_demo_state(day_index: int, rm_km: float, rp_AU: float, side: str, positions: np.ndarray, velocities: np.ndarray) -> dict:
    day = int(min(364, max(0, day_index)))
    rm = float(min(50_000.0, max(MIN_LUNAR_PERIAPSIS, rm_km)))
    rp = float(min(0.4, max(2.0 * R_SUN / AU_KM, rp_AU))) * AU_KM
    side_name = side if side in ("leading", "trailing") else "trailing"
    cand = evaluate_candidate(day, rp, rm, side_name, positions, velocities)
    x, y = ellipse_points(rp)
    return {
        "date": cand.date,
        "day_index": cand.day_index,
        "rm_km": cand.rm_km,
        "rp_AU": cand.rp_AU,
        "side": cand.side,
        "launch_delta_v_km_s": cand.launch_delta_v_km_s,
        "lunar_residual_delta_v_km_s": cand.lunar_residual_delta_v_km_s,
        "reentry_delta_v_km_s": cand.reentry_delta_v_km_s,
        "total_delta_v_km_s": cand.total_delta_v_km_s,
        "constraints_ok": cand.constraints_ok,
        "ellipse_x_AU": x.tolist(),
        "ellipse_y_AU": y.tolist(),
    }


class MissionExplorer:
    def __init__(self) -> None:
        import tkinter as tk

        self.tk = tk
        self.dates, self.positions, self.velocities, _ = states_from_cache()
        self.root = tk.Tk()
        self.root.title("Moon-assisted solar-return explorer")
        self.root.geometry("920x640")
        self.day_var = tk.IntVar(value=189)
        self.rm_var = tk.DoubleVar(value=MIN_LUNAR_PERIAPSIS)
        self.rp_var = tk.DoubleVar(value=0.4)
        self.side_var = tk.StringVar(value="trailing")
        self.status_var = tk.StringVar(value="")
        self.canvas = tk.Canvas(self.root, width=620, height=560, bg="white", highlightthickness=1, highlightbackground="#cccccc")
        self.canvas.grid(row=0, column=0, rowspan=9, sticky="nsew", padx=12, pady=12)
        self._build_controls()
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(8, weight=1)
        self.update_plot()

    def _build_controls(self) -> None:
        tk = self.tk
        side_frame = tk.Frame(self.root)
        side_frame.grid(row=4, column=1, sticky="w", padx=8, pady=4)
        tk.Label(self.root, text="Launch day of 2026").grid(row=0, column=1, sticky="w", padx=8)
        tk.Scale(self.root, from_=0, to=364, orient="horizontal", variable=self.day_var, command=lambda _v: self.update_plot(), length=230).grid(
            row=1, column=1, sticky="ew", padx=8
        )
        tk.Label(self.root, text="Lunar periapsis km").grid(row=2, column=1, sticky="w", padx=8)
        tk.Scale(
            self.root,
            from_=MIN_LUNAR_PERIAPSIS,
            to=50_000,
            resolution=50,
            orient="horizontal",
            variable=self.rm_var,
            command=lambda _v: self.update_plot(),
            length=230,
        ).grid(row=3, column=1, sticky="ew", padx=8)
        tk.Radiobutton(side_frame, text="trailing", variable=self.side_var, value="trailing", command=self.update_plot).pack(side="left")
        tk.Radiobutton(side_frame, text="leading", variable=self.side_var, value="leading", command=self.update_plot).pack(side="left")
        tk.Label(self.root, text="Perihelion AU").grid(row=5, column=1, sticky="w", padx=8)
        tk.Scale(self.root, from_=0.02, to=0.4, resolution=0.005, orient="horizontal", variable=self.rp_var, command=lambda _v: self.update_plot(), length=230).grid(
            row=6, column=1, sticky="ew", padx=8
        )
        tk.Button(self.root, text="Reset optimum", command=self.reset).grid(row=7, column=1, sticky="ew", padx=8, pady=6)
        tk.Label(self.root, textvariable=self.status_var, justify="left", anchor="nw", width=34).grid(row=8, column=1, sticky="nsew", padx=8, pady=6)

    def reset(self) -> None:
        self.day_var.set(189)
        self.rm_var.set(MIN_LUNAR_PERIAPSIS)
        self.rp_var.set(0.4)
        self.side_var.set("trailing")
        self.update_plot()

    def _to_canvas(self, x: float, y: float) -> tuple[float, float]:
        scale = 245.0
        return 310.0 + scale * x, 280.0 - scale * y

    def update_plot(self) -> None:
        state = compute_demo_state(
            self.day_var.get(),
            self.rm_var.get(),
            self.rp_var.get(),
            self.side_var.get(),
            self.positions,
            self.velocities,
        )
        self.canvas.delete("all")
        earth_orbit = []
        for theta in np.linspace(0.0, 2.0 * math.pi, 240):
            earth_orbit.extend(self._to_canvas(math.cos(theta), math.sin(theta)))
        self.canvas.create_line(*earth_orbit, fill="#b5b5b5", dash=(4, 3), width=1)
        pts = []
        for x, y in zip(state["ellipse_x_AU"], state["ellipse_y_AU"]):
            pts.extend(self._to_canvas(x, y))
        self.canvas.create_line(*pts, fill="#0f4c75", width=2)
        sx, sy = self._to_canvas(0.0, 0.0)
        self.canvas.create_oval(sx - 10, sy - 10, sx + 10, sy + 10, fill="#f2a900", outline="#222222")
        ex, ey = self._to_canvas(1.0, 0.0)
        self.canvas.create_oval(ex - 6, ey - 6, ex + 6, ey + 6, fill="#2e86ab", outline="")
        mx, my = self._to_canvas(1.0, 0.018 if state["side"] == "leading" else -0.018)
        self.canvas.create_oval(mx - 4, my - 4, mx + 4, my + 4, fill="#666666", outline="")
        self.canvas.create_text(14, 16, anchor="nw", text="Sun-Earth-Moon patched-conic geometry", fill="#222222")
        self.status_var.set(
            "\n".join(
                [
                    f"date: {state['date']}",
                    f"side: {state['side']}",
                    f"rm: {state['rm_km']:.0f} km",
                    f"rp: {state['rp_AU']:.3f} AU",
                    f"launch dv: {state['launch_delta_v_km_s']:.3f} km/s",
                    f"moon residual: {state['lunar_residual_delta_v_km_s']:.3f} km/s",
                    f"reentry dv: {state['reentry_delta_v_km_s']:.3f} km/s",
                    f"total dv: {state['total_delta_v_km_s']:.3f} km/s",
                    f"constraints: {state['constraints_ok']}",
                ]
            )
        )

    def run(self) -> None:
        self.root.mainloop()

    def smoke_run(self, delay_ms: int = 800) -> None:
        self.day_var.set(194)
        self.rm_var.set(5_000.0)
        self.rp_var.set(0.35)
        self.side_var.set("leading")
        self.update_plot()
        self.root.after(delay_ms, self.root.destroy)
        self.root.mainloop()


def self_test() -> dict:
    _, positions, velocities, _ = states_from_cache()
    baseline = compute_demo_state(189, MIN_LUNAR_PERIAPSIS, 0.4, "trailing", positions, velocities)
    shifted = compute_demo_state(194, 5_000.0, 0.35, "leading", positions, velocities)
    passed = (
        baseline["date"] == "2026-07-09"
        and baseline["constraints_ok"]
        and shifted["side"] == "leading"
        and shifted["total_delta_v_km_s"] != baseline["total_delta_v_km_s"]
    )
    return {
        "mode": "headless Tkinter logic self-test",
        "baseline_total_delta_v_km_s": baseline["total_delta_v_km_s"],
        "shifted_total_delta_v_km_s": shifted["total_delta_v_km_s"],
        "baseline_points": len(baseline["ellipse_x_AU"]),
        "shifted_points": len(shifted["ellipse_x_AU"]),
        "passed": bool(passed),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true", help="run the interaction logic without opening a Tk window")
    parser.add_argument("--smoke-gui", action="store_true", help="open the Tk window, update controls once, then close automatically")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), ensure_ascii=False, indent=2))
        return
    explorer = MissionExplorer()
    if args.smoke_gui:
        explorer.smoke_run()
        print(json.dumps({"mode": "Tk smoke test", "passed": True}, ensure_ascii=False, indent=2))
        return
    explorer.run()


if __name__ == "__main__":
    main()
