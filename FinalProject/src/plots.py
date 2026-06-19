"""Plot and animation generation."""

from __future__ import annotations

import math
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from constants import AU_KM, R_SUN


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_orbit(path: Path, best: dict) -> None:
    ensure_dir(path.parent)
    rp = best["rp_km"] / AU_KM
    r1 = 1.0
    a = 0.5 * (r1 + rp)
    e = (r1 - rp) / (r1 + rp)
    b = a * math.sqrt(1.0 - e * e)
    center = -(a - rp)
    t = np.linspace(0, 2 * math.pi, 600)
    x = center + a * np.cos(t)
    y = b * np.sin(t)
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.plot(np.cos(t), np.sin(t), "--", color="0.7", lw=1.2, label="Earth orbit")
    ax.plot(x, y, color="#0F4C75", lw=2.0, label="solar transfer ellipse")
    ax.scatter([0], [0], s=90, color="#f2a900", edgecolor="k", label="Sun")
    ax.scatter([x[np.argmin(np.abs(y))]], [0], s=28, color="#c0392b", label="perihelion")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x / AU")
    ax.set_ylabel("y / AU")
    ax.set_title("Best patched-conic solar-return ellipse")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_scan(path: Path, daily: list[dict]) -> None:
    ensure_dir(path.parent)
    x = np.arange(len(daily))
    y = np.array([d["total_delta_v_km_s"] for d in daily])
    best_i = int(np.argmin(y))
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.plot(x, y, color="#0F4C75", lw=1.5)
    ax.scatter([best_i], [y[best_i]], color="#c0392b", zorder=5)
    ax.set_xlabel("day of 2026")
    ax.set_ylabel(r"$\Delta v_{\rm total}$ / km s$^{-1}$")
    ax.set_title("Launch-window scan")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_contour(path: Path, contour: np.ndarray, rp_grid_AU: list[float]) -> None:
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    im = ax.imshow(
        contour.T,
        origin="lower",
        aspect="auto",
        extent=[0, 364, min(rp_grid_AU), max(rp_grid_AU)],
        cmap="viridis",
    )
    ax.set_xlabel("day of 2026")
    ax.set_ylabel(r"$r_p$ / AU")
    ax.set_title(r"$\Delta v_{\rm total}(t_0,r_p)$")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("km/s")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_residuals(path: Path, m3: dict) -> None:
    ensure_dir(path.parent)
    arr = np.array(m3["daily_position_residuals_km"], dtype=float)
    names = m3["body_names"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for i, name in enumerate(names):
        ax.plot(arr[:, i], label=name, lw=1.4)
    ax.axhline(6000.0, color="#c0392b", lw=1.0, ls="--", label="6000 km")
    ax.set_xlabel("day")
    ax.set_ylabel("position residual / km")
    ax.set_title("Sun-Earth-Moon propagation residual vs Horizons")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_energy(path: Path, energy: dict) -> None:
    ensure_dir(path.parent)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(energy["days"], energy["relative_drift"], color="#0F4C75", lw=1.5)
    ax.axhline(1.0e-6, color="#c0392b", lw=1.0, ls="--", label=r"$10^{-6}$")
    ax.axhline(-1.0e-6, color="#c0392b", lw=1.0, ls="--")
    ax.set_xlabel("day")
    ax.set_ylabel("relative energy drift")
    ax.set_title("One-year Sun-Earth-Moon energy conservation")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_sensitivity(path: Path, sens: dict) -> None:
    ensure_dir(path.parent)
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.2))
    date = sens["date_offset"]
    axs[0].plot([x["offset_day"] for x in date], [x["total_delta_v_km_s"] for x in date], marker="o")
    axs[0].set_xlabel("date offset / d")
    axs[0].set_ylabel("km/s")
    rm = sens["rm"]
    axs[1].plot([x["rm_km"] for x in rm], [x["total_delta_v_km_s"] for x in rm], marker="o")
    axs[1].set_xlabel(r"$r_m$ / km")
    step = sens["step"]
    axs[2].plot([x["dt_s"] for x in step], [x["energy_drift"] for x in step], marker="o")
    axs[2].set_xlabel("step / s")
    axs[2].set_ylabel("relative energy drift")
    axs[2].set_yscale("log")
    for ax in axs:
        ax.grid(alpha=0.25)
    fig.suptitle("Sensitivity and convergence near the optimum")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_animation(path: Path, best: dict) -> None:
    ensure_dir(path.parent)
    frames = []
    rp = best["rp_km"] / AU_KM
    a = 0.5 * (1.0 + rp)
    e = (1.0 - rp) / (1.0 + rp)
    b = a * math.sqrt(1.0 - e * e)
    center = -(a - rp)
    tt = np.linspace(0, 2 * math.pi, 360)
    x = center + a * np.cos(tt)
    y = b * np.sin(tt)
    earth_x = np.cos(tt)
    earth_y = np.sin(tt)
    for k in range(120):
        i = int(k / 119 * (len(tt) - 1))
        fig, ax = plt.subplots(figsize=(5.12, 5.12), dpi=100)
        ax.plot(np.cos(tt), np.sin(tt), "--", color="0.75", lw=1)
        ax.plot(x, y, color="#0F4C75", lw=1.8)
        ax.plot(x[: i + 1], y[: i + 1], color="#c0392b", lw=2.4)
        ax.scatter([0], [0], s=120, color="#f2a900", edgecolor="k")
        ax.scatter([earth_x[i]], [earth_y[i]], s=45, color="#2e86ab", label="Earth")
        ax.scatter([x[i]], [y[i]], s=35, color="#c0392b", label="rocket")
        lim = 1.15
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Moon-assisted solar-return transfer")
        ax.legend(loc="upper right", fontsize=8)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(buf)
        plt.close(fig)
    imageio.mimsave(path, frames, fps=24, macro_block_size=16)
