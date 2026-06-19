#!/usr/bin/env python3
"""
General Relativity visualization: 2D Earth orbiting 2D Sun on a curved 3D surface.
Both masses curve the space — the Sun's well is fixed and deep,
the Earth's well is shallower and moves with the planet.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# ── Parameters ─────────────────────────────────────────────────────────
grid_half, grid_n = 7.0, 160
sun_mass = 3.0
sun_eps = 0.9       # larger softening → bigger apparent size
earth_mass = 0.3
earth_eps = 0.6    # smaller softening → smaller apparent size

x = np.linspace(-grid_half, grid_half, grid_n)
y = np.linspace(-grid_half, grid_half, grid_n)
X, Y = np.meshgrid(x, y)

def well_z(cx, cy, mass, eps):
    """Gravity well: U ∝ -mass / (r + eps).  eps = softening radius."""
    r2 = (X - cx)**2 + (Y - cy)**2
    return -mass / (np.sqrt(r2) + eps)

# Sun's fixed contribution
Z_sun = well_z(0, 0, sun_mass, sun_eps)

# ── Orbit ──────────────────────────────────────────────────────────────
a, b = 3.6, 2.8
c = np.sqrt(a**2 - b**2)    # distance from centre to focus
T = 120                     # frames per orbit
num_orbits = 2
theta = np.linspace(0, 2 * np.pi * num_orbits, T * num_orbits, endpoint=False)

# Orbit centre offset so that Sun at (0,0) sits at the right focus
ex_arr = a * np.cos(theta) - c
ey_arr = b * np.sin(theta)

def height_at(px, py):
    """Total Z at a point: Sun well + Earth's own well at its position."""
    r_sun = np.sqrt(px**2 + py**2)
    z_sun = -sun_mass / (r_sun + sun_eps)
    z_self = -earth_mass / earth_eps   # Earth sits in its own well
    return z_sun + z_self

ez_arr = np.array([height_at(ex_arr[i], ey_arr[i]) for i in range(len(theta))])
ez_arr = np.clip(ez_arr, -3.5, -0.02)

# ── Figure ─────────────────────────────────────────────────────────────
plt.style.use("dark_background")
fig = plt.figure(figsize=(12, 9), dpi=100)
ax = fig.add_subplot(111, projection="3d", facecolor="black")
fig.patch.set_facecolor("black")

sun_z_bottom = Z_sun[grid_n // 2, grid_n // 2]

# ── Style helpers ──────────────────────────────────────────────────────
def style_ax():
    ax.set_xlim(-grid_half, grid_half)
    ax.set_ylim(-grid_half, grid_half)
    ax.set_zlim(-3.5, 1.0)
    ax.set_xlabel("X", fontsize=10, color="white")
    ax.set_ylabel("Y", fontsize=10, color="white")
    ax.set_zlabel("Space curvature (Z)", fontsize=10, color="white")
    ax.set_title(
        "General Relativity: Gravity as Curved Space\n"
        "Both Sun and Earth curve the fabric of space",
        fontsize=13, color="white", pad=20,
    )
    ax.grid(False)
    ax.xaxis.pane.fill = False; ax.yaxis.pane.fill = False; ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#333")
    ax.yaxis.pane.set_edgecolor("#333")
    ax.zaxis.pane.set_edgecolor("#333")
    ax.tick_params(colors="#888")

style_ax()

# Initial frame
Z_combined = Z_sun + well_z(ex_arr[0], ey_arr[0], earth_mass, earth_eps)
Z_combined = np.clip(Z_combined, -3.5, 0.0)

surf = ax.plot_surface(
    X, Y, Z_combined, cmap="coolwarm", rstride=2, cstride=2,
    linewidth=0, antialiased=True, alpha=0.85, zorder=1,
)

(sun_pt,) = ax.plot(
    [0], [0], [sun_z_bottom],
    "o", color="#FFD700", markersize=16, markeredgecolor="#FFA500",
    markeredgewidth=2, alpha=0.5, zorder=10, label="Sun",
)

(earth_pt,) = ax.plot(
    [ex_arr[0]], [ey_arr[0]], [ez_arr[0]],
    "o", color="#4169E1", markersize=7, markeredgecolor="#87CEEB",
    markeredgewidth=1.5, alpha=0.3, zorder=11, label="Earth",
)

trail_len = 50
(trail_line,) = ax.plot([], [], [], "-", color="cyan", lw=1.2, alpha=0.55, zorder=9)

ax.legend(loc="upper left", fontsize=9, facecolor="#1a1a1a", edgecolor="gray")
ax.view_init(elev=28, azim=-55)

annot = ax.text2D(0.02, 0.02, "", transform=ax.transAxes,
                  fontsize=9, color="#ccc", family="monospace")

# ── Update ─────────────────────────────────────────────────────────────
def update(frame):
    global surf
    ex, ey, ez = ex_arr[frame], ey_arr[frame], ez_arr[frame]

    # Remove old surface, draw new one with Earth well at current position
    surf.remove()
    Z_new = Z_sun + well_z(ex, ey, earth_mass, earth_eps)
    Z_new = np.clip(Z_new, -3.5, 0.0)
    surf = ax.plot_surface(
        X, Y, Z_new, cmap="coolwarm", rstride=2, cstride=2,
        linewidth=0, antialiased=True, alpha=0.85, zorder=1,
    )

    # Update Earth marker
    earth_pt.set_data([ex], [ey])
    earth_pt.set_3d_properties([ez])

    # Trail
    s = max(0, frame - trail_len)
    trail_line.set_data(ex_arr[s:frame+1], ey_arr[s:frame+1])
    trail_line.set_3d_properties(ez_arr[s:frame+1])

    # Subtle camera sway
    azim = -55 + 0.25 * np.sin(0.035 * frame)
    elev = 28 + 0.15 * np.sin(0.025 * frame + 1.0)
    ax.view_init(elev=elev, azim=azim)

    annot.set_text(
        f"Orbit: {theta[frame] / (2*np.pi) % 1:.1%}  "
        f"R: {np.sqrt(ex**2+ey**2):.2f}  "
        f"depth: {ez:.3f}"
    )
    return earth_pt, trail_line, surf, annot

ani = FuncAnimation(fig, update, frames=len(theta), interval=30, repeat=True)

print("Rendering (with dynamic Earth gravity well)...")
writer = FFMpegWriter(fps=30, bitrate=2000, codec="h264")
ani.save("gravity_curved_space.mp4", writer=writer, dpi=100)
print("Done → gravity_curved_space.mp4")
