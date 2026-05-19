"""
Example: isosurface rendering through the full 4-simplex composition space.

Uses the natural barycentric embedding  P = x1*V0 + x2*V1 + x3*V2 + x4*V3
so every composition point occupies a unique 3-D position inside the master
tetrahedron.  marching_cubes (scikit-image) extracts closed surfaces at
requested scalar levels; scipy's LinearNDInterpolator fills in the grid.

Requires:  pip install scipy scikit-image

Outputs are saved to the output/ directory.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from phase5d import PhaseDiagram5D
from phase5d.utils import generate_grid_data

OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)


def mixing_enthalpy(x0, x1, x2, x3, x4):
    """
    Toy mixing enthalpy: sum of pairwise interaction terms Ω_ij · xi · xj.
    Deliberately asymmetric to produce interesting isosurface topology.
    """
    x = [x0, x1, x2, x3, x4]
    omega = [
        [ 0, -10,   5,  -3,   8],
        [-10,   0,  12,  -7,   2],
        [  5,  12,   0,   9,  -4],
        [ -3,  -7,   9,   0,  11],
        [  8,   2,  -4,  11,   0],
    ]
    h = 0.0
    for i in range(5):
        for j in range(i + 1, 5):
            h += omega[i][j] * x[i] * x[j]
    return h


# ── Build dataset ──────────────────────────────────────────────────────────────
print("Generating grid data (step=0.04) ...")
data = generate_grid_data(step=0.04, value_fn=mixing_enthalpy)
print(f"  {len(data):,} data points")

pd5 = PhaseDiagram5D(
    data,
    value_type="continuous",
    colormap="RdBu_r",
    component_labels=["x0", "x1", "x2", "x3", "x4"],
)
print(f"  Value range: {pd5.vmin:.2f} ... {pd5.vmax:.2f}")

# ── Single isosurface ──────────────────────────────────────────────────────────
print("\nRendering single isosurface ...")
fig, ax = pd5.plot_isosurface(
    level=-1.0,
    colors=["steelblue"],
    alpha=0.55,
    grid_resolution=55,
    elev=22, azim=40,
    figsize=(8, 8),
    title="Isosurface  —  Hmix = -1  (single level)",
    show_colorbar=True,
    edgecolor="white",
    linewidth=0.1,
)
out = os.path.join(OUT_DIR, "isosurface_single.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out}")

# ── Three nested isosurfaces ───────────────────────────────────────────────────
print("Rendering three nested isosurfaces ...")
fig, ax = pd5.plot_isosurface(
    level=[-2.5, -1.0, 0.5],
    colors=["royalblue", "gold", "tomato"],
    alpha=0.38,
    grid_resolution=60,
    elev=22, azim=40,
    figsize=(8, 8),
    title="Isosurfaces  —  Hmix = -2.5 / -1.0 / +0.5",
    show_colorbar=True,
    edgecolor="white",
    linewidth=0.1,
)
out = os.path.join(OUT_DIR, "isosurface_nested.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out}")

# ── Different camera angle ─────────────────────────────────────────────────────
print("Rendering rotated view ...")
fig, ax = pd5.plot_isosurface(
    level=[-2.5, -1.0, 0.5],
    colors=["royalblue", "gold", "tomato"],
    alpha=0.38,
    grid_resolution=60,
    elev=10, azim=150,
    figsize=(8, 8),
    title="Isosurfaces  —  rotated view  (elev=10, azim=150)",
    show_colorbar=True,
    edgecolor="white",
    linewidth=0.1,
)
out = os.path.join(OUT_DIR, "isosurface_rotated.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out}")
