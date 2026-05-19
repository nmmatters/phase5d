"""
Example: visualizing a continuous scalar property (mixing enthalpy).

Generates a synthetic five-component dataset on a regular grid and creates
individual frames and a sweep video using the scatter render mode.

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
    All Ω_ij are set arbitrarily to illustrate the visualization.
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
print("Generating grid data (step=0.05) ...")
data = generate_grid_data(step=0.05, value_fn=mixing_enthalpy)
print(f"  {len(data):,} data points")

# ── Create diagram ─────────────────────────────────────────────────────────────
pd5 = PhaseDiagram5D(
    data,
    value_type="continuous",
    colormap="RdBu_r",
    component_labels=["x0", "Al", "Co", "Cr", "Fe"],
)

# ── Single frame (scatter) ─────────────────────────────────────────────────────
fig, ax = pd5.plot_frame(
    x0=0.20,
    mode="fixed",
    render="scatter",
    alpha=0.7,
    marker_size=8,
    elev=25,
    azim=30,
    title="Mixing enthalpy  —  x0 = 0.20",
)
out = os.path.join(OUT_DIR, "continuous_scatter_x0_0.20.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out}")

# ── Three scaling modes ────────────────────────────────────────────────────────
for mode in ("fixed", "shrink_center", "shrink_corner"):
    f, _ = pd5.plot_frame(x0=0.40, mode=mode, marker_size=8, alpha=0.7,
                          title=f"mode='{mode}',  x0=0.40")
    out = os.path.join(OUT_DIR, f"continuous_mode_{mode}.png")
    f.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(f)
    print(f"Saved: {out}")

# ── Video sweep ────────────────────────────────────────────────────────────────
x0_sweep = np.round(np.arange(0.0, 1.01, 0.05), 3)
output = pd5.create_video(
    x0_values=x0_sweep,
    output_path=os.path.join(OUT_DIR, "continuous_scatter.mp4"),
    fps=5, dpi=120, mode="fixed", alpha=0.7, marker_size=6, verbose=True,
)
print(f"\nVideo: {output}")
