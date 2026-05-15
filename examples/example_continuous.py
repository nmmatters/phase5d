"""
Example: visualizing a continuous scalar property (e.g. mixing enthalpy).

Generates a synthetic five-component dataset on a regular grid and creates
both individual frames and a sweep video.
"""

import numpy as np
import matplotlib.pyplot as plt

from phase5d import PhaseDiagram5D
from phase5d.utils import generate_grid_data


def mixing_enthalpy(x0, x1, x2, x3, x4):
    """
    Toy mixing enthalpy:  sum of pairwise interaction terms Ω_ij * xi * xj.
    All Ω_ij are set arbitrarily to illustrate the visualization.
    """
    x = [x0, x1, x2, x3, x4]
    omega = [
        [0,  -10,  5,  -3,   8],
        [-10,  0,  12, -7,   2],
        [5,   12,   0,  9,  -4],
        [-3,  -7,   9,  0,  11],
        [8,    2,  -4, 11,   0],
    ]
    h = 0.0
    for i in range(5):
        for j in range(i + 1, 5):
            h += omega[i][j] * x[i] * x[j]
    return h


# ── Build dataset ──────────────────────────────────────────────────────────────
print("Generating grid data (step=0.05) …")
data = generate_grid_data(step=0.05, value_fn=mixing_enthalpy)
print(f"  {len(data):,} data points")

# ── Create diagram ─────────────────────────────────────────────────────────────
pd5 = PhaseDiagram5D(
    data,
    value_type="continuous",
    colormap="RdBu_r",
    component_labels=["x₀", "x₁", "x₂", "x₃", "x₄"],
)

# ── Single frame ───────────────────────────────────────────────────────────────
fig, ax = pd5.plot_frame(
    x0=0.20,
    mode="fixed",
    alpha=0.7,
    marker_size=8,
    elev=25,
    azim=30,
    title="Mixing enthalpy  —  x₀ = 0.20",
)
fig.savefig("frame_continuous_x0_0.20.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved: frame_continuous_x0_0.20.png")

# ── All three scale modes side by side ────────────────────────────────────────
fig, axes = plt.subplots(
    1, 3,
    figsize=(18, 6),
    subplot_kw={"projection": "3d"},
)
plt.close(fig)   # we let plot_frame manage its own figures

for mode, fname in [
    ("fixed",         "frame_mode_fixed.png"),
    ("shrink_center", "frame_mode_shrink_center.png"),
    ("shrink_corner", "frame_mode_shrink_corner.png"),
]:
    f, _ = pd5.plot_frame(x0=0.40, mode=mode, title=f"mode='{mode}', x₀=0.40",
                          marker_size=8, alpha=0.7)
    f.savefig(fname, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(f)
    print(f"Saved: {fname}")

# ── Video ──────────────────────────────────────────────────────────────────────
x0_sweep = np.round(np.arange(0.0, 1.01, 0.05), 3)

output = pd5.create_video(
    x0_values=x0_sweep,
    output_path="mixing_enthalpy.mp4",
    fps=5,
    dpi=120,
    mode="fixed",
    alpha=0.7,
    marker_size=6,
    verbose=True,
)
print(f"\nVideo: {output}")
