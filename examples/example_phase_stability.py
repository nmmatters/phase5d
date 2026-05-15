"""
Example: visualizing phase stability labels (-1 / 0 / 1).

Stable regions are transparent, meta-stable are light gray, unstable are dark gray.
"""

import numpy as np
import matplotlib.pyplot as plt

from phase5d import PhaseDiagram5D
from phase5d.utils import generate_grid_data


def stability_label(x0, x1, x2, x3, x4):
    """
    Toy stability rule:
      - stable   ( 1) when x0 > 0.5
      - unstable (-1) when x1 + x3 > 0.6  and x0 <= 0.5
      - meta-stable (0) otherwise
    """
    if x0 > 0.5:
        return 1
    if x1 + x3 > 0.6:
        return -1
    return 0


# ── Build dataset ──────────────────────────────────────────────────────────────
print("Generating grid data (step=0.05) …")
data = generate_grid_data(step=0.05, value_fn=stability_label)
print(f"  {len(data):,} data points")

counts = {v: int((data[:, 4] == v).sum()) for v in (-1, 0, 1)}
print(f"  Labels: {counts}")

# ── Create diagram ─────────────────────────────────────────────────────────────
pd5 = PhaseDiagram5D(
    data,
    value_type="phase_stability",
    component_labels=["x₀", "Al", "Co", "Cr", "Fe"],  # example HEA names
    phase_alphas={-1: 1.0, 0: 0.55, 1: 0.0},
)

# ── Single frame ───────────────────────────────────────────────────────────────
fig, ax = pd5.plot_frame(
    x0=0.20,
    mode="fixed",
    marker_size=12,
    elev=20,
    azim=50,
    title="Phase stability  —  x₀ = 0.20",
)
fig.savefig("frame_stability_x0_0.20.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved: frame_stability_x0_0.20.png")

# ── Show stable regions by making them slightly visible ────────────────────────
pd5_show_stable = PhaseDiagram5D(
    data,
    value_type="phase_stability",
    component_labels=["x₀", "Al", "Co", "Cr", "Fe"],
    phase_colors={1: (0.95, 0.95, 1.0)},   # very light blue for stable
    phase_alphas={-1: 1.0, 0: 0.55, 1: 0.15},
)
fig, ax = pd5_show_stable.plot_frame(x0=0.20, mode="fixed", marker_size=12,
                                      title="Phase stability (stable shown) — x₀ = 0.20")
fig.savefig("frame_stability_all_phases.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Saved: frame_stability_all_phases.png")

# ── Video sweep ────────────────────────────────────────────────────────────────
x0_sweep = np.round(np.arange(0.0, 1.01, 0.05), 3)

output = pd5.create_video(
    x0_values=x0_sweep,
    output_path="phase_stability.mp4",
    fps=5,
    dpi=120,
    mode="fixed",
    marker_size=10,
    verbose=True,
)
print(f"\nVideo: {output}")
