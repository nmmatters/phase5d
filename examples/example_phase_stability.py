"""
Example: visualizing phase stability labels (-1 / 0 / 1).

Demonstrates both scatter and alpha-shape surface rendering for a synthetic
five-component stability dataset.

  Stable   ( 1) — fully transparent (invisible)
  Meta-stable (0) — light gray, semi-transparent
  Unstable (-1) — dark gray, opaque

Outputs are saved to the output/ directory.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from phase5d import PhaseDiagram5D
from phase5d.utils import generate_grid_data

OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)


def stability_label(x0, x1, x2, x3, x4):
    """
    Toy stability rule:
      - stable    ( 1) when x0 > 0.5
      - unstable  (-1) when x1 + x3 > 0.6 and x0 <= 0.5
      - meta-stable (0) otherwise
    """
    if x0 > 0.5:
        return 1
    if x1 + x3 > 0.6:
        return -1
    return 0


# ── Build dataset ──────────────────────────────────────────────────────────────
print("Generating grid data (step=0.05) ...")
data = generate_grid_data(step=0.05, value_fn=stability_label)
print(f"  {len(data):,} data points")

counts = {v: int((data[:, 4] == v).sum()) for v in (-1, 0, 1)}
print(f"  Labels: {counts}")

# ── Create diagram ─────────────────────────────────────────────────────────────
pd5 = PhaseDiagram5D(
    data,
    value_type="phase_stability",
    component_labels=["x0", "Al", "Co", "Cr", "Fe"],
    phase_alphas={-1: 1.0, 0: 0.55, 1: 0.0},
)

# ── Single frame: scatter ──────────────────────────────────────────────────────
fig, ax = pd5.plot_frame(
    x0=0.20, mode="fixed", render="scatter",
    marker_size=12, elev=20, azim=50,
    title="Phase stability (scatter)  —  x0 = 0.20",
)
out = os.path.join(OUT_DIR, "stability_scatter_x0_0.20.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out}")

# ── Single frame: PyVista surface ─────────────────────────────────────────────
# save_frame_surface uses PyVista with an alpha-shape (concave hull) per
# stability class.  shape_alpha is chosen adaptively by default.
# Requires: pip install pyvista scipy
try:
    out = os.path.join(OUT_DIR, "stability_surface_x0_0.20.png")
    n = pd5.save_frame_surface(
        x0=0.20,
        out_path=out,
        mode="fixed",
        # shape_alpha=90  # uncomment to fix alpha instead of using adaptive
    )
    print(f"Saved: {out}  ({n:,} points)")
except ImportError:
    print("PyVista not installed — skipping surface frame.  pip install pyvista scipy")

# ── Show stable regions faintly ────────────────────────────────────────────────
pd5_show_stable = PhaseDiagram5D(
    data,
    value_type="phase_stability",
    component_labels=["x0", "Al", "Co", "Cr", "Fe"],
    phase_colors={1: (0.95, 0.95, 1.0)},   # very light blue for stable
    phase_alphas={-1: 1.0, 0: 0.55, 1: 0.15},
)
fig, ax = pd5_show_stable.plot_frame(
    x0=0.20, mode="fixed",
    title="Phase stability (stable shown)  —  x0 = 0.20",
)
out = os.path.join(OUT_DIR, "stability_all_phases_x0_0.20.png")
fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out}")

# ── Video sweep (scatter) ──────────────────────────────────────────────────────
x0_sweep = np.round(np.arange(0.0, 1.01, 0.05), 3)
output = pd5.create_video(
    x0_values=x0_sweep,
    output_path=os.path.join(OUT_DIR, "stability_scatter.mp4"),
    fps=5, dpi=120, mode="fixed", marker_size=10, verbose=True,
)
print(f"\nVideo: {output}")

# ── Video sweep (PyVista surface) ─────────────────────────────────────────────
# render='surface' routes to PyVista (adaptive alpha shapes).
# Requires: pip install pyvista scipy
try:
    output = pd5.create_video(
        x0_values=x0_sweep,
        output_path=os.path.join(OUT_DIR, "stability_surface.mp4"),
        fps=5, mode="fixed", render="surface", verbose=True,
    )
    print(f"Video: {output}")
except ImportError:
    print("PyVista not installed — skipping surface video.  pip install pyvista scipy")
