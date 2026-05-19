"""
Example: high-quality surface rendering for FeMnNiCoCu at 873 K.

Demonstrates render='surface' — PyVista's VTK backend with smooth shading,
proper lighting, and adaptive alpha-shape concave hull surfaces.

Two data paths are supported:
  1. Real CALPHAD data  (TCHEA4, 873 K, step=0.01):
       data/tchea4_mobfe5_873k_n100.dat
     If this file is present, it is used automatically.

  2. Synthetic fallback (no data file required):
     A toy stability function mimics the FeMnNiCoCu phase boundary structure.
     Results look different from the real system but demonstrate the full API.

Requires:  pip install pyvista scipy

Outputs are saved to the output/ directory.
"""

import os
import time

import numpy as np

from phase5d import PhaseDiagram5D
from phase5d.utils import generate_grid_data

OUT_DIR  = "output"
REAL_DAT = os.path.join("data", "tchea4_mobfe5_873k_n100.dat")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load or generate data ──────────────────────────────────────────────────────
if os.path.isfile(REAL_DAT):
    print(f"Loading real CALPHAD data: {REAL_DAT}")
    t0  = time.time()
    raw = np.loadtxt(REAL_DAT, skiprows=1)
    print(f"  {len(raw):,} rows loaded in {time.time()-t0:.1f}s")

    # Column layout: Gm GmxMn GmxNi GmxCo GmxCu Hmr ... xMn xNi xCo xCu QF
    x_mn, x_ni, x_co, x_cu = raw[:, 15], raw[:, 16], raw[:, 17], raw[:, 18]
    qf = raw[:, 19]

    # Remove sentinel values (pure-component reference rows)
    valid = qf < 1e4
    x_mn, x_ni, x_co, x_cu, qf = (
        x_mn[valid], x_ni[valid], x_co[valid], x_cu[valid], qf[valid]
    )

    # QF thresholds:  <0 → unstable (-1),  0..1 → meta-stable (0),  >1 → stable (1)
    labels = np.where(qf < 0, -1, np.where(qf <= 1, 0, 1))

    # Build (N, 5) array: [x1=xMn, x2=xNi, x3=xCo, x4=xCu, value=label]
    # x0 = x(Fe) = 1 - xMn - xNi - xCo - xCu  (implicit, handled by x0='implicit')
    data = np.column_stack([x_mn, x_ni, x_co, x_cu, labels.astype(float)])

    component_labels = ["Fe", "Mn", "Ni", "Co", "Cu"]
    x0_label         = "x(Fe)"
    using_real_data  = True

else:
    print("Real data not found — using synthetic stability function.")
    print("To use real data, place tchea4_mobfe5_873k_n100.dat in data/")

    def hea_stability(x0, x1, x2, x3, x4):
        """
        Synthetic stability rule that loosely mimics a 5-component HEA.
          stable    ( 1): high x0 content
          unstable  (-1): high x1 or x3 content at moderate x0
          meta-stable (0): otherwise
        """
        if x0 > 0.55:
            return 1
        if x1 + x3 > 0.55 and x0 < 0.45:
            return -1
        if x2 + x4 > 0.60:
            return -1
        return 0

    print("Generating synthetic grid (step=0.04) ...")
    data = generate_grid_data(step=0.04, value_fn=hea_stability)
    print(f"  {len(data):,} data points")

    component_labels = ["Fe", "Mn", "Ni", "Co", "Cu"]
    x0_label         = "x(Fe)"
    using_real_data  = False

print(f"  Using {'real CALPHAD' if using_real_data else 'synthetic'} data\n")

# ── Create PhaseDiagram5D ──────────────────────────────────────────────────────
pd5 = PhaseDiagram5D(
    data,
    x0="implicit",
    value_type="phase_stability",
    component_labels=component_labels,
    phase_colors={
        -1: (0.25, 0.25, 0.25),   # dark gray  — unstable
         0: (0.75, 0.75, 0.75),   # light gray — meta-stable
         1: (1.00, 1.00, 1.00),   # white      — stable (transparent)
    },
    phase_alphas={-1: 0.90, 0: 0.40, 1: 0.0},
    tolerance=0.005,
)

# ── Single PyVista frame ───────────────────────────────────────────────────────
print("Rendering single PyVista frame (x0=0.30) ...")
t1 = time.time()
n  = pd5.save_frame_surface(
    x0=0.30,
    out_path=os.path.join(OUT_DIR, "femnnicopha_pv_x0_0.30.png"),
    # shape_alpha is adaptive by default (90 * (N/62196)^(1/3))
    # Pass shape_alpha=<value> to override, e.g. shape_alpha=90
)
print(f"  {n:,} points  ({time.time()-t1:.1f}s)")
print(f"  Saved: {os.path.join(OUT_DIR, 'femnnicopha_pv_x0_0.30.png')}")

# ── PyVista video: x(Fe) = 0.00 -> 0.40, step=0.01 ──────────────────────────
# This range covers the richest part of the phase diagram where all three
# stability classes coexist.  The full 0->1 sweep takes ~30-40 min on a
# modern desktop; see README for details on render time.
print("\nRendering PyVista video  x(Fe) = 0.00 -> 0.40, step=0.01 ...")
step   = 0.01 if using_real_data else 0.04
x0_end = 0.41 if using_real_data else 0.81
x0_values = np.round(np.arange(0.00, x0_end, step), 3)
print(f"  {len(x0_values)} frames")

t_video = time.time()
video_path = pd5.create_video(
    x0_values=x0_values,
    output_path=os.path.join(OUT_DIR, "femnnicopha_pv_surface.mp4"),
    fps=10,
    render="surface",
    verbose=True,
)
elapsed = time.time() - t_video
size_kb = os.path.getsize(video_path) // 1024
print(f"\nVideo done in {elapsed/60:.1f} min  ({size_kb} KB)")
print(f"Saved: {video_path}")
