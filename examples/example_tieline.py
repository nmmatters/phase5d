"""
Example: two-phase equilibrium overlay (tie-line).

Demonstrates how to mark a nominal alloy composition and overlay the
tie-line connecting two equilibrium phases on a surface-rendered frame
and video.

As x0 sweeps through the phase diagram, the tie-line's intersection with
the current x0 plane is shown as a red sphere.  The sphere appears when the
slice enters the two-phase window and vanishes when it exits, giving an
instant visual cue for where the two-phase region is active.

Two data paths are supported:
  1. Real CALPHAD data  (TCHEA4, 873 K, step=0.01):
       data/tchea4_mobfe5_873k_n100.dat
     If this file is present, it is used automatically.

  2. Synthetic fallback (no data file required):
     A toy stability function with a plausible two-phase window.
     Compositions look different from a real system but demonstrate
     the full API identically.

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

    # QF thresholds:  <0 -> unstable (-1),  0..1 -> meta-stable (0),  >1 -> stable (1)
    labels = np.where(qf < 0, -1, np.where(qf <= 1, 0, 1))

    # Build (N, 5) array: [xMn, xNi, xCo, xCu, label]
    # x(Fe) = 1 - xMn - xNi - xCo - xCu  (implicit, handled by x0='implicit')
    data = np.column_stack([x_mn, x_ni, x_co, x_cu, labels.astype(float)])

    component_labels = ["Fe", "Mn", "Ni", "Co", "Cu"]
    using_real_data  = True

    # ── Two-phase equilibrium compositions  [x(Fe), xMn, xNi, xCo, xCu] ────────
    # Compositions span x(Fe) = 0.00 -> 0.24 so the tie-line is visible
    # across the first 25 frames of the sweep below.
    NOMINAL = [0.10, 0.23, 0.22, 0.22, 0.23]  # near-equimolar alloy
    PHASE_A = [0.05, 0.59, 0.16, 0.12, 0.08]  # Mn-rich phase (low Fe end)
    PHASE_B = [0.22, 0.12, 0.22, 0.22, 0.22]  # Fe-rich phase (high Fe end)

    x0_start, x0_end, x0_step = 0.00, 0.25, 0.01

else:
    print("Real data not found — using synthetic stability function.")
    print("To use real data, place tchea4_mobfe5_873k_n100.dat in data/")

    def hea_stability(x0, x1, x2, x3, x4):
        """
        Toy stability rule with a two-phase window at moderate x0.
          stable    ( 1): high x0 content (> 0.55) or balanced mid-range
          unstable  (-1): high x1 or x3 content at low x0
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

    component_labels = ["x0", "x1", "x2", "x3", "x4"]
    using_real_data  = False

    # Synthetic two-phase tie-line spanning x0 = 0.10 -> 0.50
    NOMINAL = [0.25, 0.20, 0.20, 0.20, 0.15]
    PHASE_A = [0.10, 0.55, 0.15, 0.12, 0.08]  # x1-rich, low x0
    PHASE_B = [0.50, 0.10, 0.20, 0.10, 0.10]  # x0-rich, high x0

    x0_start, x0_end, x0_step = 0.00, 0.55, 0.04

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
    phase_alphas={-1: 0.45, 0: 0.40, 1: 0.0},
    tolerance=0.005,
)

# ── Single frame: nominal composition + tie-line ──────────────────────────────
# Render the frame at the nominal x0.  The tie-line intersection (red sphere)
# is shown because the nominal x0 lies inside the two-phase window.
x0_frame = NOMINAL[0]
print(f"Rendering single frame  x0 = {x0_frame:.3f} ...")
t1  = time.time()
out = os.path.join(OUT_DIR, f"tieline_frame_x0_{x0_frame:.2f}.png")
try:
    n = pd5.save_frame_surface(
        x0=x0_frame,
        out_path=out,
        markers=[NOMINAL],
        tielines=[[PHASE_A, PHASE_B]],
    )
    print(f"  {n:,} points  ({time.time()-t1:.1f}s)")
    print(f"  Saved: {out}")
except ImportError:
    print("  PyVista not installed — skipping surface frame.")
    print("  Install with:  pip install pyvista scipy")

# ── Video sweep: full tie-line window ─────────────────────────────────────────
# The red sphere enters the frame as soon as x0 crosses the lower endpoint
# of the tie-line (PHASE_A[0]) and disappears once it passes the upper
# endpoint (PHASE_B[0]).
x0_values = np.round(np.arange(x0_start, x0_end, x0_step), 3)
print(f"\nRendering tieline video  ({len(x0_values)} frames) ...")
try:
    t_video    = time.time()
    video_path = pd5.create_video(
        x0_values=x0_values,
        output_path=os.path.join(OUT_DIR, "tieline_sweep.mp4"),
        fps=10,
        render="surface",
        keep_frames=False,
        verbose=True,
        markers=[NOMINAL],
        tielines=[[PHASE_A, PHASE_B]],
    )
    elapsed = time.time() - t_video
    size_kb = os.path.getsize(video_path) // 1024
    print(f"\nVideo done in {elapsed/60:.1f} min  ({size_kb} KB)")
    print(f"Saved: {video_path}")
except ImportError:
    print("  PyVista not installed — skipping surface video.")
    print("  Install with:  pip install pyvista scipy")
