"""
Render the FeMnNiCoCu phase-stability map using phase5d.

Loads a TCHEA4 .dat file from data/ and renders phase-stability frames
or a full video sweep using PhaseDiagram5D.

Column layout of the .dat file:
  Gm GmxMn GmxNi GmxCo GmxCu Hmr ... xMn xNi xCo xCu QF
  indices:                         15   16   17   18  19

QF thresholds:
  QF < 0     -> unstable   (-1)
  0 <= QF <= 1 -> meta-stable (0)
  QF > 1     -> stable      (1)

Usage
-----
  python render_stability.py                         # single frame x(Fe)=0.10
  python render_stability.py --x0 0.30               # custom x0
  python render_stability.py --video                 # full sweep (auto-trimmed)
  python render_stability.py --video --x0max 0.50   # partial sweep
  python render_stability.py --scatter               # matplotlib scatter mode
  python render_stability.py --file tchea4_873k.dat  # specific data file
"""

import argparse
import glob
import os
import time

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_here      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(_here, "data")
OUTPUT_DIR = os.path.join(_here, "output")

PHASE_COLORS = {
    -1: (0.25, 0.25, 0.25),   # dark gray  — unstable
     0: (0.75, 0.75, 0.75),   # light gray — meta-stable
     1: (1.00, 1.00, 1.00),   # white      — stable
}

PHASE_ALPHAS = {
    -1: 0.45,
     0: 0.40,
     1: 0.00,   # stable: transparent
}

PHASE_NAMES = {
    -1: "Unstable",
     0: "Meta-stable",
     1: "Stable",
}


def _resolve_dat(requested: str = None) -> str:
    """Resolve the .dat file path from an optional user-supplied name or full path."""
    if requested:
        if os.path.isabs(requested):
            path = requested
        else:
            path = os.path.join(DATA_DIR, requested)
            if not os.path.exists(path):
                path = requested
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {requested}")
        return path

    # Auto-detect: use the first .dat file in data/
    candidates = sorted(glob.glob(os.path.join(DATA_DIR, "*.dat")))
    if not candidates:
        raise FileNotFoundError(
            f"No .dat files found in {DATA_DIR}.\n"
            "Place a TCHEA4 .dat file there or pass --file <path>."
        )
    if len(candidates) > 1:
        print(f"Multiple .dat files found — using: {os.path.basename(candidates[0])}")
        for c in candidates:
            print(f"  {os.path.basename(c)}")
        print("  (pass --file <name> to choose a specific one)")
    return candidates[0]


def load_data(dat_path: str) -> np.ndarray:
    """
    Load a TCHEA4 .dat file and return a phase5d-compatible (N, 5) array.

    Columns: [xMn, xNi, xCo, xCu, stability_label]
    phase5d reads x0 = xFe = 1 - xMn - xNi - xCo - xCu implicitly.
    """
    print(f"Loading data from {dat_path} ...")
    t0  = time.time()
    raw = np.loadtxt(dat_path, skiprows=1)
    print(f"  {len(raw):,} rows loaded in {time.time()-t0:.1f}s")

    x_mn, x_ni, x_co, x_cu = raw[:, 15], raw[:, 16], raw[:, 17], raw[:, 18]
    qf = raw[:, 19]

    # Remove sentinel values (pure-component reference rows)
    valid = qf < 1e4
    x_mn, x_ni, x_co, x_cu, qf = (
        x_mn[valid], x_ni[valid], x_co[valid], x_cu[valid], qf[valid]
    )

    labels = np.where(qf < 0, -1, np.where(qf <= 1, 0, 1))
    data   = np.column_stack([x_mn, x_ni, x_co, x_cu, labels.astype(float)])
    print(f"  {len(data):,} valid points after filtering")
    return data


def make_diagram(data: np.ndarray):
    """Build and return a PhaseDiagram5D for the phase-stability dataset."""
    from phase5d import PhaseDiagram5D

    return PhaseDiagram5D(
        data,
        x0="implicit",
        value_type="phase_stability",
        component_labels=["Fe", "Mn", "Ni", "Co", "Cu"],
        phase_colors=PHASE_COLORS,
        phase_alphas=PHASE_ALPHAS,
        phase_names=PHASE_NAMES,
        tolerance=0.005,
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--file",  metavar="FILENAME",
                        help="TCHEA4 .dat file (name or full path). "
                             "Defaults to the first .dat in data/.")
    parser.add_argument("--x0",    type=float, default=0.10,
                        help="x(Fe) value for single-frame render (default 0.10)")
    parser.add_argument("--x0max", type=float, default=1.00,
                        help="Upper limit for video sweep (default 1.00)")
    parser.add_argument("--video",   action="store_true",
                        help="Render a full video sweep instead of a single frame")
    parser.add_argument("--scatter", action="store_true",
                        help="Use matplotlib scatter instead of PyVista surface")
    parser.add_argument("--fps",   type=int, default=10,
                        help="Video frame rate (default 10)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    dat_path = _resolve_dat(args.file)
    stem     = os.path.splitext(os.path.basename(dat_path))[0]
    data     = load_data(dat_path)
    diag     = make_diagram(data)

    if args.video:
        all_x0 = np.round(np.arange(0.00, args.x0max + 1e-9, 0.01), 4).tolist()

        # Drop x0 values whose slice has fewer than min_points — those frames
        # render as an empty wireframe and add nothing to the video.
        min_pts   = 1000
        tol       = diag.tolerance
        x0_values = [
            x0 for x0 in all_x0
            if (np.abs(diag.data[:, 0] - x0) <= tol).sum() >= min_pts
        ]
        if len(x0_values) < len(all_x0):
            print(f"  Auto-trimmed to {len(x0_values)} frames "
                  f"(x(Fe) = {x0_values[0]:.2f} -> {x0_values[-1]:.2f}); "
                  f"{len(all_x0) - len(x0_values)} sparse frames skipped.")

        render     = "scatter" if args.scatter else "surface"
        video_out  = os.path.join(OUTPUT_DIR, f"stability_{stem}_{render}.mp4")
        frames_dir = os.path.join(OUTPUT_DIR, f"frames_{stem}_{render}")
        print(f"Rendering {len(x0_values)} frames ({render}) -> {video_out}")
        print(f"Frames saved to: {frames_dir}")

        t0 = time.time()
        diag.create_video(
            x0_values=x0_values,
            output_path=video_out,
            fps=args.fps,
            render=render,
            keep_frames=True,
            frames_dir=frames_dir,
        )
        elapsed = time.time() - t0
        size_kb = os.path.getsize(video_out) // 1024
        print(f"\nVideo done in {elapsed/60:.1f} min  ({size_kb} KB)")
        print(f"Saved: {video_out}")

    else:
        x0     = args.x0
        render = "scatter" if args.scatter else "surface"
        out    = os.path.join(OUTPUT_DIR, f"stability_{stem}_x{x0:.2f}_{render}.png")
        print(f"Rendering single frame x(Fe)={x0:.3f} ({render}) ...")
        t0 = time.time()
        if args.scatter:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, _ = diag.plot_frame(x0, render="scatter", marker_size=2,
                                     max_points=20000)
            fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
            plt.close(fig)
        else:
            try:
                n = diag.save_frame_surface(x0, out)
                print(f"  {n:,} points in slice")
            except ImportError:
                print("PyVista not found — falling back to scatter render.")
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                fig, _ = diag.plot_frame(x0, render="scatter", marker_size=2,
                                         max_points=20000)
                fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
                plt.close(fig)
        print(f"  Done in {time.time()-t0:.1f}s  ->  {out}")


if __name__ == "__main__":
    main()
