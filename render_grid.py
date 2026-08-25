"""
Render a 4×4 publication-quality grid of phase-stability frames.

x(Fe) values: 0.05, 0.10, 0.15, ..., 0.80  (16 frames, step 0.05)

Each panel shows the PyVista surface render without an individual scale bar
or legend.  One shared legend is added below the grid.

Outputs
-------
  output/stability_grid.pdf   — vector PDF for publication (300 Dpi)
  output/stability_grid.png   — raster PNG at 300 DPI

Usage
-----
  python render_grid.py                      # use first .dat/.npz in data/
  python render_grid.py --file <name.dat>    # specific data file
  python render_grid.py --no-cache           # re-render frames even if cached
  python render_grid.py --dpi 600           # higher output DPI

Dependencies: phase5d, pyvista, scipy, matplotlib, numpy
"""

import argparse
import glob
import os
import time

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Headless rendering — all env vars must be set before pyvista/vtk is imported.
#
# VTK's X11 backend (vtkXOpenGLRenderWindow) always tries to connect to an X
# server, even with off_screen=True.  To avoid X entirely we switch to the EGL
# backend, which renders via GPU or Mesa EGL without any display server.
# VTK_DEFAULT_RENDER_WINDOW_BACKEND=EGL is the canonical way to do this in
# VTK ≥ 9.x.  PYVISTA_OFF_SCREEN and VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN are
# kept as belt-and-suspenders for environments that support the X11 backend in
# offscreen mode.
os.environ.setdefault("VTK_DEFAULT_RENDER_WINDOW_BACKEND", "EGL")
os.environ["PYVISTA_OFF_SCREEN"] = "true"
os.environ.setdefault("VTK_DEFAULT_RENDER_WINDOW_OFFSCREEN", "1")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_here      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(_here, "data")
OUTPUT_DIR = os.path.join(_here, "output")

# Grid parameters
X0_VALUES = np.round(np.arange(0.05, 0.81, 0.05), 2).tolist()  # 16 values
GRID_ROWS, GRID_COLS = 4, 4

# Phase appearance (must match render_stability.py)
PHASE_COLORS = {
    -1: (0.25, 0.25, 0.25),
     0: (0.75, 0.75, 0.75),
     1: (1.00, 1.00, 1.00),
}
PHASE_ALPHAS = {
    -1: 0.45,
     0: 0.40,
     1: 0.00,
}
PHASE_NAMES = {
    -1: "Unstable",
     0: "Meta-stable",
     1: "Stable",
}
# Legend edge colors (visible against white background)
PHASE_EDGE = {
    -1: (0.15, 0.15, 0.15),
     0: (0.50, 0.50, 0.50),
     1: (0.40, 0.40, 0.40),
}


# ---------------------------------------------------------------------------
# Data helpers  (identical logic to render_stability.py)
# ---------------------------------------------------------------------------

def _resolve_dat(requested=None):
    if requested:
        path = requested if os.path.isabs(requested) else os.path.join(DATA_DIR, requested)
        if not os.path.exists(path):
            path = requested
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {requested}")
        return path
    candidates = sorted(glob.glob(os.path.join(DATA_DIR, "*.dat")))
    if not candidates:
        raise FileNotFoundError(f"No .dat files found in {DATA_DIR}.")
    return candidates[0]


def _resolve_npz(dat_path):
    stem   = os.path.splitext(os.path.basename(dat_path))[0]
    tokens = [t for t in stem.lower().split("_") if t.endswith("k") and t[:-1].isdigit()]
    candidates = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    if tokens:
        matched = [c for c in candidates if tokens[0] in os.path.basename(c).lower()]
        if matched:
            return matched[0]
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No .npz stability file found in {DATA_DIR}.")


def load_data(dat_path):
    from phasenumber.composition import generate_composition_space
    npz_path = _resolve_npz(dat_path)
    print(f"  stability : {os.path.basename(npz_path)}")
    stability = np.load(npz_path)["phase_diagram_data"]

    print(f"  Generating composition grid (step=0.01) ...")
    t0 = time.time()
    comp_space, _ = generate_composition_space(step=0.01)
    print(f"  {len(comp_space):,} grid points in {time.time()-t0:.1f}s")

    if len(stability) != len(comp_space):
        raise ValueError(
            f"Length mismatch: composition grid has {len(comp_space):,} points "
            f"but .npz has {len(stability):,} entries."
        )

    data = np.column_stack([comp_space, stability])
    print(f"  {len(data):,} points ready")
    return data


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------

def render_frames(diag, frames_dir, window_size, force=False):
    """Render one PNG per x0 into frames_dir (skip if already cached)."""
    os.makedirs(frames_dir, exist_ok=True)
    paths = []
    for x0 in X0_VALUES:
        out = os.path.join(frames_dir, f"frame_x0_{x0:.2f}.png")
        if os.path.exists(out) and not force:
            print(f"  [cached] x(Fe)={x0:.2f}")
        else:
            t0 = time.time()
            n  = diag.save_frame_surface(
                x0=x0,
                out_path=out,
                window_size=window_size,
                show_scalebar=False,
                dpi=100,           # DPI irrelevant when scalebar is off
                max_points=50000,
            )
            print(f"  x(Fe)={x0:.2f}  {n:,} pts  ({time.time()-t0:.1f}s)")
        paths.append(out)
    return paths


# ---------------------------------------------------------------------------
# Image cropping helper
# ---------------------------------------------------------------------------

def _crop_white(img, border_px=4, threshold=0.97):
    """Crop white border from a float32 or uint8 RGBA/RGB image.

    Finds the bounding box of all pixels darker than `threshold` on any
    channel (ignoring the alpha channel when present), then expands by
    `border_px` on each side.  Returns the cropped array.
    """
    if img.dtype != np.float32 and img.dtype != np.float64:
        img_f = img.astype(np.float32) / 255.0
    else:
        img_f = img

    # Use only RGB channels
    rgb = img_f[:, :, :3]

    # Mask of non-white pixels (any channel below threshold)
    mask = np.any(rgb < threshold, axis=2)

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]

    if rows.size == 0 or cols.size == 0:
        return img   # fully white — return as-is

    r0 = max(0, rows[0]  - border_px)
    r1 = min(img.shape[0], rows[-1]  + border_px + 1)
    c0 = max(0, cols[0]  - border_px)
    c1 = min(img.shape[1], cols[-1]  + border_px + 1)

    return img[r0:r1, c0:c1]


def _load_and_crop(fpath, border_px=4):
    """Load a PNG and crop its white border."""
    img = mpimg.imread(fpath)
    return _crop_white(img, border_px=border_px)


# ---------------------------------------------------------------------------
# Grid compositing
# ---------------------------------------------------------------------------

def build_grid(frame_paths, out_pdf, out_png, output_dpi):
    """Composite 16 frames into a 4×4 publication figure."""

    # ------------------------------------------------------------------
    # Pre-load and crop all images so we know the common cropped aspect
    # ------------------------------------------------------------------
    print("  Cropping white borders ...")
    images = [_load_and_crop(fp) for fp in frame_paths]

    # Use the median cropped aspect ratio (H/W) across all frames
    aspects = [im.shape[0] / im.shape[1] for im in images]
    aspect  = float(np.median(aspects))

    # ------------------------------------------------------------------
    # Layout constants  (all in inches unless noted)
    # ------------------------------------------------------------------
    fig_w   = 7.20          # standard journal double-column width
    pad_l   = 0.15          # left margin
    pad_r   = 0.05          # right margin
    pad_t   = 0.08          # top margin
    pad_b   = 0.50          # bottom margin (space for legend)
    hgap    = 0.00          # horizontal gap between panels
    vgap    = 0.12          # vertical gap (room for x0 labels)

    n_col  = GRID_COLS
    n_row  = GRID_ROWS

    # Panel width from available space
    avail_w  = fig_w - pad_l - pad_r - (n_col - 1) * hgap
    panel_w  = avail_w / n_col
    panel_h  = panel_w * aspect

    avail_h = n_row * panel_h + (n_row - 1) * vgap
    fig_h   = avail_h + pad_t + pad_b

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=output_dpi)

    # Normalised coordinates helper
    def norm_x(x_in):
        return x_in / fig_w

    def norm_y(y_in):
        return y_in / fig_h

    # Add image axes bottom-to-top (matplotlib origin is bottom-left)
    for idx, (x0, img) in enumerate(zip(X0_VALUES, images)):
        row = idx // n_col     # 0 = top row in display
        col = idx %  n_col

        # Display row 0 → topmost in figure
        x_pos = pad_l + col * (panel_w + hgap)
        y_pos = fig_h - pad_t - (row + 1) * panel_h - row * vgap

        ax = fig.add_axes([
            norm_x(x_pos),
            norm_y(y_pos),
            norm_x(panel_w),
            norm_y(panel_h),
        ])

        ax.imshow(img, aspect="auto", interpolation="lanczos")
        ax.axis("off")

        # Label: x(Fe) = value
        ax.set_title(
            f"$x_{{\\mathrm{{Fe}}}} = {x0:.2f}$",
            fontsize=6.5,
            pad=2.0,
            color="black",
        )

    # ------------------------------------------------------------------
    # Shared legend (centred below the grid)
    # ------------------------------------------------------------------
    legend_y = norm_y(pad_b * 0.42)

    handles = []
    for label_val in [-1, 0, 1]:
        fc = PHASE_COLORS[label_val]
        ec = PHASE_EDGE[label_val]
        al = PHASE_ALPHAS[label_val]
        # For the legend patch alpha: stable is fully transparent in the render
        # but we still need a visible patch, so force alpha >= 0.25
        patch_alpha = max(al, 0.30)
        patch = mpatches.Patch(
            facecolor=(*fc, patch_alpha),
            edgecolor=ec,
            linewidth=0.6,
            label=PHASE_NAMES[label_val],
        )
        handles.append(patch)

    leg = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=3,
        fontsize=7.5,
        frameon=True,
        framealpha=0.9,
        edgecolor="#aaaaaa",
        handlelength=1.4,
        handleheight=0.9,
        handletextpad=0.5,
        columnspacing=1.0,
        title="Phase stability",
        title_fontsize=7.5,
    )
    leg.get_frame().set_linewidth(0.5)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    plt.savefig(out_pdf, dpi=output_dpi, bbox_inches="tight",
                facecolor="white", format="pdf")
    print(f"  PDF saved: {out_pdf}")

    plt.savefig(out_png, dpi=output_dpi, bbox_inches="tight",
                facecolor="white", format="png")
    print(f"  PNG saved: {out_png}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--file", metavar="FILENAME",
                        help="TCHEA4 .dat file (name or full path). "
                             "Defaults to the first .dat in data/.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Re-render frames even if already cached in output/grid_frames/")
    parser.add_argument("--window-size", nargs=2, type=int, default=[1050, 750],
                        metavar=("W", "H"),
                        help="PyVista window size for individual frames (default 1050 750)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Output DPI for PDF/PNG (default 300)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading data ...")
    dat_path = _resolve_dat(args.file)
    data     = load_data(dat_path)

    from phase5d import PhaseDiagram5D
    diag = PhaseDiagram5D(
        data,
        x0="implicit",
        value_type="phase_stability",
        component_labels=["Fe", "Mn", "Ni", "Co", "Cu"],
        phase_colors=PHASE_COLORS,
        phase_alphas=PHASE_ALPHAS,
        phase_names=PHASE_NAMES,
        tolerance=0.005,
    )

    # ── Render frames ──────────────────────────────────────────────────────────
    npz_path   = _resolve_npz(dat_path)
    stem       = os.path.splitext(os.path.basename(npz_path))[0]
    frames_dir = os.path.join(OUTPUT_DIR, f"grid_frames_{stem}")
    window_size = tuple(args.window_size)

    print(f"\nRendering {len(X0_VALUES)} frames  (window {window_size[0]}x{window_size[1]}) ...")
    print(f"Frames directory: {frames_dir}")
    t0 = time.time()
    frame_paths = render_frames(diag, frames_dir, window_size, force=args.no_cache)
    print(f"Frames ready in {time.time()-t0:.1f}s\n")

    # ── Composite grid ─────────────────────────────────────────────────────────
    out_pdf = os.path.join(OUTPUT_DIR, "stability_grid.pdf")
    out_png = os.path.join(OUTPUT_DIR, "stability_grid.png")
    print(f"Compositing 4x4 grid (DPI={args.dpi}) ...")
    t1 = time.time()
    build_grid(frame_paths, out_pdf, out_png, args.dpi)
    print(f"Grid done in {time.time()-t1:.1f}s")


if __name__ == "__main__":
    main()
