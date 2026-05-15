"""
Main PhaseDiagram5D class for rendering and exporting frames/videos.
"""

import os
import shutil
import tempfile
from typing import Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection

from .colormap import (
    DEFAULT_PHASE_ALPHAS,
    DEFAULT_PHASE_COLORS,
    continuous_colors,
    make_scalar_mappable,
    phase_stability_colors,
)
from .geometry import (
    CENTROID,
    EDGES,
    VERTICES,
    compositions_to_cartesian,
    tetrahedron_display_vertices,
)
from .utils import (
    compute_x0,
    downsample,
    extract_x0_slice,
    validate_data,
    x0_grid,
)


class PhaseDiagram5D:
    """
    Visualize a five-component alloy phase space as a sequence of
    tetrahedron frames, one per x0 slice.

    Parameters
    ----------
    data : array-like, shape (N, 5)
        Columns: [x1, x2, x3, x4, value].
        x0 is implicit: x0 = 1 - x1 - x2 - x3 - x4.
    value_type : {'continuous', 'phase_stability'}
        How to interpret the value column.
        - 'continuous'      : scalar property (Gibbs energy, enthalpy, …)
        - 'phase_stability' : integer labels -1 / 0 / 1
    colormap : str
        Matplotlib colormap for continuous values (default 'viridis').
    vmin, vmax : float or None
        Color-scale limits for continuous values.  Defaults to data min/max.
    tolerance : float
        Half-width of the x0 acceptance window when slicing (default 0.005).
    component_labels : list of 5 str or None
        Names shown at vertices and in the scale bar.
        Default: ['x₀', 'x₁', 'x₂', 'x₃', 'x₄'].
    phase_colors : dict or None
        Override default RGB colors for stability labels.
        Keys: -1, 0, 1 → (R, G, B) tuples in [0, 1].
    phase_alphas : dict or None
        Override default alpha values for stability labels.
        Keys: -1, 0, 1 → float in [0, 1].
    """

    def __init__(
        self,
        data,
        value_type: str = "continuous",
        colormap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        tolerance: float = 0.005,
        component_labels: Optional[List[str]] = None,
        phase_colors: Optional[Dict[int, Tuple[float, float, float]]] = None,
        phase_alphas: Optional[Dict[int, float]] = None,
    ):
        if value_type not in ("continuous", "phase_stability"):
            raise ValueError("value_type must be 'continuous' or 'phase_stability'.")

        self.data = validate_data(data)
        self.value_type = value_type
        self.colormap = colormap
        self.tolerance = tolerance
        self.component_labels = component_labels or ["x₀", "x₁", "x₂", "x₃", "x₄"]

        # Continuous color limits
        values = self.data[:, 4]
        self.vmin = float(values.min()) if vmin is None else vmin
        self.vmax = float(values.max()) if vmax is None else vmax

        # Phase stability colors / alphas
        self._phase_colors = {**DEFAULT_PHASE_COLORS, **(phase_colors or {})}
        self._phase_alphas = {**DEFAULT_PHASE_ALPHAS, **(phase_alphas or {})}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _map_colors(self, values: np.ndarray, alpha: float) -> np.ndarray:
        """Return RGBA array for *values* according to value_type."""
        if self.value_type == "continuous":
            rgba, _, _ = continuous_colors(
                values,
                cmap=self.colormap,
                vmin=self.vmin,
                vmax=self.vmax,
                alpha=alpha,
            )
        else:
            rgba = phase_stability_colors(
                values,
                phase_colors=self._phase_colors,
                phase_alphas=self._phase_alphas,
            )
        return rgba

    def _draw_wireframe(
        self,
        ax: Axes3D,
        x0: float,
        mode: str,
        wireframe_alpha: float,
        wireframe_color: str,
    ) -> None:
        verts = tetrahedron_display_vertices(x0=x0, mode=mode)
        for i, j in EDGES:
            ax.plot(
                [verts[i, 0], verts[j, 0]],
                [verts[i, 1], verts[j, 1]],
                [verts[i, 2], verts[j, 2]],
                color=wireframe_color,
                alpha=wireframe_alpha,
                linewidth=0.9,
            )

    def _label_vertices(self, ax: Axes3D, mode: str, x0: float) -> None:
        """Place component name labels at (or near) each tetrahedron vertex."""
        verts = tetrahedron_display_vertices(x0=x0, mode=mode)
        offset = 0.09
        for i, label in enumerate(self.component_labels[1:]):
            v = verts[i]
            # Push label away from centroid so it clears the wireframe
            direction = v - CENTROID
            norm = np.linalg.norm(direction)
            if norm > 1e-9:
                direction /= norm
            pos = v + offset * direction
            ax.text(
                pos[0], pos[1], pos[2],
                label,
                fontsize=11,
                ha="center",
                va="center",
                fontweight="bold",
            )

    def _draw_scale_bar(
        self,
        ax,
        x0: float,
    ) -> None:
        """Draw a horizontal scale bar showing current x0 and scale = 1-x0."""
        scale = 1.0 - x0
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        bar_y = 0.45
        bar_h = 0.35

        # Background
        ax.barh(bar_y, 1.0, left=0.0, height=bar_h,
                color="#e0e0e0", edgecolor="#888888", linewidth=0.8)
        # Filled (scale portion)
        if scale > 0:
            ax.barh(bar_y, scale, left=0.0, height=bar_h,
                    color="#4477aa", edgecolor="#888888", linewidth=0.8)

        # Text: component name and values
        lbl = self.component_labels[0]
        ax.text(
            0.5, 0.92,
            f"{lbl} = {x0:.3f}    ·    scale = {scale:.3f}",
            ha="center", va="center", fontsize=9,
            transform=ax.transAxes,
        )
        ax.text(0.0, 0.05, "0", ha="left", va="top", fontsize=7,
                transform=ax.transAxes)
        ax.text(1.0, 0.05, "1", ha="right", va="top", fontsize=7,
                transform=ax.transAxes)

    def _add_colorbar(self, fig, ax_left: float = 0.88) -> None:
        ax_cb = fig.add_axes([ax_left, 0.15, 0.02, 0.65])
        sm = make_scalar_mappable(self.colormap, self.vmin, self.vmax)
        cb = fig.colorbar(sm, cax=ax_cb)
        cb.ax.tick_params(labelsize=7)

    def _add_phase_legend(self, fig) -> None:
        labels = {-1: "Unstable", 0: "Meta-stable", 1: "Stable"}
        handles = []
        for label in (-1, 0, 1):
            r, g, b = self._phase_colors[label]
            a = max(self._phase_alphas[label], 0.25)  # show even if transparent
            patch = mpatches.Patch(
                facecolor=(r, g, b, a),
                edgecolor="gray",
                linewidth=0.5,
                label=f"{labels[label]} ({label:+d})",
            )
            handles.append(patch)
        fig.legend(
            handles=handles,
            loc="upper right",
            fontsize=8,
            framealpha=0.85,
            title="Phase stability",
            title_fontsize=8,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plot_frame(
        self,
        x0: float,
        mode: str = "fixed",
        fig=None,
        alpha: float = 0.65,
        marker_size: float = 3,
        max_points: int = 15000,
        show_wireframe: bool = True,
        wireframe_alpha: float = 0.20,
        wireframe_color: str = "black",
        show_vertex_labels: bool = True,
        elev: float = 20,
        azim: float = 45,
        figsize: Tuple[float, float] = (8, 9),
        title: Optional[str] = None,
        dpi: int = 100,
    ):
        """
        Render a single tetrahedron frame for a given x0 value.

        Parameters
        ----------
        x0 : float
            Composition value for the fifth component (0 ≤ x0 ≤ 1).
        mode : {'fixed', 'shrink_center', 'shrink_corner'}
            How the tetrahedron is scaled as x0 changes:
            - 'fixed'         : tetrahedron always fills the viewport; scale
                                shown via the bar below (recommended).
            - 'shrink_center' : tetrahedron shrinks toward its centroid.
            - 'shrink_corner' : tetrahedron shrinks toward the pure-x0 corner.
        fig : matplotlib Figure or None
            Re-use an existing figure if provided; otherwise a new one is created.
        alpha : float
            Point transparency for continuous values (0 = invisible, 1 = opaque).
            For phase_stability the per-phase alphas defined at construction are
            used instead.
        marker_size : float
            Scatter marker size in points².
        max_points : int
            Maximum number of data points to render (random sub-sample if exceeded).
        show_wireframe : bool
            Draw the tetrahedron edges.
        wireframe_alpha : float
            Transparency of the wireframe edges.
        wireframe_color : str
            Color of the wireframe edges.
        show_vertex_labels : bool
            Label each vertex with its component name.
        elev, azim : float
            3-D camera elevation and azimuth angles (degrees).
        figsize : (width, height)
            Figure size in inches.
        title : str or None
            Optional title for the 3-D axes.
        dpi : int
            Figure resolution.

        Returns
        -------
        fig : matplotlib Figure
        ax  : mpl_toolkits.mplot3d.Axes3D
        """
        if fig is None:
            fig = plt.figure(figsize=figsize, dpi=dpi)

        # 3-D axes — leave bottom strip for scale bar
        ax3d: Axes3D = fig.add_axes([0.02, 0.10, 0.84, 0.86], projection="3d")
        ax3d.view_init(elev=elev, azim=azim)
        ax3d.set_axis_off()

        if title:
            ax3d.set_title(title, fontsize=12, pad=8)

        # Wireframe
        if show_wireframe:
            self._draw_wireframe(ax3d, x0, mode, wireframe_alpha, wireframe_color)

        # Data points
        slice_data = extract_x0_slice(self.data, x0, self.tolerance)
        if len(slice_data) > 0:
            slice_data = downsample(slice_data, max_points)
            x1, x2, x3, x4 = (slice_data[:, k] for k in range(4))
            values = slice_data[:, 4]

            pts = compositions_to_cartesian(x1, x2, x3, x4, x0=x0, mode=mode)
            rgba = self._map_colors(values, alpha=alpha)

            # For phase_stability, skip fully-transparent (stable) points
            visible = rgba[:, 3] > 1e-3
            if visible.any():
                ax3d.scatter(
                    pts[visible, 0],
                    pts[visible, 1],
                    pts[visible, 2],
                    c=rgba[visible],
                    s=marker_size,
                    depthshade=False,
                    linewidths=0,
                )

        # Vertex labels
        if show_vertex_labels:
            self._label_vertices(ax3d, mode, x0)

        # Fix axis limits to the FULL tetrahedron so viewport never changes
        buf = 0.12
        ax3d.set_xlim(VERTICES[:, 0].min() - buf, VERTICES[:, 0].max() + buf)
        ax3d.set_ylim(VERTICES[:, 1].min() - buf, VERTICES[:, 1].max() + buf)
        ax3d.set_zlim(VERTICES[:, 2].min() - buf, VERTICES[:, 2].max() + buf)

        # Scale bar
        ax_bar = fig.add_axes([0.08, 0.02, 0.76, 0.06])
        self._draw_scale_bar(ax_bar, x0)

        # Legend / colorbar
        if self.value_type == "continuous":
            self._add_colorbar(fig)
        else:
            self._add_phase_legend(fig)

        return fig, ax3d

    def save_frames(
        self,
        x0_values: Sequence[float],
        output_dir: str,
        dpi: int = 150,
        verbose: bool = True,
        **plot_kwargs,
    ) -> List[str]:
        """
        Render and save one PNG frame per x0 value.

        Parameters
        ----------
        x0_values : sequence of float
            x0 values to render, in the desired frame order.
        output_dir : str
            Directory where frames are saved (created if absent).
        dpi : int
            PNG resolution.
        verbose : bool
            Print progress.
        **plot_kwargs
            Forwarded to :meth:`plot_frame`.

        Returns
        -------
        list of str
            Absolute paths of saved PNG files (in order).
        """
        os.makedirs(output_dir, exist_ok=True)
        paths: List[str] = []

        for i, x0 in enumerate(x0_values):
            fig, _ = self.plot_frame(x0, dpi=dpi, **plot_kwargs)
            path = os.path.join(output_dir, f"frame_{i:04d}.png")
            fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
            plt.close(fig)
            paths.append(os.path.abspath(path))

            if verbose:
                n_slice = len(extract_x0_slice(self.data, x0, self.tolerance))
                print(
                    f"  [{i + 1:>{len(str(len(x0_values)))}}/{len(x0_values)}] "
                    f"x0 = {x0:.3f}  ({n_slice} points)  → {path}"
                )

        return paths

    def create_video(
        self,
        x0_values: Optional[Sequence[float]] = None,
        output_path: str = "phase_diagram.mp4",
        fps: int = 10,
        dpi: int = 150,
        keep_frames: bool = False,
        frames_dir: Optional[str] = None,
        verbose: bool = True,
        **plot_kwargs,
    ) -> str:
        """
        Render frames and assemble them into an mp4 video via ffmpeg.

        Parameters
        ----------
        x0_values : sequence of float or None
            x0 values to sweep.  Defaults to the full grid inferred from the
            data at the resolution set by ``step`` (pass ``step`` in plot_kwargs
            or rely on the default 0.01).
        output_path : str
            Destination file path (should end in .mp4).
        fps : int
            Video frame rate.
        dpi : int
            Frame image resolution.
        keep_frames : bool
            Retain the individual PNG frames after the video is assembled.
        frames_dir : str or None
            Where to store frames.  A temporary directory is used when None
            and deleted afterwards unless keep_frames=True.
        verbose : bool
            Print progress.
        **plot_kwargs
            Forwarded to :meth:`plot_frame` (e.g. mode, alpha, marker_size).

        Returns
        -------
        str
            Absolute path of the created video file.
        """
        from .video import assemble_video

        # Default x0 grid
        if x0_values is None:
            step = plot_kwargs.pop("step", 0.01)
            x0_values = x0_grid(self.data, step=step)

        # Frame directory
        _tmp_dir: Optional[str] = None
        if frames_dir is None:
            _tmp_dir = tempfile.mkdtemp(prefix="phase5d_")
            frames_dir = _tmp_dir

        try:
            frame_paths = self.save_frames(
                x0_values, frames_dir, dpi=dpi, verbose=verbose, **plot_kwargs
            )
            assemble_video(frame_paths, output_path, fps=fps, verbose=verbose)
        finally:
            if _tmp_dir is not None and not keep_frames:
                shutil.rmtree(_tmp_dir, ignore_errors=True)

        return os.path.abspath(output_path)
