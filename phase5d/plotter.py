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
    combined_colors,
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
    data : array-like
        Composition + value array.  The expected shape depends on *x0*:

        ``x0='implicit'`` (default) — shape **(N, 5)**,
        columns ``[x1, x2, x3, x4, value]``.
        x0 is computed internally as ``1 - x1 - x2 - x3 - x4``.

        ``x0='explicit'`` — shape **(N, 6)**,
        columns ``[x0, x1, x2, x3, x4, value]``.
        The dependent coordinate is taken directly from the first column.

    x0 : {'implicit', 'explicit'}
        Declares whether the dependent composition coordinate x0 is
        already included in *data* or should be derived from the others.
        Default: ``'implicit'``.

    value_type : {'continuous', 'phase_stability'}
        How to interpret the value column.
        - ``'continuous'``      : scalar property (Gibbs energy, enthalpy, …)
        - ``'phase_stability'`` : integer labels -1 / 0 / 1

    colormap : str
        Matplotlib colormap for continuous values (default ``'viridis'``).

    vmin, vmax : float or None
        Color-scale limits for continuous values.  Defaults to data min/max.

    tolerance : float
        Half-width of the x0 acceptance window when slicing (default 0.005).

    component_labels : list of 5 str or None
        Names shown at tetrahedron vertices and in the x0 scale bar,
        ordered ``[x0_label, x1_label, x2_label, x3_label, x4_label]``.

        Example for a Fe-Mn-Ni-Co-Cu alloy::

            component_labels=['Fe', 'Mn', 'Ni', 'Co', 'Cu']

        Default: ``['x₀', 'x₁', 'x₂', 'x₃', 'x₄']``.

    stability_data : array-like of int, shape (N,), or None
        Optional array of phase stability labels (-1, 0, 1), one per row in
        *data*.  Only used when ``value_type='continuous'``.  When provided,
        enables **combined mode**: color is taken from the continuous value
        (Gm, Hmr, …) while *alpha* is controlled by the stability label —
        stable regions stay invisible, meta-stable regions are
        semi-transparent, and unstable regions are fully opaque.

        Example::

            diag = PhaseDiagram5D(
                data_gm,
                value_type='continuous',
                stability_data=phase_labels,
                component_labels=['Fe', 'Mn', 'Ni', 'Co', 'Cu'],
            )

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
        x0: str = "implicit",
        value_type: str = "continuous",
        colormap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        tolerance: float = 0.005,
        component_labels: Optional[List[str]] = None,
        stability_data=None,
        phase_colors: Optional[Dict[int, Tuple[float, float, float]]] = None,
        phase_alphas: Optional[Dict[int, float]] = None,
    ):
        if value_type not in ("continuous", "phase_stability"):
            raise ValueError("value_type must be 'continuous' or 'phase_stability'.")

        # Internal storage is always (N, 6): [x0, x1, x2, x3, x4, value]
        self.data = validate_data(data, x0=x0)
        self.x0_mode = x0
        self.value_type = value_type
        self.colormap = colormap
        self.tolerance = tolerance
        self.component_labels = component_labels or ["x₀", "x₁", "x₂", "x₃", "x₄"]

        # Continuous color limits  (value is at column 5 in the (N,6) internal array)
        values = self.data[:, 5]
        self.vmin = float(values.min()) if vmin is None else vmin
        self.vmax = float(values.max()) if vmax is None else vmax

        # Phase stability colors / alphas
        self._phase_colors = {**DEFAULT_PHASE_COLORS, **(phase_colors or {})}
        self._phase_alphas = {**DEFAULT_PHASE_ALPHAS, **(phase_alphas or {})}

        # Optional stability mask for combined rendering
        if stability_data is not None:
            stab = np.asarray(stability_data, dtype=int).ravel()
            if len(stab) != len(self.data):
                raise ValueError(
                    f"stability_data length ({len(stab)}) must match data "
                    f"length ({len(self.data)})."
                )
            if not np.all(np.isin(stab, [-1, 0, 1])):
                raise ValueError(
                    "stability_data must contain only -1 (unstable), "
                    "0 (meta-stable), or 1 (stable)."
                )
            self._stability_data: Optional[np.ndarray] = stab
        else:
            self._stability_data = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _map_colors(
        self,
        values: np.ndarray,
        alpha: float,
        stability: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return RGBA array for *values* according to value_type.

        Parameters
        ----------
        values : np.ndarray
            Value column for the current slice.
        alpha : float
            Uniform alpha used in pure continuous mode.
        stability : np.ndarray or None
            Stability labels for the current slice.  When provided together
            with value_type='continuous', triggers combined rendering.
        """
        if self.value_type == "continuous":
            if stability is not None:
                # Combined: hue from continuous value, alpha from stability
                rgba, _, _ = combined_colors(
                    values,
                    stability,
                    cmap=self.colormap,
                    vmin=self.vmin,
                    vmax=self.vmax,
                    phase_alphas=self._phase_alphas,
                )
            else:
                rgba, _, _ = continuous_colors(
                    values,
                    cmap=self.colormap,
                    vmin=self.vmin,
                    vmax=self.vmax,
                    alpha=alpha,
                )
        else:  # 'phase_stability'
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
    # Render helpers (called by plot_frame)
    # ------------------------------------------------------------------

    def _render_scatter(
        self,
        ax3d: Axes3D,
        pts: np.ndarray,
        rgba: np.ndarray,
        marker_size: float,
        **kwargs,
    ) -> None:
        """Render data points as a scatter plot (default render mode).

        *kwargs* are merged on top of the library defaults and forwarded to
        ``Axes3D.scatter``, so any parameter accepted by that call can be
        overridden (e.g. ``marker='^'``, ``edgecolors='k'``).
        """
        visible = rgba[:, 3] > 1e-3
        if visible.any():
            kw = dict(
                c=rgba[visible],
                s=marker_size,
                depthshade=False,
                linewidths=0,
            )
            kw.update(kwargs)   # user overrides win
            ax3d.scatter(
                pts[visible, 0],
                pts[visible, 1],
                pts[visible, 2],
                **kw,
            )

    def _render_surface(
        self,
        ax3d: Axes3D,
        pts: np.ndarray,
        rgba: np.ndarray,
        values: np.ndarray,
        stability: Optional[np.ndarray],
        **kwargs,
    ) -> None:
        """
        Render data as convex-hull surfaces, one hull per phase region.

        For ``value_type='phase_stability'`` a separate hull is drawn for each
        stability class (unstable and meta-stable; stable is invisible).
        For ``value_type='continuous'`` (with or without combined stability
        masking) the hull faces are colored by the mean scalar value of their
        three vertices, taken from the diagram's colormap.

        Falls back silently to nothing on degenerate input (< 4 points or
        coplanar configuration).  Requires ``scipy >= 1.0``.

        *kwargs* are merged on top of the library defaults and forwarded to
        ``Poly3DCollection`` (e.g. ``edgecolor='white'``, ``linewidth=0.3``).
        Note: ``facecolor`` is always controlled by the colormap / phase colors;
        pass ``edgecolor`` or other non-color properties to customise the mesh.
        """
        try:
            from scipy.spatial import ConvexHull
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        except ImportError as exc:
            raise ImportError(
                "render='surface' requires scipy.  "
                "Install it with:  pip install scipy"
            ) from exc

        import matplotlib.cm as _mcm
        import matplotlib.colors as _mcolors

        cm = _mcm.get_cmap(self.colormap)
        norm = _mcolors.Normalize(vmin=self.vmin, vmax=self.vmax)

        def _add_hull(sub_pts, sub_vals, face_alpha, solid_color=None):
            """
            Compute ConvexHull of *sub_pts*, build a Poly3DCollection, and
            add it to ax3d.

            If *solid_color* is given (RGB tuple) every face gets that color.
            Otherwise face colors are derived from *sub_vals* via the colormap.
            """
            if len(sub_pts) < 4:
                return
            try:
                hull = ConvexHull(sub_pts)
            except Exception:
                return
            triangles = [sub_pts[s] for s in hull.simplices]

            if solid_color is not None:
                r, g, b = solid_color
                fcolors = np.tile([r, g, b, face_alpha], (len(hull.simplices), 1))
            else:
                fcolors = np.array([
                    [*cm(norm(sub_vals[s].mean()))[:3], face_alpha]
                    for s in hull.simplices
                ])

            poly_kw = dict(edgecolor="none")
            poly_kw.update(kwargs)   # user overrides win
            poly = Poly3DCollection(triangles, **poly_kw)
            poly.set_facecolor(fcolors)   # always driven by colormap/phase
            ax3d.add_collection3d(poly)

        if self.value_type == "phase_stability":
            # values column IS the stability label; one hull per class
            for label in (-1, 0):          # label 1 = stable = invisible
                a = self._phase_alphas[label]
                if a < 1e-3:
                    continue
                mask = values.astype(int) == label
                if mask.sum() < 4:
                    continue
                _add_hull(pts[mask], None, a, solid_color=self._phase_colors[label])

        elif stability is not None:
            # Combined mode: group by stability, colormap hue from values
            for label in (-1, 0):
                a = self._phase_alphas[label]
                if a < 1e-3:
                    continue
                mask = stability == label
                if mask.sum() < 4:
                    continue
                _add_hull(pts[mask], values[mask], a)

        else:
            # Pure continuous: one hull for all visible points
            visible = rgba[:, 3] > 1e-3
            if visible.sum() < 4:
                return
            mean_alpha = float(rgba[visible, 3].mean())
            _add_hull(pts[visible], values[visible], mean_alpha)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plot_frame(
        self,
        x0: float,
        mode: str = "fixed",
        render: str = "scatter",
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
        **kwargs,
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
        render : {'scatter', 'surface'}
            Rendering style for the data points in this slice:

            - ``'scatter'`` *(default)* — each composition point is drawn as
              an individual marker.  Fast and faithful to the raw data
              distribution; ``marker_size`` and ``max_points`` apply.

            - ``'surface'`` — the convex hull of each phase region is rendered
              as a solid (or semi-transparent) polygon mesh.  For
              ``value_type='phase_stability'`` a separate hull is built for
              each stability class.  For ``value_type='continuous'`` (or
              combined mode) hull faces are colored by the mean scalar value
              of their vertices, taken from the diagram colormap.
              Requires ``scipy``.

        fig : matplotlib Figure or None
            Re-use an existing figure if provided; otherwise a new one is created.
        alpha : float
            Point transparency for continuous values (0 = invisible, 1 = opaque).
            For phase_stability the per-phase alphas defined at construction are
            used instead.
        marker_size : float
            Scatter marker size in points² (``render='scatter'`` only).
        max_points : int
            Maximum number of data points to render (random sub-sample if exceeded).
            Applied before both scatter and surface rendering.
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
        **kwargs
            Passed directly to the underlying matplotlib rendering call,
            on top of the library's own defaults (user values take priority):

            - ``render='scatter'`` → forwarded to ``Axes3D.scatter``
              (e.g. ``marker='^'``, ``edgecolors='k'``, ``linewidths=0.5``).
            - ``render='surface'`` → forwarded to ``Poly3DCollection``
              (e.g. ``edgecolor='white'``, ``linewidth=0.2``).

            These kwargs are also accepted by :meth:`save_frames` and
            :meth:`create_video`, which forward all extra keyword arguments
            through to this method.

        Returns
        -------
        fig : matplotlib Figure
        ax  : mpl_toolkits.mplot3d.Axes3D
        """
        if render not in ("scatter", "surface"):
            raise ValueError("render must be 'scatter' or 'surface'.")
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

        # Data points  (internal format: [x0, x1, x2, x3, x4, value])
        # Compute the x0 mask directly so we can apply it to stability_data too
        x0_mask = np.abs(self.data[:, 0] - x0) <= self.tolerance
        slice_data = self.data[x0_mask]
        stab_slice = (
            self._stability_data[x0_mask]
            if self._stability_data is not None
            else None
        )

        if len(slice_data) > 0:
            # Downsample — apply the same random indices to both arrays
            if len(slice_data) > max_points:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(slice_data), size=max_points, replace=False)
                slice_data = slice_data[idx]
                if stab_slice is not None:
                    stab_slice = stab_slice[idx]

            x1, x2, x3, x4 = (slice_data[:, k] for k in range(1, 5))
            values = slice_data[:, 5]

            pts = compositions_to_cartesian(x1, x2, x3, x4, x0=x0, mode=mode)
            rgba = self._map_colors(values, alpha=alpha, stability=stab_slice)

            if render == "surface":
                self._render_surface(ax3d, pts, rgba, values, stab_slice, **kwargs)
            else:  # 'scatter' (default)
                self._render_scatter(ax3d, pts, rgba, marker_size, **kwargs)

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
            if self._stability_data is not None:
                # Combined mode: also show stability alpha legend
                self._add_phase_legend(fig)
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

    def plot_isosurface(
        self,
        level: Union[float, Sequence[float]],
        colors: Union[str, Sequence] = "auto",
        alpha: float = 0.5,
        grid_resolution: int = 50,
        max_points: int = 50000,
        fig=None,
        elev: float = 20,
        azim: float = 45,
        figsize: Tuple[float, float] = (8, 8),
        title: Optional[str] = None,
        dpi: int = 100,
        show_wireframe: bool = True,
        wireframe_alpha: float = 0.20,
        wireframe_color: str = "black",
        show_vertex_labels: bool = True,
        show_colorbar: bool = True,
        **kwargs,
    ) -> Tuple[plt.Figure, Axes3D]:
        """
        Render one or more isosurfaces through the full 4-simplex composition space.

        Unlike :meth:`plot_frame`, which shows one x₀ slice at a time, this
        method uses the **natural (barycentric) embedding**

        .. code-block:: text

            P = x₁·V₀ + x₂·V₁ + x₃·V₂ + x₄·V₃

        to place every data point at a unique position inside the master
        tetrahedron (x₀ is *not* normalised out).  A marching-cubes algorithm
        then extracts the closed surface(s) where the value column equals
        *level*, giving a true 3-D isosurface that spans all x₀ slices.

        Requires ``scipy`` and ``scikit-image``::

            pip install scipy scikit-image

        Parameters
        ----------
        level : float or sequence of float
            Value(s) at which to cut the isosurface.  Multiple levels can be
            passed as a list; each gets its own color.
        colors : 'auto' or sequence of color specs
            Colors for each isosurface.  ``'auto'`` cycles through
            ``matplotlib``'s ``tab10`` palette.  Otherwise supply one color
            per level using any matplotlib color specification.
        alpha : float
            Surface transparency (0 = invisible, 1 = fully opaque).
        grid_resolution : int
            Number of voxels along each axis of the interpolation grid.
            Higher values give smoother surfaces at the cost of memory and
            time.  Default: 50.
        max_points : int
            Maximum number of data points used to build the
            ``LinearNDInterpolator``.  A random subset is drawn when the
            dataset is larger.  Default: 50 000.
        fig : matplotlib Figure or None
            Re-use an existing figure, or create a new one.
        elev, azim : float
            Camera elevation and azimuth (degrees).
        figsize : (width, height)
            Figure size in inches.
        title : str or None
            Optional axes title.
        dpi : int
            Figure resolution.
        show_wireframe : bool
            Draw the master tetrahedron wireframe.
        wireframe_alpha : float
            Wireframe transparency.
        wireframe_color : str
            Wireframe line color.
        show_vertex_labels : bool
            Label vertices with component names.
        show_colorbar : bool
            Add a colorbar (continuous value_type only).
        **kwargs
            Passed directly to ``Poly3DCollection`` for every isosurface mesh,
            on top of the library defaults (user values take priority).
            Useful for e.g. ``edgecolor='white'``, ``linewidth=0.2``, or
            overriding ``alpha`` per call (though the *alpha* parameter above
            is more convenient for that).

        Returns
        -------
        fig : matplotlib Figure
        ax  : mpl_toolkits.mplot3d.Axes3D

        Examples
        --------
        Single isosurface at Gm = -5 kJ/mol::

            fig, ax = diagram.plot_isosurface(level=-5000)
            fig.savefig('isosurface.png', dpi=150, bbox_inches='tight')

        Multiple levels with custom colors::

            fig, ax = diagram.plot_isosurface(
                level=[-8000, -5000, -2000],
                colors=['royalblue', 'gold', 'tomato'],
                alpha=0.4,
                grid_resolution=60,
            )
        """
        try:
            from scipy.interpolate import LinearNDInterpolator
            from skimage.measure import marching_cubes
        except ImportError as exc:
            raise ImportError(
                "plot_isosurface() requires scipy and scikit-image.  "
                "Install them with:  pip install scipy scikit-image"
            ) from exc

        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        import matplotlib.cm as _mcm

        # ------------------------------------------------------------------ #
        # 1. Natural embedding: P = x1·V0 + x2·V1 + x3·V2 + x4·V3          #
        #    Every composition maps to a UNIQUE 3-D position inside the       #
        #    master tetrahedron (x0 is NOT normalised away).                  #
        # ------------------------------------------------------------------ #
        x1 = self.data[:, 1]
        x2 = self.data[:, 2]
        x3 = self.data[:, 3]
        x4 = self.data[:, 4]
        values = self.data[:, 5]

        pts3d = (
            x1[:, None] * VERTICES[0]
            + x2[:, None] * VERTICES[1]
            + x3[:, None] * VERTICES[2]
            + x4[:, None] * VERTICES[3]
        )  # shape (N, 3)

        # ------------------------------------------------------------------ #
        # 2. Downsample before building the interpolator                      #
        # ------------------------------------------------------------------ #
        N = len(pts3d)
        if N > max_points:
            rng = np.random.default_rng(42)
            idx = rng.choice(N, size=max_points, replace=False)
            pts3d_sub = pts3d[idx]
            vals_sub = values[idx]
        else:
            pts3d_sub = pts3d
            vals_sub = values

        # ------------------------------------------------------------------ #
        # 3. Build a LinearNDInterpolator on the 3-D scattered points         #
        # ------------------------------------------------------------------ #
        interp = LinearNDInterpolator(pts3d_sub, vals_sub, fill_value=np.nan)

        # Bounding box of the master tetrahedron
        x_lo, x_hi = float(VERTICES[:, 0].min()), float(VERTICES[:, 0].max())
        y_lo, y_hi = float(VERTICES[:, 1].min()), float(VERTICES[:, 1].max())
        z_lo, z_hi = float(VERTICES[:, 2].min()), float(VERTICES[:, 2].max())

        gx = np.linspace(x_lo, x_hi, grid_resolution)
        gy = np.linspace(y_lo, y_hi, grid_resolution)
        gz = np.linspace(z_lo, z_hi, grid_resolution)
        GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing="ij")
        grid_pts = np.column_stack([GX.ravel(), GY.ravel(), GZ.ravel()])

        grid_vals = interp(grid_pts).reshape(
            grid_resolution, grid_resolution, grid_resolution
        )

        # ------------------------------------------------------------------ #
        # 4. Figure and axes setup                                             #
        # ------------------------------------------------------------------ #
        if fig is None:
            fig = plt.figure(figsize=figsize, dpi=dpi)

        ax3d: Axes3D = fig.add_axes([0.02, 0.02, 0.86, 0.96], projection="3d")
        ax3d.view_init(elev=elev, azim=azim)
        ax3d.set_axis_off()

        if title:
            ax3d.set_title(title, fontsize=12, pad=8)

        # Wireframe uses shrink_corner at x0=0 → full tetrahedron = VERTICES
        if show_wireframe:
            self._draw_wireframe(
                ax3d, x0=0.0, mode="shrink_corner",
                wireframe_alpha=wireframe_alpha,
                wireframe_color=wireframe_color,
            )

        if show_vertex_labels:
            self._label_vertices(ax3d, mode="shrink_corner", x0=0.0)

        # ------------------------------------------------------------------ #
        # 5. Marching cubes for each requested level                           #
        # ------------------------------------------------------------------ #
        levels = [level] if np.isscalar(level) else list(level)

        if colors == "auto":
            cmap_tab = _mcm.get_cmap("tab10")
            color_list = [cmap_tab(i % 10) for i in range(len(levels))]
        else:
            color_list = list(colors)
            while len(color_list) < len(levels):
                color_list.append(color_list[-1])

        spacing = (
            (x_hi - x_lo) / max(grid_resolution - 1, 1),
            (y_hi - y_lo) / max(grid_resolution - 1, 1),
            (z_hi - z_lo) / max(grid_resolution - 1, 1),
        )

        # Replace NaN (outside the tetrahedron) with a sentinel ABOVE the data
        # maximum.  This ensures the exterior always appears "above" every
        # requested level, so marching_cubes finds only closed surfaces inside
        # the tetrahedron and never generates spurious flat caps at its faces.
        valid_mask = ~np.isnan(grid_vals)
        if valid_mask.any():
            sentinel = float(grid_vals[valid_mask].max()) + 1.0
        else:
            sentinel = 1.0
        grid_filled = np.where(valid_mask, grid_vals, sentinel)

        for lev, color in zip(levels, color_list):
            # Skip levels outside the actual data range
            if valid_mask.any():
                lo = float(grid_vals[valid_mask].min())
                hi = float(grid_vals[valid_mask].max())
                if not (lo < lev < hi):
                    continue
            try:
                verts, faces, _, _ = marching_cubes(
                    grid_filled,
                    level=lev,
                    spacing=spacing,
                )
            except Exception:
                continue

            # Shift vertices from voxel-space to world coordinates
            verts[:, 0] += x_lo
            verts[:, 1] += y_lo
            verts[:, 2] += z_lo

            mesh_kw = dict(alpha=alpha, facecolor=color, edgecolor="none")
            mesh_kw.update(kwargs)   # user overrides win
            mesh = Poly3DCollection(verts[faces], **mesh_kw)
            ax3d.add_collection3d(mesh)

        # ------------------------------------------------------------------ #
        # 6. Axis limits and legend/colorbar                                   #
        # ------------------------------------------------------------------ #
        buf = 0.12
        ax3d.set_xlim(VERTICES[:, 0].min() - buf, VERTICES[:, 0].max() + buf)
        ax3d.set_ylim(VERTICES[:, 1].min() - buf, VERTICES[:, 1].max() + buf)
        ax3d.set_zlim(VERTICES[:, 2].min() - buf, VERTICES[:, 2].max() + buf)

        if show_colorbar and self.value_type == "continuous":
            self._add_colorbar(fig, ax_left=0.88)

        if self.value_type == "phase_stability":
            self._add_phase_legend(fig)

        return fig, ax3d
