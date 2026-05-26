"""
Main PhaseDiagram5D class for rendering and exporting frames/videos.
"""

import base64
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


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _alpha_boundary_faces(pts: np.ndarray, shape_alpha: float):
    """
    Compute the boundary triangle indices of a 3-D alpha shape.

    A Delaunay tetrahedron is kept if its circumradius R < 1 / shape_alpha.
    Boundary faces are those belonging to exactly one kept tetrahedron.

    Parameters
    ----------
    pts : np.ndarray, shape (N, 3)
    shape_alpha : float
        Circumradius filter: keep tetrahedra with R < 1/shape_alpha.

    Returns
    -------
    np.ndarray, shape (M, 3) of int
        Row indices into *pts* for each boundary triangle, or an empty
        array if fewer than 4 points or no valid tetrahedra remain.
    """
    from scipy.spatial import Delaunay

    if len(pts) < 4:
        return np.empty((0, 3), dtype=np.intp)
    try:
        tri = Delaunay(pts)
    except Exception:
        return np.empty((0, 3), dtype=np.intp)

    vs = tri.simplices
    A, B, C, D = pts[vs[:, 0]], pts[vs[:, 1]], pts[vs[:, 2]], pts[vs[:, 3]]

    M_mat = 2.0 * np.stack([B - A, C - A, D - A], axis=1)   # (k,3,3)
    rhs   = np.stack([
        (B * B).sum(1) - (A * A).sum(1),
        (C * C).sum(1) - (A * A).sum(1),
        (D * D).sum(1) - (A * A).sum(1),
    ], axis=1)                                                 # (k,3)

    dets = np.linalg.det(M_mat)
    good = np.abs(dets) > 1e-14
    R    = np.full(len(vs), np.inf)
    if good.any():
        centers  = np.linalg.solve(
            M_mat[good], rhs[good, :, np.newaxis]
        ).squeeze(-1)
        R[good] = np.linalg.norm(centers - A[good], axis=1)

    valid = vs[R < 1.0 / shape_alpha]
    if len(valid) == 0:
        return np.empty((0, 3), dtype=np.intp)

    all_faces = np.concatenate([
        np.sort(valid[:, [0, 1, 2]], axis=1),
        np.sort(valid[:, [0, 1, 3]], axis=1),
        np.sort(valid[:, [0, 2, 3]], axis=1),
        np.sort(valid[:, [1, 2, 3]], axis=1),
    ])
    _, inv, counts = np.unique(
        all_faces, axis=0, return_inverse=True, return_counts=True
    )
    boundary = all_faces[counts[inv] == 1]
    return boundary  # (M, 3) int


# Adaptive alpha calibration constants
# Adaptive alpha calibration constants.
# PyVista backend: calibrated at max_points=50 000 (the actual N fed into
# Delaunay when a slice has >50 000 points).  For sparse slices (N < 50 000)
# the full slice is used and alpha scales down proportionally.
# Matplotlib backend: no max_points cap, calibrated at x0=0.30 (62 196 pts).
_ALPHA_REF_MPL = 2.0
_ALPHA_REF_PV  = 90.0
_N_REF_MPL     = 62_196   # matplotlib: no subsampling, calibrated at x0=0.30
_N_REF_PV      = 62_196   # PyVista: no per-phase subsampling, calibrated at x0=0.30

# PyVista camera / label defaults (work for any 5-component tetrahedron)
_PV_CAM_POS = np.array([3.0, -2.0, 1.4])
_PV_FOCAL   = np.array([0.5, 0.3,  0.2])
_PV_CAM_UP  = np.array([0.0, 0.0,  1.0])
_PV_LABEL_PUSH = [0.14, 0.10, 0.10, 0.05]   # per-vertex outward push

# Scale bar height appended below each PyVista frame (pixels)
_PV_SCALE_BAR_PX = 76


def _add_pv_scale_bar(
    img_arr: np.ndarray,
    out_path: str,
    x0: float,
    x0_label: str,
    legend_entries=None,
) -> None:
    """
    Composite a PyVista screenshot with a scale bar strip and save.

    A horizontal progress bar is rendered below the PyVista image using
    matplotlib so the visual language (colour, font) matches the scatter
    renderer.  The filled (blue) portion = 1 − x0 indicates what fraction
    of the full quaternary composition space is currently displayed.

    An optional phase legend is drawn in the lower-left corner of the PyVista
    image area using matplotlib patches — this gives full font-size control
    that PyVista's own add_legend does not expose.

    Parameters
    ----------
    img_arr : np.ndarray, shape (H, W, 3), uint8
        Raw PyVista screenshot.
    out_path : str
        Destination PNG file.
    x0 : float
        Current x₀ value.
    x0_label : str
        Name of the x₀ component (e.g. ``'Fe'``).
    legend_entries : list of (str, (R, G, B)) or None
        Phase legend entries to draw as coloured patches.
    """
    import matplotlib.patches as _mpatches
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    h, w   = img_arr.shape[:2]
    bar_px = _PV_SCALE_BAR_PX
    total  = h + bar_px
    dpi    = 100

    fig = plt.figure(figsize=(w / dpi, total / dpi), dpi=dpi)
    fig.patch.set_facecolor("white")

    # ── main PyVista image (top) ──────────────────────────────────────────
    ax_img = fig.add_axes([0.0, bar_px / total, 1.0, h / total])
    ax_img.imshow(img_arr)
    ax_img.set_axis_off()

    # ── matplotlib legend overlay (lower-left of the image area) ─────────
    if legend_entries:
        fs   = 10         # font size — freely adjustable here
        pad  = 6          # pixels of padding inside the box
        lh   = fs + 6     # row height in pixels
        sw   = fs + 2     # colour swatch width in pixels
        gap  = 5          # gap between swatch and text

        n    = len(legend_entries)
        box_h_px = n * lh + 2 * pad
        # 0.72 px per character per pt is a conservative upper bound for
        # typical sans-serif fonts at 100 dpi — keeps all labels inside the box
        max_chars = max(len(name) for name, _ in legend_entries)
        box_w_px  = sw + gap + int(max_chars * fs * 0.72) + 2 * pad

        # Convert pixel dimensions to figure-fraction coordinates
        fig_w_px = w
        fig_h_px = total
        # The image occupies the top h/total of the figure; position box at
        # lower-left of the image area with a small margin.
        margin_px = 10
        x0_fig = margin_px / fig_w_px
        y0_fig = (bar_px + margin_px) / fig_h_px
        bw_fig = box_w_px / fig_w_px
        bh_fig = box_h_px / fig_h_px

        ax_leg = fig.add_axes([x0_fig, y0_fig, bw_fig, bh_fig])
        ax_leg.set_xlim(0, box_w_px)
        ax_leg.set_ylim(0, box_h_px)
        ax_leg.axis("off")

        # Gray background box
        ax_leg.add_patch(_mpatches.FancyBboxPatch(
            (0, 0), box_w_px, box_h_px,
            boxstyle="square,pad=0",
            facecolor="#cccccc", edgecolor="#888888", linewidth=0.8,
            transform=ax_leg.transData, zorder=1,
        ))

        for i, (name, color) in enumerate(legend_entries):
            row_y = box_h_px - pad - (i + 1) * lh + (lh - sw) / 2
            # Colour swatch
            ax_leg.add_patch(_mpatches.Rectangle(
                (pad, row_y), sw, sw,
                facecolor=color, edgecolor="#555555", linewidth=0.5,
                transform=ax_leg.transData, zorder=2,
            ))
            # Label
            ax_leg.text(
                pad + sw + gap, row_y + sw / 2,
                name,
                va="center", ha="left",
                fontsize=fs, color="black",
                transform=ax_leg.transData, zorder=2,
            )

    # ── scale bar strip (bottom) ──────────────────────────────────────────
    ax_b = fig.add_axes([0.04, 0.002, 0.92, (bar_px / total) * 0.88])
    scale = 1.0 - x0
    ax_b.set_xlim(0, 1)
    ax_b.set_ylim(0, 1)
    ax_b.axis("off")

    bar_y = 0.08
    bar_h = 0.72
    lbl_y   = bar_y + 0.03            # just above bottom edge; text grows upward
    tick_bot = bar_y + bar_h * 0.42   # ticks occupy upper ~58 % of bar
    tick_top = bar_y + bar_h

    ax_b.barh(bar_y, 1.0,   left=0.0, height=bar_h,
              color="#e0e0e0", edgecolor="#888888", linewidth=0.8)
    if scale > 0:
        ax_b.barh(bar_y, scale, left=0.0, height=bar_h,
                  color="#4477aa", edgecolor="#888888", linewidth=0.8)

    # Vertical tick lines (short, upper portion) + labels at bottom of bar
    ax_b.text(0.0, lbl_y, "0", ha="left",  va="bottom", fontsize=7, color="black")
    ax_b.text(1.0, lbl_y, "1", ha="right", va="bottom", fontsize=7, color="black")
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        ax_b.plot([t, t], [tick_bot, tick_top],
                  color="white", linewidth=0.9, zorder=3, solid_capstyle="butt")
        ax_b.text(t, lbl_y, f"{t:.1f}", ha="center", va="bottom", fontsize=7, color="black")

    ax_b.text(
        0.5, 0.97,
        f"composition scale  (1 − x({x0_label}) = {scale:.3f})",
        ha="center", va="top", fontsize=8, transform=ax_b.transAxes,
    )

    # ── save without bbox_inches so pixel size is deterministic ──────────
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    fig.savefig(out_path, dpi=dpi, facecolor="white")
    plt.close(fig)


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
        RGB color per label, e.g. ``{-1: (0.2, 0.2, 0.2), 0: (0.7, 0.7, 0.7), 1: (1, 1, 1)}``.
        Defaults to dark-gray / light-gray / white for the classic -1/0/1 scheme.

    phase_alphas : dict or None
        Opacity per label, e.g. ``{-1: 1.0, 0: 0.5, 1: 0.0}``.
        Defaults to opaque / semi-transparent / invisible for the classic scheme.

    phase_names : dict or None
        Human-readable name per label shown in the legend.
        If ``None`` the label integer itself is used (e.g. ``"2"``).

        Classic stability scheme::

            phase_names={-1: "Unstable", 0: "Meta-stable", 1: "Stable"}

        Phase-number scheme::

            phase_names={1: "1-phase", 2: "2-phase", 3: "3-phase",
                         4: "4-phase", 5: "5-phase"}

    value_label : str
        Name of the quantity shown in the colorbar, e.g. ``'Gm'`` or
        ``'Hmix'``.  Combined with *value_unit* as ``"label (unit)"``.

    value_unit : str
        Physical unit for the colorbar, e.g. ``'J/mol'`` or ``'kJ·mol⁻¹'``.
        Can be supplied without *value_label* if only the unit is needed.

        Example::

            PhaseDiagram5D(
                data_gm,
                value_type='continuous',
                colormap='RdBu_r',
                component_labels=['Fe', 'Mn', 'Ni', 'Co', 'Cu'],
                value_label='Gm',
                value_unit='J/mol',
            )
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
        phase_names: Optional[Dict[int, str]] = None,
        value_label: str = "",
        value_unit: str = "",
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

        # Colorbar label and unit
        self.value_label = value_label
        self.value_unit = value_unit

        # Phase stability colors / alphas / names.
        # User dicts override defaults; defaults only fill labels not specified.
        self._phase_colors = {**DEFAULT_PHASE_COLORS, **(phase_colors or {})}
        self._phase_alphas = {**DEFAULT_PHASE_ALPHAS, **(phase_alphas or {})}
        self._phase_names: Optional[Dict[int, str]] = phase_names

        # Optional stability mask for combined rendering
        if stability_data is not None:
            stab = np.asarray(stability_data, dtype=int).ravel()
            if len(stab) != len(self.data):
                raise ValueError(
                    f"stability_data length ({len(stab)}) must match data "
                    f"length ({len(self.data)})."
                )
            # No label restriction — any integers are valid as long as
            # matching entries exist in phase_colors / phase_alphas.
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
                linewidth=1.8,
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
                fontsize=14,
                ha="center",
                va="center",
                fontweight="bold",
            )

    def _draw_scale_bar(
        self,
        ax,
        x0: float,
    ) -> None:
        """
        Draw a horizontal scale bar showing what fraction of the full
        quaternary composition space is currently displayed.

        The blue filled portion represents ``scale = 1 − x0`` — the total
        remaining composition budget available to the four axes shown.  The
        ``x(component) = value`` text label is rendered separately in the
        top-left corner of the figure and is not repeated here.
        """
        scale = 1.0 - x0
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        bar_y = 0.08
        bar_h = 0.72
        lbl_y = bar_y + 0.03    # just above bottom edge; text grows upward
        tick_bot = bar_y + bar_h * 0.42   # ticks occupy upper ~58 % of bar
        tick_top = bar_y + bar_h

        # Gray background (full composition range 0 → 1)
        ax.barh(bar_y, 1.0, left=0.0, height=bar_h,
                color="#e0e0e0", edgecolor="#888888", linewidth=0.8)
        # Blue filled portion (active scale = 1 − x0)
        if scale > 0:
            ax.barh(bar_y, scale, left=0.0, height=bar_h,
                    color="#4477aa", edgecolor="#888888", linewidth=0.8)

        # Vertical tick lines (short, upper portion) + labels at bottom of bar
        ax.text(0.0, lbl_y, "0", ha="left",  va="bottom", fontsize=6, color="black")
        ax.text(1.0, lbl_y, "1", ha="right", va="bottom", fontsize=6, color="black")
        for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            ax.plot([t, t], [tick_bot, tick_top],
                    color="white", linewidth=0.9, zorder=3, solid_capstyle="butt")
            ax.text(t, lbl_y, f"{t:.1f}",
                    ha="center", va="bottom", fontsize=6, color="black")

        # Header: show scale value so the bar is self-explanatory
        lbl = self.component_labels[0]
        ax.text(
            0.5, 0.97,
            f"composition scale  (1 − x({lbl}) = {scale:.3f})",
            ha="center", va="top", fontsize=8,
            transform=ax.transAxes,
        )

    def _add_colorbar(self, fig, ax_left: float = 0.88) -> None:
        ax_cb = fig.add_axes([ax_left, 0.15, 0.02, 0.65])
        sm = make_scalar_mappable(self.colormap, self.vmin, self.vmax)
        cb = fig.colorbar(sm, cax=ax_cb)
        cb.ax.tick_params(labelsize=7)

        # Build "Label (unit)" string from whatever the user supplied
        if self.value_label or self.value_unit:
            if self.value_label and self.value_unit:
                cb_label = f"{self.value_label} ({self.value_unit})"
            elif self.value_label:
                cb_label = self.value_label
            else:
                cb_label = f"({self.value_unit})"
            cb.set_label(cb_label, fontsize=8, labelpad=6)

    def _add_phase_legend(self, fig, bbox_to_anchor=(0.02, 0.02)) -> None:
        data_labels = set(np.unique(self.data[:, 5].astype(int)))
        handles = []
        for label in sorted(self._phase_colors):
            if label not in self._phase_alphas or label not in data_labels:
                continue
            alpha = self._phase_alphas[label]
            if alpha < 1e-3:
                continue   # fully transparent — skip
            r, g, b = self._phase_colors[label]
            name = (self._phase_names or {}).get(label, str(label))
            patch = mpatches.Patch(
                facecolor=(r, g, b, alpha),
                edgecolor="gray",
                linewidth=0.5,
                label=name,
            )
            handles.append(patch)
        fig.legend(
            handles=handles,
            loc="lower left",
            bbox_to_anchor=bbox_to_anchor,
            fontsize=9,
            frameon=False,
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
        Render data as alpha-shape surfaces, one surface per phase region.

        Replaces the former convex-hull approach with a concave hull (alpha
        shape) that follows the actual irregular phase boundary instead of the
        outermost flat envelope.

        The tightness of the surface is controlled by ``shape_alpha``
        (passed via **kwargs):

        - smaller values → looser surface, approaches convex hull
        - larger values  → tighter surface, more boundary detail

        **Default (adaptive):** when ``shape_alpha`` is not supplied, it is
        chosen automatically per frame as::

            shape_alpha = 2.0 × (N / 62196)^(1/3)

        where *N* is the number of points in the current x₀ slice.  This
        keeps the circumradius threshold proportional to the local grid
        spacing so surface quality remains consistent across all frames —
        tighter for dense slices (small x₀), looser for sparse ones (large
        x₀).  The reference point (N = 62 196, shape_alpha = 2) corresponds
        to a step = 0.01 FeMnNiCoCu grid at x₀ = 0.30.

        **Manual override:** pass ``shape_alpha=<value>`` to fix the
        threshold across all frames.

        For ``value_type='phase_stability'`` a separate surface is drawn for
        each stability class (unstable and meta-stable; stable is invisible).
        For ``value_type='continuous'`` faces are colored by the colormap.

        Falls back silently to nothing on degenerate input (< 4 points).
        Requires ``scipy >= 1.0``.

        *kwargs* are merged on top of library defaults and forwarded to
        ``Poly3DCollection`` after ``shape_alpha`` is consumed
        (e.g. ``edgecolor='white'``, ``linewidth=0.2``).
        """
        try:
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        except ImportError as exc:
            raise ImportError(
                "render='surface' requires scipy.  "
                "Install it with:  pip install scipy"
            ) from exc

        import matplotlib.cm as _mcm
        import matplotlib.colors as _mcolors

        # Adaptive alpha: scale with N^(1/3) to track local grid spacing.
        # n_slice is the pre-downsample count injected by plot_frame; falls
        # back to len(pts) if not present (e.g. direct _render_surface calls).
        n_slice  = int(kwargs.pop("_n_slice", len(pts)))
        _sa_user = kwargs.pop("shape_alpha", None)
        if _sa_user is None:
            shape_alpha = _ALPHA_REF_MPL * (n_slice / _N_REF_MPL) ** (1.0 / 3.0)
        else:
            shape_alpha = float(_sa_user)

        cm   = _mcm.get_cmap(self.colormap)
        norm = _mcolors.Normalize(vmin=self.vmin, vmax=self.vmax)

        def _add_surface(sub_pts, sub_vals, face_alpha, solid_color=None):
            boundary = _alpha_boundary_faces(sub_pts, shape_alpha)
            if len(boundary) == 0:
                return
            triangles = sub_pts[boundary]   # (M, 3, 3) vertex coords
            n_tri = len(triangles)
            if solid_color is not None:
                r, g, b = solid_color
                fcolors = np.tile([r, g, b, face_alpha], (n_tri, 1))
            else:
                mean_val = float(sub_vals.mean())
                fcolors = np.tile([*cm(norm(mean_val))[:3], face_alpha], (n_tri, 1))
            poly_kw = dict(edgecolor="none")
            poly_kw.update(kwargs)
            poly = Poly3DCollection(triangles, **poly_kw)
            poly.set_facecolor(fcolors)
            ax3d.add_collection3d(poly)

        if self.value_type == "phase_stability":
            for label, a in self._phase_alphas.items():
                if a < 1e-3:
                    continue
                mask = values.astype(int) == label
                if mask.sum() < 4:
                    continue
                _add_surface(pts[mask], None, a, solid_color=self._phase_colors[label])

        elif stability is not None:
            # Combined mode: group by stability, colormap hue from values
            for label, a in self._phase_alphas.items():
                if a < 1e-3:
                    continue
                mask = stability == label
                if mask.sum() < 4:
                    continue
                _add_surface(pts[mask], values[mask], a)

        else:
            # Pure continuous: one surface for all visible points
            visible = rgba[:, 3] > 1e-3
            if visible.sum() < 4:
                return
            mean_alpha = float(rgba[visible, 3].mean())
            _add_surface(pts[visible], values[visible], mean_alpha)

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
        wireframe_alpha: float = 0.85,
        wireframe_color: str = "black",
        show_vertex_labels: bool = True,
        elev: float = 20,
        azim: float = -45,
        figsize: Tuple[float, float] = (8, 8),
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
              Uses the matplotlib backend.

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

            These kwargs are also accepted by :meth:`save_frames` and
            :meth:`create_video`, which forward all extra keyword arguments
            through to this method.  For surface rendering, pass
            ``shape_alpha=<float>`` to override the adaptive PyVista alpha.

        Returns
        -------
        fig : matplotlib Figure
        ax  : mpl_toolkits.mplot3d.Axes3D
        """
        if render not in ("scatter",):
            raise ValueError(
                f"render={render!r} is not supported by plot_frame.  "
                "plot_frame only supports render='scatter' (matplotlib).  "
                "For surface rendering use create_video(..., render='surface') "
                "or save_frame_surface()."
            )
        if fig is None:
            fig = plt.figure(figsize=figsize, dpi=dpi)

        # 3-D axes — right strip reserved for colorbar; bottom strip for scale bar
        ax3d: Axes3D = fig.add_axes([0.02, 0.11, 0.84, 0.84], projection="3d")
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
            # Record pre-downsample count for adaptive alpha calculation
            n_slice = len(slice_data)

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
                self._render_surface(
                    ax3d, pts, rgba, values, stab_slice,
                    _n_slice=n_slice, **kwargs
                )
            else:  # 'scatter' (default)
                self._render_scatter(ax3d, pts, rgba, marker_size, **kwargs)

        # Vertex labels
        if show_vertex_labels:
            self._label_vertices(ax3d, mode, x0)

        # Equal-range axis limits so the regular tetrahedron appears undistorted.
        # All three axes must span the same data range; the 3-D box is a cube.
        buf  = 0.04
        mid  = (VERTICES.max(axis=0) + VERTICES.min(axis=0)) * 0.5
        half = (VERTICES.max() - VERTICES.min()) * 0.5 + buf
        ax3d.set_xlim(mid[0] - half, mid[0] + half)
        ax3d.set_ylim(mid[1] - half, mid[1] + half)
        ax3d.set_zlim(mid[2] - half, mid[2] + half)
        ax3d.set_box_aspect([1, 1, 1])

        # x₀ label — top-left text, matching PyVista surface render style
        x0_lbl = self.component_labels[0]
        fig.text(
            0.03, 0.97,
            f"x({x0_lbl}) = {x0:.3f}",
            fontsize=12, va="top", ha="left", color="black",
        )

        # Scale bar — bottom strip below the 3-D axes
        ax_bar = fig.add_axes([0.04, 0.01, 0.80, 0.09])
        self._draw_scale_bar(ax_bar, x0)

        # Legend / colorbar
        # Anchor raised to 0.12 so the legend clears the scale bar strip (0.01-0.10)
        if self.value_type == "continuous":
            self._add_colorbar(fig)
            if self._stability_data is not None:
                self._add_phase_legend(fig, bbox_to_anchor=(0.02, 0.12))
        else:
            self._add_phase_legend(fig, bbox_to_anchor=(0.02, 0.12))

        return fig, ax3d

    def save_frame_surface(
        self,
        x0: float,
        out_path: str,
        mode: str = "fixed",
        shape_alpha: Optional[float] = None,
        window_size: Tuple[int, int] = (1400, 1000),
        camera_position=None,
        show_wireframe: bool = True,
        show_vertex_labels: bool = True,
        max_points: int = 50000,
        min_points: int = 1000,
        markers=None,
        tielines=None,
        tietriangles=None,
        marker_color: str = "red",
        marker_size: int = 18,
        triangle_color: str = "orange",
        triangle_size: int = 14,
    ) -> int:
        """
        Render a single surface frame via PyVista and save it as a PNG.

        This is the surface-rendering equivalent of :meth:`plot_frame`.  It
        renders off-screen using PyVista's VTK backend with smooth shading and
        proper lighting, then writes the result directly to *out_path*.

        Parameters
        ----------
        x0 : float
            Composition value for the fifth component (0 ≤ x0 ≤ 1).
        out_path : str
            Destination PNG file path.
        mode : {'fixed', 'shrink_center', 'shrink_corner'}
            Tetrahedron scaling mode (same as :meth:`plot_frame`).
        shape_alpha : float or None
            Alpha-shape tightness.  When *None* (default) the value is
            computed adaptively as::

                shape_alpha = 90 × (N_eff / 50000)^(1/3)

            where *N_eff* = min(N, max_points) is the actual number of
            points fed into the Delaunay triangulation after subsampling.
            This keeps the circumradius threshold matched to the true point
            density.  Pass an explicit float to override.
        window_size : (width, height)
            Off-screen render resolution in pixels.
        camera_position : list or None
            PyVista camera position triple ``[position, focal_point, up]``.
            Defaults to a front-right elevated view of the tetrahedron.
        show_wireframe : bool
            Draw the tetrahedron edges.
        show_vertex_labels : bool
            Label each vertex with its component name.
        max_points : int
            Maximum points per phase class for the continuous rendering path.
            Has no effect on phase_stability mode (all points are used for
            the alpha-shape Delaunay to preserve surface quality).
        min_points : int
            Minimum number of points required to attempt surface rendering.
            When the x₀ slice contains fewer than *min_points* total points,
            only the tetrahedron wireframe and vertex labels are drawn (no
            phase surfaces).  This prevents artefacts at Fe-rich compositions
            where the composition space becomes very sparse.  Default: 500.

        Returns
        -------
        int
            Number of points in the x₀ slice (useful for progress logging).

        Raises
        ------
        ImportError
            If ``pyvista`` is not installed.
        """
        try:
            import pyvista as pv
        except ImportError as exc:
            raise ImportError(
                "render='surface' requires PyVista.  "
                "Install it with:  pip install pyvista"
            ) from exc

        # ── slice data ────────────────────────────────────────────────────
        x0_mask    = np.abs(self.data[:, 0] - x0) <= self.tolerance
        slice_data = self.data[x0_mask]
        stab_slice = (
            self._stability_data[x0_mask]
            if self._stability_data is not None else None
        )
        n_total = len(slice_data)

        # ── adaptive alpha ────────────────────────────────────────────────
        # Use the full slice count (no subsampling) so alpha tracks the true
        # grid density.  The alpha-shape path uses all points for quality;
        # max_points is only applied in continuous mode where a scatter-style
        # representation is used.
        if shape_alpha is None:
            sa = _ALPHA_REF_PV * (n_total / _N_REF_PV) ** (1.0 / 3.0)
        else:
            sa = float(shape_alpha)

        # ── coordinate mapping ────────────────────────────────────────────
        def _slice_to_coords(sd):
            x1, x2, x3, x4 = (sd[:, k] for k in range(1, 5))
            return compositions_to_cartesian(x1, x2, x3, x4, x0=x0, mode=mode)

        # ── plotter setup ─────────────────────────────────────────────────
        pv.global_theme.background = "white"
        pl = pv.Plotter(off_screen=True, window_size=list(window_size))
        pl.set_background("white")

        # ── surfaces ──────────────────────────────────────────────────────
        # Skip surface rendering if the slice is too sparse; render wireframe only.
        if n_total < min_points:
            print(
                f"  Note: x({self.component_labels[0]}) = {x0:.3f} — only {n_total} points "
                f"in slice (< min_points={min_points}).  Surface rendering skipped; "
                f"showing wireframe only.\n"
                f"  An empty tetrahedron does NOT indicate a fully stable phase region — "
                f"the data density is simply too low for a reliable surface at this composition."
            )
        elif self.value_type == "phase_stability":
            values_sl = slice_data[:, 5].astype(int)
            for label, face_op in self._phase_alphas.items():
                if face_op < 1e-3:
                    continue
                mask = values_sl == label
                sub = slice_data[mask]
                if len(sub) < 4:
                    continue
                pts = _slice_to_coords(sub)
                boundary = _alpha_boundary_faces(pts, sa)
                if len(boundary) == 0:
                    continue
                faces_pv = np.hstack([
                    np.full((len(boundary), 1), 3, dtype=np.int32),
                    boundary.astype(np.int32),
                ])
                mesh = pv.PolyData(pts, faces_pv)
                color = self._phase_colors.get(label, (0.5, 0.5, 0.5))
                pl.add_mesh(
                    mesh, color=color, opacity=face_op,
                    smooth_shading=True, show_edges=False, lighting=True,
                    specular=0.3, specular_power=15, diffuse=0.8, ambient=0.2,
                )

        elif n_total >= min_points:
            # Continuous or combined mode: one surface for all visible points
            if len(slice_data) > max_points:
                rng = np.random.default_rng(42)
                idx = rng.choice(len(slice_data), max_points, replace=False)
                slice_data = slice_data[idx]
                if stab_slice is not None:
                    stab_slice = stab_slice[idx]
            pts    = _slice_to_coords(slice_data)
            values = slice_data[:, 5]
            rgba   = self._map_colors(values, alpha=0.65, stability=stab_slice)
            visible = rgba[:, 3] > 1e-3
            if visible.sum() >= 4:
                boundary = _alpha_boundary_faces(pts[visible], sa)
                if len(boundary) > 0:
                    faces_pv = np.hstack([
                        np.full((len(boundary), 1), 3, dtype=np.int32),
                        boundary.astype(np.int32),
                    ])
                    mesh  = pv.PolyData(pts[visible], faces_pv)
                    color = tuple(float(c) for c in rgba[visible].mean(axis=0)[:3])
                    pl.add_mesh(
                        mesh, color=color, opacity=float(rgba[visible, 3].mean()),
                        smooth_shading=True, show_edges=False, lighting=True,
                        specular=0.3, specular_power=15, diffuse=0.8, ambient=0.2,
                    )

        # ── wireframe ─────────────────────────────────────────────────────
        if show_wireframe:
            verts = tetrahedron_display_vertices(x0, mode)
            for i, j in EDGES:
                pl.add_mesh(pv.Line(verts[i], verts[j]), color="black", line_width=2)

        # ── vertex labels ─────────────────────────────────────────────────
        if show_vertex_labels:
            verts = tetrahedron_display_vertices(x0, mode)
            centroid_3d = verts.mean(axis=0)
            labels = self.component_labels[1:]   # x1…x4
            for i, lbl in enumerate(labels):
                push = _PV_LABEL_PUSH[i] if i < len(_PV_LABEL_PUSH) else 0.10
                pos  = verts[i] + push * (verts[i] - centroid_3d)
                pl.add_point_labels(
                    [pos], [lbl],
                    font_size=18, text_color="black", bold=True,
                    show_points=False, always_visible=True,
                    shape="rect", shape_color="white", shape_opacity=1.0,
                )

        # ── camera ────────────────────────────────────────────────────────
        cam = camera_position or [_PV_CAM_POS, _PV_FOCAL, _PV_CAM_UP]
        pl.camera_position = cam
        pl.camera.zoom(1.4)

        # ── title ─────────────────────────────────────────────────────────
        x0_label = self.component_labels[0]
        pl.add_text(
            f"x({x0_label}) = {x0:.3f}",
            position="upper_left", font_size=14, color="black",
        )

        # ── legend ────────────────────────────────────────────────────────
        # Use the *full* dataset to decide which labels belong in the legend
        # so that rare phases (e.g. 4-phase regions with only ~30 total points)
        # are not omitted when they happen to be absent from the current slice.
        # Entries are rendered via matplotlib in _add_pv_scale_bar so we have
        # full font-size control (PyVista's add_legend has no font_size param).
        legend_entries = []
        if self.value_type == "phase_stability":
            data_labels = set(np.unique(self.data[:, 5].astype(int)))
            for label in sorted(self._phase_colors):
                if label not in data_labels:
                    continue
                # Include all labeled phases regardless of alpha — a transparent
                # phase (e.g. 1-phase) still deserves a legend entry to show
                # the user what the invisible region represents.
                name  = (self._phase_names or {}).get(label, str(label))
                color = self._phase_colors[label]
                legend_entries.append((name, color))
            # legend_entries collected above; rendered via matplotlib below
            # (PyVista's add_legend exposes no font_size control)

        # ── markers & tielines ────────────────────────────────────────────
        # Collect all 3-D points to render as coloured spheres this frame.
        _marker_pts = []

        # Fixed composition markers — shown when x0 matches within tolerance
        if markers is not None:
            for m in markers:
                m = np.asarray(m, dtype=float)
                if abs(m[0] - x0) <= self.tolerance:
                    pt = compositions_to_cartesian(
                        np.array([m[1]]), np.array([m[2]]),
                        np.array([m[3]]), np.array([m[4]]),
                        x0=x0, mode=mode,
                    )
                    _marker_pts.append(pt[0])

        # Tieline markers — interpolate between the two equilibrium end-points
        # and show the intersection with the current x0 plane as a sphere.
        if tielines is not None:
            for ends in tielines:
                a = np.asarray(ends[0], dtype=float)   # [x0,x1,x2,x3,x4]
                b = np.asarray(ends[1], dtype=float)
                x0_a, x0_b = a[0], b[0]
                lo, hi = min(x0_a, x0_b), max(x0_a, x0_b)
                if lo - self.tolerance <= x0 <= hi + self.tolerance:
                    denom = x0_b - x0_a
                    t = (x0 - x0_a) / denom if abs(denom) > 1e-9 else 0.5
                    t = float(np.clip(t, 0.0, 1.0))
                    interp = a + t * (b - a)
                    pt = compositions_to_cartesian(
                        np.array([interp[1]]), np.array([interp[2]]),
                        np.array([interp[3]]), np.array([interp[4]]),
                        x0=x0, mode=mode,
                    )
                    _marker_pts.append(pt[0])

        if _marker_pts:
            cloud = pv.PolyData(np.array(_marker_pts))
            pl.add_mesh(
                cloud,
                color=marker_color,
                point_size=marker_size,
                render_points_as_spheres=True,
            )

        # Tie-simplex cross-sections — works for any number of equilibrium phases
        # (tie-line=2, tie-triangle=3, tie-tetrahedron=4, …).
        # All edges (pairs of vertices) are intersected with the current x0
        # plane.  The resulting cross-section points are sorted into convex
        # order and rendered as a closed polygon + spheres in triangle_color.
        if tietriangles is not None:
            from itertools import combinations as _comb
            for tri in tietriangles:
                verts = [np.asarray(v, dtype=float) for v in tri]

                # ── intersect every edge with the x0 plane ────────────────
                raw_pts = []
                for a, b in _comb(verts, 2):
                    x0_a, x0_b = a[0], b[0]
                    lo, hi = min(x0_a, x0_b), max(x0_a, x0_b)
                    if lo - self.tolerance <= x0 <= hi + self.tolerance:
                        denom = x0_b - x0_a
                        t = (x0 - x0_a) / denom if abs(denom) > 1e-9 else 0.5
                        t = float(np.clip(t, 0.0, 1.0))
                        interp = a + t * (b - a)
                        pt = compositions_to_cartesian(
                            np.array([interp[1]]), np.array([interp[2]]),
                            np.array([interp[3]]), np.array([interp[4]]),
                            x0=x0, mode=mode,
                        )
                        raw_pts.append(pt[0])

                # ── de-duplicate (vertex exactly on the plane) ────────────
                unique_pts = []
                for p in raw_pts:
                    if not any(np.linalg.norm(p - q) < 1e-6 for q in unique_pts):
                        unique_pts.append(p)

                if not unique_pts:
                    continue

                # ── sort into convex order for 3+ points ──────────────────
                # Project onto the local 2-D plane (spanned by the first two
                # orthogonal directions of the point cloud) and sort by angle
                # around the centroid so the polygon outline is correct.
                if len(unique_pts) >= 3:
                    pts_arr = np.array(unique_pts)
                    centroid = pts_arr.mean(axis=0)
                    # Two orthogonal basis vectors in the cutting plane
                    u = pts_arr[0] - centroid
                    u_norm = np.linalg.norm(u)
                    if u_norm > 1e-9:
                        u /= u_norm
                    # v = component of (pts[1]-centroid) perpendicular to u
                    v = pts_arr[1] - centroid
                    v -= np.dot(v, u) * u
                    v_norm = np.linalg.norm(v)
                    if v_norm > 1e-9:
                        v /= v_norm
                    angles = [
                        np.arctan2(np.dot(p - centroid, v),
                                   np.dot(p - centroid, u))
                        for p in pts_arr
                    ]
                    order = np.argsort(angles)
                    unique_pts = [unique_pts[i] for i in order]

                # ── draw cross-section edges ──────────────────────────────
                n = len(unique_pts)
                if n == 2:
                    # Simple line segment (tie-line or edge of a simplex)
                    pl.add_mesh(
                        pv.Line(unique_pts[0], unique_pts[1]),
                        color=triangle_color, line_width=4,
                    )
                elif n >= 3:
                    # Closed polygon (tie-triangle cross-section, quadrilateral, …)
                    for i in range(n):
                        pl.add_mesh(
                            pv.Line(unique_pts[i], unique_pts[(i + 1) % n]),
                            color=triangle_color, line_width=4,
                        )

                # ── render intersection points as spheres ─────────────────
                cloud_tri = pv.PolyData(np.array(unique_pts))
                pl.add_mesh(
                    cloud_tri,
                    color=triangle_color,
                    point_size=triangle_size,
                    render_points_as_spheres=True,
                )

        img_arr = pl.screenshot(None, transparent_background=False, return_img=True)
        pl.close()

        # Composite PyVista render with matplotlib scale bar + legend
        _mpl_legend = legend_entries if self.value_type == "phase_stability" else None
        _add_pv_scale_bar(img_arr, out_path, x0, self.component_labels[0],
                          legend_entries=_mpl_legend)
        return n_total

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

        render = plot_kwargs.get("render", "scatter")

        if render not in ("scatter", "surface"):
            raise ValueError(
                f"render={render!r} is not supported.  "
                "Use render='scatter' (matplotlib) or render='surface' (PyVista)."
            )

        w = len(str(len(x0_values)))

        for i, x0 in enumerate(x0_values):
            path = os.path.join(output_dir, f"frame_{i:04d}.png")

            if render == "surface":
                # PyVista path — extract relevant kwargs, write PNG directly
                pv_keys = ("mode", "shape_alpha", "window_size",
                           "camera_position", "show_wireframe",
                           "show_vertex_labels", "max_points", "min_points",
                           "markers", "tielines", "tietriangles",
                           "marker_color", "marker_size",
                           "triangle_color", "triangle_size")
                pv_kw = {k: plot_kwargs[k] for k in pv_keys if k in plot_kwargs}
                n_slice = self.save_frame_surface(x0, path, **pv_kw)
            else:
                # matplotlib path (scatter)
                fig, _ = self.plot_frame(x0, dpi=dpi, **plot_kwargs)
                fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
                plt.close(fig)
                n_slice = len(extract_x0_slice(self.data, x0, self.tolerance))

            paths.append(os.path.abspath(path))

            if verbose:
                print(
                    f"  [{i + 1:>{w}}/{len(x0_values)}] "
                    f"x0 = {x0:.3f}  ({n_slice} points)  -> {path}"
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
            When *True* and *frames_dir* is not given, frames are saved in a
            ``<videoname>_frames/`` sub-directory next to *output_path*.
        frames_dir : str or None
            Where to store frames.  Defaults to a temporary directory (deleted
            after assembly) unless *keep_frames* is True, in which case it
            defaults to ``<videoname>_frames/`` next to *output_path*.
        verbose : bool
            Print progress.
        **plot_kwargs
            Forwarded to the renderer.  Common options:

            - ``render='scatter'`` *(default)* — matplotlib scatter plot;
              kwargs forwarded to :meth:`plot_frame`.
            - ``render='surface'`` — PyVista surface plot (recommended for
              publication quality); kwargs forwarded to
              :meth:`save_frame_surface`.  Pass ``shape_alpha=<float>`` to
              override the default adaptive alpha.

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
            if keep_frames:
                # Place frames next to the video: <videoname>_frames/
                base = os.path.splitext(os.path.abspath(output_path))[0]
                frames_dir = base + "_frames"
            else:
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
        buf  = 0.12
        mid  = (VERTICES.max(axis=0) + VERTICES.min(axis=0)) * 0.5
        half = (VERTICES.max() - VERTICES.min()) * 0.5 + buf
        ax3d.set_xlim(mid[0] - half, mid[0] + half)
        ax3d.set_ylim(mid[1] - half, mid[1] + half)
        ax3d.set_zlim(mid[2] - half, mid[2] + half)
        ax3d.set_box_aspect([1, 1, 1])

        if show_colorbar and self.value_type == "continuous":
            self._add_colorbar(fig, ax_left=0.88)

        if self.value_type == "phase_stability":
            self._add_phase_legend(fig)

        return fig, ax3d

    def show_interactive(
        self,
        x0_init: float = 0.0,
        mode: str = "fixed",
        render: str = "scatter",
        alpha: float = 0.65,
        marker_size: float = 3,
        max_points: int = 5000,
        show_wireframe: bool = True,
        wireframe_alpha: float = 0.85,
        wireframe_color: str = "black",
        show_vertex_labels: bool = True,
        elev: float = 20,
        azim: float = 45,
        figsize: Tuple[float, float] = (9, 10),
        **kwargs,
    ) -> None:
        """
        Open an interactive matplotlib window with sliders for x₀, elevation,
        and azimuth.

        Drag the **x₀** slider to sweep through the fifth composition axis.
        Drag **Elevation** and **Azimuth** to rotate the tetrahedron freely.
        All other rendering options (``mode``, ``render``, ``alpha``, …) are
        fixed at call time but can be changed by calling the method again.

        .. note::
            This method calls ``plt.show()`` (blocking).  It requires an
            interactive matplotlib backend — do **not** call
            ``matplotlib.use('Agg')`` before using it.  From a script, run
            with a GUI backend such as ``TkAgg`` or ``Qt5Agg``::

                import matplotlib
                matplotlib.use('TkAgg')   # before importing phase5d

        Parameters
        ----------
        x0_init : float
            Starting x₀ value (default 0).
        mode : {'fixed', 'shrink_center', 'shrink_corner'}
            Tetrahedron scaling mode.
        render : {'scatter'}
            Rendering style.  Only ``'scatter'`` (matplotlib) is supported in
            the interactive viewer; use :meth:`create_video` for surface
            rendering.
        alpha : float
            Point / face transparency for continuous values.
        marker_size : float
            Scatter marker size (``render='scatter'`` only).
        max_points : int
            Maximum data points rendered per frame.  Keep low (≤ 5 000) for
            a smooth interactive response.
        show_wireframe : bool
            Draw tetrahedron edges.
        wireframe_alpha : float
            Wireframe transparency.
        wireframe_color : str
            Wireframe color.
        show_vertex_labels : bool
            Label vertices with component names.
        elev : float
            Initial elevation angle (degrees).
        azim : float
            Initial azimuth angle (degrees).
        figsize : (width, height)
            Figure size in inches.
        **kwargs
            Forwarded to the underlying render call (same as
            :meth:`plot_frame`).
        """
        from matplotlib.widgets import Slider

        # ── Detect x0 range from data ────────────────────────────────────
        x0_all = self.data[:, 0]
        x0_min = float(x0_all.min())
        x0_max = float(x0_all.max())
        # Infer grid step from the data so the slider snaps to real values
        unique_x0 = np.unique(np.round(x0_all, 4))
        if len(unique_x0) > 1:
            x0_step = float(np.round(np.median(np.diff(unique_x0)), 4))
        else:
            x0_step = 0.01
        x0_init = float(np.clip(x0_init, x0_min, x0_max))

        lbl0 = self.component_labels[0]

        # ── Figure layout ─────────────────────────────────────────────────
        fig = plt.figure(figsize=figsize)

        ax3d: Axes3D = fig.add_axes(
            [0.02, 0.27, 0.80, 0.70], projection="3d"
        )
        ax_bar = fig.add_axes([0.08, 0.21, 0.72, 0.04])

        # Sliders
        ax_sl_x0   = fig.add_axes([0.15, 0.155, 0.65, 0.025])
        ax_sl_elev = fig.add_axes([0.15, 0.100, 0.65, 0.025])
        ax_sl_azim = fig.add_axes([0.15, 0.045, 0.65, 0.025])

        sl_x0   = Slider(ax_sl_x0,   f"x({lbl0})", x0_min, x0_max,
                         valinit=x0_init, valstep=x0_step, color="#4477aa")
        sl_elev = Slider(ax_sl_elev, "Elevation",  -90,   90,
                         valinit=elev,    valstep=1,      color="#888888")
        sl_azim = Slider(ax_sl_azim, "Azimuth",      0,  360,
                         valinit=azim,    valstep=1,      color="#888888")

        # Static decorations (colorbar / legend) — created once
        if self.value_type == "continuous":
            self._add_colorbar(fig, ax_left=0.85)
            if self._stability_data is not None:
                self._add_phase_legend(fig)
        else:
            self._add_phase_legend(fig)

        # ── Update callback ───────────────────────────────────────────────
        def _redraw(_event=None):
            x0  = float(sl_x0.val)
            e   = float(sl_elev.val)
            a   = float(sl_azim.val)

            ax3d.cla()
            ax3d.set_axis_off()

            if show_wireframe:
                self._draw_wireframe(ax3d, x0, mode,
                                     wireframe_alpha, wireframe_color)

            # Slice data
            mask = np.abs(self.data[:, 0] - x0) <= self.tolerance
            sd = self.data[mask]
            stab = (self._stability_data[mask]
                    if self._stability_data is not None else None)

            if len(sd) > 0:
                if len(sd) > max_points:
                    rng = np.random.default_rng(42)
                    idx = rng.choice(len(sd), size=max_points, replace=False)
                    sd   = sd[idx]
                    if stab is not None:
                        stab = stab[idx]

                x1, x2, x3, x4 = (sd[:, k] for k in range(1, 5))
                vals = sd[:, 5]
                pts  = compositions_to_cartesian(x1, x2, x3, x4,
                                                 x0=x0, mode=mode)
                rgba = self._map_colors(vals, alpha=alpha, stability=stab)

                self._render_scatter(ax3d, pts, rgba, marker_size, **kwargs)

            if show_vertex_labels:
                self._label_vertices(ax3d, mode, x0)

            buf  = 0.04
            mid  = (VERTICES.max(axis=0) + VERTICES.min(axis=0)) * 0.5
            half = (VERTICES.max() - VERTICES.min()) * 0.5 + buf
            ax3d.set_xlim(mid[0] - half, mid[0] + half)
            ax3d.set_ylim(mid[1] - half, mid[1] + half)
            ax3d.set_zlim(mid[2] - half, mid[2] + half)
            ax3d.set_box_aspect([1, 1, 1])
            ax3d.view_init(elev=e, azim=a)

            # Refresh scale bar
            ax_bar.cla()
            self._draw_scale_bar(ax_bar, x0)

            fig.canvas.draw_idle()

        sl_x0.on_changed(_redraw)
        sl_elev.on_changed(_redraw)
        sl_azim.on_changed(_redraw)

        _redraw()   # initial draw
        plt.show()

    def save_vtk(
        self,
        output_path: str,
        max_points: Optional[int] = None,
        include_compositions: bool = True,
    ) -> str:
        """
        Export the dataset as a VTK XML Unstructured Grid (``.vtu``) file for
        use in ParaView or any VTK-compatible viewer.

        Every data point is placed at its natural barycentric position
        ``P = x₁·V₀ + x₂·V₁ + x₃·V₂ + x₄·V₃`` — the same coordinate system
        used by :meth:`plot_isosurface`.  Scalar fields stored per point:

        ============== ==============================================
        Field          Content
        ============== ==============================================
        ``value``      The value column (or ``value_label`` if set)
        ``x0`` … ``x4`` Individual composition fractions
        ``stability``  Stability labels (−1 / 0 / 1), if available
        ============== ==============================================

        **Suggested ParaView workflow**

        1. *File → Open* the ``.vtu`` file.
        2. Apply a **Threshold** filter on ``x0`` to slice by Fe content.
        3. Apply a **Contour** filter on ``value`` to extract isosurfaces.
        4. Color by any scalar field; use *Point Gaussian* for scatter-style
           rendering.

        The file is written in VTK XML binary format with base64 encoding
        (no extra dependencies required).  For large datasets (> 1 M points)
        this produces files of roughly 60–80 MB for 4.6 M points; use
        *max_points* to downsample if storage is a concern.

        Parameters
        ----------
        output_path : str
            Destination file path (should end in ``.vtu``).
        max_points : int or None
            If set, randomly subsample the dataset to at most this many points
            before writing.  The same random seed (42) is always used so the
            output is reproducible.
        include_compositions : bool
            Write individual ``x0`` … ``x4`` scalar fields (default ``True``).
            Set to ``False`` to reduce file size.

        Returns
        -------
        str
            Absolute path of the written file.

        Examples
        --------
        ::

            diag.save_vtk('phase_diagram.vtu')
            diag.save_vtk('phase_diagram_small.vtu', max_points=200_000)
        """

        # ── Prepare data ──────────────────────────────────────────────────
        data = self.data.copy()   # (N, 6)
        stab = (self._stability_data.copy()
                if self._stability_data is not None else None)

        if max_points is not None and len(data) > max_points:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(data), size=max_points, replace=False)
            data = data[idx]
            if stab is not None:
                stab = stab[idx]

        N = len(data)
        x0c = data[:, 0]
        x1c = data[:, 1]
        x2c = data[:, 2]
        x3c = data[:, 3]
        x4c = data[:, 4]
        vals = data[:, 5]

        # ── Natural embedding ─────────────────────────────────────────────
        pts = (
            x1c[:, None] * VERTICES[0]
            + x2c[:, None] * VERTICES[1]
            + x3c[:, None] * VERTICES[2]
            + x4c[:, None] * VERTICES[3]
        )  # (N, 3)
        xyz = np.ascontiguousarray(pts, dtype=np.float32)

        # ── VTK base64 binary helper ──────────────────────────────────────
        def _b64(arr: np.ndarray) -> str:
            """Encode array as VTK inline binary (UInt32 byte-count header)."""
            raw    = np.ascontiguousarray(arr).tobytes()
            header = np.array(len(raw), dtype=np.uint32).tobytes()
            return base64.b64encode(header + raw).decode("ascii")

        _vtk_dtype = {
            np.dtype("float32"): "Float32",
            np.dtype("float64"): "Float64",
            np.dtype("int8"):    "Int8",
            np.dtype("int16"):   "Int16",
            np.dtype("int32"):   "Int32",
            np.dtype("int64"):   "Int64",
            np.dtype("uint8"):   "UInt8",
        }

        def _dtype_str(arr: np.ndarray) -> str:
            return _vtk_dtype.get(arr.dtype, "Float32")

        # ── Cell arrays (one VTK_VERTEX per point) ────────────────────────
        connectivity = np.arange(N, dtype=np.int32)
        offsets      = np.arange(1, N + 1, dtype=np.int32)
        cell_types   = np.ones(N, dtype=np.uint8)   # VTK_VERTEX = 1

        # ── Scalar fields ─────────────────────────────────────────────────
        value_name = self.value_label if self.value_label else "value"
        scalars: Dict[str, np.ndarray] = {
            value_name: vals.astype(np.float32),
        }
        if include_compositions:
            scalars["x0"] = x0c.astype(np.float32)
            scalars["x1"] = x1c.astype(np.float32)
            scalars["x2"] = x2c.astype(np.float32)
            scalars["x3"] = x3c.astype(np.float32)
            scalars["x4"] = x4c.astype(np.float32)
        if stab is not None:
            scalars["stability"] = stab.astype(np.int8)

        # ── Build VTK XML ─────────────────────────────────────────────────
        lines = [
            '<?xml version="1.0"?>',
            '<VTKFile type="UnstructuredGrid" version="0.1"'
            ' byte_order="LittleEndian" header_type="UInt32">',
            "  <UnstructuredGrid>",
            f'    <Piece NumberOfPoints="{N}" NumberOfCells="{N}">',
            "      <Points>",
            '        <DataArray type="Float32" NumberOfComponents="3"'
            ' format="binary">',
            f"          {_b64(xyz)}",
            "        </DataArray>",
            "      </Points>",
            "      <Cells>",
            '        <DataArray type="Int32" Name="connectivity"'
            ' format="binary">',
            f"          {_b64(connectivity)}",
            "        </DataArray>",
            '        <DataArray type="Int32" Name="offsets"'
            ' format="binary">',
            f"          {_b64(offsets)}",
            "        </DataArray>",
            '        <DataArray type="UInt8" Name="types"'
            ' format="binary">',
            f"          {_b64(cell_types)}",
            "        </DataArray>",
            "      </Cells>",
            f'      <PointData Scalars="{value_name}">',
        ]
        for name, arr in scalars.items():
            lines += [
                f'        <DataArray type="{_dtype_str(arr)}"'
                f' Name="{name}" format="binary">',
                f"          {_b64(arr)}",
                "        </DataArray>",
            ]
        lines += [
            "      </PointData>",
            "    </Piece>",
            "  </UnstructuredGrid>",
            "</VTKFile>",
        ]

        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

        abs_path = os.path.abspath(output_path)
        print(f"VTK export: {N:,} points -> {abs_path}")
        return abs_path
