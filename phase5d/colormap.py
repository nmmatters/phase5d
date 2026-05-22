"""
Color mapping for 5D phase diagram visualization.

Two modes are supported:
  - 'continuous' : scalar property (Gibbs energy, enthalpy, …) → colormap RGBA
  - 'phase_stability' : integer labels → fixed RGBA per label

The default color/alpha scheme uses labels -1 / 0 / 1 (unstable / meta-stable
/ stable), but any integer labels are accepted as long as matching entries are
provided in the phase_colors and phase_alphas dicts passed at construction time.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
from typing import Dict, Optional, Tuple

# Default colors and alphas for the classic phase-stability scheme (-1 / 0 / 1)
DEFAULT_PHASE_COLORS: Dict[int, Tuple[float, float, float]] = {
    -1: (0.20, 0.20, 0.20),  # dark gray  – unstable
     0: (0.70, 0.70, 0.70),  # light gray – meta-stable
     1: (1.00, 1.00, 1.00),  # white      – stable
}

DEFAULT_PHASE_ALPHAS: Dict[int, float] = {
    -1: 1.0,   # fully opaque
     0: 0.50,  # semi-transparent
     1: 0.0,   # invisible (stable regions hidden by default)
}


def continuous_colors(
    values: np.ndarray,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    alpha: float = 0.7,
) -> Tuple[np.ndarray, mcolors.Normalize, object]:
    """
    Map scalar values to RGBA colors using a matplotlib colormap.

    Parameters
    ----------
    values : array-like, shape (N,)
    cmap : matplotlib colormap name
    vmin, vmax : color scale limits (defaults to data min/max)
    alpha : uniform alpha applied to all colors

    Returns
    -------
    colors : np.ndarray, shape (N, 4)  RGBA in [0, 1]
    norm   : matplotlib Normalize instance
    cm     : matplotlib colormap
    """
    values = np.asarray(values, dtype=float)
    v_min = float(values.min()) if vmin is None else vmin
    v_max = float(values.max()) if vmax is None else vmax

    norm = mcolors.Normalize(vmin=v_min, vmax=v_max)
    cm = plt.get_cmap(cmap)
    colors = cm(norm(values))          # shape (N, 4)
    colors = colors.copy()
    colors[:, 3] = alpha
    return colors, norm, cm


def phase_stability_colors(
    values: np.ndarray,
    phase_colors: Optional[Dict[int, Tuple[float, float, float]]] = None,
    phase_alphas: Optional[Dict[int, float]] = None,
) -> np.ndarray:
    """
    Map integer phase labels to RGBA colors.

    Works with any integer label scheme.  The classic stability scheme uses
    -1 / 0 / 1 (unstable / meta-stable / stable); phase-number data uses
    1 / 2 / 3 / … — pass matching phase_colors / phase_alphas dicts for
    any label set.

    Parameters
    ----------
    values : array-like, shape (N,)
        Integer labels.
    phase_colors : dict mapping label → (R, G, B) in [0, 1]
        Overrides DEFAULT_PHASE_COLORS for listed labels.
    phase_alphas : dict mapping label → alpha in [0, 1]
        Overrides DEFAULT_PHASE_ALPHAS for listed labels.

    Returns
    -------
    colors : np.ndarray, shape (N, 4)
    """
    values = np.asarray(values, dtype=int)
    colors_map = {**DEFAULT_PHASE_COLORS, **(phase_colors or {})}
    alphas_map = {**DEFAULT_PHASE_ALPHAS, **(phase_alphas or {})}

    rgba = np.zeros((len(values), 4), dtype=float)
    for label in np.unique(values):
        mask = values == label
        if label in colors_map:
            r, g, b = colors_map[label]
            a = alphas_map.get(label, 0.5)
            rgba[mask] = [r, g, b, a]
    return rgba


def combined_colors(
    continuous_values: np.ndarray,
    phase_labels: np.ndarray,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    phase_alphas: Optional[Dict[int, float]] = None,
) -> Tuple[np.ndarray, mcolors.Normalize, object]:
    """
    Map scalar values to colors while using phase labels to set alpha.

    This is the "combined" rendering mode: the *hue* encodes a continuous
    property (e.g. Gibbs energy) and the *opacity* encodes phase stability —
    stable regions stay invisible, meta-stable regions are semi-transparent,
    and unstable regions are fully opaque.  This lets you see both *where*
    the instability boundary sits and *how deep* the energy well is.

    Parameters
    ----------
    continuous_values : array-like, shape (N,)
        Scalar property (Gm, Hmr, …) used to pick colors from the colormap.
    phase_labels : array-like, shape (N,)
        Integer labels used to set per-point alpha.
    cmap : matplotlib colormap name
    vmin, vmax : color-scale limits (defaults to data min/max).
    phase_alphas : dict or None
        Alpha per label.  Defaults to DEFAULT_PHASE_ALPHAS.

    Returns
    -------
    colors : np.ndarray, shape (N, 4)  RGBA in [0, 1]
    norm   : matplotlib Normalize instance
    cm     : matplotlib colormap
    """
    continuous_values = np.asarray(continuous_values, dtype=float)
    phase_labels = np.asarray(phase_labels, dtype=int)

    # Get hue from continuous colormap (alpha ignored here — set below)
    rgba, norm, cm = continuous_colors(continuous_values, cmap, vmin, vmax, alpha=1.0)

    # Override alpha from phase labels
    alphas_map = {**DEFAULT_PHASE_ALPHAS, **(phase_alphas or {})}
    for label in np.unique(phase_labels):
        mask = phase_labels == label
        rgba[mask, 3] = alphas_map.get(label, 0.5)

    return rgba, norm, cm


def make_scalar_mappable(
    cmap: str,
    vmin: float,
    vmax: float,
) -> mcm.ScalarMappable:
    """Return a ScalarMappable suitable for plt.colorbar."""
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    return mcm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))
