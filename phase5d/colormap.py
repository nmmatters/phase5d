"""
Color mapping for 5D phase diagram visualization.

Two modes are supported:
  - 'continuous' : scalar property (Gibbs energy, enthalpy, …) → colormap RGBA
  - 'phase_stability' : integer labels (-1 / 0 / 1) → fixed gray RGBA
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
from typing import Dict, Optional, Tuple

# Default colors and alphas for phase stability rendering
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
    Map phase stability labels (-1 / 0 / 1) to RGBA colors.

    Parameters
    ----------
    values : array-like, shape (N,)
        Integer labels: -1 (unstable), 0 (meta-stable), 1 (stable).
    phase_colors : dict mapping label → (R, G, B) in [0, 1]
    phase_alphas : dict mapping label → alpha in [0, 1]

    Returns
    -------
    colors : np.ndarray, shape (N, 4)
    """
    values = np.asarray(values, dtype=int)
    colors_map = {**DEFAULT_PHASE_COLORS, **(phase_colors or {})}
    alphas_map = {**DEFAULT_PHASE_ALPHAS, **(phase_alphas or {})}

    rgba = np.zeros((len(values), 4), dtype=float)
    for label in (-1, 0, 1):
        mask = values == label
        if mask.any():
            r, g, b = colors_map[label]
            a = alphas_map[label]
            rgba[mask] = [r, g, b, a]
    return rgba


def make_scalar_mappable(
    cmap: str,
    vmin: float,
    vmax: float,
) -> mcm.ScalarMappable:
    """Return a ScalarMappable suitable for plt.colorbar."""
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    return mcm.ScalarMappable(norm=norm, cmap=plt.get_cmap(cmap))
