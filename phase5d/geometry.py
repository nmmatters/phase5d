"""
Tetrahedron geometry for 5D phase diagram visualization.

The four vertices of the regular tetrahedron represent the four independent
composition axes (x1, x2, x3, x4). Any composition satisfying
x1 + x2 + x3 + x4 = 1 - x0 maps to a point inside or on this tetrahedron.
"""

import numpy as np

# Regular tetrahedron with unit edge length, vertices at:
#   v0 → pure x1,  v1 → pure x2,  v2 → pure x3,  v3 → pure x4
VERTICES = np.array([
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.5, np.sqrt(3.0) / 2.0, 0.0],
    [0.5, 1.0 / (2.0 * np.sqrt(3.0)), np.sqrt(2.0 / 3.0)],
])

CENTROID = VERTICES.mean(axis=0)

EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]

# Each face: triple of vertex indices, the missing vertex is the zero-component
# Face i corresponds to xi+1 = 0
FACES = [
    (1, 2, 3),  # x1 = 0
    (0, 2, 3),  # x2 = 0
    (0, 1, 3),  # x3 = 0
    (0, 1, 2),  # x4 = 0
]


def compositions_to_cartesian(
    x1: np.ndarray,
    x2: np.ndarray,
    x3: np.ndarray,
    x4: np.ndarray,
    x0: float,
    mode: str = "fixed",
) -> np.ndarray:
    """
    Convert composition arrays to 3D Cartesian coordinates.

    Parameters
    ----------
    x1, x2, x3, x4 : array-like
        Composition arrays for each component.
    x0 : float
        The fixed x0 slice value (used for normalization in 'fixed' mode).
    mode : {'fixed', 'shrink_center', 'shrink_corner'}
        Visualization mode:
        - 'fixed'         : normalize compositions so tetrahedron always fills
                            the full viewport; scale shown via bar only.
        - 'shrink_center' : tetrahedron scaled by (1-x0) around its centroid.
        - 'shrink_corner' : tetrahedron scaled by (1-x0) toward the origin
                            (pure-x0 corner).

    Returns
    -------
    np.ndarray, shape (N, 3)
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)
    x3 = np.asarray(x3, dtype=float)
    x4 = np.asarray(x4, dtype=float)

    scale = 1.0 - x0

    if mode == "fixed":
        if scale > 1e-12:
            b1, b2, b3, b4 = x1 / scale, x2 / scale, x3 / scale, x4 / scale
        else:
            # x0 ≈ 1: all compositions → centroid
            b1 = b2 = b3 = b4 = np.full_like(x1, 0.25)
        return (
            b1[:, None] * VERTICES[0]
            + b2[:, None] * VERTICES[1]
            + b3[:, None] * VERTICES[2]
            + b4[:, None] * VERTICES[3]
        )

    if mode == "shrink_corner":
        # Natural embedding: P = x1*v0 + ... → scale (1-x0) toward origin
        return (
            x1[:, None] * VERTICES[0]
            + x2[:, None] * VERTICES[1]
            + x3[:, None] * VERTICES[2]
            + x4[:, None] * VERTICES[3]
        )

    if mode == "shrink_center":
        if scale > 1e-12:
            b1, b2, b3, b4 = x1 / scale, x2 / scale, x3 / scale, x4 / scale
        else:
            b1 = b2 = b3 = b4 = np.full_like(x1, 0.25)
        p_norm = (
            b1[:, None] * VERTICES[0]
            + b2[:, None] * VERTICES[1]
            + b3[:, None] * VERTICES[2]
            + b4[:, None] * VERTICES[3]
        )
        return CENTROID + scale * (p_norm - CENTROID)

    raise ValueError(f"Unknown mode {mode!r}. Choose 'fixed', 'shrink_center', or 'shrink_corner'.")


def tetrahedron_display_vertices(x0: float, mode: str = "fixed") -> np.ndarray:
    """
    Return the 4 tetrahedron vertices as they should be drawn for a given x0 and mode.

    Parameters
    ----------
    x0 : float
    mode : {'fixed', 'shrink_center', 'shrink_corner'}

    Returns
    -------
    np.ndarray, shape (4, 3)
    """
    scale = 1.0 - x0

    if mode == "fixed":
        return VERTICES.copy()
    if mode == "shrink_corner":
        return scale * VERTICES
    if mode == "shrink_center":
        return CENTROID + scale * (VERTICES - CENTROID)

    raise ValueError(f"Unknown mode {mode!r}.")
