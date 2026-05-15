"""
Data utilities: validation, slicing, downsampling.
"""

import numpy as np
from typing import Optional


def validate_data(data: np.ndarray) -> np.ndarray:
    """
    Validate and return the data array.

    Expected shape: (N, 5) with columns [x1, x2, x3, x4, value].
    All compositions must be non-negative and x1+x2+x3+x4 <= 1.

    Raises
    ------
    ValueError on invalid input.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[1] != 5:
        raise ValueError(
            "Data must have shape (N, 5): columns are [x1, x2, x3, x4, value]."
        )

    comps = data[:, :4]
    if np.any(comps < -1e-9):
        raise ValueError("All compositions (x1…x4) must be non-negative.")

    row_sums = comps.sum(axis=1)
    if np.any(row_sums > 1.0 + 1e-9):
        raise ValueError(
            "x1 + x2 + x3 + x4 must be <= 1 for all rows "
            "(x0 = 1 - sum must be >= 0)."
        )
    return data


def compute_x0(data: np.ndarray) -> np.ndarray:
    """Return x0 = 1 - x1 - x2 - x3 - x4 for every row."""
    return 1.0 - data[:, 0] - data[:, 1] - data[:, 2] - data[:, 3]


def extract_x0_slice(
    data: np.ndarray,
    x0: float,
    tolerance: float = 0.005,
) -> np.ndarray:
    """
    Extract rows whose x0 value is within *tolerance* of the target.

    Parameters
    ----------
    data : np.ndarray, shape (N, 5)
    x0 : float
        Target x0 value.
    tolerance : float
        Half-width of the acceptance window around x0.

    Returns
    -------
    np.ndarray, shape (M, 5)
    """
    x0_values = compute_x0(data)
    mask = np.abs(x0_values - x0) <= tolerance
    return data[mask]


def downsample(
    data: np.ndarray,
    max_points: int,
    random_state: int = 42,
) -> np.ndarray:
    """
    Randomly subsample *data* to at most *max_points* rows.

    Returns *data* unchanged if it already has <= max_points rows.
    """
    if len(data) <= max_points:
        return data
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(data), size=max_points, replace=False)
    return data[idx]


def x0_grid(
    data: np.ndarray,
    step: float = 0.01,
) -> np.ndarray:
    """
    Return evenly spaced x0 values covering the range present in *data*.

    Parameters
    ----------
    data : np.ndarray, shape (N, 5)
    step : float
        Spacing between x0 values.

    Returns
    -------
    np.ndarray of x0 values rounded to avoid floating-point drift.
    """
    x0_all = compute_x0(data)
    x0_min = np.round(x0_all.min(), decimals=10)
    x0_max = np.round(x0_all.max(), decimals=10)
    n = int(round((x0_max - x0_min) / step)) + 1
    return np.round(np.linspace(x0_min, x0_max, n), decimals=10)


def generate_grid_data(
    step: float = 0.05,
    value_fn=None,
    seed: int = 0,
) -> np.ndarray:
    """
    Generate a synthetic regular-grid dataset for testing.

    Creates all (x1, x2, x3, x4) combinations on a *step* grid with
    x1+x2+x3+x4 <= 1, then optionally applies *value_fn(x0, x1, x2, x3, x4)*
    to compute the value column (defaults to Gaussian noise).

    Returns
    -------
    np.ndarray, shape (N, 5)  — columns [x1, x2, x3, x4, value]
    """
    rng = np.random.default_rng(seed)
    ticks = np.arange(0.0, 1.0 + step / 2, step)
    rows = []
    for x1 in ticks:
        for x2 in ticks:
            if x1 + x2 > 1.0 + 1e-9:
                break
            for x3 in ticks:
                if x1 + x2 + x3 > 1.0 + 1e-9:
                    break
                for x4 in ticks:
                    if x1 + x2 + x3 + x4 > 1.0 + 1e-9:
                        break
                    x0 = 1.0 - x1 - x2 - x3 - x4
                    if value_fn is not None:
                        v = float(value_fn(x0, x1, x2, x3, x4))
                    else:
                        v = float(rng.standard_normal())
                    rows.append([x1, x2, x3, x4, v])
    return np.array(rows)
