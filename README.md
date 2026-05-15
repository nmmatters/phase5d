# phase5d

**5D phase diagram visualization for high-entropy alloys.**

`phase5d` renders the full composition space of a five-component alloy system
as a sequence of tetrahedron figures — one per x₀ slice — that can be
assembled into a video.  Each frame is a regular tetrahedron whose four
vertices represent the four independent composition axes (x₁ – x₄).  The
fifth axis, x₀, is displayed via a scale bar below the tetrahedron.

---

## Concept

A five-component alloy (x₀, x₁, x₂, x₃, x₄) satisfies the constraint

```
x₀ + x₁ + x₂ + x₃ + x₄ = 1
```

For a fixed x₀ value the remaining four components satisfy
`x₁ + x₂ + x₃ + x₄ = 1 − x₀`, which forms a **3-simplex (tetrahedron)**.
By sweeping x₀ from 0 → 1 and rendering one frame per step, the full 5D
phase space is explored.

```
x₀ = 0.00  →  full tetrahedron  (scale = 1.00)
x₀ = 0.40  →  same tetrahedron, 60 % of composition range active
x₀ = 1.00  →  single point      (scale = 0.00)
```

The scale bar at the bottom of each frame encodes this information visually.

---

## Installation

### Requirements

| Package | Version | Notes |
|---------|---------|-------|
| Python  | ≥ 3.9   | |
| NumPy   | ≥ 1.24  | |
| Matplotlib | ≥ 3.7 | |
| SciPy   | ≥ 1.10  | required for `render='surface'` and `plot_isosurface()` |
| scikit-image | ≥ 0.19 | required for `plot_isosurface()` only |
| **ffmpeg** | any | for video output |

Install ffmpeg via your system package manager:

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows (winget)
winget install ffmpeg
```

### Install phase5d

```bash
git clone https://github.com/nmarschall/phase5d.git
cd phase5d
pip install -e .
```

---

## Quick start

```python
import numpy as np
from phase5d import PhaseDiagram5D
from phase5d.utils import generate_grid_data

# 1. Load or generate data  ─────────────────────────────────────────────────
#    shape (N, 5):  columns = [x1, x2, x3, x4, value]
#    x0 is implicit:  x0 = 1 − x1 − x2 − x3 − x4

data = generate_grid_data(step=0.05)   # synthetic example

# 2. Create diagram object  ──────────────────────────────────────────────────
pd5 = PhaseDiagram5D(data, value_type='continuous', colormap='viridis')

# 3. Plot a single frame  ────────────────────────────────────────────────────
fig, ax = pd5.plot_frame(x0=0.20)
fig.savefig('frame.png', dpi=150, bbox_inches='tight', facecolor='white')

# 4. Create a video sweep  ───────────────────────────────────────────────────
pd5.create_video(
    x0_values=np.arange(0.0, 1.01, 0.05),
    output_path='diagram.mp4',
    fps=8,
)
```

---

## Data format

```
data : np.ndarray, shape (N, 5)

  col 0 → x1   composition of component 1
  col 1 → x2   composition of component 2
  col 2 → x3   composition of component 3
  col 3 → x4   composition of component 4
  col 4 → value  scalar property or phase label

  x0 = 1 − x1 − x2 − x3 − x4   (implicit, not stored)
```

The library handles regular grids (e.g. 0.01 step) and arbitrary scattered
points equally.

---

## Visualization modes

### Tetrahedron scaling (`mode`)

Three modes control how the tetrahedron is scaled as x₀ changes:

| Mode | Description |
|------|-------------|
| `'fixed'` **(default)** | Tetrahedron always fills the full viewport. The scale is shown only via the bar. Easiest to read in a video. |
| `'shrink_center'` | Tetrahedron shrinks by factor `(1−x₀)` around its centroid. |
| `'shrink_corner'` | Tetrahedron shrinks toward the origin (pure-x₀ corner). |

```python
fig, ax = pd5.plot_frame(x0=0.3, mode='shrink_center')
```

### Render style (`render`)

Two rendering styles are available for the data points in each x₀ slice:

| Style | Description |
|-------|-------------|
| `'scatter'` **(default)** | Each composition point is a marker. Fast and faithful to the raw data distribution. |
| `'surface'` | The convex hull of each phase region is rendered as a polygon mesh. One hull per stability class for `phase_stability`; hull faces colored by the scalar value for `continuous` mode. Requires `scipy`. |

```python
fig, ax = pd5.plot_frame(x0=0.3, render='surface')
```

---

## Value types

### Continuous scalar

```python
pd5 = PhaseDiagram5D(
    data,
    value_type='continuous',
    colormap='RdBu_r',   # any matplotlib colormap
    vmin=-10,            # optional; defaults to data min/max
    vmax=10,
)
```

### Phase stability labels (−1 / 0 / 1)

```python
pd5 = PhaseDiagram5D(
    data,
    value_type='phase_stability',
    phase_colors={-1: (0.1, 0.1, 0.1),   # dark gray  – unstable
                   0: (0.7, 0.7, 0.7),   # light gray – meta-stable
                   1: (1.0, 1.0, 1.0)},  # white      – stable
    phase_alphas={-1: 1.0,               # opaque
                   0: 0.5,               # semi-transparent
                   1: 0.0},              # invisible (fully transparent)
)
```

Stable regions are invisible by default; set `phase_alphas={1: 0.15}` to
show them faintly.

---

## API reference

### `PhaseDiagram5D`

```python
PhaseDiagram5D(
    data,
    value_type       = 'continuous',          # 'continuous' | 'phase_stability'
    colormap         = 'viridis',             # matplotlib colormap name
    vmin             = None,                  # float or None
    vmax             = None,                  # float or None
    tolerance        = 0.005,                 # x0 slice half-width
    component_labels = ['x₀','x₁','x₂','x₃','x₄'],
    phase_colors     = None,                  # dict {label: (R,G,B)}
    phase_alphas     = None,                  # dict {label: alpha}
)
```

#### `.plot_frame(x0, …)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `x0` | — | x₀ slice value |
| `mode` | `'fixed'` | Tetrahedron scaling mode |
| `render` | `'scatter'` | `'scatter'` or `'surface'` |
| `alpha` | `0.65` | Point alpha (continuous only) |
| `marker_size` | `3` | Scatter marker size (pt²) |
| `max_points` | `15000` | Max points rendered (random sub-sample) |
| `show_wireframe` | `True` | Draw tetrahedron edges |
| `wireframe_alpha` | `0.20` | Wireframe transparency |
| `show_vertex_labels` | `True` | Show component names at vertices |
| `elev`, `azim` | `20`, `45` | Camera angles (°) |
| `figsize` | `(8, 9)` | Figure size (inches) |
| `title` | `None` | Axes title |
| `dpi` | `100` | Figure resolution |

Returns `(fig, ax)`.

#### `.plot_isosurface(level, …)`

Render one or more isosurfaces through the **full** 4-simplex composition space
using the natural barycentric embedding
`P = x₁·V₀ + x₂·V₁ + x₃·V₂ + x₄·V₃`.
A `LinearNDInterpolator` builds a continuous scalar field from the scattered
data; `marching_cubes` (scikit-image) extracts the surface(s).

| Parameter | Default | Description |
|-----------|---------|-------------|
| `level` | — | Isosurface value(s) — float or list of floats |
| `colors` | `'auto'` | Colors per level; `'auto'` uses `tab10` |
| `alpha` | `0.5` | Surface transparency |
| `grid_resolution` | `50` | Voxels per axis for interpolation grid |
| `max_points` | `50000` | Max points used for interpolation |
| `show_wireframe` | `True` | Draw master tetrahedron wireframe |
| `show_colorbar` | `True` | Add colorbar (continuous mode) |
| `elev`, `azim` | `20`, `45` | Camera angles (°) |
| `figsize` | `(8, 8)` | Figure size (inches) |

Returns `(fig, ax)`.

```python
# Single isosurface
fig, ax = pd5.plot_isosurface(level=-5000)

# Multiple levels with custom colors
fig, ax = pd5.plot_isosurface(
    level=[-8000, -5000, -2000],
    colors=['royalblue', 'gold', 'tomato'],
    alpha=0.4,
    grid_resolution=60,
)
fig.savefig('isosurfaces.png', dpi=150, bbox_inches='tight')
```

#### `.save_frames(x0_values, output_dir, dpi=150, verbose=True, **plot_kwargs)`

Saves one PNG per x₀ value.  Returns a list of file paths.

#### `.create_video(x0_values, output_path, fps=10, dpi=150, keep_frames=False, …)`

Renders frames and assembles them into an mp4 via ffmpeg.
`x0_values` defaults to the full grid inferred from the data.
Returns the absolute path of the video file.

#### `.show_interactive(x0_init=0.0, mode='fixed', render='scatter', …)`

Opens a **live matplotlib window** with three slider widgets:

| Slider | Range | Description |
|--------|-------|-------------|
| `x(Fe)` | data min → max | Sweeps the x₀ composition slice; snaps to the grid step detected from the data |
| `Elevation` | −90 → 90 | Camera elevation angle (°) |
| `Azimuth` | 0 → 360 | Camera azimuth angle (°) |

All `plot_frame` keyword arguments (`alpha`, `marker_size`, `show_wireframe`,
`wireframe_alpha`, `wireframe_color`, `show_vertex_labels`, `figsize`,
`render`, `**kwargs`) are accepted.  The colorbar / legend is created once
at startup and does not flicker on slider updates.

> **Note**: requires an interactive matplotlib backend.  If you are inside a
> Jupyter notebook, run `%matplotlib widget` (or `qt`) first.  In a plain
> script, the call blocks until the window is closed.

```python
pd5.show_interactive(x0_init=0.2, render='scatter', max_points=5000)
```

#### `.save_vtk(output_path, max_points=None, include_compositions=True)`

Exports the full dataset as a **VTK XML Unstructured Grid** (`.vtu`) file for
use in [ParaView](https://www.paraview.org/) or any VTK-compatible viewer.

Every point is placed at its natural barycentric position
`P = x₁·V₀ + x₂·V₁ + x₃·V₂ + x₄·V₃` — the same embedding used by
`plot_isosurface`.  Scalar fields written per point:

| Field | Content |
|-------|---------|
| `value` (or `value_label`) | The scalar property column |
| `x0` … `x4` | Individual composition fractions |
| `stability` | Stability labels (−1 / 0 / 1), if available |

The file uses inline base64 binary encoding and requires **no extra
dependencies** (uses Python's stdlib `base64` module).

**Suggested ParaView workflow**

1. *File → Open* the `.vtu` file and click *Apply*.
2. Add a **Threshold** filter on `x0` to interactively slice by Fe content.
3. Add a **Contour** filter on `value` to extract isosurfaces.
4. Switch the representation to *Point Gaussian* for scatter-style rendering;
   color by any scalar field.

```python
pd5.save_vtk('phase_diagram.vtu')                          # full dataset
pd5.save_vtk('phase_diagram_small.vtu', max_points=200_000)  # downsampled
```

Returns the absolute path of the written file.

---

### Utility functions

```python
from phase5d.utils import generate_grid_data, x0_grid, validate_data
from phase5d.video  import check_ffmpeg
```

| Function | Description |
|----------|-------------|
| `generate_grid_data(step, value_fn, seed)` | Synthetic regular-grid dataset |
| `x0_grid(data, step)` | x₀ values present in the dataset |
| `validate_data(data)` | Validate shape and composition constraints |
| `check_ffmpeg()` | Return ffmpeg version string, or None |

---

## Examples

See the [`examples/`](examples/) directory:

| File | Description |
|------|-------------|
| [`example_continuous.py`](examples/example_continuous.py) | Mixing enthalpy landscape, all three modes, video |
| [`example_phase_stability.py`](examples/example_phase_stability.py) | Phase stability labels, custom colors, video |
| [`example_isosurface.py`](examples/example_isosurface.py) | Isosurface rendering — single, nested, and rotated views |

Run from the repository root:

```bash
python examples/example_continuous.py
python examples/example_phase_stability.py
```

---

## Tips

- **Resolution 0.01 grid**: at x₀ = 0 a full slice has ~171 k points.
  Use `max_points=10000` (default 15000) to keep rendering fast.
- **Camera angle**: `elev=25, azim=45` usually gives a good view of all
  four vertices.  Add `azim` as a slowly changing function of the frame
  index to create a rotating video.
- **Colormap**: `'RdBu_r'` works well for enthalpy (red = positive,
  blue = negative); `'plasma'` for monotone properties.
- **ffmpeg not found?** `from phase5d.video import check_ffmpeg; print(check_ffmpeg())`
- **Interactive exploration**: `pd5.show_interactive()` opens a live window with sliders for x₀, elevation, and azimuth — no video needed.
- **ParaView export**: `pd5.save_vtk('diagram.vtu')` writes the full point cloud with all scalar fields; use Threshold on `x0` and Contour on `value` in ParaView for fully interactive 3-D analysis.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Built with Claude Code

This library was developed with the assistance of
[Claude Code](https://claude.ai/claude-code) — Anthropic's agentic coding
tool.  Claude Code was used throughout the design and implementation process,
including the coordinate geometry, color-mapping pipeline, isosurface
algorithm, and documentation.
