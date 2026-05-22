# phase5d

**5D phase diagram visualization for high-entropy alloys.**

Five-component alloy systems (HEAs) have a composition space too large to
visualize in a single plot.  `phase5d` solves this by slicing the 5D simplex
along one composition axis x₀ and rendering each slice as a 3D tetrahedron —
then sweeping x₀ from 0 → 1 to produce a video of the complete phase space.

---

## Demo

<video src="media/femnnicocu_pv_surface.mp4" width="100%" autoplay loop muted playsinline controls></video>

**FeMnNiCoCu at 873 K** — CALPHAD phase stability, x(Fe) swept from 0.00 to 1.00
in steps of 0.01.  Each frame is a cross-section of the four-component
(Mn–Ni–Co–Cu) composition space at fixed Fe content.

| Color | Phase |
|-------|-------|
| Dark gray (opaque) | Unstable |
| Light gray (semi-transparent) | Meta-stable |
| Transparent | Stable single-phase region |

Rendered with `render='surface'` (PyVista, adaptive alpha shapes, 101 frames,
`mode='fixed'`).

---

## Concept

A five-component alloy (x₀, x₁, x₂, x₃, x₄) satisfies the simplex constraint

```
x₀ + x₁ + x₂ + x₃ + x₄ = 1
```

For a fixed x₀, the remaining four components form a **3-simplex (tetrahedron)**.
Sweeping x₀ from 0 → 1 explores the full 5D phase space one slice at a time:

```
x₀ = 0.00  →  full tetrahedron  (all Mn–Ni–Co–Cu compositions available)
x₀ = 0.40  →  60 % of the range remains
x₀ = 1.00  →  single point      (pure Fe)
```

Both render modes show a `x(component) = value` text label in the top-left corner of each frame, and a **composition scale bar** at the bottom. The blue filled portion of the bar equals `1 − x₀` — the fraction of the full quaternary composition space currently displayed. The bar shrinks as x₀ increases, giving an at-a-glance sense of how much composition space is visible in each frame.

---

## Installation

### Requirements

| Package | Version | Notes |
|---------|---------|-------|
| Python  | ≥ 3.9   | |
| NumPy   | ≥ 1.24  | |
| Matplotlib | ≥ 3.7 | |
| SciPy   | ≥ 1.10  | |
| PyVista | ≥ 0.43  | |
| scikit-image | ≥ 0.19 | required for `plot_isosurface()` only — `pip install phase5d[isosurface]` |
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
git clone https://github.com/nmmatters/phase5d.git
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

data = generate_grid_data(step=0.01)   # synthetic example

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

Two input formats are supported, controlled by the `x0` parameter:

**`x0='implicit'` (default) — shape (N, 5)**

```
  col 0 → x1   composition of component 1
  col 1 → x2   composition of component 2
  col 2 → x3   composition of component 3
  col 3 → x4   composition of component 4
  col 4 → value  scalar property or phase label

  x0 = 1 − x1 − x2 − x3 − x4   (computed automatically)
```

**`x0='explicit'` — shape (N, 6)**

```
  col 0 → x0   composition of component 0  (the sweep axis)
  col 1 → x1   composition of component 1
  col 2 → x2   composition of component 2
  col 3 → x3   composition of component 3
  col 4 → x4   composition of component 4
  col 5 → value  scalar property or phase label

  x0 + x1 + x2 + x3 + x4 must be ≤ 1 for all rows
```

The library handles regular grids (e.g. 0.01 step) and arbitrary scattered
points equally.

---

## Visualization modes

### Tetrahedron scaling (`mode`)

Three modes control how the tetrahedron is scaled as x₀ changes:

| Mode | Description |
|------|-------------|
| `'fixed'` **(default)** | Tetrahedron always fills the full viewport. x₀ label top-left; scale bar at bottom. Easiest to read in a video. |
| `'shrink_center'` | Tetrahedron shrinks by factor `(1−x₀)` around its centroid. |
| `'shrink_corner'` | Tetrahedron shrinks toward the origin (pure-x₀ corner). |

```python
fig, ax = pd5.plot_frame(x0=0.3, mode='shrink_center')
```

### Render style (`render`)

Two rendering styles are available:

| Style | Description | Dependencies |
|-------|-------------|--------------|
| `'scatter'` **(default)** | Each composition point is a marker. Fast and faithful to the raw data distribution. Uses matplotlib. | — |
| `'surface'` | High-quality alpha-shape surface mesh with smooth shading, proper lighting, and specular highlights via PyVista. Recommended for publication figures and video. | `scipy`, `pyvista` |

```python
# Single scatter frame (matplotlib)
fig, ax = pd5.plot_frame(x0=0.3, render='scatter')

# Surface video (PyVista — writes directly to file, cannot return a figure)
pd5.create_video(x0_values=..., output_path='video.mp4', render='surface')

# Single surface frame
pd5.save_frame_surface(x0=0.3, out_path='frame.png')
```

### Alpha shape parameter (`shape_alpha`)

Both surface renderers use an **alpha shape** (concave hull) to reconstruct the phase
boundary from scattered composition points.  A tetrahedron from the Delaunay
triangulation is kept only if its circumradius `R < 1 / shape_alpha`.

**Default behaviour — adaptive alpha:**
When `shape_alpha` is not set, it is chosen automatically per frame as

```
shape_alpha = 90 × (N / 62196)^(1/3)
```

where `N` is the number of points in the current x₀ slice (all points, no
subsampling) and 62 196 is the reference count at x₀ = 0.30 for a step = 0.01
FeMnNiCoCu grid.  This keeps the circumradius threshold proportional to the
local grid spacing, so surface quality stays consistent as the slice becomes
sparser at higher x₀.

> **Important:** the alpha-shape path always uses **all** per-phase points —
> no random subsampling — to preserve surface quality.  Random subsampling
> destroys the regular grid structure and produces blurry, irregular surfaces.

**Manual override:**
Pass `shape_alpha` as a keyword argument to fix the value across all frames:

```python
# Fixed alpha — same threshold for every frame
pd5.create_video(..., render='surface', shape_alpha=90)

# Let the library pick per-frame (default)
pd5.create_video(..., render='surface')
```

**Choosing a value (step = 0.01 grid):**

| `shape_alpha` | Effect |
|---|---|
| 2 – 10 | Near-convex hull; smooth but hides fine phase boundary detail |
| 20 – 50 | Captures main structural features |
| 80 – 90 | Maximum detail before fragmentation on a step = 0.01 grid |
| > 100 | Surface fragments — individual triangles break apart |

The fragmentation threshold scales with grid density: for step = 0.05 the ceiling
is around `shape_alpha ≈ 18`; for step = 0.01 it is ≈ 90.  A safe starting
point is `shape_alpha ≈ 0.9 / step`.

### Composition sparsity at high x₀ (`min_points`)

As x₀ increases toward 1, the available composition space for the remaining
components shrinks rapidly.  For a step = 0.01 grid on FeMnNiCoCu data, the
slice at x₀ = 0.85 contains only ~800 points — too few for a reliable
alpha-shape surface.

The `min_points` parameter (default **1000**) controls this: frames below the
threshold skip surface rendering and show only the tetrahedron wireframe and
vertex labels.  An empty tetrahedron does **not** imply full phase stability —
it simply means the data density at that composition is too low to reconstruct
a meaningful surface.

```python
# Default: wireframe-only below 1000 points
pd5.create_video(..., render='surface')

# Lower threshold to render more frames (may show artefacts at sparse slices)
pd5.create_video(..., render='surface', min_points=500)

# Disable threshold entirely
pd5.create_video(..., render='surface', min_points=4)
```

For step = 0.01 FeMnNiCoCu data the surface quality is good up to x₀ ≈ 0.84
(~969 points); above that only the wireframe is shown.

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
    phase_names={-1: "Unstable", 0: "Meta-stable", 1: "Stable"},
)
```

Stable regions are invisible by default; set `phase_alphas={1: 0.15}` to
show them faintly.

### Phase-number labels (1–5)

Any integer label scheme is supported — pass matching `phase_colors`, `phase_alphas`,
and `phase_names` dicts.  A typical use case is a **phase-count** dataset where the
value column holds the number of co-existing equilibrium phases (1 = single-phase,
2 = two-phase, …):

```python
pd5 = PhaseDiagram5D(
    data,                                   # value column holds 1–5
    value_type='phase_stability',
    phase_colors={
        1: (1.00, 1.00, 1.00),             # white       – single-phase
        2: (0.20, 0.40, 0.80),             # blue        – two-phase
        3: (0.80, 0.20, 0.20),             # red         – three-phase
        4: (0.15, 0.65, 0.25),             # green       – four-phase
        5: (0.85, 0.55, 0.05),             # amber       – five-phase
    },
    phase_alphas={1: 0.00, 2: 0.55, 3: 0.55, 4: 0.55, 5: 0.55},
    phase_names={1: "1-phase", 2: "2-phase", 3: "3-phase", 4: "4-phase", 5: "5-phase"},
    component_labels=['Fe', 'Mn', 'Ni', 'Co', 'Cu'],
)
```

The legend automatically shows only labels that are **both present in the data
and not fully transparent** (`alpha ≥ 0.001`).  `phase_names` controls what text
appears in the legend; if omitted the raw integer is shown instead.

---

## API reference

### `PhaseDiagram5D`

```python
PhaseDiagram5D(
    data,
    x0               = 'implicit',            # 'implicit' (N×5) | 'explicit' (N×6)
    value_type       = 'continuous',          # 'continuous' | 'phase_stability'
    colormap         = 'viridis',             # matplotlib colormap name
    vmin             = None,                  # float or None
    vmax             = None,                  # float or None
    tolerance        = 0.005,                 # x0 slice half-width
    component_labels = ['x₀','x₁','x₂','x₃','x₄'],
    phase_colors     = None,                  # dict {label: (R,G,B)}
    phase_alphas     = None,                  # dict {label: alpha}
    phase_names      = None,                  # dict {label: str} — legend text per label
    value_label      = '',                    # colorbar label, e.g. 'Gm'
    value_unit       = '',                    # colorbar unit,  e.g. 'J/mol'
)
```

#### `.plot_frame(x0, …)` — scatter only (matplotlib)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `x0` | — | x₀ slice value |
| `mode` | `'fixed'` | Tetrahedron scaling mode |
| `render` | `'scatter'` | Only `'scatter'` supported; use `create_video` for surfaces |
| `alpha` | `0.65` | Point alpha (continuous only) |
| `marker_size` | `3` | Scatter marker size (pt²) |
| `max_points` | `15000` | Max points rendered (random sub-sample) |
| `show_wireframe` | `True` | Draw tetrahedron edges |
| `wireframe_alpha` | `0.85` | Wireframe transparency |
| `wireframe_color` | `'black'` | Wireframe edge color |
| `show_vertex_labels` | `True` | Show component names at vertices |
| `elev`, `azim` | `20`, `-45` | Camera angles (°) |
| `figsize` | `(8, 9)` | Figure size (inches) |
| `title` | `None` | Axes title |
| `dpi` | `100` | Figure resolution |

Returns `(fig, ax)`.

#### `.save_frame_surface(x0, out_path, …)` — surface only (PyVista)

Renders a single high-quality surface frame off-screen and saves it as a PNG.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `x0` | — | x₀ slice value |
| `out_path` | — | Output PNG file path |
| `mode` | `'fixed'` | Tetrahedron scaling mode |
| `shape_alpha` | `None` | Alpha-shape tightness; `None` = adaptive `90×(N/62196)^(1/3)` |
| `window_size` | `(1400, 1000)` | Off-screen render resolution (pixels) |
| `camera_position` | `None` | PyVista camera triple `[pos, focal, up]`; `None` = library default |
| `show_wireframe` | `True` | Draw tetrahedron edges |
| `show_vertex_labels` | `True` | Label vertices with component names |
| `max_points` | `50000` | Max points for continuous mode (no effect on phase_stability) |
| `min_points` | `1000` | Min slice points to attempt surface rendering; below threshold renders wireframe only |
| `markers` | `None` | List of `[x0,x1,x2,x3,x4]` compositions shown as spheres when the frame's x₀ matches within `tolerance` |
| `tielines` | `None` | List of endpoint pairs `[[x0_A,…],[x0_B,…]]`; the tie-line's intersection with the current x₀ plane is shown as a red sphere |
| `tietriangles` | `None` | List of N-vertex simplices; each simplex is a list of N compositions `[x0,x1,x2,x3,x4]`. The cross-section with the current x₀ plane is drawn as an orange polygon + spheres. Works for any N ≥ 2 (see below) |
| `marker_color` | `'red'` | Color of marker / tie-line spheres |
| `marker_size` | `18` | Sphere point size for markers and tie-line intersections |
| `triangle_color` | `'orange'` | Color of tie-simplex edges and spheres |
| `triangle_size` | `14` | Sphere point size for tie-simplex intersection points |

Returns the total number of points in the slice.

---

### Equilibrium phase overlays

`tielines` and `tietriangles` let you overlay multi-phase equilibrium regions on top of any surface frame or video.  As x₀ sweeps through the simplex, the cross-section of each tie-object with the current x₀ plane is computed and drawn automatically.

#### Two-phase equilibrium (tie-line)

Two equilibrium compositions connected by a tie-line.  Where the x₀ plane intersects the line, a single red sphere is shown.

```python
NOMINAL = [0.20, 0.10, 0.20, 0.20, 0.30]   # nominal alloy composition
PHASE_A = [0.22, 0.21, 0.21, 0.21, 0.15]   # equilibrium phase A
PHASE_B = [0.00, 0.07, 0.06, 0.01, 0.86]   # equilibrium phase B

pd5.save_frame_surface(
    x0=0.20, out_path='frame_tieline.png',
    markers=[NOMINAL],
    tielines=[[PHASE_A, PHASE_B]],
)

pd5.create_video(
    x0_values=np.round(np.arange(0.00, 0.23, 0.01), 3),
    output_path='tieline.mp4',
    fps=3, render='surface',
    markers=[NOMINAL],
    tielines=[[PHASE_A, PHASE_B]],
)
```

#### Three-phase equilibrium (tie-triangle)

Three equilibrium compositions forming a tie-triangle.  The x₀ plane intersects up to three edges; the resulting cross-section is rendered as an orange line segment or triangle.

```python
PHASE_A = [0.05, 0.60, 0.20, 0.10, 0.05]
PHASE_B = [0.15, 0.20, 0.50, 0.20, 0.10]
PHASE_C = [0.10, 0.20, 0.15, 0.50, 0.15]

pd5.save_frame_surface(
    x0=0.10, out_path='frame_tietriangle.png',
    tietriangles=[[PHASE_A, PHASE_B, PHASE_C]],
)
```

<video src="media/femnnicocu_pv_tietriangle.mp4" width="100%" autoplay loop muted playsinline controls></video>

**FeMnNiCoCu tie-triangle example** — three equilibrium phases, x(Fe) swept from 0.00 to 0.24.

#### Four-phase equilibrium (tie-tetrahedron) and beyond

The same `tietriangles` parameter handles any number of equilibrium phases.  Pass a list with four or more compositions and the code automatically enumerates all N(N−1)/2 edges of the simplex.  The cross-section with the x₀ plane gives a convex polygon (line, triangle, quadrilateral, …) drawn in orange.

```python
# Four equilibrium phases — cross-section at x0=0.50 is a quadrilateral
PHASE_A = [0.30, 0.60, 0.04, 0.03, 0.03]   # Mn-rich, low-Fe
PHASE_B = [0.30, 0.04, 0.60, 0.03, 0.03]   # Ni-rich, low-Fe
PHASE_C = [0.70, 0.04, 0.03, 0.18, 0.05]   # Co-rich, high-Fe
PHASE_D = [0.70, 0.04, 0.03, 0.05, 0.18]   # Cu-rich, high-Fe

pd5.save_frame_surface(
    x0=0.50, out_path='frame_4phase.png',
    tietriangles=[[PHASE_A, PHASE_B, PHASE_C, PHASE_D]],
)
```

Multiple independent simplices can be passed in the same list:

```python
tietriangles=[
    [PHASE_A, PHASE_B, PHASE_C],          # one tie-triangle
    [PHASE_D, PHASE_E, PHASE_F, PHASE_G], # one tie-tetrahedron
]
```

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
| `elev`, `azim` | `20`, `-45` | Camera angles (°) |
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
All `plot_frame` and `save_frame_surface` keyword arguments are forwarded.

#### `.create_video(x0_values, output_path, fps=10, dpi=150, keep_frames=False, frames_dir=None, …)`

Renders frames and assembles them into an mp4 via ffmpeg.
`x0_values` defaults to the full grid inferred from the data.
Returns the absolute path of the video file.

When `keep_frames=True` and `frames_dir` is not given, frames are saved to
`<videoname>_frames/` in the same directory as the output video.

All `plot_frame` / `save_frame_surface` keyword arguments (including
`render`, `mode`, `shape_alpha`, `min_points`) are forwarded via `**plot_kwargs`.

#### `.show_interactive(x0_init=0.0, mode='fixed', render='scatter', …)`

Opens a **live matplotlib window** with three slider widgets:

| Slider | Range | Description |
|--------|-------|-------------|
| `x(Fe)` | data min → max | Sweeps the x₀ composition slice; snaps to the grid step detected from the data |
| `Elevation` | −90 → 90 | Camera elevation angle (°) |
| `Azimuth` | 0 → 360 | Camera azimuth angle (°) |

All `plot_frame` keyword arguments (`alpha`, `marker_size`, `show_wireframe`,
`wireframe_alpha`, `wireframe_color`, `show_vertex_labels`, `figsize`,
`**kwargs`) are accepted.  The colorbar / legend is created once at startup
and does not flicker on slider updates.

> **Note**: `render='surface'` (PyVista) is **not** supported in the
> interactive viewer — only `render='scatter'` (default).  Use
> `create_video(..., render='surface')` for surface-rendered output.

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
| [`example_continuous.py`](examples/example_continuous.py) | Mixing enthalpy landscape, all three tetrahedron modes, scatter video |
| [`example_phase_stability.py`](examples/example_phase_stability.py) | Phase stability labels, scatter + alpha-shape surface render, video |
| [`example_isosurface.py`](examples/example_isosurface.py) | Isosurface rendering — single, nested, and rotated views |
| [`example_surface_pv.py`](examples/example_surface_pv.py) | **Surface rendering** (`render='surface'`, PyVista) — FeMnNiCoCu at 873 K, x(Fe) 0.00→0.40. Uses real TCHEA4 data if present; falls back to synthetic otherwise |
| [`example_tieline.py`](examples/example_tieline.py) | **Tie-line overlay** — nominal composition marker + two-phase equilibrium tie-line swept across the x₀ window. Uses real TCHEA4 data if present; falls back to synthetic otherwise |

Run from the repository root:

```bash
python examples/example_continuous.py
python examples/example_phase_stability.py
python examples/example_isosurface.py
python examples/example_surface_pv.py          # requires: pip install pyvista
python examples/example_tieline.py            # requires: pip install pyvista
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
- **Surface video render time**: a full 101-frame video with `render='surface'`
  at `step=0.01` takes roughly **15–20 minutes** on a modern desktop (tested on
  FeMnNiCoCu TCHEA4 at 873 K).  Dense slices near x₀ = 0 contain up to ~177 k
  points; the alpha-shape Delaunay triangulation scales super-linearly with point
  count.  Slices near x₀ = 1 finish in under 1 s.  For a quick preview use every
  5th frame (`x0_values=np.arange(0.00, 1.01, 0.05)`) which completes in ~3 minutes.
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
