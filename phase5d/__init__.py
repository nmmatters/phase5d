"""
phase5d — 5D phase diagram visualization for high-entropy alloys.

Quick start
-----------
>>> import numpy as np
>>> from phase5d import PhaseDiagram5D
>>> from phase5d.utils import generate_grid_data
>>>
>>> data = generate_grid_data(step=0.05)          # synthetic 5-component grid
>>> pd5 = PhaseDiagram5D(data, value_type='continuous', colormap='viridis')
>>> fig, ax = pd5.plot_frame(x0=0.2)
>>> pd5.create_video(output_path='diagram.mp4', fps=8)
"""

from .plotter import PhaseDiagram5D
from .video import assemble_video, check_ffmpeg
from .utils import generate_grid_data, x0_grid, validate_data

__all__ = [
    "PhaseDiagram5D",
    "assemble_video",
    "check_ffmpeg",
    "generate_grid_data",
    "x0_grid",
    "validate_data",
]

__version__ = "0.1.0"
