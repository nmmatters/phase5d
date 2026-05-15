"""
Video assembly for phase5d frame sequences.

Assembly strategy (tried in order):
  1. System ffmpeg on PATH  — best quality, requires separate install.
  2. imageio-ffmpeg bundled binary  — works out of the box if imageio is
     installed (``pip install imageio[ffmpeg]`` or ``conda install imageio``).
  3. imageio fallback  — writes via imageio.mimwrite; no ffmpeg needed but
     output quality may be lower for some formats.
"""

import os
import subprocess
from typing import List, Optional


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ffmpeg_exe() -> Optional[str]:
    """
    Return a usable ffmpeg executable path, or None.

    Tries the system PATH first, then the imageio-ffmpeg bundled binary.
    """
    # 1. System ffmpeg
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return "ffmpeg"
    except FileNotFoundError:
        pass

    # 2. imageio-ffmpeg bundled binary
    try:
        import imageio.plugins.ffmpeg as _iio_ff
        exe = _iio_ff.get_exe()
        if exe and os.path.isfile(exe):
            return exe
    except Exception:
        pass

    return None


def _assemble_via_ffmpeg(
    exe: str,
    frame_paths: List[str],
    output_path: str,
    fps: int,
    crf: int,
    verbose: bool,
) -> None:
    """Run ffmpeg concat demuxer to assemble frames into mp4."""
    frame_dir = os.path.dirname(os.path.abspath(frame_paths[0]))
    list_file = os.path.join(frame_dir, "_phase5d_frames.txt")
    duration = 1.0 / fps

    try:
        with open(list_file, "w", encoding="utf-8") as fh:
            for path in frame_paths:
                safe = os.path.abspath(path).replace("\\", "/")
                fh.write(f"file '{safe}'\n")
                fh.write(f"duration {duration:.6f}\n")
            # Concat demuxer requires a trailing entry without duration
            safe = os.path.abspath(frame_paths[-1]).replace("\\", "/")
            fh.write(f"file '{safe}'\n")

        cmd = [
            exe,
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_file,
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", str(crf),
            output_path,
        ]

        result = subprocess.run(cmd, capture_output=not verbose, text=True)
        if result.returncode != 0:
            stderr = result.stderr if not verbose else "(see ffmpeg output above)"
            raise RuntimeError(
                f"ffmpeg exited with code {result.returncode}.\n{stderr}"
            )
    finally:
        if os.path.exists(list_file):
            os.remove(list_file)


def _assemble_via_imageio(
    frame_paths: List[str],
    output_path: str,
    fps: int,
    verbose: bool,
) -> None:
    """Assemble frames using imageio (no external ffmpeg required)."""
    import imageio
    import numpy as np

    if verbose:
        print("  (using imageio for assembly — install ffmpeg for best quality)")

    frames = [imageio.imread(p) for p in frame_paths]
    imageio.mimwrite(output_path, frames, fps=fps, quality=8)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def assemble_video(
    frame_paths: List[str],
    output_path: str,
    fps: int = 10,
    crf: int = 18,
    verbose: bool = True,
) -> str:
    """
    Assemble an ordered list of PNG frames into an mp4 video.

    Assembly is attempted in this order:
      1. System ``ffmpeg`` (if on PATH).
      2. ``imageio-ffmpeg`` bundled binary (if ``imageio`` is installed).
      3. ``imageio.mimwrite`` direct fallback.

    Parameters
    ----------
    frame_paths : list of str
        Ordered list of PNG file paths.
    output_path : str
        Destination ``.mp4`` file path.
    fps : int
        Frames per second.
    crf : int
        Constant Rate Factor for libx264 (lower = higher quality; 18 is
        near-lossless).  Only used when assembling via ffmpeg.
    verbose : bool
        Print progress information.

    Returns
    -------
    str
        Absolute path of the created video.

    Raises
    ------
    RuntimeError
        If no assembly method succeeds.
    """
    if not frame_paths:
        raise ValueError("frame_paths is empty — nothing to assemble.")

    if verbose:
        print(f"Assembling {len(frame_paths)} frames -> {output_path}  (fps={fps})")

    exe = _ffmpeg_exe()

    if exe is not None:
        _assemble_via_ffmpeg(exe, frame_paths, output_path, fps, crf, verbose)
    else:
        try:
            _assemble_via_imageio(frame_paths, output_path, fps, verbose)
        except ImportError:
            raise RuntimeError(
                "No video assembly backend found.  Install one of:\n"
                "  pip install imageio[ffmpeg]   # recommended\n"
                "  conda install ffmpeg          # system ffmpeg\n"
                "  winget install ffmpeg         # Windows"
            )

    abs_output = os.path.abspath(output_path)
    if verbose:
        print(f"Video saved: {abs_output}")
    return abs_output


def check_ffmpeg() -> Optional[str]:
    """
    Return a description of the available video backend, or None.

    Checks system ffmpeg first, then imageio-ffmpeg.  Useful for
    diagnosing issues before starting a long render.
    """
    # System ffmpeg
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.splitlines()[0]
    except FileNotFoundError:
        pass

    # imageio-ffmpeg
    try:
        import imageio.plugins.ffmpeg as _iio_ff
        exe = _iio_ff.get_exe()
        if exe and os.path.isfile(exe):
            return f"imageio-ffmpeg: {exe}"
    except Exception:
        pass

    return None
