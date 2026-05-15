"""
ffmpeg-based video assembly for phase5d frame sequences.
"""

import os
import subprocess
from typing import List, Optional


def assemble_video(
    frame_paths: List[str],
    output_path: str,
    fps: int = 10,
    crf: int = 18,
    verbose: bool = True,
) -> str:
    """
    Assemble an ordered list of PNG frames into an mp4 video using ffmpeg.

    ffmpeg must be installed and available on PATH.

    Parameters
    ----------
    frame_paths : list of str
        Ordered list of absolute (or resolvable) PNG file paths.
    output_path : str
        Destination .mp4 file path.
    fps : int
        Frames per second.
    crf : int
        Constant Rate Factor for libx264 (lower = higher quality; 18 is near-lossless).
    verbose : bool
        Show ffmpeg progress.

    Returns
    -------
    str
        Absolute path of the created video.

    Raises
    ------
    FileNotFoundError
        If ffmpeg is not found on PATH.
    RuntimeError
        If ffmpeg exits with a non-zero return code.
    """
    if not frame_paths:
        raise ValueError("frame_paths is empty — nothing to assemble.")

    # Write a concat list file next to the first frame
    frame_dir = os.path.dirname(os.path.abspath(frame_paths[0]))
    list_file = os.path.join(frame_dir, "_phase5d_frames.txt")

    duration = 1.0 / fps
    try:
        with open(list_file, "w", encoding="utf-8") as fh:
            for path in frame_paths:
                # Use forward slashes — ffmpeg prefers them even on Windows
                safe_path = os.path.abspath(path).replace("\\", "/")
                fh.write(f"file '{safe_path}'\n")
                fh.write(f"duration {duration:.6f}\n")
            # ffmpeg concat demuxer needs the last entry repeated without duration
            safe_path = os.path.abspath(frame_paths[-1]).replace("\\", "/")
            fh.write(f"file '{safe_path}'\n")

        cmd = [
            "ffmpeg",
            "-y",                       # overwrite output
            "-f", "concat",
            "-safe", "0",
            "-i", list_file,
            "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # ensure even dimensions
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",      # broad player compatibility
            "-crf", str(crf),
            output_path,
        ]

        if verbose:
            print(f"Assembling {len(frame_paths)} frames → {output_path}  (fps={fps})")

        result = subprocess.run(
            cmd,
            capture_output=not verbose,
            text=True,
        )
        if result.returncode != 0:
            stderr = result.stderr if not verbose else "(see ffmpeg output above)"
            raise RuntimeError(
                f"ffmpeg exited with code {result.returncode}.\n{stderr}"
            )

    finally:
        if os.path.exists(list_file):
            os.remove(list_file)

    abs_output = os.path.abspath(output_path)
    if verbose:
        print(f"Video saved: {abs_output}")
    return abs_output


def check_ffmpeg() -> Optional[str]:
    """
    Return the ffmpeg version string, or None if ffmpeg is not found.

    Useful for providing a helpful error message before starting a long render.
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            first_line = result.stdout.splitlines()[0]
            return first_line
    except FileNotFoundError:
        pass
    return None
