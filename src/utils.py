"""
utils.py

Generic utility functions for the project (figure saving, etc.).
"""

from __future__ import annotations

import os
from typing import Optional

from matplotlib.figure import Figure

# Commonly used raster/vector formats supported by matplotlib.
SUPPORTED_FORMATS = {"png", "jpg", "jpeg", "pdf", "svg", "eps", "tiff"}


def save_figure(
    fig: Figure,
    file_name: str,
    subfolder: str = "figures",
    dpi: int = 300,
    file_format: str = "png",
    overwrite: bool = True,
) -> str:
    """Save a matplotlib figure in a subfolder relative to the project.

    The output path is computed relative to the location of this file
    (utils.py), not to the current working directory, so saving works
    consistently regardless of where the notebook/script is launched from.

    Args:
        fig: matplotlib Figure object to save.
        file_name: name of the output file, without extension.
        subfolder: subfolder (relative to the project root, i.e. one level
            above the folder containing this file) where the figure is
            saved. Default "figures".
        dpi: image resolution in dots per inch. Default 300.
        file_format: output file format (e.g. "png", "pdf", "svg").
            Default "png".
        overwrite: if False, raise an error if the file already exists
            instead of silently overwriting it. Default True.

    Returns:
        The path of the saved file, relative to the project root (the
        folder above the one containing this file), with forward slashes.
        Relative paths are returned so that no local absolute path ends
        up in the notebook outputs.

    Raises:
        ValueError: if the requested format is not supported, or if the
            file already exists and overwrite=False.
        OSError: if saving fails (e.g. permission denied).
    """
    normalized_format = file_format.lower().lstrip(".")
    if normalized_format not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Format '{file_format}' is not supported. "
            f"Valid formats: {sorted(SUPPORTED_FORMATS)}"
        )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.normpath(os.path.join(script_dir, ".."))
    output_dir = os.path.normpath(os.path.join(project_root, subfolder))

    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(
        output_dir, f"{file_name}.{normalized_format}"
    )

    if not overwrite and os.path.exists(output_file):
        raise ValueError(
            f"File '{output_file}' already exists and overwrite=False."
        )

    try:
        fig.savefig(output_file, dpi=dpi, bbox_inches="tight")
    except OSError as e:
        raise OSError(f"Unable to save the figure to '{output_file}': {e}") from e

    # Path relative to the project root (no absolute path in the output)
    try:
        relative_path = os.path.relpath(output_file, project_root)
    except ValueError:  # e.g. output on a different drive (Windows)
        relative_path = output_file
    relative_path = relative_path.replace(os.sep, "/")

    print(f"Figure saved to: {relative_path}")

    return relative_path