"""
data_loader.py

Functions for loading and converting the raw C-MAPSS data
(.txt format -> .csv).
"""

from __future__ import annotations

import csv
import os
from typing import Optional, Sequence


def txt_to_csv(
    input_file_name: str,
    output_file_name: str,
    column_names: Sequence[str],
    delimiter: Optional[str] = None,
    input_dir: str = os.path.join("..", "data", "raw"),
    output_dir: str = os.path.join("..", "data", "csv"),
    overwrite: bool = True,
) -> str:
    """Convert a .txt file to .csv format, assigning column names.

    Each line of the input file is split into fields (on the given
    delimiter, or on runs of spaces/tabs if delimiter is None) and written
    as a row of the output CSV, with column_names as the header.

    Args:
        input_file_name: name of the input .txt file (without path).
        output_file_name: name of the output .csv file (without path).
        column_names: column names to use as the header. Every line of the
            input file must have the same number of fields as
            column_names, otherwise an error is raised (to avoid silently
            misaligned columns).
        delimiter: field separator character; if None, the default split
            on runs of spaces/tabs is used. Default None.
        input_dir: input folder, relative to the location of this file.
            Default "../data/raw".
        output_dir: output folder, relative to the location of this file.
            Default "../data/csv".
        overwrite: if False, raise an error if the output file already
            exists instead of silently overwriting it. Default True.

    Returns:
        The absolute path of the created .csv file.

    Raises:
        FileNotFoundError: if the input file does not exist.
        ValueError: if the output file already exists and overwrite=False,
            or if a line of the input file has a number of fields different
            from len(column_names).
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    input_file = os.path.normpath(
        os.path.join(script_dir, input_dir, input_file_name)
    )
    output_file = os.path.normpath(
        os.path.join(script_dir, output_dir, output_file_name)
    )

    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file not found: '{input_file}'")

    if not overwrite and os.path.exists(output_file):
        raise ValueError(
            f"File '{output_file}' already exists and overwrite=False."
        )

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    expected_n_columns = len(column_names)
    rows_written = 0

    with open(input_file, "r", encoding="utf-8") as f_in, \
         open(output_file, "w", newline="", encoding="utf-8") as f_out:

        writer = csv.writer(f_out)
        writer.writerow(column_names)

        for line_number, line in enumerate(f_in, start=1):
            line = line.strip()
            if not line:
                continue  # skip empty lines

            fields = line.split(delimiter) if delimiter else line.split()

            if len(fields) != expected_n_columns:
                raise ValueError(
                    f"Line {line_number} of file '{input_file_name}' has "
                    f"{len(fields)} fields, but column_names specifies "
                    f"{expected_n_columns}. Line content: {line!r}"
                )

            writer.writerow(fields)
            rows_written += 1

    # Print the path relative to the project root (no absolute path in the output)
    project_root = os.path.normpath(os.path.join(script_dir, ".."))
    try:
        relative_output = os.path.relpath(output_file, project_root)
    except ValueError:  # e.g. output on a different drive (Windows)
        relative_output = output_file
    relative_output = relative_output.replace(os.sep, "/")

    print(f"File created: {relative_output} ({rows_written} rows written)")

    return output_file