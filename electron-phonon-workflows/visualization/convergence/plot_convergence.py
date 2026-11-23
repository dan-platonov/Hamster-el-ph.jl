#!/usr/bin/env python3

from pathlib import Path
import argparse
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# Presentation settings, in the same style as the coupling-path plots.
FONT_SIZE = 24
TICK_SIZE = 28
LABEL_SIZE = 26
FIGURE_SIZE = (14, 10)
LINE_WIDTH = 2.5
MARKER_SIZE = 6
CURVE_COLOR = "blue"


# Acceptable convergence-log base names
ACCEPTABLE_LOG_FILES = {
    "uH_convergence",
    "d_convergence",
}


def read_convergence_file(filename):
    """
    Read a HAMSTER convergence log.

    Lines beginning with '#' are ignored.

    Expected columns:
        1  N_snapshots
        2  last_frame
        3  B_norm_eV_A
        4  B_rms_eV_A
        5  B_max_abs_eV_A
        6  relative_change_B_norm
        7  r_cumulative
        8  N_added
        9  N_previous
    """

    data = np.loadtxt(
        filename,
        comments="#",
        ndmin=2,
    )

    if data.shape[1] < 7:
        raise ValueError(
            f"{filename.name} contains only {data.shape[1]} columns. "
            "At least 7 columns are required."
        )

    # Column 1
    n_snapshots = data[:, 0]

    # Column 3
    x_norm = data[:, 2]

    # Column 7
    relative_change = data[:, 6]

    return n_snapshots, x_norm, relative_change


def find_log_files(script_dir):
    """
    Find accepted convergence files.

    Examples:
        uH_convergence.txt
        d_convergence.txt
        uH_convergence_100K.txt
        d_convergence_100K.txt
        uH_convergence_250K.txt

    Returns files grouped by suffix:
        ""       -> files without suffix
        "_100K"  -> files ending in _100K.txt
        "_250K"  -> files ending in _250K.txt
    """

    groups = {}

    pattern = re.compile(
        r"^(uH_convergence|d_convergence)(_\d+K)?\.txt$"
    )

    for filename in script_dir.glob("*.txt"):
        match = pattern.match(filename.name)

        if match is None:
            continue

        base_name = match.group(1)

        if base_name not in ACCEPTABLE_LOG_FILES:
            continue

        suffix = match.group(2) or ""

        groups.setdefault(suffix, []).append(filename)

    return groups


def make_plot(log_files, suffix, plot_dir):

    fig, ax = plt.subplots(
        2,
        1,
        figsize=FIGURE_SIZE,
        sharex=True,
        layout="constrained",
    )

    for filename in sorted(log_files):

        n_snapshots, x_norm, relative_change = read_convergence_file(filename)

        ax[0].plot(
            n_snapshots,
            x_norm,
            marker="o",
            color=CURVE_COLOR,
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
        )

        ax[1].plot(
            n_snapshots,
            relative_change,
            marker="o",
            color=CURVE_COLOR,
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
        )

    ax[0].set_ylabel("Tensor norm (eV Å)")

    ax[1].set_xlabel("Number of snapshots")
    ax[1].set_ylabel("Normalized change")

    for axis in ax:
        axis.grid(True, alpha=0.2)
        axis.tick_params(which="major", labelsize=TICK_SIZE,
                         length=8, width=1.5, pad=7)
        axis.yaxis.get_offset_text().set_fontsize(TICK_SIZE)

    output_file = plot_dir / f"convergence{suffix}.svg"

    with matplotlib.rc_context({"svg.fonttype": "path"}):
        fig.savefig(output_file, format="svg", bbox_inches="tight",
                    pad_inches=0.25)

    plt.close(fig)

    print(f"Saved plot to: {output_file}")


def main():

    plt.rcParams.update({"font.size": FONT_SIZE,
                         "axes.labelsize": LABEL_SIZE,
                         "xtick.labelsize": TICK_SIZE,
                         "ytick.labelsize": TICK_SIZE})

    # Directory containing this Python script
    parser = argparse.ArgumentParser(description="Plot covariance-tensor norms and logged normalized changes.")
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    script_dir = args.log_dir.expanduser().resolve()

    # Create plots directory
    plot_dir = args.output_dir.expanduser().resolve()
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Find and group all convergence files
    groups = find_log_files(script_dir)

    if not groups:
        raise FileNotFoundError(
            f"No convergence files found in {script_dir}"
        )

    # One figure for each suffix
    for suffix, log_files in sorted(groups.items()):
        make_plot(
            log_files,
            suffix,
            plot_dir,
        )


if __name__ == "__main__":
    main()
