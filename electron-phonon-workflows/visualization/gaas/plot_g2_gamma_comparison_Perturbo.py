#!/usr/bin/env python3
"""Compare sum_s |g_nn,s(k=Gamma,q)|^2 along L-Gamma-X-W-K-Gamma.

Defaults: PERTURBO first; TB and ML 100K/400K 4x4x4; TB and ML 100K 10x10x10.
Electronic band n=5; s labels the phonon branch.
Use repeated --curve DIRECTORY LABEL arguments to replace the default
calculation curves. A directory contains electron_phonon_g_band_kq.h5;
Compact selected-k files are preferred; an explicit HDF5 filename or a standard
Ga_As calculation directory also works. --k-reduced selects another initial k.
Use --no-perturbo to omit the reference and --output-dir to choose the destination.

Input g is in eV and already includes phonon normalization; output is meV^2.
Default primitive-cell normalization removes the stored 1/sqrt(Nc) from g:
the plotted sum is 1e6 * Nc * sum_s |g_stored|^2. Nc=prod(repetitions),
including for compact files with only one initial k. PERTURBO is unchanged.
Use --normalization finite-supercell to reproduce the original stored-g plot.
There is no electronic-band sum, k average, interpolation,
fitting or extrapolation. Lines join consecutive actual q-mesh points along
the full path, including across high-symmetry segment boundaries.
Exact phonon Gamma points are omitted, as in the supplied PERTURBO plot.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_my_g2_gamma import LABELS, TICKS, cubic_frame, native_path_points, read_couplings
from visualization_common import DEFAULT_DATA_ROOT, METHODS

# Presentation settings: font sizes are in points, figure size in inches.
FONT_SIZE = 24
TICK_SIZE = 30
LABEL_SIZE = 26
TITLE_SIZE = 26
LEGEND_SIZE = 22
FIGURE_SIZE = (14, 8.5)
LINE_WIDTH = 2.5
MARKER_SIZE = 6
PERTURBO_COLOR = "black"
TB_4X4X4_100K_COLOR = "#ef8a8a"
TB_4X4X4_400K_COLOR = "#b2182b"
ML_4X4X4_100K_COLOR = "#6baed6"
ML_4X4X4_400K_COLOR = "#08519c"
TB_10X10X10_COLOR = "#74c476"
ML_10X10X10_COLOR = "#006d2c"

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__import__("os").environ.get("GAAS_PROJECT_ROOT", REPOSITORY_ROOT/"examples/GaAs"))
REFERENCE_DIR = REPOSITORY_ROOT / "reference_data/PERTURBO"
G_FILENAMES = ("electron_phonon_g_selected_k.h5", "electron_phonon_g_band_kq.h5")
DEFAULT_CURVES = (
    (DEFAULT_DATA_ROOT / "TB_only/4x4x4/100K", "TB 100K 4x4x4"),
    (DEFAULT_DATA_ROOT / "TB_only/4x4x4/400K", "TB 400K 4x4x4"),
    (DEFAULT_DATA_ROOT / "ML_corrected/4x4x4/100K", "ML 100K 4x4x4"),
    (DEFAULT_DATA_ROOT / "ML_corrected/4x4x4/400K", "ML 400K 4x4x4"),
    (DEFAULT_DATA_ROOT / "TB_only/10x10x10/100K", "TB 100K 10x10x10"),
    (DEFAULT_DATA_ROOT / "ML_corrected/10x10x10/100K", "ML 100K 10x10x10"),
)


def curve_style(curve, index):
    """Style calculation metadata, so custom legend labels do not change colours."""
    info = curve["info"]
    if info["kind"] == "PERTURBO":
        return dict(color=PERTURBO_COLOR, linestyle="-")
    parts = Path(info["input"]).parts
    method = next((part for part in parts if part in METHODS), None)
    linestyle = "--" if method == "TB_only" else "-"
    repetitions = tuple(info["repetitions"])
    temperature = next((float(match.group(1)) for part in reversed(parts)
                        if (match := re.fullmatch(r"(\d+(?:\.\d+)?)K", part, re.I))), None)
    if repetitions == (10, 10, 10) and method in METHODS:
        color = TB_10X10X10_COLOR if method == "TB_only" else ML_10X10X10_COLOR
    elif repetitions == (4, 4, 4) and method in METHODS and temperature in (100, 400):
        color = {
            ("TB_only", 100): TB_4X4X4_100K_COLOR,
            ("TB_only", 400): TB_4X4X4_400K_COLOR,
            ("ML_corrected", 100): ML_4X4X4_100K_COLOR,
            ("ML_corrected", 400): ML_4X4X4_400K_COLOR,
        }[method, temperature]
    else:
        color = f"C{index % 10}"
    return dict(color=color, linestyle=linestyle)


def coupling_file(directory: str | Path) -> Path:
    """Prefer the compact Gamma result; accept a full tensor or an explicit k-point file."""
    path = Path(directory).expanduser()
    candidates = [path] if path.is_absolute() else [path, PROJECT_ROOT/path, DEFAULT_DATA_ROOT/path]
    tried = []
    for candidate in [entry for path in candidates for entry in (path, path/"data")]:
        for file in ([candidate] if candidate.is_file() else [candidate/name for name in G_FILENAMES]):
            tried.append(str(file))
            if file.is_file():
                return file.resolve()
        try:
            relative = candidate.resolve().relative_to(PROJECT_ROOT)
        except ValueError:
            continue
        if len(relative.parts) == 3 and relative.parts[0] in METHODS:
            for name in G_FILENAMES:
                file = DEFAULT_DATA_ROOT/relative/name
                tried.append(str(file))
                if file.is_file():
                    return file.resolve()
    raise FileNotFoundError("No coupling result found. Tried:\n" + "\n".join(tried))


def path_geometry(data):
    """Recover reciprocal basis B from q_cart=B q_reduced on the complete mesh."""
    if np.linalg.matrix_rank(data["qr"]) != 3:
        raise ValueError("The stored q mesh does not determine a three-dimensional reciprocal lattice.")
    reciprocal = data["qc"] @ np.linalg.pinv(data["qr"])
    residual = float(np.max(np.abs(reciprocal @ data["qr"] - data["qc"])))
    if residual > 1e-7:
        raise ValueError(f"Stored reduced/Cartesian q coordinates disagree: {residual:.3e} inverse Angstrom.")
    # POSCAR-style lattice rows, as expected by the existing cubic_frame helper.
    primitive = 2*np.pi*np.linalg.inv(reciprocal)
    return reciprocal, cubic_frame(primitive), residual


def coupling_normalization(filename: Path, repetitions):
    """Read conventions from g or its source; never infer normalization from Nk."""
    pending = [filename]
    seen = set()
    while pending:
        source = pending.pop(0).resolve()
        if source in seen or not source.is_file():
            continue
        seen.add(source)
        with h5py.File(source, "r") as f:
            if not np.array_equal(f["repetitions_n1_n2_n3"][()], repetitions):
                raise ValueError(f"Normalization source has a different cell size: {source}")
            for key in ("conventions/g_normalization", "conventions/d_normalization", "normalization"):
                if key not in f:
                    continue
                value = f[key][()]
                value = value.decode() if isinstance(value, bytes) else str(value)
                if key == "conventions/g_normalization" and value in ("primitive-cell", "finite-supercell"):
                    return value, str(source), key
                if value == "1/sqrt(Nc)":
                    return "finite-supercell", str(source), key
                raise ValueError(f"Unknown coupling normalization {value!r} in {source}:{key}")
            for key in ("sources/orbital_basis_g", "sources/d_couplings"):
                if key in f:
                    value = f[key][()]
                    value = value.decode() if isinstance(value, bytes) else str(value)
                    path = Path(value)
                    pending.append(path if path.is_absolute() else source.parent/path)
    # Legacy band files may have been moved together with their primary d file.
    sibling = filename.parent/"electron_lattice_d_kq.h5"
    if sibling.resolve() not in seen and sibling.is_file():
        return coupling_normalization(sibling, repetitions)
    raise ValueError(f"Cannot establish coupling normalization for {filename}")


def read_our_curve(filename: Path, label: str, band: int, gamma_index: int | None,
                   initial_k=None, normalization="primitive-cell"):
    with h5py.File(filename, "r") as f:
        unit = f["units"][()] if "units" in f else None
        if isinstance(unit, bytes):
            unit = unit.decode()
        if unit is not None and str(unit).strip().lower() != "ev":
            raise ValueError(f"Expected g in eV, found {unit!r}: {filename}")
    data = read_couplings(filename, band, gamma_index, 1.0, initial_k_reduced=initial_k)
    stored, normalization_source, normalization_key = coupling_normalization(filename, data["reps"])
    nc = int(np.prod(data["reps"]))
    stored_power = nc if stored == "finite-supercell" else 1
    target_power = nc if normalization == "finite-supercell" else 1
    g2_multiplier = stored_power / target_power
    reciprocal, cubic, residual = path_geometry(data)
    x, indices, segments = native_path_points(data["qr"], reciprocal, cubic)
    # np.sum propagates missing modes; never silently sum only a valid subset.
    total = g2_multiplier * np.sum(data["g2"], axis=0)
    gamma = np.max(np.abs(data["qr"]-np.rint(data["qr"])), axis=0) < 1e-9
    total[gamma] = np.nan
    y = total[indices]
    if not np.any(np.isfinite(y) & (y > 0)):
        raise ValueError(f"No positive, finite all-mode sums on the requested path: {filename}")
    info = dict(label=label, kind="calculation", input=str(filename), band=band,
                number_of_modes=data["nm"], nk=data["nk"], nq=data["nq"],
                initial_k_index_in_file_1based=data["ik"]+1, initial_k_source=data["gamma_source"],
                repetitions=data["reps"].tolist(), q_indices_1based=(indices+1).tolist(),
                number_of_primitive_cells=nc, stored_normalization=stored,
                plotted_normalization=normalization, g2_multiplier=g2_multiplier,
                g_amplitude_multiplier=float(np.sqrt(g2_multiplier)),
                normalization_source=normalization_source, normalization_key=normalization_key,
                path_segments_1based=(segments+1).tolist(),
                reciprocal_coordinate_residual_inv_angstrom=residual)
    return dict(label=label, x=x, y=y, segments=segments, info=info)


def read_perturbo(filename: Path, label: str):
    table = np.genfromtxt(filename, delimiter=",", names=True, dtype=float, encoding="utf-8")
    if table.dtype.names is None or len(table.dtype.names) != 2:
        raise ValueError("PERTURBO CSV must have a two-column header followed by x,y values.")
    x, y = [np.atleast_1d(table[name]).astype(float) for name in table.dtype.names]
    if len(x) < 2 or not np.isfinite(x).all() or np.any(np.diff(x) <= 0):
        raise ValueError("PERTURBO x coordinates must be finite and strictly increasing.")
    if np.isinf(y).any() or not np.any(np.isfinite(y) & (y > 0)):
        raise ValueError("PERTURBO y values must contain a positive finite curve and no infinities.")
    if not np.allclose([x[0],x[-1]], [TICKS[0],TICKS[-1]], atol=1e-6, rtol=0):
        raise ValueError("PERTURBO CSV range differs from the supplied L-Gamma-X-W-K-Gamma path.")
    y[np.isclose(x,TICKS[1],atol=1e-7,rtol=0) |
      np.isclose(x,TICKS[-1],atol=1e-7,rtol=0)] = np.nan
    return dict(label=label, x=x, y=y, segments=None,
                info=dict(label=label, kind="PERTURBO", input=str(filename.resolve()),
                          band=5, number_of_modes=6, plotted_normalization="primitive-cell",
                          g2_multiplier=1.0))


def plotted_rows(curve, logarithmic):
    """Join path samples across segment boundaries; preserve undefined values."""
    y = curve["y"].copy()
    y[~np.isfinite(y)] = np.nan
    if logarithmic:
        y[y <= 0] = np.nan
    return np.column_stack([curve["x"],y])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--curve", nargs=2, action="append", metavar=("DIRECTORY","LABEL"),
                        help="Repeat for each calculation. Supplying any --curve replaces the defaults.")
    parser.add_argument("--no-perturbo", action="store_true", help="Do not read or plot the PERTURBO curve.")
    parser.add_argument("--reference-csv", type=Path,
                        default=REFERENCE_DIR/"gaas_g2_sum_all_phonon_modes.csv")
    parser.add_argument("--perturbo-label", default="PERTURBO")
    parser.add_argument("--output-dir", type=Path, default=REPOSITORY_ROOT/"results/GaAs/coupling_path_in_BZ")
    parser.add_argument("--band", type=int, default=5, help="1-based electronic band n for g_nn,s (default 5).")
    parser.add_argument("--gamma-k-index", type=int, help="Verified 1-based Gamma index, only needed if k coordinates are absent.")
    parser.add_argument("--k-reduced", nargs=3, type=float, metavar=("K1","K2","K3"),
                        help="Initial k in primitive reciprocal coordinates; default Gamma.")
    parser.add_argument("--linear-y", action="store_true", help="Use a linear y axis instead of the default logarithmic scale.")
    parser.add_argument("--normalization", choices=("primitive-cell", "finite-supercell"),
                        default="primitive-cell",
                        help="Default primitive-cell removes stored 1/sqrt(Nc); finite-supercell reproduces raw g. PERTURBO is unchanged.")
    args = parser.parse_args()
    if args.band < 1:
        parser.error("--band must be positive.")
    target = np.zeros(3) if args.k_reduced is None else np.asarray(args.k_reduced)
    if not np.isfinite(target).all():
        parser.error("Initial k must be finite.")
    initial_gamma = np.max(np.abs(target-np.rint(target))) < 1e-9
    if not initial_gamma and not args.no_perturbo:
        parser.error("The PERTURBO reference is for initial Gamma; use --no-perturbo for another initial k.")
    specs = args.curve or DEFAULT_CURVES
    if any(not str(label).strip() for _,label in specs):
        parser.error("Every curve needs a nonempty label.")
    curves = []
    if not args.no_perturbo:
        curves.append(read_perturbo(args.reference_csv.expanduser(),args.perturbo_label))
    for directory,label in specs:
        filename = coupling_file(directory)
        print(f"Reading {label}: {filename}",flush=True)
        curve = read_our_curve(filename,label,args.band,args.gamma_k_index,args.k_reduced,args.normalization)
        curves.append(curve)
        print(f"  {curve['info']['stored_normalization']} -> {args.normalization}: "
              f"multiply |g|^2 by {curve['info']['g2_multiplier']:g}",flush=True)

    plt.rcParams.update({"font.size": FONT_SIZE, "axes.labelsize": LABEL_SIZE,
                         "axes.titlesize": TITLE_SIZE, "xtick.labelsize": TICK_SIZE,
                         "ytick.labelsize": TICK_SIZE, "legend.fontsize": LEGEND_SIZE})
    fig, ax = plt.subplots(figsize=FIGURE_SIZE,layout="constrained")
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True,exist_ok=True)
    reports = []
    for index,curve in enumerate(curves,1):
        rows = plotted_rows(curve,not args.linear_y)
        style = curve_style(curve,index-1)
        ax.plot(rows[:,0],rows[:,1],**style,linewidth=LINE_WIDTH,
                marker=None if curve["segments"] is None else "o",markersize=MARKER_SIZE,
                label=curve["label"])
        slug = re.sub(r"[^\w.-]+","_",curve["label"]).strip("_.") or "curve"
        csv_file = output/f"g2_gamma_sum_curve_{index:02d}_{slug}.csv"
        np.savetxt(csv_file,rows,delimiter=",",header="path_distance,sum_abs_g_squared_meV2",
                   comments="",fmt="%.17g")
        reports.append(dict(curve["info"],plot_style=style,exported_csv=str(csv_file),
                            positive_finite_path_values=int(np.sum(np.isfinite(rows[:,1]) & (rows[:,1]>0)))))
    for position in TICKS:
        ax.axvline(position,color="0.8",linewidth=.7,zorder=0)
    ax.set_xticks(TICKS,LABELS)
    ax.set_xlim(TICKS[0],TICKS[-1])
    ax.set_xlabel(r"Phonon wavevector $\mathbf{q}$ along L--$\Gamma$--X--W--K--$\Gamma$",fontsize=LABEL_SIZE)
    k_label = r"\Gamma" if initial_gamma else "("+", ".join(f"{x:g}" for x in target)+")"
    ax.set_ylabel(r"$\sum_s |g_{nn,s}("+k_label+r",\mathbf{q})|^2$ (meV$^2$)",fontsize=LABEL_SIZE)
    title = rf"GaAs: initial $\mathbf{{k}}={k_label}$, band $n={args.band}$; all phonon modes"
    if args.normalization == "finite-supercell":
        title += "\nFinite-supercell couplings (PERTURBO retains its primitive-cell convention)" if not args.no_perturbo else "\nFinite-supercell couplings"
    if not args.no_perturbo and args.band != 5:
        title += "\nPERTURBO reference: band 5 (verify matching band counting)"
    ax.set_title(title,fontsize=TITLE_SIZE)
    if not args.linear_y:
        ax.set_yscale("log")
    ax.tick_params(labelsize=TICK_SIZE)
    ax.yaxis.get_offset_text().set_fontsize(TICK_SIZE)
    ax.grid(axis="y",alpha=.2)
    ax.legend(fontsize=LEGEND_SIZE,loc="best")
    image = output/"g2_gamma_sum_comparison.svg"
    # Outline fonts so mathematical labels render consistently in SVG viewers.
    with matplotlib.rc_context({"svg.fonttype":"path"}):
        fig.savefig(image,format="svg")
    plt.close(fig)
    report = dict(quantity="sum_s abs(g_nn,s(k_initial,q))^2",
                  notation=dict(electronic_band="n", phonon_branch="s"),
                  font_sizes=dict(base=FONT_SIZE, ticks=TICK_SIZE, labels=LABEL_SIZE,
                                  title=TITLE_SIZE, legend=LEGEND_SIZE),
                  initial_k_reduced_primitive_reciprocal=target.tolist(),
                  units="meV^2", path="L-Gamma-X-W-K-Gamma",
                  path_ticks=TICKS.tolist(), logarithmic_y=not args.linear_y,
                  exact_gamma_omitted=True, normalization=args.normalization,
                  extra_normalization="per-curve conversion recorded in g2_multiplier; PERTURBO unchanged",
                  interpolation="none; lines join consecutive actual mesh points across segment boundaries",
                  curve_order=reports, plot=str(image))
    (output/"g2_gamma_sum_comparison_sources.json").write_text(json.dumps(report,indent=2)+"\n")
    print(f"Saved {len(curves)} curves: {image}")


if __name__ == "__main__":
    main()
