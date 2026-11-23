#!/usr/bin/env python3
"""Species-resolved orbital-displacement coupling magnitudes and differences.

A single heatmap defaults to sqrt(Nc * sum_(alpha in species,a,k,q) |d|^2 / (Nk*Nq)).
This removes the explicit 1/sqrt(Nc) factor in the stored d tensor.
Finite-size, sampling and regression differences can still remain.
A comparison is the selected calculation heatmap MINUS --compare-to heatmap.
It is NOT the norm of the complex tensor difference or a relative percentage.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import json
import re
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from visualization_common import add_calculation_arguments, calculation, link_comparison

# Presentation settings (font sizes in points, figure size in inches).
FONT_SIZE = 36
TICK_SIZE = 33
LABEL_SIZE = 39
TITLE_SIZE = 42
COLORBAR_LABEL_SIZE = 36
COLORBAR_TICK_SIZE = 33
FIGURE_SIZE = (16, 15.5)
TITLE_PAD = 20

def remove_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def read_poscar_atom_species(filename: Path) -> list[str]:
    """Return the atom-species list in primitive-cell atom order."""
    lines = [line.rstrip("\n") for line in filename.read_text().splitlines()]
    if len(lines) < 8:
        raise ValueError(f"POSCAR seems too short: {filename}")

    species = lines[5].split()
    counts = [int(x) for x in lines[6].split()]

    if len(species) != len(counts):
        raise ValueError(
            f"Species/count mismatch in POSCAR {filename}: "
            f"{species} vs {counts}"
        )

    atom_species: list[str] = []
    for sp, count in zip(species, counts):
        atom_species.extend([sp] * count)
    return atom_species


def read_hconf_orbitals(filename: Path) -> dict[str, list[str]]:
    """Read the orbital token list from every 'begin atom ELEMENT ... end' block."""
    result: dict[str, list[str]] = {}
    current_species: str | None = None

    for raw_line in filename.read_text().splitlines():
        line = remove_comment(raw_line)
        if not line:
            continue

        m = re.match(r"(?i)^begin\s+atom\s+(\S+)\s*$", line)
        if m:
            current_species = m.group(1)
            if current_species in result:
                raise ValueError(
                    f"Duplicate atom block for species {current_species} in {filename}"
                )
            result[current_species] = []
            continue

        if line.lower() == "end":
            current_species = None
            continue

        if current_species is None:
            continue

        m = re.match(r"(?i)^orbitals\s*=\s*(.*)$", line)
        if m:
            tokens = m.group(1).split()
            if not tokens:
                raise ValueError(f"Empty orbital list for species {current_species}")
            result[current_species].extend(tokens)

    if not result:
        raise ValueError(f"No 'begin atom ... end' blocks found in {filename}")

    for species, orbitals in result.items():
        if not orbitals:
            raise ValueError(f"No orbitals found for species {species} in {filename}")

    return result


def build_primitive_orbital_labels(
    primitive_atom_species: list[str],
    species_orbitals: dict[str, list[str]],
) -> tuple[list[str], list[tuple[int, str, str, int]]]:
    """Build one label per primitive orbital index mu=1..Norb."""
    labels: list[str] = []
    metadata: list[tuple[int, str, str, int]] = []

    for alpha, species in enumerate(primitive_atom_species, start=1):
        if species not in species_orbitals:
            raise ValueError(
                f"Species {species} appears in POSCAR but has no 'begin atom {species}' "
                "block in hconf."
            )

        orbitals = species_orbitals[species]
        for local_index, token in enumerate(orbitals, start=1):
            labels.append(f"{species}[{local_index}]")
            metadata.append((alpha, species, token, local_index))

    return labels, metadata


# -----------------------------------------------------------------------------
# HDF5 processing
# -----------------------------------------------------------------------------

def decode_h5_strings(array) -> list[str]:
    out = []
    for x in array:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        elif hasattr(x, "decode"):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out


def load_species_resolved_orbital_matrices(
    d_filename: Path,
    primitive_atom_species: list[str],
    average_over_kq: bool = True,
) -> tuple[dict[str, np.ndarray], int, int, int]:
    """
    Compute one Norb x Norb matrix for each atomic species.

    If average_over_kq=True, each block contribution is divided by Nk*Nq.
    The square root is taken only after contributions from all same-species
    primitive atoms and Cartesian components have been accumulated.
    """
    matrices: dict[str, np.ndarray] = {}

    with h5py.File(d_filename, "r") as f:
        number_of_orbitals = int(f["number_of_primitive_orbitals"][()])
        block_count = int(f["block_index/number_of_blocks"][()])

        block_alpha = np.array(f["block_index/primitive_atom_alpha"][()], dtype=int)
        block_mu = np.array(f["block_index/mu"][()], dtype=int)
        block_nu = np.array(f["block_index/nu"][()], dtype=int)
        dataset_names = decode_h5_strings(f["block_index/dataset_name"][()])

        if block_count < 1:
            raise ValueError("No d-blocks were found in electron_lattice_d_kq.h5")

        first_block = np.array(f[f"d_blocks/{dataset_names[0]}"][()])
        nq, nk = first_block.shape  # h5py reverses Julia axes; the complete k,q sum is invariant.

        for species in sorted(set(primitive_atom_species)):
            matrices[species] = np.zeros(
                (number_of_orbitals, number_of_orbitals), dtype=float
            )

        norm = float(nk * nq) if average_over_kq else 1.0

        for i in range(block_count):
            alpha = int(block_alpha[i])
            mu = int(block_mu[i]) - 1
            nu = int(block_nu[i]) - 1

            if alpha < 1 or alpha > len(primitive_atom_species):
                raise ValueError(
                    f"Primitive atom alpha={alpha} from d-file is outside "
                    "the primitive POSCAR atom list."
                )

            species = primitive_atom_species[alpha - 1]
            dataset_name = dataset_names[i]
            d_kq = np.array(f[f"d_blocks/{dataset_name}"][()])

            contrib = np.sum(np.abs(d_kq) ** 2) / norm
            matrices[species][mu, nu] += float(contrib)

    return matrices, number_of_orbitals, nk, nq


def load_pair_resolved_orbital_matrices(directory, primitive_species, average):
    """Read direct-Fourier checkpoints: physical d = sum_d / Nsnapshots.

    HDF5.jl stores logical (k,q,ionic_component) in h5py order
    (ionic_component,q,k). Ionic components run x,y,z for each primitive atom.
    """
    with h5py.File(directory / "pair_0001_0001.h5", "r") as first:
        norb = int(first["mapping/norb"][()])
        nalpha = int(first["mapping/nalpha"][()])
        reps = np.asarray(first["mapping/reps"][()], dtype=int)
        count = int(first["number_of_snapshots"][()])
        window = int(first["completed_window"][()])
        signature = first["signature"][()]
    if len(primitive_species) != nalpha or count <= 0:
        raise ValueError("Pair-file primitive species or snapshot count is invalid.")
    nc = int(np.prod(reps))
    matrices = {s: np.zeros((norb,norb)) for s in set(primitive_species)}
    normalization = float(count)**2 * (nc*nc if average else 1)
    for mu in range(1,norb+1):
        for nu in range(1,norb+1):
            path = directory / f"pair_{mu:04d}_{nu:04d}.h5"
            with h5py.File(path,"r") as f:
                if (int(f["number_of_snapshots"][()]) != count or
                    int(f["completed_window"][()]) != window or
                    f["signature"][()] != signature):
                    raise ValueError(f"Pair files have inconsistent progress or inputs: {path}")
                if (int(f["primitive_orbital_mu"][()]),int(f["primitive_orbital_nu"][()])) != (mu,nu):
                    raise ValueError(f"Unexpected orbital-pair labels: {path}")
                dataset = f["sum_d"]
                if dataset.shape != (3*nalpha,nc,nc):
                    raise ValueError(f"Unexpected Julia sum_d storage dimensions: {dataset.shape}")
                for component in range(3*nalpha):
                    values = dataset[component,:,:]
                    if not np.isfinite(values).all():
                        raise ValueError(f"Non-finite d data: {path}")
                    species = primitive_species[component//3]
                    matrices[species][mu-1,nu-1] += float(np.sum(np.abs(values)**2))/normalization
        print(f"Read orbital-pair row {mu}/{norb}; Nsnapshots={count}",flush=True)
    return matrices,norb,nc,nc


def load_calculation(calc, average, d_file=None, hconf=None, poscar=None, normalization="primitive-cell"):
    if d_file is None:
        candidates = [calc.data_dir / name for name in (
            "electron_lattice_d_kq.h5", "electron_lattice_d_kq_direct.h5", ".direct_eph_work")]
        d_file = next((p for p in candidates if p.is_file() or (p/"pair_0001_0001.h5").is_file()),candidates[0])
    hconf = hconf or calc.source_dir / "hconf"
    poscar = poscar or calc.primitive_poscar()
    species = read_poscar_atom_species(poscar)
    labels, metadata = build_primitive_orbital_labels(species, read_hconf_orbitals(hconf))
    paired = d_file.is_dir()
    loader = load_pair_resolved_orbital_matrices if paired else load_species_resolved_orbital_matrices
    squared, norb, nk, nq = loader(d_file, species, average)
    if len(labels) != norb:
        raise ValueError(f"Orbital labels ({len(labels)}) disagree with d ({norb}): {calc.label}")
    metadata_file = d_file/"pair_0001_0001.h5" if paired else d_file
    with h5py.File(metadata_file, "r") as f:
        reps = np.asarray(f["mapping/reps" if paired else "repetitions_n1_n2_n3"][()], dtype=int)
        count = int(f["number_of_snapshots"][()]) if "number_of_snapshots" in f else None
        if tuple(reps) != tuple(map(int, calc.cell.split("x"))):
            raise ValueError(f"d tensor supercell differs from selected {calc.cell}")
    nc = int(np.prod(reps))
    amplitude_scale = np.sqrt(nc) if normalization == "primitive-cell" else 1.0
    matrices = {s: amplitude_scale * np.sqrt(values) for s, values in squared.items()}
    if not all(np.isfinite(m).all() for m in matrices.values()):
        raise ValueError(f"Non-finite raw d heatmap values: {d_file}")
    return matrices, labels, metadata, dict(calculation=str(calc.relative), d_file=str(d_file),
        input_format="direct-Fourier pair files; d=sum_d/Nsnapshots" if paired else "combined d blocks",
        number_of_snapshots=count, number_of_primitive_cells=nc,
        normalization=normalization, amplitude_scale=float(amplitude_scale),
        hconf=str(hconf), primitive_poscar=str(poscar), nk=nk, nq=nq)


def physical_label(calc):
    """Physical calculation labels; method identifiers belong in filenames."""
    cell = calc.cell.replace("x", r"\times")
    temperature = calc.temperature.removesuffix("K")
    return rf"${cell}$, {temperature} K"


def comparison_title(current, reference):
    """Keep shared parameters once and name the signed difference on one line."""
    methods = {"TB_only": "TB", "ML_corrected": "ML"}
    cell = current.cell.replace("x", r"\times")
    temperature = current.temperature.removesuffix("K")
    if current.cell == reference.cell and current.method == reference.method:
        current_t = current.temperature.removesuffix("K")
        reference_t = reference.temperature.removesuffix("K")
        return rf"${cell}$, {methods[current.method]}: {current_t} K minus {reference_t} K"
    if current.cell == reference.cell and current.temperature == reference.temperature:
        return rf"${cell}$, {temperature} K: {methods[current.method]} minus {methods[reference.method]}"
    if current.method == reference.method and current.temperature == reference.temperature:
        reference_cell = reference.cell.replace("x", r"\times")
        return rf"{methods[current.method]}, {temperature} K: ${cell}$ minus ${reference_cell}$"
    return (f"{methods[current.method]} {physical_label(current)} minus "
            f"{methods[reference.method]} {physical_label(reference)}")


def plot_matrix(matrix, labels, species, title, output, difference, average, normalization):
    n = len(labels)
    plt.rcParams.update({"font.size": FONT_SIZE, "xtick.labelsize": TICK_SIZE,
                         "ytick.labelsize": TICK_SIZE})
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, layout="constrained")
    options = {}
    if difference:
        limit = float(np.max(np.abs(matrix))) or 1e-15
        options = dict(cmap="RdBu_r", vmin=-limit, vmax=limit)
    # QuadMesh keeps every heatmap cell vector-based in the SVG.
    edges = np.arange(n + 1) - .5
    im = ax.pcolormesh(edges, edges, matrix, shading="flat", rasterized=False, **options)
    ax.set_aspect("equal")
    ax.set_xticks(np.arange(n), labels, rotation=60, ha="right", rotation_mode="anchor", fontsize=TICK_SIZE)
    ax.set_yticks(np.arange(n), labels, fontsize=TICK_SIZE)
    ax.set_xlabel(r"Orbital $\nu$", fontsize=LABEL_SIZE)
    ax.set_ylabel(r"Orbital $\mu$", fontsize=LABEL_SIZE)
    ax.set_title(f"{species} displacements\n{title}", fontsize=TITLE_SIZE, pad=TITLE_PAD)
    color_label = "Difference in coupling magnitude [eV/Å]" if difference else "Orbital–displacement coupling magnitude [eV/Å]"
    if not average or normalization != "primitive-cell":
        convention = "momentum norm" if not average else "momentum RMS"
        color_label += f"\n{normalization}; {convention}"
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(color_label, fontsize=COLORBAR_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)
    cbar.ax.yaxis.get_offset_text().set_fontsize(COLORBAR_TICK_SIZE)
    # Matplotlib otherwise rasterizes a long continuous colorbar by default.
    if cbar.solids is not None:
        cbar.solids.set_rasterized(False)
    with matplotlib.rc_context({"svg.fonttype": "path"}):
        fig.savefig(output, format="svg")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_calculation_arguments(parser)
    parser.add_argument("--compare-to", type=Path, help="Reference calculation path; plot SELECTED minus REFERENCE.")
    parser.add_argument("--d-file", type=Path)
    parser.add_argument("--hconf", type=Path)
    parser.add_argument("--primitive-poscar", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-average-kq", action="store_true", help="Use an unaveraged k,q sum; this includes mesh-size dependence.")
    parser.add_argument("--normalization", choices=("primitive-cell", "finite-supercell"), default="primitive-cell",
                        help="Default primitive-cell multiplies stored-d RMS by sqrt(Nc); finite-supercell reproduces the old heatmaps.")
    args = parser.parse_args()
    current = calculation(args)
    average = not args.no_average_kq
    values, labels, metadata, info = load_calculation(current, average, args.d_file, args.hconf, args.primitive_poscar, args.normalization)
    reference = calculation(args, args.compare_to) if args.compare_to else None
    compared = None
    if reference:
        compared, other_labels, other_metadata, other_info = load_calculation(reference, average, normalization=args.normalization)
        if labels != other_labels or metadata != other_metadata or values.keys() != compared.keys():
            raise ValueError("Comparison requires identical primitive species and orbital labels/order.")
        title = comparison_title(current, reference)
        name = f"difference_{current.tag}_minus_{reference.tag}"
        output = args.output_dir or current.source_dir / "species_orbital_d_heatmaps" / name
    else:
        title = physical_label(current)
        output = args.output_dir or current.source_dir / "species_orbital_d_heatmaps"
    output = output.expanduser().resolve()
    # Never replace an input directory with output or follow a misleading comparison symlink.
    output.mkdir(parents=True, exist_ok=True)
    with (output/"primitive_orbital_labels.txt").open("w") as f:
        f.write("# index alpha species orbital_token local_index label\n")
        for i, (entry, label) in enumerate(zip(metadata, labels), 1):
            f.write(f"{i} {' '.join(map(str,entry))} {label}\n")
    for species in sorted(values):
        matrix = values[species] - compared[species] if reference else values[species]
        stem = f"d_species_{species}{'_difference' if reference else ''}"
        plot_tag = f"{current.tag}_minus_{reference.tag}" if reference else current.tag
        plot_matrix(matrix, labels, species, title, output/f"{stem}_{plot_tag}.svg", bool(reference), average, args.normalization)
        np.savetxt(output/(stem+".dat"), matrix, fmt="%.12e")
        if reference:
            np.savetxt(output/f"d_species_{species}_selected.dat", values[species], fmt="%.12e")
            np.savetxt(output/f"d_species_{species}_reference.dat", compared[species], fmt="%.12e")
    scale_formula = "Nc * " if args.normalization == "primitive-cell" else ""
    denominator = "/(Nk*Nq)" if average else ""
    report = dict(selected=info, average_over_kq=average, normalization=args.normalization, units="eV/angstrom",
        quantity_name="Species-resolved orbital-displacement coupling magnitude",
        plot_format="svg", plot_tag=plot_tag,
        font_sizes=dict(base=FONT_SIZE, ticks=TICK_SIZE, labels=LABEL_SIZE,
                        title=TITLE_SIZE, colorbar_label=COLORBAR_LABEL_SIZE,
                        colorbar_ticks=COLORBAR_TICK_SIZE),
        orbital_labels=labels, title_parameters=title,
        quantity=f"sqrt({scale_formula}sum_species_atoms,cartesian,k,q(abs(d)^2){denominator})",
        output=str(output))
    if reference:
        report.update(reference=other_info, difference="selected heatmap minus reference heatmap; signed; after square root")
    (output/"visualization_sources.json").write_text(json.dumps(report, indent=2)+"\n")
    if reference:
        link_comparison(output, (current, reference))
    print(f"Saved vector SVG heatmaps and numerical tables: {output}")


if __name__ == "__main__":
    main()


