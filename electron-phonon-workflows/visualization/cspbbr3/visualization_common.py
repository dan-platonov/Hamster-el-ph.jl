"""Shared calculation selection for all CsPbBr3 visualization entry points."""
from dataclasses import dataclass
from pathlib import Path
import argparse
import os
import re

METHODS = ("TB_only", "ML_corrected")
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROJECT_ROOT = Path(os.environ.get("CSPBBR3_PROJECT_ROOT", REPOSITORY_ROOT / "examples/CsPbBr3"))
DEFAULT_DATA_ROOT = Path(os.environ.get("CSPBBR3_DATA_ROOT", DEFAULT_PROJECT_ROOT))


@dataclass(frozen=True)
class Calculation:
    method: str
    cell: str
    temperature: str
    project_root: Path
    data_root: Path

    @property
    def relative(self):
        return Path(self.method) / self.cell / self.temperature

    @property
    def source_dir(self):
        return self.project_root / self.relative

    @property
    def data_dir(self):
        path = self.data_root / self.relative
        return path / "data" if (path / "data").is_dir() else path

    @property
    def tag(self):
        return f"{self.method}_{self.cell}_{self.temperature}"

    @property
    def label(self):
        return f"{self.method}, {self.cell}, {self.temperature}"

    def primitive_poscar(self):
        for name in ("POSCAR_equil_primitive", "POSCAR_primitive"):
            path = self.source_dir / name
            if path.is_file():
                return path
        raise FileNotFoundError(f"No primitive POSCAR in {self.source_dir}")


def add_calculation_arguments(parser):
    parser.add_argument("--calc-dir", type=Path,
                        help="Calculation path, e.g. TB_only/4x4x4/400K; also accepts an absolute source/data path.")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--cell", help="Supercell repetitions, e.g. 4x4x4 or 10x10x10.")
    parser.add_argument("--temperature", help="Nominal temperature, e.g. 100 or 100K.")
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)


def path_identity(path):
    parts = Path(path).expanduser().parts
    for i in range(len(parts)-3, -1, -1):
        if parts[i] in METHODS and i+2 < len(parts):
            return parts[i:i+3]
    raise ValueError(f"Cannot infer method/cell/temperature from {path}")


def normalize_temperature(value):
    text = str(value).strip().removesuffix("K").removesuffix("k")
    number = float(text)
    if number <= 0:
        raise ValueError("Temperature must be positive.")
    return f"{number:g}K"


def calculation(args, path=None):
    selected = path if path is not None else args.calc_dir
    if selected is not None:
        method, cell, temperature = path_identity(selected)
    elif args.method and args.cell and args.temperature:
        method, cell, temperature = args.method, args.cell, args.temperature
    else:
        try:
            method, cell, temperature = path_identity(Path.cwd())
        except ValueError as exc:
            raise ValueError("Select --calc-dir METHOD/CELL/TEMPERATURE, or all of --method, --cell, --temperature.") from exc
    if path is None:
        method = args.method or method
        cell = args.cell or cell
        temperature = args.temperature or temperature
    if method not in METHODS or not re.fullmatch(r"[1-9]\d*x[1-9]\d*x[1-9]\d*", cell):
        raise ValueError(f"Invalid calculation identity: {method}/{cell}/{temperature}")
    result = Calculation(method, cell, normalize_temperature(temperature),
                         args.project_root.expanduser().resolve(), args.data_root.expanduser().resolve())
    if not result.source_dir.is_dir():
        raise FileNotFoundError(f"Calculation source directory missing: {result.source_dir}")
    return result


def configure_plot_args(args, output_subdir):
    calc = calculation(args)
    args.g_file = args.g_file or calc.data_dir / "electron_phonon_g_band_kq.h5"
    args.output_dir = args.output_dir or calc.source_dir / output_subdir
    if hasattr(args, "primitive_poscar"):
        args.primitive_poscar = args.primitive_poscar or calc.primitive_poscar()
    print(f"Calculation: {calc.label}\nData: {calc.data_dir}\nOutput: {args.output_dir}")
    return calc


def link_comparison(output_dir, calculations):
    """Expose one saved comparison under both participating source directories."""
    output_dir = output_dir.resolve()
    for calc in calculations:
        link = calc.source_dir / "species_orbital_d_heatmaps" / output_dir.name
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.resolve() == output_dir:
            continue
        if link.exists() or link.is_symlink():
            raise FileExistsError(f"Comparison link already points elsewhere: {link}")
        link.symlink_to(os.path.relpath(output_dir, link.parent), target_is_directory=True)
        print(f"Comparison reference: {link} -> {link.readlink()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List available calculation source directories and data availability.")
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()
    for method in METHODS:
        for source in sorted((args.project_root/method).glob("*x*x*/*K")):
            relative = source.relative_to(args.project_root)
            data = args.data_root/relative
            available = [name for name in ("electron_lattice_d_kq.h5", "electron_phonon_g_band_kq.h5") if (data/name).is_file()]
            print(f"{relative}: {', '.join(available) or 'no final d/g file yet'}")
