#!/usr/bin/env python3
"""Generate CsPbBr3 heatmaps and every higher-temperature minus lower-temperature pair."""
import argparse
from itertools import combinations
from pathlib import Path
import re
import sys

import plot_species_orbital_d_heatmaps as plotter
from visualization_common import DEFAULT_PROJECT_ROOT, DEFAULT_DATA_ROOT, METHODS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS, default='ML_corrected')
    parser.add_argument('--cell', default='4x4x4')
    parser.add_argument('--temperatures', nargs='+', help='Optional subset, e.g. 425K 525K 625K')
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()
    source = DEFAULT_PROJECT_ROOT / args.method / args.cell
    if args.temperatures:
        temperatures = args.temperatures
    else:
        temperatures = [p.name for p in source.iterdir()
                        if p.is_dir() and re.fullmatch(r'\d+(?:\.\d+)?K', p.name)
                        and (args.data_root / args.method / args.cell / p.name /
                             'electron_lattice_d_kq.h5').is_file()]
    temperatures = sorted(set(temperatures), key=lambda value: float(value.removesuffix('K')))
    if not temperatures:
        parser.error('No completed d tensors found for this method/cell.')

    # Cache small magnitude matrices, never the full d tensors.
    original_loader = plotter.load_calculation
    cache = {}
    def load_once(calc, average, d_file=None, hconf=None, poscar=None, normalization='primitive-cell'):
        key = (calc, average, d_file, hconf, poscar, normalization)
        if key not in cache:
            print('Loading d:', calc.relative, flush=True)
            cache[key] = original_loader(calc, average, d_file, hconf, poscar, normalization)
            print('Loaded d:', calc.relative, flush=True)
        return cache[key]
    plotter.load_calculation = load_once

    def run(temperature, reference=None):
        current = source / temperature
        sys.argv = ['plot_species_orbital_d_heatmaps.py', '--calc-dir', str(current),
                    '--data-root', str(args.data_root)]
        if reference:
            other = source / reference
            tag = f'difference_{args.method}_{args.cell}_{temperature}_minus_{args.method}_{args.cell}_{reference}'
            output = other / 'species_orbital_d_heatmaps' / tag
            sys.argv += ['--compare-to', str(other), '--output-dir', str(output)]
        plotter.main()
    for temperature in temperatures:
        run(temperature)
    for lower, higher in combinations(temperatures, 2):
        run(higher, lower)
    number = len(temperatures) + len(list(combinations(temperatures, 2)))
    print(f'Completed {number} heatmap sets for temperatures {temperatures}.', flush=True)


if __name__ == '__main__':
    main()
