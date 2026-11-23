#!/usr/bin/env python3
"""Compare your band-basis couplings with the PERTURBO GaAs Gamma benchmark.

Adapted from the supplied G2 momentum-analysis script. The quantity is now
    |g_{ss,nu}(k=Gamma,q)|^2
or, with PLOT_SUM=True, sum_nu |g_{ss,nu}(Gamma,q)|^2.
There is NO electronic-band sum, k average, or extra 1/Nk factor.

Put this script beside electron_phonon_g_band_kq.h5 and
POSCAR_equil_primitive (or POSCAR_primitive). Run:
    python -m pip install numpy h5py matplotlib
    python plot_my_g2_gamma.py
    python plot_my_g2_gamma.py --plot-sum
    python plot_my_g2_gamma.py --interpolate

Band indices are 1-based. BAND=5 matches the reference's spinless lowest
conduction band only if your Hamiltonian has the same band counting.
Use --band YOUR_CONDUCTION_BAND otherwise. No identification by orbital label
or automatic treatment of spin degeneracy is attempted.

Input convention retained from your script: g_blocks are logical Julia
(Nk,Nq), exposed to h5py as (Nq,Nk). Only the Gamma column is read. k=Gamma
is located from k-coordinate datasets, NEVER assumed to be the first point.
If coordinates are absent, supply --gamma-k-index with a VERIFIED 1-based
index from the program that wrote your file.

g is assumed to be in eV, including the phonon normalization. Output is meV^2.
No mass/frequency, population, Nk, Nq, or supercell correction is added.
--g-scale multiplies g before squaring, ONLY if you have independently
established a normalization conversion. Its default is 1.

Default: actual q-mesh points on L-Gamma-X-W-K-Gamma, shown as markers.
--interpolate additionally shows periodic trilinear interpolation of |g|^2,
as a visualization. This is NOT Wannier interpolation of electron-phonon
vertices, does not reconstruct polar 1/q behavior, and can mix frequency-
ordered branches across crossings. Invalid corners are NOT filled or
renormalized away. The all-mode sum is undefined if any mode is invalid.
Exact reciprocal-lattice Gamma points are omitted from both plots.

If gaas.ephmat and gaas_ephmat.yml are beside this script, a dashed PERTURBO
reference is overlaid automatically (requires: python -m pip install pyyaml).
Alternatively use --reference-dir PATH or --no-reference.
Source: https://github.com/perturbo-code/perturbo-workshop-2025/tree/main/Hands-on3/GaAs_polar/perturbo/pert-ephmat/References

Phonon modes are sorted by signed frequency at each q if frequency_THz_signed
is present. At degeneracies individual branch contributions depend on the
choice of eigenvectors; compare the multiplet sum or the full six-mode sum.
"""

from __future__ import annotations

import argparse
import csv
from itertools import product, permutations
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from visualization_common import add_calculation_arguments, configure_plot_args

# Hydrogen stores calculation data separately from these source scripts.


# -------------------------- USER SETTINGS --------------------------
PLOT_SUM = False
BAND = 5
LOG_Y = True
INTERPOLATE = False
SHOW_PLOT = False
G_AMPLITUDE_SCALE = 1.0
GAMMA_K_INDEX = None              # Optional VERIFIED 1-based index.
SAMPLES_PER_SEGMENT = 151
# -------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
LABELS = ['L', r'$\Gamma$', 'X', 'W', 'K', r'$\Gamma$']
# Exactly the PERTURBO route, in conventional cubic coordinates (2*pi/a).
VERTICES = np.array([[-.5,.5,.5], [0,0,0], [-1,0,0],
                     [-1,.5,0], [-.75,.75,0], [0,0,0]])
TICKS = np.r_[0., np.cumsum(np.linalg.norm(np.diff(VERTICES, axis=0), axis=1))]


def triplets(dataset, count):
    """Restore Julia's logical (3,N) layout, preferring HDF5.jl's (N,3)."""
    a = np.asarray(dataset[()], dtype=float)
    if a.shape == (count, 3):
        return a.T
    if a.shape == (3, count):
        return a
    raise ValueError(f'{dataset.name}: expected ({count},3) or (3,{count}), got {a.shape}')


def julia_matrix(dataset, shape, dtype=float):
    a = np.asarray(dataset[()], dtype=dtype).T
    if a.shape != shape:
        raise ValueError(f'{dataset.name}: logical shape {a.shape}, expected {shape}')
    return a


def complex_values(a):
    """Accept native complex data and common HDF5.jl complex compound types."""
    a = np.asarray(a)
    if a.dtype.names:
        for real, imag in [('r','i'), ('re','im'), ('real','imag')]:
            if real in a.dtype.names and imag in a.dtype.names:
                return a[real].astype(float) + 1j*a[imag].astype(float)
        raise ValueError(f'Unknown complex compound fields: {a.dtype.names}')
    return np.asarray(a, dtype=np.complex128)


def gamma_index(f, nk, override):
    paths = ['grids/k_reduced_primitive_reciprocal', 'grids/k_cartesian_inv_angstrom']
    available = next((p for p in paths if p in f), None)
    if available:
        k = triplets(f[available], nk)
        delta = k - np.rint(k) if 'reduced' in available else k
        hits = np.flatnonzero(np.max(np.abs(delta), axis=0) < 1e-9)
        if override is None:
            if len(hits) != 1:
                raise ValueError(f'Expected one Gamma k point in {available}; found {len(hits)}. '
                                 'If appropriate, supply --gamma-k-index (1-based).')
            return int(hits[0]), available
        if override - 1 not in hits:
            raise ValueError('--gamma-k-index is not Gamma according to the stored coordinates.')
    elif override is None:
        raise ValueError('No stored k coordinates found. Supply a VERIFIED --gamma-k-index '
                         '(1-based). The first k point is not assumed to be Gamma.')
    if not 1 <= override <= nk:
        raise ValueError(f'--gamma-k-index must lie in 1..{nk}.')
    return override - 1, available or 'explicit user index; coordinates unavailable'


def read_couplings(filename, band, gamma_override, scale, *, initial_k_reduced=None):
    with h5py.File(filename, 'r') as f:
        nk, nq, nm, nb = [int(np.asarray(f[k][()]).item()) for k in
                         ['number_of_k_points', 'number_of_q_points',
                          'number_of_modes', 'number_of_bands']]
        if not 1 <= band <= nb:
            raise ValueError(f'Band must lie in 1..{nb}.')
        if initial_k_reduced is None:
            ik, gamma_source = gamma_index(f, nk, gamma_override)
        else:
            target = np.asarray(initial_k_reduced,dtype=float)
            if target.shape != (3,):
                raise ValueError("Initial k must have three primitive reduced coordinates.")
            k = triplets(f['grids/k_reduced_primitive_reciprocal'],nk)
            delta = k-target[:,None]
            hits = np.flatnonzero(np.max(np.abs(delta-np.rint(delta)),axis=0)<1e-9)
            if len(hits) != 1:
                raise ValueError('Requested initial k is absent or nonunique in '+str(filename))
            ik = int(hits[0])
            if gamma_override is not None and gamma_override != ik+1:
                raise ValueError('Explicit k index disagrees with the requested k coordinates.')
            gamma_source = 'grids/k_reduced_primitive_reciprocal; selected initial k' 
        reps = np.asarray(f['repetitions_n1_n2_n3'][()], dtype=int).ravel()
        if reps.shape != (3,) or np.any(reps <= 0):
            raise ValueError('Expected three positive supercell repetitions.')
        qr = triplets(f['grids/q_reduced_primitive_reciprocal'], nq)
        qc = triplets(f['grids/q_cartesian_inv_angstrom'], nq)
        qi = triplets(f['grids/q_integer_indices'], nq)
        # Derive mesh locations from physical coordinates, not index origin.
        scaled = qr * reps[:, None]
        if not np.allclose(scaled, np.rint(scaled), atol=1e-7, rtol=0):
            raise ValueError('Expected an unshifted commensurate q mesh.')
        mesh = np.rint(scaled).astype(int) % reps[:, None]
        if not np.allclose(qi, np.rint(qi), atol=1e-9, rtol=0):
            raise ValueError('Non-integer q_integer_indices.')
        offsets = (np.rint(qi).astype(int) - mesh) % reps[:, None]
        if not np.all(offsets == offsets[:, :1]):
            raise ValueError('q integer indices and reduced coordinates disagree.')
        if nq != int(np.prod(reps)) or len(np.unique(mesh.T, axis=0)) != nq:
            raise ValueError('The commensurate q mesh is incomplete or duplicated.')

        index = f['block_index']
        modes = np.asarray(index['mode'][()], dtype=int).ravel()
        bm = np.asarray(index['band_m' if 'band_m' in index else 'm'][()], dtype=int).ravel()
        bn = np.asarray(index['band_n' if 'band_n' in index else 'n'][()], dtype=int).ravel()
        names = [v.decode() if isinstance(v, bytes) else str(v)
                 for v in np.ravel(index['dataset_name'][()])]
        nblocks = int(np.asarray(index['number_of_blocks'][()]).item())
        if not (len(modes) == len(bm) == len(bn) == len(names) == nblocks):
            raise ValueError('Inconsistent block-index lengths.')
        values = np.full((nm, nq), np.nan)
        seen = set()
        for i in np.flatnonzero((bm == band) & (bn == band)):
            mode = int(modes[i]) - 1
            if not 0 <= mode < nm or mode in seen:
                raise ValueError('Duplicate or invalid mode for the selected band pair.')
            seen.add(mode)
            ds = f['g_blocks'][names[i]]
            if ds.shape != (nq, nk):
                raise ValueError(f'{ds.name}: h5py shape {ds.shape}; expected {(nq,nk)} '
                                 'for a Julia (Nk,Nq) matrix.')
            # Crucial change: read only k=Gamma, rather than averaging all k.
            g = complex_values(ds[:, ik])
            values[mode] = np.abs(g * scale)**2 * 1e6  # eV^2 -> meV^2
        if len(seen) != nm:
            raise ValueError(f'Found {len(seen)} selected mode blocks, expected {nm}.')
        values[~np.isfinite(values)] = np.nan
        #if 'phonons/valid_for_g' in f:
        #    valid = julia_matrix(f['phonons/valid_for_g'], (nm,nq), bool)
        #    values[~valid] = np.nan
        sorted_modes = False
        #if 'phonons/frequency_THz_signed' in f:
        #    freq = julia_matrix(f['phonons/frequency_THz_signed'], (nm,nq))
        #    values[~np.isfinite(freq) | (freq <= 0)] = np.nan
        #    order = np.argsort(freq, axis=0, kind='stable')
        #    values = np.take_along_axis(values, order, axis=0)
        #    sorted_modes = True

        #delete gamma points
        #is_gamma = np.max(np.abs(qr - np.rint(qr)), axis=0) < 1e-9
        #values[:, is_gamma] = np.nan
    return dict(g2=values, qr=qr, qc=qc, mesh=mesh, reps=reps, nk=nk, nq=nq,
                nm=nm, ik=ik, gamma_source=gamma_source, sorted_modes=sorted_modes)


def read_lattice(path):
    """POSCAR lattice only; retain scalar, target-volume and Cartesian scaling."""
    lines = path.read_text().splitlines()
    raw = np.array([[float(x) for x in line.split()[:3]] for line in lines[2:5]])
    scales = np.array([float(x) for x in lines[1].split()])
    if len(scales) == 1 and scales[0] != 0:
        s = scales[0] if scales[0] > 0 else (abs(scales[0])/abs(np.linalg.det(raw)))**(1/3)
        return s * raw
    if len(scales) == 3 and np.all(scales > 0):
        return raw * scales[None, :]
    raise ValueError('Invalid POSCAR scale line.')


def cubic_frame(primitive):
    """Find cubic FCC axes even for signed/permuted primitive FCC vectors."""
    a = (4 * abs(np.linalg.det(primitive)))**(1/3)
    vectors = np.array(list(product(range(-2,3), repeat=3))) @ primitive
    candidates = vectors[np.abs(np.linalg.norm(vectors, axis=1)/a - 1) < 5e-3]
    best, score = None, -np.inf
    for rows in permutations(candidates, 3):
        c = np.asarray(rows)
        if np.linalg.det(c) <= 0 or not np.allclose(c@c.T/a**2, np.eye(3), atol=5e-3):
            continue
        # Choose cubic axes closest to the supplied Cartesian x,y,z frame.
        if np.trace(c/a) > score:
            best, score = c.copy(), np.trace(c/a)
    if best is None:
        raise ValueError('Cannot reconstruct a cubic FCC cell from this primitive POSCAR.')
    return 2*np.pi*np.linalg.inv(best)


def native_path_points(qr, bprim, bcubic):
    """Locate actual mesh points on each segment, including reciprocal images."""
    transform = np.linalg.solve(bcubic, bprim)
    reduced_vertices = np.linalg.solve(transform, VERTICES.T).T
    wrapped = qr.T % 1.0
    xs, ids, segs = [], [], []
    for seg, (a,b) in enumerate(zip(VERTICES[:-1], VERTICES[1:])):
        rv = reduced_vertices[seg:seg+2]
        lower = np.floor(rv.min(axis=0)).astype(int)-1
        upper = np.ceil(rv.max(axis=0)).astype(int)+1
        hits = []
        direction = b-a
        length2 = direction@direction
        for shift in product(*(range(lo,hi+1) for lo,hi in zip(lower,upper))):
            p = (wrapped + shift) @ transform.T
            t = (p-a)@direction/length2
            residual = p-a-t[:,None]*direction
            match = np.flatnonzero((t >= -1e-8) & (t <= 1+1e-8)
                                   & (np.linalg.norm(residual,axis=1) < 1e-7))
            hits.extend((float(np.clip(t[i],0,1)), int(i)) for i in match)
        unique = {(round(t,10),iq):(t,iq) for t,iq in hits}
        for t,iq in sorted(unique.values()):
            xs.append(TICKS[seg] + t*np.sqrt(length2)); ids.append(iq); segs.append(seg)
    if not ids:
        raise ValueError('No mesh points lie on the requested high-symmetry path.')
    return np.asarray(xs), np.asarray(ids), np.asarray(segs)


def interpolate_grid(grid, reduced):
    """Strict periodic trilinear interpolation; invalid nonzero-weight corners stay invalid."""
    dims = np.array(grid.shape[1:])
    scaled = (np.asarray(reduced) % 1.) * dims
    scaled = np.where(np.abs(scaled-np.rint(scaled)) < 1e-9, np.rint(scaled), scaled)
    base = np.floor(scaled).astype(int)
    fraction = scaled-base
    total = np.zeros(grid.shape[0])
    valid = np.ones(grid.shape[0], dtype=bool)
    for corner in product((0,1), repeat=3):
        weight = np.prod(np.where(corner, fraction, 1-fraction))
        if weight <= 1e-14:
            continue
        cell = (base+corner) % dims
        v = grid[:, cell[0],cell[1],cell[2]]
        valid &= np.isfinite(v)
        total += weight*np.where(np.isfinite(v),v,0.)
    total[~valid] = np.nan
    return total


def interpolated_path(data, bprim, bcubic, samples):
    grid = np.full((data['nm'], *data['reps']), np.nan)
    i,j,k = data['mesh']
    grid[:,i,j,k] = data['g2']
    xs, ys = [], []
    for seg,(a,b) in enumerate(zip(VERTICES[:-1],VERTICES[1:])):
        for t in np.linspace(0,1,samples):
            reduced = np.linalg.solve(bprim, bcubic@((1-t)*a+t*b))
            xs.append((1-t)*TICKS[seg]+t*TICKS[seg+1])
            ys.append(interpolate_grid(grid,reduced))
    return np.asarray(xs), np.asarray(ys).T


def load_reference(directory):
    import yaml
    with (directory/'gaas_ephmat.yml').open() as stream:
        meta = yaml.safe_load(stream)
    e = meta['ephmat']
    p = meta['input parameters']['after conversion']
    if p['band_min'] != 5 or p['band_max'] != 5:
        raise ValueError('Expected the band-5 PERTURBO reference.')
    if not np.allclose(e['k-point coordinates'], [[0,0,0]]):
        raise ValueError('Reference initial k is not Gamma.')
    if e['e-ph matrix elements units'] != 'meV' or e['q-point coordinate units'] != 'crystal':
        raise ValueError('Unexpected reference units.')
    q = np.asarray(e['q-point coordinates'])
    qconv = q @ np.asarray(meta['basic data']['reciprocal lattice vectors'])
    x = np.asarray(e['q-path coordinates'])
    for corner, expected_x in zip(VERTICES, TICKS):
        if not np.any((np.linalg.norm(qconv-corner,axis=1) < 1e-7) & (np.abs(x-expected_x) < 1e-6)):
            raise ValueError('Reference path differs from the specified L-Gamma-X-W-K-Gamma path.')
    tab = np.loadtxt(directory/'gaas.ephmat', ndmin=2)
    nm = int(e['number of phonon modes'])
    if tab.shape != (len(q)*nm,8) or np.any(tab[:,0] != 1):
        raise ValueError('Unexpected PERTURBO reference table shape or k index.')
    iq, mode = tab[:,2].astype(int)-1, tab[:,4].astype(int)-1
    if (np.any(iq < 0) or np.any(iq >= len(q)) or np.any(mode < 0) or np.any(mode >= nm)
            or len(np.unique(iq*nm+mode)) != len(q)*nm):
        raise ValueError('Invalid reference q/mode indices.')
    if not np.allclose(tab[:,3], x[iq], atol=6e-6, rtol=0):
        raise ValueError('Reference ASCII and YAML path distances differ.')
    g2 = np.full((nm,len(q)),np.nan)
    g2[mode,iq] = tab[:,7]**2
    g2[:,np.linalg.norm(qconv,axis=1)<1e-9] = np.nan
    return x,g2


def selected_curves(values, plot_sum):
    # np.sum deliberately propagates missing modes instead of using nansum.
    return values.sum(axis=0,keepdims=True) if plot_sum else values.copy()


def write_csv(path, header, rows):
    with path.open('w',newline='') as stream:
        writer = csv.writer(stream); writer.writerow(header); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--g-file', type=Path, default=None)
    parser.add_argument('--primitive-poscar', type=Path)
    parser.add_argument('--band', type=int, default=BAND, help='Single band, 1-based.')
    parser.add_argument('--gamma-k-index', type=int, default=GAMMA_K_INDEX)
    parser.add_argument('--g-scale', type=float, default=G_AMPLITUDE_SCALE)
    parser.add_argument('--plot-sum', action=argparse.BooleanOptionalAction, default=PLOT_SUM)
    parser.add_argument('--log-y', action=argparse.BooleanOptionalAction, default=LOG_Y)
    parser.add_argument('--interpolate', action=argparse.BooleanOptionalAction, default=INTERPOLATE)
    parser.add_argument('--samples-per-segment', type=int, default=SAMPLES_PER_SEGMENT)
    parser.add_argument('--show', action=argparse.BooleanOptionalAction, default=SHOW_PLOT)
    parser.add_argument('--reference-dir', type=Path)
    parser.add_argument('--no-reference', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=None)
    add_calculation_arguments(parser)
    args = parser.parse_args()
    calc = configure_plot_args(args, "g2_gamma_comparison")
    here = calc.source_dir
    if not np.isfinite(args.g_scale) or args.g_scale <= 0:
        parser.error('--g-scale must be finite and positive.')
    if args.samples_per_segment < 2:
        parser.error('--samples-per-segment must be at least 2.')
    poscar = args.primitive_poscar
    if poscar is None:
        poscar = next((p for p in [here/'POSCAR_equil_primitive',
                                  here/'POSCAR_primitive'] if p.is_file()),None)
    if poscar is None:
        parser.error('Cannot find primitive POSCAR; supply --primitive-poscar FILE.')
    data = read_couplings(args.g_file,args.band,args.gamma_k_index,args.g_scale)
    primitive = read_lattice(poscar)
    bprim = 2*np.pi*np.linalg.inv(primitive)
    error = float(np.max(np.abs(bprim@data['qr']-data['qc'])))
    if error > 1e-7:
        raise ValueError(f'POSCAR and stored q coordinates disagree by {error:.3e} inverse Angstrom.')
    bcubic = cubic_frame(primitive)
    x,iq,segments = native_path_points(data['qr'],bprim,bcubic)
    native = data['g2'][:,iq]
    finite_count = np.isfinite(selected_curves(native,args.plot_sum)).sum()
    if not finite_count:
        raise ValueError('No valid selected couplings on the path. Check band, mode validity, and q mesh.')
    ref = None
    directory = args.reference_dir or here
    if not args.no_reference:
        present = [(directory/name).is_file() for name in ['gaas.ephmat','gaas_ephmat.yml']]
        if all(present):
            ref = load_reference(directory)
            if ref[1].shape[0] != data['nm']:
                raise ValueError('Your file and the reference have different numbers of phonon modes.')
        elif args.reference_dir or any(present):
            raise FileNotFoundError('The reference directory must contain BOTH gaas.ephmat and gaas_ephmat.yml.')
    fig, ax = plt.subplots(figsize=(10,6),layout='constrained')
    def draw(xx, yy, style, source):
        yy = selected_curves(yy,args.plot_sum)
        if args.log_y:
            yy[yy <= 0] = np.nan
        for mode,curve in enumerate(yy):
            name = 'all modes' if args.plot_sum else rf'$\nu={mode+1}$'
            ax.plot(xx,curve,style,color=f'C{mode%10}',markersize=5,linewidth=1.25, linestyle = "-",
                    label=f'{source}: {name}')
    draw(x,native,'o','Your data')
    xi = yi = None
    if args.interpolate:
        xi,yi = interpolated_path(data,bprim,bcubic,args.samples_per_segment)
        draw(xi,yi,'-','Trilinear')
    if ref is not None:
        draw(*ref,'--','PERTURBO')
    for tick in TICKS:
        ax.axvline(tick,color='.8',lw=.7,zorder=0)
    ax.set_xticks(TICKS,LABELS); ax.set_xlim(TICKS[0],TICKS[-1])
    ax.set_xlabel(r'Phonon wavevector $\mathbf{q}$ along L-$\Gamma$-X-W-K-$\Gamma$')
    expr = r'\sum_\nu |g_{ss\nu}(\Gamma,\mathbf{q})|^2' if args.plot_sum else r'|g_{ss\nu}(\Gamma,\mathbf{q})|^2'
    ax.set_ylabel('$'+expr+r'$ (meV$^2$)')
    ax.set_title(rf'Initial $\mathbf{{k}}=\Gamma$; your band $s={args.band}$'
                 + ('; PERTURBO band 5' if ref is not None else ''))
    if args.log_y:
        ax.set_yscale('log')
    ax.grid(axis='y',alpha=.2)
    ax.legend(fontsize=8,ncol=1 if args.plot_sum else (3 if args.interpolate else 2),loc='best')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    suffix = 'sum' if args.plot_sum else 'branches'
    image_path = args.output_dir/f'g2_gamma_{suffix}.svg'
    with matplotlib.rc_context({"svg.fonttype":"path"}):
        fig.savefig(image_path,format="svg",bbox_inches="tight",pad_inches=0.25)
    headers = [f'g2_mode_{i+1}_meV2' for i in range(data['nm'])]+['sum_all_modes_meV2']
    all_values = np.vstack([data['g2'],data['g2'].sum(axis=0)]).T
    write_csv(args.output_dir/'g2_discrete_q.csv',
              ['q_index_1based','q_red1','q_red2','q_red3']+headers,
              np.column_stack([np.arange(data['nq'])+1,data['qr'].T,all_values]))
    write_csv(args.output_dir/'g2_native_path.csv',
              ['path_distance_2pi_over_a','segment_1based','q_index_1based']+headers,
              np.column_stack([x,segments+1,iq+1,all_values[iq]]))
    if xi is not None:
        write_csv(args.output_dir/'g2_interpolated_path.csv',
                  ['path_distance_2pi_over_a']+headers,
                  np.column_stack([xi,yi.T,yi.sum(axis=0)]))
    summary = (f'Input: {args.g_file}\nBand pair (1-based): {args.band},{args.band}\n'
               f'Initial Gamma k index (1-based): {data["ik"]+1}\n'
               f'Gamma determined from: {data["gamma_source"]}\n'
               f'Nk={data["nk"]}, Nq={data["nq"]}, Nmodes={data["nm"]}\n'
               f'Frequency-sorted branches: {data["sorted_modes"]}\n'
               f'g amplitude multiplier: {args.g_scale}\n'
               'Assumed input units: eV. Output: meV^2 = 1e6*abs(g*scale)^2.\n'
               'No k averaging, band summation, or added supercell normalization.\n'
               'Invalid modes and exact Gamma omitted; all-mode sums require every mode.\n'
               f'Path points plotted: {len(x)} (shared segment endpoints may repeat).\n'
               f'Optional trilinear display: {args.interpolate}; reference overlay: {ref is not None}\n'
               f'POSCAR/q-grid residual: {error:.3e} inverse Angstrom\n'
               f'Saved plot: {image_path}\n')
    (args.output_dir/f'analysis_summary_{suffix}.txt').write_text(summary)
    print(summary)
    if not data['sorted_modes']:
        print('NOTE: frequency metadata absent; original mode numbering retained.')
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == '__main__':
    main()
