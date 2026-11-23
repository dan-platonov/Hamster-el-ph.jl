#!/usr/bin/env python3
"""Plot CsPbBr3 conduction-pair couplings on Gamma-X-M-Gamma-R-X.

Defaults: ML 4x4x4, 425/525/625 K; initial electron k=R; first conduction
pair C={21,22} in 1-based spin-resolved model numbering. Twenty bands are
occupied. --band selects the first of two consecutive bands in the pair.

Quantity: 1e6 * Nc/2 * sum_{n,m in C} sum_s |g_stored[m,n,s](k+q,k)|^2
in meV^2. Average over the two initial states, sum over both final states
and all 15 phonon branches. The pair sum is independent of arbitrary
unitary rotations within the selected initial and final subspaces. At R
and R+X=M, bands 21 and 22 are degenerate in all three calculations.

The Nc factor removes the stored 1/sqrt(Nc) amplitude normalization, as
in the GaAs comparison. There is no k average, Bose factor, interpolation,
fit, or sum over higher conduction bands. Undefined phonon modes propagate
to NaN; exact phonon Gamma is omitted because its three rigid translations
have no finite phonon-normalized g in these data.

Only existing q-mesh points are used. Lines connect consecutive points across
segment boundaries but do not bridge undefined Gamma values. Primitive cubic
reciprocal coordinates: Gamma=(0,0,0), X=(1/2,0,0), M=(1/2,1/2,0),
R=(1/2,1/2,1/2). Presentation controls are near the top of this script.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np
from visualization_common import DEFAULT_DATA_ROOT

# GaAs presentation style, with larger ticks and more numbered log-axis ticks.
FONT_SIZE = 24
TICK_SIZE = 34
LABEL_SIZE = 26
TITLE_SIZE = 26
LEGEND_SIZE = 22
FIGURE_SIZE = (14, 8.5)
LINE_WIDTH = 2.5
MARKER_SIZE = 6
MAJOR_TICK_LENGTH = 10
MINOR_TICK_LENGTH = 5
Y_LOG_TICK_MULTIPLIERS = (1, 1.5, 2, 3, 5, 7)
Y_AXIS_EXPONENT = 3  # Shared multiplier above the y-axis.
TEMPERATURE_COLORS = {425: '#6baed6', 525: '#3182bd', 625: '#08519c'}

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__import__("os").environ.get("CSPBBR3_PROJECT_ROOT", REPOSITORY_ROOT/"examples/CsPbBr3"))
DEFAULT_CURVES = [(DEFAULT_DATA_ROOT/f'ML_corrected/4x4x4/{t}K', f'{t} K')
                  for t in (425, 525, 625)]
NODES = np.array([[0,0,0], [.5,0,0], [.5,.5,0], [0,0,0], [.5,.5,.5], [.5,0,0]])
NODE_LABELS = [r'$\Gamma$', 'X', 'M', r'$\Gamma$', 'R', 'X']


def text(value):
    return value.decode() if isinstance(value, bytes) else str(value)


def coupling_file(directory):
    path = Path(directory).expanduser()
    candidates = [path] if path.is_absolute() else [path, PROJECT_ROOT/path, DEFAULT_DATA_ROOT/path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
        for name in ('electron_phonon_g_band_kq.h5', 'data/electron_phonon_g_band_kq.h5'):
            if (candidate/name).is_file():
                return (candidate/name).resolve()
    raise FileNotFoundError(f'No band-basis coupling file found in {directory}')


def normalization(filename, repetitions):
    """Establish normalization from the recorded source chain, never from Nk."""
    pending, seen = [filename], set()
    while pending:
        source = pending.pop(0).resolve()
        if source in seen or not source.is_file():
            continue
        seen.add(source)
        with h5py.File(source, 'r') as f:
            if not np.array_equal(f['repetitions_n1_n2_n3'][()], repetitions):
                raise ValueError(f'Cell size mismatch in normalization source {source}')
            for key in ('conventions/g_normalization', 'conventions/d_normalization', 'normalization'):
                if key not in f:
                    continue
                value = text(f[key][()])
                if value == '1/sqrt(Nc)' or value == 'finite-supercell':
                    return int(np.prod(repetitions)), str(source), key
                if value == 'primitive-cell':
                    return 1, str(source), key
                raise ValueError(f'Unknown normalization {value!r} at {source}:{key}')
            for key in ('sources/orbital_basis_g', 'sources/d_couplings'):
                if key in f:
                    path = Path(text(f[key][()]))
                    pending.append(path if path.is_absolute() else source.parent/path)
    raise ValueError(f'Cannot establish stored normalization: {filename}')


def path_points(qr, qc):
    """Select native points on each segment in the actual reciprocal metric."""
    reciprocal = np.linalg.lstsq(qr, qc, rcond=None)[0]
    if np.linalg.matrix_rank(qr) != 3 or not np.allclose(qr@reciprocal, qc, atol=1e-8, rtol=0):
        raise ValueError('Stored Cartesian and reduced q grids are inconsistent.')
    metric = reciprocal@reciprocal.T
    if not np.allclose(metric, np.eye(3)*np.trace(metric)/3, atol=1e-8, rtol=1e-6):
        raise ValueError('This path requires the primitive simple-cubic reciprocal basis.')
    ticks = np.r_[0, np.cumsum(np.linalg.norm(np.diff(NODES, axis=0)@reciprocal, axis=1))]
    coords, indices = [], []
    # Periodic images let equivalent negative grid coordinates lie on this path.
    shifts = np.array(np.meshgrid([-1,0,1],[-1,0,1],[-1,0,1])).reshape(3,-1).T
    for segment, (start, end) in enumerate(zip(NODES[:-1], NODES[1:])):
        direction = (end-start)@reciprocal
        points = []
        for iq, q in enumerate(qr):
            for shift in shifts:
                delta = (q+shift-start)@reciprocal
                fraction = float(delta@direction/(direction@direction))
                if -1e-9 <= fraction <= 1+1e-9 and np.linalg.norm(delta-fraction*direction) < 1e-8:
                    x = ticks[segment]+np.clip(fraction,0,1)*(ticks[segment+1]-ticks[segment])
                    points.append((float(x), iq))
                    break
        points.sort()
        if not points or not np.isclose(points[0][0], ticks[segment]) or not np.isclose(points[-1][0], ticks[segment+1]):
            raise ValueError(f'The q mesh does not contain both endpoints of segment {segment+1}.')
        for x, iq in points:
            if coords and np.isclose(x, coords[-1], atol=1e-9, rtol=0):
                continue
            coords.append(x)
            indices.append(iq)
    return np.asarray(coords), np.asarray(indices), ticks


def read_curve(directory, label, band, initial_k):
    filename = coupling_file(directory)
    with h5py.File(filename, 'r') as f:
        if text(f['units'][()]).lower() != 'ev':
            raise ValueError(f'Expected eV couplings: {filename}')
        if text(f['g_matrix_axis_1'][()]) != 'k index' or text(f['g_matrix_axis_2'][()]) != 'q index':
            raise ValueError('Unrecognized Julia g-matrix axis convention.')
        kr = f['grids/k_reduced_primitive_reciprocal'][()]
        qr = f['grids/q_reduced_primitive_reciprocal'][()]
        qc = f['grids/q_cartesian_inv_angstrom'][()]
        reps = f['repetitions_n1_n2_n3'][()]
        nk, nq = len(kr), len(qr)
        nm = int(f['number_of_modes'][()])
        nb = int(f['number_of_bands'][()])
        if not 1 <= band < nb:
            raise ValueError(f'First band of the pair must lie in 1..{nb-1}')
        delta = kr-np.asarray(initial_k)
        matches = np.flatnonzero(np.max(np.abs(delta-np.rint(delta)),axis=1)<1e-9)
        if len(matches) != 1:
            raise ValueError(f'Initial k {initial_k} is not unique in the stored grid.')
        ik = int(matches[0])
        pair = (band, band+1)
        band_m = f['block_index/band_m'][()]
        band_n = f['block_index/band_n'][()]
        all_modes = f['block_index/mode'][()]
        selected = np.flatnonzero(np.isin(band_m,pair) & np.isin(band_n,pair))
        modes = all_modes[selected]
        expected = {(m,n,mode) for m in pair for n in pair for mode in range(1,nm+1)}
        actual = {(int(band_m[b]),int(band_n[b]),int(all_modes[b])) for b in selected}
        if len(selected) != 4*nm or actual != expected:
            raise ValueError('Missing or duplicate branches/band transitions for the conduction pair.')
        names = f['block_index/dataset_name'][()]
        g2 = np.zeros((nm,nq))
        for block, mode in zip(selected, modes):
            ds = f['g_blocks/'+text(names[block])]
            # h5py reverses Julia's matrix dimensions: stored here as (q,k).
            if ds.shape != (nq,nk):
                raise ValueError(f'Unexpected block shape {ds.shape}')
            # Average over two initial states, sum over two final states.
            # All four complex matrix elements are squared separately.
            g2[mode-1] += 0.5*np.abs(ds[:,ik])**2
        temperature = float(f['temperature_K'][()])
        energy = f['bands/energy_eV'][()]
        valid = f['phonons/valid_for_g'][()].astype(bool)
        if valid.shape != (nq,nm):
            raise ValueError('Unexpected validity-mask axes.')
        # Verify missing data are not accidentally represented as valid zeros.
        g2[~valid.T] = np.nan
    factor, norm_source, norm_key = normalization(filename, reps)
    total = 1e6*factor*np.sum(g2,axis=0)
    gamma = np.max(np.abs(qr-np.rint(qr)),axis=1)<1e-9
    total[gamma] = np.nan
    x, indices, ticks = path_points(qr,qc)
    y = total[indices]
    if not np.any(np.isfinite(y) & (y>0)):
        raise ValueError('No positive finite values on the requested path.')
    info = dict(label=label, input=str(filename), temperature_K=temperature,
                band_pair=list(pair), initial_state_average_factor=0.5,
                final_states_summed=list(pair), number_of_phonon_branches=nm,
                initial_k_index_1based=ik+1, initial_k_reduced=kr[ik].tolist(),
                pair_mean_energy_minimum_k_reduced=kr[np.argmin(np.mean(energy[:,band-1:band+1],axis=1))].tolist(),
                initial_pair_energy_splitting_eV=float(energy[ik,band]-energy[ik,band-1]),
                repetitions=reps.tolist(), normalization_source=norm_source,
                normalization_key=norm_key, squared_coupling_multiplier=factor,
                q_indices_1based=(indices+1).tolist(),
                positive_finite_path_values=int(np.count_nonzero(np.isfinite(y)&(y>0))))
    return dict(x=x,y=y,ticks=ticks,qr=qr[indices],info=info)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--curve', nargs=2, action='append', metavar=('DIRECTORY','LABEL'),
                        help='Replace defaults with repeated source directories and legend labels.')
    parser.add_argument('--band', type=int, default=21, help='First of two consecutive spin-resolved bands (default 21 selects the conduction pair 21,22).')
    parser.add_argument('--k-reduced', nargs=3, type=float, default=[.5,.5,.5], metavar=('K1','K2','K3'))
    parser.add_argument('--output-dir', type=Path, default=REPOSITORY_ROOT/'results/CsPbBr3/coupling_path_in_BZ')
    parser.add_argument('--linear-y', action='store_true')
    args = parser.parse_args()
    if not np.isfinite(args.k_reduced).all():
        parser.error('Initial k must be finite.')
    curves = [read_curve(path,label,args.band,args.k_reduced) for path,label in (args.curve or DEFAULT_CURVES)]
    temperatures = sorted(set(curve['info']['temperature_K'] for curve in curves))
    colors = {t:TEMPERATURE_COLORS.get(t,matplotlib.colormaps['Blues'](.5+.4*i/max(1,len(temperatures)-1)))
              for i,t in enumerate(temperatures)}
    plt.rcParams.update({'font.size':FONT_SIZE,'axes.labelsize':LABEL_SIZE,
                         'axes.titlesize':TITLE_SIZE,'xtick.labelsize':TICK_SIZE,
                         'ytick.labelsize':TICK_SIZE,'legend.fontsize':LEGEND_SIZE})
    fig, ax = plt.subplots(figsize=FIGURE_SIZE,layout='constrained')
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True,exist_ok=True)
    for index,curve in enumerate(curves,1):
        if not np.allclose(curve['ticks'],curves[0]['ticks'],rtol=1e-7,atol=1e-9):
            raise ValueError('Curves have different reciprocal lattice metrics.')
        y = curve['y'].copy()
        y[~np.isfinite(y)] = np.nan
        if not args.linear_y:
            y[y<=0] = np.nan
        color = colors[curve['info']['temperature_K']]
        ax.plot(curve['x'],y,color=color,linestyle='-',linewidth=LINE_WIDTH,
                marker='o',markersize=MARKER_SIZE,label=curve['info']['label'])
        slug = re.sub(r'[^\w.-]+','_',curve['info']['label']).strip('_.') or 'curve'
        csv = output/f'g2_path_sum_curve_{index:02d}_{slug}.csv'
        np.savetxt(csv,np.column_stack([curve['x'],y,curve['qr']]),delimiter=',',comments='',fmt='%.17g',
                   header='path_distance_inv_angstrom,conduction_pair_mean_sum_abs_g_squared_meV2,q1,q2,q3')
        curve['info'].update(exported_csv=str(csv),plot_color=matplotlib.colors.to_hex(color))
    ticks = curves[0]['ticks']
    for position in ticks:
        ax.axvline(position,color='0.8',linewidth=.7,zorder=0)
    ax.set_xticks(ticks,NODE_LABELS)
    ax.set_xlim(ticks[0],ticks[-1])
    ax.set_xlabel(r'Phonon wavevector $\mathbf{q}$ along $\Gamma$–X–M–$\Gamma$–R–X')
    target = np.asarray(args.k_reduced)
    is_r = np.max(np.abs(target-.5-np.rint(target-.5)))<1e-9
    is_gamma = np.max(np.abs(target-np.rint(target)))<1e-9
    k_label = 'R' if is_r else (r'\Gamma' if is_gamma else '('+', '.join(f'{v:g}' for v in target)+')')
    ax.set_ylabel(r'$\frac{1}{2}\sum_{n,m\in\mathcal{C}}\sum_s |g_{mn,s}('+k_label+r',\mathbf{q})|^2$ (meV$^2$)')
    ax.set_title(rf'CsPbBr$_3$: initial $\mathbf{{k}}={k_label}$; conduction pair $\mathcal{{C}}=\{{{args.band},{args.band+1}\}}$')
    if not args.linear_y:
        ax.set_yscale('log')
        # Number intermediate logarithmic ticks as well as powers of ten.
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=Y_LOG_TICK_MULTIPLIERS, numticks=100))
        ax.yaxis.set_major_formatter(FuncFormatter(
            lambda value, position: f'{value / 10.0**Y_AXIS_EXPONENT:g}'))
        ax.text(0, 1.015, rf'$\times 10^{{{Y_AXIS_EXPONENT}}}$',
                transform=ax.transAxes, ha='left', va='bottom',
                fontsize=TICK_SIZE, clip_on=False)
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(4, 6, 8, 9), numticks=100))
        ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis='both', which='major', labelsize=TICK_SIZE,
                   length=MAJOR_TICK_LENGTH, width=1.8, pad=8)
    ax.tick_params(axis='both', which='minor', length=MINOR_TICK_LENGTH, width=1.2)
    ax.yaxis.get_offset_text().set_fontsize(TICK_SIZE)
    ax.grid(axis='y',alpha=.2)
    ax.legend(fontsize=LEGEND_SIZE,loc='best')
    image = output/'g2_path_sum_temperature_comparison.svg'
    with matplotlib.rc_context({'svg.fonttype':'path'}):
        fig.savefig(image,format='svg',bbox_inches='tight',pad_inches=0.25)
    plt.close(fig)
    report = dict(quantity='(1/2) sum_n_in_C sum_m_in_C sum_s abs(g_mn,s(k_initial,q))^2',units='meV^2',
                  path='Gamma-X-M-Gamma-R-X',path_nodes_reduced=NODES.tolist(),path_ticks=ticks.tolist(),
                  initial_k_reduced=args.k_reduced, band_pair=[args.band,args.band+1],
                  initial_state_average_factor=0.5, final_state_operation='sum over both members',
                  normalization='primitive-cell',
                  exact_phonon_gamma_omitted=True,logarithmic_y=not args.linear_y,
                  y_axis_shared_exponent=None if args.linear_y else Y_AXIS_EXPONENT,
                  interpolation='none; connected native mesh samples, undefined Gamma values retained',
                  font_sizes=dict(base=FONT_SIZE,ticks=TICK_SIZE,labels=LABEL_SIZE,title=TITLE_SIZE,legend=LEGEND_SIZE),
                  curves=[curve['info'] for curve in curves],plot=str(image))
    (output/'g2_path_sum_temperature_comparison_sources.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Saved',image)
    for curve in curves:
        print(curve['info']['label'],curve['info']['positive_finite_path_values'],'finite path samples;',
              'Nc multiplier',curve['info']['squared_coupling_multiplier'])


if __name__ == '__main__':
    main()
