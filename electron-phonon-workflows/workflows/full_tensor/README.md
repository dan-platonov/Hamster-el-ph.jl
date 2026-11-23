# Full-tensor workflow: covariance, correlations, d and g

This is the reusable implementation collected from the CsPbBr3 workflow. It
also accepts GaAs XDATCAR inputs. Source code stays here; model/structure inputs
and large results are passed as paths. The earlier serial GaAs implementation
is retained in [reference_original_implementation/gaas_4x4x4](../../reference_original_implementation/gaas_4x4x4/README.md).

This workflow **accumulates the Hamiltonian over the selected MD snapshots**.
Its bands are obtained from the ensemble mean `mean_H_eV`, not from a single
Hamiltonian evaluated at the mean structure. The distinction from the present
[large-cell implementation](../large_cell/README.md) matters physically.

Installation and complete material examples are in [RUNNING](../../docs/RUNNING.md).
Equations and conventions are in [METHODS](../../docs/METHODS.md).

## Inputs, directories and selection

Run commands from the repository root:

```bash
CALC=/path/to/material/ML_corrected/4x4x4/425K
DATA=/path/to/large-results/material/ML_corrected/4x4x4/425K
./workflows/full_tensor/run.sh STAGE --calc-dir "$CALC" --data-dir "$DATA"
```

CALC contains `hconf`, its matching model files (`params.dat`, `rllm.dat`, ML
parameters and any other referenced inputs), and the required structure
information. XDATCAR preparation needs the corresponding supercell structure
and primitive topology. Native LAMMPS preparation can construct those structures
from a suitable first frame and explicit repetition counts. No fitted models
or trajectories are bundled with the repository.

DATA defaults to `CALC/data` when omitted. Covariance preparation saves generated
structures and covariance in DATA and creates references in CALC. It refuses
to replace ordinary input files with symlinks. The source directory is not an
output directory. Optional `EPH_MATERIAL_ROOT`/`EPH_STORAGE_ROOT` control mirrored
defaults; explicit `--calc-dir` and `--data-dir` avoid depending on that layout.

Use the same trajectory, atom order, geometry and intended snapshot selection
for covariance and Hamiltonian sampling. `--start`/`--final` use one-based frame
ordinals, not native LAMMPS timesteps. The normal choice is `--start 500 --final
all` with no skipped frames. Preparation uses every selected frame; a nonzero
Hamiltonian `--skip-snapshots` changes the statistics and needs a deliberate
covariance choice.

## Stage-by-stage calculation

1. **`covariance` — `compute_equilibrium_covariance.jl`.** Streams the fixed-cell
   trajectory, computes periodic mean positions, a primitive-periodic reference
   and the centered displacement covariance with denominator N. Outputs include
   `displacement_covariance.h5`, `POSCAR_equlibr`, `POSCAR_equil_primitive`, and
   the required supercell/primitive references. Native MACE input additionally
   produces `SC_XDATCAR` for HAMSTER. Periodic displacements must represent a
   stable assignment of atoms to primitive sites.
2. **`accumulate` — `accumulate_uH_parallel.jl` plus its core modules.** Computes
   snapshot Hamiltonians with HAMSTER in chunks. Accumulates sums of u, H and
   uH with explicit primitive/cell/orbital labels. Saves means and restart state
   in `sum_uH_primitive_labels.h5`, and writes `uH_convergence.txt`. The means
   refer to the Hamiltonian snapshots actually processed. Chunk HAMSTER files
   are working data and are removed after their contributions are accumulated.
3. **`d` — `compute_d_from_accumulated_streaming.jl`.** Forms the centered
   displacement–Hamiltonian correlation and applies the covariance pseudoinverse
   in electronic-column slabs. Translation averaging yields
   `electron_lattice_D.h5`; Fourier conversion yields `electron_lattice_d_kq.h5`.
   Default settings are relative covariance cutoff `1e-10`, absolute cutoff zero,
   and 512 electronic entries per slab. The unaveraged raw D is not saved by
   default. These settings are constants near the top of the stage script.
4. **`phonons` — `compute_phonons_from_covariance.jl`.** Translation-averages and
   Fourier-transforms the covariance, uses `K(q)=k_B T C(q)^+`, mass weights it
   and diagonalizes the dynamical matrix. Writes frequencies, polarization
   vectors, grids and validity information to `phonon_modes_from_covariance.h5`.
   At Gamma, rigid translations are projected out of C before inversion, with
   acoustic-sum-rule handling afterwards. The launcher passes the D-regression
   covariance threshold as an absolute phonon cutoff when available, unless
   `--pinv-atol` is supplied explicitly. Supply the actual `--temperature` and,
   for a different material, `--masses FILE`.
5. **`unitary` — `unitary_transformation_matrix_construction.jl`.** Reads the
   ensemble-mean Hamiltonian from `sum_uH_primitive_labels.h5`, averages over
   equivalent primitive translations, builds H(k), and diagonalizes it. Saves
   electronic energies and orbital-to-band eigenvector matrices in
   `unitary_transformation_matrices.h5`. It reports Hermiticity residuals,
   symmetrizes the periodic representation, and fixes an individual-vector
   phase gauge. A degenerate subspace still has freedom to rotate.
6. **`g-orbital` — `compute_electron_phonon_g.jl`.** Combines d with phonon
   polarizations and oscillator lengths, producing the full all-k, all-q,
   all-branch orbital tensor `electron_phonon_g_kq.h5`. Pass the same mass
   table used by the phonon stage with `--masses FILE` when needed.
7. **`g-band` — `transform_electron_phonon_g_to_band_basis.jl`.** Uses U(k) and
   U(k+q) to transform every orbital coupling into band indices and writes
   `electron_phonon_g_band_kq.h5`. This is another full tensor and can be large.

The essential relationships are

\[
 B=\langle uH\rangle-\bar u\bar H,\qquad D=C^+B,
\]
\[
 \bar h_{\mu\nu}(\Delta)=\frac{1}{N_c}\sum_R
 \langle H_{\mu R,\nu(R+\Delta)}\rangle_t,\qquad
 \bar H_{\mu\nu}(k)=\sum_\Delta e^{+ik\cdot\Delta}\bar h_{\mu\nu}(\Delta),
\]
\[
 g^{\mathrm{orb}}_{\mu\nu,s}(k,q)=\sum_{\alpha c}
 \sqrt{\frac{\hbar}{2M_\alpha\omega_s(q)}}\,
 e_{\alpha c,s}(q)d^{\alpha c}_{\mu\nu}(k,q),\qquad
 g_s(k,q)=U^\dagger(k+q)g^{\mathrm{orb}}_s(k,q)U(k).
\]

D/d have units eV/Angstrom and g has units eV. Stored d and g contain
`1/sqrt(Nc)`; no additional cell normalization belongs in the g conversion.
Imaginary or near-zero-frequency modes are marked invalid and their g values
are NaN. In particular the three exact-Gamma acoustic modes are not normalized
oscillators; optical Gamma entries can still be valid. These effective phonons
are inferred from the finite-temperature covariance, not from a DFPT run.

## Commands and parallel execution

The MPI/serial launch examples below are alternatives: run one accumulation
command, then the single-process stages. `jrun` is optional and specific to the
hydrogen installation; the serial and `bin/mpiexecjl` interfaces also work on
other systems with a compatible HAMSTER/MPI environment.

For XDATCAR input:

```bash
./workflows/full_tensor/run.sh covariance --calc-dir "$CALC" --data-dir "$DATA" \
  --trajectory /path/to/SC_XDATCAR --primitive "$CALC/POSCAR_primitive" \
  --start 500 --final all --no-overwrite
```

The accumulator subsequently looks for `SC_XDATCAR` in DATA or CALC. If the
trajectory lives elsewhere, create the appropriate symlink only where no file
already exists. For native MACE/LAMMPS input, replace the preparation options
with `--trajectory /path/to/positions.lammpstrj --repetitions 4 4 4` and set
`EPH_LAMMPS_SPECIES` to its actual type order, e.g. `Pb,Cs,Br` for the collected
CsPbBr3 runs. The native reader supports fixed periodic orthorhombic boxes;
tilted or changing cells require an extension. Automatic primitive extraction
requires suitable first-frame geometry; use `--primitive FILE` otherwise.

Example MPI/thread allocation on hydrogen, with four reserved cores:

```bash
export PATH="$PWD/bin:$PATH"
jrun -n 2 -t 2 -c 4 ./workflows/full_tensor/run.sh accumulate \
  --calc-dir "$CALC" --data-dir "$DATA" --start 500 --final all \
  --chunk-size 100 --correlations distributed
```

Without jrun, the equivalent launch is `JULIA_NUM_THREADS=2 bin/mpiexecjl -n 2`
followed by the same `run.sh accumulate ...` arguments. A serial baseline is
`JULIA_NUM_THREADS=1 ./workflows/full_tensor/run.sh accumulate ...`.

Only accumulation is MPI aware. HAMSTER is called collectively on the ranks;
the distributed correlation implementation partitions electronic columns and
communicates the partial statistics. Rank zero owns checkpoint writing.
`--correlations serial` keeps collective HAMSTER but uses serial correlation
reduction. This is not an independent job per rank. Chunk size need not be
divisible by rank/thread count. `run.sh` sets OMP/OpenBLAS threads to one;
Julia threads are controlled separately. More ranks do not guarantee a
proportional speedup because reductions, memory traffic and HDF5 writing remain.

Run the remaining stages once, without multi-rank launch:

```bash
./workflows/full_tensor/run.sh d --calc-dir "$CALC" --data-dir "$DATA"
./workflows/full_tensor/run.sh phonons --calc-dir "$CALC" --data-dir "$DATA" --temperature 425
./workflows/full_tensor/run.sh unitary --calc-dir "$CALC" --data-dir "$DATA"
./workflows/full_tensor/run.sh g-orbital --calc-dir "$CALC" --data-dir "$DATA"
./workflows/full_tensor/run.sh g-band --calc-dir "$CALC" --data-dir "$DATA"
```

For one initial electron momentum, the
[selected-k postprocessor](../../postprocessing/README.md) can replace the two
full-g stages after d, phonons and U. This reduces g storage independently of
how uH/d was accumulated.

## Supporting files

| File | Responsibility |
|---|---|
| `run.sh` | Selects the Julia environment (`EPH_JULIA_PROJECT`, then `HAMSTER_PROJECT`, then repository root), sets thread limits, and dispatches accumulation or a single-process stage. |
| `run_stage.jl` | Parses CALC/DATA for the non-MPI stages, establishes their path environment and propagates the covariance threshold to phonons. |
| `MaterialPaths.jl` | Resolves material/storage defaults and optional output-root containment. It provides no physics or model parameters. |
| `accumulate_uH_parallel.jl` | Parses MPI accumulation options and initializes the collective HAMSTER/correlation workflow. |
| `AccumulationCore.jl` | Distributed-correlation accumulation, centering data, checkpoint/restart and convergence bookkeeping. |
| `AccumulationCore_serial.jl` | Alternative serial-correlation core with the same collective HAMSTER frontend. |
| `DistributedCorrelations.jl` | Distributes correlation columns, combines snapshot contributions and coordinates checkpoint data transfer. |
| `MaceTrajectory.jl` | Native LAMMPS trajectory reader, type/atom-ID ordering and preparation support. |
| `atomic_masses_amu.txt` | Default species masses in amu; supply a complete matching table for another material. |
| `ElectronPhononCouplingTools.jl`, `HamsterHrLoader.jl`, `PrimitiveBasisRelabel.jl`, `VaspTrajectoryStatistics.jl` | Include wrappers for the documented [common modules](../../common/README.md). |

## Restart, convergence and primary results

The accumulator defaults to resuming its existing checkpoint and checks its
selection/labels. Changed trajectories, models or sampling belong in a new
run directory; do not mix their files. The checkpoint retains both correlation
data and the mean Hamiltonian needed for later band reconstruction.

Convergence logs record the cumulative tensor norm and the full normalized
change between N and P snapshots,

\[
 r_{N,P}=\frac{\|B_N-B_P\|_F}{\max(\|B_N\|_F,\|B_P\|_F)},\qquad P=N-M.
\]

`EPH_CONVERGENCE_M` controls M (default 1; it must fit within a chunk). This is
not necessarily the spacing between printed rows and is not a statistical error
bar. Retain the log's `N_added` and `N_previous` columns when comparing runs.

Preserve the primary correlation checkpoint, covariance and structural/basis
metadata, and the derived D/d results used in subsequent work. Full orbital/band
g files are regenerable from d, phonons and U, provided their conventions and
electronic reference are preserved. No cleanup is performed by this README.
