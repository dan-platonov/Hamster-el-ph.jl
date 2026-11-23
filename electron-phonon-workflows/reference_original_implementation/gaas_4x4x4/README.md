# Original GaAs 4x4x4 serial workflow

This directory preserves the earlier single-process scripts and their numerical
settings, with input/output paths made configurable. Shared numerical modules
are loaded from ../../common. Source hashes are recorded in the provenance file.

Prefer ../../workflows/full_tensor for new runs; it supports the shared-directory
interface, MPI HAMSTER and the corrected covariance phonon implementation.
This reference includes the historical phonon method and is not a second
independently maintained material framework.

For the historical all-in-one directory layout, place the real input files in
RUN, export its location and execute the stages from here. No scripts need to be
copied into RUN:

```bash
export EP_CALC_DIR=/path/to/legacy-run
export EP_DATA_DIR="$EP_CALC_DIR"
export ELECTRON_PHONON_DATA_DIR="$EP_DATA_DIR"
julia --project=../.. compute_equilibrium_covariance.jl --start 500 --final all
julia --project=../.. accumulate_uH_chunked_with_convergence_final_v2.jl --start 500 --final all --chunk-size 100
julia --project=../.. compute_d_from_accumulated_streaming.jl
julia --project=../.. compute_phonons_from_covariance.jl --temperature 100
julia --project=../.. unitary_transformation_matrix_construction.jl
julia --project=../.. compute_electron_phonon_g.jl
julia --project=../.. transform_electron_phonon_g_to_band_basis.jl
```

Use the HAMSTER-enabled Julia environment and executable described in RUNNING.
Supply atomic_masses_amu.txt in RUN for the historical g converter if needed.
The upstream 100K folder's v2 accumulator contained a hardcoded 400K output
path. This copy replaces that path with EP_DATA_DIR; its directory name alone
was not treated as proof of a particular completed run's provenance.


## What each preserved script does

| File | Role |
|---|---|
| `compute_equilibrium_covariance.jl` | XDATCAR mean positions, primitive/supercell equilibrium structures and displacement covariance. |
| `accumulate_uH_chunked_with_convergence_final_v2.jl` | Earlier serial chunked HAMSTER/uH accumulator; saves displacement/Hamiltonian/correlation means and convergence diagnostics. |
| `compute_d_from_accumulated_streaming.jl` | Covariance-pseudoinverse regression, translation-averaged real-space D and Fourier d. |
| `compute_phonons_from_covariance.jl` | Historical covariance-derived phonon implementation. It does not include the main workflow's corrected Gamma projection-before-inversion treatment; use the maintained full-tensor version for new runs. |
| `unitary_transformation_matrix_construction.jl` | Builds and diagonalizes the primitive Bloch Hamiltonian from the saved ensemble-mean Hamiltonian. |
| `compute_electron_phonon_g.jl` | Full orbital g from d and phonon oscillator normalization. |
| `transform_electron_phonon_g_to_band_basis.jl` | Full band g from orbital g and U(k), U(k+q). |
| `atomic_masses_amu.txt` | Species masses in amu. |
| `ElectronPhononCouplingTools.jl`, `HamsterHrLoader.jl`, `PrimitiveBasisRelabel.jl`, `VaspTrajectoryStatistics.jl` | Include wrappers for the [common modules](../../common/README.md). |

The [full-tensor README](../../workflows/full_tensor/README.md) explains the
statistical equations and main output files. The [methods](../../docs/METHODS.md)
document Fourier/cell normalization and plotted observables. This historical
directory preserves provenance; its settings are not automatically equivalent
to every later corrected or parallel implementation.
