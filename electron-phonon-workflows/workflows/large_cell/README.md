# Direct Fourier workflow for large cells

## !!!!!!! REQUIRED UPDATE: THIS CODE DOES NOT AVERAGE THE HAMILTONIAN !!!!!!!

**THE CURRENT LARGE-CELL ACCUMULATOR DOES NOT SAVE THE MD-ENSEMBLE AVERAGE
HAMILTONIAN. THE PRESENT BAND BASIS IS OBTAINED BY DIAGONALIZING A HAMSTER
HAMILTONIAN CONSTRUCTED AT THE MD-AVERAGED ATOMIC POSITIONS. THESE TWO OPERATIONS
ARE DIFFERENT. THIS PART OF THE WORKFLOW NEEDS TO BE UPDATED !!!!!!!**

Specifically, the completed GaAs 10x10x10 workflow uses

\[
 H_{\mathrm{positions}}(k)=H_{\mathrm{HAMSTER}}(k;\langle R\rangle_t)
\]

for the electronic bands, whereas the full-tensor workflow uses

\[
 \overline H(k)=\langle H_{\mathrm{HAMSTER}}(k;R_t)\rangle_t.
\]

For a nonlinear position-dependent Hamiltonian these are generally unequal.
The current large-cell d is still sampled from MD Hamiltonians: the missing
quantity is their separately saved **mean for the band reference**, not the
displacement–Hamiltonian sampling itself. Averaging geometry does not recover
the electronic ensemble mean. Translation averaging of one static Hamiltonian
also does not turn it into an MD average.

### Required future change: scripts and logic

1. Extend **`DirectFourierEph.jl`** where snapshot Hamiltonians enter the cache
   and where completed windows are committed: accumulate a separate sum of
   primitive translation-averaged Hamiltonians (or equivalently H(k)), with a
   count for exactly the snapshots used in d. This needs only the one-electron
   mean, not a stored full uH tensor. Checkpoint the sum/count with the same
   committed-window bookkeeping so resuming does not omit or double-count frames.
2. Extend **`run_direct_eph.jl`** to expose the mean-H output and save its basis,
   grid, snapshot selection and reference definition. Divide the completed sum
   by its actual snapshot count; do not substitute the Hamiltonian of a mean
   structure. The existing translation-averaged-H schema can be reused.
3. Update **`postprocessing/unitary_transformation_matrix_construction.jl`** to
   select this ensemble mean explicitly. Its current priority chooses
   `hamiltonian_at_average_positions.h5` first, even when a mean-H file also
   exists; merely adding a new file would therefore be insufficient.
4. Make **`postprocessing/compute_hamiltonian_at_average_positions.jl`** an
   explicitly optional approximation rather than the normal large-cell route.
   Regenerate U/band energies and g using the new reference, with provenance
   propagated to the selected-k output. The selected-k contraction itself can
   remain the same. Update the run instructions to match the new default.

**These are instructions for a future scientific update. This repository still
contains the existing implementation; the ensemble-mean-H update has not been
implemented.** The direct Fourier d kernel does not need a new normalization
factor to address this band-reference issue.

## What the current algorithm computes

This workflow avoids the full displacement–Hamiltonian uH tensor. It is the
direct Fourier implementation used for GaAs 10x10x10, with a portable TOML
launcher. The three numerical `Direct*.jl` modules were collected without
numerical edits. It is separate from the MPI full-tensor accumulator.
The TOML launcher supplies the paths/settings that were previously in a driver.
Separating the static-H, U, g and plotting commands accommodates the new folder
layout and lets you select the output format; it does not introduce another
electron-phonon calculation method.

For a fixed covariance C and the mean displacement of the Hamiltonian sample,

\[
 z_a(t)=\sum_b(C^+)_{ab}[u_b(t)-\bar u_{b,H}],\qquad
 Z_{\alpha c}(q,t)=\sum_L e^{+iq\cdot L}z_{\alpha cL}(t),
\]
\[
 T_{\mu\nu}(K,k,t)=\sum_{R,R'}e^{-iK\cdot R}
 H_{\mu R,\nu R'}(t)e^{+ik\cdot R'},
\]
\[
 d^{\alpha c}_{\mu\nu}(k,q)=\frac{1}{N_c\sqrt{N_c}}
 \left\langle Z_{\alpha c}(q,t)T_{\mu\nu}(k+q,k,t)\right\rangle_t.
\]

This rearranges the fixed-C regression and primitive translation average; it
does not assume that C itself is translation invariant. Equivalence to the full
regression requires the same sampling, means, covariance cutoff, geometry and
basis mapping. Reduced-coordinate phases contain `2*pi`. The kernel uses a
forward FFT on the first electronic cell coordinates and an unnormalized
backward FFT on the second; the explicit `Nc*sqrt(Nc)` denominator is required.
d is in eV/Angstrom and includes `1/sqrt(Nc)`.

## Files

| File | Responsibility |
|---|---|
| `run_direct_eph.jl` | Reads one TOML file, resolves relative paths against that file, and calls `run_direct_eph` with input, sampling, cutoff, FFT and checkpoint settings. |
| `DirectFourierEph.jl` | Main driver: indexes XDATCAR, determines the fixed sample mean displacement, loads C and its pseudoinverse, invokes external HAMSTER in chunks, caches sparse snapshot Hamiltonians/displacement transforms, and updates persistent orbital-pair d files by window. Also handles signatures, locks, restart state and optional combined output. It does not retain an MD mean Hamiltonian. |
| `DirectFourierKernel.jl` | Electronic-cell FFTs, k+q mesh lookup and the Fourier product accumulation with its normalization. |
| `DirectDConvergence.jl` | Pair-wise diagnostic slabs, cumulative d norms and full normalized changes, plus checkpoint history/compatibility bookkeeping. It avoids a second full d tensor in memory. |
| `ElectronPhononCouplingTools.jl`, `HamsterHrLoader.jl`, `PrimitiveBasisRelabel.jl`, `VaspTrajectoryStatistics.jl` | Include wrappers for the [common modules](../../common/README.md). |

## Input preparation and launch

Prepare covariance and thermal structures with the
[full-tensor covariance stage](../full_tensor/README.md), without running its
uH accumulation. The direct driver requires XDATCAR; native MACE trajectories
must first be converted by preparation. CALC needs `hconf`, all matching model
files, `POSCAR_equlibr` and `POSCAR_equil_primitive`. DATA needs
`displacement_covariance.h5`; the trajectory is set explicitly in the TOML.

Copy and edit [direct_10x10x10.toml](../../examples/GaAs/direct_10x10x10.toml).
Run from the repository root:

```bash
julia --project=. workflows/large_cell/run_direct_eph.jl /path/to/direct_run.toml
```

| Setting | Meaning |
|---|---|
| `calc_dir`, `data_dir`, `trajectory` | Input directory, large-result directory and XDATCAR; relative paths are resolved from the TOML location. |
| `model_files` | Actual files under CALC whose contents identify the model/configuration inputs; include ML parameters and all additional model inputs when applicable. |
| `start_snapshot`, `final_snapshot`, `skip_snapshots` | One-based trajectory selection; `final_snapshot="all"` selects through the end. Match the intended covariance statistics. |
| `hamster_chunk_size`, `accumulation_window` | External HAMSTER chunk and accumulation window; the window must be divisible by the chunk size. |
| `pseudoinverse_rtol`, `pseudoinverse_atol` | Covariance eigenvalue cutoffs; defaults `1e-10` and zero. |
| `h_tolerance` | Small Hamiltonian-entry cutoff; default `1e-14` eV. |
| `convergence_m`, `diagnostic_q_block` | Last M contributions used in convergence comparison, and diagnostic q-slab size. M must fit within a HAMSTER chunk. |
| `fft_threads` | FFTW thread count; default 1. |
| `hamster_executable` | External executable; default `hamster`. |
| `assemble_final_output` | Default false: retain pair files only; true also assembles `electron_lattice_d_kq_direct.h5`. |

**This driver is not MPI aware.** Schedule with `jrun -n 1` if using jrun.
Increasing `-n` is not a supported parallelization method for this code. FFTW
threads and external HAMSTER threads are separate settings and must fit inside
the reserved cores. Julia thread count alone does not distribute its pair loop.

## Output, restart and disk use

Primary outputs are `DATA/.direct_eph_work/pair_MMMM_NNNN.h5`, one per orbital
pair, plus `DATA/d_convergence.txt`. Each pair contains `sum_d`; the physical
estimate is `sum_d / number_of_snapshots`. Logical Julia axes are
`(k,q,ionic-component)`, exposed by h5py as `(ionic-component,q,k)`. Read the
mapping, component/grid labels and snapshot count, not filename order alone.

The pairs contain **all k and q**, even when the subsequent g calculation uses
only one initial electron momentum. Keep the pair files: they are the primary
d result. Resume uses input digests, pair progress, file locking and atomic
replacement of individual files. Do not mix pair files from different runs or
completed windows. Signatures include absolute paths as well as source/input
hashes: a new repository location is not automatically a valid restart of an
old hydrogen run. Use a separate DATA for new accumulation; do not edit recorded
signatures to force a restart. Diagnostic convergence compares cumulative d_N
against d_(N-M), with the same fixed displacement centering.

The algorithm uses temporary sparse HAMSTER output and window caches. It avoids
the full uH tensor, but does not guarantee zero temporary storage or small all-q
d output. Allow disk headroom for a pair-file replacement. A combined d export
is another potentially large copy and does not replace or remove the pairs.

## Current postprocessing route

See [postprocessing/README](../../postprocessing/README.md) for detailed file
dependencies and the exact electronic-reference priority. From the repository
root, with CALC and DATA set:

```bash
julia --project=. postprocessing/compute_phonons_from_covariance.jl \
  --directory "$DATA" --temperature 100
python3 postprocessing/export_direct_pair_d.py "$DATA"
# CALC/data must already point to DATA, or create that symlink if absent.
julia --project=. postprocessing/compute_hamiltonian_at_average_positions.jl "$CALC"
julia --project=. postprocessing/unitary_transformation_matrix_construction.jl --directory "$DATA"
```

Then choose the compact selected-k route:

```bash
julia --project=. postprocessing/compute_electron_phonon_selected_k.jl "$DATA"
```

Or use the full-g route, as in the earlier all-k/all-q TB output:

```bash
julia --project=. postprocessing/compute_electron_phonon_g.jl --directory "$DATA"
julia --project=. postprocessing/transform_electron_phonon_g_to_band_basis.jl --directory "$DATA"
```

**The static-H command above reproduces the present mean-position approximation;
it does not perform the required ensemble-mean-H update highlighted at the top.**
The exporter writes `electron_lattice_d_kq.h5`, the filename expected by these
postprocessors. Setting `assemble_final_output=true` in the direct driver uses
a different combined filename and is not the export step shown here.

The selected-k result contains all phonon q points/branches and all electronic
band pairs for initial k=Gamma, or another specified point on the existing mesh.
It avoids full orbital/band g intermediates; it does not mean phonon q is
restricted to Gamma. No analytic long-range polar correction or dense-q
interpolation is introduced by either workflow.
