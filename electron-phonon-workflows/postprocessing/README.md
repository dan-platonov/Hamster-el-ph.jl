# Postprocessing: phonons, electronic bands and electron–phonon g

These scripts finish a calculation from an existing d tensor and displacement
covariance. They do not accumulate uH. Only the static-reference helper calls
HAMSTER; the other conversion stages operate on saved data.

There are two output choices: a full all-k/all-q g tensor, or a compact result
for one initial electron k and every sampled phonon q. Both retain all phonon
branches and all electronic band pairs. Choosing one initial k does not mean
choosing only phonon q=Gamma.

**!!!!!!! LARGE-CELL BAND-REFERENCE LIMITATION !!!!!!!** The present direct
large-cell workflow uses a Hamiltonian at the MD-averaged positions, not the
MD-averaged Hamiltonian. See the prominent warning and required script updates
in [workflows/large_cell/README](../workflows/large_cell/README.md). These
postprocessors do not repair that distinction automatically.

## Files, dependencies and outputs

| File | Inputs / role | Output |
|---|---|---|
| `CalculationPaths.jl` | Shared path and mass-table helpers. Consumes `--directory`, resolves an existing DATA directory or a CALC/data reference, and provides optional mirrored-path/default-temperature helpers. | No scientific result; other scripts include this module. |
| `compute_phonons_from_covariance.jl` | Covariance, temperature and masses. Uses the corrected Gamma covariance projection before inversion. Does not need the uH tensor or a mean Hamiltonian. | `phonon_modes_from_covariance.h5` with frequencies, eigenvectors, masses and momentum grids. |
| `export_direct_pair_d.py` | Completed `.direct_eph_work/pair_*.h5`, covariance and phonon grid information. Divides each pair's `sum_d` by its sample count and writes indexed blocks. | `electron_lattice_d_kq.h5`; an additional combined d file, preserving all primary pairs. |
| `compute_hamiltonian_at_average_positions.jl` | CALC's `POSCAR_equlibr`, `hconf` and matching model inputs, with the direct pair mapping as a basis reference. Runs one static HAMSTER calculation, then averages equivalent primitive translations of that single Hamiltonian. | `hamiltonian_at_average_positions_hr.h5`, `hamiltonian_at_average_positions.h5`, and working config under `.average_positions_hamiltonian/`. This is H at mean positions, not a trajectory mean. |
| `unitary_transformation_matrix_construction.jl` | One of the Hamiltonian reference files described below, its primitive mapping and optionally d for a grid check. Constructs H(k) and diagonalizes it. | `unitary_transformation_matrices.h5`, including band energies, U matrices and reference metadata. |
| `compute_electron_phonon_g.jl` | Combined d, phonon modes and masses. Applies oscillator normalization to every orbital block at every k,q. | Full orbital `electron_phonon_g_kq.h5`. |
| `transform_electron_phonon_g_to_band_basis.jl` | Full orbital g and unitary matrices. Applies the U(k+q)/U(k) rotation in blocks. | Full band `electron_phonon_g_band_kq.h5`. |
| `compute_electron_phonon_selected_k.jl` | Combined d, phonon modes and unitary matrices. Reads only the chosen initial-k slice and contracts directly in memory. | `electron_phonon_g_selected_k.h5` at Gamma, or `electron_phonon_g_kpoint_INDEX.h5` elsewhere, unless `--output` is supplied. |
| `atomic_masses_amu.txt` | Default species masses in amu. A matching file in DATA takes priority; phonon/full-orbital stages also accept `--masses FILE`. | No generated result. |
| `ElectronPhononCouplingTools.jl`, `HamsterHrLoader.jl`, `PrimitiveBasisRelabel.jl`, `VaspTrajectoryStatistics.jl` | Short include wrappers for [common](../common/README.md). | No separate calculation. |

For a new full-tensor calculation, prefer its matching
[stage launcher](../workflows/full_tensor/README.md), which also propagates the
regression covariance cutoff into phonons. These standalone scripts are useful
for existing results and for direct-pair large-cell data.

## Paths and input consistency

Run commands below from the repository root. Explicit `--directory "$DATA"` is
the clearest interface for the standalone phonon, unitary and full-g scripts.
The selected-k script and the exporter take DATA as their positional argument.
The static-H helper instead takes CALC, because it needs the structure/model
inputs there; expose its corresponding DATA with `CALC/data` when using separate
directories. Inspect an existing reference rather than replacing it blindly.

The optional mirrored `EPH_DATA_ROOT` path resolution is retained for the
matching material-relative layout; it is not a universal search for an
arbitrary external CALC directory. Passing DATA explicitly, or supplying the
CALC/data reference needed by the static helper, is more reliable.

Keep repetitions, primitive basis/atom order, masses, snapshot provenance,
momentum-grid order and Fourier phases consistent. The scripts check key
dimensions/grids but a matching shape alone does not establish that two
independently supplied datasets have the same scientific provenance.

## Electronic reference: exact current selection order

The standalone `unitary_transformation_matrix_construction.jl` selects the
first existing file in this order:

1. `hamiltonian_at_average_positions.h5` — one static Hamiltonian at mean positions.
2. `sum_uH_primitive_labels.h5` — the full workflow's saved MD-ensemble mean H.
3. `mean_hamiltonian_translation_averaged.h5` — fallback translation-averaged-H file.

**A static file therefore takes priority even if a saved ensemble mean is also
present.** A future large-cell mean-H implementation must update this selection,
not just write a new file. The full-tensor workflow's own unitary stage instead
reads its correlation checkpoint directly and does not use this priority list.

For the static case the saved `number_of_snapshots=1` describes one evaluated
Hamiltonian. Geometry-average snapshot indices are separate provenance; they do
not convert that Hamiltonian into an ensemble average. The static helper ends
after writing its Hamiltonian reference: run the unitary and g stages explicitly.
On hydrogen this helper chained U, selected-k g and plotting automatically.
Here those same follow-up stages have separate commands for the reorganized
paths and for choosing selected-k versus full-g output; H itself is computed
with the existing single-structure method.

The band construction uses

\[
 h_{\mu\nu}(\Delta)=\frac{1}{N_c}\sum_R
 H^{\mathrm{ref}}_{\mu R,\nu(R+\Delta)},\qquad
 H^{\mathrm{ref}}_{\mu\nu}(k)=\sum_\Delta e^{+ik\cdot\Delta}h_{\mu\nu}(\Delta),
\]
\[
 H^{\mathrm{ref}}(k)U(k)=U(k)\operatorname{diag}\epsilon_n(k).
\]

Columns of U are band eigenvectors. Hermiticity residuals are reported and the
periodic representation is symmetrized. The individual-vector phase convention
does not remove the freedom to rotate an exactly degenerate band subspace.
Save the electronic-reference metadata with any derived g file.

## Conversion equations and units

\[
 g^{\mathrm{orb}}_{\mu\nu,s}(k,q)=\sum_{\alpha c}
 \sqrt{\frac{\hbar}{2M_\alpha\omega_s(q)}}\,
 e_{\alpha c,s}(q)d^{\alpha c}_{\mu\nu}(k,q),
\]
\[
 g_{mn,s}(k,q)=\sum_{\mu\nu}U^*_{\mu m}(k+q)
 g^{\mathrm{orb}}_{\mu\nu,s}(k,q)U_{\nu n}(k).
\]

d is in eV/Angstrom. The oscillator length is converted to Angstrom, so g is
in eV. The code uses the polarization with the convention written above; it
does not add an extra conjugate to e. The inherited stored `1/sqrt(Nc)` is
preserved. Plotting primitive-cell magnitudes subsequently uses
`|g_primitive|^2=Nc*|g_stored|^2`, rather than changing the saved g definition.

The selected-k script rejects absent/nonunique mesh points and checks the d,
phonon and unitary grids. It does no interpolation. Invalid imaginary or nearly
zero-frequency modes produce NaNs and validity information. The three rigid
translations at exact phonon Gamma are invalid for this normalization; finite
optical Gamma entries can remain available. No Bose population factor, electron
occupation factor, k average or band sum is added by these conversion scripts.

## Compact route for existing full-tensor results

After d, phonons and U have been produced by the full workflow:

```bash
DATA=/path/to/large-results/material/METHOD/CELL/TEMPERATURE
julia --project=. postprocessing/compute_electron_phonon_selected_k.jl "$DATA"
# Another initial electron momentum, if present on the mesh:
julia --project=. postprocessing/compute_electron_phonon_selected_k.jl "$DATA" \
  --k-reduced 0.5 0.5 0.5
```

Use either `--k-reduced X Y Z` or `--k-index I` (one-based). `--output FILE`
selects a filename and `--overwrite` explicitly permits replacing an existing
selected-k result. A completion marker is written when successful; a failed
conversion removes its incomplete output. It writes no temporary full-g files.

The optional **`--remove-full-g` deletes existing full orbital/band g files after
successful conversion**. It is not needed to calculate g and is not included
in the normal commands. Primary d/covariance/reference inputs are preserved.

## Current route from direct pair files

This reproduces the existing mean-position-H approximation; it does not implement
the required mean-H update.

```bash
CALC=/path/to/inputs/GaAs/ML_corrected/10x10x10/100K
DATA=/path/to/large-results/GaAs/ML_corrected/10x10x10/100K
julia --project=. postprocessing/compute_phonons_from_covariance.jl \
  --directory "$DATA" --temperature 100
python3 postprocessing/export_direct_pair_d.py "$DATA"
# Only if CALC/data is absent:
ln -s "$DATA" "$CALC/data"
julia --project=. postprocessing/compute_hamiltonian_at_average_positions.jl "$CALC"
julia --project=. postprocessing/unitary_transformation_matrix_construction.jl --directory "$DATA"
julia --project=. postprocessing/compute_electron_phonon_selected_k.jl "$DATA"
```

The exporter requires a complete pair set at a common completed window and
matching covariance sample selection. It refuses an existing destination or
partial input set; it uses an atomic temporary combined export and leaves the
pairs unchanged. Reserve disk space for both forms of d. The selected-k script
currently requires the combined filename `electron_lattice_d_kq.h5`; it does
not read the pair files directly.

The static-H helper reuses an existing `hamiltonian_at_average_positions_hr.h5`
if present and checks its basis against the pair anchor. Keep that intermediate
consistent with the intended model/structure; do not reuse it for a different
reference calculation. It refuses an already-existing final static reference.

## Full-g route

After combined d, phonons and a deliberately selected electronic reference/U:

```bash
julia --project=. postprocessing/compute_electron_phonon_g.jl --directory "$DATA"
julia --project=. postprocessing/transform_electron_phonon_g_to_band_basis.jl --directory "$DATA"
```

This retains every initial k and q, so its storage grows much faster than the
selected-k result. The full band converter processes q blocks (controlled by
`EPH_Q_BLOCK`, default 16), but the output remains a full tensor. Block streaming
reduces working memory, not the number of couplings stored. See
[visualization](../visualization/README.md) for the plotted quantities and
[METHODS](../docs/METHODS.md) for normalization and physical limitations.
