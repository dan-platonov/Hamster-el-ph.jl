# Shared numerical modules

These modules contain the common numerical definitions used by the calculation
scripts. They are libraries loaded with `include`, rather than stages to launch
directly. The same-named files in the workflow, postprocessing and historical
reference directories are short include wrappers pointing here. Change a shared
implementation here rather than editing those wrappers.

For the complete equations see [METHODS](../docs/METHODS.md). For the order in
which these modules are used see the [full-tensor workflow](../workflows/full_tensor/README.md)
and the [large-cell workflow](../workflows/large_cell/README.md).

## Files and responsibilities

| File | Role and principal interface | Data handled |
|---|---|---|
| `VaspTrajectoryStatistics.jl` | POSCAR reading/writing, streamed XDATCAR selection, periodic minimum-image displacements and cell/component indexing. Exports `VaspHeader`, `VaspStructure`, `read_poscar`, `write_poscar`, `foreach_selected_xdatcar_frame`, `minimum_image_displacements`, `enumerate_cells`, `canonical_cell`, `component_index`, `component_label`. | A structure contains a lattice with lattice vectors in columns and a `3 × Natoms` fractional-position matrix. Cartesian lengths are in Angstrom. |
| `HamsterHrLoader.jl` | Reads HAMSTER real-space sparse HDF5 matrices and assembles the periodic supercell Hamiltonian. Exports `available_hamster_snapshots`, `load_hamster_snapshot`, `get_HR`. | Recognizes static `Hr` and trajectory groups such as `Hr__4000` or `Hr_GaAs_4000`. Loads sparse CSC blocks, real-space translations and the matching basis. Hamiltonian values are in eV. |
| `PrimitiveBasisRelabel.jl` | Connects a HAMSTER supercell basis index to a primitive orbital, primitive atom, cell translation, local orbital and spin tag. Important routines are `read_simple_poscar`, `read_hconf_orbitals`, `build_primitive_basis_map`, `relabel_basis_index`, `relabel_matrix_element`, `primitive_basis_mapping_table`. | Uses supercell/primitive geometry, the HAMSTER basis and orbital definitions in `hconf`; it does not infer orbital identity from a heatmap or from a band number. |
| `ElectronPhononCouplingTools.jl` | Reads correlation/covariance data, computes the covariance regression with translation averaging, saves/loads real-space D and writes momentum-resolved d. Public routines are `load_accumulated_hamiltonian`, `load_displacement_covariance`, `compute_translation_averaged_D`, `save_translation_averaged_D`, `load_translation_averaged_D`, `save_fourier_couplings`. | Reads `sum_uH_primitive_labels.h5` and `displacement_covariance.h5`; writes `electron_lattice_D.h5` and `electron_lattice_d_kq.h5` through the calling stage. |

The VASP reader requires explicit species names. Native MACE/LAMMPS parsing is
in `workflows/full_tensor/MaceTrajectory.jl`, not in the VASP library. The basis
mapper assumes a primitive cell repeated diagonally by integer repetitions
`(n1,n2,n3)` and checks atom/species assignments and periodic geometry. Arbitrary
supercell transformation matrices require a mapping extension.

## Statistical and numerical definition

For N selected snapshots and a displacement index a,

\[
 C_{ab}=\langle(u_a-\bar u_a)(u_b-\bar u_b)\rangle,\qquad
 B_{a,ij}=\langle u_a H_{ij}\rangle-\bar u_a\bar H_{ij},\qquad
 D_{a,ij}=\sum_b(C^+)_{ab}B_{b,ij}.
\]

The sample averages use denominator N. Covariance eigenvalues must exceed the
larger of the absolute cutoff and the relative cutoff times the largest
eigenvalue. The calling script controls the cutoffs; the standard relative
cutoff is `1e-10`. Discarding a constrained or poorly sampled displacement
direction changes the retained regression subspace. Units are Angstrom squared
for C, eV Angstrom for B, and eV/Angstrom for D and d.

`compute_translation_averaged_D` checks the stored statistical labels, then
processes electronic Hamiltonian entries in slabs. It centers the correlations
using the Hamiltonian sample's displacement and Hamiltonian means, applies
C's pseudoinverse, and averages over common primitive translations. Setting
`materialize_raw_D=false` avoids retaining the enormous unaveraged D tensor;
the translation-averaged result is still saved. A slab is a memory-management
choice, not a change to the regression.

With Nc primitive cells, the Fourier convention is

\[
 d^{\alpha c}_{\mu\nu}(k,q)=\frac{1}{\sqrt{N_c}}
 \sum_{R,R'}e^{-i(k+q)\cdot R}
 D_{\alpha c;\mu R,\nu R'}e^{+ik\cdot R'}.
\]

Here D is the translation-averaged response referenced to a displacement in
cell zero. In reduced reciprocal coordinates the dot-product phases contain
`2*pi`. The stored d already includes `1/sqrt(Nc)`; the later g conversion must
not insert that factor again. No phonon mass/frequency normalization is applied
to d in this library.

## Labels, axes and reuse

Electronic indices i,j denote supercell orbitals; mu,nu denote primitive
orbitals. A displacement is identified by primitive atom alpha, cell L and
Cartesian direction c. Cell coordinates are periodic integer triplets, and
Julia basis/band indices are one-based. SOC spin labels remain part of the
electronic basis. A plotting-only spin reduction must not change this mapping.

The HDF5 files store explicit block indices and momentum grids. Read those
labels instead of assuming that a lexical dataset order defines mu,nu or that
Gamma is the first momentum. Julia `(k,q)` arrays are exposed by h5py as `(q,k)`;
the direct pair files have another explicitly documented component axis.

`load_accumulated_hamiltonian(...; load_mean_uH=false)` is used when only the
electronic mean and mapping are needed, avoiding a large correlation read.
The direct Fourier driver also reuses covariance/mapping utilities but has its
own accumulation algorithm. None of these shared modules independently
constructs a band reference, launches a trajectory calculation, or decides
between an ensemble-mean Hamiltonian and a Hamiltonian at mean positions.
