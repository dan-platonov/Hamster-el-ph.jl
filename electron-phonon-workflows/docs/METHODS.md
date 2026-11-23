# Mathematical definitions and workflow conventions

## 1. Structures, displacements and averages

Let a label an ionic Cartesian degree of freedom, including atom and supercell
translation, and let i,j label electronic orbitals in the supercell. A fixed-cell
MD trajectory provides positions. The preparation code unwraps periodic motion,
constructs an MD-averaged primitive-periodic structure and forms displacements
relative to that reference. The lattice, atom order and primitive-site mapping
must be consistent throughout the calculation.

Using the selected N snapshots, the covariance estimator has denominator N:

\[
 C_{ab}=\langle(u_a-\langle u_a\rangle)
                   (u_b-\langle u_b\rangle)\rangle.
\]

The displacement–Hamiltonian correlation is

\[
 B_{a,ij}=\langle u_a H_{ij}\rangle
          -\langle u_a\rangle\langle H_{ij}\rangle.
\]

The mean electronic Hamiltonian and displacement mean must refer to the same
selected Hamiltonian snapshots. A covariance calculated from a different frame
selection changes the regression and must be treated as an explicit choice.
Default examples select every frame starting with ordinal 500. A frame ordinal
is one-based and is distinct from the native LAMMPS timestep.

## 2. Effective linear response

The effective regression tensor is

\[
 D_{a,ij}=\sum_b(C^+)_{ab}B_{b,ij},\qquad
 H_{ij}-\langle H_{ij}\rangle\simeq
 \sum_aD_{a,ij}(u_a-\langle u_a\rangle).
\]

Here C^+ is a symmetric covariance pseudoinverse. Eigenvalues are retained only
above the larger of the absolute cutoff and the relative cutoff times the
maximum eigenvalue. The default relative cutoff is 1e-10. Constrained motion and
poorly sampled directions make this cutoff part of the numerical definition.
D has units eV/Angstrom. At finite temperature it is a best linear response over
the sampled ensemble, not automatically the zero-temperature infinitesimal
Hamiltonian derivative. Anharmonicity/nonlinear electronic response can change
it with temperature.

The full workflow accumulates raw sums for u, H and uH, then constructs the
centered covariance product. The streaming D stage processes electronic columns
in slabs and avoids materializing the entire unaveraged D tensor in RAM.
The uH checkpoint itself can still dominate memory/storage for large cells.

## 3. Translational averaging and Fourier convention

Let Nc=n1*n2*n3 be the number of primitive cells. Primitive electronic indices
are mu,nu; alpha labels a primitive atom and c a Cartesian direction. The
regression is averaged over common translations before Fourier conversion.
For a displacement at the reference cell, the stored convention is

\[
 d^{\alpha c}_{\mu\nu}(\mathbf k,\mathbf q)=\frac1{\sqrt{N_c}}
 \sum_{R,R'}e^{-i(\mathbf k+\mathbf q)\cdot R}
 D_{\alpha c;\mu R,\nu R'}e^{+i\mathbf k\cdot R'}.
\]

Reduced coordinates use phases exp(2*pi*i*k.R). Stored d includes 1/sqrt(Nc).
Momenta belong to the commensurate supercell mesh; no dense-q interpolation or
analytic long-range correction is implicit. HDF5.jl matrix axes are reversed
when exposed through h5py; for example a Julia (k,q) block appears as (q,k).
Block indices explicitly associate every array with alpha,c,mu,nu.

## 4. Direct large-cell accumulation

**!!!!!!! REQUIRED UPDATE: THE DIRECT DRIVER DOES NOT SAVE AN MD-AVERAGED
HAMILTONIAN. ITS CURRENT BAND REFERENCE IS H AT MD-AVERAGED POSITIONS, NOT MEAN(H).
The full-tensor workflow uses mean(H).** The
[large-cell README](../workflows/large_cell/README.md) specifies which scripts
need updating. This distinction concerns the band reference; the direct d
accumulation below still samples the trajectory Hamiltonians.

For the fixed covariance used in a run, define

\[
 z_a(t)=\sum_b(C^+)_{ab}[u_b(t)-\overline u_{b,H}],\qquad
 Z_{\alpha c}(\mathbf q,t)=\sum_L e^{+i\mathbf q\cdot L}z_{\alpha c L}(t),
\]
\[
 T_{\mu\nu}(K,k,t)=\sum_{R,R'}e^{-iK\cdot R}
 H_{\mu R,\nu R'}(t)e^{+ik\cdot R'},
\]
\[
 d^{\alpha c}_{\mu\nu}(k,q)=\frac{1}{N_c\sqrt{N_c}}
 \left\langle Z_{\alpha c}(q,t)T_{\mu\nu}(k+q,k,t)\right\rangle_t.
\]

This is an algebraic rearrangement of the fixed-C regression and translational
average. It does not require constructing the full uH matrix. Snapshot means,
cutoffs, mapping and sampling must match for equivalence. The implementation
uses a forward FFT on the first cell coordinates and an unnormalized backward
FFT on the second, hence the explicit Nc normalization.

Each `pair_MMMM_NNNN.h5` contains an unnormalized `sum_d`; physical d is
`sum_d / number_of_snapshots`. Julia logical axes are (k,q,ionic-component),
exposed by h5py as (ionic-component,q,k). These pairs are the primary result and
must be retained. Atomic pair-file replacement, input digests, locking and
window-progress checks support restart. Temporary HAMSTER snapshots and cache
windows are still used; the algorithm does not promise zero temporary storage.
`accumulation_window` must be a multiple of `hamster_chunk_size`.
The copied direct driver is not an MPI algorithm. Its FFT threads are separate
from any threading performed inside the external HAMSTER command.

## 5. Temperature-dependent phonons from covariance

The phonon stage Fourier-transforms the displacement covariance, then forms an
effective force-constant matrix from classical fluctuation–response:

\[
 K(q)=k_BT\,C(q)^+,\qquad
 \mathcal D(q)=M^{-1/2}K(q)M^{-1/2},\qquad
 \mathcal D(q)e_s(q)=\omega_s(q)^2e_s(q).
\]

Units are converted consistently between Angstrom, eV, amu and SI. This is an
effective covariance-based harmonic description of the finite-temperature
ensemble; it does not provide phonon linewidths or the full anharmonic spectrum.

At Gamma, rigid translations are projected out of the **covariance before its
pseudoinverse**. With T containing three uniform Cartesian translations,

\[
 P=I-T(T^\dagger T)^{-1}T^\dagger,\qquad C_{\rm used}(0)=PC(0)P.
\]

Projecting only K after inversion is not equivalent for unequal masses and
constrained centre-of-mass motion. The retained CsPbBr3 implementation includes
this correction and records raw and projected covariances separately. The full
pipeline also propagates the D-regression absolute covariance cutoff to the
phonon stage when available. The historical GaAs reference retains its original
phonon implementation for reference; use the main workflow for new calculations.

## 6. Phonon-normalized and band-basis couplings

\[
 g^{\rm orb}_{\mu\nu,s}(k,q)=\sum_{\alpha c}
 \sqrt{\frac{\hbar}{2M_\alpha\omega_s(q)}}
 e_{\alpha c,s}(q)\,d^{\alpha c}_{\mu\nu}(k,q),
\]
\[
 g_{mn,s}(k,q)=\left[U^\dagger(k+q)\,g_s^{\rm orb}(k,q)\,U(k)\right]_{mn}.
\]

The code applies the eigenvectors with this convention, without adding an
extra complex conjugation to e. g is stored in eV and inherits the finite-cell
1/sqrt(Nc) normalization of d. Imaginary/near-zero phonon modes carry validity
flags and NaN couplings. At exact phonon Gamma the three translations are
undefined under the oscillator normalization; the path plots omit that point.
Finite optical-mode Gamma data remain available separately in the files.

The full workflow constructs U(k) from the **ensemble-mean Hamiltonian**.
The successful GaAs large-cell compact conversion used one Hamiltonian at the
**MD-averaged positions**. For nonlinear H these are different references:

\[
 \langle H(R)\rangle\ne H(\langle R\rangle).
\]

The static-reference helper and this distinction are preserved explicitly.
The helper now stops after saving its reference file; subsequent U/g/plot stages
are explicit commands, rather than automatically launching other calculations.

## 7. Quantities in the plots

The species-X orbital heatmap is

\[
 A^X_{\mu\nu}=\sqrt{\frac{N_c}{N_kN_q}
 \sum_{\alpha\in X}\sum_{c,k,q}|d^{\alpha c}_{\mu\nu}(k,q)|^2}.
\]

This reports an effective sensitivity of orbital Hamiltonian matrix elements to
atomic displacements, in eV/Angstrom. Directions and same-species primitive
atoms are summed, not averaged; comparisons across species with different
multiplicities need to account for this. The magnitude contains no sign/phase.
A difference heatmap is A_selected-A_reference, taken after the square root.
It is not the norm of the complex tensor difference.

Off-diagonal matrix elements include hopping/orbital mixing. There is no
requirement that their displacement response be weaker than diagonal entries.
Primitive mu=nu alone also does not restrict R=R'. The plotting convention has
row 1 at the bottom; the true matrix diagonal slopes upward from left to right.
For CsPbBr3 only, the heatmap displays the representative up/up orbital block
when spin-flip amplitudes are negligible and up/up and down/down blocks agree.
This is a checked display reduction with no factor of two; the underlying
SOC Hamiltonians and g calculations retain all 28 band states.

The GaAs path comparison follows the original single-band convention n=5,
initial electron k=Gamma, path L-Gamma-X-W-K-Gamma. Its PERTURBO reference is the
supplied two-column curve; no independent DFPT calculation is included here.

The CsPbBr3 path comparison uses the lowest conduction pair C={21,22}, initial
k=R, and phonon path Gamma-X-M-Gamma-R-X:

\[
 G_C(q)=\frac12\sum_{n\in C}\sum_{m\in C}\sum_s
 |g_{mn,s}^{\rm primitive}(R,q)|^2.
\]

The initial two states are averaged, both final states and all 15 phonon
branches are summed. This is invariant under changes of basis within the
selected two-dimensional subspaces. Individual g_21,21 entries are not invariant
under such rotations inside degenerate bands and gave a spurious temperature
comparison at X. The supplied CsPbBr3 model has 20 occupied spin-resolved bands;
21 is not a universal conduction-band index for other materials/models.

Primitive-cell path magnitudes use

\[
 |g^{\rm primitive}|^2=N_c|g^{\rm stored}|^2,\qquad
 G[\mathrm{meV}^2]=10^6N_cG_{\rm stored}[\mathrm{eV}^2].
\]

Only this explicit normalization is removed; cell-size convergence is not
otherwise guaranteed. Neither path plot includes Bose factors or a k average.
The selected-k file keeps all q points and all band pairs for one initial k.

## 8. Convergence diagnostics

For a cumulative estimate B_N and its reference B_P, with M=N-P,

\[
 r_{N,P}=\frac{\|B_N-B_P\|_F}
 {\max(\|B_N\|_F,\|B_P\|_F)}.
\]

The scalar norm-change column is a different quantity. Plotters read the full
matrix-change column `r_cumulative`. Use `N_added` and `N_previous` to identify
what is compared; adjacent logged rows need not define P. Existing example
100 K and 400 K logs use different increments (mostly 50 versus 1), so their
relative-change magnitudes are not directly comparable. The last short update
can also have a different M. These values are stability diagnostics, not
statistical uncertainties. Direct-d convergence uses its corresponding d norm.

## 9. Limits of the retained models

A coarse commensurate q mesh cannot resolve the q->0 Fröhlich asymptote.
The supplied Wolf model uses finite-cutoff electrostatics, and the conversion
code adds no analytic Born-charge/dielectric long-range polar term. An omitted
Gamma point or a finite optical Gamma value cannot establish polaron formation
or the absence of polar coupling. A quantitative small-q/polaron study needs
appropriate long-range physics and momentum sampling.
