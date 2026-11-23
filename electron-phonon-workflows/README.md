# Electron–phonon couplings from molecular-dynamics correlations

Research workflows for reconstructing effective electron–lattice couplings from
atomic displacements and HAMSTER tight-binding Hamiltonians. The collection
contains the successful GaAs and CsPbBr3 implementations, organized so the code
is separate from material inputs and large results.

Start with [RUNNING.md](docs/RUNNING.md) for installation and commands.
Read [METHODS.md](docs/METHODS.md) for equations, conventions and limitations.

## Which workflow should I use?

| Directory | Purpose | Origin |
|---|---|---|
| [workflows/full_tensor](workflows/full_tensor/README.md) | Reusable covariance → MPI/threaded uH accumulation → D → d → phonons → orbital g → band g | Shared CsPbBr3 pipeline; also accepts XDATCAR |
| [workflows/large_cell](workflows/large_cell/README.md) | Direct Fourier accumulation of d in persistent orbital-pair checkpoints, without the full uH tensor | GaAs 10×10×10 TB/ML workflow |
| [postprocessing](postprocessing/README.md) | Compact g at one selected initial electron k; static Hamiltonian at mean positions; pair-to-combined-d exporter | Shared GaAs postprocessing |
| [reference_original_implementation/gaas_4x4x4](reference_original_implementation/gaas_4x4x4/README.md) | Earlier serial workflow with configurable input/output paths | GaAs 4×4×4 reference |
| [common](common/README.md) | Basis mapping, Hamiltonian loading, covariance regression and Fourier conversion | Shared numerical modules |
| `visualization/gaas/` | Orbital/displacement heatmaps, signed differences, PERTURBO path comparison | GaAs plotting scripts |
| `visualization/cspbbr3/` | Spin-reduced orbital heatmaps, temperature differences, conduction-pair path comparison | CsPbBr3 plotting scripts |
| `visualization/convergence/` | Tensor norm and normalized cumulative-change plots | Local convergence scripts |
| `examples/` | Material directory templates and small inputs; no trajectories/model weights | GaAs and CsPbBr3 |
| `reference_data/PERTURBO/` | Two-column reference curve used in the existing GaAs comparison | Export of the supplied PERTURBO plot |

Each workflow README explains its scripts, inputs/outputs, equations and
execution. The [visualization README](visualization/README.md) explains every
plotter and its plotted quantity; [bin/README](bin/README.md) explains MPI launch.

## !!!!!!! LARGE-CELL HAMILTONIAN AVERAGING NEEDS AN UPDATE !!!!!!!

**The current direct large-cell code does not save the MD-averaged Hamiltonian.
Its present bands come from diagonalizing a HAMSTER Hamiltonian evaluated at
MD-averaged positions. In general, H(mean positions) is not mean(H).** The full
workflow instead saves and uses the ensemble-mean Hamiltonian. The
[large-cell README](workflows/large_cell/README.md) explains the limitation and
which scripts need updating; that scientific update has not been implemented.

## Main interface

The main reusable interface is:

```bash
./workflows/full_tensor/run.sh STAGE --calc-dir /path/to/inputs --data-dir /path/to/large-results
```

The large-cell workflow has two independent storage choices:

1. Direct accumulation avoids storing the full displacement–Hamiltonian tensor.
   Its primary d result is a set of orbital-pair checkpoint files.
2. Selected-k conversion stores g for one initial **electron** momentum and all
   sampled phonon q points, branches and electronic band pairs. Gamma is the
   default initial k, not a restriction to phonon q=Gamma.

The direct algorithm still stores all k,q displacement couplings in its pair
files. These can be large. The supplied exporter makes an additional combined d
file for the established postprocessors; it does not remove the primary pairs.
The optional full-g conversion remains available for reproducing the earlier
GaAs TB large-cell calculation, subject to its much larger storage requirement.

## How this matches the existing calculations

The main full-tensor workflow is the shared implementation used for CsPbBr3,
with XDATCAR support for GaAs. The original serial GaAs 4x4x4 workflow is kept
separately for its historical settings, including its original phonon method.
For the differences in Gamma phonon treatment see [METHODS](docs/METHODS.md).

The large-cell direct d kernels and the selected-k/full-g conversion routines
come from the hydrogen implementations. The repository organizes them into
reusable stages. A TOML text file supplies the large-cell paths and run settings;
it does not introduce another calculation method. The average-position helper
saves one Hamiltonian, after which explicit commands construct the bands, choose
selected-k or full g output, and plot. These are adaptations for separate code,
input and result directories; the underlying calculation stages are retained.

## Inputs and dependencies

Use Julia >=1.10 with a compatible HAMSTER installation, and Python >=3.10.
Julia uses HDF5, MPI and FFTW. HAMSTER must be installed separately and its
matching model/configuration files supplied for each calculation. Python plotting
uses NumPy, h5py and Matplotlib; the supplied SVG repair utility also uses lxml.
MACE is a producer of the input trajectories and is not needed to postprocess
an existing native LAMMPS dump.

No trajectories, fitted parameter files, generated Hamiltonians or large HDF5
results are included. Example `hconf.example` files describe the model settings;
copy the appropriate one to `hconf` and supply the actual matching model files.
The GaAs primitive template is shared across TB/ML examples, but each run must
use its own trajectory and thermally averaged structure.

All new scientific plots export SVG with outlined lettering. Matrix heatmaps
retain the existing Cartesian convention: the first row is at the bottom.
The true mu=nu diagonal therefore runs bottom-left to top-right.

## Repository status

This is a local source collection prepared for later GitHub publication. It is
not an automatic upload and contains no GitHub credentials. Numerical kernels
were copied from the working implementations; portability changes concern
paths, launchers, configuration and plotting exports. See the source inventory
in `provenance/source_inventory.json` and [portability notes](provenance/PORTABILITY_CHANGES.md).

A clone supplies source code, input templates and the small PERTURBO reference
curve; running a material requires its external HAMSTER installation, matching
model parameters and MD trajectory. The source collection is ready to organize
on GitHub, but a redistribution license has not been selected. The author-supplied
PERTURBO source link and its scope are recorded in the reference-data
[README](reference_data/PERTURBO/README.md).
