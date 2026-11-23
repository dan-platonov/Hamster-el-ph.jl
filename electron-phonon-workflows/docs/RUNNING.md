# Installation and run commands

Run shell examples from the repository root. Replace `/path/to/...` with actual
paths; paths to data can be anywhere you have space. No model or MD trajectory is
bundled. Repository examples define the input-directory layout, not completed
calculations. Generated outputs go outside Git through the selected data path.

## 1. Julia, Python and HAMSTER

Use Julia >=1.10 compatible with your HAMSTER version, and Python >=3.10. Install this application's dependencies and develop your
existing HAMSTER checkout into the same environment. HAMSTER and the fitted
models are external research software/data: obtain the matching checkout and
model inputs from their provider. This repository does not install or download
them. The Julia Project.toml lists the workflow dependencies; `Pkg.develop`
below adds your local Hamster package:

```bash
cd /path/to/electron-phonon-workflows
export HAMSTER_SOURCE=/path/to/Hamster.jl
julia --project=. -e 'using Pkg; Pkg.develop(path=ENV["HAMSTER_SOURCE"]); Pkg.instantiate()'
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
export PATH="$PWD/bin:$PATH"
```

The external executable `hamster` must also be on PATH. Direct accumulation and
the static-reference helper invoke it as `hamster --conf FILE`. The main MPI
accumulator instead imports the Hamster Julia module and calls it collectively.
Do not replace its matching MPI launcher with a different system MPI blindly.

On hydrogen an existing HAMSTER environment can be reused without installing a
second copy:

```bash
export EPH_JULIA_PROJECT=/path/to/existing/Hamster.jl
export PATH="$PWD/bin:$PATH"
```

`EPH_JULIA_PROJECT` overrides the repository Julia environment for shell
launchers. `HAMSTER_PROJECT` is accepted as a fallback. For direct Julia commands
below, replace `--project=.` with `--project="$EPH_JULIA_PROJECT"` when reusing that
environment. HAMSTER's Julia module and executable must refer to matching code.
Ensure HDF5, MPI and FFTW are available in whichever environment you select.
`jrun` is a hydrogen scheduling utility and is not needed on another computer;
use the serial or MPI commands below there. MPI jobs require an allocation
appropriate to your workstation or cluster.

## 2. Material inputs and large-result locations

A typical calculation input directory is:

```text
/path/to/inputs/GaAs/TB_only/4x4x4/100K/
  hconf
  params.dat                 actual matching TB parameters
  rllm.dat                   actual radial-integral table
  SC_POSCAR                  supercell structure in the trajectory atom order
  POSCAR_primitive           primitive reference topology
  EIGENVAL                   only if referenced by hconf
  ml_params.dat              if the fitted ML configuration requires it
```

Keep every additional file referenced by your model alongside these inputs, or
use correct explicit paths in hconf. Choose the model's correct SOC, orbital,
electrostatic and ML settings; changing temperature does not define them.
GaAs ML examples must use the supplied ML model's configuration, not a TB-only
hconf with arbitrary extra settings. The example names are conventions only.
During full-tensor accumulation, DATA is also the default working directory;
the driver links the standard input names from CALC there. Additional model
files with custom names should be referenced by absolute paths in hconf or
made available in the selected working directory. `--work-dir` overrides that
working location; it is independent of the result destination.

Use `hconf.example` as a configuration reference and copy it to `hconf` only when
it matches your model. Small primitive POSCAR templates are included in
`examples/`. A finite-temperature equilibrium POSCAR must be generated from the
actual trajectory, not copied from another temperature.

The two paths used by the main launcher are:

```bash
CALC=/path/to/inputs/GaAs/TB_only/4x4x4/100K
DATA=/path/to/large-results/GaAs/TB_only/4x4x4/100K
mkdir -p "$DATA"
```

The launcher reads inputs from CALC and writes large outputs to DATA.
If omitted, DATA defaults to CALC/data. No hydrogen account or storage path is
hardcoded. You may optionally set EPH_STORAGE_ROOT to constrain output paths;
if using its mirrored default, also set EPH_MATERIAL_ROOT to your material root.
Explicit --data-dir is easiest to review. The preparation stage puts symlinks to
generated POSCARs/covariance in CALC and refuses to replace ordinary input files.

## 3. GaAs 4x4x4: full pipeline

Supply your XDATCAR and copy/reference SC_POSCAR and the primitive template.
The trajectory can be supplied explicitly; no manual code edit is needed.

```bash
./workflows/full_tensor/run.sh covariance \
  --calc-dir "$CALC" --data-dir "$DATA" \
  --trajectory /path/to/SC_XDATCAR --primitive "$CALC/POSCAR_primitive" \
  --start 500 --final all --no-overwrite
```

The later Hamiltonian driver expects SC_XDATCAR in DATA or CALC. If the trajectory
is elsewhere, make a reference after checking that neither destination exists:

```bash
ln -s /path/to/SC_XDATCAR "$DATA/SC_XDATCAR"
```

Use the same start/final/skip selection for the Hamiltonian sampling. Preparation
currently uses every frame in the selected interval; nonzero Hamiltonian
--skip-snapshots changes that selection and requires an intentional covariance
choice. Default examples use no skipping.

Choose **one** of the following three accumulation launches. They are
alternatives, not three consecutive calculations. All later stages use the
same DATA. One MPI rank, one Julia thread:

```bash
JULIA_NUM_THREADS=1 ./workflows/full_tensor/run.sh accumulate \
  --calc-dir "$CALC" --data-dir "$DATA" \
  --start 500 --final all --chunk-size 100
```

Two MPI ranks and two threads per rank, using MPI.jl's matching launcher:

```bash
export JULIA_NUM_THREADS=2
bin/mpiexecjl -n 2 ./workflows/full_tensor/run.sh accumulate \
  --calc-dir "$CALC" --data-dir "$DATA" \
  --start 500 --final all --chunk-size 100 --correlations distributed
```

Hydrogen `jrun` equivalent, reserving four cores:

```bash
export PATH="$PWD/bin:$PATH"
jrun -n 2 -t 2 -c 4 ./workflows/full_tensor/run.sh accumulate \
  --calc-dir "$CALC" --data-dir "$DATA" \
  --start 500 --final all --chunk-size 100 --correlations distributed
```

Only accumulate supports multiple MPI ranks. Chunk size need not be divisible by
the thread count. Keep BLAS/OMP at one thread during distributed accumulation to
avoid oversubscription; run.sh sets this. The serial correlation alternative is
`--correlations serial`, which still uses collective parallel HAMSTER.

The next stages run once, without MPI duplication:

```bash
./workflows/full_tensor/run.sh d --calc-dir "$CALC" --data-dir "$DATA"
./workflows/full_tensor/run.sh phonons --calc-dir "$CALC" --data-dir "$DATA" --temperature 100
./workflows/full_tensor/run.sh unitary --calc-dir "$CALC" --data-dir "$DATA"
./workflows/full_tensor/run.sh g-orbital --calc-dir "$CALC" --data-dir "$DATA"
./workflows/full_tensor/run.sh g-band --calc-dir "$CALC" --data-dir "$DATA"
```

These produce, in DATA:

```text
sum_uH_primitive_labels.h5
displacement_covariance.h5
electron_lattice_D.h5
electron_lattice_d_kq.h5
phonon_modes_from_covariance.h5
unitary_transformation_matrices.h5
electron_phonon_g_kq.h5
electron_phonon_g_band_kq.h5
```

The full g files can be large. Preserve the primary displacement/coupling data
needed to reconstruct later stages. Do not run the same output stage twice at
once. Changing inputs/model/selection requires an explicit new run directory.
The accumulator's existing-output policy is :resume; review the constants and
input/checkpoint validation before deliberately restarting or overwriting.

For the original serial source use the reference instructions in
[reference_original_implementation/gaas_4x4x4](../reference_original_implementation/gaas_4x4x4/README.md);
the main workflow above is the preferred interface.

## 4. CsPbBr3 native MACE preparation and temperatures

For the supplied native dumps, LAMMPS types 1,2,3 are Pb,Cs,Br. Make this mapping
explicit. A different model may use a different order.

```bash
CALC=/path/to/inputs/CsPbBr3/ML_corrected/4x4x4/425K
DATA=/path/to/large-results/CsPbBr3/ML_corrected/4x4x4/425K
export EPH_LAMMPS_SPECIES=Pb,Cs,Br
./workflows/full_tensor/run.sh covariance \
  --calc-dir "$CALC" --data-dir "$DATA" \
  --trajectory /path/to/425K/positions.lammpstrj \
  --repetitions 4 4 4 --start 500 --final all --no-overwrite
```

The first native frame supplies SC_POSCAR and the primitive reference when an
explicit primitive is absent. It must support the specified primitive topology;
supply --primitive FILE if automatic extraction is inappropriate. The reader
requires a fixed periodic orthorhombic box and recognizes Cartesian or scaled
position columns; tilted/changing cells are rejected. Atom IDs/types are sorted
consistently. The code exports SC_XDATCAR for HAMSTER and native timesteps for
provenance. Preparation creates POSCAR_equlibr, POSCAR_equil_primitive and the
covariance on DATA and links them into CALC.

Then run accumulate/d/phonons/unitary/g-orbital/g-band as in section 3, with
`--temperature 425`. Repeat with independent directories and matching trajectories
for 525 K and 625 K. For 8x8x8 use `--repetitions 8 8 8` and its actual trajectory.
A generic LAMMPS input requires EPH_LAMMPS_SPECIES in its type order; species
counts must be multiples of the number of primitive cells.

For another material, supply its primitive topology, trajectory, model hconf and
all required model files. Set EPH_LAMMPS_SPECIES to the dump type order when using
LAMMPS. Supply a mass table containing every species with --masses FILE to both
the phonons and g-orbital stages. Band indices, Brillouin-zone paths and any spin
reduction in visualization are material/model-specific and must be selected
explicitly; the GaAs and CsPbBr3 defaults are examples, not universal indices.

## 5. GaAs 10x10x10: direct accumulation without full uH

### !!!!!!! THE LARGE-CELL BAND REFERENCE NEEDS AN UPDATE !!!!!!!

**THE CURRENT DIRECT CODE DOES NOT SAVE THE MD-AVERAGED HAMILTONIAN. THE COMMANDS
BELOW USE ONE HAMSTER HAMILTONIAN EVALUATED AT MD-AVERAGED POSITIONS FOR THE BANDS.
THIS IS NOT THE ENSEMBLE-MEAN HAMILTONIAN.** See the
[large-cell README](../workflows/large_cell/README.md) for the required updates
to the accumulator, checkpoint/output and electronic-reference selection. The
update is not implemented; these commands reproduce the current approximation.

First prepare covariance and thermal POSCARs using the same covariance stage:

```bash
CALC=/path/to/inputs/GaAs/TB_only/10x10x10/100K
DATA=/path/to/large-results/GaAs/TB_only/10x10x10/100K
./workflows/full_tensor/run.sh covariance \
  --calc-dir "$CALC" --data-dir "$DATA" \
  --trajectory "$DATA/SC_XDATCAR" --primitive "$CALC/POSCAR_primitive" \
  --start 500 --final all --no-overwrite
```

Copy `examples/GaAs/direct_10x10x10.toml` to a configuration file and set calc_dir,
data_dir, trajectory and model_files. Paths are relative to the TOML file. Add
ml_params.dat and every additional model input for an ML run. Choose chunk/window
sizes before running; accumulation_window must be divisible by hamster_chunk_size.
TOML is a text configuration file: it replaces editing path/settings constants
in a Julia driver and leaves the accumulation equations unchanged. The example
uses `model_files` relative to CALC; the other path fields are relative to the
TOML file itself. Give the new repository run its own DATA: direct checkpoint
signatures include absolute input/source paths and source hashes, so copied or
relocated hydrogen checkpoints are not automatically resumable from this copy.
Existing completed data can still be used for postprocessing with the required
files; do not bypass the direct driver's signature checks.

```bash
julia --project=. workflows/large_cell/run_direct_eph.jl /path/to/direct_run.toml
```

This driver is not MPI-aware: use `jrun -n 1` if scheduling it. fft_threads in the
TOML sets FFTW threads. HAMSTER threading is controlled separately by the external
executable/environment and should fit within the cores reserved for the job.
Outputs are DATA/.direct_eph_work/pair_*.h5 plus DATA/d_convergence.txt.
Physical d is sum_d divided by each file's number_of_snapshots. Keep all primary
pair files. Do not mix pair files from different windows/runs.

To finish with the established compact postprocessors, first calculate phonons,
then export a combined d file. The export is an additional potentially large
copy, so provision disk capacity for both it and the pair checkpoints:

```bash
julia --project=. postprocessing/compute_phonons_from_covariance.jl \
  --directory "$DATA" --temperature 100
python3 postprocessing/export_direct_pair_d.py "$DATA"
```

The exporter requires the completed pair snapshots to match the covariance set
and does not run HAMSTER or change the pair checkpoints. It writes
DATA/electron_lattice_d_kq.h5.

The direct workflow did not accumulate an ensemble-mean Hamiltonian. To reproduce
the completed GaAs large-cell workflow, compute one Hamiltonian at its thermal
mean positions, then construct its band basis. The helper now saves H only;
the following steps are explicit so you can choose either g output layout:

```bash
# Only when CALC/data is absent; DATA must already exist.
ln -s "$DATA" "$CALC/data"
julia --project=. postprocessing/compute_hamiltonian_at_average_positions.jl "$CALC"
julia --project=. postprocessing/unitary_transformation_matrix_construction.jl --directory "$DATA"
```

**A. Compact g for one initial electron k** (Gamma by default), all phonon q,
branches and electronic band pairs:

```bash
julia --project=. postprocessing/compute_electron_phonon_selected_k.jl "$DATA"
```

**B. Full g for every initial k and every q**, matching the full-tensor storage
choice used in the earlier TB calculation. Run these instead of A when that
output is needed; both use the same d, phonons and U:

```bash
julia --project=. postprocessing/compute_electron_phonon_g.jl --directory "$DATA"
julia --project=. postprocessing/transform_electron_phonon_g_to_band_basis.jl --directory "$DATA"
```

The static-reference helper resolves separate data through CALC/data or through
EPH_DATA_ROOT plus the material-relative layout. With arbitrary separate DATA,
the commands above create CALC/data as a symlink to DATA. If already prepared,
verify that reference rather than recreating it:

```bash
readlink -f "$CALC/data"
```

Create that reference before running the static helper. It evaluates H at the
averaged positions; it is not an ensemble average of H. For selected k other than
Gamma, which must coincide with an existing mesh point:

```bash
julia --project=. postprocessing/compute_electron_phonon_selected_k.jl "$DATA" --k-reduced 0.1 0 0
```

Gamma output is electron_phonon_g_selected_k.h5. It contains all q points,
branches and band pairs for that one initial electron k. This calculation writes
only the final selected-k result and does not create full orbital/band g files.
Do not use the optional --remove-full-g flag unless you intend its documented
cleanup of existing full-g outputs.

Route B writes `electron_phonon_g_kq.h5` and
`electron_phonon_g_band_kq.h5`. Route A writes only its selected-k result.
The large-cell d accumulation includes all mesh k,q in either case; only the
subsequent g output is restricted in A. Full g has the additional storage cost
that compact conversion avoids.

## 6. Visualization

Set material roots to YOUR input and data trees. The path identity is
METHOD/CELL/TEMPERATURE, e.g. ML_corrected/4x4x4/425K.

```bash
export GAAS_PROJECT_ROOT=/path/to/inputs/GaAs
export GAAS_DATA_ROOT=/path/to/large-results/GaAs
export CSPBBR3_PROJECT_ROOT=/path/to/inputs/CsPbBr3
export CSPBBR3_DATA_ROOT=/path/to/large-results/CsPbBr3
```

Heatmaps and signed differences:

```bash
python3 visualization/gaas/plot_species_orbital_d_heatmaps.py --calc-dir TB_only/4x4x4/100K
python3 visualization/gaas/plot_species_orbital_d_heatmaps.py \
  --calc-dir TB_only/4x4x4/400K --compare-to TB_only/4x4x4/100K
python3 visualization/gaas/plot_species_orbital_d_heatmaps.py \
  --calc-dir ML_corrected/4x4x4/100K --compare-to TB_only/4x4x4/100K
python3 visualization/cspbbr3/plot_species_orbital_d_heatmaps.py --calc-dir ML_corrected/4x4x4/425K
python3 visualization/cspbbr3/plot_species_orbital_d_heatmaps.py \
  --calc-dir ML_corrected/4x4x4/625K --compare-to ML_corrected/4x4x4/425K
```

Direct pair heatmaps work without exporting combined d: pass
`--d-file "$DATA/.direct_eph_work"`. --output-dir can select another destination.
The CsPbBr3 heatmap checks spin-block equivalence before reducing its display.

GaAs vs PERTURBO defaults include both temperatures, ML/TB and available 10-cell
results. Supply repeated --curve DIRECTORY LABEL to replace the defaults; source
directories with data/ references and explicit HDF5 paths are accepted.

```bash
python3 visualization/gaas/plot_g2_gamma_comparison_Perturbo.py --output-dir results/GaAs/path
python3 visualization/gaas/plot_g2_gamma_comparison_Perturbo.py \
  --curve "$GAAS_DATA_ROOT/TB_only/4x4x4/100K" "TB 100K 4x4x4" \
  --no-perturbo --output-dir results/GaAs/path
python3 visualization/cspbbr3/plot_g2_path_temperature_comparison.py --output-dir results/CsPbBr3/path
```

The CsPbBr3 default reads `CSPBBR3_DATA_ROOT`, uses the conduction pair 21/22,
initial R and the three 425/525/625 K curves. It expects the full band-g file
written by the full-tensor workflow. To select a subset or another destination,
use repeated `--curve "$CSPBBR3_DATA_ROOT/ML_corrected/4x4x4/425K" "425 K"`
arguments and `--output-dir`. Model band numbering and path assumptions must be revisited
for a different material. The GaAs single-band reference convention is retained.

A plotting example that needs no large HDF5 input:

```bash
python3 visualization/convergence/plot_convergence.py \
  --log-dir examples/GaAs/convergence_logs --output-dir results/GaAs/convergence
```

These legacy convergence logs compare different N_added; see METHODS before
interpreting the two temperatures together. New full accumulation runs can use
`export EPH_CONVERGENCE_M=50` with chunk size >=50 for a matching update size.
The direct workflow sets convergence_m in its TOML file.

All maintained plotters use SVG path lettering. For older SVG files with live
text, the provided outline utility preserves the original and writes a new file:

```bash
python3 visualization/outline_matplotlib_svg.py old.svg fixed.svg
```

## 7. GitHub preparation

Outputs/results/model weights are ignored by .gitignore. This source collection
does not include a test suite. To prepare your own repository later:

```bash
git init
git add .
git status --short
```

Review the staged source, choose publication metadata/license as appropriate,
then create your GitHub remote and push using your account. This collection does
not create a remote, commit, or upload automatically.
