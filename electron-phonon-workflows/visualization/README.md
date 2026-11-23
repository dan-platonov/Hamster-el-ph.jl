# Visualization scripts and plotted quantities

Use Python >=3.10, install `requirements.txt`, and run commands from the repository root. New plots
are SVG with path-based lettering for reliable PowerPoint import. Font/tick/
legend sizes are defined near the beginning of the corresponding plotting
script. Plots and associated numerical/source records are outputs, not model
inputs. Source/data location commands are in [RUNNING](../docs/RUNNING.md).

## Source and result roots

```bash
export GAAS_PROJECT_ROOT=/path/to/inputs/GaAs
export GAAS_DATA_ROOT=/path/to/large-results/GaAs
export CSPBBR3_PROJECT_ROOT=/path/to/inputs/CsPbBr3
export CSPBBR3_DATA_ROOT=/path/to/large-results/CsPbBr3
```

The heatmap interface identifies `METHOD/CELL/TEMPERATURE` below these roots,
e.g. `TB_only/4x4x4/100K` or `ML_corrected/4x4x4/425K`. Use `--calc-dir`, or its
method/cell/temperature arguments, to select the calculation. `--output-dir`
sets a plot destination. Use `--help` on the individual script for all options.

## Orbital/displacement heatmaps

Both `gaas/plot_species_orbital_d_heatmaps.py` and
`cspbbr3/plot_species_orbital_d_heatmaps.py` plot the species-X amplitude

\[
 A^X_{\mu\nu}=\left[\frac{N_c}{N_kN_q}
 \sum_{\alpha\in X}\sum_{c,k,q}|d^{\alpha c}_{\mu\nu}(k,q)|^2\right]^{1/2}
 \quad[\mathrm{eV}/\mathrm{\AA}].
\]

The default `--normalization primitive-cell` removes the stored finite-cell
factor from d; k,q magnitudes are averaged, while Cartesian directions and
primitive atoms of that species are summed. It is an effective sensitivity of
orbital Hamiltonian elements to displacements, including hopping and orbital
mixing, not the equilibrium Hamiltonian or a signed derivative. It need not
be largest on the diagonal. Species with different atom multiplicities have
different numbers of terms in this sum. Removing the explicit cell factor does
not guarantee cell-size convergence of the sampled result.

`--compare-to REFERENCE` plots **A_selected minus A_reference**; the square root
is taken before subtraction. This differs from the norm of a complex tensor
difference. `--normalization finite-supercell` retains the stored cell factor;
`--no-average-kq` uses a mesh-dependent sum. Do not mix these options between
curves intended for quantitative comparison.

```bash
python3 visualization/gaas/plot_species_orbital_d_heatmaps.py \
  --calc-dir TB_only/4x4x4/100K
python3 visualization/gaas/plot_species_orbital_d_heatmaps.py \
  --calc-dir ML_corrected/4x4x4/100K --compare-to TB_only/4x4x4/100K
python3 visualization/cspbbr3/plot_species_orbital_d_heatmaps.py \
  --calc-dir ML_corrected/4x4x4/625K --compare-to ML_corrected/4x4x4/425K
```

Both read combined d, or direct pair data when supplied with
`--d-file /path/to/DATA/.direct_eph_work`. They save per-species SVG, numerical
matrices, orbital-label tables and `visualization_sources.json`. Comparison
directories are linked from the participating source calculations.

The GaAs orbital labels are Ga[i]/As[i] in the model's primitive local-orbital
order. CsPbBr3 labels use the model's species and s,p_x,p_y,p_z ordering. The Cs
heatmap checks negligible spin-flip amplitude and equality of the two same-spin
blocks before displaying one representative block. This is a plotting reduction
with no factor of two; all spin states remain in the coupling calculations.
Rows increase upwards: the true mu=nu matrix diagonal runs bottom-left to
top-right. Read the saved primitive-orbital table for an explicit mapping.

`cspbbr3/generate_all_heatmaps.py` produces single-temperature heatmaps and every
higher-temperature minus lower-temperature pair. It caches the small amplitude
matrices, not full d tensors. The default discovery expects completed combined
d files in the data tree; use the individual heatmap command for direct pairs.

```bash
python3 visualization/cspbbr3/generate_all_heatmaps.py \
  --method ML_corrected --cell 4x4x4 --temperatures 425K 525K 625K
```

The two `gaas/visualization_common.py` and `cspbbr3/visualization_common.py` files
implement material-specific root/path resolution, calculation discovery,
identity labels and comparison symlinks. They are supporting modules; running
them directly lists available calculations. They do not compute d or g.

## GaAs path comparison with PERTURBO

`gaas/plot_g2_gamma_comparison_Perturbo.py` is the main multi-curve comparison.
For initial electron k=Gamma and band n=5 it plots

\[
 G_n(q)=10^6N_c\sum_s|g^{\mathrm{stored}}_{nn,s}(\Gamma,q)|^2
 \quad[\mathrm{meV}^2]
\]

along L-Gamma-X-W-K-Gamma, using the actual commensurate q samples and connecting
successive available path points. The default includes available GaAs TB/ML,
4x4x4 and 10x10x10 results. PERTURBO uses the supplied two-column reference CSV.
The author-supplied documentation source and attribution details are in the
[reference-data README](../reference_data/PERTURBO/README.md).
Repeated `--curve DIRECTORY LABEL` replace the calculation defaults;
`--no-perturbo` omits the reference. `--band` changes the one-based band index.
The default primitive-cell normalization is checked from the source conventions;
PERTURBO is not rescaled by the calculation's Nc. Outputs include SVG, per-curve
CSV and a JSON source record.

```bash
python3 visualization/gaas/plot_g2_gamma_comparison_Perturbo.py \
  --output-dir results/GaAs/coupling_path_in_BZ
```

`gaas/plot_my_g2_gamma.py` supplies the original single-run reader/path analysis
used by the comparison, plus optional branch-resolved plotting and periodic
trilinear visualization. It exports sampled/path CSV and a text summary. Its
standalone default is the stored finite-supercell g, with **no automatic Nc
correction**; `--g-scale` is an explicit amplitude multiplier. Prefer the
multi-curve comparison for normalized size comparisons. Its optional raw
PERTURBO `gaas.ephmat`/YAML reader requires **PyYAML** (`pip install pyyaml`);
the main CSV-based comparison does not. Interpolation is not a new electronic
or phonon calculation and does not recover a polar small-q divergence.

## CsPbBr3 conduction-pair path comparison

`cspbbr3/plot_g2_path_temperature_comparison.py` uses initial k=R and path
Gamma-X-M-Gamma-R-X. It plots

\[
 G_C(q)=\frac{10^6N_c}{2}\sum_{n\in\{21,22\}}
 \sum_{m\in\{21,22\}}\sum_s|g^{\mathrm{stored}}_{mn,s}(R,q)|^2
 \quad[\mathrm{meV}^2].
\]

It averages the two initial conduction states, sums the two final states and all
branches. This is invariant under rotations within the selected two-dimensional
subspaces. The default 425/525/625 K curves are read from `CSPBBR3_DATA_ROOT` and use
progressively darker blue. Input can be a DATA directory, a CALC directory with
a data/ reference, or an explicit full band-g HDF5 file. This plot requires the
full band-g layout; it does not read compact selected-k files.
`--curve DIRECTORY LABEL`, `--band` (first of two consecutive bands),
`--k-reduced` and `--output-dir` customize it. The default band numbering belongs
to the collected SOC model, not to every perovskite. It saves SVG, CSV and JSON.

Both path plotters omit exact phonon Gamma from the all-branch curve because
the three translational modes have undefined oscillator normalization there.
They do not extrapolate to q=0, add a Born-charge/dielectric polar correction,
average over electron k, or apply Bose factors. Optical-only Gamma data are a
different quantity. A finite coarse-mesh curve alone cannot establish polaron
formation. See [METHODS](../docs/METHODS.md) for the physical limits.

## Convergence plots and SVG repair

`convergence/plot_convergence.py` reads `uH_convergence*.txt` or
`d_convergence*.txt` and draws the logged tensor norm (column 3) and full
normalized cumulative change (column 7), in blue without a legend:

\[
 \|B_N\|_F,\qquad
 r_{N,P}=\frac{\|B_N-B_P\|_F}{\max(\|B_N\|_F,\|B_P\|_F)}.
\]

For direct d logs replace B by d. N_previous and N_added identify the actual
comparison; adjacent plotted rows need not be the two tensors compared. The
example GaAs logs have different increments between temperatures, so their
relative changes cannot be directly interpreted as relative sampling quality.

```bash
python3 visualization/convergence/plot_convergence.py \
  --log-dir examples/GaAs/convergence_logs --output-dir results/GaAs/convergence
python3 visualization/outline_matplotlib_svg.py old.svg fixed.svg
```

`outline_matplotlib_svg.py` repairs older Matplotlib SVGs containing live text
by writing outlined text paths, using lxml/Matplotlib. It requires a separate
output filename and preserves the original. The repair is tailored to the
supplied Matplotlib DejaVu Sans/tspan layout, not an arbitrary SVG converter.
New maintained plotters already save path lettering and do not need that step.
