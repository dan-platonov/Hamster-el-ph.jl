# Source selection and portability changes

Captured from the user's working hydrogen projects. source_inventory.json records
source-directory identities and SHA-256 hashes before adaptation. The local
convergence script and user-provided outline utility were collected separately.
No production calculation or visualization source on hydrogen was modified.

- Common basis/loading/regression modules retain the working CsPbBr3 versions.
- Main full-tensor numerical stages retain the successful shared implementation.
  Calculation/data/work paths are explicit; default data is CALC/data.
- Native LAMMPS type symbols are set through EPH_LAMMPS_SPECIES. The original
  CsPbBr3 stoichiometry guard was generalized to primitive-cell multiplicities.
  Species ordering, fixed-box checks and mapping validation remain required.
- The corrected Gamma covariance projection-before-inversion is retained in the
  main and standalone phonon stages. The archival GaAs reference is historical.
- DirectFourierEph, DirectFourierKernel and DirectDConvergence are copied without
  numerical edits. Their configuration is now a portable TOML launcher.
- Static average-position H construction stops after saving the reference rather
  than launching unitary/g/plotting implicitly. Model files remain external.
- Single selected-k and full g conversion are both retained.
- Python project/data roots are configurable. SVG lettering is outlined to avoid
  PowerPoint font/tspan layout problems. Existing matrix orientation is retained.
- Fitted model weights, trajectories, caches, partial drafts, old fitted plotting
  variants, machine-specific jrun code and generated large data are excluded.

Dependency installation, model licensing and full external-dataset reruns remain
specific to the user's environment. A repository license has not been assigned.

Final repository review clarified that TOML configuration and separately
invoked H/U/g/plot stages adapt the existing workflow to separate paths; they
do not replace its numerical method. The static-H help text now describes that
separation. CsPbBr3's path plot now reads its documented CSPBBR3_DATA_ROOT for
default curves and relative curve lookup. Its quantity and styling are unchanged.
