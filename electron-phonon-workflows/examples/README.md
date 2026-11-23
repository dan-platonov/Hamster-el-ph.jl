# Input-directory templates

These folders contain small reference structures, hconf examples and one
large-cell TOML configuration. They contain no fitted models or trajectories
and are not completed calculations. Follow [RUNNING](../docs/RUNNING.md).

- `GaAs/TB_only/4x4x4/{100K,400K}` and `GaAs/ML_corrected/4x4x4/{100K,400K}`:
  primitive topology and the corresponding type of hconf. Copy a matching
  `hconf.example` to `hconf` in your CALC directory and supply the model files,
  actual supercell structure and trajectory.
- `CsPbBr3/ML_corrected/4x4x4/{425K,525K,625K}`: SOC/ML/Wolf hconf examples
  and primitive topology for native MACE preparation. These model settings
  belong to their fit; they are not generic settings for every perovskite.
- `GaAs/*/10x10x10/100K/INPUTS.txt`: input reminders for the large-cell runs.
  Use the matching GaAs primitive template and an actual 10x10x10 trajectory.
- `GaAs/direct_10x10x10.toml`: editable direct-accumulation settings. Paths
  are relative to the TOML file, except model-file names relative to CALC.
  Add the ML parameter file to `model_files` for an ML run.
- `GaAs/convergence_logs/`: small previously obtained logs for the convergence
  plotter. They are example results, not a test suite or production input.

Preparation generates the finite-temperature equilibrium structures from the
chosen trajectory. Never substitute another temperature's equilibrium POSCAR
because its primitive topology happens to match. Use separate CALC/DATA
directories for different models, cells and temperatures.
