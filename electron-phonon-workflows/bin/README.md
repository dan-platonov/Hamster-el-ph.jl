# MPI launcher

`mpiexecjl` is the shell entry point; it selects Julia's environment from
`EPH_JULIA_PROJECT`, then `HAMSTER_PROJECT`, then the repository root, and
executes `mpi_launch.jl`. The Julia helper uses `MPI.mpiexec()` from that
environment and forwards the launcher arguments. This keeps the launcher
consistent with MPI.jl used by HAMSTER instead of guessing a system MPI.

From the repository root:

```bash
export PATH="$PWD/bin:$PATH"
JULIA_NUM_THREADS=2 bin/mpiexecjl -n 2 ./workflows/full_tensor/run.sh accumulate \
  --calc-dir /path/to/inputs --data-dir /path/to/results --correlations distributed
```

Hydrogen's external `jrun` can find this wrapper through PATH when using multiple
ranks. Only the full-tensor accumulation stage supports MPI. The large-cell
direct driver and the other postprocessing stages must not be started as
independent MPI copies. These launchers do not reserve cores themselves;
allocation is the scheduler/jrun responsibility. See
[RUNNING](../docs/RUNNING.md) for complete commands.
