#!/bin/sh
set -eu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
task_code_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
task_repo_dir=$(CDPATH= cd -- "$task_code_dir/../.." && pwd)
task_project=${EPH_JULIA_PROJECT:-${HAMSTER_PROJECT:-$task_repo_dir}}
case "${1:-}" in
  accumulate)
    shift
    exec julia --startup-file=no --project="$task_project" "$task_code_dir/accumulate_uH_parallel.jl" "$@" ;;
  *)
    if [ "${PMI_SIZE:-${OMPI_COMM_WORLD_SIZE:-1}}" -gt 1 ]; then
      echo "Only accumulate supports multiple MPI ranks." >&2
      exit 1
    fi
    exec julia --startup-file=no --project="$task_project" "$task_code_dir/run_stage.jl" "$@" ;;
esac
