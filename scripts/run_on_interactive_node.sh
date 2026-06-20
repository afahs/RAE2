#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "batch" || "${1:-}" == "sbatch" || "${1:-}" == "qsub" ]]; then
  echo "Refusing non-interactive run mode. Use an interactive allocation." >&2
  exit 2
fi

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  exec salloc --qos=interactive --constraint="${RV_CONSTRAINT:-cpu}" --nodes=1 --time="${RV_TIME:-02:00:00}" "$@"
fi

exec "$@"
