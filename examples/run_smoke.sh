#!/usr/bin/env bash
set -euo pipefail

: "${RAE2_MASTER_CSV:?Set RAE2_MASTER_CSV to the local RAE-2 master CSV path.}"

python -m rylevonberg.pipeline run-smoke \
  --data "$RAE2_MASTER_CSV" \
  --start "1973-10-03 04:00:00" \
  --end "1973-10-03 06:00:00" \
  --output-dir "${RAE2_OUTPUT_DIR:-outputs}/smoke_jupiter" \
  --source jupiter \
  --frequency 4 \
  --antenna rv2_coarse \
  --run-mode local

