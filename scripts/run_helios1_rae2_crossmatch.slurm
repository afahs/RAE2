#!/bin/bash
#SBATCH --job-name=helios-rae2
#SBATCH --constraint=cpu
#SBATCH --qos=interactive
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=/pscratch/sd/a/afahs/RAE2/helios1_rae2_crossmatch_v1/slurm-%j.out

set -euo pipefail

module load python/3.12

CODE=/global/u2/a/afahs/RAE2/RyleVonberg
OUT=/pscratch/sd/a/afahs/RAE2/helios1_rae2_crossmatch_v1
RAW="$OUT/raw"
RUN1="$OUT/run1"
RUN2="$OUT/run2"
EPISODES=/pscratch/sd/a/afahs/RAE2/sun_rv_science_v1_work/noaa_low_frequency_episodes.csv
FINITE=/pscratch/sd/a/afahs/RAE2/sun_rv_external_occultation_v1/external_event_channel_metrics.csv
STRICT=/pscratch/sd/a/afahs/RAE2/sun_rv_external_occultation_strict_v1/external_event_channel_metrics.csv

mkdir -p "$RUN1" "$RUN2"

run_analysis() {
    local destination="$1"
    python "$CODE/scripts/run_helios1_rae2_crossmatch.py" \
        --helios "$OUT/helios_e5c_decoded.parquet" \
        --channel-quality "$OUT/helios_channel_quality.csv" \
        --episodes "$EPISODES" \
        --rae-finite "$FINITE" \
        --rae-strict "$STRICT" \
        --trajectory "$RAW/he1trj74.asc" "$RAW/he1trj75.asc" \
        --out-dir "$destination" \
        --seed 581204
}

run_analysis "$RUN1"
run_analysis "$RUN2"

tables=(
    helios_rae2_coverage_audit.csv
    helios_event_channel_metrics.csv
    helios_event_channel_metrics.parquet
    helios_channel_timing_summary.csv
    crossmission_frequency_pair_metrics.csv
    crossmission_day_summary.csv
    crossmission_event_support.csv
    helios_drift_fits.csv
)
for table in "${tables[@]}"; do
    cmp "$RUN1/$table" "$RUN2/$table"
done

cp -a "$RUN1"/. "$OUT"/
printf 'deterministic tabular rerun passed\n'
