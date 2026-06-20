# Usage

Run commands from the project root after installing with `pip install -e .`.

## Smoke Run

```bash
export RAE2_MASTER_CSV=/path/to/interpolatedRAE2MasterFile.csv

python -m rylevonberg.pipeline run-smoke \
  --data "$RAE2_MASTER_CSV" \
  --start "1973-10-03 04:00:00" \
  --end "1973-10-03 06:00:00" \
  --output-dir outputs/smoke_jupiter \
  --source jupiter \
  --frequency 4 \
  --antenna rv2_coarse \
  --run-mode local
```

## Staged Run

```bash
python -m rylevonberg.pipeline ingest \
  --data "$RAE2_MASTER_CSV" \
  --start "1974-11-01 00:00:00" \
  --end "1976-01-02 00:00:00" \
  --output-dir outputs/example/01_ingest \
  --value-columns rv1_coarse rv2_coarse

python -m rylevonberg.pipeline predict \
  --cleaned outputs/example/01_ingest/cleaned_timeseries.csv \
  --sources configs/bright_sources.csv \
  --source earth sun \
  --frequency 1 2 3 4 5 6 7 8 9 \
  --antenna rv1_coarse rv2_coarse \
  --output-dir outputs/example/02_events \
  --frame fk4 \
  --equinox B1950

python -m rylevonberg.pipeline detect \
  --cleaned outputs/example/01_ingest/cleaned_timeseries.csv \
  --events outputs/example/02_events/predicted_events.csv \
  --output-dir outputs/example/03_detection \
  --window-seconds 600 \
  --plots

python -m rylevonberg.pipeline stack \
  --cleaned outputs/example/01_ingest/cleaned_timeseries.csv \
  --events outputs/example/02_events/predicted_events.csv \
  --output-dir outputs/example/04_stack \
  --window-seconds 600 \
  --bin-seconds 60 \
  --plots

python -m rylevonberg.pipeline validate \
  --cleaned outputs/example/01_ingest/cleaned_timeseries.csv \
  --events outputs/example/02_events/predicted_events.csv \
  --output-dir outputs/example/05_validation

python -m rylevonberg.pipeline score-detections \
  --cleaned outputs/example/01_ingest/cleaned_timeseries.csv \
  --stepfit outputs/example/03_detection/per_event_stepfit_detections.csv \
  --matched outputs/example/03_detection/per_event_matched_filter.csv \
  --control-events outputs/example/05_validation/negative_control_event_ensemble.csv \
  --injection-grid outputs/example/05_validation/injection_recovery_grid.csv \
  --output-dir outputs/example/06_scoring
```

## Interactive HPC Use

The package has an explicit guard against batch-mode execution. On an HPC
system, request an interactive allocation first:

```bash
salloc --qos=interactive --nodes=1 --time=02:00:00
source .venv/bin/activate
python -m rylevonberg.pipeline run-smoke \
  --data "$RAE2_MASTER_CSV" \
  --output-dir outputs/smoke_interactive \
  --run-mode interactive
```

