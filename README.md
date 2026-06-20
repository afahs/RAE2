# RAE-2 Ryle-Vonberg Pipeline

Python tools for RAE-2 / Explorer 49 Ryle-Vonberg lunar-occultation
source detection and validation.

This repository contains code, tests, configs, and usage notes. It does not
include mission data or generated analysis results. Runtime products are
written under `outputs/`, which is ignored by Git.

## What This Does

- Ingest a RAE-2 master CSV into a long time-series table.
- Predict lunar-limb disappearance and reappearance events for fixed sources
  and moving bodies.
- Run local step fits, matched filters, event-aligned stacks, blind searches,
  negative controls, and injection-recovery checks.
- Score detections against empirical controls and write reproducible tables
  and plots.

The main package is `rylevonberg/`. Command-line entry points live in
`rylevonberg.pipeline`, with exploratory and paper-support scripts in
`scripts/`.

## Install

Use Python 3.10 or newer. Check this first, especially on HPC login nodes
where `python` may be Python 2.7 and `python3` may still be too old:

```bash
python --version
```

If that command does not report Python 3.10 or newer, activate or create a
newer environment first. For example, with conda:

```bash
conda create -n rae2-rylevonberg python=3.12
conda activate rae2-rylevonberg
```

Then install the package from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Data

The pipeline expects a local RAE-2 master CSV. See
`docs/data_requirements.md` for the required columns and assumptions.

Example environment variable:

```bash
export RAE2_MASTER_CSV=/path/to/interpolatedRAE2MasterFile.csv
```

## Quick Smoke Run

```bash
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

The same command is available as a script after installation:

```bash
rv-pipeline run-smoke \
  --data "$RAE2_MASTER_CSV" \
  --start "1973-10-03 04:00:00" \
  --end "1973-10-03 06:00:00" \
  --output-dir outputs/smoke_jupiter \
  --source jupiter \
  --frequency 4 \
  --antenna rv2_coarse \
  --run-mode local
```

On HPC systems, request an interactive allocation first and use
`--run-mode interactive` or `--run-mode srun-interactive`. The code rejects
explicit batch modes such as `batch`, `sbatch`, and `qsub`.

## Staged Workflow

The smoke command runs all major stages. For larger workflows, stages can be
run separately:

```bash
python -m rylevonberg.pipeline ingest \
  --data "$RAE2_MASTER_CSV" \
  --start "1974-11-01 00:00:00" \
  --end "1976-01-02 00:00:00" \
  --output-dir outputs/example/01_ingest

python -m rylevonberg.pipeline predict \
  --cleaned outputs/example/01_ingest/cleaned_timeseries.csv \
  --sources configs/bright_sources.csv \
  --source earth sun \
  --frequency 1 2 3 4 5 6 7 8 9 \
  --antenna rv1_coarse rv2_coarse \
  --output-dir outputs/example/02_events
```

See `docs/usage.md` for the full staged command sequence.

## Scripts

`scripts/` contains higher-level analyses built on top of the package. Some
scripts need extra local assets such as digitized antenna beam cuts, PySM sky
maps, Mie forward-model tables, or burst-receiver CSVs. These assets are not
included. See `docs/script_inventory.md` before running the exploratory
scripts.

## Tests

```bash
python -m pytest -q
```

The tests use synthetic and small in-memory data. They do not require the
mission master CSV.

## Repository Hygiene

The intended GitHub upload includes source code, configs, docs, examples, and
tests only. Do not upload:

- `outputs/`
- raw mission data or local derived data
- cache directories such as `__pycache__/` and `.pytest_cache/`
- cleanup inventories or result reports under `reports/`
- local planning notes under `planning_text/`
