# physionet-sleep-cognition

ML models that predict future cognitive impairment (MCI, Alzheimer's, dementia)
from multi-channel polysomnogram (PSG) recordings, using the
**PhysioNet/CinC Challenge 2026** dataset
("Screening for Cognitive Impairment During Sleep Studies").

## Identity

- **Lab:** SG Research
- **Lead Researcher:** Samantha Goncalves
- **GitHub Org:** https://github.com/sg-research
- **Repository:** https://github.com/sg-research/physionet-2026-sleep-cognition
- **PhysioNet Team Name:** SG Research

## Task

Binary classification — given a PSG + demographics, predict whether the
patient will receive a cognitive impairment diagnosis 3–7 years
post-recording. Primary metric: **AUROC** between Group 1 (future diagnosis
3–7y) and Group 2 (no diagnosis, 7+ years follow-up).

## Quickstart

```bash
# Install Python 3.12 dependencies
uv sync

# Activate the environment (optional — `uv run <cmd>` works without this)
source .venv/bin/activate

# Download the Challenge data from Kaggle (~130 GB)
uv run python scripts/download_kaggle_data.py
```

## Repo layout

```
src/physionet_cog/{data,features,models,training,evaluation,utils}/  # library
scripts/                                  # one-off entry points
notebooks/                                # exploratory only
configs/                                  # Hydra YAML
tests/                                    # pytest
submissions/                              # Challenge submission packages
data/{kaggle_raw,processed,external}/     # gitignored
```

Imports use the package prefix, e.g. `from physionet_cog.data import loaders`.

## Conventions

- Python 3.12; type hints required (mypy-clean).
- `uv` for all package management — never `pip` directly.
- `ruff format` + `ruff check` before committing.
- Hydra configs only; no hardcoded paths or magic numbers in `src/`.
- W&B for every experiment that produces a metric.
- **Subject-level CV always** — never random splits.
