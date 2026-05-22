# CLAUDE.md

> Persistent context for Claude Code. Read this first every session.

## Project Identity

- **Lab:** SG Research
- **Lead Researcher:** Samantha Goncalves
- **GitHub Org:** https://github.com/sg-research
- **Repository:** https://github.com/sg-research/physionet-2026-sleep-cognition
- **Working Directory:** `~/projects/physionet-sleep-cognition`
- **PhysioNet Team Name:** `Masam` (if Masoud co-submits) or `SG Research` (solo)

## Project Goal

Develop ML models that predict future cognitive impairment (MCI, Alzheimer's, dementia) from multi-channel polysomnogram (PSG) recordings, using the **PhysioNet/CinC Challenge 2026** dataset ("Screening for Cognitive Impairment During Sleep Studies").

- **Task:** Binary classification — given a PSG + demographics, predict whether patient will receive cognitive impairment diagnosis 3–7 years post-recording
- **Metric:** AUROC between Group 1 (future diagnosis 3–7y) and Group 2 (no diagnosis, 7+ years follow-up)

## Challenge Context (CRITICAL — read this)

This project uses the PhysioNet 2026 Challenge data as the substrate for a research project. Competition outcomes are limited because we entered late:

- **Challenge data:** Publicly available on Kaggle (no credentialing needed)
- **Phase status:** Currently in hiatus; official phase begins late May 2026
- **Prize eligibility:** ✗ Not eligible — missed unofficial phase + Apr 15 abstract deadline
- **Official ranking:** ✗ Not eligible — requires accepted CinC abstract
- **Wild card eligibility:** ✗ Not eligible — wild card requires having participated in unofficial phase
- **CinC presentation:** ✗ Not possible without accepted abstract
- **Publication restriction:** Cannot submit analysis of Challenge data to other venues until after CinC 2026 ends (~Oct 2026)

## What Remains Achievable

- ✓ **Leaderboard score visible during competition** — public, alongside other teams
- ✓ **Open-source code release** after Challenge ends
- ✓ **CS229 final project** (independent of Challenge outcome)
- ✓ **arXiv preprint** after October 2026 (when embargo lifts)
- ✓ **ML4H 2026** submission (~Sept 2026 deadline)
- ✓ **TS4H 2026 workshop** at NeurIPS (~Sept-Nov 2026 deadline)
- ✓ **ML4H 2027 / MLHC 2027 / journal paper** as the substantial publication target

## Data Sources

### Primary (publicly available, no credentialing)
- **PhysioNet 2026 Challenge data on Kaggle** (`physionet/physionetchallenge2026data`)
  - ~80 GB total
  - 622 training PSGs (~4,696 hours) from 3 sites (S0001, I0002, I0006)
  - Hidden validation set (I0004) and test set (I0007) — different sites
  - Includes: raw EDF signals, CAISR algorithmic annotations, human annotations, demographics, `Cognitive_Impairment` labels
  - Modalities: EEG, EOG, EMG, ECG, respiratory effort, airflow

### Secondary (pending credentialed access; applications submitted)
- **BDSP datasets:**
  - Ye et al. "Dementia detection from brain activity during sleep" (2023)
  - Ye et al. "Sleep EEG-Based Brain Age Index With Dementia" (2020)
  - Meulenbrugge et al. "Ordinal Sleep Depth" (2025)
  - Stanford Sleep Bench (sleep FM training data)
  - Harvard EEG Database (large EEG corpus for pre-training)
  - Human Sleep Project (full resolution)
  - Wei et al. MCI/AD EHR phenotyping (2025)
- **PhysioNet datasets:**
  - SHHS, Sleep-EDF (& Expanded), MIT-BIH PSG, CAP Sleep, Haaglanden MC, NCH Sleep DataBank, CPS Dataset, I-CARE

## Key Methodological Decisions

(Locked in from NeurIPS 2025 TS4H workshop + ML4H 2025 literature review)

1. **Compare EEG-specific vs generic TSFMs** — CBraMod (Wang 2025, current EEG SOTA) and LaBraM (Jiang 2024) vs Mantis (Feofanov 2025, generic TSFM outperforming EEG-specific in benchmarks)
2. **Channel-tokenization** for heterogeneous PSG montages across sites
3. **HiMAE-style multi-resolution SSL** — investigate which temporal resolution carries cognitive-decline signal
4. **Subject-aware contrastive pre-training** for inter-subject robustness
5. **MoE survival heads** for 3–7 year time-to-event with calibrated uncertainty
6. **Site-shift causal representation learning** for multi-center robustness (critical: val/test sites differ from train)
7. **Apple-resonant method:** Speech FM transfer (Wav2Vec2 on PSG, per Narain et al. Apple NeurIPS TS4H paper)
8. **Conformal prediction** for distribution-free uncertainty quantification
9. **Subject-level cross-validation** (never random splits)
10. **xMADD diffusion-based waveform synthesis** (Friedman et al. ML4H 2025) for augmentation

## Repo Structure

```
physionet-2026-sleep-cognition/
├── CLAUDE.md
├── README.md
├── PROJECT_BRIEF.md
├── pyproject.toml          # uv-managed
├── .gitignore
├── data/                   # gitignored, never commit
│   ├── kaggle_raw/         # challenge data from Kaggle
│   ├── processed/          # preprocessed features
│   └── external/           # SHHS, Sleep-EDF, etc. when credentialed
├── src/
│   ├── data/               # EDF loading, BIDS parsing, label extraction
│   ├── features/           # signal processing, tokenization, channel handling
│   ├── models/             # CBraMod, Mantis adapters, custom architectures
│   ├── training/           # PyTorch Lightning modules, SSL pre-training
│   ├── evaluation/         # AUROC, conformal prediction, calibration
│   └── utils/              # logging, seeding, config helpers
├── scripts/                # one-off scripts (e.g., download_kaggle_data.py)
├── notebooks/              # exploratory analysis only
├── configs/                # Hydra YAML configs
├── tests/                  # pytest
└── submissions/            # PhysioNet submission packages (each runs end-to-end)
```

## Code Conventions

- **Python:** 3.12, type hints required (mypy-clean)
- **Package management:** `uv` only — never `pip` directly
- **Linting/formatting:** `ruff` (format + check), 88 char line limit
- **Logging:** `structlog` for structured logs, never `print` in src/
- **Configs:** Hydra YAML, no hardcoded paths or magic numbers in src/
- **Training:** PyTorch Lightning for scaffolding
- **Experiment tracking:** W&B for everything that produces a metric
- **Reproducibility:** seed everywhere, log lib versions, save Hydra config with each checkpoint
- **Testing:** pytest; minimum smoke tests for data loading and model forward passes
- **Subject-level CV:** ALWAYS — never random splits, no leakage between patients

## Compute Setup

- **Primary:** M4 Mac with MPS backend for PyTorch
- **Cloud (when needed):** Lambda Labs or Vast.ai spot instances, ONLY for foundation model fine-tuning
- **Storage:** Local SSD for active data; never upload patient data to consumer cloud
- **Budget cap:** $400 total cloud spend

## Timeline

| Date | Milestone |
|---|---|
| **Late May 2026** | Official phase opens — get baseline submission to leaderboard ASAP |
| **June 2026** | CS229 starts; iterate on models |
| **Late Aug 2026** | Official phase closes; max 10 official-phase entries |
| **Aug 2026** | CS229 final report due |
| **~Sept 2026** | ML4H 2026 deadline (target with embargo-compatible methods angle) |
| **~Sept-Nov 2026** | TS4H 2026 workshop deadline |
| **Sep 20–23, 2026** | CinC 2026 Madrid (not attending — no abstract) |
| **Late Sep 2026** | Final scores released |
| **Early Oct 2026** | **Publication restriction lifts** — free to submit Challenge analysis elsewhere |
| **Oct 2026 onwards** | arXiv preprint; submissions to journals/other venues |
| **Feb 2027** | CHIL 2027 deadline |
| **April 2027** | MLHC 2027 deadline |
| **~Sept 2027** | ML4H 2027 deadline (substantial methods paper target) |

## Deliverables

| Deliverable | Status / Timing |
|---|---|
| Public leaderboard ranking | Primary — pursue actively (May-Aug 2026) |
| CS229 final project report | Required (August 2026) |
| Open-source code release | After Challenge ends (Oct 2026) |
| arXiv preprint | After Oct 2026 |
| ML4H 2026 paper | Sept 2026 if methods are ready & embargo-compatible |
| ML4H 2027 / MLHC 2027 paper | Primary publication target |

## Reading List (Priority Order)

1. **CBraMod** (Wang et al. 2025) — current EEG foundation model SOTA
2. **Mantis** (Feofanov et al. 2025) — generic TSFM beating EEG-specific
3. **HiMAE** — multi-resolution self-supervised pre-training
4. **LaBraM** (Jiang et al. 2024) — large brain model for EEG
5. **xMADD** (Friedman et al. ML4H 2025) — diffusion-based waveform synthesis
6. **TFM-Tokenizer** — time-series foundation model tokenization
7. **Segment-Then-Connect** — MCI detection from EEG
8. **Subject-Aware Contrastive** — inter-subject robustness for biosignals
9. **Apple Speech FMs** (Narain et al. NeurIPS TS4H 2025) — Wav2Vec2 transfer to physiological signals
10. **MoE Survival Heads** — time-to-event modeling
11. **Comprehensive Biosignal FM Review** — landscape paper
12. **CAISR paper** (Nasiri et al. 2025) — the framework producing our labels

## Claude Code Guidance

- **Read this file at the start of every session.**
- **Stay focused on the current task** — don't broaden scope. If you think we should pivot, ask.
- **Use `uv` for all package management** — never `pip install` directly.
- **Update this file when significant decisions are made.**
- **Don't commit data files** — `.gitignore` everything in `data/`.
- **Use Hydra configs**, not hardcoded paths or constants.
- **Log all experiments to W&B.**
- **Write smoke tests** for new data loading or model forward passes.
- **Run `ruff format` and `ruff check`** before committing.
- **Subject-level CV always** — never random splits.
- **No data leakage** — strictly separate train/val/test sources.
- **Reproducibility:** every experiment has a config, seed, and W&B run ID.
