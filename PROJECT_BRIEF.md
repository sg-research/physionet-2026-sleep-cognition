# PROJECT_BRIEF.md

> One-page outward-facing brief for SG Research's PhysioNet 2026 Challenge entry.

## One-line Description

Machine learning models that predict future cognitive impairment (MCI, Alzheimer's, dementia) from multi-channel polysomnogram recordings, using the PhysioNet/Computing in Cardiology Challenge 2026 dataset as a high-quality research substrate.

## Project Identity

- **Lab:** SG Research (independent research entity)
- **Lead:** Samantha Goncalves
- **Repository:** github.com/sg-research/physionet-2026-sleep-cognition
- **Challenge team name:** Masam (if collaborative) or SG Research (solo)

## Why It Matters

Sleep is a uniquely accessible window into brain health. Polysomnograms (PSGs) are routinely collected in clinical sleep labs and contain rich multi-channel signals (EEG, ECG, EOG, EMG, respiratory effort, airflow) that reflect underlying neurological function. If we can predict cognitive decline 3–7 years before clinical diagnosis from a single overnight recording, we open a path to earlier intervention — when treatments are most likely to be effective and when patients and families have time to plan.

## Scientific Contribution

The project addresses three open methodological questions in biosignal ML:

1. **Are EEG-specific foundation models actually better than generic time-series foundation models for clinical prediction tasks?** Mantis (Feofanov 2025) recently outperformed EEG-specific models on EEG benchmarks — surprising and unsettled. We will test this on a high-stakes clinical outcome.

2. **At what temporal resolution does the cognitive-decline signal live?** Multi-resolution self-supervised pre-training (HiMAE-style) lets us interrogate which time scales — micro-arousal, sleep stage, full-night architecture — carry predictive information for long-horizon outcomes.

3. **How do we handle multi-site clinical heterogeneity?** Validation and test sets come from different sites than training. Channel-tokenization + site-shift causal representation learning are the most credible recent approaches.

## Data

- **Primary:** PhysioNet 2026 Challenge data on Kaggle — 622 PSGs (~4,696 hours) from BIDMC, MGB, and Stanford-affiliated sites
- **Secondary (pending):** SHHS, Sleep-EDF, BDSP datasets (Ye et al. sleep-dementia work, Stanford Sleep Bench, Harvard EEG Database) for pre-training and cross-cohort validation

## Methods

- Foundation model comparison: CBraMod, LaBraM, Mantis adapted to PSG
- Channel-tokenization for heterogeneous montages
- Subject-aware contrastive pre-training
- MoE survival heads for 3–7 year time-to-event prediction
- Conformal prediction for calibrated uncertainty
- xMADD diffusion-based waveform augmentation

## Evaluation

- **Primary metric:** AUROC on hidden test set (Challenge scoring)
- **Secondary:** Calibration, conformal interval coverage, site-stratified performance, ablations across temporal resolutions and foundation model backbones

## Deliverables

| Output | Timeline |
|---|---|
| PhysioNet leaderboard submission | Late May – Aug 2026 |
| CS229 final project report | August 2026 |
| Public GitHub repository | Continuous |
| Open-source code release (with Challenge) | October 2026 |
| arXiv preprint | After Oct 2026 (publication restriction lifts) |
| ML4H 2026 paper | If embargo-compatible angle works (~Sept 2026) |
| ML4H 2027 / MLHC 2027 / journal | Primary publication target, ~2027 |

## Timeline

- **Late May 2026:** Official phase opens — baseline submission target
- **June–August 2026:** Model development concurrent with Stanford CS229
- **Late August 2026:** Official phase closes
- **September 2026:** CinC 2026 (Madrid) — not attending (no abstract)
- **Early October 2026:** Challenge ends; publication restriction lifts
- **2027:** Extended methods paper to ML4H, MLHC, or journal venue

## Career Relevance

This project sits at the intersection of three areas that matter for ML scientist roles in health tech:

- **Time-series foundation models on physiological signals** — directly aligned with Apple Health Technologies, Verily, Beacon
- **Multi-site clinical ML with calibrated uncertainty** — relevant for any biotech/health-tech ML role
- **Clinical translation discipline** — leveraging prior clinical research background in a quantitative role

## Honest Status

This is a research project using high-quality Challenge data rather than a competition entry pursuing prizes:

- ✗ Not eligible for prizes, rankings, wild card, or CinC presentation (missed unofficial phase + abstract deadline)
- ✓ Eligible for public leaderboard scoring during competition
- ✓ Code becomes publicly available after Challenge ends
- ✓ Methods publishable independently after October 2026
- ✓ Primary scholarly value comes from the publication path, not the Challenge ranking
