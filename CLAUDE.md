# PhysioNet 2026 — Sleep-PSG → Cognitive Impairment Prediction

## Project Context

This repository is the codebase for an entry in the **PhysioNet/Computing in Cardiology Challenge 2026: Screening for Cognitive Impairment During Sleep Studies**, and simultaneously serves as the final project for Stanford CS229 (Summer 2026).

**Primary task.** Given a multi-channel polysomnogram (PSG) — EEG, ECG, respiratory, EOG/EMG signals — predict whether the patient will receive a cognitive impairment diagnosis 3-7 years later. Binary classification, evaluated by AUROC on a hidden test set drawn from BIDMC, Emory, Kaiser Permanente, Mass General Brigham, and Stanford (data via the Human Sleep Project / bdsp.io).

**Deliverables and venues.**
- Official PhysioNet 2026 challenge submission with leaderboard finish
- Computing in Cardiology (CinC) 2026 conference paper (4-6 pages, Spain, September 2026)
- CS229 final project report (5 pages, August 2026)
- ML4H 2026 cross-submission (submission ~September 8, 2026)
- Public GitHub repository as portfolio artifact
- arXiv preprint posted as soon as preliminary results are stable

**Why this project.** It bridges multi-sensor physiological time-series ML (the Apple Health Technologies / Verily / Beacon profile) with long-horizon clinical outcome prediction. The candidate is a clinical researcher pivoting to ML scientist roles, with CNS/neuro domain expertise that aligns directly with the cognitive-impairment outcome.

## Methodological Priorities

These decisions are settled. Claude Code should treat them as constraints rather than re-litigate them in every session.

1. **Foundation-model-first architecture.** Don't train encoders from scratch. The substrate is a pre-trained biosignal foundation model, with downstream multi-task heads.

2. **Benchmark both EEG-specific FMs and generic time-series FMs.** The October 2025 Mantis result (generic TSFMs outperforming EEG-specific FMs on EEG tasks) is methodologically surprising and reproducing it on the PhysioNet task is a publishable finding. Include in the comparison:
   - **CBraMod** (Wang et al., 2025) — current EEG-specific SOTA
   - **LaBraM** (Jiang et al., 2024) — established EEG FM
   - **Mantis** (Feofanov et al., 2025) — generic TSFM
   - **TFM-Tokenizer** (Pradeepkumar et al., NeurIPS TS4H 2025) — alternative tokenization

3. **Channel-tokenization for heterogeneous PSG montages.** Different sites have different sensor configurations. Tokenize each channel independently and flatten into a unified sequence rather than assuming fixed channel structure.

4. **Multi-resolution self-supervised pre-training.** HiMAE-style hierarchical masked autoencoding. PSG signal exists at multiple temporal scales (sub-second micro-architecture, 30s epochs, hour-scale sleep architecture); the analysis should systematically investigate which resolution carries cognitive-decline signal.

5. **Subject-aware contrastive pre-training** for inter-subject robustness (the single biggest generalization risk in EEG).

6. **MoE survival heads** for the 3-7 year time-to-event outcome with calibrated uncertainty.

7. **Site-shift / causal representation learning** for multi-center robustness, since the hidden test set may have different site distributions.

8. **Evaluation must use drug-level / subject-level CV, not random splits.** Naive random CV catastrophically overestimates performance because of subject-level correlation in signals.

9. **Calibration is a first-class metric**, not an afterthought. Report ECE, Brier scores, reliability diagrams. Apply conformal prediction.

10. **Apple-resonant comparison.** Include a transfer experiment using a speech foundation model (Wav2Vec2-style) fine-tuned on PSG, per Narain, Aldeneh, Ren (Apple, NeurIPS TS4H 2025). Even if it underperforms, this is the single most Apple-relevant experimental choice we can make.

## Stretch goals (only after core experiments land)

- **xMADD diffusion-based waveform synthesis** for augmentation of the cognitive-impairment-positive class (Friedman et al., ML4H 2025).
- **MCP server deployment** exposing the trained predictor as a tool.

## Repository Structure

```
physionet-sleep-cognition/
├── CLAUDE.md                 # This file — persistent context for Claude Code
├── README.md                 # Public project description
├── pyproject.toml            # uv-managed Python project
├── .python-version           # Python 3.11 or 3.12
├── .gitignore
├── data/                     # NOT committed; .gitignored
│   ├── raw/                  # Raw PhysioNet 2026 data after download
│   ├── interim/              # Intermediate processed data (Parquet)
│   └── processed/            # Final feature matrices, model-ready
├── src/
│   ├── physionet_cog/        # Main package
│   │   ├── __init__.py
│   │   ├── data/             # Data loading, preprocessing pipelines
│   │   ├── features/         # Hand-crafted feature engineering
│   │   ├── models/           # Model architectures
│   │   │   ├── baselines/    # XGBoost, 1D CNN baselines
│   │   │   ├── foundation/   # FM wrappers (CBraMod, LaBraM, Mantis)
│   │   │   ├── heads/        # Multi-task / MoE / survival heads
│   │   │   └── ssl/          # Self-supervised pre-training (HiMAE, contrastive)
│   │   ├── training/         # Training loops, callbacks
│   │   ├── evaluation/       # Metrics, calibration, conformal
│   │   └── utils/            # Logging, config helpers
├── configs/                  # Hydra configs
│   ├── data/
│   ├── model/
│   ├── training/
│   └── experiment/           # Named experiment configs
├── notebooks/                # Exploratory notebooks; numbered chronologically
├── experiments/              # W&B run logs, outputs (gitignored except metadata)
├── scripts/                  # CLI entry points (data download, training, eval)
├── tests/                    # pytest test suite
├── papers/
│   ├── cinc_2026/            # CinC paper LaTeX source
│   ├── cs229_report/         # CS229 final report
│   └── ml4h_2026/            # ML4H cross-submission
└── docs/                     # Additional documentation
```

## Code Conventions

- **Python**: 3.11 or 3.12.
- **Environment management**: `uv` (not conda, not poetry). Use `uv add`, `uv sync`, `uv run`.
- **Linting/formatting**: `ruff` for both lint and format. Pre-commit hook enforces.
- **Type checking**: `mypy` (gradually adopted, not strict mode initially).
- **Testing**: `pytest`. Test files in `tests/`, mirror `src/` structure.
- **Logging**: standard library `logging`, configured via Hydra.
- **Config management**: Hydra/OmegaConf. No hardcoded paths or hyperparameters in code.
- **Experiment tracking**: Weights & Biases. Every training run logs hyperparameters, metrics, artifacts.
- **Random seeds**: explicit in every config; reproducibility is not optional.
- **Style**: descriptive names, docstrings on public functions, type hints where they help readability.

## Compute Setup

- **Primary development**: M4 Mac with MPS backend. Code should be device-agnostic: `device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"`.
- **Cloud GPU**: Lambda Labs or Vast.ai spot instances for: wide SAE training, large batch experiments, foundation model fine-tuning runs that exceed M4 memory.
- **Don't reach for cloud unnecessarily.** Most baselines, all classical ML, all inference, and most fine-tuning runs fit on M4.

## Key Libraries

Core ML:
- `torch` with MPS support
- `lightning` (PyTorch Lightning) for training loops
- `transformers` (HuggingFace) for foundation models
- `xgboost`, `scikit-learn` for classical baselines

Biosignal-specific:
- `mne` — the standard PSG/EEG processing library
- `pyEDFlib` — for raw EDF file handling if needed

Data:
- `duckdb` — for SQL-based multi-site data ETL
- `polars` — for ML feature dataframes
- `pyarrow` — for Parquet I/O

Experiment infrastructure:
- `hydra-core` for config
- `wandb` for tracking
- `ray[tune]` for hyperparameter search
- `dask` for parallel signal processing across sites

Evaluation:
- `mapie` or hand-rolled conformal prediction
- `scikit-survival` for survival analysis baselines

## Milestones

**Pre-CS229 (now → mid-June):**
- Repo set up, environment working, PhysioNet account, example entry running end-to-end
- Literature review complete (see references below)
- First exploratory notebook on the data

**Weeks 1-3 of CS229 (mid-June → early July):**
- Data pipeline complete: loading, channel harmonization, preprocessing
- Hand-crafted feature baseline (XGBoost) submitting to the unofficial leaderboard
- 1D CNN baseline trained and evaluated

**Weeks 4-7 (early July → late July):**
- Foundation model integrations: CBraMod, LaBraM, Mantis wrappers working
- Multi-task / MoE survival head architecture
- Subject-aware contrastive pre-training on the PSG corpus

**Weeks 8-10 (early August → late August):**
- HiMAE multi-resolution analysis
- Calibration / conformal prediction
- Site-shift robustness analysis
- CinC abstract submission (~July deadline, check exact date)
- CS229 final report

**Post-CS229 (September):**
- Official PhysioNet submission
- CinC paper finalization
- ML4H 2026 cross-submission
- arXiv preprint posted

## Key References (Reading List, Priority Order)

1. Wang et al. 2025, "CBraMod" — current EEG FM SOTA
2. Feofanov et al. 2025, "Mantis" — generic TSFM beating EEG-specific FMs
3. HiMAE (October 2025 arXiv) — multi-resolution masked autoencoding for wearable signals
4. Jiang et al. 2024, "LaBraM" — established EEG FM
5. Friedman, Tonekaboni et al., "xMADD" (ML4H 2025) — diffusion waveform synthesis
6. Multimodal Sleep Staging via Channel-Tokenized General-Purpose Models (Sept 2025) — channel-tokenization approach
7. Ondrus & Cribben, "Segment-Then-Connect for MCI Detection" (NeurIPS TS4H 2025) — task precedent
8. Karantonis et al., "Subject-Aware Contrastive Learning for EEG FMs" (NeurIPS TS4H 2025)
9. Pradeepkumar et al., "TFM-Tokenizer" (NeurIPS TS4H 2025)
10. Narain, Aldeneh, Ren (Apple), "Speech FMs Generalize to Time Series" (NeurIPS TS4H 2025)
11. Morrill et al., "Let the Experts Speak (MoE survival)" (NeurIPS TS4H 2025)
12. Lee et al. 2025, "Comprehensive Review of Biosignal Foundation Models" (TechRxiv) — survey

## How Claude Code Should Approach This Work

- **Read this file before doing anything else** at the start of each session.
- **Prefer small, focused changes** with clear commit messages over large rewrites.
- **Run tests before committing.** If tests don't exist for code being modified, write minimal ones.
- **Use Hydra configs, not hardcoded values.** Any number that might want to change later goes in a config.
- **Log experiments to W&B.** Manual runs that aren't logged are wasted compute.
- **When uncertain about a methodological choice**, ask the user rather than picking one silently. The priorities above are constraints; everything else is open to discussion.
- **Don't add buzzwords for buzzword's sake.** If a technique doesn't fit the data or the task, say so.
- **Keep paper-writing files clean LaTeX** — no commented-out paragraphs, no half-finished sentences. The paper drafts in `papers/` should always be in a runnable state.

## What This Project Is Not

To avoid scope creep:

- Not a real-time / on-device deployment project. The eval is offline AUROC on a hidden test set.
- Not a foundation model pre-training project from scratch. We're consuming pre-trained models, not training new ones at the billion-parameter scale.
- Not a clinical decision support system. The output is a research artifact, not a deployed clinical tool.
- Not an LLM/agent project, primarily. LLMs may appear in stretch goals (e.g., reasoning over signal-derived features) but are not the core methodology.
