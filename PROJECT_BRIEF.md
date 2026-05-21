# Project Brief: Multi-Channel Foundation Models for Long-Horizon Cognitive Impairment Prediction from Sleep Studies

## One-line description

A machine learning system that predicts a patient's risk of developing cognitive impairment 3-7 years after a routine clinical sleep study, by combining multi-channel physiological signals with modern foundation-model representations.

## Why this matters

Cognitive decline is among the most consequential late-life health outcomes, yet most diagnostic pathways detect it only after substantial irreversible neural damage. Polysomnograms (PSGs) — multi-channel sleep studies routinely conducted for sleep apnea, insomnia, and parasomnia workups — capture rich, multi-system physiological information (brain activity, cardiac dynamics, respiratory patterns, autonomic tone) and are increasingly recognized as a window into early neurodegenerative processes. If clinically routine PSGs can be repurposed to stratify cognitive risk years before clinical presentation, the implications for prevention, monitoring, and trial enrollment are substantial.

## Scientific contribution

The project addresses three open methodological questions in clinical biosignal ML:

1. **Are domain-specific foundation models necessary, or do generic time-series foundation models transfer well to multi-channel physiological signals?** Recent evidence (Feofanov et al., 2025) suggests generic TSFMs match or beat EEG-specific FMs on EEG tasks. We test this systematically on a long-horizon clinical outcome.

2. **At what temporal resolution does cognitive-decline signal live in PSG data?** Multi-resolution masked-autoencoder analysis (HiMAE-style) interrogates whether predictive structure exists in sub-second micro-architecture, 30-second epoch transitions, or hour-scale architecture.

3. **How robust are these predictions to inter-site variation?** Multi-center training (five US institutions) with explicit causal/site-shift representation learning makes generalization quantifiable rather than assumed.

## Data

The Human Sleep Project (bdsp.io) — multi-center polysomnograms from BIDMC, Emory, Kaiser Permanente, Mass General Brigham, and Stanford, with longitudinal cognitive outcome ascertainment 3-7 years post-PSG. Multi-channel signals include EEG, ECG, respiratory effort, oxygen saturation, and EOG/EMG.

## Methods

A foundation-model-based architecture combining:

- **Channel-tokenized signal encoders** that handle heterogeneous montages across sites
- **Multiple pre-trained foundation models compared head-to-head**: EEG-specific (CBraMod, LaBraM) and generic time-series (Mantis), plus a cross-domain transfer experiment using speech foundation models (Wav2Vec2-style)
- **Multi-resolution self-supervised pre-training** on the PSG corpus
- **Subject-aware contrastive learning** to address inter-subject signal variation
- **Mixture-of-experts survival heads** for calibrated 3-7 year time-to-event prediction
- **Conformal prediction** for distribution-free uncertainty intervals
- **Causal representation learning** for multi-site distributional robustness

## Evaluation

- AUROC and AUPRC on the hidden PhysioNet 2026 test set
- Subject-level cross-validation (not random splits — random splits leak signal through subject-level correlations)
- Expected calibration error, Brier score, reliability diagrams
- Per-site performance breakdown to quantify cross-institutional transfer
- Comparison to clinical baselines (age, sex, comorbidities only)

## Deliverables

- Official PhysioNet 2026 challenge submission with leaderboard ranking
- Peer-reviewed paper in Computing in Cardiology 2026 proceedings (Conference in Spain, September 2026)
- ML4H 2026 workshop cross-submission
- CS229 final project report (Stanford, Summer 2026)
- Public GitHub repository with reproducible pipeline
- arXiv preprint
- An MCP server exposing the trained predictor as a callable tool (stretch goal)

## Timeline

- **June-August 2026** — primary development concurrent with Stanford CS229
- **July 2026** — CinC abstract submission, CS229 milestone deliverables
- **September 2026** — official PhysioNet submission, CinC paper, ML4H submission
- **December 2026** — ML4H 2026 presentation if accepted

## Why this candidate

This project sits at the intersection of clinical pharmacology, neuroscience, and machine learning. The candidate brings: prior clinical research experience including CNS / neuro therapeutic areas; Stanford CS coursework in algorithms, systems, and machine learning (CS229); and a publication trajectory specifically targeting venues that span these domains (CinC, ML4H, NeurIPS Datasets & Benchmarks, ML4H 2027, and a parallel drug-safety prediction project for the Insitro / Recursion / Aitia track).

## Career relevance

The methodological stack — multi-sensor physiological time-series ML, foundation model fine-tuning, calibrated clinical prediction, multi-site robustness, multimodal fusion — is the core profile sought by Apple Health Technologies, Verily, Beacon, Anthropic Health, and clinical AI startups working on early-detection systems for neurodegenerative disease.
