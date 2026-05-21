# Getting Started — Practical Setup Guide

This guide walks you through getting from "I have these documents" to "Claude Code is set up and ready to start the project." Estimated time: 30-60 minutes.

## Step 1 — Install prerequisites (10 min)

Open your terminal on the M4 Mac:

```bash
# Install Homebrew if you don't have it
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install uv (modern Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install git if not present
brew install git

# Install Claude Code (native installer, no Node.js needed)
curl -fsSL https://claude.ai/install.sh | bash
```

Verify:
```bash
uv --version       # Should print a version
git --version      # Should print a version
claude --version   # Should print a version
```

## Step 2 — Create the project repository (5 min)

```bash
# Create the directory and move into it
mkdir -p ~/projects/physionet-sleep-cognition
cd ~/projects/physionet-sleep-cognition

# Initialize git
git init

# Initialize a uv Python project
uv init --python 3.12

# Create the standard directories
mkdir -p data/{raw,interim,processed}
mkdir -p src/physionet_cog/{data,features,models/{baselines,foundation,heads,ssl},training,evaluation,utils}
mkdir -p configs/{data,model,training,experiment}
mkdir -p notebooks experiments scripts tests
mkdir -p papers/{cinc_2026,cs229_report,ml4h_2026}
mkdir -p docs
```

## Step 3 — Drop in the foundational documents (2 min)

Copy `CLAUDE.md`, `PROJECT_BRIEF.md`, and (optionally) this `GETTING_STARTED.md` into the repository root:

```bash
# From wherever you downloaded them
cp CLAUDE.md PROJECT_BRIEF.md GETTING_STARTED.md ~/projects/physionet-sleep-cognition/
```

Create a `.gitignore`:

```bash
cat > .gitignore << 'EOF'
# Data — never commit
data/raw/
data/interim/
data/processed/
*.edf
*.h5
*.hdf5
*.parquet
*.csv

# Models
*.pt
*.pth
*.ckpt
*.safetensors

# Experiment artifacts
experiments/
wandb/
*.log

# Python
__pycache__/
*.py[cod]
.venv/
.pytest_cache/
.mypy_cache/
.ruff_cache/

# OS
.DS_Store

# Editor
.vscode/
.idea/

# Secrets
.env
*.pem
EOF
```

Commit the initial structure:

```bash
git add .
git commit -m "Initial repository structure with CLAUDE.md and project brief"
```

## Step 4 — Create a PhysioNet account and download challenge data (15-30 min)

Before opening Claude Code, do this part manually because it requires identity verification:

1. Create a PhysioNet account at https://physionet.org/register/
2. Complete the CITI "Data or Specimens Only Research" training (required for Human Sleep Project data access) — this takes about an hour but is one-time
3. Apply for access to the PhysioNet 2026 Challenge data
4. Read the challenge page carefully: https://moody-challenge.physionet.org/2026/

While that's processing, you can do everything else.

## Step 5 — Open Claude Code and let it set up the rest (10 min)

From inside the repo directory:

```bash
claude
```

This opens a Claude Code session. The first thing to say:

> Read CLAUDE.md, then set up the initial Python project. Add the dependencies we'll need: PyTorch with MPS support, MNE-Python, scikit-learn, XGBoost, transformers, lightning, hydra-core, wandb, duckdb, polars, pyarrow, pytest, ruff, mypy. Set up pre-commit hooks for ruff. Create a basic README. Confirm everything installs cleanly with `uv sync`.

Claude Code will read your CLAUDE.md, propose changes, ask permission for each significant action, and execute. Review what it does. You'll commit the changes when you're satisfied.

## Step 6 — First exploratory work (after data access is granted)

Once you have access to the PhysioNet 2026 data:

> Download the PhysioNet 2026 example entry from the challenge GitHub. Run it end-to-end on a small subset of the data to verify our environment works. Document what each stage of the pipeline does in a notebook at notebooks/01_pipeline_walkthrough.ipynb.

This is your hello-world. Once the example entry runs locally, you understand the data format, the evaluation, and the submission process — and you're ready to start writing real code.

## Step 7 — Establish a working rhythm

Suggested:

- **Mornings (~2 hr)**: pure CS229 coursework (assignments, lectures, problem sets). Don't mix with project work.
- **Afternoons (~3-4 hr)**: project work in Claude Code. Focused implementation sessions.
- **One evening per week (~1-2 hr)**: read papers, update the related-work LaTeX, sketch figures in a notebook. Slower, conceptual work.
- **End of week (~1 hr)**: commit a weekly W&B report summary, push papers/cs229_report/ progress to the report, decide priorities for the next week.

## Where to ask what

A guideline that's saved time for others:

- **High-level methodology, scoping decisions, paper writing, results interpretation, "is this approach right?"** → here in Claude.ai. The Project memory accumulates context across sessions, which is useful for things that span weeks.
- **Implementation, debugging, repo work, running experiments, fixing CI** → Claude Code. Speed of iteration matters more than long-context reasoning for this work.
- **When you're stuck and can't tell which kind of question you have** → bring it here first. We figure out what kind of question it is, and if it's an implementation question I can help you frame it for Claude Code.

## Common first-week pitfalls to avoid

- **Don't start with the fanciest method.** Get the XGBoost-on-hand-crafted-features baseline submitting to the unofficial leaderboard *before* touching CBraMod or Mantis. Knowing your evaluation pipeline works is more valuable than knowing your fancy model is.
- **Don't skip the example entry.** Many PhysioNet 2026 teams will waste a week debugging submission format issues. Run the example end-to-end first.
- **Don't commit data.** Even tiny CSV files. The `.gitignore` is your friend.
- **Don't fight Hydra.** Configs everywhere, hardcoded values nowhere. Future-you (writing the paper) will thank present-you (writing the code) for this.
- **Don't postpone the paper.** Start the CinC paper LaTeX skeleton in week 1, even if it's just title, authors, and section headers. Writing happens incrementally.

## When you're ready

Once you've done steps 1-3, you can return here and let me know — we can talk through any setup issues, and once data access is granted, we can start working through specific implementation questions together as they come up.
