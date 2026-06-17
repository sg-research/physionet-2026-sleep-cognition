"""
eda_signals.py — Exploratory analysis of PSG signals and annotations.

Sections:
  1. File inventory — which subjects have physiological + CAISR + human files
  2. Per-site subject selection — one representative subject per site
  3. Physiological signal plots — 30s epoch of all channels per site
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import polars as pl

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "data" / "kaggle_raw" / "training_set"
FIGURES_DIR = PROJECT_ROOT / "notebooks" / "figures"

SITES = ["S0001", "I0002", "I0006"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_subject_session(path: Path) -> tuple[str, str]:
    """Extract (subject_id, session) from an EDF filename.

    e.g. sub-S0001111197789_ses-2.edf  ->  ('sub-S0001111197789', 'ses-2')
    """
    name = path.stem  # drop .edf
    subject_id = name.split("_ses")[0]
    session = "ses" + name.split("_ses")[1].split("_")[0]
    return subject_id, session


def find_edf_files(subdir: str) -> list[Path]:
    """Recursively find all EDF files under BASE_DIR / subdir (all sites)."""
    pattern = str(BASE_DIR / subdir / "**" / "*.edf")
    return [Path(p) for p in glob.glob(pattern, recursive=True)]


def build_file_index(file_type: str) -> dict[tuple[str, str], Path]:
    """Return {(subject_id, session): Path} for a given file type subdir."""
    index: dict[tuple[str, str], Path] = {}
    for path in find_edf_files(file_type):
        try:
            subj, sess = parse_subject_session(path)
            index[(subj, sess)] = path
        except (IndexError, ValueError):
            pass
    return index


# ---------------------------------------------------------------------------
# Section 1 — File inventory
# ---------------------------------------------------------------------------

def file_inventory() -> pl.DataFrame:
    """Return a DataFrame listing every subject and which file types exist."""
    phys_idx = build_file_index("physiological_data")
    caisr_idx = build_file_index("algorithmic_annotations")
    human_idx = build_file_index("human_annotations")

    all_keys = set(phys_idx) | set(caisr_idx) | set(human_idx)

    rows = []
    for subj, sess in sorted(all_keys):
        phys_path = phys_idx.get((subj, sess))
        site = phys_path.parent.name if phys_path else "unknown"
        rows.append({
            "subject_id": subj,
            "session": sess,
            "site": site,
            "has_phys": (subj, sess) in phys_idx,
            "has_caisr": (subj, sess) in caisr_idx,
            "has_human": (subj, sess) in human_idx,
            "phys_path": str(phys_path) if phys_path else None,
            "caisr_path": str(caisr_idx.get((subj, sess))) if (subj, sess) in caisr_idx else None,
            "human_path": str(human_idx.get((subj, sess))) if (subj, sess) in human_idx else None,
        })

    df = pl.DataFrame(rows)

    print("=== FILE INVENTORY ===")
    print(f"Total subjects: {df.height}")
    print(f"Has all 3 files: {df.filter(pl.col('has_phys') & pl.col('has_caisr') & pl.col('has_human')).height}")
    print(f"Missing CAISR:   {df.filter(pl.col('has_phys') & ~pl.col('has_caisr')).height}")
    print(f"Missing human:   {df.filter(pl.col('has_phys') & ~pl.col('has_human')).height}")
    print()
    print("Per-site breakdown:")
    site_summary = (
        df.group_by("site")
        .agg(
            pl.len().alias("total"),
            (pl.col("has_phys") & pl.col("has_caisr") & pl.col("has_human")).sum().alias("all_three"),
        )
        .sort("site")
    )
    print(site_summary)
    print()

    return df


# ---------------------------------------------------------------------------
# Section 2 — Per-site subject selection
# ---------------------------------------------------------------------------

def select_subjects(inventory: pl.DataFrame) -> dict[str, dict]:
    """Pick the first subject with all three files for each site."""
    complete = inventory.filter(
        pl.col("has_phys") & pl.col("has_caisr") & pl.col("has_human")
    )

    selected: dict[str, dict] = {}
    for site in SITES:
        site_rows = complete.filter(pl.col("site") == site)
        if site_rows.is_empty():
            print(f"  WARNING: no complete subject found for site {site}")
            continue
        row = site_rows.row(0, named=True)
        selected[site] = row
        print(f"  {site}: {row['subject_id']} ({row['session']})")

    print()
    return selected


# ---------------------------------------------------------------------------
# Section 3 — Physiological signal plots (30s epoch)
# ---------------------------------------------------------------------------

def plot_signals(selected: dict[str, dict]) -> None:
    """Save a 30-second epoch plot for one subject per site."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for site, info in selected.items():
        phys_path = info["phys_path"]
        print(f"Loading {site}: {Path(phys_path).name}")

        raw = mne.io.read_raw_edf(phys_path, preload=True, verbose=False)
        n_channels = len(raw.ch_names)

        print(f"  Channels ({n_channels}): {raw.ch_names}")
        print(f"  Sampling rate: {raw.info['sfreq']} Hz")
        print(f"  Duration: {raw.times[-1] / 3600:.2f} hours")
        print()

        # Extract first 30 seconds
        sfreq = raw.info["sfreq"]
        n_samples = int(30 * sfreq)
        data, times = raw[:, :n_samples]

        fig, axes = plt.subplots(n_channels, 1, figsize=(14, max(2 * n_channels, 8)),
                                  sharex=True)
        if n_channels == 1:
            axes = [axes]

        for i, (ax, ch_name) in enumerate(zip(axes, raw.ch_names)):
            ax.plot(times, data[i], linewidth=0.6, color="steelblue")
            ax.set_ylabel(ch_name, fontsize=7, rotation=0, labelpad=60, va="center")
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(f"{site} — 30s PSG epoch\n{Path(phys_path).name}", fontsize=10)
        plt.tight_layout()

        out = FIGURES_DIR / f"signals_30s_{site}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.chdir(PROJECT_ROOT)
    mne.set_log_level("WARNING")

    print("=== EDA SIGNALS ===\n")

    inventory = file_inventory()
    print("=== SUBJECT SELECTION ===")
    selected = select_subjects(inventory)

    print("=== SIGNAL PLOTS ===")
    plot_signals(selected)

    print("\nDone.")


if __name__ == "__main__":
    main()
