"""Download PhysioNet/CinC Challenge 2026 data from Kaggle into data/kaggle_raw/.

Usage:
    uv run python scripts/download_kaggle_data.py
    uv run python scripts/download_kaggle_data.py --force   # re-download

Requires Kaggle credentials. Either:
    - place a `kaggle.json` API token at ~/.kaggle/kaggle.json (chmod 600), or
    - set env vars KAGGLE_USERNAME and KAGGLE_KEY.

kagglehub downloads to its own cache and we symlink/copy the snapshot path
into data/kaggle_raw/ so the rest of the codebase can use a stable path.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import kagglehub
import structlog

log = structlog.get_logger()

DATASET_HANDLE = "physionet/physionetchallenge2026data"
REPO_ROOT = Path(__file__).resolve().parent.parent
TARGET_DIR = REPO_ROOT / "data" / "kaggle_raw"


def _check_credentials() -> None:
    token_path = Path.home() / ".kaggle" / "kaggle.json"
    has_env = bool(os.environ.get("KAGGLE_USERNAME")) and bool(
        os.environ.get("KAGGLE_KEY")
    )
    if not token_path.exists() and not has_env:
        log.error(
            "kaggle_credentials_missing",
            hint=(
                "Place kaggle.json at ~/.kaggle/kaggle.json (chmod 600) or "
                "set KAGGLE_USERNAME and KAGGLE_KEY env vars."
            ),
        )
        sys.exit(2)


def _link_into_repo(snapshot_path: Path, target: Path, force: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() or target.is_symlink():
        if not force:
            log.info("target_exists_skipping", target=str(target))
            return
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)

    target.symlink_to(snapshot_path, target_is_directory=True)
    log.info("symlinked", source=str(snapshot_path), target=str(target))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-link even if data/kaggle_raw already exists.",
    )
    args = parser.parse_args()

    _check_credentials()

    log.info("downloading", dataset=DATASET_HANDLE)
    snapshot = Path(kagglehub.dataset_download(DATASET_HANDLE))
    log.info("download_complete", path=str(snapshot))

    _link_into_repo(snapshot, TARGET_DIR, force=args.force)

    log.info(
        "ready",
        kaggle_raw=str(TARGET_DIR),
        note="Run `ls data/kaggle_raw/` to inspect the dataset.",
    )


if __name__ == "__main__":
    main()
