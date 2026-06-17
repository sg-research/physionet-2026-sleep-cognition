"""
tests/test_eda_signals.py — Unit tests for eda_signals helper functions.

Run with:  uv run pytest tests/test_eda_signals.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Import helpers from the script
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from eda_signals import parse_subject_session


# ---------------------------------------------------------------------------
# parse_subject_session
# ---------------------------------------------------------------------------

class TestParseSubjectSession:

    def test_standard_ses1(self):
        path = Path("sub-S0001112284667_ses-1.edf")
        subj, sess = parse_subject_session(path)
        assert subj == "sub-S0001112284667"
        assert sess == "ses-1"

    def test_standard_ses2(self):
        path = Path("sub-S0001111197789_ses-2.edf")
        subj, sess = parse_subject_session(path)
        assert subj == "sub-S0001111197789"
        assert sess == "ses-2"

    def test_caisr_annotation_filename(self):
        """Annotation files have extra suffix after session."""
        path = Path("sub-S0001111197789_ses-2_caisr_annotations.edf")
        subj, sess = parse_subject_session(path)
        assert subj == "sub-S0001111197789"
        assert sess == "ses-2"

    def test_expert_annotation_filename(self):
        path = Path("sub-S0001112284667_ses-1_expert_annotations.edf")
        subj, sess = parse_subject_session(path)
        assert subj == "sub-S0001112284667"
        assert sess == "ses-1"

    def test_i_site_subject(self):
        path = Path("sub-I0002123456789_ses-3.edf")
        subj, sess = parse_subject_session(path)
        assert subj == "sub-I0002123456789"
        assert sess == "ses-3"

    def test_full_path_ignored(self):
        """Only the stem matters — parent directories are ignored."""
        path = Path("data/kaggle_raw/training_set/physiological_data/S0001/sub-S0001118139481_ses-1.edf")
        subj, sess = parse_subject_session(path)
        assert subj == "sub-S0001118139481"
        assert sess == "ses-1"
