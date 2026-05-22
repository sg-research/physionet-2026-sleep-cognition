"""Smoke test: package and all subpackages import cleanly."""

from __future__ import annotations


def test_package_imports() -> None:
    import physionet_cog
    from physionet_cog import (
        data,
        evaluation,
        features,
        models,
        training,
        utils,
    )

    assert physionet_cog.__name__ == "physionet_cog"
    for mod in (data, features, models, training, evaluation, utils):
        assert mod.__name__.startswith("physionet_cog.")
