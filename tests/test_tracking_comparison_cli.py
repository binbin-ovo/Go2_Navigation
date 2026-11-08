"""Protect historical private replay outputs when invoking the comparison harness."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import pytest


def harness():
    spec = importlib.util.spec_from_file_location(
        "real_comparison", Path("scripts/compare_tracking_real.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_comparison_refuses_to_overwrite_existing_run(tmp_path):
    sentinel = tmp_path / "run.json"
    sentinel.write_text("original")
    with pytest.raises(FileExistsError):
        harness().run(SimpleNamespace(output=str(tmp_path), variants=["baseline"]))
    assert sentinel.read_text() == "original"


def test_external_cache_cannot_be_reextracted(tmp_path):
    with pytest.raises(ValueError, match="External cache"):
        harness().run(
            SimpleNamespace(
                output=str(tmp_path / "new"),
                variants=["baseline"],
                cache_root=str(tmp_path),
                reuse_cache=False,
            )
        )


def test_duplicate_metric_distinguishes_coasting_from_observed_pairs():
    import numpy as np

    def track(identifier, misses):
        return dict(
            id=identifier,
            p=np.zeros(2),
            v=np.zeros(2),
            status="C",
            motion="U",
            misses=misses,
        )

    rows = [dict(t=0, tracks=[track(1, 0), track(2, 1)], latency_ms=1, fragments=0)]
    result = harness().scene(rows, True)
    assert result["duplicate_pair_frames"] == 1
    assert result["observed_duplicate_pair_frames"] == 0
