"""Evaluation regressions: stale geometry and annotation gaps must stay explicit."""

import importlib.util
from pathlib import Path


def test_real_metrics_separate_observation_availability_and_time_gaps(monkeypatch):
    monkeypatch.syspath_prepend(str(Path("scripts").resolve()))
    spec = importlib.util.spec_from_file_location(
        "scene_evaluator", "scripts/evaluate_stable_scene_real.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = dict(
        id="box",
        status="visible",
        region=[[0, 0], [1, 0], [1, 1], [0, 1]],
        motion="stationary",
    )
    times = [0.0, 0.1, 1.0]
    doc = dict(
        t0=0.0,
        objects={"box": {"type": "static_obstacle"}},
        frames=[dict(t=t, objects=[state]) for t in times],
    )
    scenes = [
        dict(
            diagnostics=dict(source_midpoint_timestamp=t, proposal_boxes=[]),
            objects=[
                dict(
                    center=[0.5, 0.5, 0.4],
                    velocity=[0, 0, 0],
                    motion_state="UNCERTAIN",
                    status="stale" if t == 0.1 else "observed_filtered",
                )
            ],
        )
        for t in times
    ]
    fresh = module.evaluate(doc, scenes)
    available = module.evaluate(doc, scenes, include_stale=True)
    assert fresh["objects"]["box"]["coverage"] == 2 / 3
    assert available["objects"]["box"]["coverage"] == 1
    assert fresh["objects"]["box"]["longest_observed_fraction"] == 1 / 3
    assert available["objects"]["box"]["longest_observed_fraction"] == 2 / 3
    assert fresh["stale_fraction"] == 1 / 3
