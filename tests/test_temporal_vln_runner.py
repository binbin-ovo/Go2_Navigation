"""Exact-package CLI selection; stubs stop before inference, never model evidence."""

import json
from pathlib import Path
import runpy
import shutil
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/vln/episode_v1"


@pytest.mark.parametrize("count", [1, 2])
def test_explicit_nested_manifests_exclude_neighbors_and_labels(
    tmp_path, monkeypatch, count
):
    manifests = []
    for i in range(3):
        package = tmp_path / f"package_{i}" / "model_input"
        shutil.copytree(FIXTURE, package)
        path = package / "episode.json"
        data = json.loads(path.read_text())
        data["episode_id"] = f"selected_{i}"
        path.write_text(json.dumps(data))
        manifests.append(path)
    runner = runpy.run_path(str(ROOT / "scripts/run_temporal_vln.py"))
    scope = runner["main"].__globals__

    class ReachedInference(Exception):
        pass

    class StubModel:
        info = {"test_stub": True}

        def __init__(self, *args, **kwargs):
            pass

        def infer(self, ep, decision_id, *args, **kwargs):
            assert ep.path == manifests[0].resolve()
            assert decision_id == "d0"
            raise ReachedInference

    def unexpected_annotations(*args, **kwargs):
        raise AssertionError("Evaluation annotations reached the pre-inference path")

    monkeypatch.setitem(scope, "SmolTemporalModel", StubModel)
    monkeypatch.setitem(scope, "load_annotations", unexpected_annotations)
    output = tmp_path / "run"
    argv = ["run_temporal_vln.py"]
    for path in manifests[:count]:
        argv.extend(["--episode", str(path)])
    argv += [
        "--annotations-name",
        "../evaluation_only/annotations.json",
        "--cache-dir",
        str(tmp_path / "unused_cache"),
        "--output",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ReachedInference):
        runner["main"]()
    declaration = json.loads((output / "run_declaration.json").read_text())
    assert [r["manifest"] for r in declaration["episodes"]] == [
        str(p.resolve()) for p in manifests[:count]
    ]
    assert [r["episode_id"] for r in declaration["episodes"]] == [
        f"selected_{i}" for i in range(count)
    ]
    assert not (tmp_path / "unused_cache").exists()
