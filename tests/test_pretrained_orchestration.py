"""Regression checks for environment identity and immutable benchmark inputs."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest

from go2_perception.vln.episodes import file_sha256, load_episode
from go2_perception.vln.pretrained_inputs import prepare_job
from go2_perception.vln.pretrained import validate_experiment_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "key,value",
    [
        ("model_revision", "unverified"),
        ("dtype", "float16"),
        ("quantization", "4bit"),
        ("max_new_tokens", 32),
        ("seed", 7),
        ("reset_policy", "retain_previous_actions"),
    ],
)
def test_declared_runtime_cannot_silently_disagree_with_worker(key, value):
    import yaml

    config = yaml.safe_load((ROOT / "configs/pretrained_vln.yaml").read_text())
    validate_experiment_config(config)
    config[key] = value
    with pytest.raises(ValueError):
        validate_experiment_config(config)


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"scripts/{name}.py")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def test_preservation_hash_rejects_premature_eof(tmp_path, monkeypatch):
    auditor = module("audit_pretrained_inputs")
    path = tmp_path / "image.png"
    path.write_bytes(b"nonempty original bytes")
    real_open = Path.open

    class EarlyEOF:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.handle.close()

        def fileno(self):
            return self.handle.fileno()

        def read(self, size):
            return b""  # reproduce the observed transient early EOF

    monkeypatch.setattr(
        Path, "open", lambda p, *a, **kw: EarlyEOF(real_open(p, *a, **kw))
    )
    with pytest.raises(OSError, match="Short read"):
        auditor.file_sha256(path)


def test_preservation_hash_allows_genuinely_empty_files(tmp_path):
    import hashlib

    auditor = module("audit_pretrained_inputs")
    path = tmp_path / "empty.log"
    path.write_bytes(b"")
    assert auditor.file_sha256(path) == hashlib.sha256(b"").hexdigest()


def test_subprocess_keeps_model_virtual_environment_launcher(tmp_path, monkeypatch):
    runner = module("run_pretrained_vln")
    launcher = tmp_path / "model-env/bin/python"
    launcher.parent.mkdir(parents=True)
    launcher.symlink_to(sys.executable)
    ep = SimpleNamespace(
        path=tmp_path / "episode.json",
        sha256="x",
        data={"episode_id": "ep", "decisions": [{"decision_id": "d0"}]},
    )
    monkeypatch.setattr(runner, "load_episode", lambda _: ep)

    def prepare(ep, did, folder, **kwargs):
        folder.mkdir(parents=True)
        return {"episode_id": "ep", "decision_id": "d0"}, {}

    monkeypatch.setattr(runner, "prepare_job", prepare)
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(runner.subprocess, "run", run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "runner",
            "--episode",
            str(ep.path),
            "--assets",
            str(tmp_path),
            "--model-python",
            str(launcher),
            "--device",
            "cpu",
            "--output",
            str(tmp_path / "out"),
        ],
    )
    runner.main()
    assert calls[0][0] == str(launcher)
    assert calls[0][0] != str(launcher.resolve())


def test_preparation_does_not_change_episode_images_calibration_or_labels(tmp_path):
    source = tmp_path / "package"
    shutil.copytree(ROOT / "tests/fixtures/vln/episode_v1", source)
    path = source / "episode.json"
    data = json.loads(path.read_text())
    # Regular synthetic stream, prepared before this test's immutable snapshot.
    data["frames"] = [f for f in data["frames"] if f["image"] is not None]
    for i, frame in enumerate(data["frames"]):
        frame["observation_s"] = 100 + i * 0.1
        frame["available_s"] = frame["observation_s"] + 0.05
    data["decisions"] = [
        {"decision_id": "d0", "time_s": data["frames"][-1]["available_s"]}
    ]
    data["gaps"] = []
    path.write_text(json.dumps(data))
    before = {str(p): file_sha256(p) for p in source.rglob("*") if p.is_file()}
    ep = load_episode(path)
    original = deepcopy(ep.data)
    job, provenance = prepare_job(ep, "d0", tmp_path / "prepared", allow_synthetic=True)
    assert job["instruction"] == data["instruction"]
    assert provenance["annotations_consumed"] is False
    assert ep.data == original
    assert before == {str(p): file_sha256(p) for p in source.rglob("*") if p.is_file()}


@pytest.mark.parametrize(
    "mutation", ["episode", "decision", "mode", "manifest", "cutoff", "labels"]
)
def test_comparison_refuses_misbound_or_label_leaking_baselines(mutation):
    evaluator = module("evaluate_pretrained_vln")
    job = {"episode_id": "ep", "decision_id": "d0", "decision_time_s": 2.0}
    baseline = {
        "episode_id": "ep",
        "decision_id": "d0",
        "mode": "temporal",
        "execution_authorized": False,
    }
    provenance = {
        "episode_manifest_sha256": "hash",
        "selection": {"decision_time_s": 2.0},
        "annotations_consumed": False,
    }
    evaluator.check_baseline_binding(job, baseline, provenance, "hash", "temporal")
    if mutation == "episode":
        baseline["episode_id"] = "another"
    elif mutation == "decision":
        baseline["decision_id"] = "d1"
    elif mutation == "mode":
        baseline["mode"] = "current_only"
    elif mutation == "manifest":
        provenance["episode_manifest_sha256"] = "changed"
    elif mutation == "cutoff":
        provenance["selection"]["decision_time_s"] = 3.0
    else:
        provenance["annotations_consumed"] = True
    with pytest.raises(ValueError, match="bound"):
        evaluator.check_baseline_binding(job, baseline, provenance, "hash", "temporal")
