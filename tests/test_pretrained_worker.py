import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def worker():
    spec = importlib.util.spec_from_file_location(
        "lightnav_worker_contract", ROOT / "scripts/lightnav_worker.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def job_data(tmp_path, worker):
    path = tmp_path / "image.png"
    path.write_bytes(b"hash-contract-only")
    return {
        "schema": "lightnav-local-jobs-v1",
        "jobs": [
            {
                "instruction": "Turn left.",
                "episode_id": "ep",
                "clock_epoch": "epoch",
                "decision_id": "d0",
                "decision_time_s": 2.0,
                "frames": [
                    {
                        "path": str(path),
                        "sha256": worker.sha(path),
                        "observation_s": 1.0,
                        "available_s": 1.2,
                    }
                ],
            }
        ],
    }


@pytest.mark.parametrize(
    "extra",
    [
        "annotations",
        "acceptable_actions",
        "future_image",
        "executed_action",
        "robot_pose",
    ],
)
def test_model_job_whitelist_excludes_label_and_action_leakage(worker, tmp_path, extra):
    data = job_data(tmp_path, worker)
    worker.validate_jobs(data)
    data["jobs"][0][extra] = "must never reach the model"
    with pytest.raises(ValueError, match="whitelist"):
        worker.validate_jobs(data)


@pytest.mark.parametrize("mutation", ["future", "changed", "repeated"])
def test_worker_checks_causality_and_prepared_frame_integrity(
    worker, tmp_path, mutation
):
    data = job_data(tmp_path, worker)
    frames = data["jobs"][0]["frames"]
    if mutation == "future":
        frames[0]["available_s"] = 2.1
    if mutation == "changed":
        Path(frames[0]["path"]).write_bytes(b"changed")
    if mutation == "repeated":
        frames.append(dict(frames[0]))
    with pytest.raises(ValueError):
        worker.validate_jobs(data)


def test_bfloat_tensor_archive_preserves_exact_bits(worker):
    import torch

    value = torch.tensor([0.3, -0.25, 1], dtype=torch.bfloat16)
    arrays, types = worker.tensor_archive({"pixels": value})
    restored = torch.from_numpy(arrays["pixels"]).view(torch.bfloat16)
    assert torch.equal(value, restored)
    assert types["pixels"]["dtype"] == "torch.bfloat16"


def test_model_and_simulator_have_no_robot_transport_imports():
    import ast

    for relative in (
        "scripts/lightnav_worker.py",
        "scripts/run_vln_simulation.py",
        "src/go2_perception/vln/simulation.py",
    ):
        tree = ast.parse((ROOT / relative).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [x.name for x in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert all(
                not any(
                    bad in name
                    for bad in (
                        "rclpy",
                        "unitree",
                        "cyclonedds",
                        "robot_deploy",
                        "socket",
                    )
                )
                for name in names
            )
