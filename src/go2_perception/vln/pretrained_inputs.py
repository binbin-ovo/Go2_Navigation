"""Reuse calibrated Phase 6B views, with LightNav's regular causal observation stream."""

from dataclasses import replace

import cv2

from .episodes import file_sha256
from .pretrained import causal_history
from .temporal_inputs import prepare_views


def prepare_job(episode, decision_id, destination, *, allow_synthetic=False):
    selection = causal_history(episode, decision_id)
    # A read-only in-memory selection; episode.json and all package bytes stay untouched.
    selected_episode = replace(
        episode, data={**episode.data, "frames": selection["frames"]}
    )
    horizon = selection["decision_time_s"] - episode.data["bounds"]["start_s"] + 1
    _, views = prepare_views(
        selected_episode,
        decision_id,
        max_frames=len(selection["frames"]),
        horizon_s=horizon,
        allow_synthetic=allow_synthetic,
    )
    destination.mkdir(parents=True, exist_ok=False)
    frames, metadata = [], []
    for i, (frame, view) in enumerate(zip(selection["frames"], views)):
        path = destination / f"frame_{i:04d}.png"
        if not cv2.imwrite(str(path), cv2.cvtColor(view.image, cv2.COLOR_RGB2BGR)):
            raise OSError(f"Could not save rectified view: {path}")
        frames.append(
            {
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "observation_s": frame["observation_s"],
                "available_s": frame["available_s"],
            }
        )
        metadata.append(
            {
                "source_frame": frame,
                "rectification": view.metadata,
                "model_geometry": {
                    "size_wh": [448, 256],
                    "K": view.intrinsics.scaled(448, 256).K.tolist(),
                    "resize": "official bilinear align_corners=False stretch",
                    "crop": None,
                    "padding": None,
                    "normalization": "official rgb_frame_to_model_tensor to [-1,1] once; _skip_normalize=true",
                    "frame": "camera_optical_rectified_unregistered_to_base",
                },
            }
        )
    job = {
        "instruction": episode.data["instruction"],
        "episode_id": episode.data["episode_id"],
        "clock_epoch": episode.data["clock"]["epoch"],
        "decision_id": decision_id,
        "decision_time_s": selection["decision_time_s"],
        "frames": frames,
    }
    provenance = {
        "episode_manifest": str(episode.path),
        "episode_manifest_sha256": episode.sha256,
        "calibration_sha256": episode.data["calibration"]["sha256"],
        "selection": selection,
        "frames": metadata,
        "annotations_consumed": False,
        "proposals_fed_back_as_executed": False,
        "T_base_camera": None,
    }
    return job, provenance
