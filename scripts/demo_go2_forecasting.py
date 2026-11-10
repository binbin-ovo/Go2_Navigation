"""Offline demonstration only; never trains on Go2 tracks or sends robot commands."""

import argparse
import hashlib
import json
from pathlib import Path
from go2_perception.core.tracking_types import TrackOutput, Detection3D
from go2_perception.forecasting.training import load_checkpoint, save_json
from go2_perception.forecasting.adapter import forecast_track
from go2_perception.tracking.multi_object_tracker import MultiObjectTracker


def synthetic_track():
    tracker = MultiObjectTracker(dict(history_size=80))
    for i in range(80):
        t = i * 0.05
        tracks = tracker.update(
            [
                Detection3D(
                    t,
                    [0.5 * t, 0.04 * t * t, 0.8],
                    "odom",
                    "synthetic",
                    class_name="person",
                )
            ],
            t,
        )
    return tracks[0]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--input")
    group.add_argument("--synthetic", action="store_true")
    p.add_argument("--resample", action="store_true")
    p.add_argument("--max-gap-s", type=float, default=0.15)
    p.add_argument("--assume-pedestrian", action="store_true")
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError(out)
    model, payload = load_checkpoint(a.checkpoint, a.device)
    if a.synthetic:
        tracks = [synthetic_track()]
    else:
        # Explicitly inspect the last snapshot only; do not splice different IDs into a trajectory.
        rows = Path(a.input).read_text().splitlines()
        if not rows:
            raise ValueError("Empty tracking input")
        tracks = [TrackOutput(**v) for v in json.loads(rows[-1])["tracks"]]
    identifier = hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()
    results = []
    rejected = []
    for track in tracks:
        try:
            results.append(
                forecast_track(
                    model,
                    track,
                    payload["config"]["dataset"]["dt"],
                    model_id=identifier,
                    resample=a.resample,
                    max_gap_s=a.max_gap_s,
                    assume_pedestrian=a.assume_pedestrian,
                ).to_dict()
            )
        except ValueError as exc:
            rejected.append(dict(track_id=track.track_id, reason=str(exc)))
    out.parent.mkdir(parents=True, exist_ok=True)
    save_json(
        out,
        dict(
            source="synthetic" if a.synthetic else "unvalidated_recorded_tracks",
            experimental=True,
            predictions=results,
            rejected=rejected,
        ),
    )
    print(f"{len(results)} predictions, {len(rejected)} rejected tracks")
