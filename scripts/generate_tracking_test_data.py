"""Generate explicitly synthetic timestamped detections, poses and truth."""

import argparse
import json
from pathlib import Path
from go2_perception.tracking.synthetic import SCENARIOS, generate_sequence
from go2_perception.io.serialization import save_json

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="data/sample/tracking_sequence")
    p.add_argument("--seed", type=int, default=13)
    a = p.parse_args()
    for scenario in SCENARIOS:
        folder = Path(a.output) / scenario
        folder.mkdir(parents=True, exist_ok=True)
        rows = generate_sequence(scenario, a.seed)
        (folder / "frames.jsonl").write_text(
            "".join(json.dumps(r, allow_nan=False) + "\n" for r in rows)
        )
        save_json(
            folder / "manifest.json",
            dict(
                schema="go2-tracking-v1", synthetic=True, scenario=scenario, seed=a.seed
            ),
        )
    print(f"Generated {len(SCENARIOS)} synthetic scenarios in {a.output}")
