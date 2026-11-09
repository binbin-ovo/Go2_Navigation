import argparse
import numpy as np
from go2_perception.forecasting.visualization import plot_examples
from go2_perception.forecasting.training import save_json
from pathlib import Path

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Paired examples: category medians and both error extremes"
    )
    p.add_argument("--predictions", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("Use a fresh visualization output directory")
    with np.load(a.predictions, allow_pickle=False) as r:
        examples = plot_examples(
            r["observed"],
            r["target"],
            r["constant_velocity"],
            r["transformer"],
            r["scenes"],
            r["keys"],
            float(r["dt"]),
            out,
        )
    save_json(out / "selection.json", examples)
