"""Train on declared training scenes; select checkpoint on validation only."""

import argparse
from go2_perception.io.config import load_config
from go2_perception.forecasting.training import train

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/forecasting_transformer.yaml")
    p.add_argument("--output", required=True)
    p.add_argument("--data-root")
    p.add_argument("--device", choices=["cpu", "cuda", "mps", "auto"])
    a = p.parse_args()
    cfg = load_config(a.config)
    if a.data_root:
        cfg["dataset"]["root"] = a.data_root
    if a.device:
        cfg["training"]["device"] = a.device
    train(cfg, a.output)
