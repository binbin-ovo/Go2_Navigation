import argparse
from go2_perception.forecasting.evaluation import evaluate

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Paired ADE/FDE on a frozen checkpoint; no training"
    )
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--split", choices=["validation", "test"], default="test")
    p.add_argument("--device", default="cpu")
    p.add_argument("--data-root")
    p.add_argument(
        "--corruption", choices=["clean", "noise", "missing"], default="clean"
    )
    a = p.parse_args()
    result = evaluate(
        a.checkpoint,
        a.output,
        split=a.split,
        device=a.device,
        data_root=a.data_root,
        corruption=a.corruption,
    )
    print({k: result[k] for k in ("transformer", "constant_velocity", "paired")})
