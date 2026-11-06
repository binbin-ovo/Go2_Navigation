from pathlib import Path
import argparse
import cv2
from go2_perception.rgb.detector import RGBDetector
from go2_perception.rgb.visualization import draw_detections
from go2_perception.io.image_loader import load_image
from go2_perception.io.config import load_config
from go2_perception.io.serialization import save_json

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--config", default="configs/rgb.yaml")
    p.add_argument("--device")
    p.add_argument("--model")
    a = p.parse_args()
    config = load_config(a.config)
    if a.device:
        config["device"] = a.device
    if a.model:
        config["model"] = a.model
    image = load_image(a.input)
    result = RGBDetector(**config).detect(image)
    out = Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "detections.json", result)
    cv2.imwrite(str(out / "rgb.jpg"), draw_detections(image, result))
    print(
        f"{len(result.detections)} detections; {result.latency_ms:.2f} ms (includes first-call overhead)"
    )
