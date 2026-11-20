"""Private offline category-box comparison on the exact LightNav image. Never used by the application."""

from hashlib import sha1
from importlib.metadata import version
import json
from pathlib import Path
from time import monotonic

from PIL import Image

from go2_perception.vln.prediction_archive import file_hash, require


REVISION = "a2bb814dd30d776dcf7e30523b00659f4f141c71"
WEIGHTS_SHA = "1a2412ef99bd74bcd3c2a246fa1e48581f8889a1300c9051974741314fc042f3"


# Git object IDs in the published Hugging Face snapshot for processor/configuration files.
SMALL_FILES = {
    "config.json": "521abd71ae4ca275a31ea98bc68bd3e41450188c",
    "preprocessor_config.json": "5cb45d963917ed130ce46a93204b349cbec21131",
    "tokenizer_config.json": "ed97a84add5f9b2091e756765ad3ba087a345e17",
    "tokenizer.json": "688882a79f44442ddc1f60d70334a7ff5df0fb47",
    "special_tokens_map.json": "a8b3208c2884c4efb86e49300fdd3dc877220cdf",
    "added_tokens.json": "f84095a3e2962f44bdd2f865e4333c35ae95d73f",
    "vocab.txt": "fb140275c155a9c7c5a3b3e0e77a9e839594a938",
}


def compare(snapshot, rows, out):
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    snapshot = Path(snapshot).resolve()
    require(
        snapshot.name == REVISION and file_hash(snapshot / "model.safetensors") == WEIGHTS_SHA,
        "Grounding DINO cached checkpoint",
    )
    for name, expected in SMALL_FILES.items():
        data = (snapshot / name).read_bytes()
        require(sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == expected, name)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    start = monotonic()
    opts = dict(local_files_only=True, trust_remote_code=False)
    processor = AutoProcessor.from_pretrained(str(snapshot), use_fast=False, **opts)
    model = (
        AutoModelForZeroShotObjectDetection.from_pretrained(str(snapshot), use_safetensors=True, **opts)
        .cpu()
        .eval()
    )
    info = dict(
        model="IDEA-Research/grounding-dino-tiny",
        revision=REVISION,
        checkpoint_sha256=WEIGHTS_SHA,
        processor=type(processor).__name__,
        device="cpu",
        dtype=str(next(model.parameters()).dtype),
        threads=4,
        load_s=monotonic() - start,
        query="box.",
        box_threshold=0.35,
        text_threshold=0.25,
        versions={k: version(k) for k in ("torch", "transformers", "Pillow")},
    )
    (out / "semantic-model.json").write_text(json.dumps(info, indent=2))
    for row in rows:
        frame = row["frames"][-1]
        require(file_hash(Path(frame["image"])) == frame["sha256"], "semantic input image")
        with Image.open(frame["image"]) as im:
            image = im.convert("RGB")
        start = monotonic()
        inputs = processor(images=image, text="box.", return_tensors="pt")
        with torch.inference_mode():
            output = model(**inputs)
        result = processor.post_process_grounded_object_detection(
            output,
            inputs.input_ids,
            threshold=0.35,
            text_threshold=0.25,
            target_sizes=[(image.height, image.width)],
        )[0]
        boxes = [
            dict(
                xyxy_norm=[v / n for v, n in zip(box.tolist(), (image.width, image.height) * 2, strict=True)],
                score=float(score),
            )
            for box, score in zip(result["boxes"], result["scores"], strict=True)
        ]
        point = row["point"]
        overlap = (
            None
            if point is None
            else any(
                x0 <= point[0] <= x1 and y0 <= point[1] <= y1
                for x0, y0, x1, y1 in (b["xyxy_norm"] for b in boxes)
            )
        )
        row["semantic"] = dict(
            boxes=boxes, mark_in_candidate=overlap, seconds=monotonic() - start, image_sha256=frame["sha256"]
        )
