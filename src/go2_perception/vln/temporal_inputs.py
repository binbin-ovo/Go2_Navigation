"""Causal calibrated multi-image inputs for the official SmolVLM processor."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .calibration import load_calibration
from .episodes import file_sha256, local_path, select_history
from .model_preprocessing import calibrated_view
from .observation import Go2VLNObservation, Stamp, VisualObservation


@dataclass
class TemporalInput:
    selection: dict
    views: list
    messages: list
    prompt_text: str
    inputs: object
    metadata: dict
    previews: list


def prepare_views(
    episode, decision_id, *, max_frames=4, horizon_s=6.0, allow_synthetic=False
):
    selection = select_history(
        episode, decision_id, max_frames=max_frames, horizon_s=horizon_s
    )
    d = episode.data
    intr, _ = load_calibration(local_path(episode.root, d["calibration"]["path"]))
    views = []
    for frame in selection["frames"]:
        path = local_path(episode.root, frame["image"]["path"])
        if file_sha256(path) != frame["image"]["sha256"]:
            raise ValueError("Selected image changed after episode validation")
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None or [bgr.shape[1], bgr.shape[0]] != frame["size_wh"]:
            raise ValueError("Undecodable image or mismatched manifest dimensions")
        v = VisualObservation(
            cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
            intr.camera_frame,
            Stamp(frame["host_receipt_s"], "host_receipt", "host_receipt"),
            Stamp(
                frame["observation_s"],
                d["clock"]["domain"],
                d["clock"]["observation_semantics"],
                d["clock"]["uncertainty_s"],
            ),
            rectified=d["preprocessing"]["source_rectified"],
            intrinsics_status=intr.status,
            K=intr.K,
            D=intr.D,
            distortion_model=intr.model,
            intrinsics_size=(intr.width, intr.height),
        )
        obs = Go2VLNObservation(
            v, None, None, instruction=d["instruction"], clock_epoch=d["clock"]["epoch"]
        )
        views.append(calibrated_view(obs, allow_synthetic=allow_synthetic))
    return selection, views


def build_messages(instruction, selection, prompt):
    """Whitelist model context; never accepts annotation objects or progress labels."""
    content = [
        {"type": "text", "text": prompt + "\nNavigation instruction: " + instruction}
    ]
    cutoff = selection["decision_time_s"]
    for i, frame in enumerate(selection["frames"]):
        role = "current" if i == len(selection["frames"]) - 1 else "history"
        content += [
            {
                "type": "text",
                "text": f"{role} view at {frame['observation_s'] - cutoff:.3f} seconds relative to decision:",
            },
            {"type": "image"},
        ]
    if selection["gaps"] or selection["missing_frame_ids"]:
        # Only causal interval extents, never future gap reasons or annotations.
        intervals = [
            (round(g["start_s"] - cutoff, 3), round(g["end_s"] - cutoff, 3))
            for g in selection["gaps"]
        ]
        content.append(
            {
                "type": "text",
                "text": f"Observation gaps (relative seconds): {intervals}. Missing selected-window frames: {len(selection['missing_frame_ids'])}.",
            }
        )
    content.append({"type": "text", "text": "Return the next-step proposal JSON only."})
    return [{"role": "user", "content": content}]


def pinhole_fov(intr):
    K, w, h = intr.K, intr.width, intr.height
    horizontal = np.arctan((w - 0.5 - K[0, 2]) / K[0, 0]) - np.arctan(
        (-0.5 - K[0, 2]) / K[0, 0]
    )
    vertical = np.arctan((h - 0.5 - K[1, 2]) / K[1, 1]) - np.arctan(
        (-0.5 - K[1, 2]) / K[1, 1]
    )
    return {
        "horizontal_deg": float(np.degrees(horizontal)),
        "vertical_deg": float(np.degrees(vertical)),
        "meaning": "pinhole border rays, not robot-frame coverage",
    }


def build_model_input(
    episode,
    decision_id,
    processor,
    prompt_path,
    *,
    max_frames=4,
    horizon_s=6.0,
    allow_synthetic=False,
):
    selection, views = prepare_views(
        episode,
        decision_id,
        max_frames=max_frames,
        horizon_s=horizon_s,
        allow_synthetic=allow_synthetic,
    )
    if not views:
        raise ValueError("No causally available image in this decision window")
    prompt_path = Path(prompt_path)
    messages = build_messages(
        episode.data["instruction"], selection, prompt_path.read_text()
    )
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Official template + official normalization once. Ordered image list matches image tokens.
    inputs = processor(
        text=text, images=[Image.fromarray(v.image) for v in views], return_tensors="pt"
    )
    pixels = inputs.pixel_values.detach().cpu().numpy()
    masks = inputs.pixel_attention_mask.detach().cpu().numpy()
    if (
        pixels.ndim != 5
        or pixels.shape[0] != 1
        or pixels.shape[1] != len(views)
        or not masks.all()
    ):
        raise ValueError(
            "Unexpected tiling/padding; refusing to assign incorrect intrinsics"
        )
    ip = processor.image_processor
    if (
        processor.do_image_splitting
        or ip.do_image_splitting
        or not ip.do_rescale
        or not ip.do_normalize
    ):
        raise ValueError(
            "Processor must use declared unsplit images and official normalization"
        )
    if (
        not np.allclose(ip.image_mean, [0.5] * 3)
        or not np.allclose(ip.image_std, [0.5] * 3)
        or ip.rescale_factor != 1 / 255
    ):
        raise ValueError("Unexpected SmolVLM normalization config")
    token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
    image_tokens = int((inputs.input_ids == token_id).sum())
    if image_tokens != len(views) * processor.image_seq_len:
        raise ValueError("History images do not match expanded model image tokens")
    stages, previews = [], []
    for i, view in enumerate(views):
        intermediate = ip.resize(
            view.image,
            size=processor.image_size,
            resample=ip.resample,
            input_data_format="channels_last",
            data_format="channels_last",
        )
        hi, wi = intermediate.shape[:2]
        hf, wf = pixels.shape[-2:]
        final_intr = view.intrinsics.scaled(wi, hi).scaled(wf, hf)
        stages.append(
            {
                "source_and_rectification": view.metadata,
                "rectified_fov": pinhole_fov(view.intrinsics),
                "official_processor_stages": [
                    {
                        "operation": "aspect_preserving_resize",
                        "size_wh": [wi, hi],
                        "K": view.intrinsics.scaled(wi, hi).K.tolist(),
                    },
                    {
                        "operation": "unsplit_square_resize",
                        "size_wh": [wf, hf],
                        "K": final_intr.K.tolist(),
                    },
                ],
                "crop_xywh": [0, 0, view.intrinsics.width, view.intrinsics.height],
                "padding_ltrb": [0, 0, 0, 0],
                "D": [0.0] * 5,
                "model": "pinhole_radtan",
                "tensor_fov": pinhole_fov(final_intr),
                "normalization": "official processor exactly once",
            }
        )
        rgb = pixels[0, i].transpose(1, 2, 0) * np.array(ip.image_std) + np.array(
            ip.image_mean
        )
        previews.append(
            np.clip(np.rint(rgb / ip.rescale_factor), 0, 255).astype(np.uint8)
        )
    metadata = {
        "selection": selection,
        "episode_manifest_sha256": episode.sha256,
        "prompt_sha256": file_sha256(prompt_path),
        "calibration_sha256": episode.data["calibration"]["sha256"],
        "per_frame": stages,
        "image_processor_config": ip.to_dict(),
        "processor_class": type(processor).__name__,
        "input_pixel_shape": list(pixels.shape),
        "image_tokens": image_tokens,
        "tokens_per_image": processor.image_seq_len,
        "input_token_count": int(inputs.input_ids.shape[-1]),
        "model_input_frame_count": len(views),
        "annotations_consumed": False,
        "history_policy": selection["policy"],
    }
    return TemporalInput(selection, views, messages, text, inputs, metadata, previews)
