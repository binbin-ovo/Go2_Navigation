"""Model-input conversion for ``VisualObservation`` without hidden geometry changes.

Resizing changes intrinsics (and aspect ratio when the scale differs per axis); these helpers
return updated intrinsics alongside the pixels so downstream geometry stays consistent.
"""

import cv2
import numpy as np

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def resize(image_rgb, size_wh, intrinsics=None, keep_aspect=True):
    """Resize (letterbox when keep_aspect) -> (image, intrinsics or None, transform info)."""
    h, w = image_rgb.shape[:2]
    tw, th = size_wh
    if not keep_aspect:
        out = cv2.resize(image_rgb, (tw, th), interpolation=cv2.INTER_AREA)
        return out, None if intrinsics is None else intrinsics.scaled(tw, th), \
            {"scale_x": tw / w, "scale_y": th / h, "pad_x": 0, "pad_y": 0}
    s = min(tw / w, th / h)
    nw, nh = round(w * s), round(h * s)
    small = cv2.resize(image_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.zeros((th, tw, 3), image_rgb.dtype)
    px, py = (tw - nw) // 2, (th - nh) // 2
    out[py:py + nh, px:px + nw] = small
    intr = None
    if intrinsics is not None:
        intr = intrinsics.scaled(nw, nh).cropped(-px, -py, tw, th)
    return out, intr, {"scale_x": nw / w, "scale_y": nh / h, "pad_x": px, "pad_y": py}


def to_tensor(image_rgb, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """uint8 HxWx3 RGB -> torch float32 3xHxW, scaled to [0,1] then normalised."""
    import torch

    if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("Expected uint8 HxWx3 RGB")
    x = torch.from_numpy(np.ascontiguousarray(image_rgb)).permute(2, 0, 1).float() / 255.0
    if mean is not None:
        x = (x - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)
    return x


def stack_history(visual):
    """Oldest-first uint8 T x H x W x 3 array of the history plus the current frame."""
    frames = [h.image for h in visual.history] + [visual.image]
    if len({f.shape for f in frames}) != 1:
        raise ValueError("History frames differ in shape")
    return np.stack(frames)
