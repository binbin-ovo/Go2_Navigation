from pathlib import Path
import cv2
import numpy as np


def load_image(source):
    """Return uint8 BGR image. Arrays are BGR, never implicit RGB."""
    image = (
        cv2.imread(str(source))
        if isinstance(source, (str, Path))
        else np.asarray(source)
    )
    if (
        image is None
        or image.ndim != 3
        or image.shape[2] != 3
        or image.dtype != np.uint8
        or min(image.shape[:2]) == 0
    ):
        raise ValueError("Expected readable image or nonempty uint8 HxWx3 BGR array")
    return image
