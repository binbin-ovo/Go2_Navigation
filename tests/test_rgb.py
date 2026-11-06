"""Wrapper tests mock model predictions; real model smoke is recorded separately."""

import sys
from types import SimpleNamespace
import numpy as np
import pytest
from go2_perception.rgb.detector import RGBDetector


def test_batch_and_metadata(monkeypatch):
    class Model:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, images, **kwargs):
            assert kwargs["device"] == "cpu" and kwargs["conf"] == 0.25
            return [SimpleNamespace(boxes=[], names={}) for _ in images]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=Model))
    detector = RGBDetector()
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    assert len(detector.detect_batch([image, image])) == 2
    result = detector.detect(image, timestamp=5.0, clock_domain="sim")
    assert (
        result.image_size == (30, 20)
        and result.timestamp == 5.0
        and result.latency_ms >= 0
    )
    assert detector.detect_batch([]) == []


def test_bad_threshold():
    with pytest.raises(ValueError):
        RGBDetector(confidence=2)
