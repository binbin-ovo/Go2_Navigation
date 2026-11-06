from time import perf_counter
from go2_perception.io.image_loader import load_image
from go2_perception.core.types import Detection2D, RGBResult


class RGBDetector:
    """Ultralytics YOLO COCO inference. Lazy model import; no training."""

    def __init__(
        self,
        model="yolov8n.pt",
        device="cpu",
        confidence=0.25,
        nms_iou=0.45,
        image_size=640,
    ):
        if not 0 <= confidence <= 1 or not 0 <= nms_iou <= 1 or image_size < 32:
            raise ValueError("Invalid detector configuration")
        from ultralytics import YOLO

        self.model = YOLO(str(model), task="detect")
        self.options = dict(
            device=device, conf=confidence, iou=nms_iou, imgsz=image_size, verbose=False
        )

    def detect(
        self, image, *, timestamp=None, frame_id="camera_optical", clock_domain=None
    ):
        result = self.detect_batch([image])[0]
        result.timestamp, result.frame_id, result.clock_domain = (
            timestamp,
            frame_id,
            clock_domain,
        )
        return result

    def detect_batch(self, images):
        """Return one result per image; latency is total batch time divided by batch size."""
        images = [load_image(i) for i in images]
        if not images:
            return []
        start = perf_counter()
        predictions = self.model.predict(images, **self.options)
        elapsed = (perf_counter() - start) * 1000 / len(images)
        results = []
        for image, pred in zip(images, predictions):
            detections = [
                Detection2D(
                    int(b.cls.item()),
                    pred.names[int(b.cls.item())],
                    float(b.conf.item()),
                    tuple(b.xyxy[0].cpu().tolist()),
                )
                for b in pred.boxes
            ]
            results.append(
                RGBResult(detections, (image.shape[1], image.shape[0]), elapsed)
            )
        return results
