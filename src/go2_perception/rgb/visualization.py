import cv2


def draw_detections(image, result):
    canvas = image.copy()
    for d in result.detections:
        x1, y1, x2, y2 = map(round, d.bbox_xyxy)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (50, 220, 50), 2)
        cv2.putText(
            canvas,
            f"{d.class_name} {d.confidence:.2f}",
            (x1, max(15, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (50, 220, 50),
            1,
        )
    return canvas
