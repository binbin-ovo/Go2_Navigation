"""Convert existing LiDAR/fused results without changing their source modules."""

from go2_perception.core.tracking_types import Detection3D, metadata


def from_lidar(result):
    metadata(result.timestamp, result.frame_id, result.clock_domain)
    return [
        Detection3D(
            result.timestamp,
            o.center,
            result.frame_id,
            result.clock_domain,
            str(o.cluster_id),
            o.bbox_min,
            o.bbox_max,
        )
        for o in result.obstacles
    ]


def from_fusion(result):
    return [
        Detection3D(
            d.timestamp,
            d.position,
            d.frame_id,
            d.clock_domain,
            str(d.cluster_id),
            d.bbox_min,
            d.bbox_max,
            d.class_name,
            d.confidence,
        )
        for d in result.detections
    ]
