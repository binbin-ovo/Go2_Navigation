from time import perf_counter
import numpy as np
from go2_perception.core.types import LidarResult
from .preprocessing import preprocess
from .ground_segmentation import ground_mask
from .clustering import cluster_points


class LidarProcessor:
    def __init__(self, preprocessing=None, ground=None, clustering=None):
        self.preprocessing = preprocessing or {}
        self.ground = ground or {}
        self.clustering = clustering or {}

    def process(self, points, *, timestamp=None, frame_id="lidar", clock_domain=None):
        start = perf_counter()
        p = preprocess(points, **self.preprocessing)
        mask = ground_mask(p, **self.ground)
        non_ground_labels, obstacles = cluster_points(p[~mask], **self.clustering)
        labels = np.full(len(p), -2, dtype=int)
        labels[~mask] = non_ground_labels
        return LidarResult(
            p,
            p[mask],
            p[labels == -1],
            obstacles,
            labels,
            (perf_counter() - start) * 1000,
            timestamp,
            frame_id,
            clock_domain,
        )
