import numpy as np
from go2_perception.core.geometry import aabb
from go2_perception.core.types import Obstacle


def cluster_points(points, eps=0.3, min_points=8):
    if eps <= 0 or min_points < 1:
        raise ValueError("Invalid DBSCAN configuration")
    if len(points) == 0:
        return np.empty(0, dtype=int), []
    import open3d as o3d

    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    labels = np.asarray(cloud.cluster_dbscan(eps=eps, min_points=min_points))
    obstacles = []
    for label in sorted(set(labels) - {-1}):
        p = points[labels == label]
        low, high, center = aabb(p)
        obstacles.append(Obstacle(int(label), p, center, low, high))
    return labels, obstacles
