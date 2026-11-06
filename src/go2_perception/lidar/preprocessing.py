import numpy as np
from go2_perception.io.pointcloud_loader import validate_points


def preprocess(points, min_range=0.1, max_range=30.0, voxel_size=0.05):
    if not 0 <= min_range < max_range or voxel_size < 0:
        raise ValueError("Invalid range or voxel size")
    p = validate_points(points)
    p = p[np.isfinite(p).all(axis=1), :3]
    radius = np.linalg.norm(p, axis=1)
    p = p[(radius >= min_range) & (radius <= max_range)]
    if voxel_size and len(p):
        import open3d as o3d

        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
        p = np.asarray(cloud.voxel_down_sample(voxel_size).points).copy()
    return p
