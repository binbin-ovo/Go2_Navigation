from pathlib import Path
import numpy as np


def validate_points(points):
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] not in (3, 4):
        raise ValueError("Expected (N,3) XYZ or (N,4) XYZI")
    return p


def load_pointcloud(source, bin_fields=4):
    """BIN is little-endian float32 XYZI (or XYZ with bin_fields=3)."""
    if not isinstance(source, (str, Path)):
        return validate_points(source)
    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return validate_points(np.load(path, allow_pickle=False))
    if suffix == ".bin":
        if bin_fields not in (3, 4) or path.stat().st_size % (4 * bin_fields):
            raise ValueError(
                "BIN byte length must be divisible by 4 * bin_fields (3 or 4)"
            )
        return validate_points(np.fromfile(path, dtype="<f4").reshape(-1, bin_fields))
    if suffix in (".ply", ".pcd"):
        import open3d as o3d

        cloud = o3d.io.read_point_cloud(
            str(path), remove_nan_points=False, remove_infinite_points=False
        )
        if not cloud.has_points():
            raise ValueError(
                "PLY/PCD is empty or unreadable; use (0,3) arrays for empty scans"
            )
        return np.asarray(cloud.points).copy()
    raise ValueError(f"Unsupported point cloud format: {suffix}")
