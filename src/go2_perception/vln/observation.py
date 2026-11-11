"""Versioned, model-agnostic Go2 observation for vision-language navigation (``go2-vln-obs-v1``).

One observation = A) visual, B) geometric, C) robot state, plus clock and calibration provenance.
Every optional field is ``None`` when unknown, never a placeholder. Units are SI (metres, seconds,
radians); every timestamp names its clock domain and its semantics; every geometric array names its
frame. Absent LiDAR returns are UNKNOWN, never free space. 3D object boxes are not part of the
interface and are never free-space evidence.
"""

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "go2-vln-obs-v1"
UNKNOWN, FREE, OCCUPIED = -1, 0, 1
ROBOT_CLOCK = "go2_robot"
HOST_CLOCK = "host_receipt"
TIME_SEMANTICS = {"exposure", "exposure_estimated", "host_receipt", "scan_midpoint", "message_stamp"}


@dataclass
class Stamp:
    """A time in seconds with its clock and meaning. ``uncertainty_s`` is a 1-sigma guess or None."""

    t: float
    clock: str
    semantics: str
    uncertainty_s: float | None = None

    def check(self, name, issues):
        if not np.isfinite(self.t):
            issues.append(f"{name}: non-finite timestamp")
        if not self.clock:
            issues.append(f"{name}: unnamed clock domain")
        if self.semantics not in TIME_SEMANTICS:
            issues.append(f"{name}: unknown time semantics {self.semantics!r}")


@dataclass
class VisualObservation:
    """RGB uint8 HxWx3 in **RGB** channel order (converted from the decoder's BGR)."""

    image: np.ndarray
    frame_id: str
    stamp_host: Stamp
    stamp_robot: Stamp | None = None   # estimated exposure time in the robot clock, if converted
    rectified: bool = False
    intrinsics_status: str = "unknown"  # unknown | synthetic | measured
    K: np.ndarray | None = None
    intrinsics_size: tuple | None = None  # (width, height) the intrinsics belong to
    distortion_model: str | None = None
    D: np.ndarray | None = None
    source: str = "rtp_h264_decoded"
    history: list = field(default_factory=list)  # earlier VisualObservation, oldest first

    @property
    def height(self):
        return int(self.image.shape[0])

    @property
    def width(self):
        return int(self.image.shape[1])


@dataclass
class GeometricObservation:
    """LiDAR-derived geometry in ``frame_id`` (the reference scan's base_link) at ``stamp``.

    ``grid`` is int8 [nx, ny] over x in [origin_x, origin_x + nx*cell) and y likewise: UNKNOWN (-1),
    FREE (0: floor returns and no obstacle return), OCCUPIED (1: a return in ``obstacle_z_band``).
    ``elevation_max`` is the highest return per cell, NaN where unobserved. ``sector_range`` is the
    nearest obstacle distance (XY, metres) per azimuth sector counter-clockwise from +x, NaN where
    the sector had no obstacle return (unknown, not clear). ``sector_clear_to`` is the farthest
    floor return before that obstacle, NaN without floor evidence. Both ignore r < blind_radius_m.
    """

    frame_id: str
    stamp: Stamp
    window_s: float
    frames_used: int
    points: np.ndarray                  # (N,3) float32 metres, obstacle window
    grid: np.ndarray
    cell_m: float
    origin_xy: tuple
    obstacle_z_band: tuple
    elevation_max: np.ndarray
    sector_deg: float
    sector_range: np.ndarray
    sector_clear_to: np.ndarray
    blind_radius_m: float
    display_points: np.ndarray | None = None
    display_window_s: float | None = None
    source_topic: str = "rt/utlidar/cloud_base"
    projected_depth: np.ndarray | None = None  # HxW metres, NaN no return; needs measured calib
    notes: list = field(default_factory=list)


@dataclass
class RobotState:
    stamp: Stamp
    T_odom_base: np.ndarray | None        # 4x4, p_odom = T @ p_base; None = unknown, not identity
    linear_velocity: np.ndarray | None = None
    angular_velocity: np.ndarray | None = None
    velocity_frame: str | None = None
    odom_source: str = "rt/utlidar/robot_odom"


@dataclass
class Go2VLNObservation:
    visual: VisualObservation | None
    geometric: GeometricObservation | None
    robot: RobotState | None
    instruction: str | None = None
    clock_epoch: str | None = None   # robot clock resets per boot: stamps compare only within one epoch
    host_minus_robot_s: float | None = None
    calibration: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION


def validate(obs, max_skew_s=0.15, strict=False):
    """Return a list of issues; with ``strict`` raise ValueError if there are any."""
    issues = []
    if obs.schema_version != SCHEMA_VERSION:
        issues.append(f"schema_version {obs.schema_version!r} != {SCHEMA_VERSION!r}")
    v, g, r = obs.visual, obs.geometric, obs.robot
    if v is not None:
        img = v.image
        if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] != 3 or img.dtype != np.uint8:
            issues.append("visual.image must be uint8 HxWx3 RGB")
        if not v.frame_id:
            issues.append("visual.frame_id missing")
        v.stamp_host.check("visual.stamp_host", issues)
        if v.stamp_host.clock != HOST_CLOCK or v.stamp_host.semantics != "host_receipt":
            issues.append("visual.stamp_host must be host_receipt time in the host_receipt clock")
        if v.stamp_robot is not None:
            v.stamp_robot.check("visual.stamp_robot", issues)
        if v.intrinsics_status not in {"unknown", "synthetic", "measured"}:
            issues.append("visual.intrinsics_status invalid")
        if v.intrinsics_status == "unknown" and (v.K is not None or v.D is not None):
            issues.append("visual: K/D present but intrinsics_status is unknown")
        if v.intrinsics_status != "unknown":
            K = None if v.K is None else np.asarray(v.K, float)
            if K is None or K.shape != (3, 3) or not np.isfinite(K).all():
                issues.append("visual.K must be finite 3x3 when intrinsics are known")
            if v.intrinsics_size is None or tuple(v.intrinsics_size) != (v.width, v.height):
                issues.append(f"visual: intrinsics are for {v.intrinsics_size}, image is "
                              f"{(v.width, v.height)}; rescale them with tensors.resize")
            if v.distortion_model is None and not v.rectified:
                issues.append("visual: raw image needs a distortion model")
        for i, h in enumerate(v.history):
            if h.stamp_host.t >= v.stamp_host.t:
                issues.append(f"visual.history[{i}] is not older than the current frame")
    if g is not None:
        g.stamp.check("geometric.stamp", issues)
        _finite_points(g.points, "geometric.points", issues)
        if g.display_points is not None:
            _finite_points(g.display_points, "geometric.display_points", issues)
        if g.grid.dtype != np.int8 or g.grid.ndim != 2 or not np.isin(g.grid, (UNKNOWN, FREE, OCCUPIED)).all():
            issues.append("geometric.grid must be 2D int8 with values -1/0/1")
        if g.elevation_max.shape != g.grid.shape:
            issues.append("geometric.elevation_max shape differs from grid")
        if not (g.cell_m > 0 and g.window_s > 0):
            issues.append("geometric cell size and window must be positive")
        n_sectors = int(round(360 / g.sector_deg))
        if g.sector_range.shape != (n_sectors,) or g.sector_clear_to.shape != (n_sectors,):
            issues.append("geometric sector arrays must have 360/sector_deg entries")
        if np.any(g.sector_range[np.isfinite(g.sector_range)] < 0):
            issues.append("geometric.sector_range has negative distances")
        if not g.frame_id:
            issues.append("geometric.frame_id missing")
        if g.projected_depth is not None:
            if v is None or g.projected_depth.shape != (v.height, v.width):
                issues.append("geometric.projected_depth must match the image size")
            if obs.calibration.get("extrinsics_status") != "measured" and \
                    obs.calibration.get("allow_synthetic") is not True:
                issues.append("projected_depth present without measured extrinsics")
    if r is not None:
        r.stamp.check("robot.stamp", issues)
        if r.T_odom_base is not None:
            try:
                from ..core.transforms import validate_transform
                validate_transform(r.T_odom_base)
            except ValueError as e:
                issues.append(f"robot.T_odom_base: {e}")
        if (r.linear_velocity is not None or r.angular_velocity is not None) and not r.velocity_frame:
            issues.append("robot velocities need velocity_frame")
    # Cross-stream timing: only stamps in one clock domain may be compared.
    if g is not None and r is not None and g.stamp.clock == r.stamp.clock and \
            abs(g.stamp.t - r.stamp.t) > max_skew_s:
        issues.append(f"robot state {r.stamp.t - g.stamp.t:+.3f} s from geometry exceeds {max_skew_s} s")
    if v is not None and g is not None:
        if v.stamp_robot is None:
            issues.append("clock_domain_mismatch: visual has only a host-receipt stamp; "
                          "convert it to the robot clock before pairing with geometry")
        elif v.stamp_robot.clock != g.stamp.clock:
            issues.append("clock_domain_mismatch: visual.stamp_robot and geometric.stamp differ")
        elif abs(v.stamp_robot.t - g.stamp.t) > max_skew_s:
            issues.append(f"visual-geometry skew {v.stamp_robot.t - g.stamp.t:+.3f} s exceeds {max_skew_s} s")
    if strict and issues:
        raise ValueError("Invalid Go2VLNObservation:\n  " + "\n  ".join(issues))
    return issues


def _finite_points(p, name, issues):
    if not isinstance(p, np.ndarray) or p.ndim != 2 or p.shape[1] != 3:
        issues.append(f"{name} must be (N,3)")
    elif not np.isfinite(p).all():
        issues.append(f"{name} has non-finite values")


# ---------------------------------------------------------------- serialization (npz, no pickle)

def save(obs, path):
    """Write one observation to ``path`` (.npz). Arrays are stored natively, metadata as JSON."""
    arrays = {}
    meta = _to_meta(obs, "", arrays)
    meta["schema_version"] = obs.schema_version
    arrays["__meta__"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    np.savez_compressed(path, **arrays)


def load(path):
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    meta = json.loads(arrays.pop("__meta__").tobytes().decode())
    if meta.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema {meta.get('schema_version')!r}")
    return _from_meta(Go2VLNObservation, meta, arrays)


_TYPES = {"visual": VisualObservation, "geometric": GeometricObservation, "robot": RobotState}
_STAMPS = {"stamp", "stamp_host", "stamp_robot"}


def _to_meta(obj, prefix, arrays):
    out = {}
    for f in fields(obj):
        value, key = getattr(obj, f.name), prefix + f.name
        if isinstance(value, np.ndarray):
            arrays[key] = value
            out[f.name] = {"__array__": key}
        elif isinstance(value, Stamp):
            out[f.name] = {"t": value.t, "clock": value.clock, "semantics": value.semantics,
                           "uncertainty_s": value.uncertainty_s}
        elif f.name == "history":
            out[f.name] = [_to_meta(h, f"{key}.{i}.", arrays) for i, h in enumerate(value)]
        elif hasattr(value, "__dataclass_fields__"):
            out[f.name] = _to_meta(value, key + ".", arrays)
        elif isinstance(value, tuple):
            out[f.name] = {"__tuple__": list(value)}
        else:
            out[f.name] = value
    return out


def _from_meta(cls, meta, arrays):
    kwargs = {}
    for f in fields(cls):
        if f.name not in meta:
            continue
        value = meta[f.name]
        if isinstance(value, dict) and "__array__" in value:
            value = arrays[value["__array__"]]
        elif isinstance(value, dict) and "__tuple__" in value:
            value = tuple(value["__tuple__"])
        elif f.name in _STAMPS and value is not None:
            value = Stamp(**value)
        elif f.name == "history":
            value = [_from_meta(VisualObservation, h, arrays) for h in value]
        elif f.name in _TYPES and value is not None:
            value = _from_meta(_TYPES[f.name], value, arrays)
        kwargs[f.name] = value
    return cls(**kwargs)
