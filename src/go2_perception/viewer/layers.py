"""Named, toggleable render layers and the three presentation modes."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

# name -> (key, label). Keys are for the interactive viewer.
LAYER_INFO = {
    "occupancy_footprint": ("o", "background occupancy (floor footprint)"),
    "fresh_boxes": ("5", "5A fresh boxes (solid)"),
    "stale_boxes": ("6", "5A stale boxes (dashed, fading)"),
    "lost_coasting": ("l", "lost / coasting / unseen states"),
    "motion": ("9", "motion classes and velocities"),
    "cv_path": ("p", "constant-velocity path (prediction)"),
    "labels": ("b", "labels"),
    "track_ids": ("i", "object IDs"),
    "robot_path": ("w", "robot path"),
    "accumulated": ("2", "accumulated LiDAR (1.0 s)"),
    "background_voxels": ("7", "background voxels (3D)"),
    "raw_incremental": ("1", "raw incremental LiDAR (current scan)"),
    "phase3_detections": ("3", "original Phase 3 detections"),
    "phase3_tracks": ("4", "original Phase 3 tracks"),
    "clusters": ("8", "5A clusters and centroids"),
    "raw_detection_boxes": ("r", "5A raw observed boxes (current support)"),
    "point_age": ("0", "colour points by age"),
    "trails": ("t", "observed trails"),
    "retrospective_gt": ("g", "RETROSPECTIVE annotation (uses whole recording)"),
}


@dataclass
class Layers:
    occupancy_footprint: bool = True
    fresh_boxes: bool = True
    stale_boxes: bool = True
    lost_coasting: bool = True
    motion: bool = True
    cv_path: bool = True
    labels: bool = True
    track_ids: bool = False
    robot_path: bool = True
    accumulated: bool = False
    background_voxels: bool = False
    raw_incremental: bool = False
    phase3_detections: bool = False
    phase3_tracks: bool = False
    clusters: bool = False
    raw_detection_boxes: bool = False
    point_age: bool = False
    trails: bool = False
    retrospective_gt: bool = False

    def toggle(self, name):
        if name not in LAYER_INFO:
            raise KeyError(name)
        setattr(self, name, not getattr(self, name))
        return getattr(self, name)

    def toggle_key(self, key):
        for name, (k, _) in LAYER_INFO.items():
            if k == key:
                return name, self.toggle(name)
        return None, None

    def enabled(self):
        return [f.name for f in fields(self) if getattr(self, f.name)]


MODES = ("clean", "hybrid", "debug")


def mode_layers(mode):
    """Preset for ``clean`` (default, no raw points), ``hybrid`` or ``debug``.

    The retrospective annotation layer is never part of a preset; it must be enabled
    explicitly and is labelled on screen because it is not causal runtime information.
    """
    base = Layers()
    if mode == "clean":
        return base
    if mode == "hybrid":
        return replace(base, accumulated=True, background_voxels=True)
    if mode == "debug":
        on = {name: True for name in LAYER_INFO if name != "retrospective_gt"}
        return replace(base, **on)
    raise ValueError(f"Unknown mode {mode!r}; choose from {MODES}")
