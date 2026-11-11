"""Conservative 2D / 2.5D geometry from accumulated base_link points.

Evidence rules (no ray casting, no inference into unobserved space):
- OCCUPIED: at least one return inside ``obstacle_z_band`` in the cell.
- FREE: floor returns (z <= floor_z + floor_tol) and no obstacle return in the cell. Low obstacles
  under floor_z + floor_tol are invisible, so FREE means "floor seen", not "traversable".
- UNKNOWN: everything else, including the sensor blind zone and any cell without returns.
"""

import numpy as np

from .observation import FREE, OCCUPIED, UNKNOWN

DEFAULTS = {
    "cell_m": 0.10, "extent_x": (-3.0, 7.0), "extent_y": (-5.0, 5.0),
    "floor_z": -0.33, "floor_tol": 0.10, "obstacle_top": 1.20, "sector_deg": 5.0,
    "blind_radius_m": 0.30,
}


def occupancy(points, cell_m, extent_x, extent_y, floor_z, floor_tol, obstacle_top, **_):
    nx = int(round((extent_x[1] - extent_x[0]) / cell_m))
    ny = int(round((extent_y[1] - extent_y[0]) / cell_m))
    grid = np.full((nx, ny), UNKNOWN, np.int8)
    elev = np.full((nx, ny), np.nan, np.float32)
    p = np.asarray(points, float)
    i = np.floor((p[:, 0] - extent_x[0]) / cell_m).astype(int)
    j = np.floor((p[:, 1] - extent_y[0]) / cell_m).astype(int)
    inside = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
    p, i, j = p[inside], i[inside], j[inside]
    floor = p[:, 2] <= floor_z + floor_tol
    obstacle = ~floor & (p[:, 2] <= floor_z + obstacle_top)
    grid[i[floor], j[floor]] = FREE
    grid[i[obstacle], j[obstacle]] = OCCUPIED  # applied last: an obstacle return always wins
    np.fmax.at(elev, (i, j), p[:, 2].astype(np.float32))
    band = (floor_z + floor_tol, floor_z + obstacle_top)
    return grid, elev, (extent_x[0], extent_y[0]), band


def sectors(points, sector_deg, floor_z, floor_tol, obstacle_top, blind_radius_m, **_):
    """Nearest obstacle range and farthest floor evidence (before it) per azimuth sector."""
    n = int(round(360 / sector_deg))
    p = np.asarray(points, float)
    r = np.hypot(p[:, 0], p[:, 1])
    keep = r >= blind_radius_m
    p, r = p[keep], r[keep]
    s = (np.floor((np.degrees(np.arctan2(p[:, 1], p[:, 0])) % 360) / sector_deg).astype(int)) % n
    floor = p[:, 2] <= floor_z + floor_tol
    obstacle = ~floor & (p[:, 2] <= floor_z + obstacle_top)
    nearest = np.full(n, np.inf)
    np.minimum.at(nearest, s[obstacle], r[obstacle])
    clear = np.full(n, -np.inf)
    floor_idx = np.flatnonzero(floor)
    before = r[floor_idx] < nearest[s[floor_idx]]
    np.maximum.at(clear, s[floor_idx[before]], r[floor_idx[before]])
    nearest[np.isinf(nearest)] = np.nan
    clear[np.isinf(clear)] = np.nan
    return nearest, clear
