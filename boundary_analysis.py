# coding: utf-8
"""
Boundary-violation analysis for the ASTM egg plots and the GB contour chart.

Each reference curve (an egg/contour outline) is a CLOSED loop. A measured
point is "outside" that boundary when it is NOT inside the loop. The rules:

  • For a curve labelled with a duration T (e.g. ASTM 0.2s, GB 0.1s): the
    measured trace may not have a CONTIGUOUS run of points outside that
    boundary lasting longer than T.
  • For a "0 s" restraint curve: NO measured point may be outside (a single
    outside point is already a violation).

This module provides the shared core (``find_boundary_violations``) plus a
small helper to turn a QLineSeries outline into a ``matplotlib.path.Path``.
The chart widgets build their boundary Paths from their reference series and
call the core to detect which outside-runs exceed each curve's threshold.
"""

from __future__ import annotations

import numpy as np
from matplotlib.path import Path


def series_to_path(series) -> Path | None:
    """Build a closed ``Path`` from a QLineSeries' points.

    Returns None if the series has fewer than 3 points (not a real loop).
    """
    try:
        pts = series.points()
    except Exception:
        return None
    if len(pts) < 3:
        return None
    verts = [(float(p.x()), float(p.y())) for p in pts]
    return Path(verts)


def find_boundary_violations(measured_xy: np.ndarray, t_arr: np.ndarray,
                             paths: list[Path], threshold: float
                             ) -> list[tuple[int, int]]:
    """Find contiguous outside-runs that violate a boundary's threshold.

    Parameters
    ----------
    measured_xy : (N, 2) array of the measured trace's (x, y) for one chart.
    t_arr : (N,) array of sample times (seconds), aligned with measured_xy.
    paths : list of ``Path`` for ONE boundary curve (a point is "inside" if
        ANY path contains it; usually a single closed loop, occasionally two
        symmetric halves — but our egg/contour outlines are already single
        closed loops).
    threshold : the boundary's sustained-duration limit in seconds. ``0`` means
        ANY outside point is a violation (the restraint "0 s" rule).

    Returns
    -------
    list of ``(i_start, i_end)`` index pairs for each violating outside-run.
    A run violates when its duration ``t[i_end] - t[i_start]`` strictly exceeds
    ``threshold``; for ``threshold == 0`` any outside point qualifies (a
    single-point run has duration 0, which is > 0 is false, so we special-case
    it: every outside point is its own violation when threshold == 0).
    """
    n = len(measured_xy)
    if n == 0 or not paths:
        return []

    # inside = contained by ANY of the boundary paths.
    inside = np.zeros(n, dtype=bool)
    for p in paths:
        inside |= p.contains_points(measured_xy)
    outside = ~inside
    if not outside.any():
        return []

    # Special case: threshold == 0 → any outside point is a violation.
    # Return contiguous outside runs (collapsed, not one-per-point) so the
    # caller draws one marker block + one label per run instead of one label
    # per sample (which would be unreadable for a sustained breach).
    if threshold <= 0:
        idx0 = np.flatnonzero(outside)
        if len(idx0) == 0:
            return []
        breaks0 = np.where(np.diff(idx0) > 1)[0]
        groups0 = np.split(idx0, breaks0 + 1)
        return [(int(g[0]), int(g[-1])) for g in groups0]

    # General case: find contiguous outside-runs, keep those whose duration
    # strictly exceeds the threshold.
    idx = np.flatnonzero(outside)
    breaks = np.where(np.diff(idx) > 1)[0]
    groups = np.split(idx, breaks + 1)
    violations = []
    for grp in groups:
        if len(grp) < 2:
            # Single outside point: duration 0, never > threshold (>0).
            continue
        start, end = int(grp[0]), int(grp[-1])
        if t_arr[end] - t_arr[start] > threshold:
            violations.append((start, end))
    return violations
