# coding: utf-8
"""
Acceleration Zone chart — single chart with coloured zone regions and a
measured data overlay.  Replaces AccZonePlot (matplotlib).
"""

from __future__ import annotations

import numpy as np
from matplotlib.path import Path

from PySide6.QtCharts import (
    QChart, QChartView, QLineSeries, QAreaSeries, QValueAxis,
    QScatterSeries,
)
from PySide6.QtCore import Qt, QPointF, QTimer
from PySide6.QtGui import QColor, QPen, QFont, QCursor
from PySide6.QtWidgets import QWidget, QLabel

from chart_base import BaseChartView, INTERACTION_ZOOM, INTERACTION_PAN, INTERACTION_DEFAULT
from chart_utils import create_line_series, clear_chart, reset_chart_view
from theme import LINE_WIDTH_DEFAULT, chart_tooltip_style


# ---------------------------------------------------------------------------
# Range constants
# ---------------------------------------------------------------------------
# The zone fills extend to the FULL range (data can exceed the standard
# envelope), but the INITIAL view and reset_view() snap to the STANDARD range
# (the original GB 8408 envelope) so the chart opens focused on the meaningful
# region. Users can still pan/zoom out to see the extended fills + outlying data.
ACCZONE_STD_X = (-2.0, 2.0)
ACCZONE_STD_Y = (-1.2, 1.2)
ACCZONE_FULL_X = (-5.0, 5.0)
ACCZONE_FULL_Y = (-3.0, 8.0)


# ---------------------------------------------------------------------------
# Zone block definitions
# ---------------------------------------------------------------------------
# Each zone is decomposed into 1 or 2 SIMPLE x-monotone blocks (convex
# polygons) so QAreaSeries(upper, lower) fills each one directly — no scanline
# decomposition, no jump-splitting, no seam artefacts. Range X∈[-5,5], Y∈[-3,8].
#
# Internal diagonal boundaries (shared between adjacent blocks, kept identical
# so blocks stay flush):
#   D1: (-0.2, 0.7) → (0.2, 0.2)        Z1 / Z2 left border
#   D2: (-0.7, 0.2) → (0, 0) → (0.7,-0.2)  Z4 notch + Z5 upper edge
#
#   Zone 1: 1 block     Zone 2: 1 block
#   Zone 3: 2 blocks (left tall, right thin)   Zone 4: 2 blocks (left tall, right thin)
#   Zone 5: 1 block
# Total: 7 blocks. Vertices listed clockwise from the top-left.

_ZONES = [
    # --- Zone 1 (greenyellow): top-right band, left edge = D1 diagonal ---
    # In the rect x∈[-0.2,0] y∈[0.7,1], the left boundary follows the diagonal
    # (-0.2,1)→(0,0.7): the upper-right triangle of that rect stays in Z1, the
    # lower-left triangle goes to Z2 (see below).
    {
        "verts": [(-0.2, 8.0), (-0.2, 1.0), (0.0, 0.7), (0.2, 0.7),
                  (0.2, 0.2), (5.0, 0.2), (5.0, 8.0)],
        "color": QColor("greenyellow"),
        "label": "Zone 1", "lx": 1.7, "ly": 0.8,
    },
    # --- Zone 2 (yellow): middle band, right edge = D1 diagonal ---
    # Gains the lower-left triangle of rect x∈[-0.2,0] y∈[0.7,1]: its right
    # boundary in that band follows the diagonal (0,0.7)→(-0.2,1).
    {
        "verts": [(-0.7, 8.0), (-0.7, 0.2), (0.2, 0.2), (0.2, 0.7),
                  (0.0, 0.7), (-0.2, 1.0), (-0.2, 8.0)],
        "color": QColor("yellow"),
        "label": "Zone 2", "lx": -0.4, "ly": 0.8,
    },
    # --- Zone 3 left (orange): tall strip x∈[-1.2,-0.7] ---
    {
        "verts": [(-1.2, 8.0), (-1.2, 0.2), (-0.7, 0.2), (-0.7, 8.0)],
        "color": QColor("orange"),
        "label": "Zone 3", "lx": -0.95, "ly": 0.8,
    },
    # --- Zone 3 right (orange): thin strip y∈[0,0.2], x∈[-0.7,5] ---
    {
        "verts": [(-0.7, 0.2), (5.0, 0.2), (5.0, 0.0), (0.0, 0.0)],
        "color": QColor("orange"),
        "label": "Zone 3", "lx": 2.0, "ly": 0.1,
    },
    # --- Zone 4 left (salmon): tall rectangle x∈[-5,-1.2] ---
    {
        "verts": [(-5.0, 8.0), (-5.0, 0.0), (-1.2, 0.0), (-1.2, 8.0)],
        "color": QColor("salmon"),
        "label": "Zone 4", "lx": -1.7, "ly": 0.8,
    },
    # --- Zone 4 right (salmon): thin band around y=0, x∈[-1.2,5], with notch ---
    {
        "verts": [(-1.2, 0.2), (-1.2, 0.0), (5.0, 0.0), (5.0, -0.2),
                  (0.7, -0.2), (0.0, 0.0), (-0.7, 0.2)],
        "color": QColor("salmon"),
        "label": "Zone 4", "lx": 1.0, "ly": -0.12,
    },
    # --- Zone 5 (red): bottom band, upper edge = D2 diagonal ---
    {
        "verts": [(-5.0, 0.0), (0.0, 0.0), (0.7, -0.2), (5.0, -0.2),
                  (5.0, -3.0), (-5.0, -3.0)],
        "color": QColor("red"),
        "label": "Zone 5", "lx": -1.0, "ly": -1.5,
    },
]


# Auto-zone classification: a ride is assigned to a single zone if it has a
# CONTIGUOUS run of (x,z) points all inside that zone's polygon(s) lasting
# strictly longer than this threshold. Zones are checked 5 → 1; first match
# wins (most-severe / lowest-priority zone takes precedence per the spec).
AUTO_ZONE_DUR = 0.2   # seconds


def _zone_number(label: str) -> int:
    """Extract the trailing integer from a zone label ("Zone 3" → 3)."""
    return int("".join(ch for ch in label if ch.isdigit()))


# Zone → list of polygon vertex lists (one entry per block; Zone 3/4 have two
# blocks each). Derived from _ZONES so there's a single source of truth for the
# geometry — classify_zone() builds a matplotlib Path per block from these.
_ZONE_POLYS: dict[int, list[list[tuple]]] = {}
for _z in _ZONES:
    _ZONE_POLYS.setdefault(_zone_number(_z["label"]), []).append(
        [tuple(float(c) for c in v) for v in _z["verts"]])



def _block_to_area_series(verts, fill_color: QColor):
    """Fill a single simple x-monotone polygon with one QAreaSeries.

    Splits the polygon boundary into an upper chain (left-most vertex → right
    most, along the top) and a lower chain (right most → left most, along the
    bottom), then ``QAreaSeries(upper, lower)`` fills between them directly.
    Every block in ``_ZONES`` is convex / x-monotone, so this never folds and
    never needs scanline sampling. Returns ``(area, upper, lower, outline)``
    where *outline* is a closed QLineSeries of the raw vertices drawn on top
    for crisp borders.
    """
    n = len(verts)
    if n < 3:
        return None, None

    # Find the indices of the left-most and right-most vertices (by x; ties
    # broken by y so the split is deterministic).
    order = sorted(range(n), key=lambda i: (verts[i][0], verts[i][1]))
    li = order[0]            # left-most
    ri = order[-1]           # right-most

    # Walk the boundary clockwise from li to ri → upper chain; ri back to li
    # (wrapping) → lower chain. Both end up x-monotone for our blocks.
    def walk(start: int, end: int) -> list:
        out = []
        i = start
        while True:
            out.append(verts[i])
            if i == end:
                break
            i = (i + 1) % n
        return out

    upper_pts = walk(li, ri)
    lower_pts = walk(ri, li)

    upper = QLineSeries()
    lower = QLineSeries()
    for x, y in upper_pts:
        upper.append(QPointF(float(x), float(y)))
    for x, y in lower_pts:
        lower.append(QPointF(float(x), float(y)))
    # The QAreaSeries border is drawn ALONG these boundary lines. In Qt Charts
    # 6.x the area's own pen is ignored, so we must camouflage the border by
    # making the boundary lines themselves (and the area borderColor/pen) the
    # same colour as the fill — that way any line drawn is invisible.
    upper.setPen(QPen(fill_color, 1, Qt.PenStyle.SolidLine))
    lower.setPen(QPen(fill_color, 1, Qt.PenStyle.SolidLine))

    area = QAreaSeries(upper, lower)
    area.setColor(fill_color)
    # QAreaSeries draws its OWN border line along the upper/lower boundary —
    # and Qt Charts 6.x IGNORES NoPen for it, so the border shows up as stray
    # dashed/diagonal lines on slanted zone edges (the bug). The only reliable
    # way to hide it is to make the border the SAME colour as the fill, so it's
    # invisible against its own area. Set BOTH borderColor and pen to the fill
    # colour so whichever code path Qt uses, the line is camouflaged.
    area.setBorderColor(fill_color)
    area.setPen(QPen(fill_color, 1, Qt.PenStyle.SolidLine))

    # Closed outline of the raw vertices, drawn on top for crisp straight edges
    # (QAreaSeries' own border can render vertical edges diagonally).
    outline = QLineSeries()
    for x, y in verts:
        outline.append(QPointF(float(x), float(y)))
    outline.append(QPointF(float(verts[0][0]), float(verts[0][1])))   # close
    outline.setPen(QPen(QColor("black"), 0.5))

    # Return upper/lower too: in PySide6 the QAreaSeries Python wrapper can be
    # GC'd after chart.addSeries takes C++ ownership, which would delete the
    # boundary lines. Callers keep strong refs to all of them.
    return area, upper, lower, outline


class AccZoneChartView(BaseChartView):
    """Acceleration Zone chart with coloured regions and data overlay."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.annot_mode: str = "Single"
        self._data_line: QLineSeries | None = None
        self._kept_refs: list = []  # prevent GC of QLineSeries used by QAreaSeries
        # When True, resizeEvent adjusts the Y range so X and Y units occupy
        # equal pixel lengths (equal aspect ratio) — zone shapes aren't
        # distorted by the window's aspect.
        self._equal_aspect: bool = True
        # Default Y centre for the home view. The zone content sits mostly
        # above y=0 (Zone 1/2/3 occupy y up to 8; the bulk of interest is
        # 0..1.2), so centring on 0.5 keeps the upper region visible without
        # wasting half the view on the lower Zone 5 band.
        self._y_centre_home: float = 0.5
        # Auto-zone state: zone number → list of QAreaSeries for that zone's
        # block(s); filled in initplotxyz so highlight_zone() can recolour.
        self._zone_areas: dict[int, list[QAreaSeries]] = {}
        # Cached matplotlib Paths per zone (one per block) for point-in-polygon.
        self._zone_paths: dict[int, list[Path]] = {}
        # Saved original zone colours so highlight can restore them.
        self._zone_orig_colors: dict[int, QColor] = {}
        # Last auto-zone result (None = not classified / no match).
        self._auto_zone_result: int | None = None
        # Overlay label widget for the "Auto Zone: X" readout.
        self._zone_overlay: QLabel | None = None
        # Accent-outline QLineSeries drawn over the selected zone (highlight).
        self._highlight_outlines: list[QLineSeries] = []
        # Last classification's run window + trajectory, cached so the Analyze
        # toggle can re-draw the big-point markers + label without recompute.
        self._last_data = None
        self._last_run: tuple[int, float, float] | None = None  # (zone,t0,t1)
        # Whether the analysis overlays (outline + big-point markers + label)
        # are currently shown. Toggled by the AccZone "Analyze" button.
        self._analysis_active: bool = False
        # Big-point scatter series + run-label widgets created by mark_run.
        self._run_marker: QScatterSeries | None = None
        self._run_marker_refs: list = []   # keep series/labels alive (Qt GC)
        # (QLabel, QPointF anchor, dx, dy) so the run label follows on zoom/pan.
        self._run_label_anchor: tuple | None = None
        self.initplotxyz()

    def resizeEvent(self, event) -> None:
        """Re-apply equal aspect after resize so X/Y stay 1:1 in pixels.

        Preserves the user's CURRENT Y centre (so a panned view stays put on
        resize); only reset_view / first-show pin the centre to 0.5.
        """
        super().resizeEvent(event)
        if self._equal_aspect:
            self._apply_equal_aspect(keep_current_centre=True)
        # Re-clamp the auto-zone overlay label to the top-right corner on resize.
        if self._zone_overlay is not None:
            vw = max(self.viewport().width(), 1)
            self._zone_overlay.move(
                max(vw - self._zone_overlay.width() - 12, 4), 8)
            self._zone_overlay.raise_()
        # Re-clamp the analysis run-label to its anchor on resize too.
        self._reposition_run_label()

    def _apply_equal_aspect(self, keep_current_centre: bool = False,
                            centre: float | None = None) -> None:
        """Adjust the Y axis range so one X unit == one Y unit in pixels.

        Keeps the X range fixed (whatever the user panned/zoomed to) and sets
        the Y span from the plot area's pixel aspect. The Y centre is taken
        from *centre* if given, else from the current Y range when
        ``keep_current_centre`` is True, else from ``_y_centre_home``.
        """
        chart = self.chart()
        if chart is None or self._axis_x is None or self._axis_y is None:
            return
        area = chart.plotArea()
        w = area.width()
        h = area.height()
        if w <= 0 or h <= 0:
            return
        x_lo, x_hi = self._axis_x.min(), self._axis_x.max()
        x_span = x_hi - x_lo
        if x_span <= 0:
            return
        # pixels per X unit
        px_per_x = w / x_span
        # desired Y span so that px_per_y == px_per_x
        y_span = h / px_per_x
        # resolve the Y centre per the call flags
        if centre is not None:
            y_mid = centre
        elif keep_current_centre:
            y_mid = (self._axis_y.min() + self._axis_y.max()) / 2.0
        else:
            y_mid = self._y_centre_home
        self._axis_y.setRange(y_mid - y_span / 2.0, y_mid + y_span / 2.0)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initplotxyz(self) -> None:
        # Drop any auto-zone highlight overlay + analysis overlays from the
        # previous chart, and reset _analysis_active. Keeping it True here used
        # to leave the AccZone Analyze button out of sync with the chart after
        # a Clear All (button unchecked but analysis silently re-applied on the
        # next plot). _generate_acczone re-runs toggle_analysis(True) when the
        # button is checked, so re-enabling on a fresh plot is handled there.
        self.clear_highlight()
        self._remove_analysis_overlays()
        self._analysis_active = False
        # Reset zone→area tracking + last-run cache for the new chart.
        self._zone_areas = {}
        self._zone_paths = {}
        self._zone_orig_colors = {}
        self._auto_zone_result = None
        self._last_data = None
        self._last_run = None

        chart = QChart()
        chart.setTitle("Acceleration Zones")
        chart.legend().hide()
        chart.layout().setContentsMargins(4, 4, 4, 4)

        # Add zone fills. Each block is a single x-monotone polygon filled by
        # one QAreaSeries(upper, lower). We do NOT draw a separate outline:
        # the original per-block outline QLineSeries double-drew every shared
        # internal edge (D1/D2 diagonals, vertical borders), which anti-aliased
        # into stray light/diagonal dashes. With block-level QAreaSeries the
        # upper/lower lines ARE the true geometry, so vertical edges stay
        # vertical and adjacent fills meet cleanly without any outline needed.
        for zone in _ZONES:
            area, upper, lower, outline = _block_to_area_series(
                zone["verts"], zone["color"])
            if area is None:
                continue
            # Keep strong Python refs: PySide6 may drop the QAreaSeries wrapper
            # after addSeries takes C++ ownership, which would delete the
            # upper/lower boundary lines too.
            self._kept_refs.extend([area, upper, lower])
            chart.addSeries(area)

            # Register this area under its zone number so highlight_zone() can
            # recolour it, and cache a matplotlib Path for the auto-classify
            # point-in-polygon test. (Zone 3/4 have two blocks/areas; both
            # register under the same zone number.)
            zn = _zone_number(zone["label"])
            self._zone_areas.setdefault(zn, []).append(area)
            if zn not in self._zone_orig_colors:
                self._zone_orig_colors[zn] = QColor(zone["color"])

        # Pre-build the per-block matplotlib Paths for classify_zone(). Done
        # once here (not per classify call) so the point-in-polygon test is fast.
        for zn, blocks in _ZONE_POLYS.items():
            self._zone_paths[zn] = [Path(b) for b in blocks]

        # Create axes. The INITIAL view snaps to the standard envelope
        # (ACCZONE_STD_*); the zone fills extend to the full range, so users
        # can pan/zoom out to reveal them.
        axis_x = QValueAxis()
        axis_x.setRange(*ACCZONE_STD_X)
        axis_x.setTitleText("Front <=> Back")
        chart.addAxis(axis_x, Qt.AlignmentFlag.AlignBottom)

        axis_y = QValueAxis()
        axis_y.setRange(*ACCZONE_STD_Y)
        axis_y.setTitleText("Up <=> Down")
        chart.addAxis(axis_y, Qt.AlignmentFlag.AlignLeft)

        # Attach axes to all series (area fills + outlines)
        for s in chart.series():
            s.attachAxis(axis_x)
            s.attachAxis(axis_y)

        # Zone labels as scatter series with point labels.
        # Bug fixed: the original used __import__("PySide6.QtGui", ...)
        # to construct a QFont and left points visible, which (a) imported
        # the module wrong and (b) rendered stray dots alongside the text.
        # Now we import QFont at the top, hide the marker dot, and only
        # show the literal zone label text.
        for zone in _ZONES:
            if zone["label"]:
                label_series = QScatterSeries()
                label_series.append(QPointF(zone["lx"], zone["ly"]))
                label_series.setMarkerSize(0.01)        # effectively invisible dot
                label_series.setPointsVisible(False)    # no dot — text only
                label_series.setPointLabelsVisible(True)
                label_series.setPointLabelsFormat(zone["label"])
                label_series.setPointLabelsColor(QColor("black"))
                label_series.setPointLabelsFont(
                    QFont("Segoe UI", 9, QFont.Weight.Bold)
                )
                chart.addSeries(label_series)
                label_series.attachAxis(axis_x)
                label_series.attachAxis(axis_y)

        self._axis_x = axis_x
        self._axis_y = axis_y
        self.setChart(chart)
        self._data_line = None
        # Sampling rate of the most recently plotted dataset (Hz). Default 500;
        # addplotxyz updates it from data.fs. Used by the click tooltip to turn
        # a point index into a time, instead of the old hardcoded 0.002 s.
        self._last_fs = 500
        # Reposition the analysis run-label when the axes change (zoom/pan).
        axis_x.rangeChanged.connect(self._reposition_run_label)
        axis_y.rangeChanged.connect(self._reposition_run_label)
        # Defer the aspect fix until the view has a real geometry (plotArea is
        # empty during construction). The first resizeEvent will apply it.
        QTimer.singleShot(0, self._apply_equal_aspect)
        # Recreate this view's viewport so no stale pixels from the previous
        # (populated) chart linger on a hidden page — a reset must look exactly
        # like the app's initial state.
        reset_chart_view(self)

    # ------------------------------------------------------------------
    # Data plotting
    # ------------------------------------------------------------------

    def addplotxyz(self, data, color: str | None = None, classification=None) -> None:
        """Overlay measured acceleration data on the zone chart.

        ``color`` (hex) drives the measured trace; falls back to legacy blue.
        After drawing, runs the auto-zone classification and highlights the
        result zone (see classify_zone / highlight_zone).
        """
        # Remove previous data line
        if self._data_line is not None:
            try:
                self.chart().removeSeries(self._data_line)
            except Exception:
                pass

        measured = color if color else "blue"
        x_arr = data.iloc[:, 1].values
        z_arr = data.iloc[:, 3].values

        # Remember the source sampling rate so the click tooltip can convert a
        # point index to a real time. Falls back to 500 Hz if the dataset has no
        # detected fs (the old behaviour hardcoded 0.002 s = 1/500).
        self._last_fs = getattr(data, "fs", None) or self._last_fs

        self._data_line = create_line_series(
            x_arr, z_arr, color=measured, name="Measured Data",
            width=LINE_WIDTH_DEFAULT, points_visible=True,
        )
        self.chart().addSeries(self._data_line)
        self._data_line.attachAxis(self._axis_x)
        self._data_line.attachAxis(self._axis_y)

        # Click annotation
        self._data_line.clicked.connect(self._on_data_clicked)

        # Cache the data + classify, so the Analyze toggle can (re)draw the
        # big-point markers + label without recomputing. The overlays are only
        # shown if analysis is currently active (default OFF).
        self._last_data = data
        zn, t0, t1 = classification if classification is not None else self.classify_zone(data)
        self._last_run = (zn, t0, t1) if zn is not None else None
        if self._analysis_active:
            self._apply_analysis()

    # ------------------------------------------------------------------
    # Auto-zone classification
    # ------------------------------------------------------------------

    def toggle_analysis(self, active: bool) -> None:
        """Show / hide the auto-zone analysis overlays.

        ``active`` True → classify the last plotted ride, draw the accent
        outline over the result zone + big-point markers on the satisfying run
        + a "Zone X, Duration ..s to ..s" label. False → remove all of those
        (the zone fills themselves are never touched).
        """
        self._analysis_active = active
        if active:
            self._apply_analysis()
        else:
            self._remove_analysis_overlays()

    def _apply_analysis(self) -> None:
        """Draw the accent outline + big-point markers + run label for the
        last classification result (kept in self._last_run)."""
        self._remove_analysis_overlays()
        if self._last_run is None:
            # No qualifying run — still show the overlay so the user sees "—".
            self.highlight_zone(None)
            return
        zn, t0, t1 = self._last_run
        self.highlight_zone(zn)
        # Zone 1 is the catch-all (zones 5→2 didn't qualify): t0/t1 are None
        # and there's no meaningful run window to mark, so draw only the
        # outline (already done above) and skip the big-point markers + label.
        if t0 is not None and t1 is not None:
            self.mark_run(self._last_data, zn, t0, t1)

    def _remove_analysis_overlays(self) -> None:
        """Strip the accent outlines, big-point markers, run label + overlay."""
        # Accent outlines + overlay label.
        self.clear_highlight()
        # Big-point scatter + run label widgets.
        chart = self.chart()
        for ref in self._run_marker_refs:
            try:
                if isinstance(ref, QLabel):
                    ref.deleteLater()
                else:
                    chart.removeSeries(ref)
            except Exception:
                pass
        self._run_marker_refs = []
        self._run_marker = None

    def mark_run(self, data, zone: int, t_start: float, t_end: float) -> None:
        """Draw enlarged markers over the points of the qualifying run + label.

        ``zone`` is the classified zone number; ``t_start`` / ``t_end`` bound
        the contiguous run that satisfied the >0.2s duration rule. The run's
        (x,z) points are re-plotted as a large scatter (on top of the measured
        line) and a label "Zone X, Duration T0s to T1s" is anchored near the
        run's centroid, following it on zoom/pan.
        """
        try:
            t_arr = np.asarray(data.iloc[:, 0].values, dtype=float)
            x_arr = np.asarray(data.iloc[:, 1].values, dtype=float)
            z_arr = np.asarray(data.iloc[:, 3].values, dtype=float)
        except Exception:
            return
        # Run = the contiguous span of indices whose time falls in [t0, t1].
        mask = (t_arr >= t_start) & (t_arr <= t_end)
        if not mask.any():
            return

        marker = QScatterSeries()
        marker.setMarkerSize(14)
        marker.setColor(QColor("#0078D4"))
        marker.setBorderColor(QColor("white"))
        for i in np.flatnonzero(mask):
            marker.append(QPointF(float(x_arr[i]), float(z_arr[i])))
        self.chart().addSeries(marker)
        marker.attachAxis(self._axis_x)
        marker.attachAxis(self._axis_y)
        self._run_marker = marker
        self._run_marker_refs.append(marker)

        # Label anchored at the run's centroid, text = "Zone X, Duration ..s to ..s".
        cx = float(np.mean(x_arr[mask]))
        cz = float(np.mean(z_arr[mask]))
        text = (f"Zone {zone}, Duration {t_start:.3f}s to {t_end:.3f}s")
        lbl = QLabel(text, self)
        lbl.setStyleSheet(
            "QLabel { background: rgba(255,255,255,220); color: #111;"
            " padding: 3px 8px; border-radius: 5px; font-weight: bold;"
            " font-size: 12px; }")
        lbl.adjustSize()
        anchor = QPointF(cx, cz)
        pos = self.chart().mapToPosition(anchor)
        lx = min(int(pos.x()) + 8, max(self.width() - lbl.width() - 4, 2))
        ly = max(int(pos.y()) - lbl.height() - 6, 2)
        lbl.move(lx, ly)
        lbl.show()
        lbl.raise_()
        self._run_marker_refs.append(lbl)
        # Remember the anchor so the label can follow on zoom/pan.
        self._run_label_anchor = (lbl, anchor, 8, -lbl.height() - 6)

    def _reposition_run_label(self, *args) -> None:
        """Re-clamp the run label to its data-coordinate anchor after zoom/pan."""
        if self._run_label_anchor is None:
            return
        lbl, anchor, dx, dy = self._run_label_anchor
        if lbl is None:
            return
        try:
            pos = self.chart().mapToPosition(anchor)
        except Exception:
            return
        lx = min(int(pos.x()) + dx, max(self.width() - lbl.width() - 4, 2))
        ly = max(int(pos.y()) + dy, 2)
        lbl.move(lx, ly)
        lbl.raise_()

    # ------------------------------------------------------------------
    # Auto-zone classification
    # ------------------------------------------------------------------

    def classify_zone(self, data) -> tuple[int | None, float | None,
                                           float | None]:
        """Classify a whole ride into a single zone by the sustained-run rule.

        The ride's (x, z) trajectory is checked against each zone's polygon(s).
        A zone qualifies if the trajectory has a CONTIGUOUS run of points that
        ALL fall inside that zone (any of its blocks) and whose duration
        (last − first point time) exceeds ``AUTO_ZONE_DUR``. Zones are checked
        in order 5 → 4 → 3 → 2 → 1; the FIRST qualifying zone wins.

        Returns ``(zone, t_start, t_end)`` where t_start/t_end are the
        satisfying run's first/last point times (seconds), or
        ``(None, None, None)`` if no zone qualifies.

        "All inside" is strict: a single point outside the zone's polygons
        breaks the run. matplotlib.path.Path.contains_points does the
        point-in-polygon test, vectorised over all points at once.
        """
        if not self._zone_paths:
            return None, None, None
        from analysis_pipeline import classify_zone
        polygons = {zone: [path.vertices for path in paths]
                    for zone, paths in self._zone_paths.items()}
        return classify_zone(data, polygons, AUTO_ZONE_DUR)

    def highlight_zone(self, zone_num: int | None) -> None:
        """Mark the classified zone WITHOUT altering the zone fills.

        The zone fills' appearance (colour + the Qt-6.x border-camouflage that
        keeps stray diagonal lines invisible) is carefully tuned in
        _block_to_area_series. Recolouring the QAreaSeries here would break
        that camouflage and re-introduce the white/black diagonal-border bug,
        so we deliberately leave every zone's fill untouched. The result is
        conveyed only by a "Auto Zone: X" overlay label in the chart corner
        plus a thick accent outline drawn over the selected zone's block edges.
        ``zone_num`` None clears the outline (overlay still shows "—").
        """
        self._auto_zone_result = zone_num

        # Remove any previous accent-outline series (e.g. from a prior classify).
        chart = self.chart()
        for s in getattr(self, "_highlight_outlines", []):
            try:
                chart.removeSeries(s)
            except Exception:
                pass
        self._highlight_outlines = []

        # Draw a thick accent outline over the selected zone's block polygon(s)
        # so the eye is drawn to it, while the fills stay exactly as built.
        if zone_num is not None:
            accent = QPen(QColor("#0078D4"), 3, Qt.PenStyle.SolidLine)
            for verts in _ZONE_POLYS.get(zone_num, []):
                out = QLineSeries()
                for vx, vz in verts:
                    out.append(QPointF(float(vx), float(vz)))
                out.append(QPointF(float(verts[0][0]), float(verts[0][1])))
                out.setPen(accent)
                chart.addSeries(out)
                out.attachAxis(self._axis_x)
                out.attachAxis(self._axis_y)
                self._highlight_outlines.append(out)

        # Overlay label in the top-right corner of the view.
        text = (f"Auto Zone: {zone_num}" if zone_num is not None
                else "Auto Zone: —")
        if self._zone_overlay is None:
            self._zone_overlay = QLabel(text, self)
            self._zone_overlay.setStyleSheet(
                "QLabel { background: rgba(255,255,255,210); color: #111;"
                " padding: 4px 10px; border-radius: 6px;"
                " font-weight: bold; font-size: 13px; }")
            self._zone_overlay.adjustSize()
        else:
            self._zone_overlay.setText(text)
            self._zone_overlay.adjustSize()
        # Position at top-right, with a small inset.
        vw = max(self.viewport().width(), 1)
        self._zone_overlay.move(
            max(vw - self._zone_overlay.width() - 12, 4), 8)
        self._zone_overlay.show()
        self._zone_overlay.raise_()

    def clear_highlight(self) -> None:
        """Remove the auto-zone overlay label + accent outlines.

        Zones get rebuilt by initplotxyz, but this also clears any accent
        outline series left over from a prior classify on the current chart.
        """
        if self._zone_overlay is not None:
            self._zone_overlay.deleteLater()
            self._zone_overlay = None
        chart = self.chart()
        for s in getattr(self, "_highlight_outlines", []):
            try:
                chart.removeSeries(s)
            except Exception:
                pass
        self._highlight_outlines = []
        self._auto_zone_result = None

    def recolor_measured(self, color: str) -> None:
        """Recolour the measured data line without recomputing points."""
        if self._data_line is not None:
            pen = self._data_line.pen()
            pen.setColor(QColor(color))
            self._data_line.setPen(pen)

    def _on_data_clicked(self, point: QPointF) -> None:
        """Show tooltip on data point click."""
        # Read source timestamps, including pre-trigger/shifted origins. Avoid
        # copying every Qt point into Python for each click on a large trace.
        if self._data_line is not None and self._last_data is not None:
            values = self._last_data.iloc[:, [1, 3]].to_numpy()
            matches = np.flatnonzero((np.abs(values[:, 0] - point.x()) < 1e-6) &
                                     (np.abs(values[:, 1] - point.y()) < 1e-6))
            idx = int(matches[0]) if len(matches) else 0
            time = float(self._last_data.iloc[idx, 0])
            text = f"t ≈ {time:.3f}s\nx = {point.x():.4f}\nz = {point.y():.4f}"
        else:
            text = f"x = {point.x():.4f}\nz = {point.y():.4f}"

        chart = self.chart()
        pos = chart.mapToPosition(point)
        label = QLabel(text, self)
        label.setStyleSheet(chart_tooltip_style())
        label.adjustSize()
        label.move(int(pos.x()) + 8, max(int(pos.y()) - label.height() - 4, 2))
        label.show()
        label.raise_()
        from PySide6.QtCore import QTimer
        QTimer.singleShot(5000, label.deleteLater)

    # ------------------------------------------------------------------
    # Compatibility
    # ------------------------------------------------------------------

    def draw_idle(self) -> None:
        pass

    def onclick(self, event) -> None:
        """Compatibility stub — Qt Charts handles clicks via signals."""
        pass

    # ------------------------------------------------------------------
    # Zoom control methods
    # ------------------------------------------------------------------

    def zoom_mode(self, mode: str) -> None:
        """Set interaction mode for the chart.

        ``"zoom"`` → rubber-band zoom; ``"pan"`` → manual drag scroll;
        anything else → default (no drag). Pan is handled by ``PanMixin``
        (inherited via BaseChartView) because ``ScrollHandDrag`` is
        unreliable on a ``QChartView``.
        """
        if mode == "zoom":
            self._interaction_mode = INTERACTION_ZOOM
            self.setRubberBand(QChartView.RubberBand.RectangleRubberBand)
            self.setDragMode(QChartView.DragMode.NoDrag)
            self.viewport().unsetCursor()
        elif mode == "pan":
            self._interaction_mode = INTERACTION_PAN
            self.setRubberBand(QChartView.RubberBand.NoRubberBand)
            self.setDragMode(QChartView.DragMode.NoDrag)
            self.viewport().setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        else:
            self._interaction_mode = INTERACTION_DEFAULT
            self._pan_cancel()
            self.setRubberBand(QChartView.RubberBand.NoRubberBand)
            self.setDragMode(QChartView.DragMode.NoDrag)
            self.viewport().unsetCursor()

    def reset_view(self) -> None:
        """Restore the chart to its standard (home) view range.

        Snaps X to ACCZONE_STD_X, then re-applies equal aspect with the home Y
        centre (0.5) so the upper zone region dominates the view instead of
        wasting half on the lower Zone 5 band.
        """
        if self._axis_x is not None:
            self._axis_x.setRange(*ACCZONE_STD_X)
        if self._axis_y is not None:
            self._axis_y.setRange(*ACCZONE_STD_Y)
        if self._equal_aspect:
            self._apply_equal_aspect(centre=self._y_centre_home)
