# coding: utf-8
"""
Trend chart — 3 vertically stacked charts showing Fore/Aft, Lateral, and
Vertical acceleration vs time.  Replaces MplCanvas (matplotlib).
"""

from __future__ import annotations

import numpy as np

from PySide6.QtCharts import QChart, QChartView, QLineSeries, QAreaSeries, QValueAxis, QScatterSeries
from PySide6.QtCore import Qt, QPointF, QTimer, QMargins
from PySide6.QtGui import QPainter, QColor, QPen, QCursor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QSizePolicy, QLabel,
)

from chart_base import BaseChartView, INTERACTION_ZOOM, INTERACTION_PAN, INTERACTION_DEFAULT, PanChartView
from chart_utils import (
    create_line_series,
    clear_chart,
    reset_chart_view,
)
from theme import LINE_WIDTH_DEFAULT, chart_tooltip_style, apply_chart_theme


class _SubChart(PanChartView):
    """A single sub-chart (one axis) inside the trend view.

    Inherits from ``PanChartView`` so manual panning works (the bare
    ``QChartView`` ignores ScrollHandDrag, see ``chart_base.PanMixin``).
    """

    def __init__(self, y_label: str, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)

        self._chart = QChart()
        self._chart.legend().hide()
        self._chart.setMargins(QMargins(8, 4, 8, 4))
        self._chart.layout().setContentsMargins(0, 0, 0, 0)

        # Axes
        self._axis_x = QValueAxis()
        self._axis_x.setTitleText("time (s)")
        self._axis_x.setTickCount(10)
        self._chart.addAxis(self._axis_x, Qt.AlignmentFlag.AlignBottom)

        self._axis_y = QValueAxis()
        # Remember the label: clear_all() rebuilds _axis_y and must restore it
        # (a fresh QValueAxis carries no title).
        self._y_label = y_label
        self._axis_y.setTitleText(self._y_label)
        self._chart.addAxis(self._axis_y, Qt.AlignmentFlag.AlignLeft)

        self.setChart(self._chart)

        # Annotation state — persistent markers + text labels. MUST be created
        # in __init__ (not in restyle_theme): the click/clear-annotation code
        # paths read these unconditionally, so missing them here used to raise
        # AttributeError unless a theme switch happened to run restyle_theme
        # first. ``annot_mode`` is mirrored from the parent TrendChartView.
        self.annot_mode: str = "Single"
        self._annot_markers: QScatterSeries | None = None
        # Persistent annotation text labels (one per placed annotation),
        # kept alive so they don't get GC'd and so we can clear them.
        self._annot_labels: list[QLabel] = []
        # Transient hover/click tooltip (auto-hides after a few seconds).
        self._tip_label: QLabel | None = None
        self._tip_timer = QTimer(self)
        self._tip_timer.setSingleShot(True)
        self._tip_timer.timeout.connect(self._hide_tip)

        # Cache of QLineSeries -> (xmin, xmax, ymin, ymax) so fit_data() can
        # compute the visible range from numpy bounds instead of pulling every
        # point back into Python (see add_data_series / fit_data).
        self._series_ranges: dict = {}
        # Analysis overlays (peak markers + 0.2s region bands). Each entry is
        # a created QSeries/QLabel that must be kept alive (Qt Charts GC) and
        # removed on clear_analysis().
        self._analysis_refs: list = []
        # Analysis labels that must follow their data-coordinate anchor when
        # the axes change (zoom/pan). Each entry: (QLabel, QPointF anchor,
        # pixel dx, pixel dy). ``anchor`` is in DATA coordinates; dx/dy are
        # the constant pixel offsets applied on top of mapToPosition(anchor).
        self._analysis_labels: list[tuple] = []
        self._axis_x.rangeChanged.connect(self._reposition_analysis_labels)
        self._axis_y.rangeChanged.connect(self._reposition_analysis_labels)

    def setChart(self, chart) -> None:
        """Override so the theme palette is applied to every attached chart
        (PanChartView's base setChart doesn't, since QSS can't reach a chart)."""
        super().setChart(chart)
        try:
            apply_chart_theme(chart)
        except Exception as e:
            print(f"trend sub-chart theme apply failed: {e}")

    def restyle_theme(self) -> None:
        """Re-apply the current palette after a light/dark switch."""
        apply_chart_theme(self._chart)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # Ten ticks become ellipses on a laptop-sized chart. Allocate roughly
        # 100px per timestamp so source times remain readable at narrow widths.
        if hasattr(self, '_axis_x'):
            self._axis_x.setTickCount(max(3, min(10, self.width() // 100)))

    @property
    def chart_obj(self) -> QChart:
        return self._chart

    @property
    def axis_x(self) -> QValueAxis:
        return self._axis_x

    @property
    def axis_y(self) -> QValueAxis:
        return self._axis_y

    def clear_all(self) -> None:
        clear_chart(self._chart)
        # Re-add clean axes
        self._axis_x = QValueAxis()
        self._axis_x.setTitleText("time (s)")
        self._axis_x.setTickCount(10)
        self._chart.addAxis(self._axis_x, Qt.AlignmentFlag.AlignBottom)

        self._axis_y = QValueAxis()
        # Restore the Y-axis title lost with the previous axis object.
        self._axis_y.setTitleText(self._y_label)
        self._chart.addAxis(self._axis_y, Qt.AlignmentFlag.AlignLeft)

        # clear_chart() rebuilt the axes, so the rangeChanged connections from
        # __init__ are gone. Re-wire them so analysis labels still follow the
        # data anchor on zoom/pan after a re-init.
        self._axis_x.rangeChanged.connect(self._reposition_analysis_labels)
        self._axis_y.rangeChanged.connect(self._reposition_analysis_labels)

        # Drop annotation state for this sub-chart.
        self._annot_markers = None
        for lbl in self._annot_labels:
            lbl.deleteLater()
        self._annot_labels = []

        # Drop any analysis overlays too.
        self.clear_analysis()

        # Drop the cached per-series data ranges (see add_data_series).
        self._series_ranges.clear()

    def add_data_series(self, time_col, value_col, color: str) -> QLineSeries:
        series = create_line_series(time_col, value_col, color=color,
                                     width=LINE_WIDTH_DEFAULT)
        self._chart.addSeries(series)
        series.attachAxis(self._axis_x)
        series.attachAxis(self._axis_y)

        # Cache this series' (xmin, xmax, ymin, ymax) computed once from the
        # source arrays via numpy. fit_data() reads these cached bounds instead
        # of calling pointsVector() and iterating every point in Python on
        # each reset_view — the old path was O(total points) and dominated
        # reset_view at ~50ms/10k pts (313ms for 6 datasets).
        try:
            xv = time_col.values if hasattr(time_col, "values") else time_col
            yv = value_col.values if hasattr(value_col, "values") else value_col
            self._series_ranges[series] = (
                float(np.min(xv)), float(np.max(xv)),
                float(np.min(yv)), float(np.max(yv)),
            )
        except Exception:
            self._series_ranges[series] = None
        return series

    def fit_data(self) -> None:
        """Recalculate x/y range so all attached series are visible.

        Reads the cached per-series bounds (set in add_data_series) instead of
        pulling every point back into Python via pointsVector(). Falls back to
        the series points only for series that have no cached range.
        """
        xmins, xmaxs, ymins, ymaxs = [], [], [], []
        for s in self._chart.series():
            if not isinstance(s, QLineSeries):
                continue
            rng = self._series_ranges.get(s)
            if rng is not None:
                xmins.append(rng[0]); xmaxs.append(rng[1])
                ymins.append(rng[2]); ymaxs.append(rng[3])
                continue
            # Fallback: series without a cached range (e.g. annotation markers
            # that are QScatterSeries are filtered above; any stray QLineSeries
            # added outside add_data_series land here).
            vx = s.pointsVector()
            if not vx:
                continue
            xmins.append(min(p.x() for p in vx))
            xmaxs.append(max(p.x() for p in vx))
            ymins.append(min(p.y() for p in vx))
            ymaxs.append(max(p.y() for p in vx))
        if xmins:
            self._axis_x.setRange(min(xmins), max(xmaxs))
        if ymins:
            self._axis_y.setRange(min(ymins), max(ymaxs))

    # ------------------------------------------------------------------
    # Analysis overlays — ±peak markers + 0.2s sustained-peak region bands.
    # Toggled by the "Analyze" button on the Trend toolbar.
    # ------------------------------------------------------------------

    def clear_analysis(self) -> None:
        """Remove all analysis overlays (peak markers, labels, region bands)."""
        for ref in self._analysis_refs:
            try:
                if isinstance(ref, QLabel):
                    ref.deleteLater()
                else:
                    # QSeries — remove from chart if still attached.
                    self._chart.removeSeries(ref)
            except Exception:
                pass
        self._analysis_refs.clear()
        # Drop the anchored-label tracking too (labels already deleted above).
        self._analysis_labels.clear()

    @staticmethod
    def set_analysis_visible(refs: list, visible: bool) -> None:
        """Show/hide a per-dataset list of analysis overlays in place.

        Used so hiding a dataset's curve (via the sidebar eye toggle) also
        hides that dataset's peak markers / sustained fills / labels, without
        affecting other datasets' overlays. QSeries use setVisible; QLabel
        use setVisible too (a no-arg QWidget method). The QAreaSeries hides
        its upper/lower boundary lines automatically.
        """
        for ref in refs:
            try:
                ref.setVisible(visible)
            except Exception:
                pass

    def _reposition_analysis_labels(self, *args) -> None:
        """Re-clamp every analysis QLabel to its data-coordinate anchor.

        QLabels are plain widget children positioned in pixels, so zoom/pan
        (which only re-maps data↔pixel) would leave them stranded at their
        original pixel spot. This re-runs mapToPosition(anchor) for each and
        re-applies the stored pixel offset. Connected to both axes' rangeChanged.
        """
        for lbl, anchor, dx, dy in self._analysis_labels:
            if lbl is None:
                continue
            try:
                pos = self._chart.mapToPosition(anchor)
            except Exception:
                continue
            lx = min(int(pos.x()) + dx,
                     max(self.width() - lbl.width() - 4, 2))
            ly = max(int(pos.y()) + dy, 2)
            lbl.move(lx, ly)
            lbl.raise_()

    def show_analysis(self, time_arr, value_arr, fs: float,
                      color: str, window: float = 0.2) -> list:
        """Draw ±peak markers + sustained-peak filled region for one series.

        ``time_arr`` / ``value_arr`` are 1-D numpy arrays (already aligned to
        the displayed plot type). ``color`` is the dataset's hex colour.
        ``window`` is the sustained-peak sliding-window width in seconds.

        Visualization per the spec:
          • 2 peak points (instantaneous max / min) as QScatterSeries, each
            with a value label.
          • For each of the +sustained and −sustained peaks: a flat horizontal
            line at the sustained value spanning the window, plus a filled
            region between that line and the actual curve segment inside the
            window, plus a value label (0.001 precision) on the line.

        Returns the list of created overlays (QSeries + QLabel) so the caller
        can track them per-dataset and toggle visibility with the dataset's
        curve. All objects are also kept in ``_analysis_refs`` (Qt Charts GC)
        and removed by clear_analysis().
        """
        import pandas as pd

        # Per-call list of created refs, returned for per-dataset visibility
        # tracking (see TrendChartView._analysis_by_dataset).
        created: list = []

        n = len(value_arr)
        if n < 2 or time_arr is None:
            return created

        # Sign-coded colours: positive peaks (max) → red, negative peaks (min)
        # → blue. Used for both the peak scatter markers and the sustained-peak
        # region fills, so the sign of an extremum is readable at a glance
        # regardless of the dataset's own curve colour.
        pos_col = QColor("#E8112D")    # red  — positive (+g) extremes
        neg_col = QColor("#0078D4")    # blue — negative (−g) extremes
        # Translucent fills for the sustained regions (low alpha so the curve
        # stays visible through them).
        pos_fill = QColor(pos_col); pos_fill.setAlpha(90)
        neg_fill = QColor(neg_col); neg_fill.setAlpha(90)

        # ---- ±peak markers (instantaneous max / min) -------------------
        # Two separate QScatterSeries so each peak is coloured by its SIGN:
        # the max point (positive) red, the min point (negative) blue.
        i_max = int(np.argmax(value_arr))
        i_min = int(np.argmin(value_arr))
        p_max = QPointF(float(time_arr[i_max]), float(value_arr[i_max]))
        p_min = QPointF(float(time_arr[i_min]), float(value_arr[i_min]))

        for p, mk_col in ((p_max, pos_col), (p_min, neg_col)):
            marker = QScatterSeries()
            marker.setMarkerSize(12)
            marker.setColor(mk_col)
            marker.setBorderColor(QColor("black"))
            marker.append(p)
            self._chart.addSeries(marker)
            marker.attachAxis(self._axis_x)
            marker.attachAxis(self._axis_y)
            self._analysis_refs.append(marker)
            created.append(marker)

            # Value label anchored to the peak point (follows on zoom/pan).
            lbl = QLabel(f"{p.y():+.4g}", self)
            lbl.setStyleSheet(chart_tooltip_style())
            lbl.adjustSize()
            dx, dy = 8, -lbl.height() - 4
            pos = self._chart.mapToPosition(p)
            lx = min(int(pos.x()) + dx,
                     max(self.width() - lbl.width() - 4, 2))
            ly = max(int(pos.y()) + dy, 2)
            lbl.move(lx, ly)
            lbl.show()
            lbl.raise_()
            self._analysis_refs.append(lbl)
            created.append(lbl)
            self._analysis_labels.append((lbl, QPointF(p), dx, dy))

        # ---- sustained-peak filled regions -----------------------------
        # Window width in samples. The sustained value over a window is the
        # min (for +peak) / max (for −peak) of the samples in that window;
        # the worst-case sustained value is then the max/min of those
        # rolling stats. The window that achieves it is reported.
        dt = max(1, int(round(window * fs)))
        if n > dt:
            ser = pd.Series(value_arr)

            # Positive sustained peak = largest of rolling-min values.
            # pandas rolling(dt).min()[i] = min over the dt samples ENDING at
            # index i, i.e. window [i-dt+1, i]. So argmax of the valid slice
            # marks the window's END index; its START is end-dt+1. An off-by-
            # one here used to shift the whole window one dt to the right
            # ("1.2-1.4s" instead of the correct "1.0-1.2s").
            roll_min = ser.rolling(dt).min()
            rm = roll_min.values
            valid = rm[dt - 1:]
            if not np.all(np.isnan(valid)):
                pos_end = int(np.nanargmax(valid) + (dt - 1))
                pos_start = max(0, pos_end - dt + 1)
                sustained_val = float(np.nanmax(valid))
                self._add_sustained_region(
                    time_arr, value_arr, pos_start, pos_end,
                    sustained_val, pos_col, pos_fill, positive=True,
                    created=created)

            # Negative sustained peak = smallest of rolling-max values; only
            # meaningful if there are negative values in the trace.
            roll_max = ser.rolling(dt).max()
            rmx = roll_max.values
            neg_valid = rmx[dt - 1:]
            if (not np.all(np.isnan(neg_valid))
                    and np.nanmin(neg_valid) < 0):
                neg_end = int(np.nanargmin(neg_valid) + (dt - 1))
                neg_start = max(0, neg_end - dt + 1)
                sustained_val = float(np.nanmin(neg_valid))
                self._add_sustained_region(
                    time_arr, value_arr, neg_start, neg_end,
                    sustained_val, neg_col, neg_fill, positive=False,
                    created=created)

        return created

    def _add_sustained_region(self, time_arr, value_arr,
                              i_start: int, i_end: int,
                              sustained_val: float,
                              line_col: QColor, fill: QColor,
                              positive: bool,
                              created: list | None = None) -> None:
        """Draw the sustained-peak region for one window.

        Adds:
          • A ``QAreaSeries`` whose upper boundary is the actual curve segment
            over [i_start, i_end] and whose lower boundary is a flat line at
            ``sustained_val`` (for a +peak), or vice-versa for a −peak. The
            fill is the closed area between the curve and the sustained line.
          • A flat horizontal ``QLineSeries`` at ``sustained_val`` drawn on top
            (the visible sustained-value line — the area's boundary lines are
            camouflaged invisible, so we need this separate visible line).
          • A value label (0.001 precision) anchored near the line.

        ``created`` (if given) is appended with every created object so the
        caller can track them per-dataset for visibility toggling.

        The area's upper/lower boundary QLineSeries use MANY sample points (the
        full curve segment, not just 2 endpoints) so Qt Charts' QAreaSeries
        renders the fill strictly within the defined x-span — the known
        2-points-per-line area overflow bug only triggers with degenerate
        2-point lines.
        """
        t0 = float(time_arr[i_start])
        t1 = float(time_arr[i_end])

        # Boundary series for the area fill. These are NOT drawn visibly —
        # their pens are camouflaged to the fill colour (Qt 6.x draws the
        # QAreaSeries border despite NoPen; see acczone). Only the fill shows.
        curve_boundary = QLineSeries()
        flat_boundary = QLineSeries()
        for i in range(i_start, i_end + 1):
            tx = float(time_arr[i])
            curve_boundary.append(QPointF(tx, float(value_arr[i])))
            flat_boundary.append(QPointF(tx, sustained_val))
        # Camouflage: pen == fill colour so the boundary is invisible against
        # its own area fill.
        inv_pen = QPen(fill, 1, Qt.PenStyle.SolidLine)
        curve_boundary.setPen(inv_pen)
        flat_boundary.setPen(inv_pen)

        # For a +peak the curve is at/above the flat (sustained = window min,
        # i.e. the floor); fill between flat (lower) and curve (upper). For a
        # −peak the curve is at/below the flat; fill between curve (lower) and
        # flat (upper).
        if positive:
            upper, lower = curve_boundary, flat_boundary
        else:
            upper, lower = flat_boundary, curve_boundary

        area = QAreaSeries(upper, lower)
        area.setColor(fill)
        area.setBorderColor(fill)
        area.setPen(inv_pen)

        # Add the area (and its hidden boundary lines) to the chart BEFORE
        # attachAxis, else Qt warns "Series not in the chart".
        self._chart.addSeries(area)
        area.attachAxis(self._axis_x)
        area.attachAxis(self._axis_y)
        # Boundary lines do NOT need their own axis attach or addSeries — the
        # QAreaSeries owns them once added. Keep strong Python refs so the
        # PySide6 wrapper isn't GC'd (which would delete the boundary lines).
        self._analysis_refs.extend([area, upper, lower])
        if created is not None:
            created.extend([area, upper, lower])

        # The VISIBLE sustained-value line, drawn on top of the fill. A simple
        # 2-point QLineSeries at the sustained value across the window.
        flat_line = QLineSeries()
        flat_line.append(QPointF(t0, sustained_val))
        flat_line.append(QPointF(t1, sustained_val))
        flat_line.setPen(QPen(line_col, 2, Qt.PenStyle.SolidLine))
        self._chart.addSeries(flat_line)
        flat_line.attachAxis(self._axis_x)
        flat_line.attachAxis(self._axis_y)
        self._analysis_refs.append(flat_line)
        if created is not None:
            created.append(flat_line)

        # Value label on the flat line: place it just above (positive) or
        # below (negative) the line, near the window's mid-x. Anchored to the
        # mid-window data point so it follows on zoom/pan.
        mid_t = 0.5 * (t0 + t1)
        anchor = QPointF(mid_t, sustained_val)
        lbl = QLabel(f"{sustained_val:+.3f}", self)
        lbl.setStyleSheet(chart_tooltip_style())
        lbl.adjustSize()
        dx = 6
        dy = -lbl.height() - 3 if positive else 3
        pos = self._chart.mapToPosition(anchor)
        lx = min(int(pos.x()) + dx,
                 max(self.width() - lbl.width() - 4, 2))
        ly = max(int(pos.y()) + dy, 2)
        lbl.move(lx, ly)
        lbl.show()
        lbl.raise_()
        self._analysis_refs.append(lbl)
        if created is not None:
            created.append(lbl)
        self._analysis_labels.append((lbl, anchor, dx, dy))

    # ------------------------------------------------------------------
    # Pan linkage — bubble up to the parent TrendChartView so it can mirror
    # this chart's new ranges onto the other two sub-charts.
    # ------------------------------------------------------------------

    def _pan_applied(self) -> None:
        """After a manual pan, tell the parent view to sync the siblings."""
        parent = self.parent()
        if isinstance(parent, TrendChartView):
            parent._sync_from_panned(self)

    # --- annotation markers (respect Single / Multiple mode) -------------

    def _ensure_annot_markers(self) -> QScatterSeries:
        if self._annot_markers is None:
            m = QScatterSeries()
            m.setName("_annotations")
            m.setMarkerSize(9)
            m.setColor(QColor(255, 0, 0, 200))
            m.setBorderColor(QColor("darkred"))
            m.setPointsVisible(True)
            # Attach to current axes so dots land at the right data coords.
            m.attachAxis(self._axis_x)
            m.attachAxis(self._axis_y)
            self._chart.addSeries(m)
            self._annot_markers = m
        return self._annot_markers

    def clear_annotations(self) -> None:
        """Remove all annotation dots + text labels from this sub-chart."""
        if self._annot_markers is not None:
            self._annot_markers.replace([])
        for lbl in self._annot_labels:
            lbl.deleteLater()
        self._annot_labels = []

    def add_annotation(self, point: QPointF) -> None:
        """Place an annotation marker + persistent text label at *point*.

        In ``Single`` mode previous annotations are cleared first; in
        ``Multiple`` mode they accumulate.
        """
        if self.annot_mode == "Single":
            self.clear_annotations()

        markers = self._ensure_annot_markers()
        pts = list(markers.points())
        pts.append(point)
        markers.replace(pts)

        # Persistent text label near the point.
        pos = self._chart.mapToPosition(point)
        text = f"t={point.x():.3f}\ny={point.y():.4f}"
        lbl = QLabel(text, self)
        lbl.setStyleSheet(chart_tooltip_style())
        lbl.adjustSize()
        lx = min(int(pos.x()) + 8, self.width() - lbl.width() - 4)
        ly = max(int(pos.y()) - lbl.height() - 4, 2)
        lbl.move(lx, ly)
        lbl.show()
        lbl.raise_()
        self._annot_labels.append(lbl)

    # --- transient hover/click tooltip -----------------------------------

    def show_tip(self, text: str, pos_x: float, pos_y: float) -> None:
        if self._tip_label is None:
            self._tip_label = QLabel(self)
        self._tip_label.setStyleSheet(chart_tooltip_style())
        self._tip_label.setText(text)
        self._tip_label.adjustSize()
        lx = min(int(pos_x) + 8, self.width() - self._tip_label.width() - 4)
        ly = max(int(pos_y) - self._tip_label.height() - 4, 2)
        self._tip_label.move(lx, ly)
        self._tip_label.show()
        self._tip_label.raise_()
        self._tip_timer.start(5000)

    def _hide_tip(self) -> None:
        if self._tip_label is not None:
            self._tip_label.hide()


class TrendChartView(QWidget):
    """Three vertically stacked charts: X, Y, Z acceleration vs time."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.annot_mode: str = "Single"
        self._series_list: list[QLineSeries] = []
        # Per-dataset registry: plot_order -> [sx, sy, sz]. Used so a single
        # dataset's colour can be changed live without redrawing the rest.
        self._series_by_dataset: dict[int, list[QLineSeries]] = {}
        # Per-dataset cached source arrays for the Analyze feature:
        # plot_order -> (time, x, y, z, fs). Stored at addplotxyz time so
        # analysis (peaks + 0.2s bands) can use the raw arrays without pulling
        # points back out of QLineSeries.
        self._data_by_dataset: dict[int, tuple] = {}
        # Whether analysis overlays are currently shown.
        self._analysis_active: bool = False
        # Per-dataset analysis overlay refs: plot_order -> list of refs per
        # sub-chart, so hiding a dataset's curve (sidebar eye toggle) also hides
        # that dataset's peak markers / sustained fills / labels. Structure:
        # {plot_order: {"x": [...], "y": [...], "z": [...]}}.
        self._analysis_by_dataset: dict[int, dict[str, list]] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)

        self._chart_x = _SubChart("Fore/Aft", self)
        self._chart_y = _SubChart("Lateral", self)
        self._chart_z = _SubChart("Vertical", self)

        layout.addWidget(self._chart_x, 1)
        layout.addWidget(self._chart_y, 1)
        layout.addWidget(self._chart_z, 1)

        # X-axis cross-link: any sub-chart's time range changing should mirror
        # onto the other two, so the three trends stay horizontally aligned
        # during pan/zoom. We connect all three and guard against recursion.
        # ``_syncing`` is set while we're propagating a range so our own
        # setRange doesn't re-trigger the chain.
        self._syncing = False
        for sub in (self._chart_x, self._chart_y, self._chart_z):
            sub.axis_x.rangeChanged.connect(self._sync_x_axis)

    # ------------------------------------------------------------------
    # Public API (matches original MplCanvas interface)
    # ------------------------------------------------------------------

    def initplotxyz(self) -> None:
        """Clear all charts and series.  Re-wires the X-axis sync (below)."""
        for sub in (self._chart_x, self._chart_y, self._chart_z):
            sub.clear_all()
            # clear_all() rebuilt axis_x, so the _sync_x_axis connection made
            # in __init__ died with the discarded axis object. Re-wire it so
            # the three time axes stay linked after every re-plot (each
            # clear_all makes a fresh axis, so this never double-connects).
            sub.axis_x.rangeChanged.connect(self._sync_x_axis)
            # Recreate the sub-chart's viewport so no stale pixels from the
            # previous (populated) chart linger — a reset must look exactly like
            # the app's initial state.
            reset_chart_view(sub)
        self._series_list.clear()
        self._series_by_dataset.clear()
        self._data_by_dataset.clear()
        self._analysis_by_dataset.clear()
        self._analysis_active = False

    def addplotxyz(self, list_data, plot_order: int, color: str = "b") -> None:
        """Add a 3-axis data series from a pandas DataFrame.

        Parameters
        ----------
        list_data : DataFrame with columns [Time, x, y, z]
        plot_order : 1-based dataset index (used as the colour identity key)
        color : hex colour string
        """
        time = list_data.iloc[:, 0]
        sx = self._chart_x.add_data_series(time, list_data.iloc[:, 1], color)
        sy = self._chart_y.add_data_series(time, list_data.iloc[:, 2], color)
        sz = self._chart_z.add_data_series(time, list_data.iloc[:, 3], color)

        for s in (sx, sy, sz):
            s.clicked.connect(lambda pt, sr=s: self._on_point_clicked(pt, sr))
            self._series_list.append(s)
        self._series_by_dataset[plot_order] = [sx, sy, sz]
        self._cache_analysis_data(list_data, plot_order, offset=0.0)
        # Fit axes tightly so the post-import default view equals Reset View.
        # Without an explicit range the axes fall to Qt Charts' nice-number
        # auto-domain (loose padding, per-chart tick granularity), which makes
        # small-span sub-charts (e.g. Vertical ≈1 g) look squashed while
        # large-span ones look full — the "weird proportions on import" that
        # a manual Reset View used to fix. Re-fitting on every add also covers
        # multi-dataset loads: the last fit spans the cumulative data.
        for sub in (self._chart_x, self._chart_y, self._chart_z):
            sub.fit_data()

    def addplotxyz_shifted(self, list_data, plot_order: int,
                           offset_seconds, color: str = "b") -> None:
        """Like addplotxyz but shifts the time axis by *offset_seconds*.

        Parameters
        ----------
        list_data : DataFrame with columns [Time, x, y, z]
        plot_order : 1-based dataset index (used as the colour identity key)
        offset_seconds : float — seconds to subtract from the Time column.
            Positive shifts the curve left (earlier); negative shifts right.
            Already in seconds — no fs assumption baked in here.
        color : hex colour string
        """
        time = list_data.iloc[:, 0].values - offset_seconds
        time = np.round(time, decimals=4)

        sx = self._chart_x.add_data_series(time, list_data.iloc[:, 1], color)
        sy = self._chart_y.add_data_series(time, list_data.iloc[:, 2], color)
        sz = self._chart_z.add_data_series(time, list_data.iloc[:, 3], color)

        for s in (sx, sy, sz):
            s.clicked.connect(lambda pt, sr=s: self._on_point_clicked(pt, sr))
            self._series_list.append(s)
        self._series_by_dataset[plot_order] = [sx, sy, sz]
        self._cache_analysis_data(list_data, plot_order, offset=offset_seconds)

    def _cache_analysis_data(self, list_data, plot_order: int,
                             offset: float = 0.0) -> None:
        """Store (time, x, y, z, fs) for one dataset so the Analyze feature
        can compute peaks / 0.2s bands from the raw arrays.

        ``fs`` is derived from the time column spacing (1 / median dt), which
        matches how accdata.AccData calculates it. The time array has the
        shift offset already applied so peak times land on the drawn curve.
        """
        try:
            t = list_data.iloc[:, 0].values.astype(float) - offset
            xv = list_data.iloc[:, 1].values.astype(float)
            yv = list_data.iloc[:, 2].values.astype(float)
            zv = list_data.iloc[:, 3].values.astype(float)
            fs = 500.0
            if len(t) > 1:
                d = float(np.median(np.diff(t)))
                if d > 0:
                    fs = 1.0 / d
            self._data_by_dataset[plot_order] = (t, xv, yv, zv, fs)
        except Exception as e:
            print(f"analysis data cache skipped: {e}")

    def addplotxyz_fitmax(self, list_data, plot_order: int,
                          offset_seconds, color: str = "b") -> None:
        """Time-shifted add — kept for backwards compatibility.

        Now takes *offset_seconds* directly (callers convert sample counts
        via ``samples / fs``). Delegates to :meth:`addplotxyz_shifted`.
        """
        self.addplotxyz_shifted(list_data, plot_order, offset_seconds, color)

    def draw_idle(self) -> None:
        """No-op for API compatibility (Qt Charts redraws automatically)."""
        pass

    # ------------------------------------------------------------------
    # Live recolour — swap one dataset's pen colour without redrawing data
    # ------------------------------------------------------------------

    def recolor_dataset(self, plot_order: int, color: str) -> bool:
        """Recolour the three sub-chart series belonging to one dataset.

        Returns True if the dataset was found and recoloured, else False.
        Only the QPen is swapped — the data points stay, so zoom/pan state
        and annotations are preserved.
        """
        series = self._series_by_dataset.get(plot_order)
        if not series:
            return False
        for s in series:
            pen = s.pen()
            pen.setColor(QColor(color))
            s.setPen(pen)
        return True

    # ------------------------------------------------------------------
    # Plot-type switch — update Y data in place (no addSeries/removeSeries)
    # ------------------------------------------------------------------

    def update_plottype(self, datalist, plottype: str,
                        dataset_colors: dict | None = None) -> None:
        """Refresh every dataset's Y data for a new plot type in place.

        A Raw/Filter/Standard switch only changes the Y values — the time axis,
        series count, colours, zoom/pan state, and analysis cache all stay the
        same. So instead of ``initplotxyz()`` (remove all series) + ``addplotxyz``
        (re-add all series, which triggers a Qt Charts layout reflow per add),
        we ``replaceNp`` the new Y onto each existing series. ``replaceNp`` is a
        direct C++ path (~0.07 ms for 27k points), so a switch over 4 datasets ×
        3 axes is nearly instant.

        Also refreshes the per-dataset analysis cache so a later Analyze toggle
        uses the new plottype's peaks. Clears any visible analysis overlays
        (their markers would no longer match the curve) — the caller resets the
        Analyze button.
        """
        cmap = dataset_colors or {}
        # Clear any visible analysis overlays first — their markers/labels were
        # positioned for the OLD plottype's curve and would no longer match.
        if self._analysis_active:
            for sub in (self._chart_x, self._chart_y, self._chart_z):
                sub.clear_analysis()
            self._analysis_by_dataset.clear()
            self._analysis_active = False
        for plot_order, data in enumerate(datalist, start=1):
            series = self._series_by_dataset.get(plot_order)
            if series is None or len(series) < 3:
                continue
            # Pick the frame for this plottype.
            src = {"Raw": "data", "Filter": "filtered_data",
                   "Standard": "std_data"}.get(plottype, "std_data")
            frame = getattr(data, src, None)
            if frame is None:
                continue
            try:
                t = frame.iloc[:, 0].values
                xv = frame.iloc[:, 1].values
                yv = frame.iloc[:, 2].values
                zv = frame.iloc[:, 3].values
            except Exception:
                continue
            # Update each axis's Y in place (time column unchanged).
            sx, sy, sz = series
            for s, yv in ((sx, xv), (sy, yv), (sz, zv)):
                try:
                    s.replaceNp(
                        np.ascontiguousarray(t, dtype=np.float64),
                        np.ascontiguousarray(yv, dtype=np.float64),
                    )
                except Exception:
                    pass
                # Refresh the cached per-series bounds (used by reset_view).
                sub = self._subchart_for_series(s)
                if sub is not None:
                    try:
                        sub._series_ranges[s] = (
                            float(np.min(t)), float(np.max(t)),
                            float(np.min(yv)), float(np.max(yv)),
                        )
                    except Exception:
                        pass
            # Refresh the analysis cache so Analyze uses the new plottype.
            self._cache_analysis_data(frame, plot_order, offset=0.0)

    def _subchart_for_series(self, series: QLineSeries):
        """Return the _SubChart that owns *series*, or None."""
        for sub in (self._chart_x, self._chart_y, self._chart_z):
            if series in sub.chart_obj.series():
                return sub
        return None

    # ------------------------------------------------------------------
    # Show / hide a dataset's trend curves without removing the data
    # ------------------------------------------------------------------

    def set_dataset_visible(self, plot_order: int, visible: bool) -> bool:
        """Toggle visibility of the three sub-chart series for one dataset.

        Returns True if the dataset was found. Hiding keeps the data in
        memory so re-showing is instant and zoom/pan state is preserved.

        Also toggles that dataset's analysis overlays (peak markers / sustained
        fills / labels) so they don't linger over an invisible curve. Each
        sub-chart's overlay list is hidden/shown via _SubChart.set_analysis_visible.
        """
        series = self._series_by_dataset.get(plot_order)
        if not series:
            return False
        for s in series:
            s.setVisible(visible)
        # Mirror visibility onto this dataset's analysis overlays, if present.
        refs = self._analysis_by_dataset.get(plot_order)
        if refs:
            for axis_key in ("x", "y", "z"):
                _SubChart.set_analysis_visible(refs[axis_key], visible)
        return True

    # ------------------------------------------------------------------
    # X-axis synchronisation
    # ------------------------------------------------------------------

    def _sync_x_axis(self, min_val: float, max_val: float) -> None:
        """Mirror one sub-chart's time range onto the other two.

        Triggered by any sub-chart's ``axis_x.rangeChanged``. ``_syncing``
        guards against recursion: while we propagate, our own ``setRange``
        would re-emit ``rangeChanged``, so we bail out until the chain ends.
        """
        if self._syncing:
            return
        self._syncing = True
        try:
            for sub in (self._chart_x, self._chart_y, self._chart_z):
                cur_min, cur_max = sub.axis_x.min(), sub.axis_x.max()
                if cur_min != min_val or cur_max != max_val:
                    sub.axis_x.setRange(min_val, max_val)
        finally:
            self._syncing = False

    def _sync_from_panned(self, panned) -> None:
        """After *panned* (the sub-chart the user dragged) is panned, mirror
        its time range onto the other two so they pan in lockstep.

        We read from the panned chart explicitly rather than relying solely
        on ``rangeChanged`` because the manual ``chart.scroll`` in
        :class:`PanMixin` shifts by pixels and only the source chart's axis
        updates — the signal-based sync then carries it to the siblings.
        """
        ax = panned.axis_x
        self._sync_x_axis(ax.min(), ax.max())

    # ------------------------------------------------------------------
    # Annotation mode (Single / Multiple) + click handling
    # ------------------------------------------------------------------

    def set_annotation_mode(self, mode: str) -> None:
        """Propagate the annotation mode to every sub-chart."""
        self.annot_mode = mode
        for sub in (self._chart_x, self._chart_y, self._chart_z):
            sub.annot_mode = mode

    def clear_annotations(self) -> None:
        """Clear all annotation markers + labels across the three sub-charts."""
        for sub in (self._chart_x, self._chart_y, self._chart_z):
            sub.clear_annotations()

    def _on_point_clicked(self, point: QPointF, series: QLineSeries) -> None:
        """Place an annotation marker + label on the owning sub-chart.

        In Single mode previous annotations are cleared first; in Multiple
        they accumulate. Falls back to a transient tooltip if the owning
        sub-chart can't be found.
        """
        for sub in (self._chart_x, self._chart_y, self._chart_z):
            if series in sub.chart_obj.series():
                sub.add_annotation(point)
                return
        # Fallback: transient tooltip if we couldn't locate the sub-chart.
        text = f"t = {point.x():.3f}\namp = {point.y():.4f}"
        self._show_chart_tooltip(series, point, text)

    def _show_chart_tooltip(self, series: QLineSeries,
                            point: QPointF, text: str) -> None:
        """Show a transient tooltip near the click point (auto-hides)."""
        for sub in (self._chart_x, self._chart_y, self._chart_z):
            if series in sub.chart_obj.series():
                pos = sub.chart_obj.mapToPosition(point)
                sub.show_tip(text, pos.x(), pos.y())
                break

    # ------------------------------------------------------------------
    # Zoom control methods — now backed by BaseChartView.set_interaction
    # ------------------------------------------------------------------

    def zoom_mode(self, mode: str) -> None:
        """Set interaction mode for all sub-charts.

        ``mode`` is one of: ``"zoom"`` (rubber-band), ``"pan"`` (drag scroll),
        or anything else → default (no drag).

        Pan is handled manually by the ``PanChartView`` base of each sub-chart
        (``ScrollHandDrag`` is unreliable on a ``QChartView``).
        """
        for sub in (self._chart_x, self._chart_y, self._chart_z):
            if mode == "zoom":
                sub._interaction_mode = INTERACTION_ZOOM
                sub.setRubberBand(QChartView.RubberBand.RectangleRubberBand)
                sub.setDragMode(QChartView.DragMode.NoDrag)
                sub.viewport().unsetCursor()
            elif mode == "pan":
                sub._interaction_mode = INTERACTION_PAN
                sub.setRubberBand(QChartView.RubberBand.NoRubberBand)
                sub.setDragMode(QChartView.DragMode.NoDrag)
                sub.viewport().setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
            else:
                sub._interaction_mode = INTERACTION_DEFAULT
                sub._pan_cancel()
                sub.setRubberBand(QChartView.RubberBand.NoRubberBand)
                sub.setDragMode(QChartView.DragMode.NoDrag)
                sub.viewport().unsetCursor()

    def reset_view(self) -> None:
        """Reset all 3 charts to fit their data.

        Bug fixed: ``chart.zoomReset()`` only undoes prior rubber-band zooms;
        on a fresh view it was a no-op. We recompute ranges from the series
        so Reset View always works.
        """
        for sub in (self._chart_x, self._chart_y, self._chart_z):
            sub.fit_data()

    def restyle_theme(self) -> None:
        """Re-apply the palette to all three sub-charts on a theme switch."""
        for sub in (self._chart_x, self._chart_y, self._chart_z):
            sub.restyle_theme()

    # ------------------------------------------------------------------
    # Analysis overlays — ±peak markers + 0.2s sustained-peak region bands.
    # Toggled by the "Analyze" button on the Trend toolbar.
    # ------------------------------------------------------------------

    def toggle_analysis(self, datalist, plottype: str,
                        dataset_colors: dict | None = None,
                        window: float = 0.2) -> bool:
        """Toggle the analysis overlays on all three sub-charts.

        ``datalist`` is MainWindow's list of AccData objects (in plot order).
        ``plottype`` selects which frame to analyse ("Standard"/"Raw"/"Filter")
        so the peaks match the currently-displayed curve. ``dataset_colors``
        maps filename -> hex colour. ``window`` is the sustained-peak
        sliding-window width in seconds.

        Returns the new state (True = overlays shown).
        """
        if self._analysis_active:
            # Turn off.
            for sub in (self._chart_x, self._chart_y, self._chart_z):
                sub.clear_analysis()
            self._analysis_by_dataset.clear()
            self._analysis_active = False
            return False

        # Turn on: iterate datasets in plot order, draw overlays on each axis.
        # Adding series can make Qt Charts auto-fit the axes to the new data,
        # which would change the user's current zoom/pan view. Save the axis
        # ranges before drawing and force-restore them after so the view is
        # identical to before the toggle.
        saved = {}
        for sub in (self._chart_x, self._chart_y, self._chart_z):
            saved[sub] = (sub.axis_x.min(), sub.axis_x.max(),
                          sub.axis_y.min(), sub.axis_y.max())

        cmap = dataset_colors or {}
        for plot_order, data in enumerate(datalist, start=1):
            cached = self._data_by_dataset.get(plot_order)
            if cached is None:
                continue
            t, xv, yv, zv, fs = cached
            # Match the colour used for this dataset's curves.
            fname = getattr(data, "filename", None)
            color = cmap.get(fname, "#0078D4")
            # Capture the per-dataset refs each sub-chart created so a later
            # set_dataset_visible can hide/show this dataset's overlays with
            # its curve (sidebar eye toggle).
            refs_x = self._chart_x.show_analysis(t, xv, fs, color, window=window)
            refs_y = self._chart_y.show_analysis(t, yv, fs, color, window=window)
            refs_z = self._chart_z.show_analysis(t, zv, fs, color, window=window)
            self._analysis_by_dataset[plot_order] = {
                "x": refs_x, "y": refs_y, "z": refs_z}

            # Respect the dataset's CURRENT visibility: if the curve is
            # hidden, hide its overlays too so a freshly-enabled analysis
            # doesn't draw markers over an invisible curve. Read the actual
            # curve's isVisible() (the source of truth) rather than a separate
            # flag, so there's no second state to keep in sync.
            curves = self._series_by_dataset.get(plot_order)
            if curves and not curves[0].isVisible():
                for refs in (refs_x, refs_y, refs_z):
                    _SubChart.set_analysis_visible(refs, False)

        # Force-restore the view so the pan/zoom state is untouched.
        for sub, (xmin, xmax, ymin, ymax) in saved.items():
            sub.axis_x.setRange(xmin, xmax)
            sub.axis_y.setRange(ymin, ymax)

        self._analysis_active = True
        return True

    @property
    def analysis_active(self) -> bool:
        return self._analysis_active
