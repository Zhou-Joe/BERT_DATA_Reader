# coding: utf-8
"""
Acceleration reversal analysis charts (Qt Charts).

Two container views ported from the archived matplotlib app:
  - ``ZReversalChartView``  — one chart: Z acceleration vs time.
  - ``XYReversalChartView`` — two stacked charts: X and Y acceleration vs time.

Both detect direction reversals via the algorithms in
``sci_calculation.detect_z_acceleration_reversal`` /
``detect_xy_acceleration_reversal`` (ported verbatim from the legacy app) and
draw them as translucent window bands + ▼/▲ min/max markers + value labels,
mirroring the trend tab's pattern (zoom/pan/reset, labels that follow the
data anchor on zoom).
"""

from __future__ import annotations

import numpy as np

from PySide6.QtCharts import (
    QChart, QChartView, QLineSeries, QAreaSeries, QValueAxis, QScatterSeries,
)
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPainter, QColor, QPen, QCursor
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy

from chart_base import PanChartView, INTERACTION_ZOOM, INTERACTION_PAN, INTERACTION_DEFAULT
from chart_utils import create_line_series, clear_chart, reset_chart_view
from sci_calculation import (
    detect_z_acceleration_reversal, detect_xy_acceleration_reversal,
)
from theme import LINE_WIDTH_DEFAULT, chart_tooltip_style, apply_chart_theme


# Per-event palette (cycled so overlapping windows read distinctly). Picked to
# stay readable on both light and dark themes.
_REVERSAL_COLORS = [
    "#E8112D",  # red
    "#107C10",  # green
    "#CA5010",  # orange
    "#8764B8",  # purple
    "#0099BC",  # cyan
    "#C239B3",  # magenta
]
# A reversal pair whose peakA or peakB exceeds mult×allowance FAILS the check;
# those pairs are drawn in this distinct dark red so the violations stand out.
# Passing pairs are NOT drawn (only the failures are highlighted).
_FAIL_COLOR = "#C00000"
# Dashed allowance-threshold lines (±0.5×allowance) colour.
_ALLOWANCE_LINE_COLOR = "#0078D4"  # accent blue, readable on light/dark


class _ReversalSubChart(PanChartView):
    """A single axis-vs-time sub-chart (one of the reversal containers).

    Mirrors ``chart_trend._SubChart``: owns a QChart + time/accel axes, a
    series-range cache for fast reset_view, and the analysis-overlay machinery
    (window bands + min/max markers + anchored labels that follow zoom/pan).
    """

    def __init__(self, y_label: str, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)

        self._chart = QChart()
        self._chart.legend().hide()
        self._chart.layout().setContentsMargins(4, 4, 4, 4)

        self._axis_x = QValueAxis()
        self._axis_x.setTitleText("time (s)")
        self._axis_x.setTickCount(10)
        self._chart.addAxis(self._axis_x, Qt.AlignmentFlag.AlignBottom)

        self._axis_y = QValueAxis()
        self._axis_y.setTitleText(y_label)
        self._chart.addAxis(self._axis_y, Qt.AlignmentFlag.AlignLeft)

        self.setChart(self._chart)

        # Annotation state (mirrors trend._SubChart).
        self.annot_mode: str = "Single"
        self._annot_markers: QScatterSeries | None = None
        self._annot_labels: list[QLabel] = []
        self._tip_label: QLabel | None = None
        from PySide6.QtCore import QTimer
        self._tip_timer = QTimer(self)
        self._tip_timer.setSingleShot(True)
        self._tip_timer.timeout.connect(self._hide_tip)

        # Series-range cache (xmin,xmax,ymin,ymax) for O(1) reset_view.
        self._series_ranges: dict = {}
        # Analysis overlays (window bands + markers + labels) — kept here so
        # they aren't GC'd and can be removed by clear_analysis().
        self._analysis_refs: list = []
        # Anchored labels that must follow their data-coordinate anchor on
        # zoom/pan. Each entry: (QLabel, QPointF anchor, dx, dy).
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
            print(f"reversal sub-chart theme apply failed: {e}")

    def restyle_theme(self) -> None:
        """Re-apply the current palette after a light/dark switch."""
        apply_chart_theme(self._chart)

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
        """Wipe everything: data series, axes, analysis overlays, annotations.

        Rebuilds clean axes and re-wires the rangeChanged connections (clear_chart
        drops the old axes, so the __init__ connections are stale afterwards).
        """
        # Capture the y-axis title BEFORE replacing self._axis_y (the old axis
        # object still holds its title text once detached from the chart).
        y_title = self._axis_y.titleText() or ""
        clear_chart(self._chart)
        # Re-add clean axes.
        self._axis_x = QValueAxis()
        self._axis_x.setTitleText("time (s)")
        self._axis_x.setTickCount(10)
        self._chart.addAxis(self._axis_x, Qt.AlignmentFlag.AlignBottom)

        self._axis_y = QValueAxis()
        self._axis_y.setTitleText(y_title)
        self._chart.addAxis(self._axis_y, Qt.AlignmentFlag.AlignLeft)

        # Re-wire label repositioning (clear_chart removed the old axes).
        self._axis_x.rangeChanged.connect(self._reposition_analysis_labels)
        self._axis_y.rangeChanged.connect(self._reposition_analysis_labels)

        # Drop annotation + analysis state.
        self._annot_markers = None
        for lbl in self._annot_labels:
            lbl.deleteLater()
        self._annot_labels = []
        self.clear_analysis()
        self._series_ranges.clear()

    def add_data_series(self, time_col, value_col, color: str) -> QLineSeries:
        """Add a measured-acceleration line series + cache its bounds."""
        series = create_line_series(time_col, value_col, color=color,
                                     width=LINE_WIDTH_DEFAULT)
        self._chart.addSeries(series)
        series.attachAxis(self._axis_x)
        series.attachAxis(self._axis_y)
        series.clicked.connect(self._on_series_clicked)
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
        """Recalculate x/y range so all attached line series are visible.

        Reads cached per-series bounds instead of iterating pointsVector() on
        every reset_view (see trend._SubChart.fit_data).
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
    # Reversal overlays — window bands + min/max markers + value labels.
    # ------------------------------------------------------------------

    def clear_analysis(self) -> None:
        """Remove all reversal overlays (bands, markers, labels)."""
        for ref in self._analysis_refs:
            try:
                if isinstance(ref, QLabel):
                    ref.deleteLater()
                else:
                    # QSeries — remove from the chart if still attached.
                    self._chart.removeSeries(ref)
            except Exception:
                pass
        self._analysis_refs.clear()
        self._analysis_labels.clear()

    def draw_allowance_lines(self, neg_full: float, pos_full: float) -> None:
        """Draw the 1× and 0.5× allowance threshold line PAIRS (4 dashed lines).

        Each reversal pair's acceptance limit depends on its peak-to-peak time
        (>0.2 s → full allowance, ≤0.2 s → 0.5×), and a chart can contain BOTH
        kinds of pairs, so BOTH reference levels are always drawn:
          • ±``neg_full`` / ±``pos_full``  → the 1×limit,
          • ±0.5× those                    → the 0.5×limit.

        ``neg_full`` is negative (the −side full allowance), ``pos_full``
        positive. Drawn BEFORE per-event marking so events overlay them, and
        tracked in ``_analysis_refs`` so ``clear_analysis()`` removes them. Each
        line gets a right-edge value label ("1×limit: +X.XX" / "0.5×limit: …")
        that follows zoom via the anchor mechanism.
        """
        if len(self._chart.series()) == 0:
            return
        x_lo = self._axis_x.min()
        x_hi = self._axis_x.max()
        pen = QPen(QColor(_ALLOWANCE_LINE_COLOR), 1.5, Qt.PenStyle.DashLine)
        # The four levels: (label, y-value, dy offset for its label).
        levels = [
            ("1×limit: %+.2f" % pos_full,        pos_full,        -16),
            ("0.5×limit: %+.2f" % (pos_full * 0.5), pos_full * 0.5,  -16),
            ("1×limit: %+.2f" % neg_full,        neg_full,         6),
            ("0.5×limit: %+.2f" % (neg_full * 0.5), neg_full * 0.5,   6),
        ]
        for label, yv, dy in levels:
            line = QLineSeries()
            line.append(QPointF(x_lo, yv))
            line.append(QPointF(x_hi, yv))
            line.setPen(pen)
            self._chart.addSeries(line)
            line.attachAxis(self._axis_x)
            line.attachAxis(self._axis_y)
            self._analysis_refs.append(line)
            # Right-edge value label anchored at (x_hi, yv).
            self._add_anchor_label(label, QPointF(x_hi, yv),
                                   dx=-120, dy=dy)

    def show_z_reversals(self, time_arr, value_arr, reversals: list) -> list:
        """Draw one Z-reversal-window overlay per detected event (min/max rule).

        Used by the Z Reversal tab. Each event renders as:
          • a translucent vertical band spanning [window_start, window_end]
            across the current y-range (the Qt-Charts equivalent of matplotlib's
            ``axvspan`` — two vertical boundary lines + QAreaSeries, with the
            border camouflaged to the fill colour so Qt 6.x's border bug is
            hidden, same trick as the trend sustained region / acczone fills),
          • a ▼ (down-triangle) ``QScatterSeries`` marker at
            (min_time, min_value) and an ▲ (up-triangle) at
            (max_time, max_value),
          • value labels (0.003g) anchored at each marker so they follow zoom.

        Returns the list of created overlays (so the caller can track them).
        """
        created: list = []
        if time_arr is None or len(value_arr) == 0 or not reversals:
            return created
        if len(self._chart.series()) == 0:
            return created

        # Vertical extent for the bands: use the CURRENT y-axis range so bands
        # cover the whole visible window regardless of where the markers sit
        # (matches the matplotlib axvspan look). Re-evaluated each call so a
        # re-analysis after a zoom still spans the view.
        y_lo = self._axis_y.min()
        y_hi = self._axis_y.max()

        for i, rev in enumerate(reversals):
            color_hex = _REVERSAL_COLORS[i % len(_REVERSAL_COLORS)]
            fill = QColor(color_hex)
            fill.setAlpha(60)
            self._add_window_band(rev["window_start"], rev["window_end"],
                                  y_lo, y_hi, fill, created)
            self._add_reversal_marker(rev, color_hex, created)
        return created

    def show_xy_reversals(self, time_arr, value_arr, reversals: list,
                          thresholds=None) -> tuple:
        """Draw one X/Y-reversal overlay per detected peak-to-peak pair.

        Used by the X/Y Reversal tab. Each event is a pair of adjacent sustained
        runs of opposite sign (run A then run B), detected by
        ``detect_xy_acceleration_reversal``. Each DRAWN pair renders as:
          (a) a translucent vertical band spanning run A start → run B end
              (the two sustained segments that form the transition),
          (b) a filled area between the y=0 line and each run's curve segment
              (one fill per run, via ``_add_run_fill``),
          (c) a ▲ at peakA (triangle up for a +sign peak) and a ▼/■ at peakB,
              with anchored value labels,
          (d) a multi-line label "Δt=…s → N×limit / peakA +… / peakB −…" anchored
              near the band top.

        ``thresholds`` (optional) is ``(neg_allowance, pos_allowance)`` — the
        FULL (1×) ASTM allowance for this axis. Each pair's acceptance multiplier
        ``mult`` (1.0 if its Δt > 0.2 s, else 0.5) scales the allowance, and the
        pair FAILS if peakA OR peakB exceeds ``mult × allowance`` on its own sign
        side. ONLY failing pairs are drawn (in the fail colour) so the chart
        highlights just the violations. When ``thresholds`` is None every pair is
        drawn in the cycled palette (no fail check).

        Returns ``(created, fail_count)`` where ``created`` is the list of drawn
        overlays and ``fail_count`` is how many pairs exceeded the limit
        (0 when ``thresholds`` is None).
        """
        created: list = []
        fail_count = 0
        if time_arr is None or len(value_arr) == 0 or not reversals:
            return created, fail_count
        if len(self._chart.series()) == 0:
            return created, fail_count

        # Vertical extent for the pair bands: the current y-range so bands span
        # the visible window regardless of where the curve sits.
        y_lo = self._axis_y.min()
        y_hi = self._axis_y.max()

        for i, rev in enumerate(reversals):
            color_hex = _REVERSAL_COLORS[i % len(_REVERSAL_COLORS)]

            # Allowance check (peak-to-peak): scale the full allowance by this
            # pair's mult (1× if Δt>0.2s else 0.5×) and compare BOTH peaks.
            failed = False
            if thresholds is not None:
                failed = self._peaks_exceed_limit(rev, thresholds)
                if not failed:
                    # Passing pair — don't plot it. Only violations are drawn.
                    continue
                fail_count += 1
                color_hex = _FAIL_COLOR

            fill = QColor(color_hex)
            fill.setAlpha(90)          # area fill (curve-to-0)
            band_fill = QColor(color_hex)
            band_fill.setAlpha(45)     # the pair's window band (lighter)

            # (a) the pair band: run A start → run B end.
            self._add_window_band(rev["window_start"], rev["window_end"],
                                  y_lo, y_hi, band_fill, created)

            # (b) fill each sustained run to y=0 (the two segments of the
            # peak-to-peak transition).
            self._add_run_fill(time_arr, value_arr,
                               int(rev["seg_a_start_idx"]),
                               int(rev["seg_a_end_idx"]),
                               int(rev["signA"]), color_hex, fill, created)
            self._add_run_fill(time_arr, value_arr,
                               int(rev["seg_b_start_idx"]),
                               int(rev["seg_b_end_idx"]),
                               int(rev["signB"]), color_hex, fill, created)

            # (c) ▲/▼ peak markers + value labels.
            self._add_peak_marker(rev["peakA_time"], rev["peakA_value"],
                                  int(rev["signA"]), color_hex, created)
            self._add_peak_marker(rev["peakB_time"], rev["peakB_value"],
                                  int(rev["signB"]), color_hex, created)

            # (d) the Δt / mult / peak-values summary label, anchored near the
            # top of the band.
            mult = float(rev.get("mult", 1.0))
            limit_txt = "0.5×limit" if mult < 1.0 else "1×limit"
            sign_sym_a = "+" if int(rev["signA"]) > 0 else "−"
            sign_sym_b = "+" if int(rev["signB"]) > 0 else "−"
            va = abs(float(rev["peakA_value"]))
            vb = abs(float(rev["peakB_value"]))
            text = (f"\u0394t={float(rev['delta_t']):.3f}s \u2192 {limit_txt}\n"
                    f"peakA {sign_sym_a}{va:.2f} / "
                    f"peakB {sign_sym_b}{vb:.2f}")
            anchor_x = float(rev["peakA_time"])
            anchor_y = float(rev["peakA_value"])
            self._add_anchor_label(text, QPointF(anchor_x, anchor_y),
                                   dx=8, dy=-22)

        return created, fail_count

    def _peaks_exceed_limit(self, rev: dict, allowance_pair) -> bool:
        """Did this peak-to-peak pair exceed its effective allowance limit?

        ``allowance_pair`` is ``(neg_allow, pos_allow)`` — the FULL (1×) ASTM
        allowance for this axis. The pair's ``mult`` (1.0 if Δt > 0.2 s else
        0.5) scales it, and the pair FAILS if peakA OR peakB exceeds
        ``mult × allowance`` on its own sign side:
          • a +sign peak fails if value > pos_allow × mult
          • a −sign peak fails if value < neg_allow × mult
            (neg_allow is negative, so × mult shrinks it toward 0).
        """
        neg_allow, pos_allow = allowance_pair
        mult = float(rev.get("mult", 1.0))
        # Peak A.
        if int(rev["signA"]) > 0:
            if float(rev["peakA_value"]) > pos_allow * mult:
                return True
        else:
            if float(rev["peakA_value"]) < neg_allow * mult:
                return True
        # Peak B.
        if int(rev["signB"]) > 0:
            if float(rev["peakB_value"]) > pos_allow * mult:
                return True
        else:
            if float(rev["peakB_value"]) < neg_allow * mult:
                return True
        return False

    def _add_peak_marker(self, t: float, v: float, sign: int,
                         color_hex: str, created: list) -> None:
        """Draw a peak marker (▲ for +peak, ■ for −peak) + an anchored label."""
        positive = (sign > 0)
        marker = QScatterSeries()
        marker.setMarkerSize(12)
        marker.setColor(QColor(color_hex))
        marker.setBorderColor(QColor("white"))
        if positive:
            marker.setMarkerShape(QScatterSeries.MarkerShape.MarkerShapeTriangle)
        else:
            marker.setMarkerShape(
                QScatterSeries.MarkerShape.MarkerShapeRectangle)
        marker.append(QPointF(t, v))
        self._chart.addSeries(marker)
        marker.attachAxis(self._axis_x)
        marker.attachAxis(self._axis_y)
        self._analysis_refs.append(marker)
        created.append(marker)
        # Value label anchored at the peak (follows zoom/pan).
        self._add_anchor_label(f"{v:+.2f}", QPointF(t, v),
                               dx=6, dy=-16 if positive else 3)

    def _add_run_fill(self, time_arr, value_arr, start_idx: int,
                      end_idx: int, sign: int, color_hex: str,
                      fill: QColor, created: list) -> None:
        """Fill between y=0 and the curve over one sustained run ``[start, end]``.

        Used by the X/Y Reversal tab: each direction-change PAIR has two runs
        (the prior run A and the new run B), and this fills each one to the
        y=0 baseline so the viewer sees the two sustained segments that form
        the peak-to-peak transition.

        Boundary QLineSeries (the curve segment + a flat y=0 line) define a
        QAreaSeries; the boundary pens are camouflaged to the fill colour (Qt
        6.x border bug). A SEPARATE visible y=0 line is drawn on top. Many
        sample points are used (the full run) so the area fill stays strictly
        within ``[start_idx, end_idx]`` (the 2-point degenerate-line overflow
        bug).
        """
        if end_idx < start_idx or end_idx >= len(value_arr):
            return
        start_idx = max(0, start_idx)
        positive = (sign > 0)

        # Boundary series over [start_idx, end_idx]: the actual curve + flat y=0.
        inv_pen = QPen(fill, 1, Qt.PenStyle.SolidLine)
        curve_boundary = QLineSeries()
        flat_boundary = QLineSeries()
        for k in range(start_idx, end_idx + 1):
            tx = float(time_arr[k])
            curve_boundary.append(QPointF(tx, float(value_arr[k])))
            flat_boundary.append(QPointF(tx, 0.0))
        curve_boundary.setPen(inv_pen)
        flat_boundary.setPen(inv_pen)

        # For a positive run the curve sits above 0: fill between flat (lower)
        # and curve (upper). For a negative run the curve sits below 0: fill
        # between curve (lower) and flat (upper).
        if positive:
            upper, lower = curve_boundary, flat_boundary
        else:
            upper, lower = flat_boundary, curve_boundary

        area = QAreaSeries(upper, lower)
        area.setColor(fill)
        area.setBorderColor(fill)
        area.setPen(inv_pen)
        self._chart.addSeries(area)
        area.attachAxis(self._axis_x)
        area.attachAxis(self._axis_y)
        # Keep strong Python refs so the boundary lines aren't GC'd.
        self._analysis_refs.extend([area, upper, lower])
        created.extend([area, upper, lower])

        # Visible y=0 line on top of the fill (2-point QLineSeries).
        t0 = float(time_arr[start_idx])
        t1 = float(time_arr[end_idx])
        flat_line = QLineSeries()
        flat_line.append(QPointF(t0, 0.0))
        flat_line.append(QPointF(t1, 0.0))
        flat_line.setPen(QPen(QColor(color_hex), 2, Qt.PenStyle.SolidLine))
        self._chart.addSeries(flat_line)
        flat_line.attachAxis(self._axis_x)
        flat_line.attachAxis(self._axis_y)
        self._analysis_refs.append(flat_line)
        created.append(flat_line)

    def _add_window_band(self, t0: float, t1: float, y_lo: float, y_hi: float,
                         fill: QColor, created: list) -> None:
        """Draw a translucent vertical band across [t0, t1] × [y_lo, y_hi].

        Builds the band as a thin rectangle: a lower horizontal QLineSeries at
        ``y_lo`` and an upper one at ``y_hi`` (each spanning [t0, t1]); a
        ``QAreaSeries`` fills the rectangle between them. (Two vertical lines
        don't fill reliably — Qt Charts' area fill evaluates the upper/lower
        series at common x values, so vertical boundaries are degenerate.)

        The boundary pens are camouflaged to the fill colour so Qt Charts 6.x's
        border-draw bug doesn't show a hard outline (same pattern as the trend
        sustained region and the acczone fills).
        """
        inv_pen = QPen(fill, 1, Qt.PenStyle.SolidLine)

        lower = QLineSeries()   # lower boundary: horizontal at y_lo
        lower.append(QPointF(t0, y_lo))
        lower.append(QPointF(t1, y_lo))
        upper = QLineSeries()   # upper boundary: horizontal at y_hi
        upper.append(QPointF(t0, y_hi))
        upper.append(QPointF(t1, y_hi))
        lower.setPen(inv_pen)
        upper.setPen(inv_pen)

        area = QAreaSeries(upper, lower)
        area.setColor(fill)
        area.setBorderColor(fill)
        area.setPen(inv_pen)

        self._chart.addSeries(area)
        area.attachAxis(self._axis_x)
        area.attachAxis(self._axis_y)
        # Keep strong Python refs so the boundary lines aren't GC'd.
        self._analysis_refs.extend([area, upper, lower])
        created.extend([area, upper, lower])

    def _add_reversal_marker(self, rev: dict, color_hex: str,
                             created: list) -> None:
        """Draw ▼ at the min point + ▲ at the max point + value labels."""
        min_t = float(rev["min_time"]); min_v = float(rev["min_value"])
        max_t = float(rev["max_time"]); max_v = float(rev["max_value"])

        # ▼ at the minimum.
        down = QScatterSeries()
        down.setMarkerShape(QScatterSeries.MarkerShape.MarkerShapeRectangle)
        down.setMarkerSize(12)
        down.setColor(QColor(color_hex))
        down.setBorderColor(QColor("white"))
        down.append(QPointF(min_t, min_v))
        self._chart.addSeries(down)
        down.attachAxis(self._axis_x)
        down.attachAxis(self._axis_y)
        self._analysis_refs.append(down)
        created.append(down)

        # ▲ at the maximum.
        up = QScatterSeries()
        up.setMarkerShape(QScatterSeries.MarkerShape.MarkerShapeTriangle)
        up.setMarkerSize(12)
        up.setColor(QColor(color_hex))
        up.setBorderColor(QColor("white"))
        up.append(QPointF(max_t, max_v))
        self._chart.addSeries(up)
        up.attachAxis(self._axis_x)
        up.attachAxis(self._axis_y)
        self._analysis_refs.append(up)
        created.append(up)

        # Value labels anchored at each marker (follow zoom/pan).
        self._add_anchor_label(f"{min_v:+.3f}", QPointF(min_t, min_v),
                               dx=6, dy=3)
        self._add_anchor_label(f"{max_v:+.3f}", QPointF(max_t, max_v),
                               dx=6, dy=-16)

    def _add_anchor_label(self, text: str, anchor: QPointF,
                          dx: int, dy: int) -> None:
        """Create a QLabel at a data-coordinate anchor + store it for reposition."""
        lbl = QLabel(text, self)
        lbl.setStyleSheet(chart_tooltip_style())
        lbl.adjustSize()
        pos = self._chart.mapToPosition(anchor)
        lx = min(int(pos.x()) + dx, max(self.width() - lbl.width() - 4, 2))
        ly = max(int(pos.y()) + dy, 2)
        lbl.move(lx, ly)
        lbl.show()
        lbl.raise_()
        self._analysis_refs.append(lbl)
        self._analysis_labels.append((lbl, anchor, dx, dy))

    def _reposition_analysis_labels(self, *args) -> None:
        """Re-clamp every reversal label to its data-coordinate anchor.

        QLabels are pixel-positioned widgets, so zoom/pan (which only re-maps
        data↔pixel) would leave them stranded. Re-run mapToPosition(anchor) and
        re-apply the stored pixel offset. Connected to both axes' rangeChanged.
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

    # ------------------------------------------------------------------
    # Pan linkage — bubble up to the parent container so it can mirror this
    # chart's time range onto the sibling(s).
    # ------------------------------------------------------------------

    def _pan_applied(self) -> None:
        parent = self.parent()
        # The container's helper mirrors this chart's time range onto siblings.
        if hasattr(parent, "_sync_from_panned"):
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
            m.attachAxis(self._axis_x)
            m.attachAxis(self._axis_y)
            self._chart.addSeries(m)
            self._annot_markers = m
        return self._annot_markers

    def clear_annotations(self) -> None:
        if self._annot_markers is not None:
            self._annot_markers.replace([])
        for lbl in self._annot_labels:
            lbl.deleteLater()
        self._annot_labels = []

    def add_annotation(self, point: QPointF) -> None:
        if self.annot_mode == "Single":
            self.clear_annotations()
        markers = self._ensure_annot_markers()
        pts = list(markers.points())
        pts.append(point)
        markers.replace(pts)
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

    def _on_series_clicked(self, point: QPointF) -> None:
        """Place an annotation marker + label at the clicked data point."""
        self.add_annotation(point)


# ===========================================================================
# Public containers
# ===========================================================================

class _ReversalContainerBase(QWidget):
    """Shared container logic for the Z and X/Y reversal views.

    Holds the cached source arrays (time + axis values + fs) captured at plot
    time so the Analyze toggle can recompute reversals without re-pulling from
    QLineSeries. Subclasses populate ``_subcharts`` and implement
    ``_plot_curve`` / ``_compute_reversals_per_axis``.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.annot_mode: str = "Single"
        # Cached source arrays per axis key: {key: (time, axis_values, fs)}.
        self._cached_data: dict[str, tuple] = {}
        self._prepared_reversals = None
        # Whether reversal overlays are currently shown.
        self._analysis_active: bool = False
        # Number of detected reversal events across all axes (for the toolbar
        # summary, read by main_window).
        self.event_count: int = 0
        # Number of peak-to-peak pairs that FAILED their effective limit (X/Y
        # tab only; the Z tab never sets an allowance so this stays 0).
        self.fail_count: int = 0
        self._subcharts: list[_ReversalSubChart] = []
        # Measured-curve series currently plotted (one per sub-chart), tracked
        # so a re-plot removes the previous curve instead of stacking on top.
        self._data_series: dict[str, QLineSeries] = {}
        # Per-axis FULL (1×) ASTM allowance, {axis_key: (neg, pos)}, or None to
        # disable the peak-to-peak fail check (the Z tab leaves this None). Each
        # reversal pair's own mult (1× if Δt>0.2s else 0.5) scales this at check
        # time.
        self._allowance: dict[str, tuple] | None = None
        # Hysteresis dead-band (g) for the X/Y sign computation: a sample only
        # commits to ±sign once |value| exceeds this, so a tiny noise/filter
        # glitch poking across zero doesn't create a spurious extra sustained
        # run. The X/Y tab tunes this via set_deadband; the Z tab ignores it.
        self._deadband: float = 0.05

    # -- subclasses override these ---------------------------------------

    def _plot_curve(self, data_df, color: str, fs: float) -> None:
        raise NotImplementedError

    def _axis_keys(self) -> list[str]:
        raise NotImplementedError

    def _compute_reversals_per_axis(self) -> dict[str, list]:
        """Return {axis_key: [reversal dicts]} for the cached data."""
        raise NotImplementedError

    def _render_reversals(self, sub, t, vals, revs: list,
                          thresholds=None) -> int:
        """Render one sub-chart's reversal overlays. Subclasses override to
        pick the Z min/▲max/▼ path vs the X/Y peak-pair band+fill path.

        ``thresholds`` is this axis's ``(neg_allow, pos_allow)`` FULL allowance
        (X/Y) or None (Z). Returns the number of failed pairs on this sub-chart.
        """
        raise NotImplementedError

    def set_allowance(self, allowance: dict | None) -> None:
        """Set the per-axis FULL (1×) allowance for the peak-to-peak fail check.

        ``allowance`` is ``{axis_key: (neg, pos)}`` where each tuple is the
        full ASTM t=0 allowance (neg negative, pos positive), or None to
        disable. Each reversal pair's own ``mult`` (1.0 if its Δt > 0.2 s else
        0.5) scales this allowance at check time, and the sub-chart draws BOTH
        the 1×limit and 0.5×limit reference lines. Takes effect on the next
        Analyze toggle / re-plot. The Z tab never calls this (stays None → no
        allowance lines, no fail marking).
        """
        self._allowance = allowance

    def set_deadband(self, deadband: float) -> None:
        """Set the hysteresis dead-band (g) for X/Y sign-change detection.

        Takes effect on the next Analyze toggle / re-plot. 0 disables
        hysteresis (plain-sign behaviour). The Z tab ignores this.
        """
        try:
            self._deadband = max(0.0, float(deadband))
        except Exception:
            pass

    # -- shared public API -----------------------------------------------

    def initplotxyz(self) -> None:
        """Clear every sub-chart + cached data + analysis state."""
        for sub in self._subcharts:
            sub.clear_all()
            # Recreate the sub-chart's viewport so no stale pixels from the
            # previous (populated) chart linger on a hidden page — a reset must
            # look exactly like the app's initial state.
            reset_chart_view(sub)
        self._cached_data.clear()
        self._prepared_reversals = None
        self._data_series.clear()
        self._analysis_active = False
        self.event_count = 0
        self.fail_count = 0

    def _remove_data_series(self) -> None:
        """Remove the previously-plotted measured curve from every sub-chart.

        Called before each re-plot so repeated Generate/Update calls don't
        stack curves on top of each other (mirrors GB/AccZone's pattern).
        """
        for key, s in list(self._data_series.items()):
            try:
                chart = s.chart()
                if chart is not None:
                    chart.removeSeries(s)
            except Exception:
                pass
            # Also drop it from the sub-chart's series-range cache.
            sub = self._subcharts_by_key(key)
            if sub is not None:
                sub._series_ranges.pop(s, None)
        self._data_series.clear()

    def _subcharts_by_key(self, key: str):
        """Return the sub-chart that plots *key* (default: by axis_keys order)."""
        keys = self._axis_keys()
        if key in keys:
            idx = keys.index(key)
            if idx < len(self._subcharts):
                return self._subcharts[idx]
        return None

    def plot_data(self, data_df, color: str, fs: float, reversals=None) -> None:
        """Plot the acceleration curve(s) + cache source arrays for analysis."""
        # Remember whether analysis overlays were ON so a Generate/Update or
        # cutoff-change re-plot can restore them (mirrors chart_acczone's
        # "if self._analysis_active: self._apply_analysis()" pattern).
        prev_active = self._analysis_active
        # Remove the previous measured curve(s) + any analysis overlays so a
        # re-plot starts clean (no stacking).
        self._remove_data_series()
        for sub in self._subcharts:
            sub.clear_analysis()
        self._analysis_active = False
        self.event_count = 0
        self.fail_count = 0
        self._plot_curve(data_df, color, fs)
        self._prepared_reversals = reversals
        # Re-draw the overlays on the fresh curve if they were on before the
        # re-plot (the shared Analyze button stays pressed — don't drop them).
        if prev_active:
            self.toggle_analysis(True)

    def toggle_analysis(self, active: bool) -> None:
        """Show or hide reversal overlays on every sub-chart."""
        if not active:
            for sub in self._subcharts:
                sub.clear_analysis()
            self._analysis_active = False
            self.event_count = 0
            self.fail_count = 0
            return

        # Adding series can auto-fit the axes; save + restore the current
        # ranges so toggling analysis doesn't change the user's view (same fix
        # as the trend tab's toggle_analysis).
        saved = []
        for sub in self._subcharts:
            saved.append((sub.axis_x.min(), sub.axis_x.max(),
                          sub.axis_y.min(), sub.axis_y.max()))

        reversals_by_axis = (self._prepared_reversals if self._prepared_reversals is not None
                             else self._compute_reversals_per_axis())
        total = 0
        failed = 0
        for sub, key in zip(self._subcharts, self._axis_keys()):
            cached = self._cached_data.get(key)
            revs = reversals_by_axis.get(key, [])
            if cached is None:
                continue
            t, vals, _fs = cached
            # X/Y tab: draw the 1×limit + 0.5×limit reference lines (if the tab
            # set a full allowance) before the per-pair overlays, and pass the
            # per-axis full allowance so each pair can be fail-marked against
            # its own mult×allowance. Z tab has no allowance → None, no lines.
            thresholds = None
            if self._allowance is not None:
                thresholds = self._allowance.get(key)
                if thresholds is not None:
                    sub.draw_allowance_lines(thresholds[0], thresholds[1])
            failed += self._render_reversals(sub, t, vals, revs, thresholds)
            total += len(revs)

        # Restore the saved view ranges.
        for sub, (xlo, xhi, ylo, yhi) in zip(self._subcharts, saved):
            sub.axis_x.setRange(xlo, xhi)
            sub.axis_y.setRange(ylo, yhi)

        self._analysis_active = True
        self.event_count = total
        self.fail_count = failed

    def draw_idle(self) -> None:
        pass

    def set_annotation_mode(self, mode: str) -> None:
        self.annot_mode = mode
        for sub in self._subcharts:
            sub.annot_mode = mode

    def clear_annotations(self) -> None:
        for sub in self._subcharts:
            sub.clear_annotations()

    def restyle_theme(self) -> None:
        for sub in self._subcharts:
            sub.restyle_theme()

    def zoom_mode(self, mode: str) -> None:
        """Set interaction mode for all sub-charts (zoom / pan / default)."""
        for sub in self._subcharts:
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
        """Reset every sub-chart to fit its data (recompute from cached ranges)."""
        for sub in self._subcharts:
            sub.fit_data()


class ZReversalChartView(_ReversalContainerBase):
    """Single chart: Z acceleration vs time + reversal-window overlays."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)

        self._chart_z = _ReversalSubChart("Vertical (g)", self)
        self._subcharts = [self._chart_z]
        layout.addWidget(self._chart_z, 1)

    def _axis_keys(self) -> list[str]:
        return ["z"]

    def _plot_curve(self, data_df, color: str, fs: float) -> None:
        t = data_df.iloc[:, 0].values.astype(float)
        zv = data_df.iloc[:, 3].values.astype(float)
        s = self._chart_z.add_data_series(t, zv, color)
        self._chart_z.fit_data()
        self._data_series["z"] = s
        self._cached_data["z"] = (t, zv, fs)

    def _compute_reversals_per_axis(self) -> dict[str, list]:
        cached = self._cached_data.get("z")
        if cached is None:
            return {}
        t, zv, fs = cached
        return {"z": detect_z_acceleration_reversal(zv, t, fs,
                                                    deadband=self._deadband)}

    def _render_reversals(self, sub, t, vals, revs: list,
                          thresholds=None) -> int:
        """Z reversals: translucent band + ▼/▲ min/max markers + value labels.

        Z tab has no allowance check, so thresholds is ignored and 0 is returned.
        """
        sub.show_z_reversals(t, vals, revs)
        return 0


class XYReversalChartView(_ReversalContainerBase):
    """Two stacked charts: X and Y acceleration vs time + reversal overlays."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)

        self._chart_x = _ReversalSubChart("Fore/Aft (g)", self)
        self._chart_y = _ReversalSubChart("Lateral (g)", self)
        self._subcharts = [self._chart_x, self._chart_y]
        layout.addWidget(self._chart_x, 1)
        layout.addWidget(self._chart_y, 1)

        # X-axis cross-link: keep the two charts horizontally aligned on
        # pan/zoom (mirror trend's _sync_x_axis). Guard against recursion.
        self._syncing = False
        for sub in (self._chart_x, self._chart_y):
            sub.axis_x.rangeChanged.connect(self._sync_x_axis)

    def _axis_keys(self) -> list[str]:
        return ["x", "y"]

    def _plot_curve(self, data_df, color: str, fs: float) -> None:
        t = data_df.iloc[:, 0].values.astype(float)
        xv = data_df.iloc[:, 1].values.astype(float)
        yv = data_df.iloc[:, 2].values.astype(float)
        sx = self._chart_x.add_data_series(t, xv, color)
        sy = self._chart_y.add_data_series(t, yv, color)
        self._chart_x.fit_data()
        self._chart_y.fit_data()
        self._data_series["x"] = sx
        self._data_series["y"] = sy
        self._cached_data["x"] = (t, xv, fs)
        self._cached_data["y"] = (t, yv, fs)

    def _compute_reversals_per_axis(self) -> dict[str, list]:
        xc = self._cached_data.get("x")
        yc = self._cached_data.get("y")
        out: dict[str, list] = {}
        # detect_xy_acceleration_reversal needs BOTH axes (it returns a dict
        # with 'x_reversals' and 'y_reversals'). Call once when at least one is
        # cached; each axis's events come from its own column.
        if xc is not None or yc is not None:
            if xc is not None and yc is not None:
                t, xv, fs = xc
                _, yv, _ = yc
            elif xc is not None:
                t, xv, fs = xc
                yv = xv
            else:
                t, yv, fs = yc
                xv = yv
            res = detect_xy_acceleration_reversal(xv, yv, t, fs,
                                                    deadband=self._deadband)
            out["x"] = res["x_reversals"]
            out["y"] = res["y_reversals"]
        return out

    def _render_reversals(self, sub, t, vals, revs: list,
                          thresholds=None) -> int:
        """X/Y reversals: pair band + two run fills + peak markers + Δt/mult
        label. When ``thresholds`` (per-axis FULL allowance) is given, pairs
        whose peakA or peakB exceeds their own mult×allowance are fail-marked
        and only those are drawn. Returns the per-sub-chart fail count.
        """
        _created, fail = sub.show_xy_reversals(t, vals, revs, thresholds)
        return fail

    # ------------------------------------------------------------------
    # X-axis synchronisation (mirrors trend._sync_x_axis / _sync_from_panned)
    # ------------------------------------------------------------------

    def _sync_x_axis(self, min_val: float, max_val: float) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            for sub in (self._chart_x, self._chart_y):
                cur_min, cur_max = sub.axis_x.min(), sub.axis_x.max()
                if cur_min != min_val or cur_max != max_val:
                    sub.axis_x.setRange(min_val, max_val)
        finally:
            self._syncing = False

    def _sync_from_panned(self, panned: _ReversalSubChart) -> None:
        """After *panned* is dragged, mirror its time range onto the sibling."""
        ax = panned.axis_x
        self._sync_x_axis(ax.min(), ax.max())
