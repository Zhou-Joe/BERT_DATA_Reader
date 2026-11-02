# coding: utf-8
"""
GB 8408 fitting chart — three sub-charts:
  - Top-left:  ay vs dt  (with allowable ay reference)
  - Top-right: az vs dt  (with allowable az reference)
  - Bottom:    ay vs az   (combined, with dt=0.05/0.1/0.2s reference curves)

Replaces GBPlot (matplotlib).
"""

from __future__ import annotations

import numpy as np

from PySide6.QtCharts import (
    QChart, QChartView, QLineSeries, QValueAxis, QAreaSeries, QScatterSeries,
)
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPainter, QPen, QColor, QCursor
from PySide6.QtWidgets import QWidget, QGridLayout, QVBoxLayout, QSizePolicy, QLabel

from chart_base import BaseChartView, PanChartView, INTERACTION_ZOOM, INTERACTION_PAN, INTERACTION_DEFAULT
from chart_utils import (
    numpy_to_qpointsf,
    create_line_series,
    add_series_with_axes,
    clear_chart,
    equalize_xy_axes,
    reset_chart_view,
)
from matplotlib.path import Path
from chart_legend import ChartLegend
from boundary_analysis import series_to_path, find_boundary_violations
from sci_calculation import ProcessGB, sampling_frequency
from theme import (LINE_WIDTH_DEFAULT, LINE_WIDTH_FITTING, chart_tooltip_style,
                   apply_chart_theme, MEASURED_POS_COLOR, MEASURED_NEG_COLOR)


class GBChartView(QWidget):
    """Three-chart widget for GB 8408 acceleration analysis."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.annot_mode: str = "Single"

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Each chart view is wrapped in a container that stacks a wrapping
        # ChartLegend on top + the view below (same pattern as the ASTM charts)
        # so the legend can take as many rows as needed instead of being
        # clipped to one by the built-in QLegend.
        self._legends: dict[str, ChartLegend] = {}
        self._charts_by_key: dict[str, QChart] = {}

        def make_view(key: str):
            container = QWidget(self)
            cl = QVBoxLayout(container)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.setSpacing(0)
            legend = ChartLegend(container)
            self._legends[key] = legend
            cl.addWidget(legend)
            cv = PanChartView(container)
            cv.setRenderHint(QPainter.RenderHint.Antialiasing)
            cv.setSizePolicy(QSizePolicy.Policy.Expanding,
                              QSizePolicy.Policy.Expanding)
            cl.addWidget(cv, 1)
            return container, cv

        # --- ay chart (top-left) ---
        c_ay, self._cv_ay = make_view("ay")
        # --- az chart (top-right) ---
        c_az, self._cv_az = make_view("az")
        # --- combined chart (bottom, spanning) ---
        c_comb, self._cv_comb = make_view("comb")

        layout.addWidget(c_ay, 0, 0)
        layout.addWidget(c_az, 0, 1)
        layout.addWidget(c_comb, 1, 0, 1, 2)

        # Measured-data series references (for removal on replot)
        self._data_series: list[QLineSeries] = []
        # Combined-chart boundary analysis overlays (scatter markers),
        # created by toggle_comb_analysis. Kept so they can be removed / GC'd.
        self._comb_analysis_refs: list = []
        # Whether the combined-chart analysis overlays are shown.
        self._comb_analysis_active: bool = False
        # Last plotted raw DataFrame, cached for the analyze toggle.
        self._last_comb_data = None
        # Set of rule labels that were violated (read by main_window to show
        # the summary in the chart-tools toolbar).
        self.failed_rules: set[str] = set()

        self.initplotxyz()

    # ------------------------------------------------------------------
    # Initialisation — draw reference (allowable) curves
    # ------------------------------------------------------------------

    def initplotxyz(self) -> None:
        """Create charts with GB 8408 allowable reference curves."""
        self._data_series.clear()
        # Drop any combined-chart analysis overlays + cached data (rebuilt).
        self._clear_comb_analysis()
        self._comb_analysis_active = False
        self._last_comb_data = None

        # --- ay chart ---
        self._chart_ay = QChart()
        # Hide the built-in legend — a wrapping ChartLegend above each view
        # shows all entries without clipping (same as the ASTM charts).
        self._chart_ay.legend().setVisible(False)
        self._chart_ay.layout().setContentsMargins(4, 4, 4, 4)
        self._cv_ay.setChart(self._chart_ay)
        apply_chart_theme(self._chart_ay)
        self._charts_by_key["ay"] = self._chart_ay

        self._ax_ay_x = QValueAxis()
        self._ax_ay_x.setRange(0, 4)
        self._ax_ay_x.setTitleText("dt (s)")
        self._chart_ay.addAxis(self._ax_ay_x, Qt.AlignmentFlag.AlignBottom)

        self._ax_ay_y = QValueAxis()
        self._ax_ay_y.setRange(-5, 5)
        self._ax_ay_y.setTitleText("ay")
        self._chart_ay.addAxis(self._ax_ay_y, Qt.AlignmentFlag.AlignLeft)

        self._add_ref(self._chart_ay, self._ax_ay_x, self._ax_ay_y,
                       [0.05, 0.2, 1, 4], [3, 2, 2, 2],
                       "Allowable ay", "red", 3)
        self._add_ref(self._chart_ay, self._ax_ay_x, self._ax_ay_y,
                       [0.05, 0.2, 1, 4], [-3, -2, -2, -2],
                       "", "red", 3)

        # --- az chart ---
        self._chart_az = QChart()
        self._chart_az.legend().setVisible(False)
        self._chart_az.layout().setContentsMargins(4, 4, 4, 4)
        self._cv_az.setChart(self._chart_az)
        apply_chart_theme(self._chart_az)
        self._charts_by_key["az"] = self._chart_az

        self._ax_az_x = QValueAxis()
        self._ax_az_x.setRange(0, 4)
        self._ax_az_x.setTitleText("dt (s)")
        self._chart_az.addAxis(self._ax_az_x, Qt.AlignmentFlag.AlignBottom)

        self._ax_az_y = QValueAxis()
        self._ax_az_y.setRange(-2, 6)
        self._ax_az_y.setTitleText("az")
        self._chart_az.addAxis(self._ax_az_y, Qt.AlignmentFlag.AlignLeft)

        self._add_ref(self._chart_az, self._ax_az_x, self._ax_az_y,
                       [0, 1, 2, 3, 4], [6, 6, 4, 4, 4],
                       "Allowable az", "red", 3)
        self._add_ref(self._chart_az, self._ax_az_x, self._ax_az_y,
                       [0, 0.5, 2, 3, 4], [-2, -1.5, -1.5, -1.5, -1.5],
                       "", "red", 3)

        # --- combined chart ---
        self._chart_comb = QChart()
        self._chart_comb.legend().setVisible(False)
        self._chart_comb.layout().setContentsMargins(4, 4, 4, 4)
        self._cv_comb.setChart(self._chart_comb)
        apply_chart_theme(self._chart_comb)
        self._charts_by_key["comb"] = self._chart_comb

        self._ax_comb_x = QValueAxis()
        self._ax_comb_x.setRange(-2, 6)
        self._ax_comb_x.setTitleText("az")
        self._chart_comb.addAxis(self._ax_comb_x, Qt.AlignmentFlag.AlignBottom)

        self._ax_comb_y = QValueAxis()
        self._ax_comb_y.setRange(-3.5, 3.5)
        self._ax_comb_y.setTitleText("ay")
        self._chart_comb.addAxis(self._ax_comb_y, Qt.AlignmentFlag.AlignLeft)

        # Reposition combined-chart analysis labels when these axes change.
        self._ax_comb_x.rangeChanged.connect(self._reposition_comb_labels)
        self._ax_comb_y.rangeChanged.connect(self._reposition_comb_labels)

        # dt=0.2s reference (yellow)
        self._add_ref(self._chart_comb, self._ax_comb_x, self._ax_comb_y,
                       [-1.8, -1.62, -0.54, 0, 1.8, 5.4, 6],
                       [0, 0.6, 1.8, 2, 1.8, 0.6, 0],
                       "dt=0.2s", "#CCCC00", 3)
        self._add_ref(self._chart_comb, self._ax_comb_x, self._ax_comb_y,
                       [-1.8, -1.62, -0.54, 0, 1.8, 5.4, 6],
                       [0, -0.6, -1.8, -2, -1.8, -0.6, 0],
                       "", "#CCCC00", 3)

        # dt=0.1s reference (orange)
        self._add_ref(self._chart_comb, self._ax_comb_x, self._ax_comb_y,
                       [-1.9, -1.71, -0.57, 0, 1.8, 5.4, 6],
                       [0, 0.741, 2.22, 2.47, 2.22, 0.741, 0],
                       "dt=0.1s", "orange", 3)
        self._add_ref(self._chart_comb, self._ax_comb_x, self._ax_comb_y,
                       [-1.9, -1.71, -0.57, 0, 1.8, 5.4, 6],
                       [0, -0.741, -2.22, -2.47, -2.22, -0.741, 0],
                       "", "orange", 3)

        # dt=0.05s reference (red)
        self._add_ref(self._chart_comb, self._ax_comb_x, self._ax_comb_y,
                       [-1.95, -1.755, -0.585, 0, 1.8, 5.4, 6],
                       [0, 0.9, 2.7, 3, 2.7, 0.9, 0],
                       "dt=0.05s", "red", 3)
        self._add_ref(self._chart_comb, self._ax_comb_x, self._ax_comb_y,
                       [-1.95, -1.755, -0.585, 0, 1.8, 5.4, 6],
                       [0, -0.9, -2.7, -3, -2.7, -0.9, 0],
                       "", "red", 3)

        # Recreate each view's viewport so no stale pixels from the previous
        # (populated) chart linger on a hidden page — a reset must look exactly
        # like the app's initial state.
        for cv in (self._cv_ay, self._cv_az, self._cv_comb):
            reset_chart_view(cv)

    def restyle_theme(self) -> None:
        """Re-apply palette to all 3 sub-charts on a theme switch."""
        for chart in (self._chart_ay, self._chart_az, self._chart_comb):
            apply_chart_theme(chart)
        for legend in self._legends.values():
            legend.refresh_theme()

    def _add_ref(self, chart, ax_x, ax_y, x_vals, y_vals,
                 name, color, width) -> QLineSeries:
        """Add a reference curve series."""
        s = create_line_series(x_vals, y_vals, color=color,
                                name=name, width=float(width))
        chart.addSeries(s)
        s.attachAxis(ax_x)
        s.attachAxis(ax_y)
        return s

    # ------------------------------------------------------------------
    # Data plotting
    # ------------------------------------------------------------------

    def addplotxyz(self, data, color: str | None = None, sustained=None) -> None:
        """Overlay measured data on the GB charts.

        The measured (+) traces use the shared dark-blue, measured (−) the
        shared dark-green (see MEASURED_POS_COLOR / MEASURED_NEG_COLOR in
        theme), consistent with the ASTM charts. The ``color`` arg is ignored
        for the ± traces so +/− read the same regardless of which dataset is
        plotted; the combined scatter overlay still takes *color*.
        """
        # Remove previous measured data
        self._remove_data_series()

        measured_pos = MEASURED_POS_COLOR
        measured_neg = MEASURED_NEG_COLOR
        # The combined scatter (3rd chart) uses the bright-blue measured colour
        # so it reads consistently with the ± traces (per user request).
        comb_color = MEASURED_POS_COLOR

        # Process GB data at various dt values
        d0py = max(0, data.iloc[:, 2].max())
        d0ny = min(0, data.iloc[:, 2].min())
        d0pz = max(1, data.iloc[:, 3].max())
        d0nz = min(1, data.iloc[:, 3].min())

        fs = sampling_frequency(data)
        def calculate(duration):
            if sustained is not None:
                return sustained[duration]
            return ProcessGB(data, max(1, round(fs * duration)))
        d1p, d1n = calculate(0.01)
        d2p, d2n = calculate(0.05)
        d08p, d08n = calculate(0.08)
        d10p, d10n = calculate(0.1)
        d3p, d3n = calculate(0.2)
        d30p, d30n = calculate(0.3)
        d40p, d40n = calculate(0.4)
        d4p, d4n = calculate(0.5)
        d5p, d5n = calculate(1)
        d6p, d6n = calculate(2)
        d7p, d7n = calculate(4)

        p1 = [d0py, d1p[0], d2p[0], d08p[0], d10p[0], d3p[0],
              d30p[0], d40p[0], d4p[0], d5p[0], d6p[0], d7p[0]]
        p2 = [d0ny, d1n[0], d2n[0], d08n[0], d10n[0], d3n[0],
              d30n[0], d40n[0], d4n[0], d5n[0], d6n[0], d7n[0]]
        p3 = [d0pz, d1p[1], d2p[1], d08p[1], d10p[1], d3p[1],
              d30p[1], d40p[1], d4p[1], d5p[1], d6p[1], d7p[1]]
        p4 = [d0nz, d1n[1], d2n[1], d08n[1], d10n[1], d3n[1],
              d30n[1], d40n[1], d4n[1], d5n[1], d6n[1], d7n[1]]

        x_gb = [1.0 / fs, 0.01, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4,
                0.5, 1, 2, 4]

        # ay chart
        s = self._add_data(self._chart_ay, self._ax_ay_x, self._ax_ay_y,
                            x_gb, p1, "Measured ay (+)", measured_pos)
        s.clicked.connect(self._on_click)
        s = self._add_data(self._chart_ay, self._ax_ay_x, self._ax_ay_y,
                            x_gb, p2, "Measured ay (-)", measured_neg)
        s.clicked.connect(self._on_click)

        # az chart
        s = self._add_data(self._chart_az, self._ax_az_x, self._ax_az_y,
                            x_gb, p3, "Measured az (+)", measured_pos)
        s.clicked.connect(self._on_click)
        s = self._add_data(self._chart_az, self._ax_az_x, self._ax_az_y,
                            x_gb, p4, "Measured az (-)", measured_neg)
        s.clicked.connect(self._on_click)

        # Combined chart — scatter overlay
        z_arr = data.iloc[:, 3].values
        y_arr = data.iloc[:, 2].values
        s = create_line_series(z_arr, y_arr, color=comb_color,
                                name="Measured Data",
                                width=LINE_WIDTH_FITTING,
                                points_visible=True)
        self._chart_comb.addSeries(s)
        s.attachAxis(self._ax_comb_x)
        s.attachAxis(self._ax_comb_y)
        self._data_series.append(s)
        s.clicked.connect(lambda pt: self._on_click_comb(pt))

        # All series added → rebuild the wrapping legends.
        self._refresh_legends()
        # Cache the raw data so the combined-chart analyze toggle can run.
        self._last_comb_data = data
        # If analysis is currently shown, re-run it against the fresh data
        # (the user may have changed the plotted dataset).
        if self._comb_analysis_active:
            self.toggle_comb_analysis(True)

        # Combined chart: apply the equal X/Y scale so the default view renders
        # the GB contour as a true circle (1 data-unit = same pixel count on
        # both axes). Deferred to the next event loop so the chart has had its
        # layout pass and plotArea() has real dimensions.
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, self.equalize_comb)

    def _refresh_legends(self) -> None:
        """Rebuild every GB ChartLegend from its chart's current series."""
        for key, chart in self._charts_by_key.items():
            legend = self._legends.get(key)
            if legend is not None:
                legend.populate_from(chart)

    def refresh_legends(self) -> None:
        """Public alias for _refresh_legends."""
        self._refresh_legends()

    def _add_data(self, chart, ax_x, ax_y, x_vals, y_vals,
                  name, color) -> QLineSeries:
        s = create_line_series(x_vals, y_vals, color=color, name=name,
                                width=LINE_WIDTH_FITTING)
        chart.addSeries(s)
        s.attachAxis(ax_x)
        s.attachAxis(ax_y)
        self._data_series.append(s)
        return s

    def _remove_data_series(self) -> None:
        for s in self._data_series:
            try:
                s.chart().removeSeries(s)
            except Exception:
                pass
        self._data_series.clear()

    def recolor_measured(self, color: str) -> None:
        """Recolour the measured traces by sign, using shared colours.

        The (+) traces stay bright blue, (−) stay bright green (shared sign
        colours), and the combined-chart scatter ("Measured Data") also uses
        the bright blue — regardless of *color*, so all measured curves read
        consistently across datasets.
        """
        for s in self._data_series:
            name = (s.name() or "")
            pen = s.pen()
            if name.endswith("(-)"):
                pen.setColor(QColor(MEASURED_NEG_COLOR))
            else:
                # (+) traces AND the combined scatter → bright blue.
                pen.setColor(QColor(MEASURED_POS_COLOR))
            s.setPen(pen)
        self._refresh_legends()

    # ------------------------------------------------------------------
    # Combined-chart boundary violation analysis (Analyze toggle)
    # ------------------------------------------------------------------
    # Combined chart axes: x=az (col3), y=ay (col2). The dt reference curves
    # are closed loops named "dt=0.05s" / "dt=0.1s" / "dt=0.2s" with matching
    # sustained-duration thresholds.

    def _comb_boundary_series(self) -> list[tuple[str, float, list]]:
        """Collect the combined chart's dt boundary curves + thresholds.

        Each dt ring (0.05/0.1/0.2s) is drawn as TWO arc series — an upper
        half (named "dt=0.2s") and a lower half (unnamed mirror). They are NOT
        individually closed loops, so we return the list of series for each
        label and let the caller merge their points into one closed Path.
        """
        # threshold by dt tag
        def threshold_for(name: str):
            low = name.lower().replace(" ", "")
            if "0.05s" in low:
                return 0.05
            if "0.1s" in low:
                return 0.1
            if "0.2s" in low:
                return 0.2
            return None

        out: list[tuple[str, float, list]] = []
        series_list = list(self._chart_comb.series())
        used = set()
        for i, s in enumerate(series_list):
            name = (s.name() or "").strip()
            if not name or name == "Measured Data" or id(s) in used:
                continue
            th = threshold_for(name)
            if th is None:
                continue
            # This is the NAMED upper half. Find the unnamed mirror half that
            # follows it (same pen colour, no name) and pair them.
            group = [s]
            used.add(id(s))
            pen_color = s.pen().color().rgb()
            for s2 in series_list[i + 1:]:
                if id(s2) in used:
                    continue
                n2 = (s2.name() or "").strip()
                # Pair only the immediately-following unnamed arc of the same
                # colour — that's this ring's lower half.
                if not n2 and s2.pen().color().rgb() == pen_color:
                    group.append(s2)
                    used.add(id(s2))
                    break
            out.append((name, th, group))
        return out

    # Marker colour priority for the combined chart: a span's points take the
    # colour of the SMALLEST-threshold rule they break (0.05s red → 0.1s orange
    # → 0.2s yellow), matching the dt reference line colours.
    _COMB_RULE_COLOR = {
        "0.05": ("#FF0000", 3),   # red, highest priority
        "0.1":  ("orange", 2),
        "0.2":  ("#CCCC00", 1),   # yellow
    }

    @staticmethod
    def _comb_marker_for(rule_label: str):
        """Return (priority, color) for a dt rule label."""
        low = rule_label.lower().replace(" ", "")
        for tag, (color, prio) in GBChartView._COMB_RULE_COLOR.items():
            if tag + "s" in low:
                return (prio, QColor(color))
        return (0, QColor("#888888"))

    def toggle_comb_analysis(self, active: bool) -> None:
        """Show / hide combined-chart boundary violation markers + record rules.

        Markers are coloured by the smallest-threshold rule a span breaks
        (0.05s red wins over 0.1s over 0.2s); the failed-rule summary is
        exposed via ``self.failed_rules`` for the main-window toolbar.
        """
        self._comb_analysis_active = active
        self._clear_comb_analysis()
        self.failed_rules: set[str] = set()
        if not active or self._last_comb_data is None:
            return
        data = self._last_comb_data
        try:
            t_arr = np.asarray(data.iloc[:, 0].values, dtype=float)
            # combined chart: x=az (col3), y=ay (col2)
            measured_xy = np.column_stack([
                np.asarray(data.iloc[:, 3].values, dtype=float),
                np.asarray(data.iloc[:, 2].values, dtype=float)])
        except Exception:
            return
        boundaries = self._comb_boundary_series()
        # Aggregate violations per (i_start, i_end) → set of rule labels.
        span_rules: dict[tuple[int, int], set[str]] = {}
        for rule_label, threshold, series_group in boundaries:
            # Merge the ring's arc halves into one closed loop's points.
            verts = []
            for s in series_group:
                for p in s.points():
                    verts.append((float(p.x()), float(p.y())))
            if len(verts) < 3:
                continue
            path = Path(verts)
            viols = find_boundary_violations(
                measured_xy, t_arr, [path], threshold)
            for (i0, i1) in viols:
                span_rules.setdefault((i0, i1), set()).add(rule_label)
            if viols:
                self.failed_rules.add(rule_label)
        for (i0, i1), rules in span_rules.items():
            self._draw_comb_violation(measured_xy, i0, i1, rules)

    def _draw_comb_violation(self, measured_xy: np.ndarray,
                             i0: int, i1: int, rules: set[str]) -> None:
        """Draw enlarged markers on the violating span, coloured by priority."""
        pts = measured_xy[i0:i1 + 1]
        # Smallest-threshold rule wins → its colour.
        best = max(rules, key=lambda r: self._comb_marker_for(r)[0])
        _prio, color = self._comb_marker_for(best)
        marker = QScatterSeries()
        marker.setMarkerSize(14)
        marker.setColor(color)
        marker.setBorderColor(QColor("white"))
        for x, y in pts:
            marker.append(QPointF(float(x), float(y)))
        self._chart_comb.addSeries(marker)
        marker.attachAxis(self._ax_comb_x)
        marker.attachAxis(self._ax_comb_y)
        self._comb_analysis_refs.append(marker)

    def _clear_comb_analysis(self) -> None:
        """Remove all combined-chart analysis scatter series (markers only)."""
        for ref in self._comb_analysis_refs:
            try:
                self._chart_comb.removeSeries(ref)
            except Exception:
                pass
        self._comb_analysis_refs = []
        self.failed_rules = set()

    def _reposition_comb_labels(self, *_args) -> None:
        """No per-span labels anymore (single fixed summary label). Kept as a
        no-op hook so the axis rangeChanged connections stay valid."""
        pass

    # ------------------------------------------------------------------
    # Compatibility
    # ------------------------------------------------------------------

    def draw_idle(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Annotations
    # ------------------------------------------------------------------

    def _on_click(self, point: QPointF) -> None:
        from PySide6.QtWidgets import QLabel
        sender = self.sender()
        chart = sender.chart() if hasattr(sender, 'chart') else None
        if chart is None:
            return
        pos = chart.mapToPosition(point)
        text = f"dt = {point.x()}\namp = {point.y():.4f}"
        # Find the chart view
        for cv in (self._cv_ay, self._cv_az, self._cv_comb):
            if cv.chart() == chart:
                label = QLabel(text, cv)
                label.setStyleSheet(chart_tooltip_style())
                label.adjustSize()
                label.move(int(pos.x()) + 8, max(int(pos.y()) - label.height() - 4, 2))
                label.show()
                label.raise_()
                from PySide6.QtCore import QTimer
                QTimer.singleShot(5000, label.deleteLater)
                break

    def _on_click_comb(self, point: QPointF) -> None:
        chart = self._cv_comb.chart()
        pos = chart.mapToPosition(point)
        text = f"az = {point.x():.4f}\nay = {point.y():.4f}"
        label = QLabel(text, self._cv_comb)
        label.setStyleSheet(chart_tooltip_style())
        label.adjustSize()
        label.move(int(pos.x()) + 8, max(int(pos.y()) - label.height() - 4, 2))
        label.show()
        label.raise_()
        from PySide6.QtCore import QTimer
        QTimer.singleShot(5000, label.deleteLater)

    # ------------------------------------------------------------------
    # Zoom control methods
    # ------------------------------------------------------------------

    def zoom_mode(self, mode: str) -> None:
        """Set interaction mode for all sub-charts.

        ``"zoom"`` → rubber-band zoom; ``"pan"`` → manual drag scroll;
        anything else → default (no drag). Pan is handled manually by
        ``PanChartView`` because ``ScrollHandDrag`` is unreliable on a
        ``QChartView``.
        """
        for cv in (self._cv_ay, self._cv_az, self._cv_comb):
            if mode == "zoom":
                cv._interaction_mode = INTERACTION_ZOOM
                cv.setRubberBand(QChartView.RubberBand.RectangleRubberBand)
                cv.setDragMode(QChartView.DragMode.NoDrag)
                cv.viewport().unsetCursor()
            elif mode == "pan":
                cv._interaction_mode = INTERACTION_PAN
                cv.setRubberBand(QChartView.RubberBand.NoRubberBand)
                cv.setDragMode(QChartView.DragMode.NoDrag)
                cv.viewport().setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
            else:
                cv._interaction_mode = INTERACTION_DEFAULT
                cv._pan_cancel()
                cv.setRubberBand(QChartView.RubberBand.NoRubberBand)
                cv.setDragMode(QChartView.DragMode.NoDrag)
                cv.viewport().unsetCursor()

    def reset_view(self) -> None:
        """Restore all 3 charts to their standard (home) ranges.

        Bug fixed: ``chart.zoomReset()`` only undoes rubber-band zooms; on a
        fresh view it was a no-op. Restoring the stored home ranges always
        works.

        The combined (3rd) chart re-equalizes its X/Y scale here so the GB
        contour reads as a true circle after Reset View (1 data-unit = same
        pixel count on both axes). Zoom/pan is left free between resets.
        """
        for axis, (lo, hi) in (
            (self._ax_ay_x, (0, 4)), (self._ax_ay_y, (-5, 5)),
            (self._ax_az_x, (0, 4)), (self._ax_az_y, (-2, 6)),
        ):
            axis.setRange(lo, hi)
        # Combined (3rd) chart: az (X) locked to its default; ay (Y) fixed at
        # ±3.5 (per request — NOT pixel-derived, so it stays a constant ±3.5).
        self._ax_comb_x.setRange(-2.0, 6.0)
        self._ax_comb_y.setRange(-3.5, 3.5)

    def equalize_comb(self) -> None:
        """Apply the combined (3rd) chart's standard ranges.

        Call after the chart is laid out (e.g. once a measured trace is
        plotted). az (X) = (-2, 6); ay (Y) = ±3.5 (fixed, per request).
        """
        self._ax_comb_x.setRange(-2.0, 6.0)
        self._ax_comb_y.setRange(-3.5, 3.5)
