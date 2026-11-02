# coding: utf-8
"""
ASTM F2291 fitting chart — six sub-charts in a 2×3 grid:
  Row 0: ax vs dt,  ay vs dt,  az vs dt
  Row 1: egg XY,    egg XZ,    egg YZ

Replaces ASTMPlot (matplotlib).  This is the largest chart module.
"""

from __future__ import annotations

import numpy as np

from PySide6.QtCharts import (
    QChart, QChartView, QLineSeries, QValueAxis, QScatterSeries,
)
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPainter, QPen, QColor, QFont, QCursor
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
from chart_legend import ChartLegend
from boundary_analysis import series_to_path, find_boundary_violations
from sci_calculation import (
    ProcessASTM, sampling_frequency, eggXY, eggXZ, eggYZ, eggXY_unclamped, eggXZ_unclamped, coef)
from theme import (LINE_WIDTH_DEFAULT, LINE_WIDTH_FITTING, chart_tooltip_style,
                   apply_chart_theme, theme_manager, MEASURED_POS_COLOR, MEASURED_NEG_COLOR)


class ASTMChartView(QWidget):
    """Chart widget for ASTM F2291 acceleration analysis.

    By default builds all six charts (ax/ay/az + xy/xz/yz). Pass a
    ``chart_names`` subset to build only some of them — used to split the
    analysis across two tabs ("Sustained acceleration" vs "Egg plots").
    """

    # All available chart keys and their axis titles/ranges.
    _CONFIGS = {
        "ax": ("dt (s)", "ax",  (0, 14), (-3, 7)),
        "ay": ("dt (s)", "ay",  (0, 14), (0, 3.2)),
        "az": ("dt (s)", "az",  (0, 14), (-3, 7)),
        "xy": ("Front <=> Back", "Left <=> Right", (-2.1, 6.1), (-3.2, 3.2)),
        "xz": ("Front <=> Back", "Up <=> Down",   (-2.1, 6.1), (-2.2, 6.2)),
        "yz": ("Left <=> Right", "Up <=> Down",   (-3.1, 3.1), (-2.2, 6.2)),
    }

    # Egg charts rendered with EQUAL X/Y scale (1 data-unit = same pixel count
    # on both axes) so the egg reads as a true circle at default / reset. The
    # X range is locked to its default; the Y range is derived from the plot
    # area's pixel aspect, centred on ``y_center``. Zoom/pan are NOT forced
    # equal afterwards (only the default + Reset View re-equalize).
    _EQUAL_EGGS = {
        # name: (x_range, y_center)
        "xy": ((-2.1, 6.1), 0.0),
        "xz": ((-2.1, 6.1), 2.0),
        "yz": ((-3.1, 3.1), 2.0),
    }

    def __init__(self, parent=None, chart_names=None):
        super().__init__(parent)
        self.annot_mode: str = "Single"

        # Which charts this instance owns (defaults to all six).
        if chart_names is None:
            chart_names = list(self._CONFIGS.keys())
        self._chart_names = list(chart_names)

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self._charts: dict[str, QChart] = {}
        self._views: dict[str, QChartView] = {}
        self._axes: dict[str, dict[str, QValueAxis]] = {}
        # Per-chart wrapping legend widgets (replaces the built-in QLegend,
        # which lays markers out in a single clipped row). Built in __init__,
        # populated from the chart's legend markers after each plot.
        self._legends: dict[str, ChartLegend] = {}

        # Lay the active charts out left-to-right in a single row. Each chart
        # view is wrapped in a container that stacks a wrapping ChartLegend on
        # top + the view below, so the legend can take as many rows as it needs
        # instead of being clipped to one.
        for col, name in enumerate(self._chart_names):
            container = QWidget(self)
            cl = QVBoxLayout(container)
            cl.setContentsMargins(0, 0, 0, 0)
            cl.setSpacing(0)
            legend = ChartLegend(container)
            self._legends[name] = legend
            cl.addWidget(legend)
            cv = PanChartView(container)
            cv.setRenderHint(QPainter.RenderHint.Antialiasing)
            cv.setSizePolicy(QSizePolicy.Policy.Expanding,
                              QSizePolicy.Policy.Expanding)
            self._views[name] = cv
            cl.addWidget(cv, 1)
            layout.addWidget(container, 0, col)

        # Data series references
        self._data_series: list[QLineSeries] = []
        self._ref_series: list[QLineSeries] = []
        # Home axis ranges per chart, recorded in initplotxyz for reset_view.
        self._home_ranges: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {}
        # Egg-boundary analysis overlays: scatter markers (violation points),
        # created by toggle_egg_analysis. Kept here so they can be removed and
        # so they aren't GC'd.
        self._egg_analysis_refs: list = []
        # id(scatter) → chart_name, so _clear_egg_analysis removes each scatter
        # from the chart that actually owns it (removeSeries is chart-specific).
        self._egg_scatter_charts: dict = {}
        # Whether egg analysis overlays are currently shown.
        self._egg_analysis_active: bool = False
        # Last plotted raw DataFrame, cached for the analyze toggle.
        self._last_egg_data = None
        # Whether prone posture was last plotted (cached for the analyze toggle).
        self._last_prone: bool = False
        # Set of rule labels that were violated (read by main_window to show
        # the summary in the chart-tools toolbar).
        self.failed_rules: set[str] = set()
        # Sustained-acceleration analysis (ax/ay/az vs dt): scatter markers +
        # value labels on measured sustained points that exceed the ASTM limit
        # at their dt (dt in [0.2s, 14s] only; 0-0.2s has no requirement).
        self._sust_analysis_refs: list = []
        self._sust_analysis_active: bool = False
        # Sustained-axis rule labels that failed ("ax (+)", "az (−)", …), shown
        # in the Sustained toolbar summary by main_window.
        self.sust_failed_rules: set[str] = set()

        self.initplotxyz()

    # ------------------------------------------------------------------
    # Chart initialisation
    # ------------------------------------------------------------------

    def initplotxyz(self) -> None:
        self._data_series.clear()
        self._ref_series.clear()
        # Drop any egg-analysis overlays + cached data (chart rebuilt).
        self._clear_egg_analysis()
        self._egg_analysis_active = False
        self._last_egg_data = None
        # Drop any sustained-acceleration analysis overlays too.
        self._clear_sust_analysis()
        self._sust_analysis_active = False

        for name in self._chart_names:
            x_title, y_title, x_range, y_range = self._CONFIGS[name]
            chart = QChart()
            # Hide the built-in legend: it lays markers in ONE row and clips
            # overflow, truncating the many ASTM entries. A wrapping
            # ChartLegend widget above each view shows them all (see __init__).
            chart.legend().setVisible(False)
            chart.layout().setContentsMargins(4, 4, 4, 4)

            ax_x = QValueAxis()
            ax_x.setTitleText(x_title)
            ax_x.setRange(x_range[0], x_range[1])
            chart.addAxis(ax_x, Qt.AlignmentFlag.AlignBottom)

            ax_y = QValueAxis()
            ax_y.setTitleText(y_title)
            ax_y.setRange(y_range[0], y_range[1])
            chart.addAxis(ax_y, Qt.AlignmentFlag.AlignLeft)

            self._charts[name] = chart
            self._axes[name] = {"x": ax_x, "y": ax_y}
            self._home_ranges[name] = (tuple(x_range), tuple(y_range))
            self._views[name].setChart(chart)
            apply_chart_theme(chart)
            # Reposition egg-analysis labels when this chart's axes change.
            ax_x.rangeChanged.connect(self._reposition_egg_labels)
            ax_y.rangeChanged.connect(self._reposition_egg_labels)

        # Recreate each view's viewport so no stale pixels from the previous
        # (populated) chart linger on a hidden page — a reset must look exactly
        # like the app's initial state.
        for cv in self._views.values():
            reset_chart_view(cv)

    def restyle_theme(self) -> None:
        """Re-apply palette to all 6 sub-charts on a theme switch."""
        for chart in self._charts.values():
            apply_chart_theme(chart)
        for series in self._ref_series:
            if series.property("themeReference"):
                pen = series.pen()
                pen.setColor(QColor(theme_manager.tokens['chart_axis']))
                series.setPen(pen)
        # Refresh the wrapping legends' label text colour for the new theme.
        for legend in self._legends.values():
            legend.refresh_theme()

    # ------------------------------------------------------------------
    # Data plotting
    # ------------------------------------------------------------------

    def addplotxyz(self, data, restraint: str, cond: str,
                   input_height: int, color: str | None = None,
                   prone: bool = False, sustained=None) -> None:
        """Plot measured data and ASTM reference curves.

        ``color`` (hex) drives the measured (+) traces and the egg overlays.
        If None, falls back to legacy blue/cyan. The measured (−) traces use
        a contrasting dark tone so + / − stay distinguishable.

        ``prone`` (prone posture) swaps the XY/XZ base eggs to the wider prone
        duck-egg shape and overrides the sustained ax− limit (both the ASTM
        base line and the restraint line), regardless of restraint type.
        """
        self._remove_data_series()

        # Shared sign colours: measured (+) dark blue, measured (−) dark green,
        # consistent with the GB charts (see MEASURED_POS/NEG_COLOR in theme).
        # The dataset's own ``color`` is ignored for the ± traces so +/− read
        # the same across datasets; the egg overlays keep the dataset colour.
        measured_pos = MEASURED_POS_COLOR
        measured_neg = MEASURED_NEG_COLOR
        # Egg plots use the bright-blue measured colour so the overlay reads
        # consistently with the ± traces (per user request).
        egg_color = MEASURED_POS_COLOR

        # The split Eggplot canvas has no sustained charts. Avoid seventeen
        # rolling-window passes on data it never displays.
        if any(name in self._charts for name in ('ax', 'ay', 'az')):
            fs = sampling_frequency(data)
            def calculate(duration):
                if sustained is not None:
                    return sustained[duration]
                return ProcessASTM(data, max(1, round(fs * duration)))
            # Process data at various dt values
            d0p, d0n = calculate(0.01)
            d1p, d1n = calculate(0.02)
            d2p, d2n = calculate(0.05)
            d3p, d3n = calculate(0.1)
            d7p, d7n = calculate(0.2)
            d03p, d03n = calculate(0.3)
            d04p, d04n = calculate(0.4)
            d4p, d4n = calculate(0.5)
            d5p, d5n = calculate(1)
            d9p, d9n = calculate(2)
            d6p, d6n = calculate(4)
            d10p, d10n = calculate(5)
            d77p, d77n = calculate(6)
            d99p, d99n = calculate(7)
            d88p, d88n = calculate(8)
            d111p, d111n = calculate(9)
            d8p, d8n = calculate(14)

            x_axis = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                      1, 2, 4, 5, 6, 7, 8, 9, 14]

            # ax chart
            self._add_data_line("ax", x_axis,
                [d0p[0], d1p[0], d2p[0], d3p[0], d7p[0], d03p[0], d04p[0],
                 d4p[0], d5p[0], d9p[0], d6p[0], d10p[0], d77p[0], d99p[0],
                 d88p[0], d111p[0], d8p[0]],
                "Measured ax (+)", measured_pos)
            self._add_data_line("ax", x_axis,
                [d0n[0], d1n[0], d2n[0], d3n[0], d7n[0], d03n[0], d04n[0],
                 d4n[0], d5n[0], d9n[0], d6n[0], d10n[0], d77n[0], d99n[0],
                 d88n[0], d111n[0], d8n[0]],
                "Measured ax (-)", measured_neg)

            # ay chart
            self._add_data_line("ay", x_axis,
                [d0p[1], d1p[1], d2p[1], d3p[1], d7p[1], d03p[1], d04p[1],
                 d4p[1], d5p[1], d9p[1], d6p[1], d10p[1], d77p[1], d99p[1],
                 d88p[1], d111p[1], d8p[1]],
                "Measured ay (Right)", measured_pos)
            self._add_data_line("ay", x_axis,
                [-d0n[1], -d1n[1], -d2n[1], -d3n[1], -d7n[1], -d03n[1],
                 -d04n[1], -d4n[1], -d5n[1], -d9n[1], -d6n[1], -d10n[1],
                 -d77n[1], -d99n[1], -d88n[1], -d111n[1], -d8n[1]],
                "Measured ay (Left)", measured_neg)

            # az chart
            self._add_data_line("az", x_axis,
                [d0p[2], d1p[2], d2p[2], d3p[2], d7p[2], d03p[2], d04p[2],
                 d4p[2], d5p[2], d9p[2], d6p[2], d10p[2], d77p[2], d99p[2],
                 d88p[2], d111p[2], d8p[2]],
                "Measured az (Down)", measured_pos)
            self._add_data_line("az", x_axis,
                [d0n[2], d1n[2], d2n[2], d3n[2], d7n[2], d03n[2], d04n[2],
                 d4n[2], d5n[2], d9n[2], d6n[2], d10n[2], d77n[2], d99n[2],
                 d88n[2], d111n[2], d8n[2]],
                "Measured az (Up)", measured_neg)

        # Egg plots — measured data
        x_data = data.iloc[:, 1].values
        y_data = data.iloc[:, 2].values
        z_data = data.iloc[:, 3].values

        self._add_data_line("xy", x_data, y_data, "Measured", egg_color, True)
        self._add_data_line("xz", x_data, z_data, "Measured", egg_color, True)
        self._add_data_line("yz", y_data, z_data, "Measured", egg_color, True)

        # Add reference curves
        try:
            self._add_reference_curves(restraint, cond, input_height, prone)
        except Exception as e:
            print(f"ASTM reference error: {e}")

        # Cache the raw data + current restraint/cond/height so the analysis
        # toggles can run later without the caller having to pass them again.
        self._last_egg_data = data
        self._last_restraint = restraint
        self._last_cond = cond
        self._last_height = input_height
        self._last_prone = prone
        # If analysis is currently shown, re-run it against the freshly plotted
        # data + reference curves (restraint/condition/height may have changed).
        if self._egg_analysis_active:
            self.toggle_egg_analysis(True)
        if self._sust_analysis_active:
            self.toggle_sust_analysis(True, restraint, cond, input_height, prone)

        # Now that all series (measured + reference + restraint) are added,
        # populate each chart's wrapping legend from its series pens/labels.
        self._refresh_legends()

        # Egg charts: apply the equal X/Y scale so the default view renders a
        # circular egg as a true circle (1 data-unit = same pixel count on both
        # axes). Deferred to the next event loop so the chart has had its
        # layout pass and plotArea() has real dimensions.
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, self.equalize_eggs)

    def _refresh_legends(self) -> None:
        """Rebuild every ChartLegend from its chart's current series.

        Call after addplotxyz / recolor_measured so the legend matches what's
        drawn (right colours, no stale entries). Skips empty-label series.
        """
        for name, chart in self._charts.items():
            legend = self._legends.get(name)
            if legend is not None:
                legend.populate_from(chart)

    def refresh_legends(self) -> None:
        """Public alias for _refresh_legends (used after recolour, etc.)."""
        self._refresh_legends()

    # ------------------------------------------------------------------
    # Reference curves (ASTM + restraint types)
    # ------------------------------------------------------------------

    def _add_reference_curves(self, restraint: str, cond: str,
                               height: int, prone: bool = False) -> None:
        """Add ASTM F2291 standard and restraint-type curves.

        ``prone`` swaps the XY/XZ base eggs to the wider prone duck-egg shape
        (scaled by height) and overrides the ASTM base ax− limit with the
        prone ax− curve. Restraint-type eggs and the YZ egg are unaffected.
        """
        frontASTM = np.array([-2, -2, -1.5, -1.5])
        backASTM = np.array([6, 6, 6, 4, 4, 3, 3, 2.5, 2.5])
        lrASTM = np.array([3, 3, 3, 2, 2])
        upASTM = np.array([-2, -2, -1.5, -1.5, -1.2, -1.2])
        downASTM = np.array([6, 6, 6, 4, 4, 3, 3, 2, 2])

        ht = float(height)
        cond_mult = ASTMChartView._cond_mult(cond)

        coefX = cond_mult * coef(ht, 'x')
        coefZ = cond_mult * coef(ht, 'z')

        # Egg curves (0.2s envelope). Prone posture uses the wider prone shape
        # for XY/XZ (left extent 3.5·coef, exceeding the standard 2g clamp); the
        # standard egg generators clamp the left radius to 2, so the unclamped
        # variants are required. YZ is unchanged in prone posture.
        if prone:
            x3, y3 = eggXY_unclamped(3.5 * coef(ht, 'x'),
                                     6 * coef(ht, 'x'),
                                     3 * coef(ht, 'x'))
            x4, y4 = eggXZ_unclamped(3.5 * coef(ht, 'z'),
                                     6 * coef(ht, 'z'),
                                     2 * coef(ht, 'z'),
                                     6 * coef(ht, 'z'))
        else:
            x3, y3 = eggXY(2 * coef(ht, 'x'), 6 * coef(ht, 'x'), 3 * coef(ht, 'x'))
            x4, y4 = eggXZ(2 * coef(ht, 'z'), 6 * coef(ht, 'z'),
                             2 * coef(ht, 'z'), 6 * coef(ht, 'z'))
        x5, y5 = eggYZ(3 * coef(ht, 'z'), 2 * coef(ht, 'z'),
                         6 * coef(ht, 'z'))

        # --- ASTM base limits (red, thick) ---
        pen = QPen(QColor("red"), 3)

        # ax limits
        if prone:
            self._add_ref_line("ax", *self._PRONE_ASTM_AXNEG, "ASTM ax", pen)
        else:
            self._add_ref_line("ax", [0.2, 0.5, 14], [-2, -1.5, -1.5],
                                "ASTM ax", pen)
        self._add_ref_line("ax",
            [0.2, 1, 2, 4, 5, 11.8, 12, 14],
            (np.array([6, 6, 4, 4, 3, 3, 2.5, 2.5]) * coef(ht, 'x')).tolist(),
            "", pen)

        # ay limits
        self._add_ref_line("ay",
            [0.2, 1, 2, 14],
            (np.array([3, 3, 2, 2]) * coef(ht, 'x')).tolist(),
            "ASTM ay", pen)

        # az limits
        self._add_ref_line("az",
            [0.2, 0.5, 4, 7, 14],
            (np.array([-2, -1.5, -1.5, -1.2, -1.2]) * coef(ht, 'x')).tolist(),
            "ASTM az", pen)
        self._add_ref_line("az",
            [0.2, 1, 2, 4, 5, 11.8, 12, 14],
            (np.array([6, 6, 4, 4, 3, 3, 2, 2]) * coef(ht, 'x')).tolist(),
            "", pen)

        # Egg reference outlines. The horizontal crosshair extends to the left
        # edge of the base egg so it spans the full shape — for prone posture
        # that's 3.5·coef (wider than the standard 2g).
        xneg = (3.5 * coef(ht, 'x')) if prone else 2
        self._add_ref_line("xy", x3, y3, "ASTM (0.2s)",
                            QPen(QColor("red"), 2, Qt.PenStyle.DashLine))
        self._add_crosshairs("xy", -xneg, 6, 3, -3)

        self._add_ref_line("xz", x4, y4, "ASTM (0.2s)",
                            QPen(QColor("red"), 2, Qt.PenStyle.DashLine))
        self._add_crosshairs("xz", -xneg, 6, 6, -2)

        self._add_ref_line("yz", x5, y5, "ASTM (0.2s)",
                            QPen(QColor("red"), 2, Qt.PenStyle.DashLine))
        self._add_crosshairs("yz", -3, 3, 6, -2)

        # --- Restraint-type specific curves ---
        self._add_restraint_curves(restraint, cond_mult, ht, coefX, coefZ,
                                   prone=prone)

    def _add_restraint_curves(self, restraint: str, cond: float,
                               ht: float, coefX: float,
                               coefZ: float, prone: bool = False) -> None:
        """Add restraint-type-specific egg curves and limit lines.

        The ax/ay/az limit values (a31..a35) come from the shared
        _restraint_arrays helper — the SAME source the Analyze check
        (_restraint_limit_axes) uses — so the drawn red lines and the limit
        check can never drift apart.

        ``restraint`` 'None' draws NO restraint limit lines: the Analyze check
        falls back to the ASTM base limit (_astm_limit_axes), so a "None" copy
        of the restraint lines here would show limits the check never uses.

        ``prone`` overrides ONLY the ax− restraint line (with
        _PRONE_RESTR_AXNEG), regardless of restraint type; every other limit
        and the restraint eggs are unchanged.
        """
        pen_solid = QPen(QColor("black"), 2, Qt.PenStyle.SolidLine)
        pen_dash = QPen(QColor("black"), 2, Qt.PenStyle.DashLine)

        if restraint == 'Upper Body':
            xa, ya = eggXY(2 * coefX, 3.6 * coefX, 3 * coefX)
            xaa, yaa = eggXY(1.6 * coefX, 3.6 * coefX, 2.4 * coefX)
            xb, yb = eggXZ(2 * coefZ, 3.6 * coefZ, 2 * coefZ, 5 * coefZ)
            xbb, ybb = eggXZ(1.6 * coefZ, 3.6 * coefZ, 1.4 * coefZ, 4.8 * coefZ)
            xc, yc = eggYZ(3 * coefZ, 2 * coefZ, 5 * coefZ)
            xcc, ycc = eggYZ(2.4 * coefZ, 1.4 * coefZ, 4.8 * coefZ)

            self._add_ref_line("xy", xa, ya, f"{restraint} (0s)", pen_solid)
            self._add_ref_line("xy", xaa, yaa, f"{restraint} (0.2s)", pen_dash)
            self._add_ref_line("xz", xb, yb, f"{restraint} (0s)", pen_solid)
            self._add_ref_line("xz", xbb, ybb, f"{restraint} (0.2s)", pen_dash)
            self._add_ref_line("yz", xc, yc, f"{restraint} (0s)", pen_solid)
            self._add_ref_line("yz", xcc, ycc, f"{restraint} (0.2s)", pen_dash)

        elif restraint == 'Group Lower Body':
            xa, ya = eggXY(1.7 * coefX, 2.5 * coefX, 2.4 * coefX)
            xaa, yaa = eggXY(1.4 * coefX, 2.5 * coefX, 2.1 * coefX)
            xb, yb = eggXZ(1.7 * coefZ, 2.5 * coefZ, 2 * coefZ, 3.5 * coefZ)
            xbb, ybb = eggXZ(1.4 * coefZ, 2.5 * coefZ, 1 * coefZ, 3 * coefZ)
            xc, yc = eggYZ(2.4 * coefZ, 2 * coefZ, 3.5 * coefZ)
            xcc, ycc = eggYZ(2.1 * coefZ, 1 * coefZ, 3 * coefZ)

            self._add_ref_line("xy", xa, ya, f"{restraint} (0s)", pen_solid)
            self._add_ref_line("xy", xaa, yaa, f"{restraint} (0.2s)", pen_dash)
            self._add_ref_line("xz", xb, 1 + yb, f"{restraint} (0s)", pen_solid)
            self._add_ref_line("xz", xbb, 1 + ybb, f"{restraint} (0.2s)", pen_dash)
            self._add_ref_line("yz", xc, 1 + yc, f"{restraint} (0s)", pen_solid)
            self._add_ref_line("yz", xcc, 1 + ycc, f"{restraint} (0.2s)", pen_dash)

        elif restraint == 'Individual Lower Body':
            xa, ya = eggXY(1.8 * coefX, 2.5 * coefX, 2.6 * coefX)
            xaa, yaa = eggXY(1.5 * coefX, 2.5 * coefX, 2.2 * coefX)
            xb, yb = eggXZ(1.8 * coefZ, 2.5 * coefZ, 1.8 * coefZ, 4.8 * coefZ)
            xbb, ybb = eggXZ(1.5 * coefZ, 2.5 * coefZ, 1.2 * coefZ, 4.5 * coefZ)
            xc, yc = eggYZ(2.6 * coefZ, 1.8 * coefZ, 4.8 * coefZ)
            xcc, ycc = eggYZ(2.2 * coefZ, 1.2 * coefZ, 4.5 * coefZ)

            self._add_ref_line("xy", xa, ya, f"{restraint} (0s)", pen_solid)
            self._add_ref_line("xy", xaa, yaa, f"{restraint} (0.2s)", pen_dash)
            self._add_ref_line("xz", xb, yb, f"{restraint} (0s)", pen_solid)
            self._add_ref_line("xz", xbb, ybb, f"{restraint} (0.2s)", pen_dash)
            self._add_ref_line("yz", xc, yc, f"{restraint} (0s)", pen_solid)
            self._add_ref_line("yz", xcc, ycc, f"{restraint} (0.2s)", pen_dash)

        elif restraint in ('No Restraint', 'Convenience Restraint'):
            xa, ya = eggXY(1.5 * coefX, 2.5 * coefX, 1.8 * coefX)
            xaa, yaa = eggXY(1.2 * coefX, 2.5 * coefX, 1.2 * coefX)
            xb, yb = eggXZ(1.5 * coefZ, 2.5 * coefZ, 1.2 * coefZ, 3 * coefZ)
            xbb, ybb = eggXZ(1.2 * coefZ, 2.5 * coefZ, 0.8 * coefZ, 2.8 * coefZ)
            xc, yc = eggYZ(1.8 * coefZ, 1.2 * coefZ, 3 * coefZ)
            xcc, ycc = eggYZ(1.2 * coefZ, 0.8 * coefZ, 2.8 * coefZ)

            self._add_ref_line("xy", xa, ya, f"{restraint} (0s)", pen_solid)
            self._add_ref_line("xy", xaa, yaa, f"{restraint} (0.2s)", pen_dash)
            self._add_ref_line("xz", xb, 1 + yb, f"{restraint} (0s)", pen_solid)
            self._add_ref_line("xz", xbb, 1 + ybb, f"{restraint} (0.2s)", pen_dash)
            self._add_ref_line("yz", xc, 1 + yc, f"{restraint} (0s)", pen_solid)
            self._add_ref_line("yz", xcc, 1 + ycc, f"{restraint} (0.2s)", pen_dash)
        # 'None' (else): no egg curves and no restraint limit lines — the
        # Analyze check falls back to _astm_limit_axes (the base ASTM lines
        # already drawn by _add_reference_curves), so drawing a "None" copy
        # here would show limits the check never uses.

        # Restraint-specific limit lines on ax/ay/az, via the shared helper
        # (single source of truth with the Analyze check). In prone posture the
        # ax− restraint line is overridden by the prone curve regardless of the
        # selected restraint type (see _PRONE_RESTR_AXNEG).
        if restraint != 'None':
            a31, a32, a33, a34, a35 = self._restraint_arrays(
                restraint, coefX, coefZ)
            if prone:
                self._add_ref_line("ax", *self._PRONE_RESTR_AXNEG,
                                    f"{restraint}", QPen(QColor("red"), 2))
            else:
                self._add_ref_line("ax", [0, 0.2, 0.5, 14], a31.tolist(),
                                    f"{restraint}", QPen(QColor("red"), 2))
            self._add_ref_line("ax",
                [0, 0.2, 1, 2, 4, 5, 11.8, 12, 14], a32.tolist(),
                "", QPen(QColor("red"), 2))
            self._add_ref_line("ay",
                [0, 0.2, 1, 2, 14], a33.tolist(),
                f"{restraint}", QPen(QColor("red"), 2))
            self._add_ref_line("az",
                [0, 0.2, 0.5, 4, 7, 14], a34.tolist(),
                "", QPen(QColor("red"), 2))
            self._add_ref_line("az",
                [0, 0.2, 1, 2, 4, 5, 11.8, 12, 14], a35.tolist(),
                f"{restraint}", QPen(QColor("red"), 2))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _add_data_line(self, chart_name: str, x_data, y_data,
                        name: str, color: str,
                        points: bool = False) -> QLineSeries | None:
        # No-op if this instance doesn't own the requested chart (split tabs).
        if chart_name not in self._charts:
            return None
        chart = self._charts[chart_name]
        axes = self._axes[chart_name]
        s = create_line_series(x_data, y_data, color=color, name=name,
                                width=LINE_WIDTH_FITTING,
                                points_visible=points)
        chart.addSeries(s)
        s.attachAxis(axes["x"])
        s.attachAxis(axes["y"])
        self._data_series.append(s)
        s.clicked.connect(lambda pt: self._on_click(pt, chart_name))
        return s

    def _add_ref_line(self, chart_name: str, x_data, y_data,
                       name: str, pen: QPen) -> QLineSeries | None:
        if chart_name not in self._charts:
            return None
        chart = self._charts[chart_name]
        axes = self._axes[chart_name]
        s = QLineSeries()
        s.setName(name)
        if pen.color() == QColor("black"):
            pen = QPen(pen)
            pen.setColor(QColor(theme_manager.tokens['chart_axis']))
            s.setProperty("themeReference", True)
        s.setPen(pen)

        # Convert data
        if hasattr(x_data, '__iter__'):
            for x, y in zip(x_data, y_data):
                s.append(QPointF(float(x), float(y)))

        chart.addSeries(s)
        s.attachAxis(axes["x"])
        s.attachAxis(axes["y"])
        self._ref_series.append(s)
        s.clicked.connect(lambda pt: self._on_click(pt, chart_name))
        return s

    def _add_crosshairs(self, chart_name: str,
                         x1: float, x2: float,
                         y1: float, y2: float) -> None:
        """Add horizontal and vertical reference lines."""
        if chart_name not in self._charts:
            return
        axes = self._axes[chart_name]
        chart = self._charts[chart_name]
        pen = QPen(QColor(theme_manager.tokens['chart_axis']), 1.2)

        h = QLineSeries()
        h.setProperty("themeReference", True)
        h.append(QPointF(x1, 0))
        h.append(QPointF(x2, 0))
        h.setPen(pen)
        chart.addSeries(h)
        h.attachAxis(axes["x"])
        h.attachAxis(axes["y"])
        self._ref_series.append(h)

        v = QLineSeries()
        v.setProperty("themeReference", True)
        v.append(QPointF(0, y1))
        v.append(QPointF(0, y2))
        v.setPen(pen)
        chart.addSeries(v)
        v.attachAxis(axes["x"])
        v.attachAxis(axes["y"])
        self._ref_series.append(v)

    def _remove_data_series(self) -> None:
        for s in self._data_series:
            try:
                s.chart().removeSeries(s)
            except Exception:
                pass
        self._data_series.clear()

        for s in self._ref_series:
            try:
                s.chart().removeSeries(s)
            except Exception:
                pass
        self._ref_series.clear()

    def recolor_measured(self, color: str) -> None:
        """Recolour the measured series by sign, using shared colours.

        The (−/Left/Up) traces become the shared bright green, the
        (+/Right/Down) traces and egg overlays ("Measured") the shared bright
        blue — regardless of *color*, so all measured curves read consistently
        across datasets. The user-chosen dataset colour still applies to the
        Trend plot and the sidebar swatch; this sign-coded colour scheme is an
        intentional ASTM convention. Reference curves (``_ref_series``) are
        untouched.
        """
        neg_markers = ("(-)", "(left)", "(up)")
        for s in self._data_series:
            name = (s.name() or "")
            pen = s.pen()
            if name.lower().endswith(neg_markers):
                pen.setColor(QColor(MEASURED_NEG_COLOR))
            else:
                # (+) traces AND egg overlays → bright blue.
                pen.setColor(QColor(MEASURED_POS_COLOR))
            s.setPen(pen)
        # Swatch colours changed → rebuild the wrapping legends.
        self._refresh_legends()

    # ------------------------------------------------------------------
    # Egg-boundary violation analysis (Analyze toggle)
    # ------------------------------------------------------------------
    # Per-egg-chart column mapping: which data columns feed the x/y axes.
    _EGG_AXIS_COLS = {
        "xy": (1, 2),   # x=accel-x (col1), y=accel-y (col2)
        "xz": (1, 3),   # x=accel-x (col1), y=accel-z (col3)
        "yz": (2, 3),   # x=accel-y (col2), y=accel-z (col3)
    }

    def _egg_boundary_series(self, chart_name: str) -> list[tuple[str, float, QLineSeries]]:
        """Collect this egg chart's boundary curves + their thresholds.

        Returns a list of ``(rule_label, threshold_seconds, series)`` for each
        reference series on *chart_name* whose name carries a duration tag:
          • "ASTM (0.2s)"   → 0.2 s
          • "... (0s)"      → 0   (no point may cross — restraint solid line)
          • "... (0.2s)"    → 0.2 s
        Other reference series (crosshairs, unnamed) are ignored.
        """
        out = []
        chart = self._charts.get(chart_name)
        if chart is None:
            return out
        for s in chart.series():
            name = (s.name() or "").strip()
            if not name or name == "Measured":
                continue
            low = name.lower()
            threshold = None
            if low.endswith("(0s)") or low.endswith("(0 s)"):
                threshold = 0.0
            elif low.endswith("(0.2s)") or low.endswith("(0.2 s)"):
                threshold = 0.2
            if threshold is None:
                continue
            out.append((name, threshold, s))
        return out

    # Marker-style priority for egg violations (highest wins per span):
    #   ASTM 0.2s   → red filled circle
    #   restraint 0.2s → black filled circle
    #   restraint 0s   → black triangle
    # Rule label → (priority, marker_shape, color). marker_shape is the
    # QScatterSeries.MarkerShape enum value.
    @staticmethod
    def _egg_marker_for(rule_label: str):
        """Return (priority, color, use_triangle) for a rule label.

        ASTM (0.2s) → red circle (priority 3, highest).
        restraint "(0.2s)" → black circle (priority 2).
        restraint "(0s)" → black triangle (priority 1).
        """
        from PySide6.QtCharts import QScatterSeries
        low = rule_label.lower()
        if low.startswith("astm"):
            return (3, QColor("#E8112D"), False)        # red circle
        if low.endswith("(0.2s)") or low.endswith("(0.2 s)"):
            return (2, QColor("#111111"), False)         # black circle
        # "(0s)" restraint
        return (1, QColor("#111111"), True)              # black triangle

    def toggle_egg_analysis(self, active: bool) -> None:
        """Show / hide egg-boundary violation markers + record failed rules.

        Markers are styled by the highest-priority rule a span breaks; the
        summary (which rules failed) is exposed via ``self.failed_rules`` for
        the main window to display in its toolbar (not a floating label here).
        """
        self._egg_analysis_active = active
        self._clear_egg_analysis()
        self.failed_rules: set[str] = set()
        if not active or self._last_egg_data is None:
            return
        data = self._last_egg_data
        try:
            t_arr = np.asarray(data.iloc[:, 0].values, dtype=float)
        except Exception:
            return
        for chart_name in ("xy", "xz", "yz"):
            if chart_name not in self._charts:
                continue
            cx, cy = self._EGG_AXIS_COLS[chart_name]
            measured_xy = np.column_stack([
                np.asarray(data.iloc[:, cx].values, dtype=float),
                np.asarray(data.iloc[:, cy].values, dtype=float)])
            boundaries = self._egg_boundary_series(chart_name)
            # Collect violations: map (i_start,i_end) → set of rule labels.
            span_rules: dict[tuple[int, int], set[str]] = {}
            for rule_label, threshold, series in boundaries:
                path = series_to_path(series)
                if path is None:
                    continue
                viols = find_boundary_violations(
                    measured_xy, t_arr, [path], threshold)
                for (i0, i1) in viols:
                    span_rules.setdefault((i0, i1), set()).add(rule_label)
                if viols:
                    self.failed_rules.add(rule_label)
            # Draw one marker per span, styled by its highest-priority rule.
            for (i0, i1), rules in span_rules.items():
                self._draw_egg_violation(chart_name, measured_xy, i0, i1, rules)

    def _draw_egg_violation(self, chart_name: str, measured_xy: np.ndarray,
                            i0: int, i1: int, rules: set[str]) -> None:
        """Draw enlarged markers on the violating span, styled by priority."""
        from PySide6.QtCharts import QScatterSeries
        chart = self._charts[chart_name]
        pts = measured_xy[i0:i1 + 1]
        # Pick the highest-priority rule among those this span breaks.
        best = max(rules, key=lambda r: self._egg_marker_for(r)[0])
        _prio, color, use_triangle = self._egg_marker_for(best)
        marker = QScatterSeries()
        marker.setMarkerSize(14)
        marker.setColor(color)
        marker.setBorderColor(QColor("white"))
        if use_triangle:
            marker.setMarkerShape(QScatterSeries.MarkerShape.MarkerShapeTriangle)
        for x, y in pts:
            marker.append(QPointF(float(x), float(y)))
        chart.addSeries(marker)
        marker.attachAxis(self._axes[chart_name]["x"])
        marker.attachAxis(self._axes[chart_name]["y"])
        self._egg_analysis_refs.append(marker)
        # Remember which chart owns this scatter so _clear can remove it from
        # the RIGHT chart reliably (removeSeries only works on the owning chart).
        self._egg_scatter_charts[id(marker)] = chart_name

    def _clear_egg_analysis(self) -> None:
        """Remove all egg-analysis scatter series (markers only, no labels)."""
        for ref in self._egg_analysis_refs:
            try:
                # Remove the scatter from the SPECIFIC chart that owns it.
                chart_name = self._egg_scatter_charts.get(id(ref))
                chart = self._charts.get(chart_name) if chart_name else None
                if chart is not None:
                    chart.removeSeries(ref)
            except Exception:
                pass
        self._egg_analysis_refs = []
        self._egg_scatter_charts = {}
        self.failed_rules = set()

    def _reposition_egg_labels(self, *_args) -> None:
        """No per-span labels anymore (summary shown in the toolbar by
        main_window). Kept as a no-op hook so the axis rangeChanged
        connections stay valid."""
        pass

    # ------------------------------------------------------------------
    # Sustained-acceleration limit analysis (Analyze toggle, ax/ay/az)
    # ------------------------------------------------------------------
    # Per-dt measured sustained values are already plotted by addplotxyz as the
    # "Measured ax/ay/az (+/−)" series, one point per dt in _SUST_DTS. The ASTM
    # limit is a piecewise curve limit(dt). This check compares each measured
    # point against the limit AT THE SAME dt, for dt in [0.2s, 14s] only — the
    # standard has no requirement in 0–0.2s.
    _SUST_DTS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                 1, 2, 4, 5, 6, 7, 8, 9, 14]
    # dt range with an ASTM requirement (the limit lines start at 0.2s; t>14s is
    # off-chart).
    _SUST_DT_MIN = 0.2
    _SUST_DT_MAX = 14.0

    @staticmethod
    def _astm_limit_axes(height: float, prone: bool = False) -> dict:
        """Build the ASTM F2291 Sustained limit curves limit(dt) per axis/side.

        Mirrors the red base-limit lines drawn in ``_add_reference_curves``:
        each is a piecewise-linear function sampled at knot dt values, scaled by
        ``coef(height, 'x')``. Returns a dict keyed by axis, each value a dict
        ``{'pos': (dts, values), 'neg': (dts, values)}`` where ``pos`` is the
        +side limit (ax+/ay+/az+) and ``neg`` the −side (ax−/az−). ay is
        symmetric (its neg limit equals −pos). Height scaling uses 'x' for all
        three (matches _add_reference_curves which applies coef(ht,'x') to ax,
        ay, AND az limits).

        ``prone`` swaps the ax− curve for the prone-posture limit
        (_PRONE_ASTM_AXNEG), matching the red line _add_reference_curves draws.
        """
        c = coef(float(height), 'x')
        # ax− is prone-aware; everything else is identical.
        ax_neg = ASTMChartView._PRONE_ASTM_AXNEG if prone else (
            [0.2, 0.5, 14], (np.array([-2, -1.5, -1.5])).tolist())
        return {
            'ax': {
                # ax− : standard [-2,-1.5,-1.5] over [0.2,0.5,14], or prone curve.
                'neg': ax_neg,
                # ax+ : [6,6,4,4,3,3,2.5,2.5]*c over [0.2,1,2,4,5,11.8,12,14]
                'pos': ([0.2, 1, 2, 4, 5, 11.8, 12, 14],
                        (np.array([6, 6, 4, 4, 3, 3, 2.5, 2.5]) * c).tolist()),
            },
            'ay': {
                # ay is symmetric: ±( [3,3,2,2]*c over [0.2,1,2,14] )
                'pos': ([0.2, 1, 2, 14],
                        (np.array([3, 3, 2, 2]) * c).tolist()),
                'neg': ([0.2, 1, 2, 14],
                        (-(np.array([3, 3, 2, 2]) * c)).tolist()),
            },
            'az': {
                # az− : [-2,-1.5,-1.5,-1.2,-1.2]*c over [0.2,0.5,4,7,14]
                'neg': ([0.2, 0.5, 4, 7, 14],
                        (np.array([-2, -1.5, -1.5, -1.2, -1.2]) * c).tolist()),
                # az+ : [6,6,4,4,3,3,2,2]*c over [0.2,1,2,4,5,11.8,12,14]
                'pos': ([0.2, 1, 2, 4, 5, 11.8, 12, 14],
                        (np.array([6, 6, 4, 4, 3, 3, 2, 2]) * c).tolist()),
            },
        }

    # Knot dt values for the restraint limit lines, matching _add_restraint_curves.
    # The restraint limit is drawn from dt=0, so these arrays start at 0 — the
    # ASTM base limit (_astm_limit_axes) instead starts at 0.2.
    _RESTR_DT = {
        'ax_neg': [0, 0.2, 0.5, 14],                       # a31 (front)
        'ax_pos': [0, 0.2, 1, 2, 4, 5, 11.8, 12, 14],      # a32 (back)
        'ay':     [0, 0.2, 1, 2, 14],                      # a33 (left/right)
        'az_neg': [0, 0.2, 0.5, 4, 7, 14],                 # a34 (up)
        'az_pos': [0, 0.2, 1, 2, 4, 5, 11.8, 12, 14],      # a35 (down)
    }

    # Prone-posture ax− limit (overrides the standard ax− when prone is ON).
    # The ASTM base curve has NO requirement over 0–0.2s (knots start at 0.2s);
    # the restraint curve applies from t=0 (matches _RESTR_DT['ax_neg']).
    _PRONE_ASTM_AXNEG = (
        [0.2, 0.5, 2, 3, 4, 5, 14],
        [-3.5, -3.5, -3.5, -2.5, -2.5, -2, -2])
    _PRONE_RESTR_AXNEG = (
        [0, 0.2, 0.5, 2, 3, 4, 5, 14],
        [-2.77, -2.67, -2.58, -2.58, -1.97, -1.97, -1.66, -1.66])

    @staticmethod
    def _restraint_arrays(restraint: str, coefX: float, coefZ: float):
        """Return the restraint-specific coefficient arrays (a31..a35).

        Single source of truth for the restraint limit values — used by BOTH
        _add_restraint_curves (drawing) and _restraint_limit_axes (Analyze),
        so the drawn red lines and the limit check can never drift apart.

        ``restraint`` 'None' returns the height-scaled ASTM base arrays (kept
        as a defensive fallback — both callers now guard against 'None':
        _add_restraint_curves skips the restraint lines, and the Analyze check
        uses _astm_limit_axes instead). For real restraint types it returns
        the tighter per-restraint min/max envelopes.
        Returns (a31, a32, a33, a34, a35) as numpy arrays.
        """
        frontASTM = np.array([-2, -2, -1.5, -1.5])
        backASTM = np.array([6, 6, 6, 4, 4, 3, 3, 2.5, 2.5])
        lrASTM = np.array([3, 3, 3, 2, 2])
        upASTM = np.array([-2, -2, -1.5, -1.5, -1.2, -1.2])
        downASTM = np.array([6, 6, 6, 4, 4, 3, 3, 2, 2])

        if restraint == 'Upper Body':
            a31 = np.maximum(frontASTM, np.array([-2, -1.6, -1.2, -1.2]) * coefX)
            a32 = np.minimum(backASTM, np.array([3.6, 3.6, 3.6, 2.5, 2.5, 2, 2, 2, 2]) * coefX)
            a33 = np.minimum(lrASTM, np.array([3, 2.4, 2.4, 1.6, 1.6]) * coefX)
            a34 = np.maximum(upASTM, np.array([-2, -1.4, -1, -1, -0.7, -0.7]) * coefZ)
            a35 = np.minimum(downASTM, np.array([5, 4.8, 4.8, 3.4, 3.4, 2.6, 2.6, 1.8, 1.8]) * coefZ)
        elif restraint == 'Group Lower Body':
            a31 = np.maximum(frontASTM, np.array([-2, -1.6, -1.2, -1.2]) * coefX)
            a32 = np.minimum(backASTM, np.array([2.5]*5 + [2]*4) * coefX)
            a33 = np.minimum(lrASTM, np.array([2.4, 2.1, 2.1, 1.4, 1.4]) * coefX)
            a34 = np.maximum(upASTM, np.array([-1, 0, 0.2, 0.2, 0.2, 0.2]) * coefZ)
            a35 = np.minimum(downASTM, np.array([4.5, 4, 4, 3.1, 3.1, 2.4, 2.4, 1.7, 1.7]) * coefZ)
        elif restraint == 'Individual Lower Body':
            a31 = np.maximum(frontASTM, np.array([-1.8, -1.5, -1.1, -1.1]) * coefX)
            a32 = np.minimum(backASTM, np.array([2.5]*5 + [2]*4) * coefX)
            a33 = np.minimum(lrASTM, np.array([2.6, 2.2, 2.2, 1.5, 1.5]) * coefX)
            a34 = np.maximum(upASTM, np.array([-1.8, -1.2, -0.9, -0.9, -0.6, -0.6]) * coefZ)
            a35 = np.minimum(downASTM, np.array([4.8, 4.5, 4.5, 3.2, 3.2, 2.5, 2.5, 1.8, 1.8]) * coefZ)
        elif restraint in ('No Restraint', 'Convenience Restraint'):
            a31 = np.maximum(frontASTM, np.array([-1.5, -1.2, -0.7, -0.7]) * coefX)
            a32 = np.minimum(backASTM, np.array([2.5]*5 + [2]*4) * coefX)
            a33 = np.minimum(lrASTM, np.array([1.8, 1.2, 1.2, 0.7, 0.7]) * coefX)
            a34 = np.maximum(upASTM, np.array([-0.2, 0.2, 0.2, 0.2, 0.2, 0.2]) * coefZ)
            a35 = np.minimum(downASTM, np.array([4, 3.8, 3.8, 2.8, 2.8, 2.2, 2.2, 1.6, 1.6]) * coefZ)
        else:
            # 'None' — height-scaled ASTM base (no restraint tightening).
            a31 = np.maximum(frontASTM, frontASTM * coefX)
            a32 = np.minimum(backASTM, backASTM * coefX)
            a33 = np.minimum(lrASTM, lrASTM * coefX)
            a34 = np.maximum(upASTM, upASTM * coefZ)
            a35 = np.minimum(downASTM, downASTM * coefZ)
        # az(+) is the DOWN limit. At rest, gravity alone reads az = +1g, so a
        # limit below 1.0 would falsely flag a stationary ride as a failure.
        # Apply the az(+) >= 1 floor here — this is the single source of truth,
        # so the drawn red line and the Analyze check get the floor together.
        a35 = np.maximum(a35, 1.0)
        return a31, a32, a33, a34, a35

    @staticmethod
    def _cond_mult(cond: str) -> float:
        """Condition multiplier applied to the restraint limits.

        1.25 for 'E-Stop' / 'Expected/Permitted Bumping', else 1.0 — the same
        factor _add_reference_curves applies when drawing the restraint red
        lines (cond_mult * coefX/coefZ), so the Analyze check scales with it.
        """
        return 1.25 if cond in ('E-Stop', 'Expected/Permitted Bumping') else 1.0

    @staticmethod
    def _restraint_limit_axes(restraint: str, height: float,
                              cond: str = 'Normal', prone: bool = False):
        """Build the limit curves for the Analyze check, restraint-aware.

        For real restraint types ('No Restraint', 'Upper Body', ...) the limit
        starts at dt=0 (restraint rules apply from t=0) — so the 0–0.2s range
        IS checked. For 'None' the limit is the ASTM base, which starts at 0.2s,
        so callers must skip 0–0.2s (see toggle_sust_analysis).

        ``cond`` applies the same condition multiplier as the drawn restraint
        lines (1.25 for 'E-Stop' / 'Expected/Permitted Bumping', see
        _cond_mult), so the check can never disagree with the red lines.

        Returns a dict shaped like _astm_limit_axes: axis -> {'pos'/'neg':
        (dts, values)}. All five limit arrays are height- and condition-scaled
        via _restraint_arrays (coefX/coefZ), matching the drawn red lines —
        including az's neg side (a34).

        ``prone`` overrides the ax− curve with the prone restraint limit
        (_PRONE_RESTR_AXNEG), regardless of the selected restraint type —
        matching the line _add_restraint_curves draws.
        """
        coefX = ASTMChartView._cond_mult(cond) * coef(float(height), 'x')
        coefZ = ASTMChartView._cond_mult(cond) * coef(float(height), 'z')
        a31, a32, a33, a34, a35 = ASTMChartView._restraint_arrays(
            restraint, coefX, coefZ)
        dt = ASTMChartView._RESTR_DT
        ax_neg = ASTMChartView._PRONE_RESTR_AXNEG if prone else (
            dt['ax_neg'], a31.tolist())
        return {
            'ax': {
                'neg': ax_neg,
                'pos': (dt['ax_pos'], a32.tolist()),
            },
            'ay': {
                'pos': (dt['ay'], a33.tolist()),
                'neg': (dt['ay'], (-a33).tolist()),
            },
            'az': {
                'neg': (dt['az_neg'], a34.tolist()),
                'pos': (dt['az_pos'], a35.tolist()),
            },
        }

    # Map each measured-series name suffix to (axis, side) for the limit check.
    # side 'pos' = +acceleration (Right/Down), 'neg' = −acceleration (Left/Up).
    # NB the ay (Left) trace is plotted as -neg, so its values are already in the
    # same sign convention as the limit (neg limit is negative); compare directly.
    _SUST_SERIES_MAP = {
        "measured ax (+)":  ('ax', 'pos'),
        "measured ax (-)":  ('ax', 'neg'),
        "measured ay (right)": ('ay', 'pos'),
        "measured ay (left)":  ('ay', 'neg'),
        "measured az (down)":  ('az', 'pos'),
        "measured az (up)":    ('az', 'neg'),
    }

    @staticmethod
    def _interp_limit(dts, vals, target_dt: float) -> float:
        """Piecewise-linear interpolation of limit(dt) at ``target_dt``.

        ``dts``/``vals`` are the limit curve's knots. Clamps to the endpoints
        outside the knot range (the limit is constant beyond the first/last
        knot, matching how the red line is drawn).
        """
        dts = list(dts); vals = list(vals)
        n = len(dts)
        if n == 0:
            return float('inf')
        if target_dt <= dts[0]:
            return float(vals[0])
        if target_dt >= dts[-1]:
            return float(vals[-1])
        # Linear search is fine (<=8 knots).
        for k in range(n - 1):
            if dts[k] <= target_dt <= dts[k + 1]:
                x0, x1 = dts[k], dts[k + 1]
                y0, y1 = vals[k], vals[k + 1]
                if x1 == x0:
                    return float(y0)
                t = (target_dt - x0) / (x1 - x0)
                return float(y0 + t * (y1 - y0))
        return float(vals[-1])

    def toggle_sust_analysis(self, active: bool, restraint: str,
                             cond: str, height: int,
                             prone: bool = False) -> None:
        """Show/hide out-of-limit markers on the measured sustained points.

        Compares each plotted measured sustained point (one per dt) against the
        limit at that dt. Out-of-limit points get a red circle + a "value > limit"
        label. Records failing axis/side labels (e.g. "ax (+)") in
        ``self.sust_failed_rules`` for the toolbar summary.

        Limit selection (matches the red lines drawn by _add_restraint_curves):
          • Real restraint types ('No Restraint', 'Upper Body', ...) → the
            restraint-specific limit, which starts at dt=0, so the 0–0.2s range
            IS checked (restraint rules apply from t=0). ``cond`` scales this
            limit by _cond_mult (1.25 for 'E-Stop' / 'Expected/Permitted
            Bumping'), exactly like the drawn restraint red lines.
          • 'None' → the ASTM base limit, which only starts at dt=0.2s, so the
            0–0.2s range is NOT checked (no ASTM requirement below 0.2s).
        dt > 14s is always off-chart / ignored.

        ``prone`` overrides the ax− limit on BOTH the ASTM base curve and the
        restraint curve (see _PRONE_ASTM_AXNEG / _PRONE_RESTR_AXNEG), matching
        the lines drawn by _add_reference_curves / _add_restraint_curves.
        """
        self._sust_analysis_active = active
        self._clear_sust_analysis()
        self.sust_failed_rules: set[str] = set()
        if not active:
            return

        # 'None' = no restraint selected → ASTM base limit (starts at 0.2s).
        # Any real restraint type → restraint limit (starts at 0).
        is_restraint = restraint != 'None'
        limits = (self._restraint_limit_axes(restraint, height, cond, prone)
                  if is_restraint
                  else self._astm_limit_axes(height, prone))
        for s in self._data_series:
            name = (s.name() or "").strip().lower()
            axis_side = self._SUST_SERIES_MAP.get(name)
            if axis_side is None:
                continue
            axis, side = axis_side
            curve = limits[axis][side]
            pts = s.points()
            if not pts:
                continue
            chart = s.chart()
            if chart is None:
                continue
            for p in pts:
                dt = float(p.x())
                val = float(p.y())
                # Restraint types are checked from dt=0; 'None' only from 0.2s.
                # dt > 14s is always off-chart.
                dt_lo = 0.0 if is_restraint else self._SUST_DT_MIN
                if dt < dt_lo or dt > self._SUST_DT_MAX:
                    continue
                limit = self._interp_limit(curve[0], curve[1], dt)
                # pos side: value above the positive limit fails.
                # neg side: value below the negative limit fails.
                failed = val > limit if side == 'pos' else val < limit
                if not failed:
                    continue
                rule = f"{axis} ({'+' if side == 'pos' else '−'})"
                self.sust_failed_rules.add(rule)
                self._draw_sust_marker(chart, axis, dt, val, limit, side)

    def _draw_sust_marker(self, chart: QChart, axis_name: str,
                          dt: float, val: float, limit: float,
                          side: str) -> None:
        """Draw a red circle at (dt, val) to highlight an out-of-limit point.

        Only the marker is drawn — no value label — so a chart with many
        out-of-limit points stays readable instead of being buried under a pile
        of overlapping "value > limit" labels. The summary toolbar (and the
        report) already names the failing axes.
        """
        # Find the chart_name for this chart (to attach axes correctly).
        chart_name = None
        for nm, ch in self._charts.items():
            if ch is chart:
                chart_name = nm
                break
        if chart_name is None:
            return
        axes = self._axes[chart_name]
        marker = QScatterSeries()
        marker.setMarkerSize(14)
        marker.setColor(QColor("#E8112D"))   # red circle (matches egg ASTM fails)
        marker.setBorderColor(QColor("white"))
        marker.append(QPointF(dt, val))
        chart.addSeries(marker)
        marker.attachAxis(axes["x"])
        marker.attachAxis(axes["y"])
        self._sust_analysis_refs.append(marker)

    def _clear_sust_analysis(self) -> None:
        """Remove all sustained-analysis markers + their value labels."""
        for ref in self._sust_analysis_refs:
            try:
                if isinstance(ref, QLabel):
                    ref.deleteLater()
                else:
                    # QScatterSeries — remove from whichever chart owns it.
                    chart = ref.chart()
                    if chart is not None:
                        chart.removeSeries(ref)
            except Exception:
                pass
        self._sust_analysis_refs = []
        self.sust_failed_rules = set()

    # ------------------------------------------------------------------
    # Annotations
    # ------------------------------------------------------------------

    def _on_click(self, point: QPointF, chart_name: str) -> None:
        chart = self._charts[chart_name]
        cv = self._views[chart_name]
        pos = chart.mapToPosition(point)

        if chart_name in ("xy", "xz", "yz"):
            labels = {"xy": ("x", "y"), "xz": ("x", "z"), "yz": ("y", "z")}
            l1, l2 = labels[chart_name]
            # Estimate time from point index
            text = f"{l1} = {point.x():.4f}\n{l2} = {point.y():.4f}"
        else:
            text = f"dt = {point.x()}\namp = {point.y():.4f}"

        label = QLabel(text, cv)
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
        pass

    # ------------------------------------------------------------------
    # Zoom control methods
    # ------------------------------------------------------------------

    def zoom_mode(self, mode: str) -> None:
        """Set interaction mode for all 6 sub-charts.

        ``"zoom"`` → rubber-band zoom; ``"pan"`` → manual drag scroll;
        anything else → default (no drag). Pan is handled manually by
        ``PanChartView`` because ``ScrollHandDrag`` is unreliable on a
        ``QChartView``.
        """
        for cv in self._views.values():
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
        """Restore all 6 charts to their standard (home) ranges.

        Bug fixed: ``chart.zoomReset()`` only undoes rubber-band zooms; on a
        fresh view it was a no-op. Restoring stored home ranges always works.

        Egg charts (xy/xz/yz) re-equalize their X/Y scale here so a circular
        egg reads as a true circle after Reset View (1 data-unit = same pixel
        count on both axes). Zoom/pan is left free between resets.
        """
        for name, axes in self._axes.items():
            if name in self._EQUAL_EGGS:
                # Equal-scale egg: lock X, derive Y from the plot-area aspect.
                xr, yc = self._EQUAL_EGGS[name]
                equalize_xy_axes(self._charts[name], axes["x"], axes["y"],
                                 xr, yc)
            else:
                xr, yr = self._home_ranges.get(name, ((0, 1), (0, 1)))
                axes["x"].setRange(xr[0], xr[1])
                axes["y"].setRange(yr[0], yr[1])

    def equalize_eggs(self) -> None:
        """Apply the equal-X/Y-scale to every egg chart this view owns.

        Call after the chart is laid out (e.g. once a measured trace is
        plotted). No-op for the Sustained-only instance (no egg charts).
        """
        for name in self._EQUAL_EGGS:
            if name not in self._charts:
                continue
            xr, yc = self._EQUAL_EGGS[name]
            equalize_xy_axes(self._charts[name],
                             self._axes[name]["x"], self._axes[name]["y"],
                             xr, yc)
