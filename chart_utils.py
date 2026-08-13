# coding: utf-8
"""
Shared utilities for all Qt Charts chart classes.
"""

from __future__ import annotations

from PySide6.QtCharts import QChart, QLineSeries, QValueAxis
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QPen, QColor

from theme import LINE_WIDTH_DEFAULT


# ---------------------------------------------------------------------------
# Data conversion helpers
# ---------------------------------------------------------------------------

def pandas_to_qpointsf(x_col, y_col) -> list[QPointF]:
    """Convert two pandas/numpy Series to a list of QPointF."""
    xv = x_col.values if hasattr(x_col, "values") else x_col
    yv = y_col.values if hasattr(y_col, "values") else y_col
    return [QPointF(float(x), float(y)) for x, y in zip(xv, yv)]


def numpy_to_qpointsf(x_arr, y_arr) -> list[QPointF]:
    """Convert two numpy arrays to a list of QPointF."""
    return [QPointF(float(x), float(y)) for x, y in zip(x_arr, y_arr)]


# ---------------------------------------------------------------------------
# Series factory
# ---------------------------------------------------------------------------

def create_line_series(
    x_data,
    y_data,
    color: str | None = None,
    name: str = "",
    width: float = LINE_WIDTH_DEFAULT,
    dash: bool = False,
    points_visible: bool = False,
) -> QLineSeries:
    """Create a styled QLineSeries from x/y data.

    Parameters
    ----------
    x_data, y_data : array-like
        Numeric sequences (pandas Series, numpy array, or plain list).
    color : str, optional
        Hex colour string, e.g. "#FF0000".
    name : str
        Series legend name.
    width : float
        Pen width.
    dash : bool
        If True, use a dashed line style.
    points_visible : bool
        Show individual data-point markers.
    """
    series = QLineSeries()
    series.setName(name)
    series.setPointsVisible(points_visible)

    pen = QPen()
    pen.setWidthF(width)
    if dash:
        pen.setStyle(Qt.PenStyle.DashLine)
    else:
        pen.setStyle(Qt.PenStyle.SolidLine)
    if color:
        pen.setColor(QColor(color))
    series.setPen(pen)

    # Batch-set the data. PySide6's QLineSeries.replaceNp(numpy, numpy) is a
    # direct C++ path that is ~400× faster than building a list of QPointF in
    # Python (30 ms → 0.07 ms for 27k points) — the Python list build was the
    # dominant cost of every plot/replot. Use it whenever the inputs are
    # array-like; fall back to the QPointF path only for plain iterables.
    import numpy as _np
    try:
        xa = x_data.values if hasattr(x_data, "values") else x_data
        ya = y_data.values if hasattr(y_data, "values") else y_data
        if (hasattr(xa, "__array__") and hasattr(ya, "__array__")
                and len(xa) == len(ya)):
            # replaceNp needs matching-length float arrays; ascontiguousarray
            # handles non-contiguous / strided views and dtype normalisation.
            series.replaceNp(_np.ascontiguousarray(xa, dtype=_np.float64),
                             _np.ascontiguousarray(ya, dtype=_np.float64))
        else:
            raise ValueError("non-array or mismatched inputs")
    except Exception:
        # Fallback: the explicit QPointF list (handles plain lists, odd types).
        if hasattr(x_data, "values"):
            pts = pandas_to_qpointsf(x_data, y_data)
        elif hasattr(x_data, "__array__"):
            pts = numpy_to_qpointsf(x_data, y_data)
        else:
            pts = [QPointF(float(x), float(y)) for x, y in zip(x_data, y_data)]
        series.replace(pts)

    # GPU offload for large line series. Qt Charts falls back to its software
    # rasteriser on platforms/drivers without OpenGL, so wrap in try/except —
    # if acceleration is unavailable we silently keep the (slower) default.
    # NOTE: an OpenGL series renders onto an opaque GL surface, so it can no
    # longer show the GlowBackground halos bleeding through the chart area.
    # This is an accepted trade-off for the large per-frame speed-up (a
    # 27000-point × 3-axis line scrolls ~5× faster under GL).
    try:
        series.setUseOpenGL(True)
    except Exception:
        pass
    return series


# ---------------------------------------------------------------------------
# Axis helpers
# ---------------------------------------------------------------------------

def create_value_axis(
    title: str = "",
    min_val: float = 0,
    max_val: float = 1,
    alignment: Qt.AlignmentFlag = Qt.AlignmentFlag.AlignLeft,
    tick_count: int = 5,
) -> QValueAxis:
    axis = QValueAxis()
    axis.setTitleText(title)
    axis.setRange(min_val, max_val)
    axis.setTickCount(tick_count)
    return axis


def add_series_with_axes(
    chart: QChart,
    series: QLineSeries,
    axis_x: QValueAxis,
    axis_y: QValueAxis,
) -> None:
    """Add a series to an existing chart and attach it to existing axes.

    Assumes axes have already been added to the chart via chart.addAxis().
    """
    chart.addSeries(series)
    series.attachAxis(axis_x)
    series.attachAxis(axis_y)


# ---------------------------------------------------------------------------
# Chart reset
# ---------------------------------------------------------------------------

def clear_chart(chart: QChart) -> None:
    """Remove all series and axes from a QChart."""
    chart.removeAllSeries()
    for axis in chart.axes():
        chart.removeAxis(axis)


def reset_chart_view(view) -> None:
    """Recreate a QChartView's viewport so it has a pristine render surface.

    ``QChartView`` is a ``QGraphicsView``; its viewport caches the last rendered
    frame. When a chart is rebuilt (``initplotxyz``) while the view is on a
    HIDDEN nav page, the cached pixels of the previous (populated) chart can
    persist on top of the new (empty) one — stale curves that can't be panned,
    even though the live chart is correct. ``setViewport(QWidget())`` swaps in a
    brand-new viewport widget, forcing Qt Charts to allocate a fresh render
    surface and repaint the current chart from scratch. This is exactly what a
    freshly-constructed view does, so a chart reset looks identical to the
    app's initial state.
    """
    from PySide6.QtWidgets import QWidget
    try:
        view.setViewport(QWidget())
    except Exception:
        pass


def equalize_xy_axes(chart: QChart, ax_x: QValueAxis, ax_y: QValueAxis,
                     x_range, y_center: float) -> None:
    """Adjust the Y axis range so 1 data-unit = the same pixel count on X and Y.

    Qt Charts has no native "equal aspect" mode. This sets the X axis to
    ``x_range`` (= ``(lo, hi)``), reads the chart's current plot area, and
    derives a Y range centred on ``y_center`` whose units-per-pixel matches X's.
    The result: a circular egg / contour renders as a true circle at default /
    reset (not during an in-progress zoom/pan, which is the intended behaviour).

    Call once after the chart is laid out (e.g. after the view is shown / a
    resize has happened) and again from reset_view. Needs a non-degenerate
    plot area, so it's a no-op when the chart isn't yet sized.
    """
    lo, hi = float(x_range[0]), float(x_range[1])
    plot = chart.plotArea()
    w = plot.width()
    h = plot.height()
    if w <= 0 or h <= 0 or hi <= lo:
        # Chart not yet laid out — set the X range anyway so it's correct when
        # it does lay out; the caller should re-invoke after show/resize.
        ax_x.setRange(lo, hi)
        return
    ax_x.setRange(lo, hi)
    x_units_per_px = (hi - lo) / w
    y_half = x_units_per_px * h / 2.0
    ax_y.setRange(y_center - y_half, y_center + y_half)
