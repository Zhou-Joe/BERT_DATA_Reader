# coding: utf-8
"""
Base class for all Qt Charts chart views used in BERT Reader.

Provides:
  - Antialiased rendering
  - Interaction modes: zoom (rubber-band), pan (drag with open-hand cursor),
    or default (none).  This fixes the original "Pan" toolbox button which was
    a no-op (it only turned the rubber band off and never enabled dragging).
  - Annotation mode ("Single" clears previous, "Multiple" keeps all)
  - Click-to-annotate via floating QLabel tooltips
"""

from __future__ import annotations

from PySide6.QtCharts import QChartView, QChart, QLineSeries, QScatterSeries
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPainter, QPen, QColor, QCursor
from PySide6.QtWidgets import QLabel, QSizePolicy

from chart_utils import clear_chart
from theme import theme_manager, chart_tooltip_style, apply_chart_theme


# Modes accepted by set_interaction()
INTERACTION_ZOOM = "zoom"
INTERACTION_PAN = "pan"
INTERACTION_DEFAULT = "default"


class PanMixin:
    """Reusable manual-pan behaviour for a ``QChartView`` subclass.

    Why manual: ``QGraphicsView.ScrollHandDrag`` is unreliable on a
    ``QChartView`` — it sets the open-hand cursor but the chart's own scene
    handling swallows the drag, so the data never scrolls. This mixin tracks
    left-button drags itself and calls ``chart().scroll(dx, dy)``.

    Panning is enabled by default (in BOTH ``INTERACTION_DEFAULT`` and
    ``INTERACTION_PAN`` modes) so the user can drag the chart around without
    first clicking the Pan toolbox button. Only ``INTERACTION_ZOOM`` disables
    it, because there a left-drag draws the rubber-band zoom rectangle.
    """

    # Subclasses must define:  self._interaction_mode: str
    _panning: bool = False
    _pan_last_pos: QPointF | None = None

    def _pan_allows_drag(self) -> bool:
        """Whether a left-drag should pan right now.

        Panning is allowed unless we're in explicit rubber-band zoom mode
        (where the drag must draw the rectangle instead). Both default and
        pan modes accept drags.
        """
        return getattr(self, "_interaction_mode", INTERACTION_DEFAULT) != INTERACTION_ZOOM

    def _pan_mouse_press(self, event) -> bool:
        """Handle mousePressEvent start-of-pan. Returns True if consumed."""
        if (self._pan_allows_drag()
                and event.button() == Qt.MouseButton.LeftButton):
            self._panning = True
            self._pan_last_pos = event.position()
            self.viewport().setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
            event.accept()
            return True
        return False

    def _pan_mouse_move(self, event) -> bool:
        """Handle mouseMoveEvent drag. Returns True if consumed.

        Grab-the-data semantics: the curves follow the cursor. X is inverted
        (drag right → viewport scrolls left, revealing earlier data); Y is
        NOT inverted because the chart's data Y-axis points up while screen
        Y points down, so the two flips already cancel out — a direct
        ``dy`` gives "drag down → curves move down". After scrolling we fire
        :meth:`_pan_applied` so containers (e.g. the Trend view) can mirror
        the new range onto linked charts.
        """
        if self._panning and self._pan_last_pos is not None:
            cur = event.position()
            dx = int(cur.x() - self._pan_last_pos.x())
            dy = int(cur.y() - self._pan_last_pos.y())
            self._pan_last_pos = cur
            chart = self.chart()
            if chart is not None and (dx or dy):
                # X inverted (scroll left to reveal earlier data); Y direct
                # so vertical drag feels natural (screen-Y and data-Y cancel).
                chart.scroll(-dx, dy)
                self._pan_applied()
            event.accept()
            return True
        return False

    def _pan_applied(self) -> None:
        """Hook called after a pan scroll. Override in containers to sync
        linked charts (e.g. mirror this chart's new axis range onto peers)."""
        pass

    def _pan_mouse_release(self, event) -> bool:
        """Handle mouseReleaseEvent end-of-pan. Returns True if consumed.

        On release we clear the cursor only if we set it (pan/zoom modes); in
        default mode we restore whatever cursor the view had (typically none)
        so the open-hand doesn't linger after a casual drag.
        """
        if self._panning and event.button() == Qt.MouseButton.LeftButton:
            self._panning = False
            self._pan_last_pos = None
            # Only swap back to the open-hand cursor in explicit pan mode;
            # in default mode unset so the pointer returns to normal.
            if getattr(self, "_interaction_mode", INTERACTION_DEFAULT) == INTERACTION_PAN:
                self.viewport().setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
            else:
                self.viewport().unsetCursor()
            event.accept()
            return True
        return False

    def _pan_cancel(self) -> None:
        """Abort any in-flight pan (call when interaction mode changes)."""
        self._panning = False
        self._pan_last_pos = None

    # ------------------------------------------------------------------
    # Wheel zoom — zoom toward the cursor. Active in every mode (a scroll
    # wheel never conflicts with a left-drag rubber band), so the user can
    # zoom without touching the toolbox at all.
    # ------------------------------------------------------------------

    def _wheel_zoom(self, event) -> bool:
        """Zoom the chart toward the cursor on wheel rotation.

        Returns True if consumed. Keeps the data point under the cursor fixed
        while zooming (the usual "zoom where I point" expectation).

        We avoid ``QChart.mapToValue`` here because it needs a series for a
        reliable mapping and behaves inconsistently across Qt versions. Instead
        we compute the cursor's data value directly from the axis range and the
        plot-area geometry, which is robust for any series count.
        """
        chart = self.chart()
        if chart is None:
            return False
        delta = event.angleDelta().y()
        if delta == 0:
            return False
        factor = 1.15 if delta > 0 else 1 / 1.15

        # Plot area in VIEW coordinates (where event.position() lives).
        # QChart.plotArea() returns scene coords; on a QChartView with the
        # default identity scene transform these coincide with view coords, so
        # we can use plotArea() directly. (mapFromScene returns a QPolygonF,
        # which lacks left()/width(); boundingRect() would work but the direct
        # value is simpler and correct here.)
        view_plot = chart.plotArea()
        cursor = event.position()

        for axis in chart.axes():
            try:
                lo, hi = axis.min(), axis.max()
                span = hi - lo
                if span <= 0:
                    continue
                new_span = span / factor
                if new_span < 1e-9:
                    continue
                # Fractional position of the cursor along this axis within the
                # plot rect. Clamp to [0,1] so a cursor outside the plot still
                # zooms sanely (anchored at the nearest edge).
                if axis.orientation() == Qt.Orientation.Horizontal:
                    if view_plot.width() > 0:
                        frac = (cursor.x() - view_plot.left()) / view_plot.width()
                    else:
                        frac = 0.5
                else:
                    # Vertical axis: screen y grows down, data y grows up.
                    if view_plot.height() > 0:
                        frac = 1.0 - (cursor.y() - view_plot.top()) / view_plot.height()
                    else:
                        frac = 0.5
                frac = max(0.0, min(1.0, frac))
                # Data value under the cursor stays fixed.
                anchor_val = lo + frac * span
                new_lo = anchor_val - frac * new_span
                new_hi = new_lo + new_span
                axis.setRange(new_lo, new_hi)
            except Exception:
                continue
        event.accept()
        return True


class PanChartView(QChartView, PanMixin):
    """A bare ``QChartView`` with manual pan support.

    Drop-in replacement for ``QChartView`` in chart containers that build
    their own sub-views (ASTM, GB, Trend). Set the interaction mode via
    ``set_pan_active(bool)``.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._interaction_mode = INTERACTION_DEFAULT

    def set_pan_active(self, active: bool) -> None:
        self._interaction_mode = INTERACTION_PAN if active else INTERACTION_DEFAULT
        if active:
            self.setRubberBand(QChartView.RubberBand.NoRubberBand)
            self.setDragMode(QChartView.DragMode.NoDrag)
            self.viewport().setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        else:
            self._pan_cancel()
            self.viewport().unsetCursor()

    def mousePressEvent(self, event) -> None:
        if self._pan_mouse_press(event):
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._pan_mouse_move(event):
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._pan_mouse_release(event):
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:
        """Scroll wheel zooms toward the cursor (works in every mode)."""
        if self._wheel_zoom(event):
            return
        super().wheelEvent(event)


class BaseChartView(QChartView, PanMixin):
    """Base chart view with zoom/pan and click-annotation support.

    Pan is implemented manually (mouse-press → scroll on mouse-move) instead
    of relying on ``QGraphicsView.ScrollHandDrag`` — see ``PanMixin``.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)

        # Annotation state
        self.annot_mode: str = "Single"
        self._annot_markers: QScatterSeries | None = None
        self._tooltip_label: QLabel | None = None

        # Interaction state
        self._interaction_mode = INTERACTION_DEFAULT
        self._panning = False
        self._pan_last_pos: QPointF | None = None
        self.set_interaction(INTERACTION_DEFAULT)

    # ------------------------------------------------------------------
    # Interaction mode — the real fix for the toolbox Pan/Zoom buttons
    # ------------------------------------------------------------------

    def set_interaction(self, mode: str) -> None:
        """Configure how the mouse interacts with this chart.

        ``"zoom"``    : left-drag draws a rubber-band zoom rectangle.
        ``"pan"``     : left-drag pans the chart (manual chart.scroll()).
        ``"default"`` : no drag interaction (clicks still register).
        """
        self._interaction_mode = mode
        # Always cancel any in-flight pan when the mode changes.
        self._pan_cancel()
        if mode == INTERACTION_ZOOM:
            self.setRubberBand(QChartView.RubberBand.RectangleRubberBand)
            self.setDragMode(QChartView.DragMode.NoDrag)
            self.viewport().unsetCursor()
        elif mode == INTERACTION_PAN:
            # Keep NoDrag so our manual mouse handlers receive the events;
            # ScrollHandDrag would fight Qt Charts' own scene handling.
            self.setRubberBand(QChartView.RubberBand.NoRubberBand)
            self.setDragMode(QChartView.DragMode.NoDrag)
            self.viewport().setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
        else:  # default
            self.setRubberBand(QChartView.RubberBand.NoRubberBand)
            self.setDragMode(QChartView.DragMode.NoDrag)
            self.viewport().unsetCursor()

    @property
    def interaction_mode(self) -> str:
        return self._interaction_mode

    # ------------------------------------------------------------------
    # Theme refresh — QChart backgrounds don't follow QSS, restyle manually
    # ------------------------------------------------------------------

    def restyle_theme(self) -> None:
        """Re-apply the current theme palette to this view's chart.

        Called by MainWindow.refresh_theme() on a light/dark switch.
        """
        chart = self.chart()
        if chart is not None:
            apply_chart_theme(chart)
        # Tooltip label is re-styled on next show via chart_tooltip_style().

    def setChart(self, chart) -> None:
        """Override to apply the theme palette whenever a chart is attached."""
        super().setChart(chart)
        try:
            apply_chart_theme(chart)
        except Exception as e:
            print(f"chart theme apply failed: {e}")

    # ------------------------------------------------------------------
    # Manual panning — delegate to PanMixin
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        if self._pan_mouse_press(event):
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._pan_mouse_move(event):
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._pan_mouse_release(event):
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:
        """Scroll wheel zooms toward the cursor (works in every mode)."""
        if self._wheel_zoom(event):
            return
        super().wheelEvent(event)

    # ------------------------------------------------------------------
    # Annotation mode
    # ------------------------------------------------------------------

    def set_annotation_mode(self, mode: str) -> None:
        self.annot_mode = mode

    def _ensure_annot_markers(self, chart: QChart) -> QScatterSeries:
        """Lazily create the scatter series used for annotation dots."""
        if self._annot_markers is None:
            self._annot_markers = QScatterSeries()
            self._annot_markers.setName("_annotations")
            self._annot_markers.setMarkerSize(10)
            self._annot_markers.setColor(QColor(255, 0, 0, 180))
            self._annot_markers.setBorderColor(QColor("darkred"))
            self._annot_markers.setPointsVisible(True)
            chart.addSeries(self._annot_markers)
        return self._annot_markers

    def _clear_annotations(self) -> None:
        if self._annot_markers is not None:
            self._annot_markers.replace([])

    def _show_tooltip(self, series: QLineSeries, point: QPointF) -> None:
        """Display a (reused, theme-styled) tooltip label near the click."""
        if self._tooltip_label is None:
            self._tooltip_label = QLabel(self)
            self._tooltip_label.raise_()

        # Re-apply style on every show so a theme toggle is reflected.
        self._tooltip_label.setStyleSheet(chart_tooltip_style())

        chart = self.chart()
        pos = chart.mapToPosition(point)
        text = self._format_tooltip(point, series)

        self._tooltip_label.setText(text)
        self._tooltip_label.adjustSize()

        label_x = min(pos.x() + 12, self.width() - self._tooltip_label.width() - 4)
        label_y = max(pos.y() - self._tooltip_label.height() - 4, 4)
        self._tooltip_label.move(int(label_x), int(label_y))
        self._tooltip_label.show()
        self._tooltip_label.raise_()

    def _format_tooltip(self, point: QPointF, series: QLineSeries) -> str:
        """Override in subclasses for domain-specific tooltip text."""
        return f"x = {point.x():.3f}\ny = {point.y():.3f}"

    def _on_series_clicked(self, point: QPointF) -> None:
        """Slot to connect to QLineSeries.clicked."""
        chart = self.chart()

        if self.annot_mode == "Single":
            self._clear_annotations()

        markers = self._ensure_annot_markers(chart)
        pts = list(markers.points())
        pts.append(point)
        markers.replace(pts)

        sender_series = self.sender()
        if isinstance(sender_series, QLineSeries):
            self._show_tooltip(sender_series, point)
