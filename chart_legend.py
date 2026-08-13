# coding: utf-8
"""
Custom multi-row legend for Qt Charts.

Qt Charts' built-in ``QLegend`` lays its markers out in a SINGLE row and clips
overflow, so charts with many series (e.g. ASTM F2291 with several measured +
reference + restraint curves) end up with a truncated legend. This module
provides a wrapping ``ChartLegend`` widget (colour swatch + label per entry,
laid out by a ``FlowLayout`` that wraps to as many rows as needed) plus a
``FlowLayout`` itself, since PySide6 doesn't ship one.

Usage in a chart view: hide the built-in legend, build the wrapper layout so
each ``QChartView`` sits below a ``ChartLegend``, and call
``ChartLegend.populate_from(chart)`` whenever series are added/removed.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QPoint, QRect, QSize, QTimer
from PySide6.QtGui import QColor, QPainter, QFont
from PySide6.QtWidgets import (
    QLayout, QWidget, QSizePolicy, QLabel, QFrame,
)


class FlowLayout(QLayout):
    """Lays out child widgets left-to-right, wrapping to a new row when the
    available width is exhausted (like text flow).

    This is the standard Qt "flow layout" pattern, adapted for PySide6. Each
    item keeps its size hint; spacing is uniform between items and between
    rows. ``expandingDirections()`` returns no expansion so the layout stays
    top-left and rows wrap naturally.
    """

    def __init__(self, parent=None, h_spacing: int = 8, v_spacing: int = 4):
        super().__init__(parent)
        self._h_spacing = h_spacing
        self._v_spacing = v_spacing
        self._items: list = []

    # -- QLayout API -------------------------------------------------------
    def addItem(self, item):
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index):
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index):
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QRect):
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        left, top, right, bottom = self.getContentsMargins()
        size += QSize(left + right, top + bottom)
        return size

    # -- internals ---------------------------------------------------------
    def _smart_spacing(self, pm: QLayout.SizePolicy.PixelMetric) -> int:
        parent = self.parent()
        if parent is None:
            return -1 if pm == QLayout.SizePolicy.PM_TopMargin else 0
        if parent.isWidgetType():
            pw = parent.parent()
            if pw is None:
                return -1
            return pw.style().pixelMetric(pm, None, pw) if pw else 0
        return parent.spacing()

    def _do_layout(self, rect: QRect, test_only: bool) -> int:
        """Arrange items into wrapped rows. Returns the total height used.

        When ``test_only`` is True, nothing is positioned — the height is
        returned so heightForWidth() can compute the wrapped height for a
        given width.
        """
        left, top, right, bottom = self.getContentsMargins()
        eff = rect.adjusted(+left, +top, -right, -bottom)
        x = eff.x()
        y = eff.y()
        line_height = 0
        for item in self._items:
            wid = item.widget()
            if wid is not None and not wid.isVisible():
                continue
            hint = item.sizeHint()
            next_x = x + hint.width() + self._h_spacing
            if next_x - self._h_spacing > eff.right() and line_height > 0:
                # Wrap to a new row.
                x = eff.x()
                y = y + line_height + self._v_spacing
                next_x = x + hint.width() + self._h_spacing
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y() + bottom


# ---------------------------------------------------------------------------
# Legend entry widget (swatch + label) and the legend container
# ---------------------------------------------------------------------------

_LEGEND_QSS = (
    "QLabel { color: palette(text); padding: 0px 2px; }"
)


class _LegendEntry(QWidget):
    """One legend entry: a coloured swatch + a text label, side by side."""

    def __init__(self, color: QColor, text: str, parent=None):
        super().__init__(parent)
        self._color = QColor(color)
        self._text = text
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        # Size hint drives FlowLayout wrapping; keep it compact.
        lbl = QLabel(text, self)
        lbl.setStyleSheet(_LEGEND_QSS)
        lbl.setFont(QFont("Segoe UI", 9))
        # The swatch is painted in paintEvent; reserve a little width for it.
        self._label = lbl
        from PySide6.QtWidgets import QHBoxLayout
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        sw = QFrame(self)
        sw.setFixedSize(12, 12)
        sw.setStyleSheet(
            f"background: {color.name(QColor.NameFormat.HexArgb)};"
            " border: 1px solid rgba(0,0,0,80); border-radius: 2px;")
        lay.addWidget(sw)
        lay.addWidget(lbl)

    def text(self) -> str:
        return self._text


class ChartLegend(QWidget):
    """A wrapping legend widget that mirrors one QChart's series list.

    Call :meth:`populate_from` after series are added/removed (or visibility
    changes) to rebuild the entry list. Entries with an empty label are
    skipped (they correspond to unnamed reference curves and would just clutter
    the legend). The layout is a :class:`FlowLayout`, so entries wrap to new
    rows when the legend's width can't fit them all on one line.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._flow = FlowLayout(self, h_spacing=10, v_spacing=3)
        self._flow.setContentsMargins(6, 2, 6, 2)

    def populate_from(self, chart) -> None:
        """Rebuild entries from *chart*'s series list.

        Reads each series' name (label) and pen colour (swatch), skips empty
        names and invisible series. We read from the SERIES rather than the
        legend markers because (a) the markers only exist while the built-in
        legend is visible, and (b) the marker pen colour is unreliable — it
        reads as black; the series pen holds the true line colour.
        """
        # Clear existing entries.
        while self._flow.count():
            item = self._flow.takeAt(0)
            w = item.widget() if item is not None else None
            if w is not None:
                w.deleteLater()

        try:
            series_list = list(chart.series())
        except Exception:
            series_list = []
        for s in series_list:
            try:
                if not s.isVisible():
                    continue
                text = (s.name() or "").strip()
                if not text:
                    continue
                pen = s.pen()
                color = QColor(pen.color()) if pen is not None else QColor("#888")
                entry = _LegendEntry(color, text, self)
                self._flow.addWidget(entry)
            except Exception:
                continue

        # Force the legend (and its parent layout) to recompute height for the
        # new entry count. Without this the parent keeps the OLD height and the
        # FlowLayout's wrapped rows overlap until the next resizeEvent nudges a
        # relayout — the "two rows overlap until I drag the window" bug.
        # Deferred via QTimer.singleShot so it runs AFTER the chart view has a
        # real geometry (during populate_from the width can still be stale,
        # which would compute the wrong wrapped height).
        QTimer.singleShot(0, self._relayout)

    def _relayout(self) -> None:
        """Invalidate cached geometry so the legend reflows immediately.

        ``FlowLayout`` reports ``hasHeightForWidth``; the parent caches the
        legend's height and only re-queries it on a resize. After we rebuild
        entries the cached height is stale (too short for the new row count),
        so rows overlap. We compute the wrapped height for the current width
        and set it as a MINIMUM height on the widget — this forces the parent
        layout to reserve enough vertical space for all rows immediately,
        without waiting for a resizeEvent.
        """
        self._flow.invalidate()
        self.updateGeometry()
        # Recompute the wrapped height for the current width.
        w = max(self.width(), 1)
        h = self._flow.heightForWidth(w)
        # Set minimum height so the parent layout allocates enough space; also
        # apply the geometry directly so the rows are placed correctly even if
        # the parent's activate() doesn't run synchronously.
        self.setMinimumHeight(h)
        self._flow.setGeometry(QRect(0, 0, w, h))
        # Ask the parent chain to re-layout with the new minimum height.
        parent = self.parentWidget()
        if parent is not None:
            parent.updateGeometry()
            pl = parent.layout()
            if pl is not None:
                pl.activate()
            # The grandparent (the grid cell) may also need a nudge.
            gp = parent.parentWidget()
            if gp is not None:
                gpl = gp.layout()
                if gpl is not None:
                    gpl.activate()

    def refresh_theme(self) -> None:
        """Re-style label text after a light/dark switch.

        The swatch colours come from the series pens (re-read on
        populate_from), so a full theme switch should re-populate; this just
        refreshes the label text colour via the stylesheet which already uses
        ``palette(text)``.
        """
        for i in range(self._flow.count()):
            item = self._flow.itemAt(i)
            w = item.widget() if item is not None else None
            if isinstance(w, _LegendEntry):
                w._label.setStyleSheet(_LEGEND_QSS)
