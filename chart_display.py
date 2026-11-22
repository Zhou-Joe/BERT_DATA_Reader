"""Peak-preserving display resolution for long time-series charts."""

import numpy as np

from PySide6.QtCharts import QLineSeries
from PySide6.QtCore import Qt, QTimer


def envelope_indices(values, bins):
    count = len(values)
    if count <= bins * 2:
        return np.arange(count)
    stride = int(np.ceil(count / bins))
    padded = np.full(int(np.ceil(count / stride)) * stride, np.nan)
    padded[:count] = values
    blocks = padded.reshape(-1, stride)
    starts = np.arange(len(blocks)) * stride
    extrema = np.column_stack((starts + np.nanargmin(blocks, axis=1),
                               starts + np.nanargmax(blocks, axis=1))).ravel()
    return np.unique(np.r_[0, extrema, count - 1])


class TrajectorySeries(QLineSeries):
    """Draw the endpoints of each consecutive run within a half-pixel cell."""

    def __init__(self, horizontal, vertical):
        super().__init__()
        self._horizontal = np.ascontiguousarray(horizontal, dtype=np.float64)
        self._vertical = np.ascontiguousarray(vertical, dtype=np.float64)
        self._scale = None
        self._extrema = [self._horizontal.argmin(), self._horizontal.argmax(),
                         self._vertical.argmin(), self._vertical.argmax()]

    def bind_resolution(self, chart, horizontal_axis, vertical_axis):
        self._horizontal_axis = horizontal_axis
        self._vertical_axis = vertical_axis
        chart.plotAreaChanged.connect(self.refresh_resolution)
        horizontal_axis.rangeChanged.connect(self.refresh_resolution)
        vertical_axis.rangeChanged.connect(self.refresh_resolution)
        self.refresh_resolution()

    def refresh_resolution(self, *_args):
        chart = self.chart()
        if chart is None:
            return
        area = chart.plotArea()
        horizontal_step = (self._horizontal_axis.max() - self._horizontal_axis.min()) / (2 * max(1, area.width()))
        vertical_step = (self._vertical_axis.max() - self._vertical_axis.min()) / (2 * max(1, area.height()))
        scale = (horizontal_step, vertical_step)
        if scale == self._scale or min(scale) <= 0:
            return
        self._scale = scale
        columns = np.floor(self._horizontal / horizontal_step)
        rows = np.floor(self._vertical / vertical_step)
        crosses_cell = (columns[1:] != columns[:-1]) | (rows[1:] != rows[:-1])
        keep = np.ones(len(columns), dtype=bool)
        keep[1:-1] = crosses_cell[:-1] | crosses_cell[1:]
        keep[self._extrema] = True
        self.replaceNp(self._horizontal[keep], self._vertical[keep])


class DisplayLineSeries(QLineSeries):
    """Retain full source samples and draw the visible min/max envelope."""

    def __init__(self):
        super().__init__()
        self._source_x = self._source_y = np.array([], dtype=float)
        self._time_axis = None
        self._display_key = None
        self._can_reduce = False
        self._display_timer = QTimer(self)
        self._display_timer.setSingleShot(True)
        self._display_timer.setInterval(16)
        self._display_timer.timeout.connect(self.refresh_display)

    def replaceNp(self, time, values):
        self._source_x = np.array(time, dtype=np.float64, order='C', copy=True)
        self._source_y = np.array(values, dtype=np.float64, order='C', copy=True)
        self._can_reduce = (len(time) > 12000 and len(time) == len(values)
                            and np.isfinite(self._source_x).all()
                            and np.isfinite(self._source_y).all()
                            and np.all(np.diff(self._source_x) >= 0))
        self._display_key = None
        self.refresh_display()

    def attachAxis(self, axis):
        attached = super().attachAxis(axis)
        if attached and axis.orientation() == Qt.Orientation.Horizontal:
            self._time_axis = axis
            axis.rangeChanged.connect(self._schedule_display)
            self.chart().plotAreaChanged.connect(self._schedule_display)
            self._schedule_display()
        return attached

    def _schedule_display(self, *_args):
        if not self._display_timer.isActive():
            self._display_timer.start()

    def refresh_display(self):
        start, end = 0, len(self._source_x)
        bins = 1024
        if self._can_reduce and self._time_axis is not None:
            # A view switch can detach the series before its pending resize timer fires.
            chart = self.chart()
            if chart is None:
                return
            start = max(0, int(np.searchsorted(self._source_x, self._time_axis.min())) - 1)
            end = min(end, int(np.searchsorted(self._source_x, self._time_axis.max(), side='right')) + 1)
            bins = max(256, min(4096, int(chart.plotArea().width())))
        key = (start, end, bins)
        if key == self._display_key:
            return
        self._display_key = key
        if self._can_reduce:
            indices = envelope_indices(self._source_y[start:end], bins) + start
            super().replaceNp(self._source_x[indices], self._source_y[indices])
        else:
            super().replaceNp(self._source_x, self._source_y)
