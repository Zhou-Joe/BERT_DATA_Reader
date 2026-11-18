"""Compact recording overview for reviewing trim windows."""
import numpy as np
from PySide6.QtCore import Qt, QRectF, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import QWidget
from qfluentwidgets import DoubleSpinBox

from theme import theme_manager


class TimeField(DoubleSpinBox):
    """Display readable seconds but retain the exact detected sample boundary."""
    secondsChanged = Signal(float)

    def __init__(self, times, end, fs):
        super().__init__()
        self._times, self._end = times, end
        self.setDecimals(6)
        self.setRange(min(0, times[0]) - 1, end + 1)
        self.setSuffix(" s")
        self.setSingleStep(1 / fs)
        self._exact = None
        self._setting = False
        self.valueChanged.connect(self._edited)

    def _edited(self, value):
        if not self._setting:
            self._exact = None
            self.secondsChanged.emit(value)

    def set_seconds(self, seconds):
        self._setting = True
        self.setValue(seconds)
        self._setting = False
        self._exact = seconds
        self.secondsChanged.emit(seconds)

    def seconds(self):
        return self.value() if self._exact is None else self._exact

    def stepBy(self, steps):
        # Step by acquired samples, including across clock gaps. Adding a
        # rounded 1/fs to the display can jump two samples in a 512 Hz file.
        if steps > 0:
            i = self._times.searchsorted(self.seconds(), side="right") + steps - 1
        elif steps < 0:
            i = self._times.searchsorted(self.seconds(), side="left") + steps
        else:
            return
        self.set_seconds(self._end if i >= len(self._times) else self._times[max(0, i)])


class SegmentPreview(QWidget):
    """Show acceleration magnitude by source time, leaving logger gaps blank."""
    def __init__(self, data, first, end):
        super().__init__()
        self.setFixedHeight(80)
        self.setAccessibleName("Acceleration magnitude and selected time windows")
        self.setToolTip("Shading marks the time windows you will keep.")
        self._first, self._end = first, end
        self._windows = []
        # Compute the envelope once, preserving narrow peaks on long recordings.
        samples = data.data.iloc[:, :4].to_numpy(dtype=float)
        samples = samples[np.isfinite(samples).all(axis=1)]
        self._bins = 1024
        self._low = np.full(self._bins, np.inf)
        self._high = np.full(self._bins, -np.inf)
        if samples.size and end > first:
            bins = np.clip(((samples[:, 0] - first) / (end - first)
                            * self._bins).astype(int), 0, self._bins - 1)
            amplitude = np.linalg.norm(samples[:, 1:4], axis=1)
            np.minimum.at(self._low, bins, amplitude)
            np.maximum.at(self._high, bins, amplitude)
        self._occupied = np.flatnonzero(np.isfinite(self._low))

    def set_windows(self, windows):
        self._windows = windows
        self.update()

    def paintEvent(self, event):
        tk = theme_manager.tokens
        painter = QPainter(self)
        width = self.width() - 2
        top, height = 3, 48
        painter.fillRect(1, top, width, height, QColor(tk["surface_alt"]))
        shade = QColor(tk["accent"])
        shade.setAlpha(42)
        span = self._end - self._first
        if span <= 0:
            return
        for start, end in self._windows:
            x1 = 1 + width * (start - self._first) / span
            x2 = 1 + width * (end - self._first) / span
            painter.fillRect(QRectF(x1, top, max(0, x2 - x1), height), shade)
            painter.fillRect(QRectF(x1, top, max(0, x2 - x1), 2), QColor(tk["accent"]))
        if self._occupied.size:
            lo, hi = self._low.min(), self._high.max()
            scale = (height - 9) / max(.02, hi - lo)
            painter.setPen(QColor(tk["text_secondary"]))
            for i in self._occupied:
                x = round(1 + (i + .5) * width / self._bins)
                y1 = top + height - 4 - round((self._high[i] - lo) * scale)
                y2 = top + height - 4 - round((self._low[i] - lo) * scale)
                painter.drawLine(x, y1, x, max(y1 + 1, y2))
        legend = QRectF(1, top + height + 1, width, 24)
        painter.setPen(QColor(tk["text_secondary"]))
        painter.drawText(legend, Qt.AlignLeft | Qt.AlignVCenter, f"{self._first:g} s")
        painter.drawText(legend, Qt.AlignCenter, "Acceleration magnitude  ·  selected time shaded")
        painter.drawText(legend, Qt.AlignRight | Qt.AlignVCenter, f"{self._end:g} s")
