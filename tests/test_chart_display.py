import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import unittest

import numpy as np
from PySide6.QtCharts import QChart, QValueAxis
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from chart_display import DisplayLineSeries, envelope_indices
from chart_utils import add_trajectory


class ChartDisplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_envelope_keeps_impulses_endpoints_and_time_order(self):
        values = np.zeros(71000)
        values[23456], values[23457] = 12, -9
        indices = envelope_indices(values, 800)
        self.assertLessEqual(len(indices), 1602)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], len(values) - 1)
        self.assertTrue(np.all(np.diff(indices) > 0))
        self.assertIn(23456, indices)
        self.assertIn(23457, indices)

    def test_pending_refresh_after_series_is_detached(self):
        chart, series, axis = QChart(), DisplayLineSeries(), QValueAxis()
        time = np.arange(13000, dtype=float)
        series.replaceNp(time, time)
        chart.addSeries(series)
        chart.addAxis(axis, Qt.AlignmentFlag.AlignBottom)
        series.attachAxis(axis)
        chart.removeSeries(series)
        series.refresh_display()

    def test_zoom_restores_every_sample_and_replacement_invalidates_display(self):
        chart = QChart()
        series = DisplayLineSeries()
        time = np.arange(71000, dtype=float) / 500
        values = np.sin(time)
        series.replaceNp(time, values)
        self.assertLess(series.count(), 2100)
        chart.addSeries(series)
        axis = QValueAxis()
        chart.addAxis(axis, Qt.AlignmentFlag.AlignBottom)
        series.attachAxis(axis)
        axis.setRange(20, 20.1)
        series.refresh_display()
        expected = time[(time >= 19.998) & (time <= 20.102)]
        np.testing.assert_allclose([point.x() for point in series.points()], expected)
        series.replaceNp(time, values * 2)
        np.testing.assert_allclose([point.y() for point in series.points()], np.sin(expected) * 2)
        np.testing.assert_array_equal(series._source_x, time)
        series._display_timer.stop()

    def test_short_and_nonfinite_signals_keep_original_samples(self):
        series = DisplayLineSeries()
        series.replaceNp(np.arange(600, dtype=float), np.ones(600))
        self.assertEqual(series.count(), 600)
        values = np.ones(13000)
        values[50] = np.nan
        series.replaceNp(np.arange(13000, dtype=float), values)
        self.assertEqual(series.count(), 13000)

    def test_trajectory_preserves_excursion_and_revisits_at_display_resolution(self):
        from PySide6.QtCore import QRectF
        chart = QChart()
        horizontal, vertical = QValueAxis(), QValueAxis()
        chart.addAxis(horizontal, Qt.AlignmentFlag.AlignBottom)
        chart.addAxis(vertical, Qt.AlignmentFlag.AlignLeft)
        horizontal.setRange(-5, 5)
        vertical.setRange(-5, 5)
        chart.setPlotArea(QRectF(0, 0, 500, 350))
        ramp = np.linspace(.01, .09, 10001)
        source = np.r_[ramp, 3., ramp[::-1]]
        series = add_trajectory(chart, horizontal, vertical, source, -source, 'blue', 'trace', 1.5)
        overview = series.points()
        self.assertLess(len(overview), 200)
        self.assertEqual((overview[0].x(), overview[-1].x()), (source[0], source[-1]))
        self.assertIn((3., -3.), [(point.x(), point.y()) for point in overview])
        horizontal.setRange(.01, .02)
        self.assertGreater(series.count(), len(overview))
        self.assertTrue(all(point.x() == -point.y() for point in series.points()))
