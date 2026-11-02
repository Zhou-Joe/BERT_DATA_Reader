"""Headless Qt integration checks; run from the repository root."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import unittest
from unittest.mock import patch

from PySide6.QtCore import QPoint, Qt, QEventLoop, QTimer
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from accdata import AccData
from dialogs import TruncateTimeDialog
from main_window import MainWindow
from sci_calculation import ProcessASTM
from theme import theme_manager
from test_data import recording


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        if os.path.exists('C:/Windows/Fonts/segoeui.ttf'):
            QFontDatabase.addApplicationFont('C:/Windows/Fonts/segoeui.ttf')
        cls.app.setFont(QFont('Segoe UI', 10))
        theme_manager.apply(cls.app)

    def setUp(self):
        theme_manager.set_mode('light')
        self.window = MainWindow()
        self.window.stackedWidget.setAnimationEnabled(False)
        self.window._refresh_timer.setInterval(20)
        self.window.show()
        self.app.processEvents()

    def tearDown(self):
        self.window.cancel_processing()
        self.window._jobs.pool.waitForDone(5000)
        self.app.processEvents()
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def load(self, name='recording.sup', fs=200):
        data = AccData(recording(fs=fs, start=-1, seconds=3),
                       is_raw=True, path=name)
        self.window.set_datalist(data)
        self.window.add_trend_data(data, 'Standard', len(self.window.datalist))
        self.app.processEvents()
        return data

    def wait_for_idle(self, timeout=5000):
        # Use the real Qt event loop: a Python loop around QTest.qWait can
        # starve worker threads of the GIL despite appearing to yield.
        loop = QEventLoop()
        poll = QTimer(loop)
        poll.setInterval(10)
        def idle():
            return not (self.window._jobs.active_count or self.window._refresh_timer.isActive()
                        or self.window._cutoff_timer.isActive() or self.window._import_commit is not None)
        poll.timeout.connect(lambda: loop.quit() if idle() else None)
        deadline = QTimer(loop)
        deadline.setSingleShot(True)
        deadline.timeout.connect(loop.quit)
        poll.start()
        deadline.start(timeout)
        loop.exec()
        poll.stop()
        deadline.stop()
        self.assertTrue(idle(), 'Background jobs did not finish before the deadline')
        self.app.processEvents()

    def refresh(self):
        self.window._generate_active_page()
        self.wait_for_idle()

    def render_page(self, index):
        self.window.stackedWidget.setCurrentIndex(index)
        QTest.qWait(10)
        self.app.processEvents()

    def test_selector_keeps_selection_when_more_data_is_added(self):
        self.load('one.sup')
        self.load('two.sup')
        self.window.select_data.setCurrentText('two.sup')
        self.load('three.sup')
        self.assertEqual(self.window.select_data.currentText(), 'two.sup')
        self.assertTrue(all(combo.currentText() == 'two.sup'
                            for combo in self.window._data_combos))

    def test_all_seven_views_generate_and_use_acquisition_rate(self):
        data = self.load(fs=200)
        self.refresh()
        self.render_page(1)
        self.assertEqual(self.window._processing_state, 'ready')
        canvas = self.window.canvasASTMSust
        measured = next(s for s in canvas._charts['ax'].series()
                        if s.name() == 'Measured ax (+)')
        expected, _ = ProcessASTM(data.std_data, 4)  # .02 seconds at 200 Hz
        self.assertAlmostEqual(measured.at(1).y(), expected[0])
        for index in range(self.window.stackedWidget.count()):
            self.render_page(index)
            self.assertTrue(self.window._page_is_current())
            self.assertFalse(self.window.grab().isNull())

    def test_egg_canvas_skips_sustained_calculations(self):
        data = self.load()
        with patch('chart_astm.ProcessASTM', side_effect=AssertionError('Unneeded rolling pass')):
            self.window.canvasASTMEgg.addplotxyz(data.std_data, 'None', 'Normal', 0)

    def test_analysis_off_clears_hidden_pages(self):
        self.load()
        self.refresh()
        self.window._analyze_btn.setChecked(True)
        self.render_page(1)
        self.assertTrue(self.window.trendplot._analysis_active)
        self.assertTrue(self.window.canvasASTMSust._sust_analysis_active)
        self.window._analyze_btn.setChecked(False)
        self.assertFalse(self.window.trendplot._analysis_active)
        self.assertFalse(self.window.canvasASTMSust._sust_analysis_active)

    def test_unplotted_view_does_not_report_a_pass(self):
        self.load()
        self.window.stackedWidget.setCurrentIndex(1)
        self.window._analyze_btn.setChecked(True)
        self.assertFalse(self.window.canvasASTMSust._sust_analysis_active)
        self.assertFalse(self.window._page_is_current())
        self.assertTrue(self.window.stackedWidget.currentWidget()._pending_label.isVisible())
        self.wait_for_idle()
        self.assertTrue(self.window._page_is_current())

    def test_external_theme_change_keeps_toggle_in_sync(self):
        theme_manager.set_mode('dark')
        self.assertTrue(self.window.theme_toggle.isChecked())
        theme_manager.set_mode('light')
        self.assertFalse(self.window.theme_toggle.isChecked())

    def test_generate_errors_are_visible(self):
        self.load()
        self.window.stackedWidget.setCurrentIndex(3)
        with patch.object(self.window.canvasGB, 'addplotxyz', side_effect=ValueError('bad frame')):
            self.refresh()
        self.assertEqual(self.window._processing_state, 'error')
        self.assertIn('bad frame', self.window.statusBar().currentMessage())

    def test_recording_row_selects_dataset_and_inspector(self):
        self.load('one.sup')
        second = self.load('two.sup')
        QTest.mouseClick(self.window.sidebar._cards[1], Qt.MouseButton.LeftButton)
        self.assertEqual(self.window.select_data.currentText(), 'two.sup')
        self.assertEqual(self.window.sidebar._selected_name, 'two.sup')
        self.assertIs(self.window.sidebar._inspector_frame, second.std_data)

    def test_recording_search_and_clear(self):
        self.load('one.sup')
        self.load('two.sup')
        sidebar = self.window.sidebar
        sidebar._search.setText('TWO')
        self.assertTrue(sidebar._cards[0].isHidden())
        self.assertFalse(sidebar._cards[1].isHidden())
        self.assertEqual(sidebar._count.text(), '1 / 2')
        self.window.reset_trend(reset_datalist=True)
        self.assertEqual(sidebar._search.text(), '')
        self.assertFalse(sidebar._inspector_scroll.isVisible())

    def test_visibility_button_does_not_change_selection(self):
        self.load('one.sup')
        self.load('two.sup')
        QTest.mouseClick(self.window.sidebar._cards[1]._plot_check,
                         Qt.MouseButton.LeftButton)
        self.assertEqual(self.window.select_data.currentText(), 'one.sup')
        self.assertFalse(self.window.dataset_visibility['two.sup'])

    def test_inspector_follows_plot_mode(self):
        data = self.load()
        self.window.switch_plot_type('Raw')
        self.assertIs(self.window.sidebar._inspector_frame, data.data)
        values = data.data.iloc[:, 1]
        self.assertEqual(self.window.sidebar._inspector_values[('X', 1)].text(),
                         f'{values.max():+.3f} / {values.min():+.3f}')

    def test_only_selected_recording_computes_inspector_statistics(self):
        from sci_calculation import ProcessASTM as process
        self.window._refresh_suspended = True
        with patch('sci_calculation.ProcessASTM', wraps=process) as sustained:
            self.load('one.sup')
            count = sustained.call_count
            self.load('two.sup')
            self.assertEqual(sustained.call_count, count)

    def test_removing_selected_recording_selects_remaining_row(self):
        self.load('one.sup')
        self.load('two.sup')
        self.window.select_data.setCurrentText('two.sup')
        self.window.sidebar._cards[1]._more.menu().actions()[-1].trigger()
        self.assertEqual(self.window.select_data.currentText(), 'one.sup')
        self.assertEqual(self.window.sidebar._selected_name, 'one.sup')
        self.assertEqual(len(self.window.sidebar._cards), 1)

    def test_long_filename_and_footer_fit_laptop_window(self):
        from PySide6.QtWidgets import QPushButton
        name = 'long-recording-name-with-multiple-test-identifiers-and-versions.sup'
        self.load(name)
        self.window.resize(1100, 720)
        self.app.processEvents()
        row = self.window.sidebar._cards[0]
        self.assertEqual(row._title_label.toolTip(), name)
        self.assertLessEqual(row._title_label.fontMetrics().horizontalAdvance(
            row._title_label.text()), row._title_label.width())
        export = self.window.sidebar._actions_host.findChild(QPushButton)
        position = export.mapTo(self.window, QPoint(0, 0))
        self.assertTrue(export.isVisible())
        self.assertLessEqual(position.y() + export.height(),
                             self.window.statusBar().geometry().top())

    def test_clear_resets_cards_selection_visibility_and_empty_state(self):
        self.load()
        self.window.dataset_visibility['recording.sup'] = False
        self.window.reset_trend(reset_datalist=True)
        self.app.processEvents()
        self.assertFalse(self.window.datalist)
        self.assertFalse(self.window.dataset_visibility)
        self.assertFalse(self.window.sidebar._cards)
        self.assertEqual(self.window._shift_combo.count(), 0)
        self.assertTrue(self.window._trend_empty.isVisible())
        self.assertFalse(self.window._generate_button.isEnabled())

    def test_command_controls_fit_at_supported_window_sizes(self):
        for mode in ['light', 'dark']:
            theme_manager.set_mode(mode)
            for width, height in [(1100, 720), (1366, 768), (1600, 900)]:
                self.window.resize(width, height)
                self.app.processEvents()
                bar = self.window._top_bar
                self.assertLessEqual(bar.minimumSizeHint().width(), bar.width())
                for widget in [self.window.select_data, self.window._generate_button,
                               self.window.cutoff_spin]:
                    top_left = widget.mapTo(bar, QPoint(0, 0))
                    self.assertGreaterEqual(top_left.x(), 0)
                    self.assertLessEqual(top_left.x() + widget.width(), bar.width())

    def test_cutoff_changes_are_coalesced(self):
        with patch.object(self.window, '_on_cutoff_changed') as changed:
            for value in [6, 7, 8]:
                self.window.cutoff_spin.setValue(value)
            self.assertEqual(changed.call_count, 0)
            QTest.qWait(350)
            changed.assert_called_once_with(8)

    def test_trim_dialog_exposes_fractional_and_negative_time(self):
        data = self.load()
        dialog = TruncateTimeDialog(self.window, data)
        self.assertAlmostEqual(dialog._start_spins[0].value(), -1)
        self.assertAlmostEqual(dialog._end_spins[0].value(), 2)
        dialog._start_spins[0].setValue(-.755)
        self.assertAlmostEqual(dialog._start_spins[0].value(), -.755)
        dialog.deleteLater()


if __name__ == '__main__':
    unittest.main()
