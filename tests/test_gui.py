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

    def test_recording_colour_applies_to_every_view_and_survives_refresh(self):
        self.load('one.sup')
        self.load('two.sup')
        self.refresh()
        other_colour = self.window.dataset_colors['two.sup']
        for index in range(1, 7):
            self.render_page(index)
        canvas = self.window.canvasASTMSust
        measured = list(canvas._data_series)
        points = [s.points() for s in measured]
        limits = [(s, s.pen().color().name()) for s in canvas._ref_series]
        canvas._axes['ay']['x'].setRange(.1, 2.)
        self.window.set_dataset_color('one.sup', '#b04baf')

        def assert_colours(expected):
            for view in (canvas, self.window.canvasASTMEgg, self.window.canvasGB):
                self.assertTrue(view._data_series)
                for line in view._data_series:
                    self.assertEqual(line.pen().color().name(), expected)
            self.assertEqual(self.window.canvasAccZone._data_line.pen().color().name(), expected)
            for view in (self.window.canvasZReversal, self.window.canvasXYReversal):
                for line in view._data_series.values():
                    self.assertEqual(line.pen().color().name(), expected)
            for line in self.window.trendplot._series_by_dataset[1]:
                self.assertEqual(line.pen().color().name(), '#b04baf')
            for line in self.window.trendplot._series_by_dataset[2]:
                self.assertEqual(line.pen().color().name(), other_colour)

        assert_colours('#b04baf')
        self.assertEqual([s.points() for s in measured], points)
        self.assertEqual((canvas._axes['ay']['x'].min(), canvas._axes['ay']['x'].max()), (.1, 2.))
        self.assertTrue(all(s.pen().color().name() == colour for s, colour in limits))
        negative = next(s for s in measured if s.name() == 'Measured ay (-)')
        self.assertEqual(negative.pen().style(), Qt.PenStyle.DashLine)

        # Each dataset keeps its colour when selected, recalculated, or themed.
        for name, expected in [('two.sup', other_colour), ('one.sup', '#b04baf')]:
            self.window.select_data.setCurrentText(name)
            self.window.cutoff_spin.setValue(6 if name == 'two.sup' else 7)
            self.wait_for_idle()
            for index in range(1, 7):
                self.render_page(index)
            theme_manager.set_mode('dark')
            assert_colours(expected)

    def test_colour_swatch_accept_and_cancel_preserve_recording_selection(self):
        from PySide6.QtGui import QColor
        from PySide6.QtWidgets import QDialog
        self.load('one.sup')
        self.load('two.sup')
        self.refresh()
        row = self.window.sidebar._cards[1]
        first_colour = self.window.dataset_colors['one.sup']
        with patch('sidebar.QColorDialog') as picker:
            picker.ColorDialogOption.ShowAlphaChannel = 1
            picker.DialogCode.Accepted = QDialog.DialogCode.Accepted
            picker.return_value.exec.return_value = QDialog.DialogCode.Accepted
            picker.return_value.currentColor.return_value = QColor('#c76e19')
            QTest.mouseClick(row._indicator, Qt.MouseButton.LeftButton)
            picker.assert_called_once()
        self.assertEqual(self.window.dataset_colors['two.sup'], '#c76e19')
        self.assertEqual(self.window.dataset_colors['one.sup'], first_colour)
        self.assertEqual(self.window.select_data.currentText(), 'one.sup')
        with patch('sidebar.QColorDialog') as picker:
            picker.ColorDialogOption.ShowAlphaChannel = 1
            picker.DialogCode.Accepted = QDialog.DialogCode.Accepted
            picker.return_value.exec.return_value = QDialog.DialogCode.Rejected
            QTest.mouseClick(row._indicator, Qt.MouseButton.LeftButton)
        self.assertEqual(self.window.dataset_colors['two.sup'], '#c76e19')

    def test_word_export_uses_each_recordings_chosen_colour(self):
        from io import BytesIO
        from pathlib import Path
        from tempfile import TemporaryDirectory
        import numpy as np
        from docx import Document
        from PIL import Image
        from dialogs import ReportDialog

        colours = [('one.sup', '#b04baf'), ('two.sup', '#c76e19')]
        for name, colour in colours:
            self.load(name)
            self.window.set_dataset_color(name, colour)
        self.refresh()
        self.window.select_restraint.setCurrentText('Upper Body')
        self.wait_for_idle()
        dialog = ReportDialog(self.window)
        for checkbox in dialog._checkboxes:
            checkbox.setChecked(True)
        for key, checkbox in dialog._module_boxes.items():
            checkbox.setChecked(key == 'sust')
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'colours.docx'
            with patch('dialogs.QFileDialog.getSaveFileName', return_value=(str(path), '')):
                dialog._on_ok()
            doc = Document(path)
            self.assertEqual(len(doc.inline_shapes), 6)
            for index, shape in enumerate(doc.inline_shapes):
                colour = colours[index // 3][1]
                rgb = tuple(int(colour[i:i + 2], 16) for i in (1, 3, 5))
                relationship = shape._inline.graphic.graphicData.pic.blipFill.blip.embed
                blob = doc.part.related_parts[relationship].blob
                pixels = np.asarray(Image.open(BytesIO(blob)).convert('RGB'))
                self.assertGreater(np.count_nonzero(np.all(pixels == rgb, axis=2)), 50,
                                   f'Figure {index + 1} lost its recording colour')
        self.wait_for_idle()
        self.assertEqual(self.window.select_data.currentText(), 'one.sup')

    def test_signed_lateral_sustained_curves_and_verdicts(self):
        from contextlib import nullcontext
        canvas = self.window.canvasASTMSust
        # Exercise both the synchronous report path and the prepared GUI path.
        for cached, right, left, restraint, failing in [
                (False, 1.25, -4.6, 'None', '−'),
                (True, 4.25, -.8, 'Upper Body', '+')]:
            with self.subTest(cached=cached, failing=failing):
                frame = recording(fs=100, seconds=6)
                frame['X'], frame['Y'], frame['Z'] = 0., 0., 1.
                frame.loc[frame['Time'].between(1, 2), 'Y'] = right
                frame.loc[frame['Time'].between(3, 4), 'Y'] = left
                windows = ({dt: ProcessASTM(frame, max(1, round(dt * 100)))
                            for dt in canvas._SUST_DTS} if cached else None)
                with (patch('chart_astm.ProcessASTM', side_effect=AssertionError('Unexpected recalculation'))
                      if cached else nullcontext()):
                    canvas.addplotxyz(frame, restraint, 'E-Stop', 60, sustained=windows)
                chart = canvas._charts['ay']
                lines = {s.name(): s for s in chart.series()}
                for suffix, expected in [('+', right), ('-', left)]:
                    pts = {p.x(): p.y() for p in lines[f'Measured ay ({suffix})'].points()}
                    self.assertEqual(pts[.2], expected)
                self.assertEqual(lines['ASTM ay (+)'].at(0).y(), 3.)
                self.assertEqual(lines['ASTM ay (-)'].at(0).y(), -3.)
                if restraint != 'None':
                    for label, sign in [('+', 1), ('-', -1)]:
                        # Upper Body: 2.4 g at 0.2 s, increased by 1.25 for E-Stop.
                        self.assertAlmostEqual(lines[f'{restraint} ({label})'].at(1).y(),
                                               sign * 2.4 * 1.25)
                canvas.toggle_sust_analysis(True, restraint, 'E-Stop', 60)
                self.assertEqual(canvas.sust_failed_rules, {f'ay ({failing})'})
                markers = [m.at(0) for m in canvas._sust_analysis_refs if m.chart() == chart]
                self.assertTrue(any(p.x() == .2 for p in markers))
                self.assertTrue(all(p.y() == (left if failing == '−' else right) for p in markers))
                canvas.reset_view()
                limits = canvas._axes['ay']['y']
                self.assertLess(limits.min(), left)
                self.assertGreater(limits.max(), right)

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

    def test_find_reopens_library_without_changing_active_recording(self):
        self.load('one.sup')
        self.load('two.sup')
        original_width = self.window.stackedWidget.width()
        self.window.sidebar._search.setText('one')
        self.window._library_toggle.click()
        self.app.processEvents()
        self.assertTrue(self.window._library_host.isHidden())
        self.assertGreater(self.window.stackedWidget.width(), original_width)
        QTest.keyClick(self.window, Qt.Key.Key_F, Qt.KeyboardModifier.ControlModifier)
        self.app.processEvents()
        self.assertTrue(self.window._library_host.isVisible())
        self.assertTrue(self.window.sidebar._search.hasFocus())
        self.assertEqual(self.window.sidebar._search.selectedText(), 'one')
        self.assertEqual(self.window.select_data.currentText(), 'one.sup')

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

    def test_rail_navigation_and_settings_keep_the_current_view(self):
        from PySide6.QtWidgets import QToolButton
        self.window._nav_buttons[3].click()
        self.assertEqual(self.window._view_heading.text(), 'GB Contour')
        settings = self.window.findChild(QToolButton, 'railSettings')
        settings.click()
        self.assertTrue(self.window.select_cond.isVisible())
        settings.click()
        self.assertEqual(self.window.stackedWidget.currentIndex(), 3)
        self.assertTrue(self.window._nav_buttons[3].isChecked())
        self.window.stackedWidget.setCurrentIndex(6)
        self.assertTrue(self.window._nav_buttons[6].isChecked())

    def test_time_shift_disclosure_and_library_preference(self):
        self.assertFalse(self.window._library_host.isVisible())
        self.load('one.sup')
        self.assertTrue(self.window._library_host.isVisible())
        self.assertFalse(self.window._shift_tools.isVisible())
        self.window._shift_disclosure.trigger()
        self.assertTrue(self.window._shift_spin.isVisible())
        self.window._library_toggle.click()
        self.load('two.sup')
        self.assertFalse(self.window._library_host.isVisible())

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
