"""Worker isolation, freshness, cancellation and delivery on the GUI thread."""
import threading
import unittest
from unittest.mock import patch

from PySide6.QtCore import QEventLoop, QThread, QTimer

import test_gui as gui
from analysis_pipeline import prepare_analysis
from accdata import AccData
from sci_calculation import allowance_at_t0
from test_data import recording


class ProcessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gui.GuiTests.setUpClass.__func__(cls)

    setUp = gui.GuiTests.setUp
    tearDown = gui.GuiTests.tearDown
    load = gui.GuiTests.load
    wait_for_idle = gui.GuiTests.wait_for_idle
    refresh = gui.GuiTests.refresh
    render_page = gui.GuiTests.render_page

    def spin_until(self, predicate, timeout=5000):
        loop = QEventLoop()
        poll = QTimer(loop)
        poll.setInterval(10)
        poll.timeout.connect(lambda: loop.quit() if predicate() else None)
        deadline = QTimer(loop)
        deadline.setSingleShot(True)
        deadline.timeout.connect(loop.quit)
        poll.start()
        deadline.start(timeout)
        loop.exec()
        self.assertTrue(predicate(), 'Timed out waiting for worker state')

    def test_calculations_run_on_worker_and_chart_mutations_on_gui(self):
        self.load()
        self.window.stackedWidget.setCurrentIndex(3)
        calculation_threads, rendering_threads = [], []
        draw = self.window.canvasGB.addplotxyz
        def prepare(*args):
            calculation_threads.append(QThread.currentThread())
            return prepare_analysis(*args)
        def render(*args, **kwargs):
            rendering_threads.append(QThread.currentThread())
            return draw(*args, **kwargs)
        with patch('processing_controller.prepare_analysis', side_effect=prepare), \
             patch.object(self.window.canvasGB, 'addplotxyz', side_effect=render), \
             patch('chart_gb.ProcessGB', side_effect=AssertionError('GUI recalculated GB')):
            self.refresh()
        self.assertTrue(calculation_threads)
        self.assertTrue(all(thread != self.app.thread() for thread in calculation_threads))
        self.assertEqual(rendering_threads, [self.app.thread()])

    def test_selection_refreshes_automatically_and_hides_old_verdict(self):
        self.load('one.sup')
        self.load('two.sup')
        self.render_page(1)
        self.refresh()
        self.window._analyze_btn.setChecked(True)
        self.window.select_data.setCurrentText('two.sup')
        page = self.window.stackedWidget.currentWidget()
        self.assertTrue(page._pending_label.isVisible())
        self.assertFalse(page._chart_content.isVisible())
        self.assertFalse(self.window._sust_summary_host.isVisible())
        self.wait_for_idle()
        self.assertEqual(self.window._prepared['selected'].filename, 'two.sup')
        self.assertTrue(self.window._page_is_current())
        self.assertTrue(self.window.canvasASTMSust._sust_analysis_active)

    def test_late_result_cannot_overwrite_new_selection(self):
        self.load('one.sup')
        self.load('two.sup')
        entered, release = threading.Event(), threading.Event()
        def delayed(request, token, progress):
            result = prepare_analysis(request, token, progress)
            if request.selected == 'one.sup':
                entered.set()
                release.wait(5)  # Simulate a non-interruptible library call.
            return result
        try:
            with patch('processing_controller.prepare_analysis', side_effect=delayed):
                self.window.request_chart_update(force=True)
                self.assertTrue(entered.wait(2))
                self.window.select_data.setCurrentText('two.sup')
                self.spin_until(lambda: bool(self.window._prepared and
                    self.window._prepared['selected'].filename == 'two.sup'))
                release.set()
                self.wait_for_idle()
            self.assertEqual(self.window._prepared['selected'].filename, 'two.sup')
            self.assertEqual(self.window._prepared['key'], self.window._current_analysis_key())
        finally:
            release.set()

    def test_settings_changes_coalesce_and_use_latest_allowances(self):
        self.load()
        self.render_page(6)
        with patch('processing_controller.prepare_analysis', wraps=prepare_analysis) as prepare:
            self.window.height_input.setValue(30)
            self.window.height_input.setValue(55)
            self.window.select_restraint.setCurrentText('Upper Body')
            self.window.select_cond.setCurrentText('E-Stop')
            self.window._xr_hyst.setValue(.2)
            self.wait_for_idle()
        self.assertEqual(prepare.call_count, 1)
        self.assertEqual(self.window.canvasXYReversal._allowance['x'],
                         allowance_at_t0('x', 'Upper Body', 'E-Stop', 55))
        self.assertTrue(self.window._page_is_current())

    def test_cutoff_failure_never_applies_partially_filtered_snapshots(self):
        data = self.load()
        self.render_page(3)
        self.refresh()
        before = data.std_data
        def fail(request, token, progress):
            request.datasets[0][1].filter_data(request.cutoff)
            raise ValueError('calculation failed')
        with patch('processing_controller.prepare_analysis', side_effect=fail):
            self.window.cutoff_spin.setValue(9)
            self.window.request_chart_update(force=True)
            self.wait_for_idle()
        self.assertIs(data.std_data, before)
        self.assertEqual(data._filtered_cutoff, 5)
        self.assertEqual(self.window._processing_state, 'error')
        self.assertFalse(self.window._page_is_current())
        self.refresh()
        self.assertEqual(data._filtered_cutoff, 9)
        self.assertTrue(self.window._page_is_current())

    def test_pending_filter_hides_old_inspector_statistics(self):
        self.load()
        self.refresh()
        self.window.cutoff_spin.setValue(9)
        self.assertFalse(self.window.sidebar._stats_panel.isVisible())
        self.window.cancel_processing()
        self.assertFalse(self.window.sidebar._stats_panel.isVisible())
        self.refresh()
        self.assertTrue(self.window.sidebar._stats_panel.isVisible())

    def test_prepared_refresh_repairs_duplicate_or_missing_trend_series(self):
        self.load()
        self.window.replot_all()
        self.assertEqual(len(self.window.trendplot._series_list), 6)
        self.refresh()
        self.assertEqual(len(self.window.trendplot._series_list), 3)
        self.window.trendplot.initplotxyz()
        self.refresh()
        self.assertEqual(len(self.window.trendplot._series_list), 3)

    def test_gui_commit_errors_are_reported(self):
        with patch.object(self.window, '_accept_import', side_effect=ValueError('cannot display data')):
            self.window._background_result('import', [])
        self.assertIn('cannot display data', self.window.statusBar().currentMessage())
        self.assertEqual(self.window._processing_state, 'error')

    def test_filtered_export_waits_for_requested_cutoff(self):
        from dialogs import ExportDialog
        data = self.load()
        self.window.cutoff_spin.setValue(9)
        dialog = ExportDialog(self.window, data)
        dialog._filename.setText('export')
        dialog._export_type.setCurrentText('Filtered Clean Data')
        with patch('dialogs.qfw.InfoBar.warning') as warning, \
             patch('dialogs.QFileDialog.getSaveFileName') as save:
            dialog._on_ok()
        warning.assert_called_once()
        save.assert_not_called()
        dialog.deleteLater()

    def test_cancel_pending_update_stays_cancelled_until_requested(self):
        self.load()
        self.window.select_cond.setCurrentText('E-Stop')
        self.window.cancel_processing()
        self.render_page(3)
        self.wait_for_idle()
        self.assertFalse(self.window._jobs.busy())
        self.assertIsNone(self.window._prepared)
        self.assertEqual(self.window._processing_state, 'cancelled')
        self.refresh()
        self.assertTrue(self.window._page_is_current())

    def test_cancelled_import_never_adds_recordings(self):
        entered, release = threading.Event(), threading.Event()
        imported = AccData(recording(), is_raw=True, path='cancelled.sup')
        def importer(path, raw, cutoff, token, progress):
            entered.set()
            release.wait(5)
            return [imported]
        try:
            with patch('processing_controller.import_recording', side_effect=importer):
                self.window._start_import('unused', raw=False)
                self.assertTrue(entered.wait(2))
                self.window.cancel_processing()
                release.set()
                self.wait_for_idle()
            self.assertEqual(self.window.datalist, [])
            self.assertEqual(self.window.sidebar._cards, [])
        finally:
            release.set()

    def test_import_errors_are_visible_and_do_not_create_partial_data(self):
        with patch('processing_controller.import_recording', side_effect=ValueError('bad header')):
            self.window._start_import('unused', raw=True)
            self.wait_for_idle()
        self.assertEqual(self.window._processing_state, 'error')
        self.assertIn('bad header', self.window.statusBar().currentMessage())
        self.assertFalse(self.window.datalist)
        self.assertTrue(self.window.sidebar._import_button.isEnabled())

    def test_gui_timer_runs_while_preparation_is_in_progress(self):
        self.load()
        release = threading.Event()
        ticks = []
        timer = QTimer()
        timer.setInterval(5)
        def tick():
            ticks.append(1)
            if len(ticks) >= 5:
                release.set()
        timer.timeout.connect(tick)
        def prepare(request, token, progress):
            while not release.wait(.005):
                token.check()
            return prepare_analysis(request, token, progress)
        try:
            with patch('processing_controller.prepare_analysis', side_effect=prepare):
                timer.start()
                self.refresh()
            self.assertGreaterEqual(len(ticks), 5)
        finally:
            timer.stop()
            release.set()

    def test_clear_all_prevents_late_import_from_reappearing(self):
        entered, release = threading.Event(), threading.Event()
        imported = AccData(recording(), is_raw=True, path='late.sup')
        def importer(path, raw, cutoff, token, progress):
            entered.set()
            release.wait(5)
            return [imported]
        try:
            with patch('processing_controller.import_recording', side_effect=importer):
                self.window._start_import('unused', raw=False)
                self.assertTrue(entered.wait(2))
                self.window.reset_trend(reset_datalist=True)
                release.set()
                self.wait_for_idle()
            self.assertFalse(self.window.datalist)
            self.assertEqual(self.window._processing_state, 'empty')
        finally:
            release.set()

    def test_report_rendering_defers_import_and_automatic_refresh(self):
        self.load('one.sup')
        self.window._begin_report_processing()
        imported = AccData(recording(), is_raw=True, path='deferred.sup')
        self.window._background_result('import', [imported])
        self.assertEqual(len(self.window.datalist), 1)
        self.assertFalse(self.window._refresh_timer.isActive())
        self.window._end_report_processing()
        self.wait_for_idle()
        self.assertEqual(len(self.window.datalist), 2)
        self.assertTrue(self.window._prepared)

    def test_close_waits_for_cancelled_worker_without_destroying_it(self):
        self.load()
        entered, release = threading.Event(), threading.Event()
        def prepare(request, token, progress):
            entered.set()
            release.wait(5)
            token.check()
        try:
            with patch('processing_controller.prepare_analysis', side_effect=prepare):
                self.window.request_chart_update(force=True)
                self.assertTrue(entered.wait(2))
                self.window.close()
                self.assertTrue(self.window._closing)
                release.set()
                self.wait_for_idle()
            self.assertFalse(self.window.isVisible())
        finally:
            release.set()


if __name__ == '__main__':
    unittest.main()
