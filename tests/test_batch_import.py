"""Actual raw/SUP batches, failure isolation and incremental GUI commits."""
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import test_gui as gui
from test_data import recording
from accdata import AccData
from analysis_pipeline import ImportBatch, import_recording, import_recordings, prepare_analysis
from background import Cancelled, CancellationToken


def write_raw(path):
    path.write_text('Channel Description,ASTM_X,ASTM_Y,ASTM_Z,GB_X,GB_Y,GB_Z\n'
                    'Time,ASTM_X,ASTM_Y,ASTM_Z,GB_X,GB_Y,GB_Z\n' + ''.join(
                        f'{-1 + index / 200:.4f},0,0,1,0,0,1\n' for index in range(120)),
                    encoding='utf-8')
    return path


def write_sup(path):
    AccData(recording(fs=200, seconds=1), is_raw=True, path=path.name).export_to_file(path)
    return path


class BatchReaderTests(unittest.TestCase):
    def test_multiple_raw_files_keep_both_groups_and_report_batch_progress(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = [write_raw(Path(folder) / f'{index}.csv') for index in range(2)]
            progress = []
            result = import_recordings(paths, True, 9, CancellationToken(),
                                       lambda value, message: progress.append((value, message)))
        self.assertEqual(result.total, 2)
        self.assertEqual(result.errors, [])
        self.assertEqual([len(signals) for _, signals in result.recordings], [2, 2])
        self.assertTrue(all(data._filtered_cutoff == 9 and data.fs == 200
                            for _, signals in result.recordings for data in signals))
        self.assertEqual([value for value, _ in progress], sorted(value for value, _ in progress))
        self.assertEqual(progress[-1][0], 100)
        self.assertTrue(any('File 2/2' in message for _, message in progress))

    def test_bad_sup_file_does_not_stop_later_files(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = [write_sup(Path(folder) / 'one.sup'), Path(folder) / 'missing.sup',
                     write_sup(Path(folder) / 'two.sup')]
            result = import_recordings(paths, False, 5, CancellationToken(), lambda *args: None)
        self.assertEqual([signals[0].filename for _, signals in result.recordings], ['one.sup', 'two.sup'])
        self.assertEqual(len(result.errors), 1)
        self.assertTrue(result.errors[0][0].endswith('missing.sup'))

    def test_cancel_between_files_is_not_treated_as_a_file_error(self):
        token = CancellationToken()
        with tempfile.TemporaryDirectory() as folder:
            paths = [write_sup(Path(folder) / f'{index}.sup') for index in range(2)]
            def progress(value, message):
                if message.startswith('Checked file 1/2'):
                    token.cancel()
            with patch('analysis_pipeline.import_recording', wraps=import_recording) as reader, \
                 self.assertRaises(Cancelled):
                import_recordings(paths, False, 5, token, progress)
            self.assertEqual(reader.call_count, 1)


class BatchGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        gui.GuiTests.setUpClass.__func__(cls)

    setUp = gui.GuiTests.setUp
    tearDown = gui.GuiTests.tearDown
    load = gui.GuiTests.load
    wait_for_idle = gui.GuiTests.wait_for_idle

    def test_both_file_dialogs_select_multiple_files(self):
        for method, extension, raw in [(self.window.open_raw_data, 'csv', True),
                                        (self.window.open_accel_data, 'sup', False)]:
            paths = [f'C:/data/one.{extension}', f'C:/data/two.{extension}']
            with patch('main_window.QFileDialog.getOpenFileNames', return_value=(paths, '')), \
                 patch.object(self.window, '_start_import') as start:
                method()
            start.assert_called_once_with(paths, raw=raw)

    def test_cancelled_file_dialog_starts_no_import(self):
        with patch('main_window.QFileDialog.getOpenFileNames', return_value=([], '')), \
             patch.object(self.window, '_start_import') as start:
            self.window.open_raw_data()
            self.window.open_accel_data()
        start.assert_not_called()

    def test_sup_batch_keeps_good_files_and_reports_failures_once(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = [write_sup(Path(folder) / 'one.sup'), Path(folder) / 'bad.sup',
                     write_sup(Path(folder) / 'two.sup')]
            with patch('qfluentwidgets.InfoBar.warning') as warning, \
                 patch('processing_controller.prepare_analysis', wraps=prepare_analysis) as prepare:
                self.window._start_import(paths, raw=False)
                self.wait_for_idle()
            warning.assert_called_once()
            self.assertIn('bad.sup', warning.call_args.kwargs['content'])
            self.assertEqual(prepare.call_count, 1)
        self.assertEqual(self.window.data_namelist, ['one.sup', 'two.sup'])
        self.assertEqual(len(self.window.sidebar._cards), 2)
        self.assertEqual(len(self.window.trendplot._series_list), 6)
        self.assertIn('Imported 2/3 files', self.window.statusBar().currentMessage())

    def test_raw_batch_imports_four_distinct_signals(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = [write_raw(Path(folder) / f'{index}.csv') for index in range(2)]
            self.window._start_import(paths, raw=True)
            self.wait_for_idle()
        self.assertEqual(self.window.data_namelist,
                         ['(GB)0.csv', '(ASTM)0.csv', '(GB)1.csv', '(ASTM)1.csv'])
        self.assertEqual(len(set(self.window.dataset_colors.values())), 4)
        self.assertEqual(len(self.window.trendplot._series_list), 12)
        self.assertEqual(self.window._prepared['key'], self.window._current_analysis_key())

    def test_duplicate_basenames_remain_independently_selectable(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = []
            for directory in ('a', 'b'):
                target = Path(folder) / directory
                target.mkdir()
                paths.append(write_sup(target / 'same.sup'))
            self.window._start_import(paths, raw=False)
            self.wait_for_idle()
        self.assertEqual(self.window.data_namelist, ['same.sup', 'same (2).sup'])
        self.window.select_data.setCurrentText('same (2).sup')
        self.wait_for_idle()
        self.assertEqual(self.window._prepared['selected'].filename, 'same (2).sup')

    def test_all_failed_batch_leaves_existing_workspace_unchanged(self):
        existing = self.load()
        with tempfile.TemporaryDirectory() as folder, patch('qfluentwidgets.InfoBar.warning') as warning:
            self.window._start_import([Path(folder) / 'a.sup', Path(folder) / 'b.sup'], raw=False)
            self.wait_for_idle()
        self.assertEqual(self.window.datalist, [existing])
        self.assertEqual(len(self.window._last_import_summary['errors']), 2)
        self.assertIn('Imported 0/2 files', self.window.statusBar().currentMessage())
        warning.assert_called_once()
        self.assertTrue(self.window.sidebar._import_button.isEnabled())

    def test_cancel_during_gui_commit_keeps_only_completed_files(self):
        self.load('existing.sup')
        signals = [AccData(recording(), is_raw=True, path=f'{index}.sup') for index in range(3)]
        result = ImportBatch(3, [(f'{index}.sup', [data]) for index, data in enumerate(signals)], [])
        self.window._accept_import_batch(result)
        self.window._import_commit_timer.stop()
        self.window._commit_import_file()
        self.window.cancel_processing()
        self.wait_for_idle()
        self.assertEqual(self.window.data_namelist, ['existing.sup', '0.sup'])
        self.assertIsNone(self.window._import_commit)
        self.assertIn('Kept 1 imported file', self.window.statusBar().currentMessage())

    def test_cancel_during_batch_parsing_never_commits_a_late_result(self):
        entered, release = threading.Event(), threading.Event()
        signal = AccData(recording(), is_raw=True, path='late.sup')
        def reader(paths, raw, cutoff, token, progress):
            entered.set()
            release.wait(5)
            return ImportBatch(2, [('late.sup', [signal])], [])
        try:
            with patch('processing_controller.import_recordings', side_effect=reader):
                self.window._start_import(['one.sup', 'two.sup'], raw=False)
                self.assertTrue(entered.wait(2))
                self.window.cancel_processing()
                release.set()
                self.wait_for_idle()
            self.assertFalse(self.window.datalist)
        finally:
            release.set()

    def test_report_pauses_batch_commit_until_capture_finishes(self):
        signals = [AccData(recording(), is_raw=True, path=f'{index}.sup') for index in range(3)]
        result = ImportBatch(3, [(f'{index}.sup', [data]) for index, data in enumerate(signals)], [])
        self.window._accept_import_batch(result)
        self.window._import_commit_timer.stop()
        self.window._commit_import_file()
        self.window._begin_report_processing()
        self.app.processEvents()
        self.assertEqual(len(self.window.datalist), 1)
        self.assertFalse(self.window._import_commit_timer.isActive())
        self.window._end_report_processing()
        self.wait_for_idle()
        self.assertEqual(len(self.window.datalist), 3)
        self.assertEqual(self.window._processing_state, 'ready')

    def test_clear_all_stops_queued_batch_additions(self):
        signals = [AccData(recording(), is_raw=True, path=f'{index}.sup') for index in range(2)]
        result = ImportBatch(2, [(f'{index}.sup', [data]) for index, data in enumerate(signals)], [])
        self.window._accept_import_batch(result)
        self.window._import_commit_timer.stop()
        self.window._commit_import_file()
        self.window.reset_trend(reset_datalist=True)
        self.wait_for_idle()
        self.assertFalse(self.window.datalist)
        self.assertIsNone(self.window._import_commit)
        self.assertEqual(self.window._processing_state, 'empty')

    def test_cancel_discards_import_deferred_during_report(self):
        existing = self.load('existing.sup')
        data = AccData(recording(), is_raw=True, path='deferred.sup')
        self.window._begin_report_processing()
        self.window._background_result('import', ImportBatch(1, [('deferred.sup', [data])], []))
        self.window.cancel_processing()
        self.window._end_report_processing()
        self.wait_for_idle()
        self.assertEqual(self.window.datalist, [existing])

    def test_cancel_releases_uncommitted_recording_buffers(self):
        import gc
        import weakref
        data = AccData(recording(), is_raw=True, path='cancelled.sup')
        reference = weakref.ref(data)
        batch = ImportBatch(1, [('cancelled.sup', [data])], [])
        self.window._accept_import_batch(batch)
        self.window._import_commit_timer.stop()
        del data, batch
        self.window.cancel_processing()
        gc.collect()
        self.assertIsNone(reference())


if __name__ == '__main__':
    unittest.main()
