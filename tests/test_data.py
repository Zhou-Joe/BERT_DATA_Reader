"""Regression coverage for acquisition timing, editing and analysis."""
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from accdata import AccData, RawData, detect_test_boundaries
from sci_calculation import ProcessASTM, ProcessGB, detect_z_acceleration_reversal


def recording(fs=500, start=0.0, seconds=2):
    time = start + np.arange(int(fs * seconds)) / fs
    return pd.DataFrame({'Time': time, 'X': np.sin(time * 7),
                         'Y': np.cos(time * 4), 'Z': 1 + np.sin(time * 3)})


class DataTests(unittest.TestCase):
    def test_csv_preserves_time_rate_and_reordered_axes(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'reordered.csv'
            path.write_text('Sample Rate (Hz),200,200,200,200,200,200\n'
                            'Channel Description,GB_Z,ASTM_Y,GB_X,ASTM_Z,GB_Y,ASTM_X\n'
                            'Time,Z,Y,X,Z,Y,X\n' + ''.join(
                                f'{-1 + i / 200:.4f},3,5,1,6,2,4\n'
                                for i in range(400)), encoding='utf-8')
            gb, astm = RawData(path).export_data()
            self.assertEqual(gb.fs, 200)
            self.assertEqual(gb.data.iloc[0, 0], -1)
            np.testing.assert_allclose(gb.data.iloc[0, 1:], [1, 2, 3])
            np.testing.assert_allclose(astm.data.iloc[0, 1:], [4, 5, 6])

    def test_tab_sample_rate_without_time(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'six.txt'
            path.write_text('Sample Rate (Hz)\t250\n' + '1\t2\t3\t4\t5\t6\n' * 500)
            gb, _ = RawData(path).export_data()
            self.assertEqual(gb.fs, 250)
            self.assertAlmostEqual(gb.data['Time'].iloc[-1], 1.996)

    def test_csv_rejects_missing_channels_and_nonfinite_samples(self):
        with tempfile.TemporaryDirectory() as folder:
            for name, body in [('short', '0,1,2,3\n.01,1,2,3\n'),
                               ('invalid', '0,1,2,3,4,5,6\n.01,1,2,3,4,5,inf\n')]:
                path = Path(folder) / f'{name}.csv'
                path.write_text(body)
                with self.assertRaises(RuntimeError):
                    RawData(path)

    def test_export_roundtrip_preserves_filter_and_pretrigger_samples(self):
        data = AccData(recording(start=-1), is_raw=True, path='a.sup', cutoff=7)
        data.set_angle(5.5, 2, 0, 0)
        data.filter_data(7)
        before = data.std_data.copy()
        with tempfile.TemporaryDirectory() as folder, warnings.catch_warnings():
            warnings.simplefilter('error', FutureWarning)
            path = Path(folder) / 'roundtrip.sup'
            data.export_to_file(path)
            loaded = AccData(path, cutoff=7)
        self.assertEqual(data._filtered_cutoff, 7)
        pd.testing.assert_frame_equal(data.std_data, before)
        np.testing.assert_allclose(loaded.data, data.data)
        self.assertEqual(loaded.pitch_angle, 5.5)
        self.assertEqual(loaded.data['Time'].iloc[0], -1)

    def test_fractional_truncate_uses_timestamp_and_drops_old_filter_rows(self):
        data = AccData(recording(start=-1), is_raw=True, path='a.sup')
        original = data.data.copy()
        data.truncate_data(-.75, -.25, cutoff=9)
        self.assertEqual(len(data.data), 250)
        np.testing.assert_allclose(data.data['X'], original['X'].iloc[125:375])
        self.assertAlmostEqual(data.data['Time'].iloc[0], 0)
        self.assertAlmostEqual(data.data['Time'].iloc[-1], .498)
        self.assertEqual(len(data.filtered_data), 250)
        self.assertEqual(len(data.std_data), 250)

    def test_invalid_truncate_leaves_source_unchanged(self):
        data = AccData(recording(), is_raw=True, path='a.sup')
        before = data.data.copy()
        with self.assertRaises(ValueError):
            data.truncate_data(10, 11, cutoff=5)
        pd.testing.assert_frame_equal(before, data.data)

    def test_peak_times_come_from_time_column(self):
        frame = recording(start=10)
        data = AccData(frame, is_raw=True, path='a.sup')
        stats = data.get_data_stats(data.data)
        self.assertEqual(stats.loc['X', 't (Max)'], frame.loc[frame['X'].idxmax(), 'Time'])

    def test_edit_retains_numeric_channels(self):
        data = AccData(recording(), is_raw=True, path='a.sup')
        before = data.data.iloc[:, 1:].to_numpy(copy=True)
        data.edit_data('multiply', [2, 3, 4], cutoff=5)
        np.testing.assert_allclose(data.data.iloc[:, 1:], np.round(before * [2, 3, 4], 4))
        self.assertTrue(all(pd.api.types.is_numeric_dtype(dtype) for dtype in data.data.dtypes))

    def test_filter_cache_and_acczone_do_not_change_other_views(self):
        data = AccData(recording(), is_raw=True, path='a.sup', cutoff=7)
        before = data.std_data
        data.filter_data(7)
        self.assertIs(data.std_data, before)
        data.get_1hz_standard_frame()
        self.assertIs(data.std_data, before)
        self.assertEqual(data._filtered_cutoff, 7)

    def test_sustained_values_match_brute_force(self):
        rng = np.random.default_rng(42)
        frame = pd.DataFrame(np.column_stack([np.arange(50) / 500,
                                              rng.normal(size=(50, 3))]))
        for window in [1, 5, 50, 51]:
            expected_pos, expected_neg = [], []
            for axis, floor in zip(range(1, 4), [0, 0, 1]):
                values = frame.iloc[:, axis].to_numpy()
                spans = [values[i:i + window] for i in range(len(values) - window + 1)]
                expected_pos.append(round(max([floor] + [float(v.min()) for v in spans]), 4))
                expected_neg.append(round(min([floor] + [float(v.max()) for v in spans]), 4))
            pos, neg = ProcessASTM(frame, window)
            self.assertEqual(pos, expected_pos)
            self.assertEqual(neg, expected_neg)
            self.assertEqual(ProcessGB(frame, window), (pos[1:], neg[1:]))

    def test_z_reversal_requires_full_pre_window(self):
        fs = 500
        early = np.concatenate([[-1.0] * 5, [3.0] * 75])
        self.assertEqual(detect_z_acceleration_reversal(early, np.arange(len(early)) / fs, fs), [])
        valid = np.concatenate([[-1.0] * 150, [3.0] * 100])
        events = detect_z_acceleration_reversal(valid, np.arange(len(valid)) / fs, fs)
        self.assertTrue(events)
        self.assertTrue(all(event['window_start_idx'] >= 100 for event in events))

    def test_detected_boundary_is_exclusive_and_preserves_origin(self):
        frame = recording(start=-10, seconds=5)
        data = AccData(frame, is_raw=True, path='a.sup')
        intervals = detect_test_boundaries(data)
        self.assertEqual(len(intervals), 1)
        self.assertAlmostEqual(intervals[0][0], -10)
        self.assertAlmostEqual(intervals[0][1], -5)


if __name__ == '__main__':
    unittest.main()
