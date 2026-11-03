import tempfile
import unittest
from pathlib import Path

import numpy as np

from accdata import AccData, SensorConnectData, detect_test_boundaries
from analysis_pipeline import import_recordings
from background import CancellationToken
from test_data import recording
from sci_calculation import sampling_frequency


class SensorConnectTests(unittest.TestCase):
    def test_legacy_batch_keeps_existing_trims_and_roundtrips_512_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            paths, frames = [], []
            for index in range(3):
                frame = recording(fs=512, seconds=3)
                frame.Time = (frame.Time + .0003 * index).round(4)
                original = AccData(frame, is_raw=True, path=f'lap{index}')
                path = Path(folder) / f'lap{index}.sup'
                original.export_to_file(path)
                paths.append(path)
                frames.append(frame)
            result = import_recordings(paths, 'sensorconnect_sup', 5,
                                       CancellationToken(), lambda *_: None)
            self.assertFalse(result.errors)
            self.assertEqual(len(result.recordings), 3)
            for index, (_, datasets) in enumerate(result.recordings):
                data = datasets[0]
                self.assertEqual(data.fs, 512)
                np.testing.assert_array_equal(data.data.Time, frames[index].Time)
                np.testing.assert_allclose(data.data, frames[index], rtol=0, atol=1e-15)
                for frame in (data.data, data.filtered_data, data.std_data,
                              data.get_1hz_standard_frame()):
                    self.assertEqual(sampling_frequency(frame), 512)
                corrected = Path(folder) / f'corrected{index}.sup'
                data.export_to_file(corrected)
                reloaded = AccData(corrected)
                self.assertEqual(reloaded.fs, 512)
                np.testing.assert_array_equal(reloaded.data.Time, data.data.Time)
                np.testing.assert_allclose(reloaded.data, data.data, rtol=0, atol=1e-15)
                np.testing.assert_array_equal(reloaded.std_data, data.std_data)
                reloaded.truncate_data(.1, 1.1, 5)
                self.assertEqual(reloaded.fs, 512)

    def load_sensor(self, folder, times, rate='512Hz'):
        path = Path(folder) / 'sensor.csv'
        path.write_text('CHANNEL_INFO\nChannel,Type,SampleRate\n'
                        f'1:ch1,Timeseries,{rate}\nDATA_START\n'
                        'Time,1:ch1,1:ch2,1:ch3\n' + ''.join(
                            f'{t:.9f},{np.sin(i / 10)},0,-1\n'
                            for i, t in enumerate(times)), encoding='utf-8')
        return SensorConnectData(path).export_data()[0]

    def test_preserves_precision_and_initial_filters_use_declared_rate(self):
        with tempfile.TemporaryDirectory() as folder:
            times = np.arange(1024) / 512
            data = self.load_sensor(folder, times)
            np.testing.assert_allclose(data.data.Time, times, atol=1e-9, rtol=0)
            self.assertEqual(data.fs, 512)
            filtered = data.filtered_data.copy()
            data._filtered_cutoff = None
            data.filter_data(data.cutoff)
            np.testing.assert_allclose(filtered, data.filtered_data)
            data.truncate_data(.2, 1.2, data.cutoff)
            self.assertEqual(data.fs, 512)

    def test_raw_rate_near_default_is_detected(self):
        data = AccData(recording(fs=512), is_raw=True, path='raw')
        self.assertEqual(data.fs, 512)

    def test_fallback_rate_ignores_initial_duplicate(self):
        with tempfile.TemporaryDirectory() as folder:
            data = self.load_sensor(folder, np.r_[0, np.arange(1024) / 512], '')
            self.assertEqual(data.fs, 512)

    def test_backward_and_duplicate_times_remain_strictly_increasing(self):
        with tempfile.TemporaryDirectory() as folder:
            data = self.load_sensor(folder, [0, .002, .002, .001, .004, .006])
            np.testing.assert_allclose(data.data.Time, [0, .002, .004, .006])

    def test_collapsed_clock_and_infinite_samples_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'increasing timestamps'):
                self.load_sensor(folder, [0, 0, 0])
            with self.assertRaisesRegex(ValueError, 'finite'):
                self.load_sensor(folder, [0, .002, float('inf')])

    def test_sensorconnect_batch_preserves_rate_and_reports_bad_file(self):
        with tempfile.TemporaryDirectory() as folder:
            self.load_sensor(folder, np.arange(1024) / 512)
            path = Path(folder) / 'sensor.csv'
            result = import_recordings([path, Path(folder) / 'missing.csv', path],
                                       'sensorconnect', 10, CancellationToken(),
                                       lambda *_: None)
            self.assertEqual(len(result.recordings), 2)
            self.assertEqual(len(result.errors), 1)
            for _, datasets in result.recordings:
                self.assertEqual(len(datasets), 1)
                self.assertEqual(datasets[0].fs, 512)
                self.assertEqual(datasets[0]._filtered_cutoff, 10)

    def test_auto_trim_splits_clock_jump_with_inverted_rest_orientation(self):
        from types import SimpleNamespace
        import pandas as pd
        local_time = np.arange(1500) / 100
        motion = np.where(local_time < 3, 0, np.sin(local_time * 5))
        frame = pd.DataFrame({'Time': np.r_[local_time, local_time + 100],
                              'X': np.tile(motion + .2, 2), 'Y': 0., 'Z': -1.})
        intervals = detect_test_boundaries(SimpleNamespace(data=frame, fs=100))
        self.assertEqual(len(intervals), 2)
        self.assertAlmostEqual(intervals[0][1], 15)
        self.assertAlmostEqual(intervals[1][0], 100)
        self.assertAlmostEqual(intervals[1][1], 115)
