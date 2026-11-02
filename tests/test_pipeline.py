import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from accdata import AccData
from analysis_pipeline import AnalysisRequest, classify_zone, prepare_analysis, snapshot
from background import Cancelled, CancellationToken
from sci_calculation import ProcessASTM, ProcessGB, sampling_frequency
from test_data import recording


class PipelineTests(unittest.TestCase):
    def request(self, data, cutoff=5):
        return AnalysisRequest(('key',), [(id(data), snapshot(data))], data.filename,
                               cutoff, 'Standard', 'Standard', 'Standard', .05, {}, .2)

    def test_preparation_matches_direct_sustained_calculations(self):
        data = AccData(recording(fs=250, seconds=3), is_raw=True, path='a.sup')
        result = prepare_analysis(self.request(data), CancellationToken(), lambda *args: None)
        for duration, values in result['astm'].items():
            self.assertEqual(values, ProcessASTM(data.std_data, max(1, round(250 * duration))))
        for duration, values in result['gb'].items():
            self.assertEqual(values, ProcessGB(data.std_data, max(1, round(250 * duration))))

    def test_snapshot_filtering_does_not_mutate_live_model(self):
        data = AccData(recording(), is_raw=True, path='a.sup')
        original, standard = data.data.copy(), data.std_data
        result = prepare_analysis(self.request(data, cutoff=9), CancellationToken(), lambda *args: None)
        pd.testing.assert_frame_equal(original, data.data)
        self.assertIs(data.std_data, standard)
        self.assertEqual(data._filtered_cutoff, 5)
        self.assertEqual(result['selected']._filtered_cutoff, 9)
        result['selected'].data.iloc[0, 1] = 123
        self.assertNotEqual(data.data.iloc[0, 1], 123)

    def test_noninteger_acquisition_rate_keeps_timestamp_based_window_sizes(self):
        data = AccData(recording(fs=333.5, seconds=15), is_raw=True, path='a.sup')
        request = self.request(data)
        with patch('analysis_pipeline.ProcessASTM', wraps=ProcessASTM) as process:
            result = prepare_analysis(request, CancellationToken(), lambda *args: None)
        expected = round(sampling_frequency(result['frame']) * 14)
        self.assertIn(expected, [call.args[1] for call in process.call_args_list])

    def test_cancelled_request_stops_before_calculating(self):
        data = AccData(recording(), is_raw=True, path='a.sup')
        token = CancellationToken()
        token.cancel()
        with patch('analysis_pipeline.ProcessASTM') as process, self.assertRaises(Cancelled):
            prepare_analysis(self.request(data), token, lambda *args: None)
        process.assert_not_called()

    def test_zone_run_duration_is_strict_and_uses_source_time(self):
        polygon = [[-2, -2], [2, -2], [2, 2], [-2, 2], [-2, -2]]
        frame = pd.DataFrame({'Time': [-1, -.9, -.8], 'X': [0, 0, 0],
                              'Y': [0, 0, 0], 'Z': [1, 1, 1]})
        self.assertEqual(classify_zone(frame, {5: [polygon]}, .21), (1, None, None))
        self.assertEqual(classify_zone(frame, {5: [polygon]}, .19), (5, -1, -.8))

    def test_zone_accumulation_combines_outer_zone_membership(self):
        left = [[-2, -2], [0, -2], [0, 2], [-2, 2], [-2, -2]]
        right = [[0, -2], [2, -2], [2, 2], [0, 2], [0, -2]]
        frame = pd.DataFrame({'Time': np.arange(6) / 10, 'X': [-1, -1, 1, 1, -1, -1],
                              'Y': np.zeros(6), 'Z': np.ones(6)})
        self.assertEqual(classify_zone(frame, {5: [left], 4: [right]}, .2), (4, 0, .5))


if __name__ == '__main__':
    unittest.main()
