"""Behavior of automatic test-window proposals in recording time."""
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from accdata import detect_test_boundaries


def propose(time, x, fs=100, **options):
    frame = pd.DataFrame({'Time': time, 'X': x, 'Y': -.02, 'Z': -.99})
    return detect_test_boundaries(SimpleNamespace(data=frame, fs=fs), **options)


class SegmentationTests(unittest.TestCase):
    def test_long_idle_recording_with_isolated_impulses_is_not_a_test(self):
        t = np.arange(10000) / 100
        x = np.zeros(len(t))
        x[400::800] = 8  # one bad reading every 8 s; no 10 s quiet run
        self.assertEqual(propose(t, x), [])

    def test_short_rides_near_both_file_edges_survive_long_stop(self):
        t = np.arange(8000) / 100 - 40
        x = np.where((t < -38) | (t >= 38), np.sin(9 * t), 0)
        intervals = propose(t, x)
        self.assertEqual(len(intervals), 2)
        (start, tail), (head, end) = intervals
        self.assertEqual(start, -40)
        self.assertAlmostEqual(end, 40)
        self.assertTrue(-24 < tail < -22)
        self.assertTrue(22 < head < 24)

    def test_brief_pause_and_a_single_spike_do_not_add_rides(self):
        t = np.arange(16500) / 100
        x = np.where(((t >= 35) & (t < 45)) | ((t >= 48) & (t < 58))
                     | ((t >= 110) & (t < 120)), np.sin(5 * t), 0)
        x[8300] = 12
        intervals = propose(t, x)
        self.assertEqual(len(intervals), 2)
        for (s, e), (first, last) in zip(intervals, [(35, 58), (110, 120)]):
            self.assertLessEqual(s, first)
            self.assertGreaterEqual(e, last)
            self.assertFalse(s <= 83 < e)

    def test_flat_acceleration_at_start_does_not_become_rest_reference(self):
        t = np.arange(5000) / 100
        x = np.where((t < 2) | (t >= 48), .6, 0.)
        intervals = propose(t, x)
        self.assertEqual(len(intervals), 2)
        self.assertEqual(intervals[0][0], 0.)
        self.assertAlmostEqual(intervals[1][1], 50.)
        self.assertFalse(any(s <= 25 < e for s, e in intervals))

    def test_no_motion_on_one_side_of_logger_restart(self):
        t = np.r_[np.arange(3500) / 100, 80 + np.arange(2000) / 100]
        x = np.where(t >= 85, np.sin(5 * t), 0)
        self.assertEqual(propose(t, x), [(80, 100)])

    def test_lone_reading_after_restart_cannot_be_a_trim_window(self):
        t = np.r_[np.arange(20), 100]
        self.assertEqual(propose(t, np.r_[np.zeros(20), .5], fs=1), [])

    def test_short_shared_stop_is_split_on_one_exclusive_boundary(self):
        t = np.arange(4000) / 100 - 10
        x = np.where((t < 2) | (t >= 18), np.sin(9 * t), 0)
        intervals = propose(t, x)
        self.assertEqual(len(intervals), 2)
        self.assertEqual(intervals[0][1], intervals[1][0])
        self.assertAlmostEqual(intervals[0][1], 10, delta=.25)
