"""Exercise proposal review using real Qt inputs and recording trims."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import unittest
from unittest.mock import Mock, patch

from PySide6.QtWidgets import QApplication
from accdata import AccData
from dialogs import TruncateTimeDialog
from test_data import recording


class SegmentReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.data = AccData(recording(fs=512, seconds=4, start=0),
                            is_raw=True, path="sensor")
        self.ui = Mock(cutoff=5, plottype="Raw", datalist=[self.data],
                       datadict={"sensor": self.data})
        self.ui.set_datalist.side_effect = self.ui.datalist.append
        self.dialog = TruncateTimeDialog(self.ui, self.data)
        self.addCleanup(self.dialog.deleteLater)
        self.addCleanup(self.dialog.close)

    def detect(self, windows):
        with patch("dialogs.detect_test_boundaries", return_value=windows):
            self.dialog._on_auto_detect()

    def test_only_kept_window_is_applied_at_exact_sensor_boundaries(self):
        t = self.data.data.Time.to_numpy()
        self.detect([(t[1], t[515]), (t[800], t[1500])])
        self.dialog._keeps[1].setChecked(False)
        self.dialog._on_ok()
        source, segment = self.ui.datalist
        self.assertIs(source, self.data)
        self.assertEqual(len(source.data), len(t))
        self.assertEqual(len(segment.data), 514)
        self.assertAlmostEqual(segment.data.Time.iloc[0], 0, places=12)
        self.ui._remove_dataset.assert_not_called()

    def test_one_proposal_creates_copy_and_keeps_original(self):
        self.detect([(.5, 1.5)])
        self.dialog._on_ok()
        self.assertEqual(len(self.ui.datalist), 2)
        self.assertEqual(len(self.data.data), 2048)
        self.ui._remove_dataset.assert_not_called()

    def test_bad_second_window_does_not_apply_first_or_remove_source(self):
        self.detect([(.1, 1.0), (1.5, 2.5)])
        self.dialog._start_spins[1].setValue(.8)
        self.dialog._replace_original.setChecked(True)
        self.dialog._on_ok()
        self.assertIn("overlap", self.dialog._notice.text())
        self.ui.set_datalist.assert_not_called()
        self.ui._remove_dataset.assert_not_called()

    def test_excess_proposals_do_not_silently_truncate_list(self):
        self.dialog._start_spins[0].setValue(.25)
        self.detect([(i, i + .5) for i in range(51)])
        self.assertEqual(self.dialog._count.value(), 1)
        self.assertEqual(self.dialog._start_spins[0].value(), .25)
        self.assertIn("51", self.dialog._notice.text())

    def test_edit_does_not_close_a_detected_gap(self):
        self.detect([(.1, 1.), (2., 3.)])
        self.dialog._end_spins[0].setValue(.8)
        self.assertEqual(self.dialog._start_spins[1].value(), 2.)

    def test_no_proposals_cannot_accidentally_apply_the_default_window(self):
        self.detect([])
        self.assertFalse(self.dialog._apply_button.isEnabled())
        self.dialog._on_ok()
        self.ui.set_datalist.assert_not_called()
        self.ui.replot_all.assert_not_called()
        self.dialog._keeps[0].setChecked(True)
        self.assertTrue(self.dialog._apply_button.isEnabled())

    def test_step_by_moves_one_acquired_sample_at_512_hz(self):
        t = self.data.data.Time.to_numpy()
        self.detect([(t[4], t[100])])
        start = self.dialog._start_spins[0]
        start.stepUp()
        self.assertEqual(start.seconds(), t[5])
        start.stepDown()
        self.assertEqual(start.seconds(), t[4])
        end = self.dialog._end_spins[0]
        end.set_seconds(t[-1])
        end.stepUp()
        self.assertEqual(end.seconds(), t[-1] + 1 / 512)
        end.stepDown()
        self.assertEqual(end.seconds(), t[-1])
