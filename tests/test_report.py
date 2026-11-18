"""Check what a reader sees for a selective, multi-record Word export."""
from io import BytesIO
from types import SimpleNamespace
import unittest

from docx import Document
from PySide6.QtCharts import QLineSeries, QScatterSeries

from accdata import AccData
from dialogs import ReportDialog
from report_document import AccelerationReport, MODULES
from report_lateral import lateral_traces
from test_data import recording


class ReportTests(unittest.TestCase):
    def test_lateral_figure_preserves_signed_limits_and_failure_markers(self):
        series = []
        for suffix, sign in (("+", 1), ("-", -1)):
            for label, value in (("Measured ay", 2.4), ("ASTM ay", 2.8), ("Upper Body", 2.1)):
                line = QLineSeries()
                line.setName(f"{label} ({suffix})")
                line.append(0.2, value * sign)
                series.append(line)
        exceeded = QScatterSeries()
        exceeded.append(0.2, -2.4)
        chart = SimpleNamespace(series=lambda: [*series, exceeded])
        directions, markers = lateral_traces(chart, "Upper Body")
        self.assertEqual(directions["+"]["ASTM limit"], [(0.2, 2.8)])
        self.assertEqual(directions["-"]["ASTM limit"], [(0.2, -2.8)])
        self.assertEqual(directions["-"]["Restraint limit"], [(0.2, -2.1)])
        self.assertEqual(markers, {"+": [], "-": [(0.2, -2.4)]})

    def test_selected_results_and_missing_captures_survive_round_trip(self):
        report = AccelerationReport(2)
        data = AccData(recording(), is_raw=True, path="test.sup")
        for name in ("Run A", "Run B"):
            report.add_record(data, name, {"acczone": ("info", "Classified as Zone 2"),
                                           "sust": (False, "Exceeded ax"),
                                           "egg": (True, "Excluded result")},
                              ".", {"acczone": True, "sust": True, "gb": True}, {},
                              plot_type="Raw", cutoff=5, restraint="None", condition="Normal",
                              height=0, prone=False)
        stream = BytesIO()
        report.save(stream)
        doc = Document(stream)
        self.assertEqual(len(doc.tables), 2)
        for table in doc.tables:
            self.assertEqual([row.cells[1].text for row in table.rows],
                             ["Result", "Fail", "Unavailable", "Info"])
            self.assertEqual(table.rows[-1].cells[-1].text, "Classified as Zone 2")
        text = "\n".join(p.text for p in doc.paragraphs)
        self.assertIn("Run A", text)
        self.assertIn("Run B", text)
        self.assertIn("Acceleration zone: chart image unavailable", text)
        self.assertNotIn("Axis statistics", text)

    def test_no_pass_for_unevaluated_or_partial_checks(self):
        window = SimpleNamespace(
            canvasASTMSust=SimpleNamespace(sust_failed_rules=set(), sust_axes_checked=set()),
            canvasASTMEgg=SimpleNamespace(failed_rules=set(), egg_planes_checked={"xy"}),
            canvasGB=SimpleNamespace(failed_rules=set(), comb_rules_checked=set()),
            canvasAccZone=SimpleNamespace(_auto_zone_result=None),
            canvasZReversal=SimpleNamespace(_analysis_active=True, _cached_data={}),
            canvasXYReversal=SimpleNamespace(_analysis_active=False))
        selected = dict.fromkeys(MODULES, True)
        self.assertEqual(ReportDialog._collect_verdicts(window, selected), {})
        window.canvasGB.failed_rules.add("dt=0.2s")
        self.assertFalse(ReportDialog._collect_verdicts(window, selected)["gb"][0])
