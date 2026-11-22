"""Exercise the same stdio endpoint an agent launches, including fresh-session isolation."""
import base64
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import asynccontextmanager

from docx import Document

from accdata import AccData
from test_data import recording


@asynccontextmanager
async def client():
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    script = Path(__file__).resolve().parents[1] / "bert_mcp.py"
    async with stdio_client(StdioServerParameters(command=sys.executable, args=[str(script)])) as pipes:
        async with ClientSession(*pipes) as connection:
            await connection.initialize()
            yield connection


@unittest.skipUnless(importlib.util.find_spec("mcp"), "Install requirements-mcp.txt for protocol tests")
class MCPTests(unittest.IsolatedAsyncioTestCase):
    async def test_analysis_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            source = folder / "example.sup"
            frame = recording(fs=100, start=-1, seconds=7)
            frame["X"], frame["Y"], frame["Z"] = 0., 0., 1.
            frame.loc[frame.Time.between(1., 2.), "Y"] = -4.6
            AccData(frame, is_raw=True, path=source.name).export_to_file(source)
            initial = source.read_bytes()
            async with client() as c:
                async def call(name, **args):
                    result = await c.call_tool(name, args)
                    self.assertFalse(result.isError, result.content)
                    return result.structuredContent

                tools = await c.list_tools()
                self.assertIn("compare_standards", [t.name for t in tools.tools])
                self.assertEqual((await call("list_recordings"))["recordings"], [])
                imported = await call("import_recordings", paths=[str(source)], file_format="sup")
                key = imported["recordings"][0]["id"]
                settings = await call("configure_analysis", settings={"source": "Raw", "height_inches": 60})
                self.assertEqual(settings["source"], "Raw")
                result = await call("compare_standards", recording_ids=[key])
                checks = result["recordings"][0]["checks"]
                self.assertEqual(set(checks), {"sust", "egg", "gb", "acczone", "zr", "xr"})
                self.assertEqual(checks["sust"]["status"], "fail")
                self.assertIn("ay", checks["sust"]["detail"])
                self.assertTrue(any(p["measured_g"] < p["limit_g"]
                                    for p in checks["sust"]["exceedances"]))
                self.assertEqual(checks["acczone"]["status"], "info")
                await call("detect_segments", recording_id=key)
                await call("signal_statistics", recording_id=key)
                children = await call("split_recording", recording_id=key, windows=[[-1, 0], [1, 2]])
                part = children["recordings"][0]["id"]
                await call("edit_axes", recording_id=part, operation="add", xyz=[2, 0, 0])
                first = await call("read_samples", recording_id=part, limit=1)
                self.assertEqual(first["rows"][0], [0., 2., 0., 1.])
                await call("set_orientation", recording_id=part, pitch=10, seatback=0, roll=0, yaw=0)
                await call("trim_recording", recording_id=part, start_s=0.1, end_s=0.4)
                invalid = await c.call_tool("split_recording", {"recording_id": key,
                                            "windows": [[0, 0.1], [1, 1.00001]]})
                self.assertTrue(invalid.isError)
                self.assertEqual(len((await call("list_recordings"))["recordings"]), 3)
                await call("align_peaks", recording_ids=[key, part], axis="Y", extremum="min")
                await call("set_trace", recording_id=part, shift_s=0.25, colour="#112233")
                for view, ids in [("trend", [key, part]), ("egg", [key])]:
                    chart = await c.call_tool("render_chart", {"recording_ids": ids, "view": view})
                    self.assertFalse(chart.isError, chart.content[0].type)
                    self.assertTrue(base64.b64decode(chart.content[0].data).startswith(b"\x89PNG"))
                saved = folder / "processed.sup"
                await call("export_recording", recording_id=part, output_path=str(saved))
                self.assertEqual(len(AccData(saved).data), 30)
                collision = await c.call_tool("export_recording", {"recording_id": part, "output_path": str(source)})
                self.assertTrue(collision.isError)
                batch = await call("export_all", output_directory=str(folder / "batch"))
                self.assertEqual(len(batch["paths"]), 3)
                report = folder / "results.docx"
                await call("export_word_report", recording_ids=[key], output_path=str(report),
                           sections=["trend", "sust"])
                doc = Document(report)
                self.assertIn("Fail", [cell.text for table in doc.tables for row in table.rows for cell in row.cells])
                self.assertGreater(len(doc.inline_shapes), 0)
                await call("remove_recordings", recording_ids=[part])
                self.assertEqual(len((await call("list_recordings"))["recordings"]), 2)
                self.assertEqual(source.read_bytes(), initial)
                async with client() as other:
                    fresh = await other.call_tool("list_recordings", {})
                    self.assertEqual(fresh.structuredContent["recordings"], [])
