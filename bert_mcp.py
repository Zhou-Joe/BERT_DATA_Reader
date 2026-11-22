"""Local stdio MCP server for an independent BERT Signal Lab analysis session."""
import copy
import os
import sys
import tempfile
from contextlib import asynccontextmanager, contextmanager, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from mcp.server.fastmcp import FastMCP, Image
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from accdata import AccData, detect_test_boundaries
from analysis_pipeline import FRAME_ATTR, import_recordings as load_batch
from background import CancellationToken
from report_document import MODULES
from sci_calculation import ProcessASTM
from theme import DATASET_COLORS


Source = Literal["Raw", "Filter", "Standard"]
Section = Literal["trend", "sust", "egg", "gb", "acczone", "zr", "xr"]
Check = Literal["sust", "egg", "gb", "acczone", "zr", "xr"]
Positive = Annotated[FiniteFloat, Field(gt=0)]
Angle = Annotated[int, Field(ge=-180, le=180)]
Colour = Annotated[str, Field(pattern="^#[0-9a-fA-F]{6}$")]
XYZ = tuple[FiniteFloat, FiniteFloat, FiniteFloat]


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: Source = "Standard"
    cutoff_hz: Annotated[int, Field(ge=1, le=249)] = 5
    restraint: Literal["None", "Individual Lower Body", "Upper Body", "Group Lower Body",
                       "Convenience Restraint", "No Restraint"] = "None"
    condition: Literal["Normal", "E-Stop", "Expected/Permitted Bumping"] = "Normal"
    height_inches: Annotated[int, Field(ge=0, le=80)] = 0
    prone: bool = False
    trend_window_s: Annotated[FiniteFloat, Field(ge=0.001, le=60)] = 0.2
    z_reversal_source: Source = "Standard"
    xy_reversal_source: Source = "Standard"
    xy_deadband_g: Annotated[FiniteFloat, Field(ge=0, le=2)] = 0.05


@dataclass
class Recording:
    data: AccData
    path: str
    colour: str
    shift_s: float = 0.0
    visible: bool = True


class Session:
    def __init__(self):
        self.records = {}
        self.next_id = 1
        self.settings = Settings()
        self.app = self.window = None

    def add(self, data, path):
        key = f"r{self.next_id}"
        colour = DATASET_COLORS[(self.next_id - 1) % len(DATASET_COLORS)]
        self.records[key] = Recording(data, path, colour)
        self.next_id += 1
        return key

    def get(self, key):
        if key not in self.records:
            raise ValueError(f"Unknown recording ID {key!r}. Use list_recordings.")
        return self.records[key]

    def select(self, keys):
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("Supply one or more distinct recording IDs.")
        return [(key, self.get(key)) for key in keys]

    def frame(self, record, source=None):
        source = source or self.settings.source
        if source != "Raw":
            record.data.filter_data(self.settings.cutoff_hz)
        return getattr(record.data, FRAME_ATTR[source])

    def renderer(self):
        from PySide6.QtWidgets import QApplication
        from main_window import MainWindow
        from theme import theme_manager

        if self.window is None:
            os.environ["QT_QPA_PLATFORM"] = "offscreen"
            # The MCP stdio client forwards SYSTEMROOT on Windows. QtAwesome
            # uses its WINDIR alias to locate the bundled icon fonts.
            if os.name == "nt" and "SYSTEMROOT" in os.environ:
                os.environ.setdefault("WINDIR", os.environ["SYSTEMROOT"])
            self.app = QApplication.instance() or QApplication(["bert-mcp"])
            theme_manager.apply(self.app)
            self.window = MainWindow()
            # Tools evaluate their explicit inputs synchronously on the Qt thread.
            self.window._refresh_suspended = True
        w, s = self.window, self.settings
        w.set_cutoff(s.cutoff_hz)
        w.plottype = s.source
        w.select_restraint.setCurrentText(s.restraint)
        w.select_cond.setCurrentText(s.condition)
        w.height_input.setValue(s.height_inches)
        w._prone_btn.setChecked(s.prone)
        return w


session = Session()


@asynccontextmanager
async def lifespan(server):
    # The SDK opens its protocol stream before entering the lifespan. BERT's
    # readers and charts print diagnostics, which belong on stderr.
    with redirect_stdout(sys.stderr):
        try:
            yield {}
        finally:
            if session.window is not None:
                session.window.close()


mcp = FastMCP("BERT Signal Lab", lifespan=lifespan, instructions=(
    "Each connection owns an independent in-memory session. Import named local files, "
    "then use the returned recording IDs. Times are seconds and accelerations are g. "
    "Configure the test conditions before comparing the app's built-in standards."))


def describe(key):
    record = session.get(key)
    data = record.data
    times = data.data["Time"]
    return dict(id=key, name=data.filename, source_path=record.path, samples=len(times),
                sample_rate_hz=float(data.fs), first_s=float(times.iloc[0]),
                last_s=float(times.iloc[-1]), end_exclusive_s=float(times.iloc[-1] + 1 / data.fs),
                orientation_deg=dict(pitch=data.pitch_angle, seatback=data.seatback_angle,
                                     roll=data.roll_angle, yaw=data.yaw_angle),
                colour=record.colour, visible=record.visible, trend_shift_s=record.shift_s)


@mcp.tool()
def list_recordings() -> dict[str, Any]:
    """List this session's recordings, IDs, timing, display shifts and current analysis settings."""
    return dict(recordings=[describe(key) for key in session.records],
                settings=session.settings.model_dump())


@mcp.tool()
def import_recordings(paths: list[str], file_format: Literal["daq", "sup", "sensorconnect"]) -> dict[str, Any]:
    """Import local files by absolute path. DAQ creates GB and ASTM signals. Returns IDs and per-file errors."""
    if not paths or any(not Path(p).is_absolute() for p in paths):
        raise ValueError("Supply one or more absolute local file paths.")
    mode = {"daq": True, "sup": False, "sensorconnect": "sensorconnect"}[file_format]
    result = load_batch(paths, mode, session.settings.cutoff_hz, CancellationToken(), lambda *_: None)
    added = [describe(session.add(data, path)) for path, signals in result.recordings for data in signals]
    return dict(recordings=added, errors=[dict(path=p, message=e) for p, e in result.errors])


@mcp.tool()
def configure_analysis(settings: Settings) -> dict[str, Any]:
    """Replace session settings. Omitted fields take the displayed schema defaults. Standard means calibrated."""
    session.settings = settings
    return settings.model_dump()


@mcp.tool()
def read_samples(recording_id: str, offset: Annotated[int, Field(ge=0)] = 0,
                 limit: Annotated[int, Field(ge=1, le=2000)] = 100,
                 source: Source | None = None) -> dict[str, Any]:
    """Read a page of full-resolution [time_s, x_g, y_g, z_g] samples in acquisition order."""
    frame = session.frame(session.get(recording_id), source)
    page = frame.iloc[offset:offset + limit, :4]
    end = offset + len(page)
    return dict(columns=["time_s", "x_g", "y_g", "z_g"], rows=page.to_numpy().tolist(),
                next_offset=end if end < len(frame) else None, total=len(frame))


@mcp.tool()
def signal_statistics(recording_id: str, source: Source | None = None,
                      window_s: Positive | None = None) -> dict[str, Any]:
    """Return signed axis extrema, their acquisition timestamps and sustained positive/negative peaks in g."""
    rec = session.get(recording_id)
    frame = session.frame(rec, source)
    seconds = window_s or session.settings.trend_window_s
    width = max(1, round(seconds * rec.data.fs))
    if width > len(frame):
        raise ValueError("The sustained window exceeds the recording duration.")
    pos, neg = ProcessASTM(frame, width)
    return dict(recording_id=recording_id, source=source or session.settings.source,
                window_s=seconds, window_samples=width,
                axes=rec.data.get_data_stats(frame).to_dict(orient="index"),
                sustained_g={axis: dict(positive=float(pos[i]), negative=float(neg[i]))
                             for i, axis in enumerate("XYZ")})


def edit_copy(key, change):
    record = session.get(key)
    staged = copy.deepcopy(record.data)
    change(staged)
    record.data = staged
    record.shift_s = 0
    return describe(key)


@mcp.tool()
def edit_axes(recording_id: str, operation: Literal["add", "multiply"], xyz: XYZ) -> dict[str, Any]:
    """Add offsets in g or multiply X, Y and Z by three factors. Recalculate calibrated and filtered data."""
    return edit_copy(recording_id, lambda d: d.edit_data(operation, xyz, session.settings.cutoff_hz))


@mcp.tool()
def set_orientation(recording_id: str, pitch: Angle, seatback: Angle, roll: Angle, yaw: Angle) -> dict[str, Any]:
    """Set orientation in degrees and recompute the recording's calibration."""
    return edit_copy(recording_id, lambda d: d.reformat(
        overwrite=True, setting_angle=True, pitch_angle=pitch, seatback_angle=seatback,
        roll_angle=roll, yaw_angle=yaw, cutoff=session.settings.cutoff_hz))


def check_windows(data, windows):
    first, last = data.data["Time"].iloc[[0, -1]]
    previous = first
    if not windows:
        raise ValueError("Supply at least one time window.")
    for start, end in windows:
        if not previous <= start < end <= last + 1 / data.fs:
            raise ValueError("Use ordered, non-overlapping [start_s, end_s) windows inside the recording.")
        previous = end


@mcp.tool()
def trim_recording(recording_id: str, start_s: FiniteFloat, end_s: FiniteFloat) -> dict[str, Any]:
    """Keep [start_s, end_s) in the session copy and rebase its clock to the requested start."""
    check_windows(session.get(recording_id).data, [(start_s, end_s)])
    return edit_copy(recording_id, lambda d: d.truncate_data(start_s, end_s, session.settings.cutoff_hz))


@mcp.tool()
def detect_segments(recording_id: str, quiet_gap_s: Positive = 10) -> dict[str, Any]:
    """Propose half-open test windows using BERT's automatic ride-boundary detector."""
    return dict(windows=detect_test_boundaries(session.get(recording_id).data, min_gap_s=quiet_gap_s))


@mcp.tool()
def split_recording(recording_id: str, windows: list[tuple[FiniteFloat, FiniteFloat]]) -> dict[str, Any]:
    """Create one new recording per [start_s, end_s) window. Keep the original recording."""
    record = session.get(recording_id)
    check_windows(record.data, windows)
    parts = []
    for i, (start, end) in enumerate(windows, 1):
        part = copy.deepcopy(record.data)
        part.truncate_data(start, end, session.settings.cutoff_hz)
        part.filename = f"{part.filename} [Trunc {start:.8g}-{end:.8g}s #{i}]"
        parts.append(part)
    return dict(recordings=[describe(session.add(p, record.path)) for p in parts])


@mcp.tool()
def set_trace(recording_id: str, shift_s: FiniteFloat | None = None,
              colour: Colour | None = None, visible: bool | None = None) -> dict[str, Any]:
    """Set trend display properties. Positive seconds shift the trace earlier. Zero resets its shift."""
    rec = session.get(recording_id)
    if shift_s is not None:
        rec.shift_s = shift_s
    if colour is not None:
        rec.colour = colour
    if visible is not None:
        rec.visible = visible
    return describe(recording_id)


@mcp.tool()
def align_peaks(recording_ids: list[str], axis: Literal["X", "Y", "Z"],
                extremum: Literal["max", "min"]) -> dict[str, Any]:
    """Align trend traces to the first recording's chosen extremum using acquired timestamps."""
    records = session.select(recording_ids)
    if len(records) < 2:
        raise ValueError("Peak alignment requires at least two recordings.")
    times = []
    for _, rec in records:
        frame = session.frame(rec)
        values = frame.iloc[:, "XYZ".index(axis) + 1]
        index = values.idxmax() if extremum == "max" else values.idxmin()
        times.append(float(frame.loc[index, "Time"]))
    for (_, rec), time in zip(records, times):
        rec.shift_s = time - times[0]
    return dict(shifts_s={key: rec.shift_s for key, rec in records})


@mcp.tool()
def remove_recordings(recording_ids: list[str]) -> dict[str, Any]:
    """Discard the named in-memory recordings. Source files remain on disk."""
    for key, _ in session.select(recording_ids):
        del session.records[key]
    return dict(removed=recording_ids)


def module_flags(keys):
    if not keys:
        raise ValueError("Select at least one analysis section.")
    return {key: key in keys for key in MODULES}


def reversal_options():
    s = session.settings
    return dict(z_source=s.z_reversal_source, xy_source=s.xy_reversal_source, deadband=s.xy_deadband_g)


def evaluate(record, flags):
    from dialogs import evaluate_recording
    w = session.renderer()
    return evaluate_recording(w, record.data, flags, color=record.colour, **reversal_options())


@mcp.tool()
def compare_standards(recording_ids: list[str], checks: list[Check] | None = None) -> dict[str, Any]:
    """Compare recordings with BERT's ASTM F2291 sustained/egg, GB 8408, zone and reversal checks.
    Returns each check's pass, fail, informational or unavailable result and the applied settings.
    """
    flags = module_flags([name for name in MODULES if name != "trend"] if checks is None else checks)
    results = []
    for key, rec in session.select(recording_ids):
        verdicts = evaluate(rec, flags)
        readings = {}
        for check, enabled in flags.items():
            if enabled:
                value, detail = verdicts.get(check, (None, "Check was not fully evaluated"))
                status = {True: "pass", False: "fail", "info": "info", None: "unavailable"}[value]
                readings[check] = dict(name=MODULES[check], status=status, detail=detail)
        w = session.window
        if flags["sust"]:
            readings["sust"]["exceedances"] = list(w.canvasASTMSust.sust_exceedances)
        if flags["acczone"] and w.canvasAccZone._last_run is not None:
            zone, start, end = w.canvasAccZone._last_run
            readings["acczone"].update(zone=zone, start_s=start, end_s=end)
        if flags["zr"]:
            readings["zr"]["event_count"] = w.canvasZReversal.event_count
        if flags["xr"]:
            readings["xr"].update(event_count=w.canvasXYReversal.event_count,
                                  failed_pairs=w.canvasXYReversal.fail_count)
        results.append(dict(recording_id=key, name=rec.data.filename, checks=readings))
    return dict(settings=session.settings.model_dump(), recordings=results)


@contextmanager
def new_output(path, extensions):
    dest = Path(path)
    if not dest.is_absolute() or dest.suffix.lower() not in extensions:
        raise ValueError(f"Supply an absolute output path ending in {', '.join(extensions)}.")
    if dest.exists():
        raise FileExistsError(f"Output already exists: {dest}")
    with tempfile.TemporaryDirectory(dir=dest.parent, prefix=".bert-") as staging:
        temp = Path(staging) / dest.name
        yield str(temp)
        # Publish a complete result and refuse to replace a concurrent writer.
        if os.name == "nt":
            os.rename(temp, dest)
        else:
            os.link(temp, dest)


@mcp.tool()
def export_recording(recording_id: str, output_path: str,
                     file_format: Literal["sup", "tsv"] = "sup", source: Source | None = None) -> dict[str, Any]:
    """Write a new SUP with metadata or headerless tab-separated Time/X/Y/Z samples. Requires a new path."""
    rec = session.get(recording_id)
    suffixes = (".sup", ".sups") if file_format == "sup" else (".tsv", ".txt")
    with new_output(output_path, suffixes) as temp:
        if file_format == "sup":
            rec.data.export_to_file(temp)
        else:
            session.frame(rec, source)
            rec.data.export_filter(temp, source or session.settings.source)
    return dict(path=output_path, samples=len(rec.data.data))


@mcp.tool()
def export_all(output_directory: str) -> dict[str, Any]:
    """Export all session recordings as SUP into a new directory. Each filename includes its recording ID."""
    import re
    dest = Path(output_directory)
    if not dest.is_absolute() or dest.exists() or not session.records:
        raise ValueError("Supply a new absolute directory and import at least one recording.")
    with tempfile.TemporaryDirectory(dir=dest.parent, prefix=".bert-") as staging:
        folder = Path(staging) / "exports"
        folder.mkdir()
        names = []
        for key, rec in session.records.items():
            stem = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", Path(rec.data.filename).stem)
            name = f"{key}-{stem}.sup"
            rec.data.export_to_file(folder / name)
            names.append(name)
        folder.rename(dest)
    return dict(paths=[str(dest / name) for name in names])


@mcp.tool()
def export_word_report(recording_ids: list[str], output_path: str,
                       sections: list[Section] | None = None) -> dict[str, Any]:
    """Generate the built-in Word report with selected figures, statistics and verdicts at a new .docx path."""
    from dialogs import ReportDialog
    records = session.select(recording_ids)
    flags = module_flags(list(MODULES) if sections is None else sections)
    w = session.renderer()
    with new_output(output_path, (".docx",)) as temp:
        w.datadict = {f"{key} · {rec.data.filename}": rec.data for key, rec in records}
        w.dataset_colors = {f"{key} · {rec.data.filename}": rec.colour for key, rec in records}
        dialog = ReportDialog(w)
        try:
            for name, box in dialog._module_boxes.items():
                box.setChecked(flags[name])
            dialog._generate(list(w.datadict), output_path=temp, **reversal_options())
        finally:
            w.datadict = {}
            dialog.deleteLater()
    return dict(path=output_path, recording_ids=recording_ids, settings=session.settings.model_dump())


@mcp.tool()
def render_chart(recording_ids: list[str], view: Section = "trend",
                 trend_analysis: bool = False) -> Image:
    """Return a PNG chart. Trend overlays the visible named recordings, including display shifts.
    Each standards or reversal view takes one recording and shows its evaluated overlays.
    """
    from PySide6.QtCore import QBuffer, QIODevice
    from dialogs import _disable_series_opengl
    records = session.select(recording_ids)
    w = session.renderer()
    if view == "trend":
        canvas = w.trendplot
        canvas.initplotxyz()
        shown = [rec for _, rec in records if rec.visible]
        if not shown:
            raise ValueError("All selected traces are hidden.")
        for index, rec in enumerate(shown, 1):
            canvas.addplotxyz_shifted(session.frame(rec), index, rec.shift_s, rec.colour)
        canvas.reset_view()
        if trend_analysis:
            canvas.toggle_analysis([r.data for r in shown], session.settings.source,
                                   {r.data.filename: r.colour for r in shown},
                                   window=session.settings.trend_window_s)
        canvas.show()
    else:
        if len(records) != 1:
            raise ValueError("Select exactly one recording for this chart.")
        evaluate(records[0][1], module_flags([view]))
        canvas = {"sust": w.canvasASTMSust, "egg": w.canvasASTMEgg, "gb": w.canvasGB,
                  "acczone": w.canvasAccZone, "zr": w.canvasZReversal, "xr": w.canvasXYReversal}[view]
    w.stackedWidget.setCurrentIndex(list(MODULES).index(view))
    w._trend_empty.hide()
    w.show()
    _disable_series_opengl([canvas])
    session.app.processEvents()
    buffer = QBuffer()
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    if not canvas.grab().save(buffer, "PNG"):
        raise OSError("Could not render the selected chart")
    return Image(data=bytes(buffer.data()), format="png")


if __name__ == "__main__":
    mcp.run(transport="stdio")
