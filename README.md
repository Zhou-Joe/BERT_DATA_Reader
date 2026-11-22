# BERT_READER X

A desktop application for **child-restraint (CR) crash-test acceleration data
analysis**, built for reviewing SARC-style (`.sup`), tab-separated, and DAQ-CSV
acceleration recordings against the key injury criteria in
**ASTM F2137 / GB 8408** style standards.

BERT_READER X turns raw tri-axial accelerometer logs into the seven analysis
views a CR test engineer actually uses day-to-day: time-history trends,
ASTM Sustained + Egg-crate limit checks, GB-8408 contour, Acceleration Zone
classification, and Z / X-Y reversal signature detection — with one-click batch
export and a Word failure report.

![BERT_READER X](icon.svg)

---

## Features

The desktop workspace uses React 19 and Radix Themes. Navigation, import, and
analysis controls run from local assets in Qt WebEngine. Qt Charts handles the
interactive scientific plots, and Python processes recordings on the desktop.

- **Seven analysis views** are always visible in the sidebar:
  - **Trend** — X/Y/Z time-history for all loaded datasets, synced zoom.
  - **ASTM Sustained** — 17-duration sustained-peak window analysis with
    pass/fail limit overlays (restraint / condition / height / prone aware).
  - **ASTM Egg** — X-Y / X-Z / Y-Z "egg-crate" boundary plots with the new
    **prone posture** egg shapes and sustained ax− override.
  - **GB Contour** — GB-8408-style contour, Y-axis ±3.5 g default.
  - **Acceleration Zone** — automatic zone classification with click-to-read.
  - **Z Reversal** — detects the down-then-up acceleration-reversal signature.
  - **X/Y Reversal** — X vs Y hysteresis with per-axis allowance checks.
- **ASTM az(+) floor** — az(+) ≥ 1.0 g at rest (gravity reads 1 g), enforced.
- **Automatic refresh / Analyze** — changing files or settings prepares fresh
  results; each chart renders and applies analysis when opened.
- **Background import and calculations** — progress, cancellation, and rejection
  of results from superseded inputs.
- **Prone posture** toggle — re-plots the ASTM egg shapes immediately.
- **Auto test-boundary detection** — splits concatenated multi-ride recordings.
- **Acquisition-aware timing** — preserves CSV timestamps and derives the sample
  rate; sustained windows use seconds at the detected acquisition rate.
- **Fractional trimming** — negative/pre-trigger and sub-second timestamps are
  supported, with an exclusive end time.
- **Signal Lab workspace** — grouped analysis views, three labelled signal
  channels and a searchable recording library with a selected-signal inspector.
- **Batch export** — writes every loaded dataset back to `.sup`.
- **Word analysis report** — recording settings, results, axis statistics and
  numbered figures for the views you choose.
- **Light / Dark theme** with theme-aware icons.

---

## Requirements

| Dependency              | Version used / tested |
|-------------------------|----------------------|
| Python                  | 3.13 (3.11+ works)   |
| PySide6                 | 6.11.1               |
| QtAwesome               | 1.4.2 (Phosphor icons) |
| pandas                  | **2.3.x** (do NOT use 3.x — see *Known issue* below) |
| numpy                   | 2.5.x                |
| scipy                   | 1.18.x               |
| matplotlib              | 3.11.x (only `matplotlib.path` is used) |
| python-docx             | (latest)             |

> **Windows 10/11** is the primary platform. Qt Charts and native Qt Widgets run on
> macOS/Linux too, but the packaged `.exe` is Windows x64.

---

## Installation

### Option A — Run from source (recommended for development)

```bash
# 1. Clone
git clone https://github.com/Zhou-Joe/BERT_DATA_Reader.git
cd BERT_DATA_Reader

# 2. Create a clean virtual environment (uv is fastest; venv works too)
uv venv .venv --python 3.13            #  OR:  python -m venv .venv

# 3. Activate
#    Windows (Git Bash / PowerShell):
. .venv/Scripts/activate
#    macOS / Linux:
source .venv/bin/activate

# 4. Install dependencies
#    From PyPI (default):
uv pip install PySide6 "QtAwesome>=1.4,<2" "pandas<3" numpy scipy matplotlib python-docx pyinstaller
#    Or with the Aliyun mirror (faster inside the Disney network / China):
uv pip install PySide6 "QtAwesome>=1.4,<2" "pandas<3" numpy scipy matplotlib python-docx pyinstaller \
    -i https://mirrors.aliyun.com/pypi/simple/

# 5. Build the desktop UI (Node.js 22+)
npm --prefix frontend ci
npm --prefix frontend run build

# 6. Run
python main.py
```

### Option B — Use the prebuilt Windows `.exe`

If a `dist/BERT_READER_X.exe` is available (built by Option C below), just
double-click it. No Python install required. The first "Add data" dialog opens
in the folder where the `.exe` lives.

---

## Use BERT from an AI agent

`bert_mcp.py` is a local stdio MCP server. Each client connection starts with
an empty recording library and its own analysis settings. Import files by
absolute path. Edits apply to that session's copies. Export the results you
want to keep before disconnecting.

After installing the Python application dependencies, install the MCP dependency
in the same environment:

```powershell
python -m pip install -r requirements-mcp.txt
```

For Codex, run this from the application directory with that environment active:

```powershell
$bertPython = python -c "import sys; print(sys.executable)"
$bertServer = (Resolve-Path .\bert_mcp.py).Path
codex mcp add bert-signal-lab -- $bertPython $bertServer
```

For another stdio MCP client, use the same Python executable as its `command`
and the absolute path to `bert_mcp.py` as its sole argument. The agent can run
the analysis with the desktop application closed.

| Task | Tools |
| --- | --- |
| Import and inspect | `import_recordings` (`daq`, `sup`, or `sensorconnect`), `list_recordings`, `read_samples`, `signal_statistics` |
| Edit signals | `edit_axes`, `set_orientation`, `trim_recording`, `detect_segments`, `split_recording`, `remove_recordings` |
| Compare and plot | `configure_analysis`, `compare_standards`, `align_peaks`, `set_trace`, `render_chart` |
| Export | `export_recording` (SUP or TSV), `export_all` (SUP batch), `export_word_report` |

Use the IDs returned by import for subsequent calls. A DAQ file can return
both a GB and an ASTM channel group. The comparison tool evaluates each
selected recording against the app's built-in checks and returns its settings,
check names, findings and verdicts. `sust` and `egg` select the ASTM F2291
checks, `gb` selects GB 8408, `acczone` classifies acceleration zones, and
`zr` and `xr` select the reversal checks. `render_chart` returns a PNG for
any of these views or the multi-recording trend.

For example, ask your agent:

> Import the DAQ file at the absolute path I supplied. List the two recording
> IDs, use calibrated data at 5 Hz, and compare both against ASTM sustained
> acceleration and GB 8408 with Upper Body restraint, Normal condition, and
> a 60-inch height. Show me the failing checks and the corresponding plots.

Time windows use seconds and exclude their end. A trim rebases the selected
interval to its requested start. Peak alignment and time shifts affect the
trend display. Exports require a new destination. The Word report uses the
same evaluation routines as the comparison tool.

## Building a Windows `.exe` (one-file PyInstaller)

A ready-to-use spec is included:

```bash
# from the repo root, with the venv from Option A active
pyinstaller --noconfirm --clean build.spec
```

The output is `dist/BERT_READER_X.exe`. Build the frontend before packaging.
The spec includes the React workspace, local fonts, and Qt WebEngine runtime. It also:

- bundles `qtawesome` icon fonts, `PySide6`, `python-docx` data files + hidden imports,
- excludes unused matplotlib GUI backends (only `matplotlib.path` is kept),
- embeds `icon.ico` and ships `icon.svg` for the window icon.

To regenerate the multi-resolution icon from the SVG source:

```bash
python gen_icon.py     # writes icon.ico + icon_preview.png from icon.svg
```

---

## Usage

### 1. Load data

| Action           | Accepts                                                |
|------------------|--------------------------------------------------------|
| **Acceleration recording / SUP** | `.sup` (SARC tab-separated, the native format) |
| **Raw DAQ recording** | `.csv` / `.txt` from a DAQ export (multi-channel) |
| **SensorConnect CSV** | SensorConnect accelerometer CSV with timestamps |

> The CSV reader auto-detects the header length and maps channels by label
> (`ASTM_X/Y/Z`, `GB_X/Y/Z`) — it no longer hardcodes a 23-row skip. GB / ASTM
> column ordering is detected from the `Channel Description` header line.

Multiple datasets can be loaded; each gets an auto-assigned palette colour and
a selectable row in the searchable right-hand recording library.

The DAQ and acceleration recording importers accept multiple files in one dialog.
Use Ctrl-click or Shift-click to select files, then Open. Each raw file adds
its GB and ASTM recordings; each `.sup` file adds one recording. Import shows
batch progress and a completion summary. Unreadable files are skipped with
per-file errors while the remaining files continue. Duplicate filenames get
unique display names. Cancel keeps any files already added and skips those
still pending.

### 2. Configure the analysis

The seven analysis views stay visible in the left navigation rail. Analysis
settings sit directly below them and scroll independently on smaller screens.

- **Restraint Type** — affects the limit overlays (None / Convenience / …).
- **Condition** — test condition class.
- **Height** (in) — scales the allowance limits (taller dummies → more allowance).
- **Prone** — toggles the prone-posture egg shapes + sustained ax− override.
- **Analyze** (command bar) — one global on/off that runs on every chart tab.

### 3. Top control bar (always visible)

- **Data selector** — pick the active dataset for the standard charts
  (ASTM / GB / AccZone / Reversal). Trend shows all datasets at once.
- **Update charts** — manually recalculates analysis from the current inputs.
  File and settings changes refresh automatically; hidden views draw when opened.
- **Freshness status** — shows outdated, updating, ready, cancelled or failed.
  Previous plots and verdicts stay hidden while results are outdated.
- **Progress / Cancel** — import and numerical work run in the background.
  Cancel stops applying unfinished work; use Update charts to retry.
- **Cutoff** (Hz) — live low-pass filter cutoff (SpinBox, no modal dialog).
- **Export all** — writes every loaded dataset to `.sup` in a chosen folder.
- **Recording tools** — select a recording to trim or split it, edit orientation or
  axis values, change its colour, export it, or remove it from the workspace.
  These buttons remain accessible when the library is closed.

### 4. Per-chart interaction

- **Zoom / Pan / Reset View** toolbars on each chart.
- **Long recordings** — Trend and Reversal draw a peak-preserving envelope at
  the current display resolution. Zooming in restores the original samples;
  numerical analysis and recording exports always use the full recording.
- **Trend** — use the six **Align peaks** buttons to align two or more recordings
  by the maximum or minimum on each axis. Use **Time shift** for a manual
  offset or **Annotate** to change annotation mode.
- **Click a data point** on the Acc Zone chart to read its time + acceleration.
- **Recording library** — search imported files, click a row to select it, and
  toggle visibility using the eye. Click the coloured square beside a recording
  to choose its trace colour across all charts and Word reports. Negative
  sustained curves use a dashed line. The selected signal's
  statistics appear in a separate inspector; drag the divider to resize it.
- **Library** — hide the recordings panel for a wider chart. Press **Ctrl+F**
  to reopen it and search without changing the active recording.
- **Trim / split** — select a recording and click **Trim / split**. Choose
  **Auto-detect** to see suggested windows over the acceleration trace.
  Shading marks the time ranges to keep. Adjust the start or exclusive end,
  or uncheck a segment to omit it. A longer **Quiet gap** treats longer pauses
  as part of the same test. Detected segments are copies; the original stays
  in the library unless you select **Remove original after creating segments**.
  A single manual window uses **Trim original** instead.
- **Word report** — choose recordings and analysis views. Each recording's
  section opens with its settings and results, followed by the selected charts
  with numbered captions. The trend option also includes axis statistics.
  The ASTM sustained lateral acceleration figure overlays rightward (+ay) and
  leftward (−ay) measurements with the upper and lower limits on one signed axis.

### 5. File dialog defaults

"Add data" dialogs remember the last-used folder. On first run they open in the
folder containing the `.exe` (or the source root when running `python main.py`),
**not** the PyInstaller temp extraction folder.

---

## Project layout

```
BERT_READER/
├── main.py                 # entry point
├── main_window.py          # main window, bars, control panel, chart pages
├── background.py           # cancellable Qt worker pool and queued signals
├── analysis_pipeline.py    # numeric snapshots, filtering and prepared results
├── processing_controller.py # automatic refresh, input validation and GUI delivery
├── accdata.py              # data loading (AccData, RawData), filter, reformat,
│                           #   detect_test_boundaries, CSV column detection
├── sci_calculation.py      # butter_lowpass_filter, ProcessASTM, ProcessGB,
│                           #   egg shapes, allowance_at_t0, coordTransform
├── chart_base.py           # QChartView base + shared toolbars
├── chart_trend.py          # Trend (synced X-axis time history)
├── chart_astm.py           # ASTM Sustained + Egg canvases
├── chart_gb.py             # GB-8408 contour
├── chart_acczone.py        # Acceleration Zone classification
├── chart_reversal.py       # Z + X/Y reversal
├── chart_legend.py         # legend widgets
├── chart_utils.py          # series helpers
├── sidebar.py              # recording library and selected-signal inspector
├── dialogs.py              # TruncateTimeDialog, ReportDialog, CutoffFrequencyDialog
├── segment_preview.py      # trim timeline and sample-accurate time controls
├── report_document.py      # Word layout, results tables and figure captions
├── theme.py                # token system, app palette/QSS, theme switching
├── ui_controls.py          # shared Qt controls and Phosphor icons
├── boundary_analysis.py    # boundary-violation helpers
├── models.py               # data model shims
├── icon.svg / icon.ico     # app icon (vector + multi-resolution)
├── gen_icon.py             # regenerates icon.ico from icon.svg
├── build.spec              # PyInstaller one-file spec
└── README.md
```

---

## Known issues / notes

- Validation currently uses **pandas 2.3.x**; keep `pandas<3` until the full
  dependency stack has been validated with 3.x. Export headers now use an
  explicit object table, leaving the numeric analysis frames unchanged.
- The packaged `.exe` launches as a GUI app (no console window). Unhandled
  exceptions still surface in a dialog (`disable_windowed_traceback=False`).

---

## License

Internal tooling — © Disney / SARC test engineering. Not for redistribution.


## Development checks

Run the numerical and headless Qt regression suite from the repository root:

```bash
python -m unittest discover -s tests -v
```

Exercise background import, cancellation and all seven views in both themes
with a local DAQ recording:

```bash
python tools/review_gui.py --sample path/to/recording.csv --out path/to/previews
# Batch import smoke check:
python tools/review_gui.py --sample path/to/first.csv path/to/second.csv --out path/to/previews
```

See [GUI_REVIEW.md](GUI_REVIEW.md) for the changes, measured component timings,
and remaining performance opportunities. Real recordings and preview images
are kept outside the source tree.
