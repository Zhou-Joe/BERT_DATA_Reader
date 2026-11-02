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

- **7 analysis tabs** (custom SVG icons per tab):
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
- **Compact workspace** — two-row command bar, scrollable analysis settings,
  a searchable recording library, and readable light/dark chart styling.
- **One-click batch export** — writes every loaded dataset back to `.sup`.
- **Word failure report** — snapshot of all charts + a pass/fail verdict table.
- **Light / Dark theme** with theme-aware icons.

---

## Requirements

| Dependency              | Version used / tested |
|-------------------------|----------------------|
| Python                  | 3.13 (3.11+ works)   |
| PySide6                 | 6.11.1               |
| PySide6-Fluent-Widgets  | 1.11.2 (`qfluentwidgets`) |
| pandas                  | **2.3.x** (do NOT use 3.x — see *Known issue* below) |
| numpy                   | 2.5.x                |
| scipy                   | 1.18.x               |
| matplotlib              | 3.11.x (only `matplotlib.path` is used) |
| python-docx             | (latest)             |

> **Windows 10/11** is the primary platform. Qt Charts + Fluent Widgets run on
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
uv pip install PySide6 "PySide6-Fluent-Widgets" "pandas<3" numpy scipy matplotlib python-docx pyinstaller
#    Or with the Aliyun mirror (faster inside the Disney network / China):
uv pip install PySide6 "PySide6-Fluent-Widgets" "pandas<3" numpy scipy matplotlib python-docx pyinstaller \
    -i https://mirrors.aliyun.com/pypi/simple/

# 5. Run
python main.py
```

### Option B — Use the prebuilt Windows `.exe`

If a `dist/BERT_READER_X.exe` is available (built by Option C below), just
double-click it. No Python install required. The first "Add data" dialog opens
in the folder where the `.exe` lives.

---

## Building a Windows `.exe` (one-file PyInstaller)

A ready-to-use spec is included:

```bash
# from the repo root, with the venv from Option A active
pyinstaller --noconfirm --clean build.spec
```

The output is `dist/BERT_READER_X.exe` (≈ 315 MB, GUI subsystem — **no console
window** on launch). The spec already:

- bundles `qfluentwidgets`, `PySide6`, `python-docx` data files + hidden imports,
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
| **Add Accel Data** | `.sup` (SARC tab-separated, the native format)       |
| **Add Raw Data**   | `.csv` / `.txt` from a DAQ export (multi-channel)    |

> The CSV reader auto-detects the header length and maps channels by label
> (`ASTM_X/Y/Z`, `GB_X/Y/Z`) — it no longer hardcodes a 23-row skip. GB / ASTM
> column ordering is detected from the `Channel Description` header line.

Multiple datasets can be loaded; each gets an auto-assigned palette colour and
a selectable row in the searchable right-hand recording library.

Both **Add Raw Data** and **Add Accel Data** accept multiple files in one dialog.
Use Ctrl-click or Shift-click to select files, then Open. Each raw file adds
its GB and ASTM recordings; each `.sup` file adds one recording. Import shows
batch progress and a completion summary. Unreadable files are skipped with
per-file errors while the remaining files continue. Duplicate filenames get
unique display names. Cancel keeps any files already added and skips those
still pending.

### 2. Configure the analysis (left control panel, below the tab list)

- **Restraint Type** — affects the limit overlays (None / Convenience / …).
- **Condition** — test condition class.
- **Height** (in) — scales the allowance limits (taller dummies → more allowance).
- **Prone** — toggles the prone-posture egg shapes + sustained ax− override.
- **Analyze** — one global on/off that runs on every chart tab.

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
- **Export All** — writes every loaded dataset to `.sup` in a chosen folder.

### 4. Per-chart interaction

- **Zoom / Pan / Reset View** toolbars on each chart.
- **Click a data point** on the Acc Zone chart to read its time + acceleration.
- **Recording library** — search imported files, click a row to select it, and
  toggle visibility using the eye. Each row's menu offers trimming, orientation,
  axis edits, colour selection, export and removal. The selected signal's
  statistics appear in a separate inspector; drag the divider to resize it.
- **Generate Word report** — snapshots every chart + a pass/fail verdict table
  into a `.docx`.

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
├── theme.py                # token system, app palette/QSS, theme switching
├── nav_icons.py            # custom SVG nav-tab icons
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
