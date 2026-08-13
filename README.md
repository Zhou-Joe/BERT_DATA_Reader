# BERT_READER X

A desktop application for **child-restraint (CR) crash-test acceleration data
analysis**, built for reviewing SARC-style (`.sup`), tab-separated, and DAQ-CSV
acceleration recordings against the key injury criteria in
**ASTM F2135 / FMVSS 213 / GB 27812** style standards.

BERT_READER X turns raw tri-axial accelerometer logs into the seven analysis
views a CR test engineer actually uses day-to-day: time-history trends,
ASTM Sustained + Egg-crate limit checks, GB-27812 contour, Acceleration Zone
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
  - **GB Contour** — GB-27812-style contour, Y-axis ±3.5 g default.
  - **Acceleration Zone** — automatic zone classification with click-to-read.
  - **Z Reversal** — detects the down-then-up acceleration-reversal signature.
  - **X/Y Reversal** — X vs Y hysteresis with per-axis allowance checks.
- **ASTM az(+) floor** — az(+) ≥ 1.0 g at rest (gravity reads 1 g), enforced.
- **Global Generate / Analyze** — one click plots / analyzes every chart tab.
- **Prone posture** toggle — re-plots the ASTM egg shapes immediately.
- **Auto test-boundary detection** — splits concatenated multi-ride recordings.
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
git clone https://github.disney.com/ZHOUC056/BERT_READER.git
cd BERT_READER

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
double-click it. No Python install required. The first "Add Data" dialog opens
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
a card in the right-hand sidebar.

### 2. Configure the analysis (left control panel, below the tab list)

- **Restraint Type** — affects the limit overlays (None / Convenience / …).
- **Condition** — test condition class.
- **Height** (cm) — scales the allowance limits (taller dummies → more allowance).
- **Prone** — toggles the prone-posture egg shapes + sustained ax− override.
- **Analyze** — one global on/off that runs on every chart tab.

### 3. Top control bar (always visible)

- **Data selector** — pick the active dataset for the standard charts
  (ASTM / GB / AccZone / Reversal). Trend shows all datasets at once.
- **Generate / Update** — re-plots **all** chart tabs from the current
  selection + control-panel settings.
- **Cutoff** (Hz) — live low-pass filter cutoff (SpinBox, no modal dialog).
- **Export All** — writes every loaded dataset to `.sup` in a chosen folder.

### 4. Per-chart interaction

- **Zoom / Pan / Reset View** toolbars on each chart.
- **Click a data point** on the Acc Zone chart to read its time + acceleration.
- **Right sidebar** — per-dataset card: colour swatch (click to recolour),
  show/hide eye, pitch/seatback/roll/yaw angles, and min/max stats.
- **Generate Word report** — snapshots every chart + a pass/fail verdict table
  into a `.docx`.

### 5. File dialog defaults

"Add Data" dialogs remember the last-used folder. On first run they open in the
folder containing the `.exe` (or the source root when running `python main.py`),
**not** the PyInstaller temp extraction folder.

---

## Project layout

```
BERT_READER/
├── main.py                 # entry point
├── main_window.py          # main window, floating bars, control panel, page generation
├── accdata.py              # data loading (AccData, RawData), filter, reformat,
│                           #   detect_test_boundaries, CSV column detection
├── sci_calculation.py      # butter_lowpass_filter, ProcessASTM, ProcessGB,
│                           #   egg shapes, allowance_at_t0, coordTransform
├── chart_base.py           # QChartView base + shared toolbars
├── chart_trend.py          # Trend (synced X-axis time history)
├── chart_astm.py           # ASTM Sustained + Egg canvases
├── chart_gb.py             # GB-27812 contour
├── chart_acczone.py        # Acceleration Zone classification
├── chart_reversal.py       # Z + X/Y reversal
├── chart_legend.py         # legend widgets
├── chart_utils.py          # series helpers
├── sidebar.py              # right dataset sidebar (cards, colour picker, stats)
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

- **pandas 3.x is not supported.** pandas 3.0 removed the silent dtype upcast,
  which breaks the `.sup` header-injection pattern (`loc[-6, 'Time'] = '...'`).
  Pin `pandas<3` (2.3.x is the tested version). The build venv already does this.
- The packaged `.exe` launches as a GUI app (no console window). Unhandled
  exceptions still surface in a dialog (`disable_windowed_traceback=False`).

---

## License

Internal tooling — © Disney / SARC test engineering. Not for redistribution.
