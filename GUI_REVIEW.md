# GUI and performance review

## Appearance

- Replaced the overflowing command row with two rows: file actions and the
  active dataset/filter controls. Secondary readouts live in the status bar.
- Replaced the right-hand statistics cards with a searchable recording library.
  Compact rows show filename, channel group, acquisition rate, duration, colour,
  visibility and an actions menu. Selection stays synchronized with the active
  dataset selector. A separate inspector shows only the selected signal's
  statistics and orientation; Export and Edit stay accessible in a fixed footer.
  A vertical splitter adjusts the balance between files and details.
- Added useful empty states, recording counts and a trace palette that remains
  readable in both themes. Long filenames elide by available space with a full
  tooltip, and search is case-insensitive.
- Consolidated peak alignment into a menu; plot modes now show their selection.
- Made analysis settings scroll at shorter window heights. Timestamp tick
  density follows the chart width. Dark-mode reference lines remain visible.
- Validated command controls at 1100×720, 1366×768 and 1600×900 logical pixels.

## Bugs corrected

- CSV imports discarded timestamps and assumed 500 Hz. The reader now preserves
  source time, derives the acquisition rate and maps labeled axes in XYZ order.
  Six-channel tab files use the header sample rate when time is absent.
- Sustained charts used hard-coded sample counts. Their duration windows now
  scale with acquisition rate; the limit formulas themselves are unchanged.
- Fractional trims were rounded down to whole seconds and treated timestamps
  as row offsets. Trims now select `[start, end)` using source timestamps,
  preserve pre-trigger data, and reject empty intervals without changing data.
- Shortened recordings retained old rows in shared filter frames. Filtering
  now builds fresh frames and keeps the original time precision.
- Exporting rebuilt live data and silently reset filtering to 5 Hz. Exports now
  build a separate header table and leave live analysis frames untouched.
- Peak statistics and Acceleration Zone click readouts now use actual timestamps.
- Adding a dataset no longer resets the current selector. Clear All also clears
  visibility state and the time-shift selector.
- Turning Analyze off clears hidden pages too. Unplotted pages no longer show
  an empty analysis as a passing result. Generation errors appear in the UI.
- External theme changes synchronize the title-bar toggle.
- Z-reversal detection now requires the documented full 0.2-second pre-window;
  the previous implementation accepted truncated pre-windows near file start.

## Automatic refresh and background processing

- Dataset selection, plot mode, restraint, condition, height, prone posture,
  reversal mode, hysteresis and cutoff changes automatically refresh results.
  Rapid settings changes coalesce; cutoff edits retain their 250 ms debounce.
- Parsing, filtering, sustained-window calculations, reversal detection and
  zone classification run in a Qt thread pool. Workers use source snapshots
  and reuse unchanged filtered frames as read-only inputs. Qt widgets and
  chart series are only modified on the GUI thread.
- Each result carries the complete input signature. Superseded jobs are
  cancelled and late results are discarded. Failed or cancelled numerical
  preparation does not apply partially filtered snapshots to live data.
- Outdated plots and verdicts are hidden immediately. The command bar shows
  whether results are outdated, updating, ready, cancelled or failed. Filtered
  sidebar statistics are hidden while their cutoff is pending, and their
  heading shows the applied cutoff once ready.
- Progress and Cancel remain available during background work. Cancellation
  is cooperative: a library read or calculation already in progress finishes
  its current step before stopping; its cancelled result is never applied.
- Numeric results are prepared once; hidden chart pages draw when opened.
  This avoids redrawing every chart during each selection or settings change.
- Report capture suspends automatic chart refresh to prevent background
  completions from changing figures mid-export. Reports apply the requested
  cutoff before reading any filtered data. Filtered-data export waits for the
  requested cutoff, and report restoration avoids duplicate trend traces.
- Closing cancels outstanding work and waits for workers to finish without
  destroying running threads.

## Multi-file import

- Both raw and acceleration/SUP dialogs support selecting multiple files.
  Each raw file creates its GB/ASTM pair; each SUP file creates one recording.
- The background reader reports file counts and filenames, preserves the
  selected-file sequence, and continues after individual parse failures.
  A single completion summary lists failures; full errors are available in
  its tooltip. Duplicate basenames retain independent display names and colours.
- Complete files are added to the GUI one at a time, then charts refresh once
  for the finished batch. Report capture pauses these GUI additions safely.
- Cancellation during reading discards the staged batch. During GUI addition,
  completed files remain and pending files are skipped. Clear All and shutdown
  stop queued additions; cancellation also discards report-deferred imports.
  Import diagnostics retain filenames and errors without retaining cancelled
  recording data buffers.

## Performance

Measurements on the local Windows environment, using a 141,994-sample recording
at 500 Hz. Median of three calls, comparing against commit `5161cb5` in the same
process. These are component timings, not interactive frame-rate benchmarks.
Qt plotting was measured with its offscreen renderer.

| Operation | Before | After | Speedup |
| --- | ---: | ---: | ---: |
| Filter all three axes | 14.67 ms | 11.43 ms | 1.3× |
| GB sustained calculation, 0.2 s | 79.75 ms | 24.41 ms | 3.3× |
| Z-reversal detection | 247.62 ms | 5.29 ms | 46.8× |
| Eggplot update | 813.14 ms | 436.82 ms | 1.9× |

Filtering computes each axis once and shares it between filtered/calibrated
frames. GB avoids Python list conversions. Z-reversal detection uses rolling
extrema to reject quiet windows before the Python event loop. The Eggplot-only
canvas skips 17 unused sustained-window calculations. Cutoff changes coalesce
for 250 ms before applying. Small/offscreen series avoid OpenGL setup.
Only the selected recording computes inspector statistics, and unchanged
selections reuse the existing inspector. Importing more files does not rebuild
all the other files' statistics.

The asynchronous smoke run took about 1.93 s for A3 (141,994 samples per group)
and 1.17 s for A4 (70,997 samples per group) to import, plot Trend and prepare
analysis. A 10 ms GUI timer fired 72 and 28 times respectively during these
runs. These are observed workflow timings, not latency guarantees; the figures
exclude painting the hidden standard views until those pages are opened.

## Validation and remaining opportunities

- Regression suite covers reordered/missing channels, non-500 Hz acquisition,
  pre-trigger export round-trips, fractional/invalid trimming, filter caching,
  numeric edits, peak times, sustained extrema against a brute-force reference,
  reversal pre-windows, selector state, clear/reset, analysis state, generation
  errors, cutoff debounce, theme synchronization and layout containment.
- All 67 regression and integration checks pass. Additional coverage verifies
  worker/GUI thread separation, rapid settings changes, out-of-order results,
  cancellation, clear/reset, close during work, snapshot isolation, failures,
  pending filter statistics/export, report refresh suspension and trend repair.
- Batch coverage includes real raw/SUP parsing, multiple selection dialogs,
  per-file failure isolation, duplicate basenames, cancellation in both phases,
  report pausing, clear/reset and release of uncommitted recording buffers.
  The supplied A3 and A4 raw files were also imported together as four signals
  and rendered across all seven views in both themes.
- Rendered all seven views in both themes with the supplied A3 and A4 recordings;
  previews and timing evidence are kept in the parent `gui_review` folder.
- Chart creation, painting, overlays and report capture still use the GUI
  thread. Large full-resolution traces can cause short pauses while drawing.
  Display-only extrema-preserving decimation is a remaining opportunity;
  full-resolution numerical analysis should continue using original samples.
- Validation used Python 3.14, PySide6 6.11.1 and pandas 2.3.3. Native GPU/driver
  behavior and packaged executable builds were not part of the headless checks.
  Numerical regression checks do not constitute a standards compliance audit.
