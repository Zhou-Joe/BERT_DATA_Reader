# coding: utf-8
"""
Modernized dialog classes for BERT Reader.

All dialogs use QDialogButtonBox and QFormLayout for consistent look & feel.
"""

from __future__ import annotations

import copy
import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFormLayout, QVBoxLayout, QHBoxLayout,
    QGroupBox, QLabel, QWidget,
)

import qfluentwidgets as qfw
from qfluentwidgets import (
    SpinBox, DoubleSpinBox, LineEdit, ComboBox, CheckBox,
    PushButton, PrimaryPushButton, SmoothScrollArea,
)

import docx
from docx.shared import Inches, RGBColor

from theme import theme_manager
from sci_calculation import allowance_at_t0
from accdata import detect_test_boundaries


class _ThemedDialog(QDialog):
    """QDialog that paints a theme-aware background AND palette.

    Plain QDialog keeps the default light window colour even in dark mode
    (qfluentwidgets only restyles Fluent widgets, not raw QDialogs). This base
    paints the current theme's surface colour so the dialog matches the rest
    of the dark UI, sets the widget palette so child widgets (QLabel, QGroupBox
    titles, QFormLayout labels) inherit readable text colours, and re-applies
    both on theme switches.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._apply_theme_palette()

    def _apply_theme_palette(self) -> None:
        from PySide6.QtGui import QPalette
        tk = theme_manager.tokens
        pal = QPalette()
        bg = QColor(tk["surface"])
        pal.setColor(QPalette.ColorRole.Window, bg)
        pal.setColor(QPalette.ColorRole.Base, bg)
        pal.setColor(QPalette.ColorRole.AlternateBase, QColor(tk["surface_alt"]))
        txt = QColor(tk["text"])
        pal.setColor(QPalette.ColorRole.WindowText, txt)
        pal.setColor(QPalette.ColorRole.Text, txt)
        pal.setColor(QPalette.ColorRole.ToolTipText, txt)
        pal.setColor(QPalette.ColorRole.ButtonText, txt)
        # disabled
        pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText,
                     QColor(tk["text_disabled"]))
        pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text,
                     QColor(tk["text_disabled"]))
        self.setPalette(pal)
        # propagate to existing children
        for child in self.findChildren(QWidget):
            child.setPalette(pal)

    def paintEvent(self, event) -> None:
        from PySide6.QtGui import QPainter
        tk = theme_manager.tokens
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(tk["surface"]))
        painter.end()

    def refresh_theme(self) -> None:
        self._apply_theme_palette()
        self.update()


class _ThemedWidget(QWidget):
    """QWidget variant of _ThemedDialog for the non-modal EditDataDialog."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._apply_theme_palette()

    _apply_theme_palette = _ThemedDialog._apply_theme_palette

    def paintEvent(self, event) -> None:
        from PySide6.QtGui import QPainter
        tk = theme_manager.tokens
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(tk["surface"]))
        painter.end()

    def refresh_theme(self) -> None:
        self._apply_theme_palette()
        self.update()


def _apply_dialog_margins(layout: QVBoxLayout) -> None:
    """Comfortable Fluent spacing for dialog root layouts."""
    layout.setContentsMargins(20, 20, 20, 20)
    layout.setSpacing(12)


# ---------------------------------------------------------------------------
# Truncate Data Dialog
# ---------------------------------------------------------------------------

class TruncateTimeDialog(_ThemedDialog):
    """Dialog for truncating acceleration data to one or more time windows.

    A segment-count selector at the top controls how many (start, end) input
    pairs are shown. On OK:
      - 1 segment  → truncate the source dataset in place (legacy behaviour).
      - N segments → deep-copy the source dataset once per segment, truncate
        each copy to its window, and append each as a brand-new dataset in the
        sidebar. The original dataset is left untouched.
    """

    def __init__(self, mainui, data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Truncate Data")
        self.setMinimumWidth(360)
        self._mainui = mainui
        self._data = data

        layout = QVBoxLayout(self)
        _apply_dialog_margins(layout)

        # --- Segment-count selector -------------------------------------
        count_row = QFormLayout()
        self._count = SpinBox()
        self._count.setRange(1, 50)
        self._count.setValue(1)
        self._count.valueChanged.connect(self._rebuild_time_rows)
        count_row.addRow("Number of segments:", self._count)
        layout.addLayout(count_row)

        # --- Auto-detect + sensitivity ----------------------------------
        # Auto-Detect runs the stop-time detector on the recording and fills in
        # the segment start/end times. The min-gap slider controls how long a
        # quiet period must be to count as a test boundary (longer = fewer,
        # coarser splits; shorter = more, finer splits). Default 10s matches
        # the validated separation between real boundaries (~30s) and
        # intra-test pauses (~3s).
        detect_row = QHBoxLayout()
        detect_row.setSpacing(8)
        self._auto_detect_btn = PrimaryPushButton("Auto-Detect")
        self._auto_detect_btn.setStatusTip(
            "Detect stop-times between tests and fill the segments")
        self._auto_detect_btn.clicked.connect(self._on_auto_detect)
        detect_row.addWidget(self._auto_detect_btn)
        detect_row.addWidget(QLabel("Min gap:"))
        self._min_gap_spin = DoubleSpinBox()
        self._min_gap_spin.setRange(1.0, 120.0)
        self._min_gap_spin.setSingleStep(1.0)
        self._min_gap_spin.setDecimals(1)
        self._min_gap_spin.setSuffix(" s")
        self._min_gap_spin.setValue(10.0)
        self._min_gap_spin.setToolTip(
            "Minimum quiet-run duration (s) that counts as a test boundary.\n"
            "Higher = fewer splits; lower = more splits.")
        detect_row.addWidget(self._min_gap_spin)
        detect_row.addStretch(1)
        layout.addLayout(detect_row)

        # --- Replace-original toggle -------------------------------------
        # When ON (default), applying a multi-segment truncate removes the
        # source dataset after adding the truncated segments, so the original
        # (un-split) curve doesn't linger alongside its splits. Only applies to
        # the multi-segment path (1 segment mutates the source in place anyway).
        self._replace_original = CheckBox("Replace original after split")
        self._replace_original.setChecked(True)
        self._replace_original.setToolTip(
            "Remove the source dataset once the truncated segments are added,\n"
            "so only the split results remain. Uncheck to keep the original.")
        layout.addWidget(self._replace_original)

        # --- Scrollable time-window form --------------------------------
        # A scroll area keeps the dialog usable when many segments are added.
        self._scroll = SmoothScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(SmoothScrollArea.Shape.NoFrame)

        form_host = QWidget()
        self._form = QFormLayout(form_host)
        self._form.setContentsMargins(0, 0, 0, 0)
        self._form.setSpacing(8)

        # Per-segment start/end spinboxes, created/destroyed by _rebuild.
        self._start_spins: list = []
        self._end_spins: list = []

        self._scroll.setWidget(form_host)
        layout.addWidget(self._scroll)

        # Data duration (seconds) — used to seed sensible end-time defaults so
        # the user doesn't have to look up how long the recording is.
        try:
            self._data_duration = float(self._data.data["Time"].iloc[-1])
        except Exception:
            self._data_duration = 100000.0

        # Build the initial single row.
        self._rebuild_time_rows(1)
        # Seed a useful default for the first segment: end = full duration, so
        # segment 2+ auto-fill their start from it as the user adds rows.
        if self._end_spins:
            self._end_spins[0].setValue(int(self._data_duration))
        # Establish Tab-key navigation order across every input (see
        # _set_tab_order). Re-run whenever rows change (it depends on row count).
        self._set_tab_order()

        # --- OK / Cancel ------------------------------------------------
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------
    # Dynamic segment rows
    # ------------------------------------------------------------------

    def _rebuild_time_rows(self, n: int) -> None:
        """Rebuild exactly *n* (start, end) rows in the form.

        PRESERVES existing start/end values across the rebuild (the old code
        wiped everything, which would discard segment 1's end just when the
        user wants segment 2 to inherit it). New rows beyond the previous count
        auto-fill their start time from the PREVIOUS segment's end — so defining
        contiguous laps requires entering only the end times.
        """
        # Snapshot the previous values before tearing the rows down.
        prev_start = [sp.value() for sp in self._start_spins]
        prev_end = [sp.value() for sp in self._end_spins]

        while self._form.rowCount():
            self._form.removeRow(0)
        self._start_spins.clear()
        self._end_spins.clear()

        for i in range(n):
            start = SpinBox()
            start.setRange(-100000, 100000)
            start.setSuffix(" s")
            end = SpinBox()
            end.setRange(-100000, 100000)
            end.setSuffix(" s")

            # Restore the previous value if this row existed before; otherwise
            # auto-fill: start = previous segment's end (contiguous laps),
            # end = full duration (a reasonable default).
            if i < len(prev_start):
                start.setValue(prev_start[i])
                end.setValue(prev_end[i])
            else:
                # Newly added row: start = the previous row's END. Use the
                # freshly-restored spinbox (prev_end may be shorter than i),
                # so the chain stays contiguous even when the user jumps from
                # e.g. 1 segment straight to 3.
                prev_end_val = (self._end_spins[i - 1].value()
                                if i > 0 and self._end_spins else 0)
                start.setValue(prev_end_val)
                end.setValue(int(self._data_duration))

            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(8)
            row.addWidget(start)
            row.addWidget(end, 1)
            self._form.addRow(f"Segment {i + 1}  (start / end):", row)
            self._start_spins.append(start)
            self._end_spins.append(end)

        # Cascade: editing an end time pushes it into the NEXT segment's start
        # so contiguous laps stay linked as the user types. Connected after the
        # rebuild so each end spinbox targets the correct next start spinbox.
        # Spinboxes are freshly created each rebuild, so there's nothing to
        # disconnect — connecting here is enough.
        for i, end_sp in enumerate(self._end_spins[:-1]):
            next_start = self._start_spins[i + 1]
            end_sp.valueChanged.connect(
                lambda v, ns=next_start: ns.setValue(v))

        self._adjust_height()
        self._set_tab_order()

    def _set_tab_order(self) -> None:
        """Set Tab-key focus order: segment 1 start → end → segment 2 start →
        end → … so the user can walk every input with the keyboard only.
        """
        prev = None
        for s, e in zip(self._start_spins, self._end_spins):
            if prev is not None:
                self.setTabOrder(prev, s)
            self.setTabOrder(s, e)
            prev = e

    def _adjust_height(self) -> None:
        """Keep the dialog a sensible height for the current segment count."""
        n = self._count.value()
        # ~56px per segment row + ~180px chrome; capped so huge counts scroll.
        self._scroll.setMinimumHeight(min(56 * n + 40, 320))

    def _on_auto_detect(self) -> None:
        """Run the stop-time detector and fill the segment rows.

        Detects test boundaries from quiet periods in the recording, sets the
        segment count, and populates each segment's start/end times. The
        values remain user-editable afterwards (this just pre-fills them).
        """
        if self._data is None:
            return
        try:
            min_gap = float(self._min_gap_spin.value())
            tests = detect_test_boundaries(self._data, min_gap_s=min_gap)
        except Exception as e:
            print(f"Auto-detect error: {e}")
            return
        if not tests:
            self._mainui.statusBar().showMessage(
                "No tests detected (recording too short)", 4000)
            return
        # Set the segment count (triggers _rebuild_time_rows, which preserves
        # old values — then we overwrite below). Cap at the spinbox maximum.
        self._count.setValue(len(tests))
        # Now fill each segment's start/end with the detected boundaries.
        for i, (s, e) in enumerate(tests):
            if i >= len(self._start_spins):
                break
            self._start_spins[i].setValue(s)
            self._end_spins[i].setValue(e)
        self._mainui.statusBar().showMessage(
            f"Auto-detected {len(tests)} test(s)", 4000)

    # ------------------------------------------------------------------
    # Apply
    # ------------------------------------------------------------------

    def _on_ok(self) -> None:
        try:
            requested = self._count.value()
            pairs = [
                (self._start_spins[i].value(), self._end_spins[i].value())
                for i in range(requested)
            ]
            # Drop empty (start>=end) windows; treat them as "unused".
            valid = [(s, e) for s, e in pairs if e > s]
            if not valid:
                self.reject()
                return

            if requested == 1:
                # Legacy single-segment path: mutate the source in place.
                s, e = valid[0]
                self._data.truncate_data(starttime=s, endtime=e,
                                         cutoff=self._mainui.cutoff)
                self._mainui.trendplot.initplotxyz()
                self._mainui.clear_sidebar()
                self._mainui.replot_all()
            else:
                # Multi-segment: each valid window becomes its own new dataset.
                base_name = self._data.filename
                # Remember the source's position so we can drop it afterwards
                # (the clones are appended after it, so its index is stable).
                src_order = self._mainui._plot_order_of(base_name)
                for i, (s, e) in enumerate(valid, 1):
                    clone = copy.deepcopy(self._data)
                    clone.truncate_data(starttime=s, endtime=e,
                                        cutoff=self._mainui.cutoff)
                    clone.filename = f"{base_name} [Trunc {s}-{e}s #{i}]"
                    self._mainui.set_datalist(clone)
                    self._mainui.add_trend_data(
                        data=self._mainui.datalist[-1],
                        plottype=self._mainui.plottype,
                        plot_order=len(self._mainui.datalist),
                    )
                # Optionally remove the source so only the split results remain.
                if (self._replace_original.isChecked() and src_order > 0):
                    self._mainui._remove_dataset(src_order)
                self._mainui.statusBar().showMessage(
                    f"Added {len(valid)} truncated datasets", 4000)
            self.accept()
        except Exception as e:
            print(f"Truncate error: {e}")


# ---------------------------------------------------------------------------
# Export Dialog
# ---------------------------------------------------------------------------

class ExportDialog(_ThemedDialog):
    """Dialog for exporting acceleration data to file."""

    def __init__(self, mainui, data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export")
        self.setMinimumWidth(360)
        self._mainui = mainui
        self._data = data

        layout = QVBoxLayout(self)
        _apply_dialog_margins(layout)

        # Export type
        self._export_type = ComboBox()
        self._export_type.addItems([
            "Formated Data for Newtonviewer",
            "Filtered Clean Data",
        ])

        # Filename + suffix (combined row — bug fix: previously the
        # filename field was added twice, once on its own row and again
        # inside this suffix layout, producing a duplicated control).
        self._filename = LineEdit()
        self._suffix = ComboBox()
        self._suffix.addItems([".sup", ".sups", ".txt"])

        form = QFormLayout()
        form.addRow("Export Type:", self._export_type)

        suffix_layout = QHBoxLayout()
        suffix_layout.setContentsMargins(0, 0, 0, 0)
        suffix_layout.addWidget(self._filename, 1)
        suffix_layout.addWidget(self._suffix, 0)
        form.addRow("File Name:", suffix_layout)

        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_ok(self) -> None:
        try:
            name = self._filename.text()
            if not name:
                return

            if self._export_type.currentText() == "Filtered Clean Data":
                self._data.export_filter(
                    plottype=self._mainui.plottype,
                    path=name + "_filtered.txt",
                )
            else:
                self._data.export_to_file(
                    path=name + self._suffix.currentText(),
                )

            self._mainui.statusBar().showMessage(
                f"Exported: {name}", 5000
            )
            self.accept()
        except Exception as e:
            print(f"Export error: {e}")


# ---------------------------------------------------------------------------
# Edit Data Dialog
# ---------------------------------------------------------------------------

class EditDataDialog(_ThemedWidget):
    """Non-modal dialog for adding/multiplying offsets to data."""

    def __init__(self, mainui, data, parent=None):
        super().__init__(parent)
        self.setObjectName("popoutWindow")
        self.setWindowTitle("Edit Data")
        self.setMinimumWidth(340)
        self._mainui = mainui
        self._data = data

        layout = QVBoxLayout(self)
        _apply_dialog_margins(layout)

        # --- Add section ---
        add_group = QGroupBox("Add Offset")
        add_form = QFormLayout()
        self._add_x = DoubleSpinBox()
        self._add_x.setRange(-10, 10)
        self._add_y = DoubleSpinBox()
        self._add_y.setRange(-10, 10)
        self._add_z = DoubleSpinBox()
        self._add_z.setRange(-10, 10)
        add_form.addRow("Fore/Back (x):", self._add_x)
        add_form.addRow("Left/Right (y):", self._add_y)
        add_form.addRow("Up/Down (z):", self._add_z)
        add_group.setLayout(add_form)
        layout.addWidget(add_group)

        btn_add = PrimaryPushButton("Apply Add")
        btn_add.clicked.connect(self._on_add)
        layout.addWidget(btn_add)

        # --- Multiply section ---
        mul_group = QGroupBox("Multiply Scale")
        mul_form = QFormLayout()
        self._mul_x = DoubleSpinBox()
        self._mul_x.setRange(-10, 10)
        self._mul_x.setValue(1.0)
        self._mul_y = DoubleSpinBox()
        self._mul_y.setRange(-10, 10)
        self._mul_y.setValue(1.0)
        self._mul_z = DoubleSpinBox()
        self._mul_z.setRange(-10, 10)
        self._mul_z.setValue(1.0)
        mul_form.addRow("Fore/Back (x):", self._mul_x)
        mul_form.addRow("Left/Right (y):", self._mul_y)
        mul_form.addRow("Up/Down (z):", self._mul_z)
        mul_group.setLayout(mul_form)
        layout.addWidget(mul_group)

        btn_mul = PrimaryPushButton("Apply Multiply")
        btn_mul.clicked.connect(self._on_multiply)
        layout.addWidget(btn_mul)

        btn_close = PushButton("Close")
        btn_close.clicked.connect(self.close)
        layout.addWidget(btn_close)

    def _on_add(self) -> None:
        try:
            self._data.edit_data(
                method="add",
                value_array=[self._add_x.value(), self._add_y.value(),
                             self._add_z.value()],
                cutoff=self._mainui.cutoff,
            )
            self._mainui.trendplot.initplotxyz()
            self._mainui.clear_sidebar()
            self._mainui.replot_all()
        except Exception as e:
            print(f"Edit add error: {e}")

    def _on_multiply(self) -> None:
        try:
            self._data.edit_data(
                method="multiply",
                value_array=[self._mul_x.value(), self._mul_y.value(),
                             self._mul_z.value()],
                cutoff=self._mainui.cutoff,
            )
            self._mainui.trendplot.initplotxyz()
            self._mainui.clear_sidebar()
            self._mainui.replot_all()
        except Exception as e:
            print(f"Edit multiply error: {e}")


# ---------------------------------------------------------------------------
# Edit Angle Dialog
# ---------------------------------------------------------------------------

class EditAngleDialog(_ThemedDialog):
    """Dialog for editing sensor orientation angles."""

    def __init__(self, data, mainui, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Angle")
        self.setMinimumWidth(320)
        self._data = data
        self._mainui = mainui

        layout = QVBoxLayout(self)
        _apply_dialog_margins(layout)

        form = QFormLayout()
        self._pitch = SpinBox()
        self._pitch.setRange(-180, 180)
        self._pitch.setSuffix(" °")
        self._seatback = SpinBox()
        self._seatback.setRange(-180, 180)
        self._seatback.setSuffix(" °")
        self._roll = SpinBox()
        self._roll.setRange(-180, 180)
        self._roll.setSuffix(" °")
        self._yaw = SpinBox()
        self._yaw.setRange(-180, 180)
        self._yaw.setSuffix(" °")

        form.addRow("Pitch Angle:", self._pitch)
        form.addRow("Seatback Angle:", self._seatback)
        form.addRow("Roll Angle:", self._roll)
        form.addRow("Yaw Angle:", self._yaw)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_ok(self) -> None:
        try:
            self._data.reformat(
                overwrite=True,
                setting_angle=True,
                pitch_angle=self._pitch.value(),
                seatback_angle=self._seatback.value(),
                roll_angle=self._roll.value(),
                yaw_angle=self._yaw.value(),
                cutoff=self._mainui.cutoff,
            )
            self._mainui.trendplot.initplotxyz()
            self._mainui.clear_sidebar()
            self._mainui.replot_all()
            self.accept()
        except Exception as e:
            print(f"Angle edit error: {e}")


# ---------------------------------------------------------------------------
# Cutoff Frequency Dialog
# ---------------------------------------------------------------------------

class CutoffFrequencyDialog(_ThemedDialog):
    """Dialog for setting the low-pass filter cutoff frequency."""

    def __init__(self, mainui, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cutoff Frequency")
        self.setMinimumWidth(320)
        self._mainui = mainui

        layout = QVBoxLayout(self)
        _apply_dialog_margins(layout)

        form = QFormLayout()
        self._freq = SpinBox()
        self._freq.setRange(1, 249)
        self._freq.setValue(mainui.cutoff)
        self._freq.setSuffix(" Hz")
        form.addRow("Cutoff Frequency:", self._freq)
        layout.addLayout(form)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_ok(self) -> None:
        try:
            freq = self._freq.value()
            self._mainui.set_cutoff(freq)
            self._mainui.update_status_label()
            self._mainui.trendplot.initplotxyz()
            self._mainui.clear_sidebar()
            self._mainui.replot_all()
            self._mainui.statusBar().showMessage(
                f"Cutoff frequency set to {freq} Hz", 4000
            )
            self.accept()
        except Exception as e:
            print(f"Cutoff error: {e}")


# ---------------------------------------------------------------------------
# Report Generation Dialog
# ---------------------------------------------------------------------------

class ReportDialog(_ThemedDialog):
    """Dialog for selecting datasets and generating a Word report."""

    def __init__(self, mainui, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generate Report")
        self.setMinimumWidth(420)
        self._mainui = mainui
        self._checkboxes: list = []

        layout = QVBoxLayout(self)
        _apply_dialog_margins(layout)

        layout.addWidget(QLabel("Select Acceleration Records to Generate:"))
        notice = QLabel(
            "Please specify restraint type in advance for ASTM Standard Fit "
            "and make sure to use filtered and zero-adjusted data."
        )
        notice.setWordWrap(True)
        notice.setStyleSheet(
            "color: {warn}; background-color: {glass}; "
            "border: 1px solid {warn}; border-top: 1px solid {shine}; "
            "border-radius: 12px; padding: 8px; font-style: italic;".format(
                warn=theme_manager.t("warning"),
                glass=theme_manager.t("glass_bg_subtle"),
                shine=theme_manager.t("glass_shine"),
            )
        )
        layout.addWidget(notice)

        # Scrollable checkbox list
        scroll = SmoothScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        for name in mainui.datadict.keys():
            cb = CheckBox(name)
            self._checkboxes.append(cb)
            scroll_layout.addWidget(cb)
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_ok(self) -> None:
        data_list = [cb.text() for cb in self._checkboxes if cb.isChecked()]
        if not data_list:
            return

        mu = self._mainui
        doc = docx.Document()

        # Snapshot each canvas's analysis state so we can FORCE Analyze on for
        # the report images and restore the user's prior choice afterwards (no
        # UI state leak). Keys match the canvas analyze methods. Analysis is now
        # driven by a single shared Analyze button (per-canvas flags), so we
        # read the canvas flags directly rather than per-page buttons.
        analyze_canvas_attr = {
            "sust": "canvasASTMSust",
            "egg": "canvasASTMEgg",
            "gb": "canvasGB",
            "acczone": "canvasAccZone",
            "zr": "canvasZReversal",
            "xr": "canvasXYReversal",
        }
        prior = {k: mu._canvas_analysis_active(attr)
                 for k, attr in analyze_canvas_attr.items()}

        try:
            for data_name in data_list:
                data = mu.datadict.get(data_name)
                if data is None:
                    continue
                restraint = mu.select_restraint.currentText()
                cond = mu.select_cond.currentText()
                height = mu.height_input.value()
                prone = mu._prone_btn.isChecked()

                # --- Plot every chart for this dataset -----------------------
                mu.plotASTM_checkbox(canvas=mu.canvasASTMSust, name=data_name)
                mu.plotASTM_checkbox(canvas=mu.canvasASTMEgg, name=data_name)
                mu.plotGB_checkbox(canvas=mu.canvasGB, name=data_name)
                mu.plotGB_checkbox(canvas=mu.canvasAccZone, name=data_name)
                mu.replot_single(data_name)

                # Reversal charts (new in the report). Plot the standard frame
                # at the current cutoff, then set allowance/deadband so Analyze
                # has the right limits to check against.
                if hasattr(data, "calculate_sampling_frequency"):
                    data.calculate_sampling_frequency()
                data.filter_data(cutoff=mu.cutoff)
                fs = getattr(data, "fs", 500)
                df_rev = mu._data_for_plottype(data, "Standard")
                color = mu.dataset_colors.get(data_name)
                mu.canvasZReversal.plot_data(df_rev, color, fs)
                mu.canvasXYReversal.plot_data(df_rev, color, fs)
                mu.canvasXYReversal.set_allowance({
                    "x": allowance_at_t0("x", restraint, cond, height),
                    "y": allowance_at_t0("y", restraint, cond, height),
                })
                mu.canvasXYReversal.set_deadband(0.05)

                # --- Force Analyze ON for every chart so the image includes
                #     the overlays (markers/bands/zone outline). -------------
                mu.canvasASTMSust.toggle_sust_analysis(
                    True, restraint, cond, height, prone)
                mu.canvasASTMEgg.toggle_egg_analysis(True)
                mu.canvasGB.toggle_comb_analysis(True)
                mu.canvasAccZone.toggle_analysis(True)
                mu.canvasZReversal.toggle_analysis(True)
                mu.canvasXYReversal.toggle_analysis(True)

                # --- Gather verdicts (pass/fail + message) -------------------
                verdicts = self._collect_verdicts(mu)

                # --- Capture all 7 chart images -----------------------------
                # 1=Sustained 2=Eggplot 3=GB 4=AccZone 5=Trend
                # 6=Z Reversal 7=X/Y Reversal
                img_idx = 0
                for chart_view in [
                    mu.canvasASTMSust, mu.canvasASTMEgg, mu.canvasGB,
                    mu.canvasAccZone, mu.trendplot,
                    mu.canvasZReversal, mu.canvasXYReversal,
                ]:
                    img_idx += 1
                    try:
                        chart_view.grab().save(f"img{img_idx}.jpg", "JPG", 95)
                    except Exception as e:
                        print(f"Image save error: {e}")

                mu.clear_sidebar()
                mu.replot_all()
                self._write_doc_section(doc, data, data_name, verdicts)

            doc.save("acceleration_plot.docx")
            mu.statusBar().showMessage(
                "Report generated: acceleration_plot.docx", 5000)
            self.accept()

        except Exception as e:
            print(f"Report error: {e}")

        finally:
            # Restore the user's prior Analyze state (and the live canvases).
            self._restore_analyze(mu, prior)

    @staticmethod
    def _collect_verdicts(mu) -> dict:
        """Build {chart_key: (passed_bool, message_str)} from canvas state.

        Called AFTER every canvas has been plotted + analyzed. Each verdict
        becomes a green/red line below that chart's image in the report.
        """
        v: dict = {}

        # ASTM Sustained — failed axes (e.g. "ax (+)").
        sust = sorted(getattr(mu.canvasASTMSust, "sust_failed_rules", set()))
        if sust:
            v["sust"] = (False, "exceeded limit: " + ", ".join(sust))
        else:
            v["sust"] = (True, "all axes within limit")

        # ASTM Eggplot — failed boundary rules.
        egg = sorted(getattr(mu.canvasASTMEgg, "failed_rules", set()))
        if egg:
            txt = ", ".join(r.replace("(0.2s)", "0.2s").replace("(0s)", "0s")
                            for r in egg)
            v["egg"] = (False, "out of boundary: " + txt)
        else:
            v["egg"] = (True, "all points within boundary")

        # GB 8408 — failed dt loops.
        gb = sorted(getattr(mu.canvasGB, "failed_rules", set()))
        if gb:
            txt = ", ".join(r.replace("(0.2s)", "0.2s").replace("(0.1s)", "0.1s")
                            .replace("(0.05s)", "0.05s").replace("(0s)", "0s")
                            for r in gb)
            v["gb"] = (False, "exceeded: " + txt)
        else:
            v["gb"] = (True, "all points within loops")

        # AccZone — this is a CLASSIFICATION (which zone the ride falls in),
        # not a pass/fail check, so report it as informational, not PASS/FAIL.
        zn = getattr(mu.canvasAccZone, "_auto_zone_result", None)
        if zn is not None:
            v["acczone"] = ("info", f"Classified as Zone {zn}")
        else:
            v["acczone"] = ("info", "No zone classified")

        # Z Reversal — the rule IS the detection: a recognized acceleration-
        # reversal pattern (down-then-up crossing 0 and exceeding 2 g within the
        # time window) is itself the failure signature, so ANY detected event is
        # a FAIL. Unlike X/Y Reversal there is no per-event allowance check.
        zr_n = int(getattr(mu.canvasZReversal, "event_count", 0))
        if zr_n > 0:
            v["zr"] = (False, f"{zr_n} Z-reversal pattern(s) detected")
        else:
            v["zr"] = (True, "no Z-reversal pattern detected")

        # X/Y Reversal — FAIL if any pair exceeded its effective limit; a
        # detected reversal that stays within its allowance is NOT a violation.
        xr_fail = int(getattr(mu.canvasXYReversal, "fail_count", 0))
        xr_n = int(getattr(mu.canvasXYReversal, "event_count", 0))
        if xr_fail > 0:
            v["xr"] = (False, f"{xr_fail} reversal pair(s) exceeding the limit")
        else:
            v["xr"] = (True, "no reversal pair exceeding the limit "
                       f"({xr_n} detected)")
        return v

    @staticmethod
    def _restore_analyze(mu, prior: dict) -> None:
        """Turn each canvas's analysis back to its pre-report on/off state."""
        try:
            mu.canvasASTMSust.toggle_sust_analysis(
                prior.get("sust", False),
                mu.select_restraint.currentText(),
                mu.select_cond.currentText(),
                mu.height_input.value(),
                mu._prone_btn.isChecked())
        except Exception:
            pass
        try:
            mu.canvasASTMEgg.toggle_egg_analysis(prior.get("egg", False))
        except Exception:
            pass
        try:
            mu.canvasGB.toggle_comb_analysis(prior.get("gb", False))
        except Exception:
            pass
        try:
            mu.canvasAccZone.toggle_analysis(prior.get("acczone", False))
        except Exception:
            pass
        try:
            mu.canvasZReversal.toggle_analysis(prior.get("zr", False))
        except Exception:
            pass
        try:
            mu.canvasXYReversal.toggle_analysis(prior.get("xr", False))
        except Exception:
            pass
        # Analyze is now a global toggle. The report forced it on per-canvas;
        # after restoring each canvas, re-sync the global flag + button to
        # reflect whether the VISIBLE page is still analyzed.
        vis = getattr(mu, "_canvas_analysis_active", None)
        if vis is not None:
            page = mu.stackedWidget.currentWidget()
            canvas_attr = {
                "page_trend": "trendplot",
                "page_astm_sust": "canvasASTMSust",
                "page_astm_egg": "canvasASTMEgg",
                "page_gb": "canvasGB",
                "page_acczone": "canvasAccZone",
                "page_zreversal": "canvasZReversal",
                "page_xyreversal": "canvasXYReversal",
            }.get(page.objectName() if page is not None else "")
            on = bool(vis(canvas_attr)) if canvas_attr else False
            mu._analyze_global_on = on
            btn = getattr(mu, "_analyze_btn", None)
            if btn is not None and btn.isChecked() != on:
                btn.blockSignals(True)
                btn.setChecked(on)
                btn.blockSignals(False)

    @staticmethod
    def _add_image_with_verdict(doc, img_path: str, verdict) -> None:
        """Add a chart image + a coloured status line beneath it.

        ``verdict`` formats:
          • ``(passed_bool, message_str)`` — PASS → green ✓, FAIL → red ✗
          • ``("info", message_str)``       — informational neutral blue ●,
            no pass/fail (used by AccZone, which is a classification not a check)
          • None                           — no line (e.g. the Trend plot)
        """
        doc.add_picture(img_path, width=Inches(6))
        if verdict is None:
            return
        kind, msg = verdict
        para = doc.add_paragraph()
        if kind == "info":
            run = para.add_run(f"\u25cf {msg}")
            run.bold = True
            run.font.color.rgb = RGBColor(0x00, 0x78, 0xD4)   # accent blue
            return
        passed = bool(kind)
        mark = "\u2713 PASS" if passed else "\u2717 FAIL"
        run = para.add_run(f"{mark} — {msg}")
        run.bold = True
        # Green for pass (#107C10), red for fail (#C00000) — readable when
        # printed and on screen.
        run.font.color.rgb = RGBColor(0x10, 0x7C, 0x10) if passed \
            else RGBColor(0xC0, 0x00, 0x00)

    @staticmethod
    def _write_doc_section(doc, data, name: str, verdicts: dict) -> None:
        """Write one data section into the Word document.

        Each chart image is followed by a green ✓ PASS / red ✗ FAIL line built
        from *verdicts* (see ReportDialog._collect_verdicts).
        """
        key_info = data.get_data_stats(data.std_data)
        v = verdicts or {}

        doc.add_heading(f"Acceleration File Name: {name}", level=2)
        doc.add_heading("Trend Plot:", level=4)
        # Trend has no analysis → no verdict line.
        ReportDialog._add_image_with_verdict(doc, "img5.jpg", None)

        # Stats table
        table = doc.add_table(rows=1, cols=6, style="Table Grid")
        table.autofit = True
        hdr = table.rows[0].cells
        hdr[0].text = ""
        for j, col_name in enumerate(key_info.columns):
            hdr[j + 1].text = col_name

        for i in range(key_info.shape[0]):
            row = table.add_row().cells
            row[0].text = str(key_info.index[i])
            for j in range(key_info.shape[1]):
                row[j + 1].text = str(key_info.iloc[i, j])

        doc.add_heading("ASTM F2291 Sustained Acceleration:", level=4)
        ReportDialog._add_image_with_verdict(doc, "img1.jpg", v.get("sust"))
        doc.add_heading("ASTM F2291 Eggplot:", level=4)
        ReportDialog._add_image_with_verdict(doc, "img2.jpg", v.get("egg"))
        doc.add_heading("GB 8408 Fitting:", level=4)
        ReportDialog._add_image_with_verdict(doc, "img3.jpg", v.get("gb"))
        doc.add_heading("Acceleration Zone Plot:", level=4)
        ReportDialog._add_image_with_verdict(doc, "img4.jpg", v.get("acczone"))
        doc.add_heading("Z-Axis Reversal Analysis:", level=4)
        ReportDialog._add_image_with_verdict(doc, "img6.jpg", v.get("zr"))
        doc.add_heading("X/Y-Axis Reversal Analysis:", level=4)
        ReportDialog._add_image_with_verdict(doc, "img7.jpg", v.get("xr"))

        # Cleanup temp images (7 charts now).
        for i in range(1, 8):
            try:
                os.remove(f"img{i}.jpg")
            except OSError:
                pass
