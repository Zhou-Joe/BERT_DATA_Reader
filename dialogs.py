# coding: utf-8
"""
Modernized dialog classes for BERT Reader.

All dialogs use QDialogButtonBox and QFormLayout for consistent look & feel.
"""

from __future__ import annotations

import copy
import os
import shutil
import tempfile
from datetime import datetime

from PySide6.QtCore import Qt, QStandardPaths, QCoreApplication, QRectF
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFileDialog, QFormLayout, QVBoxLayout,
    QHBoxLayout, QGroupBox, QLabel, QWidget,
)

import qfluentwidgets as qfw
from qfluentwidgets import (
    SpinBox, DoubleSpinBox, LineEdit, ComboBox, CheckBox,
    PushButton, PrimaryPushButton, SmoothScrollArea,
)

import docx
from docx.shared import Inches, RGBColor, Pt

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
            self._data_start = float(self._data.data["Time"].iloc[0])
            self._data_duration = float(self._data.data["Time"].iloc[-1]) + 1 / self._data.fs
        except Exception:
            self._data_start = 0.0
            self._data_duration = 100000.0

        # Build the initial single row.
        self._rebuild_time_rows(1)
        # Seed a useful default for the first segment: end = full duration, so
        # segment 2+ auto-fill their start from it as the user adds rows.
        if self._end_spins:
            self._end_spins[0].setValue(self._data_duration)
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
            start = DoubleSpinBox()
            start.setDecimals(4)
            start.setSingleStep(1 / self._data.fs)
            start.setRange(-100000, 100000)
            start.setSuffix(" s")
            end = DoubleSpinBox()
            end.setDecimals(4)
            end.setSingleStep(1 / self._data.fs)
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
                                if i > 0 and self._end_spins else self._data_start)
                start.setValue(prev_end_val)
                end.setValue(self._data_duration)

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
            self._mainui.statusBar().showMessage(f"Cannot truncate: {e}", 6000)


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
        name = self._filename.text().strip()
        if not name:
            # 空文件名：明确提示并保持对话框打开（旧代码静默 return）。
            qfw.InfoBar.warning(
                title="Export",
                content="请输入文件名",
                orient=Qt.Orientation.Horizontal,
                isClosable=True,
                duration=3000,
                position=qfw.InfoBarPosition.TOP,
                parent=self,
            )
            return

        if (self._export_type.currentText() == 'Filtered Clean Data'
                and self._mainui.plottype != 'Raw'
                and self._data._filtered_cutoff != self._mainui.cutoff_spin.value()):
            qfw.InfoBar.warning(
                title='Filter update pending',
                content='Wait for the current filter results, then export again.',
                duration=4000, parent=self)
            return

        # 用系统保存对话框选择完整路径（旧代码把相对名直接写进进程 CWD）。
        # 默认名 = 用户输入的文件名 + 后缀下拉当前值；"Filtered Clean Data"
        # 分支同样尊重后缀下拉，不再硬编码 name+"_filtered.txt"。
        suggested = name + self._suffix.currentText()
        docs_dir = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.DocumentsLocation
        ) or os.path.expanduser("~")
        default_path = os.path.join(docs_dir, suggested)
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save Export", default_path,
            "Export Files (*.sup *.sups *.txt);;All Files (*.*)")
        if not out_path:
            return  # 用户取消了保存对话框，Export 对话框保持打开

        try:
            if self._export_type.currentText() == "Filtered Clean Data":
                self._data.export_filter(
                    plottype=self._mainui.plottype,
                    path=out_path,
                )
            else:
                self._data.export_to_file(
                    path=out_path,
                )

            self._mainui.statusBar().showMessage(
                f"Exported: {os.path.abspath(out_path)}", 5000
            )
            self.accept()
        except Exception as e:
            # 写盘失败（含文件被 Excel 等占用）：InfoBar 让失败可见（打包版无
            # 控制台，仅 print 用户看不到），且不 accept —— 对话框保持打开。
            print(f"Export error: {e}")
            qfw.InfoBar.error(
                title="Export Failed",
                content=f"导出失败：{e}",
                orient=Qt.Orientation.Horizontal,
                isClosable=True,
                duration=-1,
                position=qfw.InfoBarPosition.TOP,
                parent=self,
            )


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

def _render_view_figure(view, out_path: str, w: int, h: int,
                        before_render=None) -> bool:
    """把一个 QChartView 临时固定成报告画幅 (w, h) 后截图渲染成 JPG。

    两条已踩过的坑，说明为什么是"固定视图尺寸 + grab"：
    1. 直接截 UI 里的子图控件：宽高比被布局绑架（egg 三列并排又窄又高、
       GB 合成图又扁又长），放进 Word 很难看。
    2. 对 chart setGeometry 后 scene.render：Qt Charts 的系列绘制走视图
       的渲染管线，脱离视图直接渲场景只会画出坐标轴/参考线，实测测量
       曲线丢失。

    所以这里把视图本身 setFixedSize 到报告画幅（布局无法改回它），泵事
    件循环让布局和系列落地，再 view.grab() —— 与屏幕显示同一条渲染路
    径，曲线/分析叠加/坐标轴所见即所得，且画幅任意。最后恢复原尺寸。
    ``before_render`` 在固定尺寸后调用（egg 等比例按新画幅重跑）。
    """
    try:
        old_size = view.size()
        view.setFixedSize(w, h)
        QCoreApplication.processEvents()
        # 弃用截图：强制 chart 在新尺寸下完成真实布局。隐藏页/尺寸切换
        # 瞬间 chart.plotArea() 是陈旧值（曾读到 23×21），等比例若据此计
        # 算会得出 ±68 这类荒谬范围；先 grab 一次让布局落地再取值。
        view.grab()
        QCoreApplication.processEvents()
        if before_render is not None:
            try:
                before_render()
            except Exception as e:
                print(f"before_render error: {e}")
            QCoreApplication.processEvents()
        ok = view.grab().save(out_path, "JPG", 95)
        return ok
    except Exception as e:
        print(f"View render error: {e}")
        return False
    finally:
        view.setMinimumSize(0, 0)
        view.setMaximumSize(16777215, 16777215)
        try:
            view.resize(old_size)
            QCoreApplication.processEvents()
        except Exception:
            pass


def _grab_widget(widget, out_path: str) -> bool:
    """整控件截图（仅 Trend 用：三联时序整体对比，构图不受拆分影响）。"""
    try:
        return widget.grab().save(out_path, "JPG", 95)
    except Exception as e:
        print(f"Image save error: {e}")
        return False


def _egg_equalize_to_data(chart):
    """返回闭包：按图上全部系列的实际极值设等刻度范围（报告 egg 专用）。

    应用内的 equalize_eggs 锁定 X 家范围再按 plotArea 像素反推 Y —— 依赖
    plotArea 与视图尺寸一致，隐藏页/尺寸切换瞬间它是陈旧值（曾据此算出
    ±68 的荒谬 Y 范围）。报告版完全数据驱动：所有曲线（蛋形边界 + 测量
    轨迹 + 违规标记）的联合极值各加 8% 边距，units/px = max(ex/pw, ey/ph)
    保证双轴同刻度 —— 两轴都只比实际内容略大，余量多的一边对称留白。
    须在 _render_view_figure 的弃用截图之后调用（plotArea 已真实）。
    """
    def _apply():
        try:
            xs, ys = [], []
            for s in chart.series():
                try:
                    pts = s.pointsVector()
                except Exception:
                    continue
                for p in pts:
                    xs.append(p.x())
                    ys.append(p.y())
            if not xs:
                return
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            # 8% 边距（跨度退化时给绝对下限，避免零跨度除零/贴边）。
            mx = max((xmax - xmin) * 0.08, 0.2)
            my = max((ymax - ymin) * 0.08, 0.2)
            ex = (xmax + mx) - (xmin - mx)
            ey = (ymax + my) - (ymin - my)
            pa = chart.plotArea()
            if pa.width() <= 0 or pa.height() <= 0:
                return
            units = max(ex / pa.width(), ey / pa.height())
            cx = (xmin + xmax) / 2.0
            cy = (ymin + ymax) / 2.0
            for a in chart.axes():
                orient = str(a.orientation())
                if orient.endswith("Horizontal"):
                    a.setRange(cx - units * pa.width() / 2,
                               cx + units * pa.width() / 2)
                elif orient.endswith("Vertical"):
                    a.setRange(cy - units * pa.height() / 2,
                               cy + units * pa.height() / 2)
        except Exception as e:
            print(f"Egg equalize error: {e}")
    return _apply


def _disable_series_opengl(canvases) -> list:
    """临时关闭这些画布上所有 XY 系列的 OpenGL 渲染，返回被切换的系列。

    chart_utils.create_line_series 对大曲线开了 setUseOpenGL(True) 加速
    滚动 —— 但 GL 系列画在叠加于视图之上的独立 GL 表面，widget.grab() /
    场景渲染都合成不到它：报告截图里测量曲线整条消失（只剩非 GL 的参考
    线）。报告渲染前统一切回软件渲染（一次性出图没有交互性能顾虑），
    finally 里调用方对返回的系列 setUseOpenGL(True) 恢复。

    注意 QChart 不在 QObject 子对象树里（findChildren(QChart) 搜不到），
    要经子控件 QChartView 的 chart() 拿；画布本身可能就是视图（AccZone）。
    """
    switched = []
    try:
        from PySide6.QtCharts import QChartView, QXYSeries
        views = []
        for c in canvases:
            if c is None:
                continue
            if isinstance(c, QChartView):
                views.append(c)
            views.extend(c.findChildren(QChartView))
        for v in views:
            ch = v.chart()
            if ch is None:
                continue
            for s in ch.series():
                try:
                    if isinstance(s, QXYSeries) and s.useOpenGL():
                        s.setUseOpenGL(False)
                        switched.append(s)
                except Exception:
                    continue
    except Exception as e:
        print(f"OpenGL disable error: {e}")
    return switched


def _chart_legend_entries(chart, limit: int = 8) -> list:
    """收集 chart 当前可见系列的 (颜色, 名称)，作为报告图下的文字图例。

    ChartLegend 是叠在视图上的 QWidget，不在 chart 场景里 —— 按场景渲染
    报告图时不会带图例。这里把系列清单读出来，写进 Word 单元格里当
    彩色文字图例（矢量、始终清晰）。
    """
    out = []
    try:
        for s in chart.series():
            if not s.isVisible():
                continue
            name = (s.name() or "").strip()
            if not name:
                continue
            pen = s.pen()
            col = pen.color().name() if pen is not None else "#888888"
            out.append((col, name))
    except Exception:
        pass
    return out[:limit]


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

        # ---- Report module selection -------------------------------------
        # 7 modules map 1:1 onto the report sections AND the canvases the
        # generation loop plots/analyzes. An unchecked module is skipped
        # entirely — not plotted, not screenshotted, not written — which also
        # speeds up generation. Defaults keep the previous all-in behavior.
        layout.addWidget(QLabel("Report Modules:"))
        module_defs = [
            ("trend",   "Trend Plot (+ stats table)"),
            ("sust",    "ASTM Sustained Acceleration"),
            ("egg",     "ASTM Eggplot"),
            ("gb",      "GB 8408 Fitting"),
            ("acczone", "Acceleration Zone Plot"),
            ("zr",      "Z-Axis Reversal Analysis"),
            ("xr",      "X/Y-Axis Reversal Analysis"),
        ]
        self._module_boxes: dict = {}
        mod_widget = QWidget()
        mod_layout = QVBoxLayout(mod_widget)
        mod_layout.setContentsMargins(8, 0, 0, 0)
        for mod_key, mod_label in module_defs:
            mod_cb = CheckBox(mod_label)
            mod_cb.setChecked(True)
            self._module_boxes[mod_key] = mod_cb
            mod_layout.addWidget(mod_cb)
        layout.addWidget(mod_widget)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_ok(self) -> None:
        # 生成期间会泵事件循环（见下方 processEvents），期间用户可能再点
        # OK —— 防重入，避免两份报告循环交错执行。
        if getattr(self, "_generating", False):
            return
        data_list = [cb.text() for cb in self._checkboxes if cb.isChecked()]
        if not data_list:
            return
        self._generating = True
        self._mainui._begin_report_processing()
        try:
            self._generate(data_list)
        finally:
            self._generating = False
            self._mainui._end_report_processing()

    def _generate(self, data_list) -> None:
        if not data_list:
            return
        mods = {k: cb.isChecked() for k, cb in self._module_boxes.items()}
        if not any(mods.values()):
            qfw.InfoBar.warning(
                title="No Module Selected",
                content="请至少勾选一个报告模块",
                orient=Qt.Orientation.Horizontal,
                isClosable=True,
                duration=3000,
                position=qfw.InfoBarPosition.TOP,
                parent=self,
            )
            return

        mu = self._mainui
        doc = docx.Document()

        # 让用户选择输出路径：打包版（console=False）的进程 CWD 可能是只读的
        # 安装目录，固定相对名既可能写不进去，也会静默覆盖同名旧文件。默认名
        # acceleration_plot_<yyyymmdd_HHMM>.docx，落在系统"文档"目录。
        docs_dir = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.DocumentsLocation
        ) or os.path.expanduser("~")
        default_path = os.path.join(
            docs_dir,
            datetime.now().strftime("acceleration_plot_%Y%m%d_%H%M.docx"))
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save Report", default_path, "Word Documents (*.docx)")
        if not out_path:
            return  # 用户取消了保存
        if not out_path.lower().endswith(".docx"):
            out_path += ".docx"

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

        # 临时截图写入独立临时目录：不再往进程 CWD 丢 img*.jpg，杜绝上次报告
        # 残留的同名旧图被本次 add_picture 静默嵌入、与新判定文案错配。
        tmpdir = tempfile.mkdtemp(prefix="bert_report_")
        saved_path = None
        # 测量曲线和 restraint 限值线段都走 create_line_series 的 OpenGL
        # 路径 —— GL 系列画在叠加于视图的独立 GL 表面上，截图合成不到。
        # 每份数据绘制完成后、截图前统一切回软件渲染（循环内重绘会生成
        # 新的 GL 系列，所以在循环内做）；finally 恢复（对象已被重绘销毁
        # 的会抛异常，逐个 try 包住）。
        gl_series: list = []

        try:
            # 生成整份报告期间显示等待光标（finally 中恢复）。
            QGuiApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            for data_name in data_list:
                data = mu.datadict.get(data_name)
                if data is None:
                    continue
                # A background refresh may have been cancelled for export.
                # Bring this recording to the requested cutoff BEFORE any
                # chart or verdict reads its filtered/calibrated frame.
                data.filter_data(cutoff=mu.cutoff)
                restraint = mu.select_restraint.currentText()
                cond = mu.select_cond.currentText()
                height = mu.height_input.value()
                prone = mu._prone_btn.isChecked()

                # --- Plot every SELECTED chart for this dataset --------------
                # Unchecked modules are skipped entirely: not plotted, not
                # analyzed, not captured — faster, and no stale canvas state
                # gets written into the report.
                if mods.get("sust", True):
                    mu.plotASTM_checkbox(canvas=mu.canvasASTMSust, name=data_name)
                if mods.get("egg", True):
                    mu.plotASTM_checkbox(canvas=mu.canvasASTMEgg, name=data_name)
                if mods.get("gb", True):
                    mu.plotGB_checkbox(canvas=mu.canvasGB, name=data_name)
                # Acc Zone 分析强制 1 Hz 数据：与主窗口 _generate_acczone 同口径
                # （calculate_sampling_frequency → get_1hz_standard_frame），不能
                # 走 plotGB_checkbox→_select_data —— 后者按全局 plottype 选帧，
                # 会导致 Zone 分类结论写错进报告。
                if mods.get("acczone", True):
                    if hasattr(data, "calculate_sampling_frequency"):
                        data.calculate_sampling_frequency()
                    df_acczone = data.get_1hz_standard_frame()
                    mu.canvasAccZone.addplotxyz(
                        data=df_acczone, color=mu.dataset_colors.get(data_name))
                if mods.get("trend", True):
                    mu.replot_single(data_name)

                # Reversal charts. Plot the standard frame at the current
                # cutoff, then set allowance/deadband so Analyze has the right
                # limits to check against (setup shared by both, plotting
                # per-selection).
                if mods.get("zr", True) or mods.get("xr", True):
                    if hasattr(data, "calculate_sampling_frequency"):
                        data.calculate_sampling_frequency()
                    data.filter_data(cutoff=mu.cutoff)
                    fs = getattr(data, "fs", 500)
                    df_rev = mu._data_for_plottype(data, "Standard")
                    color = mu.dataset_colors.get(data_name)
                    if mods.get("zr", True):
                        mu.canvasZReversal.plot_data(df_rev, color, fs)
                    if mods.get("xr", True):
                        mu.canvasXYReversal.plot_data(df_rev, color, fs)
                        mu.canvasXYReversal.set_allowance({
                            "x": allowance_at_t0("x", restraint, cond, height),
                            "y": allowance_at_t0("y", restraint, cond, height),
                        })
                        mu.canvasXYReversal.set_deadband(0.05)

                # --- Force Analyze ON for every SELECTED chart so the image
                #     includes the overlays (markers/bands/zone outline). ----
                if mods.get("sust", True):
                    mu.canvasASTMSust.toggle_sust_analysis(
                        True, restraint, cond, height, prone)
                if mods.get("egg", True):
                    mu.canvasASTMEgg.toggle_egg_analysis(True)
                if mods.get("gb", True):
                    mu.canvasGB.toggle_comb_analysis(True)
                if mods.get("acczone", True):
                    mu.canvasAccZone.toggle_analysis(True)
                if mods.get("zr", True):
                    mu.canvasZReversal.toggle_analysis(True)
                if mods.get("xr", True):
                    mu.canvasXYReversal.toggle_analysis(True)

                # --- Gather verdicts (pass/fail + message) -------------------
                verdicts = self._collect_verdicts(mu)

                # --- 泵事件循环 + 显式等比例：让延迟任务在截图前跑完 --------
                # egg/GB/AccZone 的等比例（equalize_eggs / equalize_comb /
                # _apply_equal_aspect）和图例重排（ChartLegend._relayout）都
                # 是 QTimer.singleShot(0, ...) 延迟任务；而本循环从绘制到截图
                # 一直同步执行、事件循环从未运转 —— 截图时这些任务还没执行，
                # 导致报告里蛋形/等值线被拉成椭圆、图例缺失。先泵一次事件让
                # 全部延迟任务执行，再显式调用三个等比例方法兜底（不依赖定时
                # 器时序），最后再泵一次让新轴范围下的标签/标注重排完成。
                QCoreApplication.processEvents()
                for equalizer in (
                    getattr(mu.canvasASTMEgg, "equalize_eggs", None),
                    getattr(mu.canvasGB, "equalize_comb", None),
                    getattr(mu.canvasAccZone, "_apply_equal_aspect", None),
                ):
                    if callable(equalizer):
                        try:
                            equalizer()
                        except Exception as e:
                            print(f"Equalize error: {e}")
                QCoreApplication.processEvents()

                # --- 生成报告图：QChart 按报告画幅重新布局渲染（不截控件）---
                # 出图前先关掉本份数据刚画上去的全部 GL 系列（测量曲线 +
                # restraint 限值线段），否则截图里只剩非 GL 的参考蛋形。
                gl_series.extend(_disable_series_opengl([
                    mu.canvasASTMSust, mu.canvasASTMEgg, mu.canvasGB,
                    mu.canvasAccZone, mu.canvasZReversal, mu.canvasXYReversal,
                    mu.trendplot,
                ]))
                # 子图控件的宽高比由界面布局决定（三列并排的 egg 又窄又高、
                # GB 合成图又扁又长），直接截图放进 Word 很难看。这里把视图
                # 临时固定成报告画幅后走与屏幕同路径的 grab：横向时序图
                # 1100×480，egg 按各自跨度比例，单图模块 1200 宽。Trend 保持
                # 整幅截图（三联时序整体对比）。sections 记录 (子标题, 文件名)。
                sections: dict[str, list] = {}

                def _p(fname):
                    return os.path.join(tmpdir, fname)

                if mods.get("sust", True):
                    entries = []
                    for i, (key, label) in enumerate(
                            [("ax", "ax vs dt"), ("ay", "ay vs dt"),
                             ("az", "az vs dt")], 1):
                        view = mu.canvasASTMSust._views.get(key)
                        ch = mu.canvasASTMSust._charts.get(key)
                        if view is not None and _render_view_figure(
                                view, _p(f"img_sust_{i}.jpg"), 1100, 480):
                            entries.append((label, f"img_sust_{i}.jpg",
                                            _chart_legend_entries(ch)
                                            if ch is not None else []))
                    if entries:
                        sections["sust"] = entries

                if mods.get("egg", True):
                    entries = []
                    egg_labels = {
                        "xy": "X-Y plane (Fore/Aft vs Left/Right)",
                        "xz": "X-Z plane (Fore/Aft vs Up/Down)",
                        "yz": "Y-Z plane (Left/Right vs Up/Down)",
                    }
                    # 画幅按每个蛋图自身的 X/Y 跨度比例取（偏高留余量）：
                    # xy 跨度 8.2×6.4、xz 8.2×8.4、yz 6.2×8.4。等比例按渲染
                    # 画幅的 plotArea 像素比反推 Y 范围，画幅太横会把 Y 压
                    # 扁、蛋上下被裁。
                    egg_sizes = {"xy": (920, 780), "xz": (840, 910),
                                 "yz": (710, 990)}
                    for i, key in enumerate(["xy", "xz", "yz"], 1):
                        view = mu.canvasASTMEgg._views.get(key)
                        ch = mu.canvasASTMEgg._charts.get(key)
                        ew, eh = egg_sizes[key]
                        if view is not None and _render_view_figure(
                                view, _p(f"img_egg_{i}.jpg"), ew, eh,
                                before_render=(_egg_equalize_to_data(ch)
                                               if ch is not None else None)):
                            entries.append((egg_labels[key], f"img_egg_{i}.jpg",
                                            _chart_legend_entries(ch)
                                            if ch is not None else []))
                    if entries:
                        sections["egg"] = entries
                    # 渲染循环后各图 Y 范围混着渲染画幅推导值，重跑一次恢复
                    # 屏幕显示比例。
                    try:
                        mu.canvasASTMEgg.equalize_eggs()
                    except Exception:
                        pass

                if mods.get("gb", True):
                    entries = []
                    gb_specs = [
                        ("_cv_ay", "ay (+) sustained", 1000, 460),
                        ("_cv_az", "az (+) sustained", 1000, 460),
                        ("_cv_comb", "Combined (ay vs az)", 820, 720),
                    ]
                    for i, (attr, label, w, h) in enumerate(gb_specs, 1):
                        view = getattr(mu.canvasGB, attr, None)
                        if view is not None and _render_view_figure(
                                view, _p(f"img_gb_{i}.jpg"), w, h):
                            entries.append((label, f"img_gb_{i}.jpg",
                                            _chart_legend_entries(view.chart())))
                    if entries:
                        sections["gb"] = entries

                if mods.get("acczone", True) and _render_view_figure(
                        mu.canvasAccZone, _p("img_acczone.jpg"), 1200, 640):
                    sections["acczone"] = [
                        (None, "img_acczone.jpg",
                         _chart_legend_entries(mu.canvasAccZone.chart()))]

                if mods.get("trend", True) and _grab_widget(
                        mu.trendplot, _p("img_trend.jpg")):
                    sections["trend"] = [(None, "img_trend.jpg", None)]

                if mods.get("zr", True):
                    sub = getattr(mu.canvasZReversal, "_chart_z", None)
                    if sub is not None and _render_view_figure(
                            sub, _p("img_zr.jpg"), 1200, 500):
                        sections["zr"] = [
                            (None, "img_zr.jpg", _chart_legend_entries(sub.chart()))]

                if mods.get("xr", True):
                    entries = []
                    for i, (attr, label) in enumerate(
                            [("_chart_x", "X (Fore/Aft)"),
                             ("_chart_y", "Y (Lateral)")], 1):
                        sub = getattr(mu.canvasXYReversal, attr, None)
                        if sub is not None and _render_view_figure(
                                sub, _p(f"img_xr_{i}.jpg"), 1100, 480):
                            entries.append((label, f"img_xr_{i}.jpg",
                                            _chart_legend_entries(sub.chart())))
                    if entries:
                        sections["xr"] = entries

                mu.trendplot.initplotxyz()
                mu.clear_sidebar()
                mu.replot_all()
                self._write_doc_section(doc, data, data_name, verdicts, tmpdir,
                                        mods, sections)

            try:
                doc.save(out_path)
            except PermissionError:
                # 输出文件被 Word 等程序占用 —— 打包版无控制台，仅 print 用户
                # 完全看不到，必须用 InfoBar 让失败可见。对话框保持打开以便重试。
                print(f"Report save error (file locked): {out_path}")
                qfw.InfoBar.error(
                    title="Report Failed",
                    content="输出文件被占用，请关闭 Word 后重试",
                    orient=Qt.Orientation.Horizontal,
                    isClosable=True,
                    duration=-1,
                    position=qfw.InfoBarPosition.TOP,
                    parent=self,
                )
            else:
                saved_path = out_path
                mu.statusBar().showMessage(
                    f"Report generated: {os.path.abspath(out_path)}", 5000)
                self.accept()

        except Exception as e:
            print(f"Report error: {e}")
            qfw.InfoBar.error(
                title="Report Failed",
                content=f"报告生成失败：{e}",
                orient=Qt.Orientation.Horizontal,
                isClosable=True,
                duration=-1,
                position=qfw.InfoBarPosition.TOP,
                parent=self,
            )

        finally:
            QGuiApplication.restoreOverrideCursor()
            # 恢复报告前各画布测量系列的 OpenGL 加速渲染（重绘已销毁的
            # 系列对象恢复时抛异常，逐个吞掉）。
            for s in gl_series:
                try:
                    s.setUseOpenGL(True)
                except Exception:
                    pass
            # Restore the user's prior Analyze state (and the live canvases).
            self._restore_analyze(mu, prior)
            # 报告循环会把各图表页画成循环里最后一个数据集 —— 按顶栏 Data
            # 选择器当前选中的数据集重绘所有图表页，恢复用户看到的页面。
            try:
                mu._generate_active_page()
            except Exception as e:
                print(f"Post-report regenerate error: {e}")
            if saved_path:
                # _generate_active_page 会刷新状态栏消息，这里补一次带完整
                # 路径的成功提示，确保用户看得到报告写到了哪里。
                mu.statusBar().showMessage(
                    f"Report generated: {os.path.abspath(saved_path)}", 5000)
            # 兜底清理临时图目录（无论成功/失败/异常都会执行）。
            shutil.rmtree(tmpdir, ignore_errors=True)

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

        截图文件缺失（单图保存失败）时只跳过图片、保留结论行：既不抛错中断
        整份报告，也不会误嵌任何残留的旧图（临时目录按次新建，天然隔离）。
        """
        if os.path.isfile(img_path):
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
    def _write_doc_section(doc, data, name: str, verdicts: dict,
                           img_dir: str, mods: dict | None = None,
                           sections: dict | None = None) -> None:
        """Write one data section into the Word document.

        Each module gets a level-4 heading; its subplots (from *sections*,
        built by the capture step as ``{module: [(sub_label, filename)]}``)
        each get a level-5 sub-heading + image. The module's single verdict
        line (green ✓ PASS / red ✗ FAIL, from *verdicts*) is written once
        after the module's images. *sections*=None falls back to one whole-
        module image per module using the legacy img1..img7 slot names.
        *mods* 为勾选的报告模块（None → 全部写入）；未勾选的模块整节跳过。
        """
        v = verdicts or {}
        on = (lambda k: True) if mods is None \
            else (lambda k: mods.get(k, True))

        if sections is None:
            # Legacy fallback: one whole-module image per module.
            sections = {
                "trend": [(None, "img5.jpg", None)],
                "sust": [(None, "img1.jpg", None)],
                "egg": [(None, "img2.jpg", None)],
                "gb": [(None, "img3.jpg", None)],
                "acczone": [(None, "img4.jpg", None)],
                "zr": [(None, "img6.jpg", None)],
                "xr": [(None, "img7.jpg", None)],
            }

        def _legend_line(cell, legend) -> None:
            """图例文字行：彩色 ■ + 系列名（小号灰字）。"""
            if not legend:
                return
            p = cell.add_paragraph()
            for col, series_name in legend:
                r1 = p.add_run("\u25a0 ")
                r1.font.color.rgb = RGBColor.from_string(col.lstrip("#").upper())
                r1.font.size = Pt(8)
                r2 = p.add_run(series_name + "   ")
                r2.font.size = Pt(8)

        def _subplot_grid(entries) -> None:
            """子图两列并排放进无边框表格：单元格 = 小标题 + ~2.9in 图 +
            彩色文字图例。图是按报告画幅重新布局渲染的正常构图图。"""
            if not entries:
                return
            rows = (len(entries) + 1) // 2
            table = doc.add_table(rows=rows, cols=2)
            for idx, (label, fname, legend) in enumerate(entries):
                cell = table.cell(idx // 2, idx % 2)
                p = cell.paragraphs[0]
                if label:
                    run = p.add_run(label)
                    run.bold = True
                    run.font.size = Pt(9)
                img_p = cell.add_paragraph()
                try:
                    img_p.add_run().add_picture(
                        os.path.join(img_dir, fname), width=Inches(2.9))
                except Exception as e:
                    print(f"Image insert error: {e}")
                _legend_line(cell, legend)

        def _write_module(key: str, heading: str) -> None:
            if not on(key):
                return
            doc.add_heading(heading, level=4)
            entries = sections.get(key, [])
            if entries and all(label is None for label, _, _ in entries):
                # 单图模块：整幅全宽。
                for _sub, fname, legend in entries:
                    ReportDialog._add_image_with_verdict(
                        doc, os.path.join(img_dir, fname), None)
                    if legend:
                        para = doc.add_paragraph()
                        for col, series_name in legend:
                            r1 = para.add_run("\u25a0 ")
                            r1.font.color.rgb = RGBColor.from_string(
                                col.lstrip("#").upper())
                            r1.font.size = Pt(8)
                            r2 = para.add_run(series_name + "   ")
                            r2.font.size = Pt(8)
            else:
                _subplot_grid(entries)
            # 模块级判定行只写一次（空路径 → 跳过图片、只写结论）。
            ReportDialog._add_image_with_verdict(doc, "", v.get(key))

        doc.add_heading(f"Acceleration File Name: {name}", level=2)
        if on("trend"):
            doc.add_heading("Trend Plot:", level=4)
            for _sub, fname, _leg in sections.get("trend", []):
                # Trend has no analysis → no verdict line.
                ReportDialog._add_image_with_verdict(
                    doc, os.path.join(img_dir, fname), None)

            # Stats table
            key_info = data.get_data_stats(data.std_data)
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

        _write_module("sust", "ASTM F2291 Sustained Acceleration:")
        _write_module("egg", "ASTM F2291 Eggplot:")
        _write_module("gb", "GB 8408 Fitting:")
        _write_module("acczone", "Acceleration Zone Plot:")
        _write_module("zr", "Z-Axis Reversal Analysis:")
        _write_module("xr", "X/Y-Axis Reversal Analysis:")
