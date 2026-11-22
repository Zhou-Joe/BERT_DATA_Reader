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

from PySide6.QtWidgets import QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox, QCheckBox, QPushButton, QScrollArea
from ui_controls import PrimaryButton, warning, error

from report_document import AccelerationReport
from report_lateral import render_lateral_figure

from theme import theme_manager
from sci_calculation import allowance_at_t0
from accdata import detect_test_boundaries
from segment_preview import SegmentPreview, TimeField


class _ThemedDialog(QDialog):
    """Apply the workspace surface and text colours to secondary windows."""

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
    """Apply shared margins to dialog content."""
    layout.setContentsMargins(20, 20, 20, 20)
    layout.setSpacing(12)


# ---------------------------------------------------------------------------
# Truncate Data Dialog
# ---------------------------------------------------------------------------

class TruncateTimeDialog(_ThemedDialog):
    """Review source-time windows before trimming or creating recordings."""

    def __init__(self, mainui, data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Trim and split recording")
        self.setMinimumWidth(570)
        self._mainui, self._data = mainui, data
        self._source_frame = data.data
        self._data_start = float(data.data["Time"].iloc[0])
        self._data_duration = float(data.data["Time"].iloc[-1]) + 1 / data.fs
        self._auto_proposal = False
        self._rebuilding = False

        layout = QVBoxLayout(self)
        _apply_dialog_margins(layout)
        title = QLabel(str(data.filename))
        title.setTextFormat(Qt.TextFormat.PlainText)
        title.setWordWrap(True)
        layout.addWidget(title)
        self._preview = SegmentPreview(data, self._data_start, self._data_duration)
        layout.addWidget(self._preview)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Quiet gap"))
        self._min_gap_spin = QDoubleSpinBox()
        self._min_gap_spin.setRange(1.0, 120.0)
        self._min_gap_spin.setDecimals(1)
        self._min_gap_spin.setSuffix(" s")
        self._min_gap_spin.setValue(10.0)
        self._min_gap_spin.setToolTip("Minimum stop between tests. Increase to make fewer splits.")
        controls.addWidget(self._min_gap_spin)
        self._auto_detect_btn = QPushButton("Auto-detect")
        self._auto_detect_btn.clicked.connect(self._on_auto_detect)
        controls.addWidget(self._auto_detect_btn)
        controls.addStretch()
        controls.addWidget(QLabel("Segments"))
        self._count = QSpinBox()
        self._count.setRange(1, 50)
        controls.addWidget(self._count)
        layout.addLayout(controls)

        self._notice = QLabel("Set the start and end times, or detect segments from pauses.")
        self._notice.setWordWrap(True)
        self._notice.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self._notice)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        host = QWidget()
        self._form = QFormLayout(host)
        self._form.setContentsMargins(0, 0, 8, 0)
        self._form.setSpacing(8)
        self._scroll.setWidget(host)
        layout.addWidget(self._scroll)
        self._start_spins, self._end_spins, self._keeps = [], [], []

        self._replace_original = QCheckBox("Remove original after creating segments")
        self._replace_original.setChecked(False)
        layout.addWidget(self._replace_original)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self._apply_button = buttons.button(QDialogButtonBox.Ok)
        buttons.accepted.connect(self._on_ok)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._count.valueChanged.connect(self._rebuild_time_rows)
        self._rebuild_time_rows(1)

    def _windows(self):
        return [(i, a.seconds(), b.seconds()) for i, (a, b, keep) in
                enumerate(zip(self._start_spins, self._end_spins, self._keeps), 1)
                if keep.isChecked()]

    def _sync_preview(self, *_):
        if self._rebuilding:
            return
        windows = self._windows()
        self._preview.set_windows([(a, b) for _, a, b in windows if b > a])
        creates_copies = self._count.value() > 1 or self._auto_proposal
        self._replace_original.setVisible(creates_copies)
        self._apply_button.setText(("Create segment" if len(windows) == 1 else
                                    f"Create {len(windows)} segments")
                                   if creates_copies else "Trim original")
        self._apply_button.setEnabled(bool(windows))

    def _rebuild_time_rows(self, n, proposals=None):
        self._rebuilding = True
        previous = [(a.seconds(), b.seconds(), k.isChecked()) for a, b, k in
                    zip(self._start_spins, self._end_spins, self._keeps)]
        while self._form.rowCount():
            self._form.removeRow(0)
        self._start_spins.clear()
        self._end_spins.clear()
        self._keeps.clear()
        for i in range(n):
            if proposals is not None:
                a, b = proposals[i]
                included = True
            elif i < len(previous):
                a, b, included = previous[i]
            else:
                a = self._end_spins[-1].seconds() if i else self._data_start
                b, included = self._data_duration, True
            times = self._source_frame["Time"].to_numpy()
            start = TimeField(times, self._data_duration, self._data.fs)
            end = TimeField(times, self._data_duration, self._data.fs)
            start.set_seconds(a)
            end.set_seconds(b)
            for field, label in ((start, "Start"), (end, "End (exclusive)")):
                field.setAccessibleName(f"Segment {i + 1}: {label}")
                field.setToolTip(label + " in source seconds")
            keep = QCheckBox(f"Segment {i + 1}")
            keep.setChecked(included)
            row = QHBoxLayout()
            row.addWidget(start)
            row.addWidget(QLabel("to"))
            row.addWidget(end)
            self._form.addRow(keep, row)
            self._start_spins.append(start)
            self._end_spins.append(end)
            self._keeps.append(keep)
            start.setEnabled(included)
            end.setEnabled(included)
            keep.toggled.connect(start.setEnabled)
            keep.toggled.connect(end.setEnabled)
            keep.toggled.connect(self._sync_preview)
            start.secondsChanged.connect(self._sync_preview)
            end.secondsChanged.connect(self._sync_preview)
        # Link contiguous manual laps only. An edit must not fill a detected
        # pause or overwrite a deliberately independent start time.
        if not self._auto_proposal:
            for end, following in zip(self._end_spins, self._start_spins[1:]):
                def move_seam(value, nxt=following, previous_end=[end.seconds()]):
                    if nxt.seconds() == previous_end[0]:
                        nxt.set_seconds(value)
                    previous_end[0] = value
                end.secondsChanged.connect(move_seam)
        self._scroll.setMinimumHeight(min(44 * n + 12, 220))
        self._rebuilding = False
        self._sync_preview()

    def _on_auto_detect(self):
        try:
            tests = detect_test_boundaries(self._data, min_gap_s=self._min_gap_spin.value())
        except Exception as error:
            self._notice.setText(f"Detection failed: {error}. Current entries were kept.")
            return
        if not tests:
            for keep in self._keeps:
                keep.setChecked(False)
            self._notice.setText("No segments suggested. To use a manual window, "
                                 "check its box and review the times.")
            return
        if len(tests) > self._count.maximum():
            self._notice.setText(f"Found {len(tests)} segments; only {self._count.maximum()} "
                                 "can be reviewed at once. Increase the quiet gap or trim "
                                 "a shorter recording. Current entries were kept.")
            return
        self._auto_proposal = True
        self._count.blockSignals(True)
        self._count.setValue(len(tests))
        self._count.blockSignals(False)
        self._rebuild_time_rows(len(tests), proposals=tests)
        label = "segment" if len(tests) == 1 else "segments"
        self._notice.setText(f"Found {len(tests)} {label}. Review the times and uncheck "
                             "any segment you want to omit.")

    def _on_ok(self):
        windows = self._windows()
        if not windows:
            return
        try:
            if (self._mainui.datadict.get(self._data.filename) is not self._data or
                    self._data.data is not self._source_frame):
                raise ValueError("The recording changed. Reopen Trim and split to review it.")
            times = self._source_frame["Time"].to_numpy()
            last_end = None
            for i, start, end in windows:
                if start < self._data_start or end > self._data_duration or start >= end:
                    raise ValueError(f"Segment {i}: enter a start before the end, inside the recording.")
                if last_end is not None and start < last_end:
                    raise ValueError(f"Segment {i}: selected windows must be in order and cannot overlap.")
                lo, hi = times.searchsorted([start, end])
                if hi - lo < 2:
                    raise ValueError(f"Segment {i}: the window must contain at least two samples.")
                last_end = end
            if self._count.value() == 1 and not self._auto_proposal:
                _, start, end = windows[0]
                self._data.truncate_data(start, end, self._mainui.cutoff)
                self._mainui.trendplot.initplotxyz()
                self._mainui.clear_sidebar()
                self._mainui.replot_all()
            else:
                # Prepare every selected slice before exposing any to the library.
                clones = []
                for i, start, end in windows:
                    clone = copy.deepcopy(self._data)
                    clone.truncate_data(start, end, self._mainui.cutoff)
                    clone.filename = f"{self._data.filename} [Trunc {start:.8g}-{end:.8g}s #{i}]"
                    clones.append(clone)
                for clone in clones:
                    self._mainui.set_datalist(clone)
                    self._mainui.add_trend_data(clone, self._mainui.plottype,
                                                len(self._mainui.datalist))
                if self._replace_original.isChecked():
                    self._mainui._remove_dataset(self._mainui._plot_order_of(self._data.filename))
                self._mainui.statusBar().showMessage(f"Created {len(clones)} segment(s)", 4000)
            self.accept()
        except Exception as error:
            self._notice.setText(str(error))


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
        self._export_type = QComboBox()
        self._export_type.addItems([
            "Formated Data for Newtonviewer",
            "Filtered Clean Data",
        ])

        # Filename + suffix (combined row — bug fix: previously the
        # filename field was added twice, once on its own row and again
        # inside this suffix layout, producing a duplicated control).
        self._filename = QLineEdit()
        self._suffix = QComboBox()
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
            warning(
                title="Export",
                content="请输入文件名",
                duration=3000,
                parent=self,
            )
            return

        if (self._export_type.currentText() == 'Filtered Clean Data'
                and self._mainui.plottype != 'Raw'
                and self._data._filtered_cutoff != self._mainui.cutoff_spin.value()):
            warning(
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
            error(
                title="Export Failed",
                content=f"导出失败：{e}",
                duration=-1,
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
        self._add_x = QDoubleSpinBox()
        self._add_x.setRange(-10, 10)
        self._add_y = QDoubleSpinBox()
        self._add_y.setRange(-10, 10)
        self._add_z = QDoubleSpinBox()
        self._add_z.setRange(-10, 10)
        add_form.addRow("Fore/Back (x):", self._add_x)
        add_form.addRow("Left/Right (y):", self._add_y)
        add_form.addRow("Up/Down (z):", self._add_z)
        add_group.setLayout(add_form)
        layout.addWidget(add_group)

        btn_add = PrimaryButton("Apply Add")
        btn_add.clicked.connect(self._on_add)
        layout.addWidget(btn_add)

        # --- Multiply section ---
        mul_group = QGroupBox("Multiply Scale")
        mul_form = QFormLayout()
        self._mul_x = QDoubleSpinBox()
        self._mul_x.setRange(-10, 10)
        self._mul_x.setValue(1.0)
        self._mul_y = QDoubleSpinBox()
        self._mul_y.setRange(-10, 10)
        self._mul_y.setValue(1.0)
        self._mul_z = QDoubleSpinBox()
        self._mul_z.setRange(-10, 10)
        self._mul_z.setValue(1.0)
        mul_form.addRow("Fore/Back (x):", self._mul_x)
        mul_form.addRow("Left/Right (y):", self._mul_y)
        mul_form.addRow("Up/Down (z):", self._mul_z)
        mul_group.setLayout(mul_form)
        layout.addWidget(mul_group)

        btn_mul = PrimaryButton("Apply Multiply")
        btn_mul.clicked.connect(self._on_multiply)
        layout.addWidget(btn_mul)

        btn_close = QPushButton("Close")
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
        self._pitch = QSpinBox()
        self._pitch.setRange(-180, 180)
        self._pitch.setSuffix(" °")
        self._seatback = QSpinBox()
        self._seatback.setRange(-180, 180)
        self._seatback.setSuffix(" °")
        self._roll = QSpinBox()
        self._roll.setRange(-180, 180)
        self._roll.setSuffix(" °")
        self._yaw = QSpinBox()
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
        self._freq = QSpinBox()
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
    """把一个 QChartView 临时固定成报告画幅 (w, h) 后截图渲染成 PNG。

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
        ok = view.grab().save(out_path, "PNG")
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


def _render_trend_figure(frame, color, out_path: str):
    """Draw one recording onto a fresh canvas at print size.

    Capturing the live trend immediately after resetting its three viewports
    can composite their previous axes over the new ones. The report also needs
    the selected recording even if its trace is hidden in the workspace.
    """
    from PySide6.QtCore import QEvent
    from chart_trend import TrendChartView

    figure = TrendChartView()
    try:
        figure.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen)
        figure.setFixedSize(960, 560)
        figure.addplotxyz(frame, 1, color)
        _disable_series_opengl([figure])
        for chart, label in [(figure._chart_x, "Fore/aft (g)"),
                             (figure._chart_y, "Lateral (g)"),
                             (figure._chart_z, "Vertical (g)")]:
            chart.axis_y.setTitleText(label)
            chart.axis_x.setTitleText("Time (s)" if chart is figure._chart_z else "")
        figure.show()
        figure.layout().activate()
        if not figure.grab().save(out_path, "PNG"):
            raise OSError("Could not save the trend figure")
    finally:
        figure.close()
        figure.deleteLater()
        # Multi-record exports run in a single GUI callback. Release this
        # recording's full-resolution series before moving on to the next.
        QCoreApplication.sendPostedEvents(figure, QEvent.Type.DeferredDelete)


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


def evaluate_recording(mu, data, mods, *, color=None, z_source="Standard",
                       xy_source="Standard", deadband=0.05):
    """Evaluate the selected checks with the same charts and rules as the Word report."""
    data.filter_data(cutoff=mu.cutoff)
    restraint = mu.select_restraint.currentText()
    cond = mu.select_cond.currentText()
    height = mu.height_input.value()
    prone = mu._prone_btn.isChecked()

    # --- Plot every SELECTED chart for this dataset --------------
    # Unchecked modules are skipped entirely: not plotted, not
    # analyzed, not captured — faster, and no stale canvas state
    # gets written into the report.
    # Plot directly: the UI convenience wrappers swallow exceptions,
    # which could export the preceding recording's chart as this one.
    for key, canvas in (("sust", mu.canvasASTMSust), ("egg", mu.canvasASTMEgg)):
        if mods[key]:
            canvas.addplotxyz(data=mu._select_data(data), restraint=restraint,
                              cond=cond, input_height=height, prone=prone,
                              color=color)
    if mods.get("gb", True):
        mu.canvasGB.addplotxyz(data=mu._select_data(data), color=color)
    # Acc Zone 分析强制 1 Hz 数据：与主窗口 _generate_acczone 同口径
    # （calculate_sampling_frequency → get_1hz_standard_frame），不能
    # 走 plotGB_checkbox→_select_data —— 后者按全局 plottype 选帧，
    # 会导致 Zone 分类结论写错进报告。
    if mods.get("acczone", True):
        if hasattr(data, "calculate_sampling_frequency"):
            data.calculate_sampling_frequency()
        df_acczone = data.get_1hz_standard_frame()
        mu.canvasAccZone.addplotxyz(data=df_acczone, color=color)

    # Reversal charts. Plot the standard frame at the current
    # cutoff, then set allowance/deadband so Analyze has the right
    # limits to check against (setup shared by both, plotting
    # per-selection).
    if mods.get("zr", True) or mods.get("xr", True):
        if hasattr(data, "calculate_sampling_frequency"):
            data.calculate_sampling_frequency()
        data.filter_data(cutoff=mu.cutoff)
        fs = getattr(data, "fs", 500)
        if mods.get("zr", True):
            mu.canvasZReversal.plot_data(mu._data_for_plottype(data, z_source), color, fs)
        if mods.get("xr", True):
            mu.canvasXYReversal.plot_data(mu._data_for_plottype(data, xy_source), color, fs)
            mu.canvasXYReversal.set_allowance({
                "x": allowance_at_t0("x", restraint, cond, height),
                "y": allowance_at_t0("y", restraint, cond, height),
            })
            mu.canvasXYReversal.set_deadband(deadband)

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

    return ReportDialog._collect_verdicts(mu, mods)


class ReportDialog(_ThemedDialog):
    """Dialog for selecting datasets and generating a Word report."""

    def __init__(self, mainui, parent=None):
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QGridLayout

        super().__init__(parent or mainui)
        self.setWindowTitle("Word report")
        # The displayed analysis settings stay in effect until export or cancel.
        self.setWindowModality(Qt.WindowModality.WindowModal)
        self.setMinimumWidth(520)
        self._mainui = mainui
        self._checkboxes: list = []

        layout = QVBoxLayout(self)
        _apply_dialog_margins(layout)
        layout.addWidget(QLabel("Create a Word report"))
        plot = {"Standard": "Calibrated", "Filter": "Filtered", "Raw": "Raw"}[mainui.plottype]
        summary = QLabel(
            f"Signal: {plot}   ·   Low-pass setting: {mainui.cutoff_spin.value()} Hz\n"
            f"Restraint: {mainui.select_restraint.currentText()}   ·   Condition: "
            f"{mainui.select_cond.currentText()}\nHeight: {mainui.height_input.value()} in   ·   "
            f"Prone: {'Yes' if mainui._prone_btn.isChecked() else 'No'}")
        summary.setWordWrap(True)
        tk = theme_manager.tokens
        summary.setStyleSheet(
            f"color: {tk['text']}; background: {tk['window_bg']}; "
            f"border: 1px solid {tk['divider']}; border-radius: 6px; padding: 8px 12px;")
        layout.addWidget(summary)

        # Scrollable checkbox list
        layout.addWidget(QLabel("Recordings"))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(68)
        scroll.setMaximumHeight(160)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        active = mainui.select_data.currentText()
        for name in mainui.datadict.keys():
            cb = QCheckBox(name)
            cb.setChecked(name == active)
            cb.setToolTip(name)
            self._checkboxes.append(cb)
            scroll_layout.addWidget(cb)
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll, 1)
        current_box = next((cb for cb in self._checkboxes if cb.isChecked()), None)
        if current_box is not None:
            QTimer.singleShot(0, self, lambda: scroll.ensureWidgetVisible(current_box))

        # ---- Report module selection -------------------------------------
        # 7 modules map 1:1 onto the report sections AND the canvases the
        # generation loop plots/analyzes. An unchecked module is skipped
        # entirely — not plotted, not screenshotted, not written — which also
        # speeds up generation. Defaults keep the previous all-in behavior.
        layout.addWidget(QLabel("Report sections"))
        module_defs = [
            ("trend",   "Trend and signal statistics"),
            ("sust",    "ASTM Sustained Acceleration"),
            ("egg",     "ASTM Eggplot"),
            ("gb",      "GB 8408 Fitting"),
            ("acczone", "Acceleration Zone Plot"),
            ("zr",      "Z-Axis Reversal Analysis"),
            ("xr",      "X/Y-Axis Reversal Analysis"),
        ]
        self._module_boxes: dict = {}
        mod_widget = QWidget()
        mod_layout = QGridLayout(mod_widget)
        mod_layout.setContentsMargins(0, 0, 0, 0)
        for index, (mod_key, mod_label) in enumerate(module_defs):
            mod_cb = QCheckBox(mod_label)
            mod_cb.setChecked(True)
            self._module_boxes[mod_key] = mod_cb
            mod_layout.addWidget(mod_cb, index % 4, index // 4)
        layout.addWidget(mod_widget)

        footer = QHBoxLayout()
        count = QLabel()
        footer.addWidget(count, 1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        footer.addWidget(cancel)
        generate = PrimaryButton("Generate Word report")
        generate.setDefault(True)
        generate.clicked.connect(self._on_ok)
        footer.addWidget(generate)
        layout.addLayout(footer)

        def update_selection():
            recordings = sum(cb.isChecked() for cb in self._checkboxes)
            sections = sum(cb.isChecked() for cb in self._module_boxes.values())
            generate.setEnabled(bool(recordings and sections))
            count.setText(f"{recordings} recording{'s' if recordings != 1 else ''} · "
                          f"{sections} section{'s' if sections != 1 else ''}")

        for cb in [*self._checkboxes, *self._module_boxes.values()]:
            cb.toggled.connect(update_selection)
        update_selection()

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

    def _generate(self, data_list, *, output_path=None, z_source="Standard",
                  xy_source="Standard", deadband=0.05):
        if not data_list:
            return
        mods = {k: cb.isChecked() for k, cb in self._module_boxes.items()}
        if not any(mods.values()):
            warning(
                title="No Module Selected",
                content="请至少勾选一个报告模块",
                duration=3000,
                parent=self,
            )
            return

        mu = self._mainui
        doc = AccelerationReport(len(data_list))

        # 让用户选择输出路径：打包版（console=False）的进程 CWD 可能是只读的
        # 安装目录，固定相对名既可能写不进去，也会静默覆盖同名旧文件。默认名
        # acceleration_plot_<yyyymmdd_HHMM>.docx，落在系统"文档"目录。
        out_path = output_path
        if out_path is None:
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
                verdicts = evaluate_recording(mu, data, mods, color=mu.dataset_colors.get(data_name),
                                              z_source=z_source, xy_source=xy_source, deadband=deadband)
                restraint = mu.select_restraint.currentText()
                cond = mu.select_cond.currentText()
                height = mu.height_input.value()
                prone = mu._prone_btn.isChecked()

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
                        if key == "ay" and ch is not None:
                            filename = f"img_sust_{i}.png"
                            render_lateral_figure(ch, restraint, _p(filename))
                            entries.append(("Signed lateral acceleration by duration", filename, []))
                            continue
                        if view is not None and _render_view_figure(
                                view, _p(f"img_sust_{i}.png"), 1100, 480):
                            entries.append((label, f"img_sust_{i}.png",
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
                                view, _p(f"img_egg_{i}.png"), ew, eh,
                                before_render=(_egg_equalize_to_data(ch)
                                               if ch is not None else None)):
                            entries.append((egg_labels[key], f"img_egg_{i}.png",
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
                                view, _p(f"img_gb_{i}.png"), w, h):
                            entries.append((label, f"img_gb_{i}.png",
                                            _chart_legend_entries(view.chart())))
                    if entries:
                        sections["gb"] = entries

                if mods.get("acczone", True) and _render_view_figure(
                        mu.canvasAccZone, _p("img_acczone.png"), 1200, 640):
                    sections["acczone"] = [
                        (None, "img_acczone.png",
                         _chart_legend_entries(mu.canvasAccZone.chart()))]

                if mods.get("trend", True):
                    _render_trend_figure(mu._select_data(data),
                                         mu.dataset_colors.get(data_name, "#2563EB"),
                                         _p("img_trend.png"))
                    sections["trend"] = [(None, "img_trend.png", None)]

                if mods.get("zr", True):
                    sub = getattr(mu.canvasZReversal, "_chart_z", None)
                    if sub is not None and _render_view_figure(
                            sub, _p("img_zr.png"), 1200, 500):
                        sections["zr"] = [
                            (None, "img_zr.png", _chart_legend_entries(sub.chart()))]

                if mods.get("xr", True):
                    entries = []
                    for i, (attr, label) in enumerate(
                            [("_chart_x", "X (Fore/Aft)"),
                             ("_chart_y", "Y (Lateral)")], 1):
                        sub = getattr(mu.canvasXYReversal, attr, None)
                        if sub is not None and _render_view_figure(
                                sub, _p(f"img_xr_{i}.png"), 1100, 480):
                            entries.append((label, f"img_xr_{i}.png",
                                            _chart_legend_entries(sub.chart())))
                    if entries:
                        sections["xr"] = entries

                doc.add_record(data, data_name, verdicts, tmpdir, mods, sections,
                               plot_type=mu.plottype, cutoff=mu.cutoff, restraint=restraint,
                               condition=cond, height=height, prone=prone,
                               z_source=z_source, xy_source=xy_source, deadband=deadband)

            try:
                doc.save(out_path)
            except PermissionError:
                if output_path is not None:
                    raise
                # 输出文件被 Word 等程序占用 —— 打包版无控制台，仅 print 用户
                # 完全看不到，必须用 InfoBar 让失败可见。对话框保持打开以便重试。
                print(f"Report save error (file locked): {out_path}")
                error(
                    title="Report Failed",
                    content="输出文件被占用，请关闭 Word 后重试",
                    duration=-1,
                    parent=self,
                )
            else:
                saved_path = out_path
                mu.statusBar().showMessage(
                    f"Report generated: {os.path.abspath(out_path)}", 5000)
                self.accept()

        except Exception as e:
            if output_path is not None:
                raise
            print(f"Report error: {e}")
            error(
                title="Report Failed",
                content=f"报告生成失败：{e}",
                duration=-1,
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
    def _collect_verdicts(mu, selected) -> dict:
        """Publish passes only for selected checks with evaluated data."""
        verdicts = {}
        if selected["sust"]:
            canvas = mu.canvasASTMSust
            failed = sorted(canvas.sust_failed_rules)
            if failed:
                verdicts["sust"] = (False, "Exceeded limit: " + ", ".join(failed))
            elif len(getattr(canvas, "sust_axes_checked", ())) == 6:
                verdicts["sust"] = (True, "All measured axes within limit")
        if selected["egg"]:
            canvas = mu.canvasASTMEgg
            failed = sorted(canvas.failed_rules)
            if failed:
                verdicts["egg"] = (False, "Outside boundary: " + ", ".join(failed))
            elif getattr(canvas, "egg_planes_checked", set()) == {"xy", "xz", "yz"}:
                verdicts["egg"] = (True, "All three measured planes within boundary")
        if selected["gb"]:
            canvas = mu.canvasGB
            failed = sorted(canvas.failed_rules)
            if failed:
                verdicts["gb"] = (False, "Exceeded: " + ", ".join(failed))
            elif len(getattr(canvas, "comb_rules_checked", ())) == 3:
                verdicts["gb"] = (True, "All measured points within the three reference loops")
        if selected["acczone"]:
            zone = mu.canvasAccZone._auto_zone_result
            if zone is not None:
                verdicts["acczone"] = ("info", f"Classified as Zone {zone}")
        if selected["zr"]:
            canvas = mu.canvasZReversal
            if canvas._analysis_active and "z" in canvas._cached_data:
                count = canvas.event_count
                verdicts["zr"] = (count == 0, f"Z reversal patterns detected: {count}")
        if selected["xr"]:
            canvas = mu.canvasXYReversal
            if (canvas._analysis_active and {"x", "y"} <= canvas._cached_data.keys()
                    and {"x", "y"} <= (canvas._allowance or {}).keys()):
                failed = canvas.fail_count
                verdicts["xr"] = (failed == 0, f"Reversal pairs exceeding allowance: "
                                  f"{failed} of {canvas.event_count}")
        return verdicts

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
