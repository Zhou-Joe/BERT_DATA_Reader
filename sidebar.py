# coding: utf-8
"""Recording library: compact selectable rows and one focused signal inspector."""
from __future__ import annotations

import re
from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QPushButton,
    QToolButton, QLineEdit, QScrollArea, QFrame, QSplitter,
    QSizePolicy, QColorDialog,
)
from ui_controls import icon
from theme import DATASET_COLORS, theme_manager


class _FilenameLabel(QLabel):
    """Elide by available pixels while keeping the complete name in a tooltip."""
    def __init__(self, text, parent=None):
        super().__init__(parent)
        self._full_text = text
        self.setToolTip(text)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._update_text()

    def _update_text(self):
        self.setText(self.fontMetrics().elidedText(
            self._full_text, Qt.TextElideMode.ElideMiddle, max(0, self.width())))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_text()


class InfoCard(QWidget):
    """Compact recording row. The historical name is kept for MainWindow."""
    selected = Signal(str)

    def __init__(self, data, plottype, plot_order, remove_callback, mainui, parent=None):
        super().__init__(parent)
        self._data = data
        self._mainui = mainui
        self._plottype = plottype
        self._plot_order = plot_order
        self._remove_callback = remove_callback
        self._active_dialogs = []
        self._selected = False
        self._color = mainui.dataset_colors.get(
            data.filename, DATASET_COLORS[(plot_order - 1) % len(DATASET_COLORS)])
        self.setObjectName('recordingRow')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setFixedHeight(62)
        self.setAccessibleName(f'Recording {data.filename}')
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 10, 8, 10)
        layout.setSpacing(10)
        self._indicator = QToolButton()
        self._indicator.setFixedSize(22, 24)
        self._indicator.setCursor(Qt.CursorShape.PointingHandCursor)
        self._indicator.setAccessibleName(f'Change trace colour for {data.filename}')
        self._indicator.clicked.connect(self._pick_color)
        layout.addWidget(self._indicator)
        text = QVBoxLayout()
        text.setSpacing(6)
        display_name = re.sub(r'^\((GB|ASTM)\)', '', data.filename)
        self._title_label = _FilenameLabel(display_name)
        text.addWidget(self._title_label)
        duration = len(data.data) / data.fs
        group = ('GB' if data.filename.startswith('(GB)') else
                 'ASTM' if data.filename.startswith('(ASTM)') else 'ACCEL')
        self._meta = QLabel(f'{group}  /  {data.fs:g} Hz  /  {duration:,.2f} s')
        self._meta.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        text.addWidget(self._meta)
        layout.addLayout(text, 1)
        self._plot_check = QToolButton()
        self._plot_check.setFixedSize(30, 30)
        self._plot_check.setIconSize(QSize(17, 17))
        self._plot_check.setCheckable(True)
        self._plot_check.setChecked(mainui.dataset_visibility.get(data.filename, True))
        self._plot_check.setToolTip('Show or hide this recording on Trend')
        self._plot_check.setAccessibleName(f'Show {display_name} on Trend')
        self._plot_check.toggled.connect(self._on_plot_toggled)
        layout.addWidget(self._plot_check)
        self.refresh_theme()

    def set_selected(self, selected):
        if self._selected == selected:
            return
        self._selected = selected
        self.refresh_theme()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.selected.emit(self._data.filename)
        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.selected.emit(self._data.filename)
            event.accept()
            return
        super().keyPressEvent(event)

    def update_plottype(self, plottype):
        self._plottype = plottype

    def refresh_theme(self):
        tk = theme_manager.tokens
        bg = tk['accent_secondary'] if self._selected else tk['surface']
        border = tk['accent'] if self._selected else tk['card_stroke']
        self.setStyleSheet(
            f'QWidget#recordingRow {{ background: {bg}; border: 1px solid {border};'
            f' border-radius: 9px; }}'
            f'QWidget#recordingRow:hover {{ border-color: {tk["accent"]}; }}')
        self._title_label.setStyleSheet(f'color: {tk["text"]}; font-size: 11pt; font-weight: 600;')
        self._meta.setStyleSheet(f'color: {tk["text_secondary"]}; font-size: 9pt;')
        self._indicator.setToolTip(f'Change trace colour ({self._color})')
        self._indicator.setStyleSheet(
            f'QToolButton {{ background: {self._color}; border: 1px solid {tk["card_stroke"]};'
            ' border-radius: 5px; }'
            f'QToolButton:hover, QToolButton:focus {{ border: 2px solid {tk["text"]}; }}')
        self._plot_check.setIcon(icon('eye' if self._plot_check.isChecked() else 'eye-slash'))
        self._title_label._update_text()

    def apply_color(self, color):
        self._color = color
        self.refresh_theme()
        if self._selected:
            self.selected.emit(self._data.filename)

    def _on_plot_toggled(self, checked):
        self._plot_check.setIcon(icon('eye' if checked else 'eye-slash'))
        self._mainui.set_dataset_visible(self._data.filename, checked)

    def _pick_color(self) -> None:
        """Open a native colour dialog and apply the chosen colour."""
        initial = QColor(self._color)
        dlg = QColorDialog(initial, self)
        dlg.setWindowTitle(f"Trace colour — {self._data.filename}")
        # The main window sets a QSS that targets QLabel/QGroupBox/QStatusBar —
        # those rules cascade into QColorDialog and corrupt its native picker
        # (custom-colour swatches and the OK/Cancel buttons end up unreadable,
        # most visibly in light mode). Reparent to None and force the platform
        # style so the dialog renders with its OWN styling, not the window's.
        dlg.setParent(None)
        dlg.setOption(QColorDialog.ColorDialogOption.ShowAlphaChannel, False)
        try:
            from PySide6.QtWidgets import QStyleFactory
            dlg.setStyle(QStyleFactory.create("Windows") or dlg.style())
        except Exception:
            pass
        dlg.setStyleSheet("")  # block inherited QSS
        if dlg.exec() == QColorDialog.DialogCode.Accepted:
            chosen = dlg.currentColor()
            if chosen.isValid():
                self._mainui.set_dataset_color(self._data.filename, chosen.name())


    def _track_dialog(self, win) -> None:
        """Keep *win* alive until it closes, then drop the reference."""
        self._active_dialogs.append(win)
        if hasattr(win, "destroyed"):
            win.destroyed.connect(lambda *_: self._discard_dialog(win))
        if hasattr(win, "finished"):
            win.finished.connect(lambda *_: self._discard_dialog(win))

    def _discard_dialog(self, win) -> None:
        try:
            self._active_dialogs.remove(win)
        except ValueError:
            pass

    def _popup_truncate(self) -> None:
        from dialogs import TruncateTimeDialog
        try:
            win = TruncateTimeDialog(self._mainui, self._data)
            self._track_dialog(win)
            win.show()
        except Exception as e:
            print(e)

    def _popup_angle(self) -> None:
        from dialogs import EditAngleDialog
        try:
            win = EditAngleDialog(self._data, self._mainui)
            self._track_dialog(win)
            win.show()
        except Exception as e:
            print(e)

    def _popup_edit(self) -> None:
        from dialogs import EditDataDialog
        try:
            self._edit_win = EditDataDialog(self._mainui, self._data)
            self._track_dialog(self._edit_win)
            self._edit_win.show()
        except Exception as e:
            print(e)

    def _popup_export(self) -> None:
        from dialogs import ExportDialog
        try:
            win = ExportDialog(self._mainui, self._data)
            self._track_dialog(win)
            win.show()
        except Exception as e:
            print(e)


class _ResizeHandle(QWidget):
    """A thin vertical strip the user drags to resize its parent DataSidebar.

    Sits on the LEFT edge of the sidebar (next to the chart area). Captures
    mouse drag and calls ``parent.resize_width(delta)``.
    """

    def __init__(self, sidebar: "DataSidebar"):
        super().__init__(sidebar)
        self._sidebar = sidebar
        self.setFixedWidth(5)
        self.setCursor(Qt.CursorShape.SplitHCursor)
        self._press_x: int | None = None
        self._press_w: int = 0

    def paintEvent(self, event) -> None:
        from PySide6.QtGui import QPainter
        tk = theme_manager.tokens
        painter = QPainter(self)
        painter.fillRect(0, 0, 1, self.height(), QColor(tk["divider"]))
        painter.end()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_x = int(event.globalPosition().x())
            self._press_w = self._sidebar.width()

    def mouseMoveEvent(self, event) -> None:
        if self._press_x is None:
            return
        # Dragging LEFT (negative delta) widens the sidebar (it's on the right
        # side of the window), dragging RIGHT narrows it.
        delta = self._press_x - int(event.globalPosition().x())
        self._sidebar.resize_width(self._press_w + delta)

    def mouseReleaseEvent(self, event) -> None:
        self._press_x = None


class DataSidebar(QWidget):
    """Searchable recording library with a resizable selected-signal inspector."""
    MIN_WIDTH = 340
    MAX_WIDTH = 560

    def __init__(self, parent=None):
        super().__init__(parent)
        self._mainui = parent
        self._cards = []
        self._selected_name = None
        self._inspector_key = None
        self._pending_details = False
        self._pending_message = 'Updating signal summary...'
        self._plottype = 'Standard'
        self.setObjectName('recordingLibrary')
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setMinimumWidth(self.MIN_WIDTH)
        self.setMaximumWidth(self.MAX_WIDTH)
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self._handle = _ResizeHandle(self)
        outer.addWidget(self._handle)
        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(14, 18, 14, 14)
        body_layout.setSpacing(9)
        heading = QHBoxLayout()
        self._heading = QLabel('Recordings')
        self._heading.setObjectName('libraryTitle')
        heading.addWidget(self._heading)
        self._count = QLabel('0')
        self._count.setObjectName('libraryCount')
        heading.addWidget(self._count)
        heading.addStretch()
        body_layout.addLayout(heading)
        self._search = QLineEdit()
        self._search.setObjectName('recordingSearch')
        self._search.setPlaceholderText('Find a recording…')
        self._search.setToolTip('Find a recording (Ctrl+F)')
        self._search.setAccessibleName('Find a recording')
        self._search.setClearButtonEnabled(True)
        self._search.setFixedHeight(36)
        self._search.textChanged.connect(self._filter_rows)
        body_layout.addWidget(self._search)
        self._splitter = QSplitter(Qt.Orientation.Vertical)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.setHandleWidth(8)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._scroll.setMinimumHeight(134)
        self._content = QWidget()
        self._content.setObjectName('recordingList')
        self._layout = QVBoxLayout(self._content)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self._layout.setSpacing(7)
        self._empty_hint = QLabel('Imported recordings appear here.')
        self._empty_hint.setObjectName('libraryHint')
        self._empty_hint.setWordWrap(True)
        self._empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._layout.addWidget(self._empty_hint)
        self._layout.addStretch()
        self._scroll.setWidget(self._content)
        self._splitter.addWidget(self._scroll)
        self._inspector_scroll = QScrollArea()
        self._inspector_scroll.setWidgetResizable(True)
        self._inspector_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._inspector_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._inspector_scroll.setMinimumHeight(210)
        self._inspector = QWidget()
        self._inspector.setObjectName('signalInspector')
        self._inspector_layout = QVBoxLayout(self._inspector)
        self._inspector_layout.setContentsMargins(14, 14, 14, 14)
        self._inspector_layout.setSpacing(6)
        self._inspector_scroll.setWidget(self._inspector)
        self._splitter.addWidget(self._inspector_scroll)
        self._splitter.setStretchFactor(0, 1)
        self._splitter.setStretchFactor(1, 0)
        self._splitter.setSizes([175, 290])
        self._inspector_scroll.hide()
        body_layout.addWidget(self._splitter, 1)
        outer.addWidget(body, 1)
        self.refresh_theme()

    def refresh_theme(self):
        tk = theme_manager.tokens
        self.setStyleSheet(
            f'QWidget#recordingLibrary {{ background: {tk["surface_alt"]};'
            f' border-left: 1px solid {tk["divider"]}; }}'
            f'QWidget#recordingList, QScrollArea {{ background: transparent; border: none; }}'
            f'QWidget#signalInspector {{ background: {tk["surface"]};'
            f' border: 1px solid {tk["card_stroke"]}; border-radius: 10px; }}'
            f'QLabel#libraryTitle {{ color: {tk["text"]}; font: bold 21px "Segoe UI"; }}'
            f'QLabel#metricValue {{ color: {tk["text"]}; font: bold 17px "Segoe UI"; }}'
            f'QLabel#libraryCount {{ color: {tk["text_secondary"]}; font-size: 10pt; padding: 2px 7px;'
            f' background: {tk["surface"]}; border-radius: 8px; }}'
            f'QLabel#libraryHint {{ color: {tk["text_secondary"]}; padding: 20px 8px; font-size: 10pt; }}'
            f'QLineEdit#recordingSearch {{ background: {tk["surface"]}; color: {tk["text"]};'
            f' border: 1px solid {tk["control_stroke"]}; border-radius: 7px; padding: 0 10px; }}'
            f'QLineEdit#recordingSearch:focus {{ border-color: {tk["accent"]}; }}'
            f'QSplitter::handle {{ background: transparent; }}'
            f'QScrollBar:vertical {{ background: transparent; width: 7px; margin: 0; }}'
            f'QScrollBar::handle:vertical {{ background: {tk["control_stroke"]}; border-radius: 3px; min-height: 24px; }}'
            'QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }'
            'QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }')
        for row in self._cards:
            row.refresh_theme()
        self._render_inspector()

    def add_card(self, card):
        self._cards.append(card)
        card.selected.connect(self._choose_recording)
        self._layout.insertWidget(self._layout.count() - 1, card)
        self._filter_rows(self._search.text())
        current = self._mainui.select_data.currentText()
        self.select_filename(current)

    def _choose_recording(self, filename):
        self._mainui.select_data.setCurrentText(filename)
        self.select_filename(filename)

    def focus_search(self):
        self._search.setFocus(Qt.FocusReason.ShortcutFocusReason)
        self._search.selectAll()

    def select_filename(self, filename):
        if not any(row._data.filename == filename for row in self._cards):
            return
        self._selected_name = filename
        for row in self._cards:
            row.set_selected(row._data.filename == filename)
        self._render_inspector()

    def _filter_rows(self, text):
        visible = 0
        for row in self._cards:
            matches = text.casefold() in row._data.filename.casefold()
            row.setVisible(matches)
            visible += int(matches)
        self._count.setText(f'{visible} / {len(self._cards)}' if text else str(len(self._cards)))
        self._empty_hint.setText('No matching recordings. Try a shorter name.' if self._cards else
                                 'Imported recordings appear here.')
        self._empty_hint.setVisible(visible == 0)

    def _selected_row(self):
        return next((row for row in self._cards if row._data.filename == self._selected_name), None)

    def create_recording_tools(self):
        tools = QWidget()
        layout = QHBoxLayout(tools)
        layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel('RECORDING TOOLS')
        label.setObjectName('caption')
        layout.addWidget(label)

        def run(callback):
            row = self._selected_row()
            if row is not None:
                callback(row)

        for text, callback in [
            ('Trim / split', InfoCard._popup_truncate),
            ('Edit orientation', InfoCard._popup_angle),
            ('Edit axis values', InfoCard._popup_edit),
            ('Trace colour', InfoCard._pick_color),
            ('Export recording', InfoCard._popup_export),
            ('Remove', lambda row: row._remove_callback(False)),
        ]:
            button = QPushButton(text)
            button.setToolTip(f'{text} for the selected recording')
            button.clicked.connect(lambda checked=False, cb=callback: run(cb))
            layout.addWidget(button)
        layout.addStretch()
        return tools

    def _render_inspector(self):
        row = self._selected_row()
        self._inspector_scroll.setVisible(row is not None)
        frame = None if row is None else getattr(row._data, {
            'Standard': 'std_data', 'Filter': 'filtered_data', 'Raw': 'data'}[self._plottype])
        key = None if row is None else (id(row._data), id(frame), self._plottype,
                                        row._color, theme_manager.mode,
                                        self._pending_details, self._pending_message)
        if key is not None and key == self._inspector_key:
            return
        self._inspector_key = key
        self._inspector_frame = frame
        self._inspector_values = {}
        while self._inspector_layout.count():
            item = self._inspector_layout.takeAt(0)
            if item.widget():
                item.widget().hide()
                item.widget().deleteLater()
        if row is None:
            return
        data = row._data
        tk = theme_manager.tokens
        layout = self._inspector_layout
        cutoff = getattr(data, '_filtered_cutoff', None)
        title = QLabel('Signal summary' if self._plottype == 'Raw' or cutoff is None
                       else f'Signal summary / {cutoff:g} Hz')
        title.setStyleSheet(f'font-size: 12pt; font-weight: bold; color: {tk["text"]};')
        layout.addWidget(title)
        mode = {'Standard': 'Calibrated', 'Filter': 'Filtered', 'Raw': 'Raw'}[self._plottype]
        metrics = QWidget()
        metrics_layout = QGridLayout(metrics)
        metrics_layout.setContentsMargins(0, 2, 0, 2)
        metrics_layout.setVerticalSpacing(0)
        for column, (label, reading) in enumerate([('SAMPLE RATE', f'{data.fs:g} Hz'),
                                                   ('SAMPLES', f'{len(data.data):,}')]):
            key_label = QLabel(label)
            key_label.setObjectName('caption')
            value_label = QLabel(reading)
            value_label.setObjectName('metricValue')
            metrics_layout.addWidget(key_label, 0, column)
            metrics_layout.addWidget(value_label, 1, column)
            metrics_layout.setColumnStretch(column, 1)
        layout.addWidget(metrics)
        first, last = data.data['Time'].iloc[[0, -1]]
        times = QLabel(f'{mode}   ·   {first:,.3f} to {last:,.3f} s')
        times.setStyleSheet(f'color: {tk["text_secondary"]}; font-size: 9pt;')
        layout.addWidget(times)
        pending = self._pending_details and self._plottype != 'Raw'
        pending_label = QLabel(self._pending_message)
        pending_label.setWordWrap(True)
        pending_label.setStyleSheet(f'color: {tk["warning"]}; font-size: 10pt; padding: 10px 0;')
        layout.addWidget(pending_label)
        pending_label.setVisible(pending)
        from sci_calculation import ProcessASTM
        pos, neg = ([0] * 3, [0] * 3) if pending else ProcessASTM(frame, max(1, round(.2 * data.fs)))
        table = QFrame()
        table.setObjectName('signalStats')
        table.setStyleSheet(f'QFrame#signalStats {{ background: {tk["window_bg"]};'
                            f' border: 1px solid {tk["divider"]}; border-radius: 6px; }}')
        self._stats_panel = table
        grid = QGridLayout(table)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setHorizontalSpacing(5)
        grid.setVerticalSpacing(8)
        for col, text in enumerate(['Axis', 'Max / min (g)', '0.2 s + / - (g)']):
            label = QLabel(text)
            label.setStyleSheet(f'color: {tk["text_secondary"]}; font-size: 8pt;')
            if col:
                label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            grid.addWidget(label, 0, col)
        self._inspector_values = {}
        for axis, name in enumerate('XYZ'):
            label = QLabel(name)
            label.setStyleSheet(f'font-size: 10pt; font-weight: 600; color: {tk["text"]};')
            grid.addWidget(label, axis + 1, 0)
            values = frame.iloc[:, axis + 1]
            for col, pair in enumerate([(values.max(), values.min()), (pos[axis], neg[axis])], 1):
                value = QLabel(f'{pair[0]:+.3f} / {pair[1]:+.3f}')
                value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                value.setStyleSheet(f'color: {tk["text"]}; font-size: 9pt;')
                grid.addWidget(value, axis + 1, col)
                self._inspector_values[(name, col)] = value
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(2, 1)
        layout.addWidget(table)
        table.setVisible(not pending)
        orientation = QLabel(f'Pitch {data.pitch_angle:g}°  /  Seat {data.seatback_angle:g}°\n'
                             f'Roll {data.roll_angle:g}°  /  Yaw {data.yaw_angle:g}°')
        orientation.setStyleSheet(f'color: {tk["text_secondary"]}; font-size: 9pt;')
        layout.addWidget(orientation)
        layout.addStretch()

    def set_pending_details(self, pending, message='Updating signal summary...'):
        self._pending_details = pending
        self._pending_message = message
        self._render_inspector()

    def update_plottype(self, plottype):
        self._plottype = plottype
        for row in self._cards:
            row.update_plottype(plottype)
        self._render_inspector()

    def clear_all(self):
        for row in self._cards:
            self._layout.removeWidget(row)
            row.hide()
            row.deleteLater()
        self._cards.clear()
        self._selected_name = None
        if not self._mainui.datalist:
            self._search.clear()
        self._filter_rows(self._search.text())
        self._render_inspector()

    def resize_width(self, width):
        self.setFixedWidth(max(self.MIN_WIDTH, min(self.MAX_WIDTH, int(width))))
