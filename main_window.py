# coding: utf-8
"""
BERT Reader V3.1 — Main Window (PySide6 + Qt Charts).

Modernised reconstruction of the original PyQt5 + matplotlib application.
"""

from __future__ import annotations

import os
import sys

import numpy as np

from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtGui import QIcon, QColor, QAction, QActionGroup, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QToolBar, QWidget, QFileDialog,
    QLabel, QFrame, QToolButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QSizePolicy,
    QApplication,
)

from PySide6.QtWidgets import QComboBox, QSpinBox, QDoubleSpinBox, QPushButton
from ui_controls import icon, icon_action, PrimaryButton

import accdata
from sidebar import DataSidebar, InfoCard
from dialogs import CutoffFrequencyDialog, ReportDialog
from chart_trend import TrendChartView
from chart_gb import GBChartView
from chart_acczone import AccZoneChartView
from chart_astm import ASTMChartView
from chart_reversal import ZReversalChartView, XYReversalChartView
from sci_calculation import allowance_at_t0
from workspace_shell import WorkspaceWindow, caption
from processing_controller import ProcessingController
from theme import DATASET_COLORS, theme_manager

# When frozen by PyInstaller, __file__ points inside the temp _MEIxxxxx
# extraction folder, which is useless as a "current folder" default for the
# file dialogs. Use the real exe location in that case instead.
if getattr(sys, "frozen", False):
    SCRIPT_DIR = os.path.dirname(sys.executable)
else:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ======================================================================
# Helper: build chart-control toolbar (Zoom In / Zoom Out / Reset View)
# ======================================================================

def _add_chart_toolbar(toolbar: QToolBar, chart_widget) -> None:
    """Append zoom/pan/reset actions to a QToolBar for a chart widget.

    chart_widget can be a QChartView (AccZoneChartView) or a QWidget container
    (TrendChartView, ASTMChartView, GBChartView) that has zoom_mode() method.

    Zoom and pan are mutually exclusive.
    """
    zoom_btn = toolbar.addAction('Zoom')
    zoom_btn.setCheckable(True)
    zoom_btn.setStatusTip("Enable zoom by dragging (rubber band)")

    pan_btn = toolbar.addAction('Pan')
    pan_btn.setCheckable(True)
    pan_btn.setStatusTip("Enable pan mode (drag to scroll)")

    def _on_zoom(checked: bool) -> None:
        if checked:
            pan_btn.blockSignals(True)
            pan_btn.setChecked(False)
            pan_btn.blockSignals(False)
            chart_widget.zoom_mode("zoom")
        else:
            chart_widget.zoom_mode("default")

    def _on_pan(checked: bool) -> None:
        if checked:
            zoom_btn.blockSignals(True)
            zoom_btn.setChecked(False)
            zoom_btn.blockSignals(False)
            chart_widget.zoom_mode("pan")
        else:
            chart_widget.zoom_mode("default")

    zoom_btn.toggled.connect(_on_zoom)
    pan_btn.toggled.connect(_on_pan)

    def _reset() -> None:
        zoom_btn.blockSignals(True)
        pan_btn.blockSignals(True)
        zoom_btn.setChecked(False)
        pan_btn.setChecked(False)
        zoom_btn.blockSignals(False)
        pan_btn.blockSignals(False)
        chart_widget.zoom_mode("default")
        chart_widget.reset_view()

    a_reset = icon_action('arrow-counter-clockwise', "Reset", toolbar)
    a_reset.setToolTip('Reset view')
    a_reset.setStatusTip("Reset chart to fit all data")
    a_reset.triggered.connect(_reset)
    toolbar.addAction(a_reset)


class MainWindow(ProcessingController, WorkspaceWindow):
    """BERT Reader analysis workspace."""

    def __init__(self):
        super().__init__()
        self.version = "3.1"
        self.setWindowTitle("BERT_READER X")
        self.resize(1600, 900)
        self.setMinimumSize(1100, 720)

        # State
        self.plottype = "Standard"
        self.annot_mode = "Single"
        self.cutoff = 5
        self.datalist: list = []
        self.data_namelist: list = []
        self.datadict: dict = dict()
        self.stat_content: list = []
        # Directory to open the file dialog in. Remembers the folder of the
        # last loaded file so repeat loads don't make the user navigate back
        # from the app directory each time. Falls back to SCRIPT_DIR.
        self._last_open_dir: str = SCRIPT_DIR
        # filename -> hex colour. Single source of truth for each dataset's
        # colour, so users can change it at runtime and every chart follows.
        self.dataset_colors: dict = {}
        # filename -> bool. Tracks whether each dataset's trend curves are
        # shown. Lets users hide a dataset's plot without deleting the data.
        self.dataset_visibility: dict = {}

        # Combo boxes for per-tab data selection (created before _setup_ui).
        # These are now HIDDEN mirrors of the top-bar ``select_data`` combo:
        # they stay as attributes because the per-page Generate handlers and
        # set_dataset_color read them, but they are no longer placed in any
        # toolbar. _wire_data_selector_sync() (called from _build_top_bar)
        # fans the top-bar selection out to all of them.
        self.select_plot_tab2 = QComboBox()
        self.select_plot_tab2b = QComboBox()   # mirror for the split ASTM tab
        self.select_plot_tab3 = QComboBox()
        self.select_plot_tab4 = QComboBox()
        self.select_plot_tab5 = QComboBox()    # Z Reversal tab
        self.select_plot_tab6 = QComboBox()    # X/Y Reversal tab
        self._data_combos = [
            self.select_plot_tab2, self.select_plot_tab2b,
            self.select_plot_tab3, self.select_plot_tab4,
            self.select_plot_tab5, self.select_plot_tab6,
        ]

        # Build UI
        self._setup_ui()
        self._update_status_label()

        # Wire theme changes: re-apply QSS + refresh inline styles.
        theme_manager.changed.connect(self._on_theme_changed)
        # Apply current theme immediately (covers re-entrancy on toggle).
        self._on_theme_changed(theme_manager.mode)
        self._init_processing()

        # Real chart rendering warms naturally when data is loaded. Keeping
        # startup free of synthetic plots also leaves the empty state stable.

    def _on_theme_changed(self, _mode: str) -> None:
        """Refresh the workspace when the appearance changes."""
        if hasattr(self, 'theme_toggle'):
            self.theme_toggle.blockSignals(True)
            self.theme_toggle.setChecked(theme_manager.is_dark)
            self.theme_toggle.blockSignals(False)

        self.refresh_theme()
        if hasattr(self, '_processing_state'):
            self._set_processing_state(self._processing_state, self._workspace_title.toolTip())

    def refresh_theme(self) -> None:
        """Apply the active palette to controls, icons and chart canvases."""
        self.setStyleSheet(theme_manager.app_qss)
        self._refresh_action_icons()
        self.sidebar.refresh_theme()
        for canvas in (self.trendplot, self.canvasASTMSust, self.canvasASTMEgg,
                       self.canvasGB, self.canvasAccZone,
                       self.canvasZReversal, self.canvasXYReversal):
            if hasattr(canvas, "restyle_theme"):
                canvas.restyle_theme()

    def _refresh_action_icons(self):
        for action in self.findChildren(QAction):
            if action.property('glyph'):
                action.setIcon(icon(action.property('glyph')))
        for button in self._import_buttons:
            button.setIcon(icon(button.property('formatGlyph') or 'folder-plus'))
        self._library_toggle.setIcon(icon('books'))
        self.theme_toggle.setIcon(icon('sun' if theme_manager.is_dark else 'moon'))

    # ==================================================================
    # UI Setup
    # ==================================================================

    def _setup_ui(self) -> None:
        # All import entry points use the same labels and readers.
        self._import_formats = [
            ('activity', 'Raw DAQ recording', self.open_raw_data),
            ('file-text', 'Acceleration recording / SUP', self.open_accel_data),
            ('waves', 'SensorConnect CSV', self.open_sensorconnect_data),
        ]
        self._import_buttons = []
        self.sidebar = DataSidebar(self)
        self._library_host = self.sidebar
        self._work_area.addWidget(self.sidebar)

        # --- Chart canvases (created before pages so toolbars can wire up) ---
        self.canvasASTMSust = ASTMChartView(chart_names=["ax", "ay", "az"])
        self.canvasASTMEgg = ASTMChartView(chart_names=["xy", "xz", "yz"])
        self.canvasGB = GBChartView()
        self.canvasAccZone = AccZoneChartView()
        self.trendplot = TrendChartView()
        self.canvasZReversal = ZReversalChartView()
        self.canvasXYReversal = XYReversalChartView()

        # --- Persistent top control bar (Add Data / selector / cutoff / Annot).
        # Built before pages so the shared selector exists when pages sync to it.
        self._build_top_bar()

        # --- Build each page (a QWidget) and register it as a nav item ---
        self._setup_trend_page()
        self._setup_astm_sust_page()
        self._setup_astm_egg_page()
        self._setup_gb_page()
        self._setup_acczone_page()
        self._setup_zreversal_page()
        self._setup_xyreversal_page()
        self.stackedWidget.currentChanged.connect(self._update_view_heading)
        self._update_view_heading()

        self._build_control_panel()
        self._update_workspace_state()

        self._setup_theme_toggle()

        # --- Status bar ---
        self.statusBar().showMessage("Ready")

    def _find_recording(self):
        self._library_manually_set = True
        self._library_toggle.setChecked(True)
        self.sidebar.focus_search()

    def _update_view_heading(self, *_):
        page = self.stackedWidget.currentWidget()
        if page:
            index = self.stackedWidget.currentIndex()
            self._view_heading.setText('Signal timeline' if index == 0 else page.windowTitle())
            self._nav_buttons[index].setChecked(True)
            for i in range(self.stackedWidget.count()):
                self.stackedWidget.widget(i)._tools.setVisible(i == index)

    def _add_chart_page(self, page, icon, title):
        page.setWindowTitle(title)
        self._register_view(page, icon, title)

    # -- helper: wrap a page widget + its toolbar into a vertical layout ------
    def _make_page(self, object_name: str, toolbars, central, margins=(16, 6, 16, 10)):
        """Keep chart commands above the plot and the resizable library."""
        page = QWidget()
        page.setObjectName(object_name)
        lay = QVBoxLayout(page)
        lay.setContentsMargins(*margins)
        lay.setSpacing(6)
        page._tools = QWidget()
        tools_layout = QVBoxLayout(page._tools)
        tools_layout.setContentsMargins(0, 0, 0, 0)
        tools_layout.setSpacing(4)
        for tb in toolbars:
            tools_layout.addWidget(tb)
        self._chart_tools_layout.addWidget(page._tools)
        page._chart_content = central
        page._pending_label = QLabel(page)
        page._pending_label.setObjectName('pendingChart')
        page._pending_label.setTextFormat(Qt.TextFormat.PlainText)
        page._pending_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        page._pending_label.setWordWrap(True)
        lay.addWidget(page._pending_label, 1)
        page._pending_label.hide()
        lay.addWidget(central, 1)
        return page

    def _setup_theme_toggle(self) -> None:
        """Place the appearance switch beside the workspace actions."""
        self.theme_toggle = QToolButton(self)
        self.theme_toggle.setIcon(icon('moon'))
        self.theme_toggle.setToolTip("Toggle light / dark theme")
        self.theme_toggle.setStatusTip("Toggle light / dark theme")
        self.theme_toggle.setCheckable(True)
        self.theme_toggle.setChecked(theme_manager.is_dark)
        self.theme_toggle.toggled.connect(self._toggle_theme)
        self._header_actions.addWidget(self.theme_toggle)

    @staticmethod
    def _apply_elevation(widget: QWidget, blur: int = 18, dy: int = 3,
                         alpha: int = 40) -> None:
        """Attach a soft drop shadow to *widget*.

        Kept for callers that still want a drop shadow (e.g. dialog cards).
        Skips silently if QGraphicsDropShadowEffect isn't usable.
        """
        try:
            from PySide6.QtWidgets import QGraphicsDropShadowEffect
            effect = QGraphicsDropShadowEffect(widget)
            effect.setBlurRadius(blur)
            effect.setOffset(0, dy)
            effect.setColor(QColor(0, 0, 0, alpha))
            widget.setGraphicsEffect(effect)
        except Exception as e:
            print(f"Drop shadow skipped: {e}")

    @staticmethod
    def _mk_toolbar(name: str) -> QToolBar:
        """Create a non-movable QToolBar whose actions show icon + text.

        QToolBar defaults to ``IconOnly``, which hides the text of every
        ``Action`` it holds (the migration's "buttons lost their labels"
        symptom). Setting ``TextBesideIcon`` here keeps the label readable
        on every toolbar (Main, Trend, per-tab controls, chart tools).
        """
        tb = QToolBar(name)
        tb.setMovable(False)
        tb.setIconSize(QSize(16, 16))
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        return tb

    # ------------------------------------------------------------------
    # Persistent workspace commands and active source.
    # ------------------------------------------------------------------

    def _tb_button_for(self, action):
        button = QToolButton(self._top_bar)
        button.setDefaultAction(action)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        return button

    def _build_top_bar(self) -> None:
        """Keep file commands, import formats and analysis inputs visible."""
        lay = self._top_bar_layout
        actions = self._header_actions = QHBoxLayout()
        actions.setSpacing(8)
        heading = QVBoxLayout()
        heading.setSpacing(2)
        self._view_heading = caption('', 'workspaceTitle')
        heading.addWidget(self._view_heading)
        actions.addLayout(heading)
        actions.addSpacing(8)
        self._workspace_title = QLabel()
        actions.addWidget(self._workspace_title, 0, Qt.AlignmentFlag.AlignVCenter)
        actions.addStretch(1)
        self._data_actions = {}
        files = QHBoxLayout()
        files.addWidget(caption('IMPORT'))
        for glyph, label, reader in self._import_formats:
            button = QPushButton(label)
            button.setProperty('formatGlyph', glyph)
            button.clicked.connect(reader)
            self._import_buttons.append(button)
            files.addWidget(button)
        files.addStretch(1)
        for glyph, text, slot in [
            ('export', "Export all", self._export_all_to_sup),
            ('file-text', "Word report", self._show_report_dialog),
            ('trash', "Clear all", lambda: self.reset_trend(reset_datalist=True)),
        ]:
            action = icon_action(glyph, text, self)
            action.triggered.connect(slot)
            actions.addWidget(self._tb_button_for(action))
            self._data_actions[text] = action
        self._library_toggle = QToolButton()
        self._library_toggle.setText("Library")
        self._library_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._library_toggle.setCheckable(True)
        self._library_manually_set = False
        self._library_toggle.clicked.connect(self._remember_library_choice)
        self._library_toggle.setToolTip("Show or hide the recording library")
        self._library_toggle.toggled.connect(self._library_host.setVisible)
        actions.addWidget(self._library_toggle)
        QShortcut(QKeySequence.StandardKey.Find, self, activated=self._find_recording)
        lay.addLayout(actions)
        lay.addLayout(files)

        inputs = QHBoxLayout()
        inputs.setSpacing(10)
        inputs.addWidget(caption('SOURCE'))
        self.select_data = QComboBox()
        self.select_data.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.select_data.setMinimumWidth(150)
        self.select_data.setSizePolicy(QSizePolicy.Policy.Expanding,
                                       QSizePolicy.Policy.Fixed)
        self.select_data.setToolTip("Active dataset for the standard and reversal charts")
        inputs.addWidget(self.select_data, 1)
        self._generate_button = PrimaryButton("Update charts")
        self._generate_button.setToolTip("Refresh all analysis charts for the selected dataset")
        self._generate_button.clicked.connect(self._generate_active_page)
        inputs.addWidget(caption('LOW-PASS'))
        self.cutoff_spin = QSpinBox()
        self.cutoff_spin.setRange(1, 249)
        self.cutoff_spin.setValue(self.cutoff)
        self.cutoff_spin.setSuffix(" Hz")
        self.cutoff_spin.setFixedWidth(120)
        self.cutoff_spin.setToolTip("Low-pass cutoff. Acceleration Zone always uses 1 Hz.")
        self.cutoff_spin.setKeyboardTracking(False)
        self.cutoff_spin.valueChanged.connect(self._schedule_cutoff_change)
        inputs.addWidget(self.cutoff_spin)
        inputs.addWidget(self._generate_button)
        self._input_actions = inputs
        lay.addLayout(inputs)
        self._recording_tools = self.sidebar.create_recording_tools()
        self._chart_tools_layout.addWidget(self._recording_tools)
        self._cutoff_timer = QTimer(self)
        self._cutoff_timer.setSingleShot(True)
        self._cutoff_timer.setInterval(250)
        self._cutoff_timer.timeout.connect(
            lambda: self._on_cutoff_changed(self.cutoff_spin.value()))

        # Keep secondary readouts out of the command rows.
        self.annot_label = QLabel(f"Annotations: {self.annot_mode}")
        self.status_label = QLabel()
        self.statusBar().addPermanentWidget(self.annot_label)
        self.statusBar().addPermanentWidget(self.status_label)
        self._wire_data_selector_sync()

    def _remember_library_choice(self):
        self._library_manually_set = True

    def _update_workspace_state(self) -> None:
        loaded = bool(self.datalist)
        if not self._library_manually_set:
            self._library_toggle.setChecked(loaded)
            self._library_host.setVisible(loaded)
        for action in self._data_actions.values():
            action.setEnabled(loaded)
        self._generate_button.setEnabled(loaded)
        self._recording_tools.setEnabled(loaded)
        self.select_data.setEnabled(loaded)
        if not loaded:
            self.select_data.setPlaceholderText("Select a recording")
        self.select_data.setToolTip(self.select_data.currentText())
        empty = getattr(self, '_trend_empty', None)
        if empty is not None:
            empty.setVisible(not loaded)
            self.trendplot.setVisible(loaded)
            self._trend_tools.setEnabled(loaded)
            self._shift_tools.setEnabled(loaded)
            for action in self._peak_actions:
                action.setEnabled(len(self.datalist) > 1)

    def _schedule_cutoff_change(self, _frequency: int) -> None:
        if hasattr(self, '_jobs') and self.datalist:
            self._refresh_timer.stop()
            self._mark_charts_outdated(trend=True)
        self._cutoff_timer.start()
        if hasattr(self, '_jobs'):
            self._update_job_controls()

    def _wire_data_selector_sync(self) -> None:
        """One-way sync: top-bar ``select_data`` → every per-page combo.

        The per-page combos are now hidden mirrors; they're read by the
        per-page Generate handlers and set_dataset_color, so they must track
        the shared selector. Signals are blocked on the mirrors to avoid any
        feedback. (No reverse wiring — the mirrors are not user-interactable.)
        """
        def _fanout(text):
            if text is None:
                return
            self.select_data.setToolTip(text)
            self.sidebar.select_filename(text)
            for cb in self._data_combos:
                if cb.currentText() == text:
                    continue
                cb.blockSignals(True)
                cb.setCurrentText(text)
                cb.blockSignals(False)
        self.select_data.currentTextChanged.connect(_fanout)

    def _on_cutoff_changed(self, freq: int) -> None:
        """The debounced cutoff is filtered by the background preparation job."""
        self.set_cutoff(int(freq))
        self.request_chart_update(force=True)

    def _toggle_theme(self, checked: bool) -> None:
        """checked == True → dark mode."""
        new_mode = "dark" if checked else "light"

        theme_manager.set_mode(new_mode)

    def _toggle_analysis(self, checked: bool) -> None:
        """Toggle ±peak markers + sustained-peak regions on the Trend chart.

        The sustained-peak sliding-window width is taken from the spinbox next
        to the Analyze button (default 0.2 s). Changing the spinbox value while
        analysis is shown re-applies it with the new window.
        """
        if not self.datalist:
            return
        if self.trendplot._analysis_active == checked:
            return
        window = 0.2
        if getattr(self, "_analyze_window_spin", None) is not None:
            try:
                window = float(self._analyze_window_spin.value())
            except Exception:
                window = 0.2
        active = self.trendplot.toggle_analysis(
            self.datalist, self.plottype, self.dataset_colors,
            window=window)
        self._analyze_btn.setChecked(active)

    def _on_analysis_window_changed(self, _value: float) -> None:
        """Re-run analysis with the new window width if it's currently shown."""
        if getattr(self, "_analyze_btn", None) is not None and \
                self._analyze_btn.isChecked():
            # Refresh: clear overlays, then re-apply with the new window width.
            for sub in (self.trendplot._chart_x, self.trendplot._chart_y,
                        self.trendplot._chart_z):
                sub.clear_analysis()
            self.trendplot._analysis_active = False
            self._toggle_analysis(True)

    # ------------------------------------------------------------------
    # Tab 1: Trend
    # ------------------------------------------------------------------

    def _setup_trend_page(self) -> None:
        tb = self._mk_toolbar("Trend view")
        self._plot_type_actions = {}
        group = QActionGroup(self)
        group.setExclusive(True)
        for text, value in [("Raw", "Raw"), ("Filtered", "Filter"),
                            ("Calibrated", "Standard")]:
            action = QAction(text, self)
            action.setCheckable(True)
            action.setChecked(self.plottype == value)
            action.triggered.connect(lambda checked=False, v=value: self.switch_plot_type(v))
            group.addAction(action)
            tb.addAction(action)
            self._plot_type_actions[value] = action
        tb.addSeparator()
        tb.addWidget(caption('ALIGN PEAKS'))
        self._peak_actions = []
        for which, callback in [('Max', self.switch_to_fitmax), ('Min', self.switch_to_fitmin)]:
            for axis, name in enumerate('XYZ', 1):
                action = tb.addAction(f'{which} {name}')
                action.setToolTip(f'Align recordings at their {name}-axis {which.lower()}imum')
                action.triggered.connect(lambda checked=False, a=axis, cb=callback: cb(a))
                self._peak_actions.append(action)

        self._trend_tools = tb
        tb_shift = self._shift_tools = self._mk_toolbar("Time alignment")
        tb_shift.addWidget(caption('TIME SHIFT'))
        self._shift_combo = QComboBox()
        self._shift_combo.setFixedWidth(165)
        self._shift_combo.setToolTip("Select a dataset to shift in time")
        tb_shift.addWidget(self._shift_combo)
        self._shift_spin = QDoubleSpinBox()
        self._shift_spin.setRange(-10000.0, 10000.0)
        self._shift_spin.setDecimals(3)
        self._shift_spin.setSingleStep(0.001)
        self._shift_spin.setSuffix(" s")
        self._shift_spin.setFixedWidth(112)
        self._shift_spin.setToolTip("Display shift in seconds; positive moves the trace earlier")
        tb_shift.addWidget(self._shift_spin)
        self._shift_apply = QPushButton("Apply")
        self._shift_apply.clicked.connect(self._apply_manual_shift)
        tb_shift.addWidget(self._shift_apply)
        self._shift_reset = QPushButton("Reset shift")
        self._shift_reset.clicked.connect(self._reset_manual_shift)
        tb_shift.addWidget(self._shift_reset)
        tb_shift.addSeparator()
        annotate = icon_action('tag', "Annotate", self)
        annotate.triggered.connect(self.switch_annot_mode)
        tb_shift.addAction(annotate)
        _add_chart_toolbar(tb_shift, self.trendplot)

        self._analyze_window_spin = QDoubleSpinBox(self._ctrl_panel)
        self._analyze_window_spin.setRange(0.001, 60.0)
        self._analyze_window_spin.setDecimals(3)
        self._analyze_window_spin.setSingleStep(0.1)
        self._analyze_window_spin.setSuffix(" s")
        self._analyze_window_spin.setValue(0.2)
        self._analyze_window_spin.setKeyboardTracking(False)
        self._analyze_window_spin.setToolTip("Trend sustained-peak window")
        self._analyze_window_spin.valueChanged.connect(self._on_analysis_window_changed)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        self._trend_empty = QFrame()
        self._trend_empty.setObjectName("emptyState")
        empty_layout = QVBoxLayout(self._trend_empty)
        empty_layout.setContentsMargins(24, 16, 24, 16)
        empty_layout.addStretch()
        chooser = QWidget()
        chooser.setMaximumWidth(530)
        chooser_layout = QVBoxLayout(chooser)
        chooser_layout.setContentsMargins(0, 0, 0, 0)
        chooser_layout.setSpacing(10)
        chooser_layout.addWidget(caption('Open a recording', 'emptyTitle'))
        chooser_layout.addWidget(caption('Choose a format in the Import row above to plot your signals.', 'secondaryText'))
        chooser_row = QHBoxLayout()
        chooser_row.addStretch(1)
        chooser_row.addWidget(chooser, 4)
        chooser_row.addStretch(1)
        empty_layout.addLayout(chooser_row)
        empty_layout.addStretch()
        content_layout.addWidget(self._trend_empty, 1)
        content_layout.addWidget(self.trendplot, 1)
        self.trendplot.hide()
        page = self._make_page("page_trend", [tb, tb_shift], content)
        self._add_chart_page(page, 'activity', "Trend")

    # ------------------------------------------------------------------
    # Pages 3 & 4: ASTM F2291
    # ------------------------------------------------------------------

    def _setup_astm_sust_page(self) -> None:
        """Sustained-acceleration page.

        Restraint / condition / height / prone / Analyze / Generate all live in
        shared global controls (control panel + top bar); this page keeps only
        the chart tools + the analysis summary.
        """
        tb2 = self._mk_toolbar("ASTM Sustained Chart Tools")
        _add_chart_toolbar(tb2, self.canvasASTMSust)
        tb2.addSeparator()
        # Analysis summary: a row of coloured "Fail X" labels (one per failing
        # axis/side). Cleared when all pass / analyze is off. The Analyze toggle
        # itself lives in the shared control panel (routes here on this page).
        self._sust_summary_host = QWidget(tb2)
        QHBoxLayout(self._sust_summary_host).setContentsMargins(0, 0, 0, 0)
        tb2.addWidget(self._sust_summary_host)

        page = self._make_page("page_astm_sust", [tb2], self.canvasASTMSust)
        self._add_chart_page(page, 'chart-line', "ASTM Sustained")

    def _setup_astm_egg_page(self) -> None:
        """Eggplot page — chart tools + analysis summary.

        Restraint / condition / height / prone / Analyze / Generate all live in
        shared global controls (control panel + top bar); this page keeps only
        the chart tools + the analysis summary.
        """
        tb2 = self._mk_toolbar("ASTM Eggplot Chart Tools")
        _add_chart_toolbar(tb2, self.canvasASTMEgg)
        tb2.addSeparator()
        # Analysis summary: a row of coloured "Fail X" labels (one per broken
        # rule, coloured to match its marker). Cleared when all pass / analyze
        # is off. The Analyze toggle itself lives in the shared control panel.
        self._egg_summary_host = QWidget(tb2)
        QHBoxLayout(self._egg_summary_host).setContentsMargins(0, 0, 0, 0)
        tb2.addWidget(self._egg_summary_host)

        page = self._make_page("page_astm_egg", [tb2], self.canvasASTMEgg)
        self._add_chart_page(page, 'circles-three', "ASTM Eggplot")

    def _build_control_panel(self) -> None:
        """Populate the shared control panel under the nav tabs.

        These are the SINGLE instances of the restraint / condition / height /
        prone / Analyze controls, replacing the per-page mirrored copies. Every
        page's Generate handler and analyze toggle reads these (by the canonical
        attribute names ``select_restraint`` / ``select_cond`` / ``height_input``
        / ``_prone_btn``), so no read-site rewiring is needed beyond the X/Y
        Reversal handler (which previously read its own _xr_* copies).
        """
        lay = self._ctrl_panel_layout
        heading = QLabel("ANALYSIS SETTINGS")
        heading.setObjectName("sectionHeading")
        lay.addWidget(heading)

        # --- Restraint type ---
        lay.addWidget(QLabel("Restraint:"))
        self.select_restraint = QComboBox()
        self.select_restraint.addItems([
            'None', 'Individual Lower Body', 'Upper Body',
            'Group Lower Body', 'Convenience Restraint', 'No Restraint',
        ])
        lay.addWidget(self.select_restraint)

        # --- Condition ---
        lay.addWidget(QLabel("Condition:"))
        self.select_cond = QComboBox()
        self.select_cond.addItems(['Normal', 'E-Stop', 'Expected/Permitted Bumping'])
        for combo in (self.select_restraint, self.select_cond):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(9)
            combo.setToolTip(combo.currentText())
            combo.currentTextChanged.connect(combo.setToolTip)
        lay.addWidget(self.select_cond)

        # --- Height (inches) ---
        lay.addWidget(QLabel("Height (in):"))
        self.height_input = QSpinBox()
        self.height_input.setRange(0, 80)
        self.height_input.setValue(0)
        lay.addWidget(self.height_input)

        # --- Prone toggle ---
        # Toggling prone re-plots every chart immediately (it changes the ASTM
        # egg shapes + the sustained ax- limit line), so the user sees the
        # update without needing to click Generate.
        self._prone_btn = QToolButton(self._ctrl_panel)
        self._prone_btn.setText("Prone")
        self._prone_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self._prone_btn.setCheckable(True)
        self._prone_btn.setStatusTip(
            "Prone posture: swap XY/XZ base eggs to the prone shape + "
            "override the sustained ax- limit")
        self._prone_btn.toggled.connect(self._on_prone_toggled)
        lay.addWidget(self._prone_btn)

        lay.addWidget(QLabel("Trend window"))
        lay.addWidget(self._analyze_window_spin)
        lay.addSpacing(6)

        # --- Shared Analyze button (routes to the visible page's handler) ---
        self._analyze_btn = QToolButton(self._ctrl_panel)
        self._analyze_btn.setText("Analyze")
        self._analyze_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self._analyze_btn.setCheckable(True)
        self._analyze_btn.setStatusTip(
            "Analyze the currently-visible chart (switch tabs to analyze another)")
        self._analyze_btn.toggled.connect(self._toggle_active_page_analysis)
        self._input_actions.addWidget(self._analyze_btn)
        self._analyze_btn.setEnabled(bool(self.datalist))
        lay.addStretch(1)

        # Re-sync the Analyze button's checked state when the user switches
        # pages (each canvas tracks its own analysis on/off state).
        # Shared Analyze is a GLOBAL on/off: when ON, every tab the user switches
        # to auto-runs its analysis; when OFF, every tab clears. The button's
        # checked state is the global truth — it does NOT follow per-canvas state.
        self._analyze_global_on = False
        self.stackedWidget.currentChanged.connect(self._on_page_changed_for_analyze)

    # ------------------------------------------------------------------
    # Shared Analyze dispatch (GLOBAL on/off)
    # ------------------------------------------------------------------

    def _analyze_dispatch(self):
        """Return the analyze handler for the currently-visible page."""
        page = self.stackedWidget.currentWidget()
        if page is None:
            return None
        name = page.objectName()
        entry = {
            "page_trend": (self._toggle_analysis, self.trendplot, '_data_by_dataset'),
            "page_astm_sust": (self._toggle_sust_analysis, self.canvasASTMSust, '_last_egg_data'),
            "page_astm_egg": (self._toggle_egg_analysis, self.canvasASTMEgg, '_last_egg_data'),
            "page_gb": (self._toggle_gb_analysis, self.canvasGB, '_last_comb_data'),
            "page_acczone": (self.canvasAccZone.toggle_analysis, self.canvasAccZone, '_last_data'),
            "page_zreversal": (self._toggle_zreversal_analysis, self.canvasZReversal, '_cached_data'),
            "page_xyreversal": (self._toggle_xyreversal_analysis, self.canvasXYReversal, '_cached_data'),
        }.get(name)
        if entry is None:
            return None
        handler, canvas, field = entry

        def dispatch(checked):
            if checked and hasattr(self, '_jobs') and not self._page_is_current():
                self._queue_chart_refresh()
                return
            source = getattr(canvas, field, None)
            if checked and (source is None or len(source) == 0):
                self.statusBar().showMessage("Update charts before analyzing this view", 4000)
                return
            handler(checked)
        return dispatch

    def _toggle_active_page_analysis(self, checked: bool) -> None:
        """Global Analyze toggle: set the global state + apply to the visible page.

        The shared button drives a single global on/off. Turning it ON runs
        analysis on the visible page (and any page switched to afterwards);
        turning it OFF clears analysis everywhere. Each page's own on/off is
        driven by this global state, so the button stays consistent across tabs.
        """
        if checked and not self.datalist:
            self._analyze_btn.blockSignals(True)
            self._analyze_btn.setChecked(False)
            self._analyze_btn.blockSignals(False)
            return
        self._analyze_global_on = checked
        if not checked:
            for handler in (self._toggle_analysis, self._toggle_sust_analysis,
                            self._toggle_egg_analysis, self._toggle_gb_analysis,
                            self.canvasAccZone.toggle_analysis,
                            self._toggle_zreversal_analysis,
                            self._toggle_xyreversal_analysis):
                handler(False)
            return
        handler = self._analyze_dispatch()
        if handler is not None:
            handler(checked)

    def _on_page_changed_for_analyze(self, *_args) -> None:
        """On page switch: if global Analyze is ON, run it on the new page.

        If global Analyze is OFF, do nothing (the new page's canvas keeps its
        own cleared state). This keeps the shared button's checked state stable
        across tab switches — it reflects the GLOBAL intent, not per-canvas state.
        """
        if not getattr(self, "_analyze_global_on", False):
            return
        if hasattr(self, '_jobs') and not self._page_is_current():
            return
        page = self.stackedWidget.currentWidget()
        if page is None:
            return
        # If the new page's analysis is already on, leave it; otherwise turn on.
        canvas_attr = {
            "page_trend": "trendplot",
            "page_astm_sust": "canvasASTMSust",
            "page_astm_egg": "canvasASTMEgg",
            "page_gb": "canvasGB",
            "page_acczone": "canvasAccZone",
            "page_zreversal": "canvasZReversal",
            "page_xyreversal": "canvasXYReversal",
        }.get(page.objectName())
        if canvas_attr and not self._canvas_analysis_active(canvas_attr):
            handler = self._analyze_dispatch()
            if handler is not None:
                handler(True)

    # Map a canvas attribute name -> the specific analysis-state flag to read.
    # NB canvasASTMSust and canvasASTMEgg are the SAME ASTMChartView class, which
    # carries BOTH _sust_analysis_active and _egg_analysis_active — so the flag
    # must be chosen by WHICH canvas (page), not by trying flags in priority order.
    _CANVAS_ANALYSIS_FLAG = {
        "canvasASTMSust": "_sust_analysis_active",
        "canvasASTMEgg": "_egg_analysis_active",
        "canvasGB": "_comb_analysis_active",
        "canvasAccZone": "_analysis_active",
        "canvasZReversal": "_analysis_active",
        "canvasXYReversal": "_analysis_active",
        "trendplot": "_analysis_active",
    }

    def _canvas_analysis_active(self, canvas_attr: str) -> bool:
        """Whether analysis is currently shown on a given canvas.

        Replaces the old per-page ``_*_analyze_btn.isChecked()`` reads now that
        a single shared Analyze button drives whichever page is visible. Reads
        the canvas's own analysis-state flag (each chart module tracks it).
        """
        canvas = getattr(self, canvas_attr, None)
        if canvas is None:
            return False
        flag = self._CANVAS_ANALYSIS_FLAG.get(canvas_attr, "_analysis_active")
        return bool(getattr(canvas, flag, False))

    def _generate_astm_egg(self) -> None:
        """Generate from the egg-page controls: the mirrors ARE already synced
        to the Sustained originals, so just re-run the shared both-canvases plot."""
        self._generate_astm_both()
        # The chart's addplotxyz auto re-runs analysis if active, but it can't
        # reach the toolbar label — refresh it here.
        self._refresh_egg_summary()

    def _on_prone_toggled(self, _checked: bool) -> None:
        self._queue_limit_refresh()

    def _plot_error(self, context: str, error: Exception) -> None:
        message = f"{context}: {error}"
        errors = getattr(self, '_generation_errors', None)
        if errors is not None:
            errors.append(message)
        self.statusBar().showMessage(message, 8000)

    def _generate_active_page(self) -> None:
        """Prepare all numeric results off-thread, then draw the visible view."""
        self.request_chart_update(force=True)

    def _toggle_egg_analysis(self, checked: bool) -> None:
        """Toggle egg analysis + refresh the toolbar summary readout."""
        self.canvasASTMEgg.toggle_egg_analysis(checked)
        self._refresh_egg_summary()

    def _toggle_sust_analysis(self, checked: bool) -> None:
        """Toggle the ASTM Sustained limit check + refresh the toolbar summary.

        Passes the Sustained page's restraint/condition/height/prone to the
        canvas (height drives the limit scaling; prone overrides the ax- limit;
        restraint/cond don't affect this base check but are kept for API
        symmetry with the egg toggle).
        """
        self.canvasASTMSust.toggle_sust_analysis(
            checked,
            self.select_restraint.currentText(),
            self.select_cond.currentText(),
            self.height_input.value(),
            self._prone_btn.isChecked(),
        )
        self._refresh_sust_summary()

    def _refresh_sust_summary(self) -> None:
        """Rebuild the Sustained-page coloured "Fail X" labels from failing axes.

        Each failing axis/side (e.g. "ax (+)") becomes a red "● Fail ax (+)"
        label. When analysis is on but nothing fails, a green "✓ No issues
        found" label is shown instead. All labels are removed when analysis off.
        """
        host = getattr(self, "_sust_summary_host", None)
        if host is None:
            return
        checked = self._canvas_analysis_active("canvasASTMSust")
        specs = self._sust_fail_specs() if checked else []
        self._rebuild_summary_labels(host, specs, checked)

    def _sust_fail_specs(self) -> list[tuple[str, str]]:
        """Return (text, colour) per failing Sustained axis/side — all red."""
        out = []
        for r in sorted(getattr(self.canvasASTMSust, "sust_failed_rules", set())):
            out.append((f"\u25cf Fail {r}", "#E8112D"))   # red circle
        return out

    def _refresh_egg_summary(self) -> None:
        """Rebuild the egg-page coloured "Fail X" labels from failed rules.

        Each broken rule becomes its own QLabel, coloured to match the marker
        its points get (red circle = ASTM 0.2s, black circle = restraint 0.2s,
        black triangle = restraint 0s). When analysis is on but nothing fails, a
        green "✓ No issues found" label is shown instead.
        """
        host = getattr(self, "_egg_summary_host", None)
        if host is None:
            return
        checked = self._canvas_analysis_active("canvasASTMEgg")
        specs = self._egg_fail_specs() if checked else []
        self._rebuild_summary_labels(host, specs, checked)

    def _egg_fail_specs(self) -> list[tuple[str, str]]:
        """Return (text, colour) per failed egg rule, coloured to match markers."""
        out = []
        for r in sorted(getattr(self.canvasASTMEgg, "failed_rules", set())):
            low = r.lower()
            txt = r.replace("(0.2s)", "0.2s").replace("(0s)", "0s").strip()
            if low.startswith("astm"):
                out.append((f"● Fail {txt}", "#E8112D"))      # red circle
            elif low.endswith("(0.2s)") or low.endswith("(0.2 s)"):
                out.append((f"● Fail {txt}", "#111111"))      # black circle
            else:
                out.append((f"▲ Fail {txt}", "#111111"))      # black triangle
        return out

    def _rebuild_summary_labels(self, host: QWidget,
                                specs: list[tuple[str, str]],
                                checked: bool = False) -> None:
        """Rebuild *host*'s layout into one coloured QLabel per (text, colour).

        Clears previous labels first. When *checked* is True (analysis active)
        and *specs* is empty, shows a green "✓ No issues found" label instead of
        leaving the host blank — so the user gets a clear positive confirmation
        rather than an empty bar. When *checked* is False (analysis off) the host
        is left empty.
        """
        lay = host.layout()
        # Remove + delete old labels.
        while lay.count():
            it = lay.takeAt(0)
            w = it.widget()
            if w is not None:
                w.deleteLater()
        if not specs and checked:
            # Analysis is on but nothing failed — positive confirmation.
            lbl = QLabel("\u2713 No issues found", host)
            lbl.setStyleSheet(
                "QLabel { color: #107C10; font-weight: 600; padding: 0 6px; }")
            lay.addWidget(lbl)
            return
        for text, colour in specs:
            lbl = QLabel(text, host)
            lbl.setStyleSheet(
                f"QLabel {{ color: {colour}; font-weight: 600;"
                f" padding: 0 6px; }}")
            lay.addWidget(lbl)

    def _generate_astm_both(self) -> None:
        """Re-generate both ASTM canvases from the current controls.

        Each tab owns its own data combo (``select_plot_tab2`` for Sustained,
        ``select_plot_tab2b`` for Eggplot). They are kept in sync, so either
        reads the same value — but each canvas reads its own combo so the
        behaviour stays correct even if the sync wiring changes.
        """
        self.plotASTM(canvas=self.canvasASTMSust, name=self.select_plot_tab2)
        self.plotASTM(canvas=self.canvasASTMEgg, name=self.select_plot_tab2b)
        # addplotxyz re-runs an active analysis on the canvas, but can't reach
        # the toolbar summary labels — refresh them here.
        if self._canvas_analysis_active("canvasASTMSust"):
            self._refresh_sust_summary()
        if self._canvas_analysis_active("canvasASTMEgg"):
            self._refresh_egg_summary()

    # ------------------------------------------------------------------
    # Page 5: GB 8408
    # ------------------------------------------------------------------

    def _setup_gb_page(self) -> None:
        # Generate/Update + Analyze live in the shared global controls.
        tb2 = self._mk_toolbar("GB Chart Tools")
        _add_chart_toolbar(tb2, self.canvasGB)
        tb2.addSeparator()
        # Analysis summary: a row of coloured "Fail X" labels (one per broken
        # rule, coloured to match its marker). Cleared when all pass / analyze
        # is off.
        self._gb_summary_host = QWidget(tb2)
        QHBoxLayout(self._gb_summary_host).setContentsMargins(0, 0, 0, 0)
        tb2.addWidget(self._gb_summary_host)

        page = self._make_page("page_gb", [tb2], self.canvasGB)
        self._add_chart_page(page, 'intersect', "GB Contour")

    def _toggle_gb_analysis(self, checked: bool) -> None:
        """Toggle GB combined-chart analysis + refresh the toolbar summary."""
        self.canvasGB.toggle_comb_analysis(checked)
        self._refresh_gb_summary()

    def _generate_gb(self) -> None:
        """Re-plot the GB charts + refresh the analysis summary (the chart
        auto re-runs analysis if active; the toolbar label needs a nudge)."""
        self.plotGB(canvas=self.canvasGB, combobox=self.select_plot_tab3)
        self._refresh_gb_summary()

    def _generate_acczone(self) -> None:
        """Re-plot the Acc Zone chart with FORCED 1 Hz standard-filtered data.

        Acc Zone analysis requires 1 Hz data regardless of the global cutoff, so
        this ignores self.plottype / self.cutoff and always uses the 1 Hz frame.
        The 1 Hz frame is built non-mutatingly (get_1hz_standard_frame) so it
        does NOT clobber the shared std_data/filtered_data frames that the
        Trend/GB/ASTM pages read.
        """
        try:
            name = self.select_plot_tab4.currentText()
            data = self.datadict.get(name)
            if data is None:
                return
            color = self.dataset_colors.get(name)
            if hasattr(data, "calculate_sampling_frequency"):
                data.calculate_sampling_frequency()
            df = data.get_1hz_standard_frame()
            self.canvasAccZone.addplotxyz(data=df, color=color)
            if self._canvas_analysis_active("canvasAccZone"):
                self.canvasAccZone.toggle_analysis(True)
        except Exception as e:
            self._plot_error("AccZone plot", e)

    def _refresh_gb_summary(self) -> None:
        """Rebuild the GB-page coloured "Fail X" labels from failed rules.

        Each broken rule becomes its own QLabel coloured to match its marker
        (red 0.05s / orange 0.1s / yellow 0.2s). When analysis is on but nothing
        fails, a green "✓ No issues found" label is shown instead.
        """
        host = getattr(self, "_gb_summary_host", None)
        if host is None:
            return
        checked = self._canvas_analysis_active("canvasGB")
        specs = self._gb_fail_specs() if checked else []
        self._rebuild_summary_labels(host, specs, checked)

    def _gb_fail_specs(self) -> list[tuple[str, str]]:
        """Return (text, colour) per failed GB rule, coloured to match markers."""
        out = []
        for r in sorted(getattr(self.canvasGB, "failed_rules", set())):
            txt = r.replace("(0.2s)", "0.2s").replace("(0.1s)", "0.1s") \
                   .replace("(0.05s)", "0.05s").replace("(0s)", "0s").strip()
            low = r.lower().replace(" ", "")
            if "0.05s" in low:
                colour = "#FF0000"     # red
            elif "0.1s" in low:
                colour = "#E8A317"     # orange-ish (readable on light/dark)
            elif "0.2s" in low:
                colour = "#B8B800"     # yellow (dark enough to read)
            else:
                colour = "#888888"
            out.append((f"● Fail {txt}", colour))
        return out

    # ------------------------------------------------------------------
    # Page 6: Acceleration Zone
    # ------------------------------------------------------------------

    def _setup_acczone_page(self) -> None:
        tb = self._mk_toolbar("AccZone Controls")

        # Acc Zone analysis REQUIRES 1 Hz data. Filtering is forced to 1 Hz by
        # _generate_acczone (independent of the global cutoff), so the readout is
        # a STATIC label — it must NOT mirror self.cutoff (the global value is
        # irrelevant here and would be misleading). Generate/Update + Analyze
        # live in the shared global controls (top bar + control panel).
        self.acczone_cutoff_label = QLabel("  Fixed: 1 Hz  ")
        self.acczone_cutoff_label.setFixedHeight(22)
        tb.addWidget(self.acczone_cutoff_label)

        tb2 = self._mk_toolbar("AccZone Chart Tools")
        _add_chart_toolbar(tb2, self.canvasAccZone)

        # Info hint: 1 Hz filtering is applied automatically on this page. A
        # soft amber label so it reads as guidance, not a status readout.
        tb2.addSeparator()
        self._acczone_hint = QLabel(
            "ℹ Acc Zone uses 1 Hz filtered data automatically.")
        self._acczone_hint.setStyleSheet(
            "QLabel { color: #B8860B; font-weight: 600; padding: 0 6px; }")
        tb2.addWidget(self._acczone_hint)

        page = self._make_page("page_acczone", [tb, tb2], self.canvasAccZone)
        self._add_chart_page(page, 'squares-four', "Acc Zone")
        # Seed the readout now that the widget exists.
        self._update_status_label()

    # ------------------------------------------------------------------
    # Pages 7 & 8: Acceleration reversal analysis (Z, X/Y)
    # Ported from the archived matplotlib app; rebuilt on Qt Charts. Each tab
    # owns its own plottype selector (independent of the Trend tab's global
    # plottype) and an Analyze toggle that draws the detected reversal-window
    # bands + min/max markers, mirroring how the ASTM/GB/AccZone tabs work.
    # ------------------------------------------------------------------

    @staticmethod
    def _data_for_plottype(data, plottype: str):
        """Return the AccData frame for *plottype* ("Raw"/"Filter"/"Standard").

        Reversal tabs have their own plottype selector (independent of the
        Trend tab's global one), so they need their own frame lookup instead
        of using self._select_data.
        """
        src = {"Raw": "data", "Filter": "filtered_data",
               "Standard": "std_data"}
        return getattr(data, src.get(plottype, "std_data"))

    def _setup_zreversal_page(self) -> None:
        """Z-axis reversal analysis page.

        Row 1: data + plot-type + cutoff + Generate (analysis-relevant controls).
        Row 2: chart tools (zoom/pan/reset) + Analyze + summary.
        Kept to these two rows so nothing overflows off-screen at normal window
        widths — Analyze lives on the same row as the chart tools, where every
        other tab's Analyze button also sits.
        """
        tb = self._mk_toolbar("Z Reversal Controls")

        tb.addWidget(QLabel(" Type: "))
        self._zr_plottype = QComboBox()
        self._zr_plottype.addItems(["Raw", "Filter", "Standard"])
        self._zr_plottype.setCurrentText("Standard")
        self._zr_plottype.setMinimumWidth(110)
        tb.addWidget(self._zr_plottype)

        # Generate/Update + Analyze live in the shared global controls.
        tb2 = self._mk_toolbar("Z Reversal Chart Tools")
        _add_chart_toolbar(tb2, self.canvasZReversal)
        tb2.addSeparator()

        # The Analyze toggle lives in the shared control panel (routes here).
        # Summary: "● N reversal(s)" when Analyze is on and events exist; empty
        # otherwise (reuses the ASTM/GB coloured-label host machinery).
        self._zr_summary_host = QWidget(tb2)
        QHBoxLayout(self._zr_summary_host).setContentsMargins(0, 0, 0, 0)
        tb2.addWidget(self._zr_summary_host)

        page = self._make_page("page_zreversal", [tb, tb2], self.canvasZReversal)
        self._add_chart_page(page, 'arrows-down-up', "Z Reversal")

    def _setup_xyreversal_page(self) -> None:
        """X/Y-axis reversal analysis page (two stacked charts).

        Three-row toolbar layout:
          Row 1: data + plot-type + hysteresis + Generate (data + detection).
          Row 2: Set Cutoff + cutoff readout + Restraint/Condition/Height (the
                 allowance inputs).
          Row 3: Zoom / Pan / Reset + Analyze + summary label (chart tools).
        Splitting into three rows keeps each row narrow enough that nothing
        overflows off-screen at normal window widths.
        """
        # ---- Row 1: data selection + detection controls + Generate ----
        tb = self._mk_toolbar("X/Y Reversal Controls")
        tb.addWidget(QLabel(" Type: "))
        self._xr_plottype = QComboBox()
        self._xr_plottype.addItems(["Raw", "Filter", "Standard"])
        self._xr_plottype.setCurrentText("Standard")
        self._xr_plottype.setMinimumWidth(110)
        tb.addWidget(self._xr_plottype)

        tb.addSeparator()
        # Hysteresis dead-band for the X/Y sign-change detection. A sample only
        # commits to ±sign once |value| exceeds this, so a tiny noise/filter
        # glitch poking across zero no longer creates a spurious 2nd sign
        # change — a macro single-crossing with a small blip still counts as ONE
        # reversal. 0 disables hysteresis (plain-sign behaviour).
        tb.addWidget(QLabel(" Hysteresis(g): "))
        self._xr_hyst = QDoubleSpinBox()
        self._xr_hyst.setRange(0.0, 2.0)
        self._xr_hyst.setDecimals(3)
        self._xr_hyst.setSingleStep(0.01)
        self._xr_hyst.setValue(0.05)
        self._xr_hyst.setMinimumWidth(80)
        self._xr_hyst.setToolTip(
            "Hysteresis dead-band (g) for the sign-change detection: a sample "
            "only flips the sign once |value| exceeds this, so a tiny glitch "
            "across zero is ignored as noise")
        tb.addWidget(self._xr_hyst)

        # Generate/Update + Analyze live in the shared global controls.
        # Restraint / Condition / Height live in the shared control panel; the
        # cutoff lives in the persistent top bar. _generate_xyreversal reads the
        # shared instances and applies self.cutoff before plotting.

        # ---- Row 2: chart tools + summary ----
        tb2 = self._mk_toolbar("X/Y Reversal Chart Tools")
        _add_chart_toolbar(tb2, self.canvasXYReversal)
        tb2.addSeparator()

        # The Analyze toggle lives in the shared control panel (routes here).
        self._xr_summary_host = QWidget(tb2)
        QHBoxLayout(self._xr_summary_host).setContentsMargins(0, 0, 0, 0)
        tb2.addWidget(self._xr_summary_host)

        page = self._make_page("page_xyreversal", [tb, tb2],
                               self.canvasXYReversal)
        self._add_chart_page(page, 'arrows-left-right', "X/Y Reversal")

    # -- generate / analyze handlers ------------------------------------

    def _generate_zreversal(self) -> None:
        """Re-plot the Z-reversal chart from the current data + tab plottype."""
        try:
            name = self.select_plot_tab5.currentText()
            data = self.datadict.get(name)
            if data is None:
                return
            color = self.dataset_colors.get(name)
            if hasattr(data, "calculate_sampling_frequency"):
                data.calculate_sampling_frequency()
            # Apply the current cutoff before plotting so the curve reflects the
            # Set Cutoff Freq control. filter_data is cached, so when the cutoff
            # is unchanged this is a no-op (free).
            data.filter_data(cutoff=self.cutoff)
            fs = getattr(data, "fs", 500)
            df = self._data_for_plottype(data, self._zr_plottype.currentText())
            self.canvasZReversal.plot_data(df, color, fs)
            # If Analyze is on, re-run it against the fresh data + refresh label.
            if self._canvas_analysis_active("canvasZReversal"):
                self.canvasZReversal.toggle_analysis(True)
            self._refresh_zreversal_summary()
        except Exception as e:
            self._plot_error("Z reversal plot", e)

    def _generate_xyreversal(self) -> None:
        """Re-plot the X/Y-reversal charts from the current data + tab plottype."""
        try:
            name = self.select_plot_tab6.currentText()
            data = self.datadict.get(name)
            if data is None:
                return
            color = self.dataset_colors.get(name)
            if hasattr(data, "calculate_sampling_frequency"):
                data.calculate_sampling_frequency()
            # Apply the current cutoff before plotting (cached when unchanged).
            data.filter_data(cutoff=self.cutoff)
            fs = getattr(data, "fs", 500)
            df = self._data_for_plottype(data, self._xr_plottype.currentText())

            # Compute the ASTM Sustained t=0 allowance for the X and Y axes from
            # this tab's restraint / condition / height. Store the FULL (1×)
            # allowance: the chart draws both 1×limit and 0.5×limit reference
            # lines, and each reversal pair's own peak-to-peak Δt picks its
            # effective multiplier (>0.2s → 1×, else 0.5×) at check time.
            # Reads the shared restraint/condition/height (control panel).
            restraint = self.select_restraint.currentText()
            condition = self.select_cond.currentText()
            height = int(self.height_input.value())
            allowance = {
                "x": allowance_at_t0("x", restraint, condition, height),
                "y": allowance_at_t0("y", restraint, condition, height),
            }
            self.canvasXYReversal.set_allowance(allowance)
            # Hysteresis dead-band for the sign computation (tunable via the
            # toolbar spinbox): absorbs tiny noise/filter glitches near zero so
            # they don't create spurious extra sustained runs.
            self.canvasXYReversal.set_deadband(float(self._xr_hyst.value()))

            self.canvasXYReversal.plot_data(df, color, fs)
            if self._canvas_analysis_active("canvasXYReversal"):
                self.canvasXYReversal.toggle_analysis(True)
            self._refresh_xyreversal_summary()
        except Exception as e:
            self._plot_error("X/Y reversal plot", e)

    def _toggle_zreversal_analysis(self, checked: bool) -> None:
        """Toggle Z-reversal overlays + refresh the toolbar summary."""
        self.canvasZReversal.toggle_analysis(checked)
        self._refresh_zreversal_summary()

    def _toggle_xyreversal_analysis(self, checked: bool) -> None:
        """Toggle X/Y-reversal overlays + refresh the toolbar summary."""
        self.canvasXYReversal.toggle_analysis(checked)
        self._refresh_xyreversal_summary()

    def _reversal_summary_specs(self, canvas) -> list[tuple[str, str]]:
        """Return the reversal summary labels for one canvas.

        Two modes, depending on the canvas:
        • X/Y canvas (has an allowance → a real pass/fail check): mirrors
          ASTM/GB — a red "● Fail M" when events exceed 0.5×allowance, else an
          empty list (the caller then shows the green "✓ No issues found").
        • Z canvas (no allowance → no pass/fail concept): always shows an amber
          "● N reversal(s)" event count so the user still sees how many were
          detected.
        Empty (no labels) when Analyze is off or no events were detected.
        """
        n = getattr(canvas, "event_count", 0)
        if n <= 0:
            return []
        # X/Y canvas: allowance-based — show fails, or nothing (→ green ✓).
        if getattr(canvas, "_allowance", None) is not None:
            fails = getattr(canvas, "fail_count", 0)
            if fails > 0:
                return [(f"● Fail {fails}", "#C00000")]  # red
            return []  # no fails → caller shows green "✓ No issues found"
        # Z canvas: no pass/fail, just the event count.
        return [(f"● {n} reversal(s)", "#CA5010")]  # amber, readable light/dark

    def _refresh_zreversal_summary(self) -> None:
        host = getattr(self, "_zr_summary_host", None)
        if host is None:
            return
        checked = self._canvas_analysis_active("canvasZReversal")
        specs = (self._reversal_summary_specs(self.canvasZReversal)
                 if checked else [])
        self._rebuild_summary_labels(host, specs, checked)

    def _refresh_xyreversal_summary(self) -> None:
        host = getattr(self, "_xr_summary_host", None)
        if host is None:
            return
        checked = self._canvas_analysis_active("canvasXYReversal")
        specs = (self._reversal_summary_specs(self.canvasXYReversal)
                 if checked else [])
        self._rebuild_summary_labels(host, specs, checked)

    # ==================================================================
    # State management
    # ==================================================================

    def set_cutoff(self, cf: int) -> None:
        self.cutoff = cf
        # Keep the status readouts (global + AccZone label) in sync regardless
        # of which code path sets the cutoff.
        self._update_status_label()

    def _update_status_label(self) -> None:
        self.status_label.setText(
            f"  Plot: {self.plottype}, Cutoff: {self.cutoff} Hz  "
        )
        # Keep the top-bar Cutoff SpinBox showing the effective value (its
        # valueChanged handler is NOT re-triggered here — signals are blocked
        # so the live-replot only fires when the USER turns the knob, not when
        # set_cutoff is called from another path).
        spin = getattr(self, "cutoff_spin", None)
        if spin is not None and spin.value() != self.cutoff:
            spin.blockSignals(True)
            spin.setValue(self.cutoff)
            spin.blockSignals(False)

    def update_status_label(self) -> None:
        self._update_status_label()

    # ==================================================================
    # Data management
    # ==================================================================

    def set_datalist(self, data) -> None:
        # Ensure a unique display name: the colour/visibility maps and the
        # datadict are all keyed by filename, so a duplicate would silently
        # collide (shared colour, lost datadict entry, ambiguous combo box).
        # Append (2), (3), … to duplicates instead.
        data.filename = self._unique_filename(data.filename)
        self.datalist.append(data)
        self.data_namelist.append(data.filename)
        self.datadict = dict(zip(self.data_namelist, self.datalist))
        # Auto-assign a colour if this dataset doesn't have one yet.
        if data.filename not in self.dataset_colors:
            self.dataset_colors[data.filename] = self._auto_color_for(data.filename).lower()
        self._repopulate_data_selectors()
        # Keep the trend-tab shift selector in sync with the dataset list.
        self._refresh_shift_combo()
        self._update_workspace_state()
        self._queue_chart_refresh()

    def _repopulate_data_selectors(self) -> None:
        """Rebuild every data selector (top bar + hidden per-page mirrors).

        The top-bar ``select_data`` is the single source of truth; the six
        per-page combos are hidden mirrors of it. All are repopulated here
        from ``datadict`` and aligned to the same current item. ``select_data``'s
        fan-out sync is blocked during this bulk update so it doesn't re-emit
        per-combo.
        """
        keys = list(self.datadict.keys())
        previous = self.select_data.currentText()
        # Repopulate the top-bar selector (source of truth) without triggering
        # its fan-out mid-rebuild.
        sd = getattr(self, "select_data", None)
        if sd is not None:
            sd.blockSignals(True)
            sd.clear()
            sd.addItems(keys)
        for cb in self._data_combos:
            cb.blockSignals(True)
            cb.clear()
            cb.addItems(keys)
            cb.blockSignals(False)
        # Pick a sensible current item: keep the previous selection if still
        # present, else the first (or empty).
        cur = previous
        if cur not in keys:
            cur = keys[0] if keys else ""
        for cb in self._data_combos:
            cb.blockSignals(True)
            cb.setCurrentText(cur)
            cb.blockSignals(False)
        if sd is not None:
            sd.setCurrentText(cur)
            sd.blockSignals(False)
        self.sidebar.select_filename(cur)

    def _unique_filename(self, filename: str) -> str:
        """Return *filename* made unique against already-loaded datasets.

        Splits name/ext so the suffix lands before the extension, e.g.
        ``data.sup`` → ``data (2).sup``.
        """
        if filename not in self.data_namelist:
            return filename
        import os as _os
        stem, ext = _os.path.splitext(filename)
        n = 2
        while f"{stem} ({n}){ext}" in self.data_namelist:
            n += 1
        return f"{stem} ({n}){ext}"

    def reduce_data(self, plot_order: int) -> None:
        removed_name = self.data_namelist[plot_order - 1] if plot_order - 1 < len(self.data_namelist) else None
        self.datalist.pop(plot_order - 1)
        self.data_namelist.pop(plot_order - 1)
        self.datadict = dict(zip(self.data_namelist, self.datalist))
        if removed_name:
            self.dataset_colors.pop(removed_name, None)
            self.dataset_visibility.pop(removed_name, None)
        # Keep the trend-tab shift selector in sync with the dataset list.
        self._refresh_shift_combo()

    def _auto_color_for(self, filename: str) -> str:
        """Pick the first palette colour not already used by another dataset."""
        # Normalise to lowercase for comparison: set_dataset_color() stores
        # picked colours lowercased, while DATASET_COLORS is uppercase. Without
        # this, the uppercase palette entries never match the lowercase used-set
        # and every dataset collapses to the first palette colour.
        used = {c.lower() for c in self.dataset_colors.values()}
        for c in DATASET_COLORS:
            if c.lower() not in used:
                return c
        # All palette colours taken — cycle by index.
        return DATASET_COLORS[len(self.dataset_colors) % len(DATASET_COLORS)]

    def _plot_order_of(self, filename: str) -> int:
        """1-based position of a dataset by filename, or -1 if absent."""
        try:
            return self.data_namelist.index(filename) + 1
        except ValueError:
            return -1

    def set_dataset_color(self, filename: str, color_hex: str) -> None:
        """Change one dataset's colour and recolour every chart it appears on.

        Called by the InfoCard colour swatch. Trend is recoloured in place
        (no recompute); analysis charts recolour their currently-plotted
        measured overlay only — so this is cheap and preserves zoom state.
        """
        colour = QColor(color_hex)
        if filename not in self.datadict or not colour.isValid():
            return
        color_hex = colour.name()
        self.dataset_colors[filename] = color_hex

        # 1) Trend — in-place pen swap, preserves data + zoom.
        po = self._plot_order_of(filename)
        if po > 0:
            self.trendplot.recolor_dataset(po, color_hex)

        # 2) Standard charts — recolour the measured overlay IF this dataset
        #    is the one currently shown on each tab. (Each tab plots exactly
        #    one dataset at a time.)
        if self.select_plot_tab2.currentText() == filename:
            self.canvasASTMSust.recolor_measured(color_hex)
            self.canvasASTMEgg.recolor_measured(color_hex)
        if self.select_plot_tab3.currentText() == filename:
            self.canvasGB.recolor_measured(color_hex)
        if self.select_plot_tab4.currentText() == filename:
            self.canvasAccZone.recolor_measured(color_hex)
        if self.select_plot_tab5.currentText() == filename:
            self.canvasZReversal.recolor_measured(color_hex)
        if self.select_plot_tab6.currentText() == filename:
            self.canvasXYReversal.recolor_measured(color_hex)

        # 3) Refresh the sidebar card's colour swatch.
        for card in getattr(self, "stat_content", []):
            if getattr(card, "_data", None) is not None and \
                    getattr(card._data, "filename", None) == filename:
                if hasattr(card, "apply_color"):
                    card.apply_color(color_hex)

    def set_dataset_visible(self, filename: str, visible: bool) -> None:
        """Show or hide one dataset's trend curves without deleting the data.

        Called by the InfoCard 'Plot' checkbox. Toggles visibility in place
        on the trend chart; the central ``dataset_visibility`` map is the
        source of truth and is honoured on every replot (so hidden datasets
        stay hidden after a switch-plot-type / fit-max / truncate).
        """
        self.dataset_visibility[filename] = visible
        po = self._plot_order_of(filename)
        if po > 0:
            self.trendplot.set_dataset_visible(po, visible)

    # ==================================================================
    # Sidebar management
    # ==================================================================

    def clear_sidebar(self) -> None:
        self.sidebar.clear_all()
        self.stat_content.clear()

    def _add_info_card(self, data, plottype: str, plot_order: int) -> None:
        card = InfoCard(
            data=data,
            plottype=plottype,
            plot_order=plot_order,
            remove_callback=lambda _, po=plot_order: self._remove_dataset(po),
            mainui=self,
        )
        self.sidebar.add_card(card)
        self.stat_content.append(card)

    # ==================================================================
    # Plotting
    # ==================================================================

    def add_trend_data(self, data, plottype: str, plot_order: int) -> None:
        color = self.dataset_colors.get(data.filename,
                                        DATASET_COLORS[(plot_order - 1) % len(DATASET_COLORS)])
        if plottype == "Raw":
            self.trendplot.addplotxyz(data.data, plot_order, color)
        elif plottype == "Standard":
            self.trendplot.addplotxyz(data.std_data, plot_order, color)
        elif plottype == "Filter":
            self.trendplot.addplotxyz(data.filtered_data, plot_order, color)
        self.trendplot.draw_idle()
        # Apply current visibility (defaults to visible when first loaded).
        visible = self.dataset_visibility.get(data.filename, True)
        self.trendplot.set_dataset_visible(plot_order, visible)
        self._add_info_card(data, plottype, plot_order)

    def replot_all(self) -> None:
        i = 1
        for d in self.datalist:
            d.filter_data(cutoff=self.cutoff)
            self.add_trend_data(data=d, plottype=self.plottype, plot_order=i)
            i += 1
        self._queue_chart_refresh()

    def replot_single(self, name: str) -> None:
        d = self.datadict[name]
        self.trendplot.initplotxyz()
        d.filter_data(cutoff=self.cutoff)
        self.add_trend_data(data=d, plottype=self.plottype, plot_order=1)

    # ==================================================================
    # Plot type switching
    # ==================================================================

    def switch_plot_type(self, plottype: str) -> None:
        self.plottype = plottype
        if plottype in getattr(self, '_plot_type_actions', {}):
            self._plot_type_actions[plottype].setChecked(True)
        self._update_status_label()
        # initplotxyz clears analysis overlays; reflect that in the button.
        if getattr(self, "_analyze_btn", None) is not None:
            self._analyze_btn.setChecked(False)

        if not self.datalist:
            return

        # Fast path: a plot-type switch only changes Y values, so update the
        # existing trend series + sidebar stat tables IN PLACE instead of
        # clearing + rebuilding everything (initplotxyz + replot_all + sidebar
        # rebuild was ~530ms for 4 datasets; the in-place path is ~50ms).
        # Both methods are no-ops/cheap when nothing has been plotted yet and
        # fall back gracefully, so this is safe on the first switch too.
        try:
            self.trendplot.update_plottype(self.datalist, plottype,
                                           self.dataset_colors)
        except Exception:
            # Fallback to the full rebuild if the in-place path hits a problem
            # (e.g. series count mismatch after a partial state).
            self.trendplot.initplotxyz()
            self.clear_sidebar()
            self.replot_all()
        # Refresh the sidebar stat tables for the new plot type in place
        # (no card rebuild — keeps the swatch/eye/title/buttons untouched).
        self.sidebar.update_plottype(plottype)
        self._queue_chart_refresh()

    def switch_to_fitmax(self, axis: int) -> None:
        """Align every dataset's MAX (peak) on the given axis.

        Requires >1 dataset. Dataset 0 is the reference (no shift); every
        other dataset is time-shifted so its peak lines up with dataset 0's.
        """
        if len(self.datalist) > 1:
            self.trendplot.initplotxyz()
            self.clear_sidebar()
            self._replot_fitextreme(axis, "max")

    def switch_to_fitmin(self, axis: int) -> None:
        """Align every dataset's MIN (trough) on the given axis.

        Mirror of switch_to_fitmax. Dataset 0 is the reference; every other
        dataset shifts so its trough lines up with dataset 0's.
        """
        if len(self.datalist) > 1:
            self.trendplot.initplotxyz()
            self.clear_sidebar()
            self._replot_fitextreme(axis, "min")

    def _replot_fitextreme(self, axis: int, which: str) -> None:
        """Re-plot all datasets with per-dataset time offsets so the chosen
        extremum (peak or trough) on *axis* lines up across datasets.

        ``which`` is "max" (idxmax) or "min" (idxmin). Offsets are computed
        in SECONDS via each dataset's own ``fs`` (the old code assumed a
        fixed 500 Hz, which broke for other sample rates).
        """
        arg = self._get_argextreme(axis, which)
        ref = arg[0]
        for i, d in enumerate(self.datalist, 1):
            offset_seconds = (arg[i - 1] - ref) / max(getattr(d, "fs", 500), 1)
            self._add_shifted_data(d, self.plottype, i, offset_seconds)

    def _add_shifted_data(self, data, plottype: str, plot_order: int,
                          offset_seconds) -> None:
        """Plot one dataset on the trend chart with a time-axis shift.

        Shared by Fit Max, Fit Min and the manual Shift control. The shift is
        display-only — the underlying AccData frames are not modified.
        """
        color = self.dataset_colors.get(data.filename,
                                        DATASET_COLORS[(plot_order - 1) % len(DATASET_COLORS)])
        if plottype == "Raw":
            self.trendplot.addplotxyz_shifted(data.data, plot_order, offset_seconds, color)
        elif plottype == "Standard":
            self.trendplot.addplotxyz_shifted(data.std_data, plot_order, offset_seconds, color)
        elif plottype == "Filter":
            self.trendplot.addplotxyz_shifted(data.filtered_data, plot_order, offset_seconds, color)
        self.trendplot.draw_idle()
        # Re-apply this dataset's visibility (mirrors add_trend_data). Without
        # this, a Shift / Fit Max / Fit Min replot would show every dataset —
        # including ones the user had hidden via the eye toggle — because
        # addplotxyz_shifted draws curves visible by default.
        visible = self.dataset_visibility.get(data.filename, True)
        self.trendplot.set_dataset_visible(plot_order, visible)
        self._add_info_card(data, plottype, plot_order)

    def _get_argextreme(self, axis: int, which: str) -> list:
        """Return the row index of the extremum on *axis* for each dataset.

        ``which`` is "max" → ``idxmax`` or "min" → ``idxmin``.
        """
        src = {"Raw": "data", "Standard": "std_data", "Filter": "filtered_data"}
        attr = src.get(self.plottype, "std_data")
        method = "idxmax" if which == "max" else "idxmin"
        return [getattr(d, attr).iloc[:, axis]
                .agg(method) for d in self.datalist]

    # ------------------------------------------------------------------
    # Manual time-shift (display-only)
    # ------------------------------------------------------------------

    def _apply_manual_shift(self) -> None:
        """Shift ONE selected dataset by the spinbox value (seconds).

        Only the chosen dataset moves; every other dataset is drawn with
        zero offset. Clear + replot so the chart reflects the new layout.
        """
        if not self.datalist:
            return
        target_po = self._shift_combo.currentData()
        dt = self._shift_spin.value()
        self.trendplot.initplotxyz()
        self.clear_sidebar()
        for i, d in enumerate(self.datalist, 1):
            offset = dt if i == target_po else 0.0
            self._add_shifted_data(d, self.plottype, i, offset)

    def _reset_manual_shift(self) -> None:
        """Undo any shift and replot normally (all offsets back to zero)."""
        if not self.datalist:
            return
        self._shift_spin.setValue(0.0)
        self.trendplot.initplotxyz()
        self.clear_sidebar()
        for i, d in enumerate(self.datalist, 1):
            self._add_shifted_data(d, self.plottype, i, 0.0)

    def _refresh_shift_combo(self) -> None:
        """Repopulate the shift dataset selector from the current datalist.

        Keeps the current selection when possible; called whenever datasets
        are added/removed."""
        if not hasattr(self, "_shift_combo"):
            return
        current = self._shift_combo.currentData()
        self._shift_combo.blockSignals(True)
        self._shift_combo.clear()
        for i, d in enumerate(self.datalist, 1):
            self._shift_combo.addItem(self._short_name(d.filename), i)
        # Restore selection if still present, else first item.
        idx = self._shift_combo.findData(current)
        if idx >= 0:
            self._shift_combo.setCurrentIndex(idx)
        elif self._shift_combo.count() > 0:
            self._shift_combo.setCurrentIndex(0)
        self._shift_combo.blockSignals(False)

    @staticmethod
    def _short_name(filename: str) -> str:
        """Trim long filenames for the compact shift combo."""
        name = filename or "dataset"
        return name if len(name) <= 30 else name[:27] + "…"

    # ==================================================================
    # GB / ASTM plotting
    # ==================================================================

    def _select_data(self, data_obj):
        """Return the correct data slice based on current plottype."""
        src = {"Raw": "data", "Standard": "std_data", "Filter": "filtered_data"}
        return getattr(data_obj, src.get(self.plottype, "std_data"))

    def plotGB(self, canvas, combobox) -> None:
        try:
            name = combobox.currentText()
            data = self.datadict[name]
            color = self.dataset_colors.get(name)
            canvas.addplotxyz(data=self._select_data(data), color=color)
        except Exception as e:
            self._plot_error("GB plot", e)

    def plotGB_checkbox(self, canvas, name: str) -> None:
        try:
            data = self.datadict[name]
            color = self.dataset_colors.get(name)
            canvas.addplotxyz(data=self._select_data(data), color=color)
        except Exception as e:
            print(f"GB checkbox plot error: {e}")

    def plotASTM(self, canvas, name) -> None:
        try:
            ds_name = name.currentText()
            data = self.datadict[ds_name]
            color = self.dataset_colors.get(ds_name)
            canvas.addplotxyz(
                data=self._select_data(data),
                restraint=self.select_restraint.currentText(),
                cond=self.select_cond.currentText(),
                input_height=self.height_input.value(),
                color=color,
                prone=self._prone_btn.isChecked(),
            )
        except Exception as e:
            self._plot_error("ASTM plot", e)

    def plotASTM_checkbox(self, canvas, name: str) -> None:
        try:
            data = self.datadict[name]
            color = self.dataset_colors.get(name)
            canvas.addplotxyz(
                data=self._select_data(data),
                restraint=self.select_restraint.currentText(),
                cond=self.select_cond.currentText(),
                input_height=self.height_input.value(),
                color=color,
                prone=self._prone_btn.isChecked(),
            )
        except Exception as e:
            print(f"ASTM checkbox plot error: {e}")

    # ==================================================================
    # Dataset operations
    # ==================================================================

    def _remove_dataset(self, plot_order: int) -> None:
        self.reduce_data(plot_order)  # also drops the colour from dataset_colors
        self.clear_sidebar()
        self.trendplot.initplotxyz()
        self.replot_all()
        self._repopulate_data_selectors()
        self._update_workspace_state()

    def reset_trend(self, reset_datalist: bool = False) -> None:
        self._reset_processing()
        try:
            if reset_datalist:
                self.datalist.clear()
                self.data_namelist.clear()
                self.datadict.clear()
                self.dataset_colors.clear()
                self.dataset_visibility.clear()
                self._refresh_shift_combo()
                self._repopulate_data_selectors()

            self.trendplot.initplotxyz()
            self.canvasASTMSust.initplotxyz()
            self.canvasASTMEgg.initplotxyz()
            self.canvasGB.initplotxyz()
            self.canvasAccZone.initplotxyz()
            self.canvasZReversal.initplotxyz()
            self.canvasXYReversal.initplotxyz()
            # Reset the shared Analyze button — initplotxyz clears each canvas's
            # overlays + internal _analysis_active flag, so the button must be
            # un-checked to stay in sync (a stuck checked state leaves stale
            # toolbar summary labels and a misleading button).
            btn = getattr(self, "_analyze_btn", None)
            if btn is not None:
                btn.blockSignals(True)
                btn.setChecked(False)
                btn.blockSignals(False)
            # Clear the global Analyze flag too (initplotxyz cleared each canvas).
            self._analyze_global_on = False
            # Reset the prone posture toggle too (return to the initial
            # non-prone state). Block signals so it doesn't fire mid-reset.
            btn = getattr(self, "_prone_btn", None)
            if btn is not None:
                btn.blockSignals(True)
                btn.setChecked(False)
                btn.blockSignals(False)
            # Clear the per-page toolbar summary labels too. initplotxyz wipes
            # the in-chart overlays, but the coloured "Fail X" / "✓ No issues
            # found" / "● N reversal(s)" QLabels live in the toolbar hosts and
            # would otherwise survive a Clear All.
            self._refresh_sust_summary()
            self._refresh_egg_summary()
            self._refresh_gb_summary()
            self._refresh_zreversal_summary()
            self._refresh_xyreversal_summary()
            self.clear_sidebar()
            self._update_workspace_state()
        except Exception as e:
            print(f"Reset error: {e}")

    # ==================================================================
    # File dialogs
    # ==================================================================

    def open_accel_data(self) -> None:
        filenames, _ = QFileDialog.getOpenFileNames(
            self, "Open Acceleration Data Files", self._last_open_dir,
            "Supported data (*.sup *.sups *.txt *.csv *.dat);;All Files (*)")
        if filenames:
            self._last_open_dir = os.path.dirname(filenames[0])
            self._start_import(filenames, raw=False)

    def open_raw_data(self) -> None:
        filenames, _ = QFileDialog.getOpenFileNames(
            self, "Open Raw Data Files", self._last_open_dir,
            "Supported data (*.sup *.sups *.txt *.csv *.dat);;All Files (*)")
        if filenames:
            self._last_open_dir = os.path.dirname(filenames[0])
            self._start_import(filenames, raw=True)

    def open_sensorconnect_data(self) -> None:
        """SensorConnect 软件导出的单传感器 CSV（Time + 三通道 X/Y/Z）。

        走与 raw/accel 相同的后台导入管线（raw='sensorconnect' 分支），
        加载器解析 FILE_INFO/CHANNEL_INFO 元数据与 DATA_START 数据块，
        并把 fs 校正为文件声明的采样率（如 512Hz）。
        """
        filenames, _ = QFileDialog.getOpenFileNames(
            self, "Open SensorConnect Data Files", self._last_open_dir,
            "SensorConnect export (*.csv);;All Files (*)")
        if filenames:
            self._last_open_dir = os.path.dirname(filenames[0])
            self._start_import(filenames, raw='sensorconnect')

    def _export_all_to_sup(self) -> None:
        """Export every loaded dataset to a .sup file in one go.

        Prompts for an output folder, then writes one .sup per dataset (using
        the existing ``export_to_file`` → ``reformat`` → ``to_csv`` pipeline,
        which produces the standard SARC header + tab-separated data layout).
        Each file is named after the dataset's stored filename (stem only),
        sanitized for the filesystem.
        """
        if not self.datalist:
            self.statusBar().showMessage("No datasets to export", 3000)
            return
        out_dir = QFileDialog.getExistingDirectory(
            self, "Select folder to export .sup files", self._last_open_dir)
        if not out_dir:
            return
        self._last_open_dir = out_dir

        def _safe_stem(name: str) -> str:
            # Strip the existing extension + sanitize filesystem-illegal chars.
            import re
            stem = os.path.splitext(name)[0]
            return re.sub(r'[<>:"/\\|?*]', '_', stem).strip() or "dataset"

        count = 0
        failed = []  # export_to_file 不再吞异常，这里收集失败文件名一并展示
        for d in self.datalist:
            stem = _safe_stem(d.filename)
            path = os.path.join(out_dir, stem + ".sup")
            try:
                d.export_to_file(path=path)
                count += 1
            except Exception as e:
                failed.append(d.filename)
                print(f"Export error ({d.filename}): {e}")
        if failed:
            self.statusBar().showMessage(
                f"Exported {count}/{len(self.datalist)}; failed: "
                f"{', '.join(failed)}", 8000)
        else:
            self.statusBar().showMessage(
                f"Exported {count}/{len(self.datalist)} dataset(s) to {out_dir}",
                5000)

    # ==================================================================
    # Dialogs
    # ==================================================================

    def _show_cutoff_dialog(self) -> None:
        try:
            win = CutoffFrequencyDialog(self)
            self._track_dialog(win)
            win.show()
        except Exception as e:
            print(f"Cutoff dialog error: {e}")

    def _show_report_dialog(self) -> None:
        try:
            win = ReportDialog(self)
            self._track_dialog(win)
            win.show()
        except Exception as e:
            print(f"Report dialog error: {e}")

    def _track_dialog(self, win) -> None:
        """Hold a reference so the dialog isn't garbage-collected before show.

        Bug fix: the original ``CutoffFrequencyDialog(self).show()`` and
        ``ReportDialog(self).show()`` created dialogs with no ref holder,
        so CPython could collect them immediately and they never appeared.
        """
        if not hasattr(self, "_active_dialogs"):
            self._active_dialogs = []
        self._active_dialogs.append(win)
        if hasattr(win, "destroyed"):
            win.destroyed.connect(lambda *_: self._discard_dialog(win))
        if hasattr(win, "finished"):
            win.finished.connect(lambda *_: self._discard_dialog(win))

    def _discard_dialog(self, win) -> None:
        try:
            self._active_dialogs.remove(win)
        except (ValueError, AttributeError):
            pass

    def switch_annot_mode(self) -> None:
        self.annot_mode = "Multiple" if self.annot_mode == "Single" else "Single"
        # Standard charts (QChartView subclasses) read annot_mode directly.
        for canvas in (self.canvasASTMSust, self.canvasASTMEgg,
                       self.canvasGB, self.canvasAccZone):
            canvas.annot_mode = self.annot_mode
        # Trend + reversal views are QWidget containers of sub-charts —
        # propagate explicitly so each sub-chart honours Single vs Multiple.
        self.trendplot.set_annotation_mode(self.annot_mode)
        self.canvasZReversal.set_annotation_mode(self.annot_mode)
        self.canvasXYReversal.set_annotation_mode(self.annot_mode)
        self.annot_label.setText(f"Annotations: {self.annot_mode}")
