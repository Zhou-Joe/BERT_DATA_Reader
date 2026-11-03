# coding: utf-8
"""
BERT Reader V3.1 — Main Window (PySide6 + Qt Charts).

Modernised reconstruction of the original PyQt5 + matplotlib application.
"""

from __future__ import annotations

import os
import sys

import numpy as np

from PySide6.QtCharts import QChartView
from PySide6.QtCore import Qt, QTimer, QSize
from PySide6.QtGui import QIcon, QColor, QCursor, QAction, QActionGroup
from PySide6.QtWidgets import (
    QToolBar, QWidget, QFileDialog, QStatusBar,
    QLabel, QFrame, QToolButton,
    QVBoxLayout, QHBoxLayout, QGridLayout, QSizePolicy,
    QApplication, QScrollArea, QMenu,
)

import qfluentwidgets as qfw
from qfluentwidgets import (
    FluentWindow,
    Action, ComboBox, SpinBox, DoubleSpinBox, PushButton, PrimaryPushButton,
    ToggleToolButton, TransparentToolButton, FluentIcon as FI, InfoBadge,
)

import accdata
from sidebar import DataSidebar, InfoCard
from dialogs import CutoffFrequencyDialog, ReportDialog
from chart_trend import TrendChartView
from chart_gb import GBChartView
from chart_acczone import AccZoneChartView
from chart_astm import ASTMChartView
from chart_reversal import ZReversalChartView, XYReversalChartView
from sci_calculation import allowance_at_t0
from nav_icons import NavIcon
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

    Uses qfluentwidgets ToggleToolButton + FluentIcon so the buttons match the
    rest of the Fluent UI instead of emoji. Interaction logic is unchanged:
    zoom/pan are mutually-exclusive toggles, and un-checking returns the chart
    to default mode.
    """
    zoom_btn = ToggleToolButton(toolbar)
    # ToggleToolButton pins IconOnly internally, so icon+text overlap. Use
    # text-only (no icon) for a clean readable label.
    zoom_btn.setText("Zoom")
    zoom_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
    zoom_btn.setCheckable(True)
    zoom_btn.setStatusTip("Enable zoom by dragging (rubber band)")

    pan_btn = ToggleToolButton(toolbar)
    pan_btn.setText("Pan")
    pan_btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
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

    toolbar.addWidget(zoom_btn)
    toolbar.addWidget(pan_btn)

    def _reset() -> None:
        zoom_btn.blockSignals(True)
        pan_btn.blockSignals(True)
        zoom_btn.setChecked(False)
        pan_btn.setChecked(False)
        zoom_btn.blockSignals(False)
        pan_btn.blockSignals(False)
        chart_widget.zoom_mode("default")
        chart_widget.reset_view()

    a_reset = Action(FI.SYNC, "Reset View", toolbar)
    a_reset.setStatusTip("Reset chart to fit all data")
    a_reset.triggered.connect(_reset)
    toolbar.addAction(a_reset)


class BertFluentWindow(FluentWindow):
    """FluentWindow + a full-width QStatusBar at the bottom.

    FluentWindow is a QWidget (not QMainWindow), so it has no status bar.
    This subclass rebuilds the top-level layout into a QVBoxLayout so a
    QStatusBar can sit beneath the whole (navigation + stacked pages) area.
    Also exposes ``addTitleBarWidget`` to drop controls (e.g. the theme
    toggle) into the frameless title bar.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        # Disable the Mica/acrylic effect so the window paints its own solid
        # Fluent background (light #f0f4f9 / dark #202020). With Mica enabled
        # the window background is transparent and relies on the OS compositor
        # to paint Mica behind it — which silently fails on Win10 / offscreen /
        # some setups, leaving the window black or unchanging on theme switch.
        # Disabling it guarantees a correct themed background everywhere.
        self.setMicaEffectEnabled(False)
        self._install_status_bar()
        self._install_top_bar()
        self._install_control_panel()

    def _install_status_bar(self) -> None:
        """Add a QStatusBar across the bottom of the window.

        FluentWindowBase installs ``QHBoxLayout(self)`` as the widget layout
        and you CANNOT replace an already-installed widget layout (the earlier
        takeAt + QVBoxLayout(self) attempt silently broke page visibility).
        Instead we keep FluentWindow's layout untouched and place the status
        bar as a child widget, manually positioned in resizeEvent. The bottom
        margin of ``widgetLayout`` is bumped so content never sits under it.
        """
        self._status_bar = QStatusBar(self)
        self._status_bar.setSizeGripEnabled(False)
        # Reserve room at the bottom of the content column for the status bar.
        sb_h = self._status_bar.sizeHint().height()
        # widgetLayout holds the stacked pages; widen its bottom margin.
        try:
            lm, tm, rm, bm = self.widgetLayout.getContentsMargins()
            self.widgetLayout.setContentsMargins(lm, tm, rm, sb_h)
        except Exception:
            pass
        self._position_status_bar()

    def _position_status_bar(self) -> None:
        """Place the status bar flush with the bottom edge, full width."""
        sb = getattr(self, "_status_bar", None)
        if sb is None:
            return
        sb_h = sb.sizeHint().height()
        sb.setGeometry(0, self.height() - sb_h, self.width(), sb_h)
        sb.raise_()

    def resizeEvent(self, e) -> None:
        """Keep the manually-positioned bars pinned (top + bottom + ctrl)."""
        super().resizeEvent(e)
        self._position_top_bar()
        self._position_status_bar()
        self._position_control_panel()

    def statusBar(self) -> QStatusBar:
        """Compatibility shim — callers use ``self.statusBar().showMessage``."""
        return self._status_bar

    def addTitleBarWidget(self, widget, index: int = 2) -> None:
        """Insert *widget* into the frameless title bar.

        ``index`` is the position in the title bar's hBoxLayout (0=icon,
        1=title, 2=after title / before spacer, 3=before min/max/close).
        Default 2 places it on the left side; use 3 for the right side.
        """
        self.titleBar.hBoxLayout.insertWidget(index, widget, 0,
                                              Qt.AlignmentFlag.AlignVCenter)

    def _install_top_bar(self) -> None:
        """Add a persistent control bar below the title bar (always visible).

        Mirrors ``_install_status_bar``: a child QWidget placed at the top of
        the content area (just under the title bar), manually repositioned in
        ``resizeEvent``. The top margin of ``widgetLayout`` is bumped by the
        bar's height so pages never render underneath it. The bar's own layout
        (a QHBoxLayout) is exposed via ``topBarLayout()`` so subclasses can
        populate it.
        """
        self._top_bar = QWidget(self)
        self._top_bar.setObjectName("topBar")
        # A subtle bottom border separates the bar from the page content.
        self._top_bar.setStyleSheet(
            "#topBar { border-bottom: 1px solid rgba(128,128,128,90); }")
        self._top_bar_layout = QVBoxLayout(self._top_bar)
        self._top_bar_layout.setContentsMargins(16, 8, 16, 8)
        self._top_bar_layout.setSpacing(8)
        # Height: enough for a toolbar row (combo/spinbox ~ 24px + margins).
        self._top_bar_h = 100
        # Reserve room below the title bar for this bar in the content column.
        try:
            lm, tm, rm, bm = self.widgetLayout.getContentsMargins()
            tb_h = self.titleBar.height() if self.titleBar else 48
            self.widgetLayout.setContentsMargins(lm, tb_h + self._top_bar_h,
                                                 rm, bm)
        except Exception:
            pass
        self._position_top_bar()

    def _position_top_bar(self) -> None:
        """Place the top bar flush under the title bar, to the RIGHT of the nav.

        The left navigation panel's expand/collapse button sits at its top, so
        the top bar must NOT span the full window width (it would overlap that
        button). Start it at the navigation panel's right edge — mirroring how
        the FluentTitleBar positions itself — and let it track the panel width
        so it stays clear as the nav expands/collapses.
        """
        bar = getattr(self, "_top_bar", None)
        if bar is None:
            return
        tb_h = self.titleBar.height() if self.titleBar else 48
        h = getattr(self, "_top_bar_h", 34)
        # x = current nav panel width (0 if it isn't laid out yet).
        try:
            nav_w = self.navigationInterface.width()
        except Exception:
            nav_w = 0
        bar.setGeometry(nav_w, tb_h, self.width() - nav_w, h)
        bar.raise_()

    def topBarLayout(self) -> QVBoxLayout:
        """Return the top bar's content layout (for subclasses to populate)."""
        return self._top_bar_layout

    def _install_control_panel(self) -> None:
        """Add a persistent control panel directly below the nav tab list.

        Sits OVER the navigation column, pinned just under the LAST nav tab
        (not at the window bottom, not overlapping the tabs). The nav tab list is
        top-aligned and its end position is fixed (tab count × 36px), so the
        panel anchors to that fixed y. Spans the nav width (~200px). Mirrors the
        top/status-bar overlay pattern: a child QWidget repositioned in
        ``resizeEvent``. Its content layout (a QVBoxLayout) is exposed via
        ``controlPanelLayout()`` for the subclass to populate with the shared
        restraint/condition/height/prone/analyze controls.
        """
        self._ctrl_panel = QWidget(self)
        self._ctrl_panel.setObjectName("ctrlPanel")
        # A themed surface + top border so it reads as a distinct panel.
        self._ctrl_panel.setStyleSheet(
            "#ctrlPanel { border-top: 1px solid rgba(128,128,128,90); }")
        self._ctrl_panel_layout = QVBoxLayout(self._ctrl_panel)
        self._ctrl_panel_layout.setContentsMargins(10, 8, 10, 8)
        self._ctrl_panel_layout.setSpacing(6)
        self._ctrl_scroll = QScrollArea(self)
        self._ctrl_scroll.setObjectName("analysisScroll")
        self._ctrl_scroll.setWidgetResizable(True)
        self._ctrl_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._ctrl_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._ctrl_scroll.setWidget(self._ctrl_panel)
        self._position_control_panel()

    def _position_control_panel(self) -> None:
        """Pin the control panel just below the last nav tab (over the nav column).

        Measures the last nav tab's bottom edge and places the panel a few px
        below it, so the panel sits directly under the tab list without
        overlapping any tab and without sinking to the window bottom.
        """
        panel = getattr(self, "_ctrl_panel", None)
        if panel is None:
            return
        try:
            nav_w = self.navigationInterface.width()
        except Exception:
            nav_w = 0
        # Find where the tab list ends: scan the nav panel's tab widgets for the
        # lowest bottom edge (mapped into window coords).
        try:
            from qfluentwidgets.components.navigation.navigation_widget import \
                NavigationWidget
            nav_panel = self.navigationInterface.panel
            last_bottom = 0
            for w in nav_panel.findChildren(NavigationWidget):
                if not w.isVisible():
                    continue
                b = w.mapTo(self, w.rect().bottomLeft()).y()
                if b > last_bottom:
                    last_bottom = b
            top_y = last_bottom + 4   # 4px gap under the last tab
        except Exception:
            # Fallback: a fixed offset (7 tabs × 36px + header) if the scan fails.
            top_y = (self.titleBar.height() if self.titleBar else 48) + 7 * 36 + 10
        scroll = getattr(self, "_ctrl_scroll", None)
        if scroll is not None:
            # The navigation can collapse on a narrow window. A full settings
            # form must never be squeezed into its 48px icon rail.
            scroll.setVisible(nav_w >= 180)
            available = self.height() - self.statusBar().height() - top_y - 8
            scroll.setGeometry(8, top_y, max(0, nav_w - 16), max(0, available))
            scroll.raise_()

    def controlPanelLayout(self) -> QVBoxLayout:
        """Return the control panel's content layout (for subclasses)."""
        return self._ctrl_panel_layout


class MainWindow(ProcessingController, BertFluentWindow):
    """BERT_READER X main window — FluentWindow + Qt Charts."""

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
        self.select_plot_tab2 = ComboBox()
        self.select_plot_tab2b = ComboBox()   # mirror for the split ASTM tab
        self.select_plot_tab3 = ComboBox()
        self.select_plot_tab4 = ComboBox()
        self.select_plot_tab5 = ComboBox()    # Z Reversal tab
        self.select_plot_tab6 = ComboBox()    # X/Y Reversal tab
        self._data_combos = [
            self.select_plot_tab2, self.select_plot_tab2b,
            self.select_plot_tab3, self.select_plot_tab4,
            self.select_plot_tab5, self.select_plot_tab6,
        ]

        # Build UI
        self._setup_ui()
        # Keep the left navigation panel expanded by default (showing the full
        # item labels) instead of collapsing to icon-only on mouse-leave. The
        # nav still collapses on very narrow windows, but at normal widths it
        # stays open. expand() pops it open immediately (no hover needed).
        self.navigationInterface.setCollapsible(False)
        # Narrow the expanded panel (default 322px) so the charts get more
        # room. The labels are short (e.g. "ASTM Sustained", "GB Contour"),
        # so ~200px is plenty at the app's font size.
        self.navigationInterface.setExpandWidth(224)
        self.navigationInterface.setMinimumExpandWidth(1000)
        self.navigationInterface.expand(useAni=False)
        # Reposition the top bar when the nav panel expands/collapses so it
        # always starts to the RIGHT of the nav (clear of the collapse button).
        # The expand() call lays out the nav asynchronously, so also re-position
        # on the next event loop pass once the nav's final width is settled.
        self.navigationInterface.displayModeChanged.connect(
            lambda: (self._position_top_bar(), self._position_control_panel()))
        QTimer.singleShot(0, self._position_top_bar)
        QTimer.singleShot(0, self._position_control_panel)
        self._update_status_label()

        # Wire theme changes: re-apply QSS + refresh inline styles.
        theme_manager.changed.connect(self._on_theme_changed)
        # Apply current theme immediately (covers re-entrancy on toggle).
        self._on_theme_changed(theme_manager.mode)
        self._init_processing()

        # Real chart rendering warms naturally when data is loaded. Keeping
        # startup free of synthetic plots also leaves the empty state stable.

    def _on_theme_changed(self, _mode: str) -> None:
        """Refresh chart palettes after a light/dark switch.

        qfluentwidgets restyles all Fluent widgets automatically on theme
        change — no global QSS re-apply, no inline-style re-render needed
        (this is the big perf win over the old hand-rolled theme system).
        Only the Qt Charts layer needs a manual nudge because charts live on
        a QGraphicsScene and ignore QSS.
        """
        if hasattr(self, 'theme_toggle'):
            self.theme_toggle.blockSignals(True)
            self.theme_toggle.setChecked(theme_manager.is_dark)
            self.theme_toggle.blockSignals(False)
            self._theme_toggle_icon = FI.QUIET_HOURS if theme_manager.is_dark else FI.BRIGHTNESS
        self.refresh_theme()
        if hasattr(self, '_processing_state'):
            self._set_processing_state(self._processing_state, self._workspace_title.toolTip())

    def refresh_theme(self) -> None:
        """Re-apply chart palettes + non-Fluent widget QSS after a theme switch.

        Fluent widgets self-update via qfluentwidgets' themeChanged signal.
        This (1) restyles the Qt Charts canvases (which bypass QSS), and
        (2) applies token-driven QSS on the WINDOW widget so non-Fluent widgets
        (QToolBar action buttons, QLabel, QGroupBox titles) become readable in
        dark mode — the app-level stylesheet can't be used because qfluentwidgets
        owns it. It also (3) re-renders every toolbar action's FluentIcon for
        the current theme, because QAction caches the icon pixmap at build time
        and doesn't refresh it on a theme switch.
        """
        # Theme-aware QSS for non-Fluent widgets (toolbar buttons, labels...).
        self.setStyleSheet(theme_manager.app_qss)
        # Re-render FluentIcons on toolbar actions + title-bar buttons for the
        # current theme (they're cached at light-mode build time).
        self._refresh_action_icons()
        self.sidebar.refresh_theme()
        for canvas in (self.trendplot, self.canvasASTMSust, self.canvasASTMEgg,
                       self.canvasGB, self.canvasAccZone,
                       self.canvasZReversal, self.canvasXYReversal):
            if hasattr(canvas, "restyle_theme"):
                canvas.restyle_theme()

    def _refresh_action_icons(self) -> None:
        """Re-render every FluentIcon-bearing action/button for the live theme.

        QAction icons (and TransparentToolButton icons in the title bar) are
        rasterised once at construction; qfluentwidgets only auto-refreshes its
        own Fluent widget classes, so plain QToolBar actions stay on the
        build-time palette. We re-emit ``fluentIcon.icon(theme=...)`` for each.
        """
        from qfluentwidgets import Theme
        target_theme = Theme.DARK if theme_manager.is_dark else Theme.LIGHT
        # Toolbar actions.
        for tb in self.findChildren(QToolBar):
            for a in tb.actions():
                fi = getattr(a, "fluentIcon", None)
                if fi is not None:
                    try:
                        a.setIcon(fi.icon(theme=target_theme))
                    except Exception:
                        pass
        # Title-bar toggle icon.
        tt = getattr(self, "theme_toggle", None)
        if tt is not None:
            fi = getattr(self, "_theme_toggle_icon", None)
            if fi is not None:
                try:
                    tt.setIcon(fi.icon(theme=target_theme))
                except Exception:
                    pass
        # Top-bar QToolButtons (Add/Clear/Generate/Annot) — these aren't in a
        # QToolBar so the loop above misses them. Re-apply a fresh theme-aware
        # qicon (FluentIconEngine reads the live theme at paint time) + nudge a
        # repaint so the new-theme icon shows immediately.
        top_bar = getattr(self, "_top_bar", None)
        if top_bar is not None:
            for btn in top_bar.findChildren(QToolButton):
                a = getattr(btn, "_action", None)
                if a is None:
                    # Fall back to matching by text — _tb_button_for doesn't tag
                    # the action on the button, so walk the actions list.
                    continue
                fi = getattr(a, "fluentIcon", None)
                if fi is not None and hasattr(fi, "qicon"):
                    try:
                        btn.setIcon(fi.qicon())
                        btn.update()
                    except Exception:
                        pass

    # ==================================================================
    # UI Setup
    # ==================================================================

    def _setup_ui(self) -> None:
        # --- Datasets sidebar: persistent RIGHT panel (not a nav page) ---
        # FluentWindow has no built-in right dock, but its hBoxLayout is
        # nav | stacked-pages | <extra>; appending the sidebar here keeps it
        # visible regardless of which page is selected. DataSidebar manages its
        # own min/max width and paints a theme-aware background itself.
        self.sidebar = DataSidebar(self)
        self._add_right_panel(self.sidebar)

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

        # All tabs are built now — build the shared control panel (restraint /
        # condition / height / prone / Analyze) that sits under the nav tabs.
        # These are the SINGLE instances every page's Generate/analyze reads.
        self._build_control_panel()

        # --- Theme toggle lives in the frameless title bar ---
        self._setup_theme_toggle()

        # --- Status bar ---
        self.statusBar().showMessage("Ready")

    def _add_right_panel(self, panel: QWidget) -> None:
        """Add a persistent widget to the right of the stacked pages.

        FluentWindow's hBoxLayout is [navigationInterface, widgetLayout(stacked)].
        Appending a third item puts it on the far right, outside the
        stackedWidget, so it never switches with nav pages.

        The FluentTitleBar spans the FULL window width across the top ~48px,
        and the persistent top control bar sits just below it (~34px). If the
        panel widget itself starts at y=0 (which it does, being a layout item),
        it covers the title-bar's right side — hiding the theme toggle and the
        min/max/close buttons — and overlaps the top control bar. To prevent
        that, we wrap the panel in a container whose top inset equals the
        title-bar height PLUS the top bar height: the empty strip above the
        panel lets the title bar and top bar show through and receive clicks.
        """
        try:
            tb_h = self.titleBar.height()
        except Exception:
            tb_h = 48
        # Add the persistent top control bar's height so the panel clears it too.
        top_bar_h = getattr(self, "_top_bar_h", 34)
        container = QWidget(self)
        container.setObjectName("rightPanelHost")
        cl = QVBoxLayout(container)
        cl.setContentsMargins(0, tb_h + top_bar_h, 0, self.statusBar().sizeHint().height())
        cl.setSpacing(0)
        cl.addWidget(panel, 1)
        self.hBoxLayout.addWidget(container)
        # Keep the title bar above the container so its buttons always win z-order.
        self.titleBar.raise_()

    # -- helper: wrap a page widget + its toolbar into a vertical layout ------
    @staticmethod
    def _make_page(object_name: str, toolbars, central, margins=(0, 0, 0, 0)):
        """Build a nav page: stacked toolbars on top, central widget below.

        ``toolbars`` is a list of QToolBar (top-to-bottom). ``central`` is the
        chart/content widget (given stretch 1). Returns the page QWidget ready
        to pass to ``addSubInterface`` (caller sets objectName before adding).
        """
        page = QWidget()
        page.setObjectName(object_name)
        lay = QVBoxLayout(page)
        lay.setContentsMargins(*margins)
        lay.setSpacing(0)
        for tb in toolbars:
            lay.addWidget(tb)
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
        """Place the Light/Dark toggle in the title bar (right side)."""
        # TransparentToolButton renders cleanly in the title bar (auto-themed).
        self.theme_toggle = TransparentToolButton(self)
        # Store the FluentIcon so refresh_theme can re-render it for the
        # current theme (TransparentToolButton caches the icon pixmap).
        self._theme_toggle_icon = FI.BRIGHTNESS
        self.theme_toggle.setIcon(FI.BRIGHTNESS)
        self.theme_toggle.setToolTip("Toggle light / dark theme")
        self.theme_toggle.setStatusTip("Toggle light / dark theme")
        self.theme_toggle.setCheckable(True)
        self.theme_toggle.setChecked(theme_manager.is_dark)
        self.theme_toggle.toggled.connect(self._toggle_theme)
        # index 3 = just before the min/max/close buttons group.
        self.addTitleBarWidget(self.theme_toggle, index=3)

    @staticmethod
    def _apply_elevation(widget: QWidget, blur: int = 18, dy: int = 3,
                         alpha: int = 40) -> None:
        """Attach a soft Fluent-style drop shadow to *widget*.

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
    # Persistent top control bar (Add Data / selector / cutoff / Annot).
    # FluentWindow has no addToolBar and is not a QMainWindow, so these
    # global controls live in the custom top bar (see BertFluentWindow) that
    # stays visible across every nav page.
    # ------------------------------------------------------------------

    @staticmethod
    def _vline() -> QFrame:
        """A thin vertical separator for the top bar."""
        line = QFrame()
        line.setFrameShape(QFrame.Shape.VLine)
        line.setFixedWidth(2)
        line.setStyleSheet("color: rgba(128,128,128,110);")
        return line

    def _tb_button_for(self, action: QAction) -> QToolButton:
        """Build a flat QToolButton wired to an Action for the top bar.

        Uses ``fluentIcon.qicon()`` (a FluentIconEngine-backed QIcon) when the
        action carries a FluentIcon, so the icon recolours automatically on a
        light/dark theme switch. ``action.icon()`` alone bakes a single-theme
        pixmap at construction and stays stale after a theme change.
        """
        btn = QToolButton(self._top_bar)
        btn._action = action   # tagged for _refresh_action_icons theme refresh
        btn.setText(action.text())
        btn.setIcon(self._themed_action_icon(action))
        btn.setToolTip(action.toolTip() or action.text())
        btn.setStatusTip(action.statusTip() or action.text())
        btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        btn.setAutoRaise(True)
        btn.setIconSize(QSize(16, 16))
        # Connect both triggered paths (click + action re-emit).
        btn.clicked.connect(action.trigger)
        action.changed.connect(
            lambda a=action, b=btn: (b.setText(a.text()),
                                     b.setIcon(self._themed_action_icon(a)),
                                     b.setEnabled(a.isEnabled())))
        return btn

    @staticmethod
    def _themed_action_icon(action: QAction) -> QIcon:
        """Return a theme-aware QIcon for a Fluent action, else the raw icon."""
        fi = getattr(action, "fluentIcon", None)
        if fi is not None and hasattr(fi, "qicon"):
            return fi.qicon()
        return action.icon()

    def _build_top_bar(self) -> None:
        """Two deliberate rows: file actions, then the active analysis input."""
        lay = self.topBarLayout()
        actions = QHBoxLayout()
        actions.setSpacing(8)
        title = QLabel("Acceleration workspace")
        title.setObjectName("workspaceTitle")
        self._workspace_title = title
        title.setMinimumWidth(80)
        title.setMaximumWidth(220)
        actions.addWidget(title)
        actions.addStretch(1)
        self._data_actions = []
        self._import_actions = []
        for icon, text, slot, needs_data in [
            (FI.FOLDER_ADD, "Add raw data", self.open_raw_data, False),
            (FI.ADD, "Add accel data", self.open_accel_data, False),
            (FI.IOT, "SensorConnect", self.open_sensorconnect_data, False),
            (FI.SHARE, "Export all", self._export_all_to_sup, True),
            (FI.DOCUMENT, "Report", self._show_report_dialog, True),
        ]:
            action = Action(icon, text, self)
            action.triggered.connect(slot)
            actions.addWidget(self._tb_button_for(action))
            if needs_data:
                self._data_actions.append(action)
            else:
                self._import_actions.append(action)
        clear = Action(FI.DELETE, "Clear all", self)
        clear.triggered.connect(lambda: self.reset_trend(reset_datalist=True))
        actions.addWidget(self._tb_button_for(clear))
        self._data_actions.append(clear)
        lay.addLayout(actions)

        inputs = QHBoxLayout()
        inputs.setSpacing(10)
        inputs.addWidget(QLabel("Dataset"))
        self.select_data = ComboBox()
        self.select_data.setMinimumWidth(150)
        self.select_data.setSizePolicy(QSizePolicy.Policy.Expanding,
                                       QSizePolicy.Policy.Fixed)
        self.select_data.setToolTip("Active dataset for the standard and reversal charts")
        inputs.addWidget(self.select_data, 1)
        self._generate_button = PrimaryPushButton("Update charts")
        self._generate_button.setToolTip("Refresh all analysis charts for the selected dataset")
        self._generate_button.clicked.connect(self._generate_active_page)
        inputs.addWidget(self._generate_button)
        inputs.addWidget(self._vline())
        inputs.addWidget(QLabel("Low-pass"))
        self.cutoff_spin = SpinBox()
        self.cutoff_spin.setRange(1, 249)
        self.cutoff_spin.setValue(self.cutoff)
        self.cutoff_spin.setSuffix(" Hz")
        self.cutoff_spin.setFixedWidth(120)
        self.cutoff_spin.setToolTip("Low-pass cutoff. Acceleration Zone always uses 1 Hz.")
        self.cutoff_spin.setKeyboardTracking(False)
        self.cutoff_spin.valueChanged.connect(self._schedule_cutoff_change)
        inputs.addWidget(self.cutoff_spin)
        lay.addLayout(inputs)
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
        self._update_workspace_state()

    def _update_workspace_state(self) -> None:
        loaded = bool(self.datalist)
        for action in getattr(self, '_data_actions', []):
            action.setEnabled(loaded)
        self._generate_button.setEnabled(loaded)
        self.select_data.setEnabled(loaded)
        if not loaded:
            self.select_data.setText("Add a dataset to begin")
        self.select_data.setToolTip(self.select_data.currentText())
        empty = getattr(self, '_trend_empty', None)
        if empty is not None:
            empty.setVisible(not loaded)
            self.trendplot.setVisible(loaded)

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
        # Track which FluentIcon is shown so refresh_theme can re-render it.
        self._theme_toggle_icon = FI.QUIET_HOURS if checked else FI.BRIGHTNESS
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
            action = Action(text, self)
            action.setCheckable(True)
            action.setChecked(self.plottype == value)
            action.triggered.connect(lambda checked=False, v=value: self.switch_plot_type(v))
            group.addAction(action)
            tb.addAction(action)
            self._plot_type_actions[value] = action
        tb.addSeparator()
        fit = QToolButton()
        fit.setText("Align peaks")
        fit.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(fit)
        for which in ('Max', 'Min'):
            for axis, name in enumerate('XYZ', 1):
                callback = self.switch_to_fitmax if which == 'Max' else self.switch_to_fitmin
                menu.addAction(f"{which} {name}", lambda checked=False, a=axis, cb=callback: cb(a))
        fit.setMenu(menu)
        tb.addWidget(fit)
        annotate = Action(FI.TAG, "Annotate", self)
        annotate.triggered.connect(self.switch_annot_mode)
        tb.addAction(annotate)
        _add_chart_toolbar(tb, self.trendplot)

        tb_shift = self._mk_toolbar("Time alignment")
        tb_shift.addWidget(QLabel(" Shift "))
        self._shift_combo = ComboBox()
        self._shift_combo.setMinimumWidth(90)
        self._shift_combo.setMaximumWidth(180)
        self._shift_combo.setToolTip("Select a dataset to shift in time")
        tb_shift.addWidget(self._shift_combo)
        self._shift_spin = DoubleSpinBox()
        self._shift_spin.setRange(-10000.0, 10000.0)
        self._shift_spin.setDecimals(3)
        self._shift_spin.setSingleStep(0.001)
        self._shift_spin.setSuffix(" s")
        self._shift_spin.setFixedWidth(140)
        self._shift_spin.setToolTip("Display shift in seconds; positive moves the trace earlier")
        tb_shift.addWidget(self._shift_spin)
        self._shift_apply = PushButton("Apply")
        self._shift_apply.clicked.connect(self._apply_manual_shift)
        tb_shift.addWidget(self._shift_apply)
        self._shift_reset = PushButton("Reset")
        self._shift_reset.clicked.connect(self._reset_manual_shift)
        tb_shift.addWidget(self._shift_reset)

        self._analyze_window_spin = DoubleSpinBox(self._ctrl_panel)
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
        self._trend_empty = QWidget()
        self._trend_empty.setObjectName("emptyState")
        empty_layout = QVBoxLayout(self._trend_empty)
        empty_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title = QLabel("Start with an acceleration recording")
        title.setObjectName("emptyTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint = QLabel("Load a raw DAQ file or a standard .sup recording.\n"
                      "Compare traces, inspect peaks, and review all seven analysis views.")
        hint.setObjectName("secondaryText")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setWordWrap(True)
        add = PrimaryPushButton("Add raw data")
        add.clicked.connect(self.open_raw_data)
        empty_layout.addWidget(title)
        empty_layout.addWidget(hint)
        empty_layout.addSpacing(12)
        empty_layout.addWidget(add, 0, Qt.AlignmentFlag.AlignHCenter)
        content_layout.addWidget(self._trend_empty, 1)
        content_layout.addWidget(self.trendplot, 1)
        self.trendplot.hide()
        page = self._make_page("page_trend", [tb, tb_shift], content)
        self.addSubInterface(page, NavIcon.TREND, "Trend")

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
        self.addSubInterface(page, NavIcon.SUST, "ASTM Sustained")

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
        self.addSubInterface(page, NavIcon.EGG, "ASTM Eggplot")

    def _build_control_panel(self) -> None:
        """Populate the shared control panel under the nav tabs.

        These are the SINGLE instances of the restraint / condition / height /
        prone / Analyze controls, replacing the per-page mirrored copies. Every
        page's Generate handler and analyze toggle reads these (by the canonical
        attribute names ``select_restraint`` / ``select_cond`` / ``height_input``
        / ``_prone_btn``), so no read-site rewiring is needed beyond the X/Y
        Reversal handler (which previously read its own _xr_* copies).
        """
        lay = self.controlPanelLayout()
        heading = QLabel("ANALYSIS SETTINGS")
        heading.setObjectName("sectionHeading")
        lay.addWidget(heading)

        # --- Restraint type ---
        lay.addWidget(QLabel("Restraint:"))
        self.select_restraint = ComboBox()
        self.select_restraint.addItems([
            'None', 'Individual Lower Body', 'Upper Body',
            'Group Lower Body', 'Convenience Restraint', 'No Restraint',
        ])
        lay.addWidget(self.select_restraint)

        # --- Condition ---
        lay.addWidget(QLabel("Condition:"))
        self.select_cond = ComboBox()
        self.select_cond.addItems(['Normal', 'E-Stop', 'Expected/Permitted Bumping'])
        lay.addWidget(self.select_cond)

        # --- Height (inches) ---
        lay.addWidget(QLabel("Height (in):"))
        self.height_input = SpinBox()
        self.height_input.setRange(0, 80)
        self.height_input.setValue(0)
        lay.addWidget(self.height_input)

        # --- Prone toggle ---
        # Toggling prone re-plots every chart immediately (it changes the ASTM
        # egg shapes + the sustained ax- limit line), so the user sees the
        # update without needing to click Generate.
        self._prone_btn = ToggleToolButton(self._ctrl_panel)
        self._prone_btn.setText("Prone")
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
        self._analyze_btn = ToggleToolButton(self._ctrl_panel)
        self._analyze_btn.setText("Analyze")
        self._analyze_btn.setCheckable(True)
        self._analyze_btn.setStatusTip(
            "Analyze the currently-visible chart (switch tabs to analyze another)")
        self._analyze_btn.toggled.connect(self._toggle_active_page_analysis)
        lay.addWidget(self._analyze_btn)
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
        self._queue_chart_refresh()

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
        self.addSubInterface(page, NavIcon.GB, "GB Contour")

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
        self.addSubInterface(page, NavIcon.ACC_ZONE, "Acc Zone")
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
        self._zr_plottype = ComboBox()
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
        self.addSubInterface(page, NavIcon.Z_REVERSAL, "Z Reversal")

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
        self._xr_plottype = ComboBox()
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
        self._xr_hyst = DoubleSpinBox()
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
        self.addSubInterface(page, NavIcon.XY_REVERSAL, "X/Y Reversal")

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
        (no recompute); ASTM/GB/AccZone recolour their currently-plotted
        measured overlay only — so this is cheap and preserves zoom state.
        """
        color_hex = color_hex.lower()
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

        # 3) Refresh the sidebar card's swatch + label stripe.
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
