# coding: utf-8
"""Shared Qt palette, widget styles and analytical chart colours."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Qt
from PySide6.QtGui import QColor


# ---------------------------------------------------------------------------
# 19-colour dataset palette (kept identical to the original)
# ---------------------------------------------------------------------------
DATASET_COLORS = [
    '#2563EB', '#D97706', '#0D9488', '#9333EA', '#E11D48',
    '#0284C7', '#65A30D', '#C026D3', '#64748B', '#F97316',
    '#14B8A6', '#8B5CF6', '#DC2626', '#16A34A', '#CA8A04',
    '#6366F1', '#06B6D4', '#A855F7', '#DB2777',
]

# Backwards-compatible aliases (a few modules import these names)
PRIMARY = "#226746"
PRIMARY_HOVER = "#105A4A"
PRIMARY_PRESSED = "#0C4539"
DANGER = "#C42B1C"
SUCCESS = "#107C10"
WARNING = "#F7630C"

# Type scale (point sizes)
FONT_XS = 8
FONT_SM = 9
FONT_MD = 10
FONT_LG = 12
FONT_XL = 14
FONT_TITLE = 16

# Layout constants
SIDEBAR_MIN_WIDTH = 380
LINE_WIDTH_DEFAULT = 1.3
# Thicker line for the fitting charts (GB + ASTM), where curves need to read
# clearly against the reference grid and overlapping rides.
LINE_WIDTH_FITTING = 2.0

# Defaults for GB/ASTM callers that don't supply a recording colour.
MEASURED_POS_COLOR = "#1E90FF"   # bright blue (dodger blue) — positive traces
MEASURED_NEG_COLOR = "#32CD32"   # bright green (lime green) — negative traces

# Font stack (also consumed by main.py for the app font)
FONT_FAMILY = '"Segoe UI Variable Text", "Segoe UI", "Microsoft YaHei UI", "Lucida Grande", Verdana, sans-serif'
FONT_FAMILY_DISPLAY = '"Segoe UI Variable Display", "Segoe UI", "Microsoft YaHei UI", sans-serif'

# Radii kept for the inline-style shims below.
_RADIUS = "8px"
_RADIUS_LG = "12px"
_RADIUS_PILL = "999px"


# ---------------------------------------------------------------------------
# Theme tokens shared by widgets, charts and per-recording styles.
# ---------------------------------------------------------------------------

def _darken(hex_color: str, factor: float) -> str:
    """Multiply an ``#RRGGBB`` colour's channels by ``factor``."""
    c = QColor(hex_color)
    c = QColor(
        max(0, min(255, int(c.red() * factor))),
        max(0, min(255, int(c.green() * factor))),
        max(0, min(255, int(c.blue() * factor))),
    )
    return c.name()


def _lighten(hex_color: str, factor: float) -> str:
    """Blend an ``#RRGGBB`` colour toward white by ``factor`` (0..1)."""
    c = QColor(hex_color)
    f = max(0.0, min(1.0, factor))
    c = QColor(
        int(c.red() + (255 - c.red()) * f),
        int(c.green() + (255 - c.green()) * f),
        int(c.blue() + (255 - c.blue()) * f),
    )
    return c.name()


def _with_alpha(hex_color: str, alpha: float) -> str:
    c = QColor(hex_color)
    return f"rgba({c.red()}, {c.green()}, {c.blue()}, {alpha})"


def _build_tokens(dark: bool) -> dict:
    accent = "#93dab5" if dark else PRIMARY
    accent_hover = _darken(accent, 0.88) if not dark else _lighten(accent, 0.18)
    accent_pressed = _darken(accent, 0.76) if not dark else _lighten(accent, 0.32)
    accent_disabled = _lighten(accent, 0.55) if not dark else _darken(accent, 0.5)

    caut_fg_pair = ('#795014', '#F5C972')
    crit_fg_pair = ('#A33240', '#FFABB3')
    caut_bg_pair = ('#FFF1D4', '#4C3820')
    crit_bg_pair = ('#FDE8EC', '#4B2A32')
    idx = 1 if dark else 0

    if dark:
        tk = {
            "mode": "dark",
            # chrome
            "window_bg": "#17201d",
            "bg": "#17201d",
            "surface": "#202c27",
            "surface_alt": "#24312b",
            "surface_hover": "#304137",
            "surface_pressed": "#3C524B",
            # text
            "text": "#eef4ec",
            "text_secondary": "#b4c3b9",
            "text_disabled": "#6E6E6E",
            "text_on_accent": "#12281b",
            # strokes
            "card_stroke": "#394a40",
            "divider": "#384940",
            "control_stroke": "#404040",
            "control_stroke_strong": "#5A5A5A",
            # accent ramp (live from themeColor)
            "accent": accent,
            "accent_hover": accent_hover,
            "accent_pressed": accent_pressed,
            "accent_disabled": accent_disabled,
            "accent_secondary": "#284837",
            "accent_secondary_hover": "#345e48",
            "subtle_fill": "#2D2D2D",
            "subtle_fill_hover": "#3A3A3A",
            # chart palette
            "chart_bg": "#202c27",
            "chart_grid": "#36473e",
            "chart_axis": "#a0b7a9",
            "chart_title": "#eef4ec",
            "chart_text": "#cad7cc",
            # status colours
            "danger": crit_fg_pair[idx],
            "danger_bg": crit_bg_pair[idx],
            "danger_rgb": _hex_to_rgb(crit_fg_pair[idx]),
            "success": "#ACDEC2",
            "success_bg": "#244737",
            "warning": caut_fg_pair[idx],
            "warning_bg": caut_bg_pair[idx],
            # tooltip
            "tooltip_bg": "#2B2B2B",
            "tooltip_border": "#4A4A4A",
            "tooltip_text": "#FFFFFF",
            # scrollbars
            "scrollbar_bg": "transparent",
            "scrollbar_handle": "#5A5A5A",
            "scrollbar_handle_hover": "#7A7A7A",
            # status pill
            "pill_bg": _darken(accent, 0.40),
            "pill_border": accent,
            "pill_text": _lighten(accent, 0.55),
            # glass layers (charts/cards layer subtle translucency over mica)
            "glass_bg": "rgba(255, 255, 255, 0.06)",
            "glass_bg_hover": "rgba(255, 255, 255, 0.10)",
            "glass_bg_subtle": "rgba(255, 255, 255, 0.04)",
            "glass_stroke": "rgba(255, 255, 255, 0.12)",
            "glass_stroke_inner": "rgba(255, 255, 255, 0.06)",
            "glass_shine": "rgba(255, 255, 255, 0.08)",
            # glow blobs
            "glow_a": _with_alpha(accent, 0.22),
            "glow_b": "rgba(139, 92, 246, 0.18)",
            "glow_c": "rgba(14, 165, 233, 0.16)",
        }
    else:
        tk = {
            "mode": "light",
            "window_bg": "#f8f9f5",
            "bg": "#f8f9f5",
            "surface": "#FFFFFF",
            "surface_alt": "#f6f8f4",
            "surface_hover": "#e8f1e7",
            "surface_pressed": "#d8e5d8",
            "text": "#192823",
            "text_secondary": "#586a61",
            "text_disabled": "#A0A0A0",
            "text_on_accent": "#FFFFFF",
            "card_stroke": "#d6e1d5",
            "divider": "#dce4db",
            "control_stroke": "#C7C7C7",
            "control_stroke_strong": "#B0B0B0",
            "accent": accent,
            "accent_hover": accent_hover,
            "accent_pressed": accent_pressed,
            "accent_disabled": accent_disabled,
            "accent_secondary": _lighten(accent, 0.88),
            "accent_secondary_hover": _lighten(accent, 0.80),
            "subtle_fill": "#F9F9F9",
            "subtle_fill_hover": "#EFEFEF",
            "chart_bg": "#FFFFFF",
            "chart_grid": "#e2e9e1",
            "chart_axis": "#64796c",
            "chart_title": "#192823",
            "chart_text": "#43594d",
            "danger": crit_fg_pair[idx],
            "danger_bg": crit_bg_pair[idx],
            "danger_rgb": _hex_to_rgb(crit_fg_pair[idx]),
            "success": "#215C49",
            "success_bg": "#E3F1E3",
            "warning": caut_fg_pair[idx],
            "warning_bg": caut_bg_pair[idx],
            "tooltip_bg": "#F2F2F2",
            "tooltip_border": "#CCCCCC",
            "tooltip_text": "#1A1A1A",
            "scrollbar_bg": "transparent",
            "scrollbar_handle": "#C7C7C7",
            "scrollbar_handle_hover": "#A0A0A0",
            "pill_bg": _lighten(accent, 0.88),
            "pill_border": accent,
            "pill_text": _darken(accent, 0.6),
            "glass_bg": "rgba(255, 255, 255, 0.65)",
            "glass_bg_hover": "rgba(255, 255, 255, 0.78)",
            "glass_bg_subtle": "rgba(255, 255, 255, 0.45)",
            "glass_stroke": "rgba(255, 255, 255, 0.85)",
            "glass_stroke_inner": "rgba(255, 255, 255, 0.55)",
            "glass_shine": "rgba(255, 255, 255, 0.55)",
            "glow_a": _with_alpha(accent, 0.18),
            "glow_b": "rgba(139, 92, 246, 0.14)",
            "glow_c": "rgba(14, 165, 233, 0.12)",
        }
    return tk


def _hex_to_rgb(h: str) -> str:
    h = h.lstrip("#")
    if len(h) == 6:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"{r},{g},{b}"
    return "255,255,255"


# ===========================================================================
# Application palette and appearance settings
# ===========================================================================

class ThemeManager(QObject):
    changed = Signal(str)

    def __init__(self):
        super().__init__()
        self._mode = 'light'
        self._tokens_cache = {}

    @property
    def app_qss(self) -> str:
        """Workspace stylesheet for the current palette."""
        return _app_qss(self.tokens)

    def _apply_app_palette(self) -> None:
        """Supply application colours to widgets and native popups."""
        from PySide6.QtGui import QPalette
        from PySide6.QtWidgets import QApplication as _QApp
        app = _QApp.instance()
        if app is None:
            return
        tk = self.tokens
        pal = QPalette()
        bg = QColor(tk["window_bg"])
        surface = QColor(tk["surface"])
        pal.setColor(QPalette.ColorRole.Window, bg)
        pal.setColor(QPalette.ColorRole.Base, surface)
        pal.setColor(QPalette.ColorRole.AlternateBase, QColor(tk["surface_alt"]))
        pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(tk["tooltip_bg"]))
        pal.setColor(QPalette.ColorRole.ToolTipText, QColor(tk["tooltip_text"]))
        txt = QColor(tk["text"])
        txt2 = QColor(tk["text_secondary"])
        for role in (QPalette.ColorRole.WindowText, QPalette.ColorRole.Text,
                     QPalette.ColorRole.ButtonText):
            pal.setColor(role, txt)
        # Brighten button face so QGroupBox / buttons read in dark.
        pal.setColor(QPalette.ColorRole.Button, surface)
        pal.setColor(QPalette.ColorRole.Highlight, QColor(tk["accent"]))
        pal.setColor(QPalette.ColorRole.HighlightedText, QColor(tk["text_on_accent"]))
        pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText,
                     QColor(tk["text_disabled"]))
        pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text,
                     QColor(tk["text_disabled"]))
        app.setPalette(pal)

    # -- read-only properties -------------------------------------------------
    @property
    def mode(self) -> str:
        return self._mode

    @property
    def is_dark(self) -> bool:
        return self._mode == "dark"

    @property
    def tokens(self) -> dict:
        """Cached colour tokens for the current appearance."""
        dark = self.is_dark
        key = "dark" if dark else "light"
        tk = self._tokens_cache.get(key)
        if tk is None:
            tk = _build_tokens(dark)
            self._tokens_cache[key] = tk
        return tk

    def t(self, key: str) -> str:
        return self.tokens.get(key, "")

    # -- mutation -------------------------------------------------------------
    def set_mode(self, mode: str) -> None:
        if mode not in ('light', 'dark'):
            raise ValueError(f'Unknown theme: {mode}')
        if self._mode != mode:
            self._mode = mode
            self._apply_app_palette()
            self.changed.emit(mode)

    def toggle(self) -> str:
        self.set_mode('light' if self.is_dark else 'dark')
        return self.mode

    def apply(self, app) -> None:
        app.setStyle('Fusion')
        self.set_mode('light')
        self._apply_app_palette()


theme_manager = ThemeManager()


# ===========================================================================
# Chart theming — QChart/QChartView live on a QGraphicsScene and ignore QSS,
# so this applies the palette directly. Values come from ``theme_manager.tokens``
# for both appearances.
# ===========================================================================

def apply_chart_theme(chart) -> None:
    """Apply the current theme palette to a QChart and its axes/series."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QPen, QFont, QBrush

    tk = theme_manager.tokens
    bg = QColor(tk["chart_bg"])
    grid = QColor(tk["chart_grid"])
    axis_color = QColor(tk["chart_axis"])
    title_color = QColor(tk["chart_title"])
    text_color = QColor(tk["chart_text"])

    chart.setBackgroundBrush(QBrush(bg))
    chart.setBackgroundPen(QPen(Qt.PenStyle.NoPen))
    try:
        chart.setPlotAreaBackgroundBrush(QBrush(bg))
        chart.setPlotAreaBackgroundVisible(True)
    except Exception:
        pass

    title_font = QFont("Segoe UI", 10)
    title_font.setBold(True)
    chart.setTitleFont(title_font)
    chart.setTitleBrush(QBrush(title_color))

    try:
        legend = chart.legend()
        legend.setLabelColor(text_color)
        legend.setFont(QFont("Segoe UI", 9))
        # Place at top so Qt allocates enough height for the legend to WRAP to
        # multiple rows when entries exceed the chart width (default AlignBottom
        # reserves a single row and clips overflow).
        from PySide6.QtCore import Qt as _Qt
        legend.setAlignment(_Qt.AlignmentFlag.AlignTop)
        legend.setShowToolTips(True)
    except Exception:
        pass

    grid_pen = QPen(grid)
    grid_pen.setWidth(1)
    axis_pen = QPen(axis_color)
    axis_pen.setWidth(1)

    from PySide6.QtCharts import QValueAxis
    for axis in chart.axes():
        if isinstance(axis, QValueAxis):
            axis.setGridLineColor(grid)
            axis.setMinorGridLineColor(grid)
            axis.setLinePen(axis_pen)
            axis.setLabelsColor(text_color)
            axis.setLabelsBrush(QBrush(text_color))
            axis.setLabelsFont(QFont("Segoe UI", 9))
            try:
                axis.setTitleBrush(QBrush(axis_color))
                f = QFont("Segoe UI", 9); f.setBold(True)
                axis.setTitleFont(f)
            except Exception:
                pass
        else:
            try:
                axis.setLinePen(axis_pen)
                axis.setLabelsColor(text_color)
                axis.setLabelsBrush(QBrush(text_color))
                axis.setLabelsFont(QFont("Segoe UI", 9))
            except Exception:
                pass


def _app_qss(tk: dict) -> str:
    """Styles shared by the workspace and its controls."""
    return (
        f"QComboBox, QAbstractSpinBox, QLineEdit {{ color: {tk['text']}; background: {tk['surface']}; border: 1px solid {tk['control_stroke']}; border-radius: 5px; padding: 3px 7px; min-height: 23px; }}"
        f"QComboBox:focus, QAbstractSpinBox:focus, QLineEdit:focus {{ border-color: {tk['accent']}; }}"
        f"QComboBox QAbstractItemView, QMenu {{ color: {tk['text']}; background: {tk['surface']}; selection-color: {tk['text']}; selection-background-color: {tk['accent_secondary']}; }}"
        f"QPushButton {{ color: {tk['text']}; background: {tk['surface']}; border: 1px solid {tk['divider']}; border-radius: 5px; padding: 5px 12px; }}"
        f"QPushButton:hover {{ background: {tk['surface_hover']}; border-color: {tk['accent']}; }}"
        f"QPushButton#primaryButton {{ background: {tk['accent']}; border-color: {tk['accent']}; color: {tk['text_on_accent']}; font-weight: bold; }}"
        f"QPushButton:disabled, QPushButton#primaryButton:disabled, QComboBox:disabled {{ background: {tk['surface_alt']}; color: {tk['text_disabled']}; border-color: {tk['divider']}; }}"
        f"QPushButton#formatButton {{ text-align: left; padding: 11px 16px; font: 14px 'Segoe UI'; background: {tk['surface_alt']}; }}"
        f"QPushButton#formatButton:hover, QPushButton#formatButton:focus {{ background: {tk['accent_secondary']}; border-color: {tk['accent']}; }}"
        f"QWidget#workbench {{ background: {tk['window_bg']}; }}"
        "QFrame#rail { background: #182137; border: none; }"
        "QLabel#brandName { color: #F5F7FE; font: bold 24px 'Segoe UI'; }"
        "QLabel#brandTag, QLabel#railCaption { color: #AAB8D0; font: bold 11px 'Segoe UI'; }"
        "QLabel#railCaption { padding: 8px 10px 3px; }"


        "QPushButton#viewLink { color: #E3E9F6; background: transparent; border: none; text-align: left; padding: 3px 12px; font: 14px 'Segoe UI'; border-radius: 5px; }"
        "QPushButton#viewLink:hover { background: #293652; }"
        "QPushButton#viewLink:checked { color: #F5FFEA; background: #35456A; border-left: 3px solid #ABBFFF; padding-left: 9px; font-weight: bold; }"
        "QPushButton#viewLink:focus { border: 1px solid #ABBFFF; }"
        f"QToolBar {{ background: {tk['surface']}; border: 1px solid {tk['divider']}; border-radius: 6px; spacing: 2px; padding: 3px; }}"
        f"QWidget#topBar {{ background: {tk['surface']}; border-bottom: 1px solid {tk['divider']}; }}"
        "QWidget#ctrlPanel, QScrollArea#analysisScroll { background: #202B43; border: none; border-radius: 7px; }"
        "QWidget#ctrlPanel QLabel { color: #D3DDF2; }"
        "QWidget#ctrlPanel QToolButton { color: #E8ECFF; background: #34425E; }"
        "QWidget#ctrlPanel QToolButton:checked { color: #182137; background: #BBC8FF; }"
        f"QScrollArea#analysisScroll QScrollBar:vertical {{ background: transparent; width: 6px; margin: 0; }}"
        f"QScrollArea#analysisScroll QScrollBar::handle:vertical {{ background: {tk['control_stroke']}; border-radius: 3px; min-height: 24px; }}"
        "QScrollArea#analysisScroll QScrollBar::add-line:vertical, QScrollArea#analysisScroll QScrollBar::sub-line:vertical { height: 0; }"
        "QScrollArea#analysisScroll QScrollBar::add-page:vertical, QScrollArea#analysisScroll QScrollBar::sub-page:vertical { background: transparent; }"
        f"QLabel#pendingChart {{ color: {tk['text_secondary']}; background: {tk['chart_bg']}; font-size: 12pt; border-radius: 10px; padding: 20px; }}"
        f"QLabel#workspaceTitle {{ color: {tk['text']}; font: bold 27px 'Segoe UI'; }}"
        f"QLabel#caption {{ color: {tk['text_secondary']}; font: bold 11px 'Segoe UI'; }}"
        f"QFrame#channelPanel {{ background: {tk['chart_bg']}; border: 1px solid {tk['divider']}; border-radius: 6px; }}"
        f"QLabel#channelCode {{ color: {tk['pill_text']}; background: {tk['pill_bg']}; border-radius: 3px; font: bold 11px 'Segoe UI'; padding: 1px 5px; }}"
        f"QLabel#channelName {{ color: {tk['text']}; font: bold 13px 'Segoe UI'; }}"
        f"QLabel#sectionHeading {{ color: {tk['text_secondary']}; font-size: 8pt; font-weight: 600; padding: 8px 0; }}"
        f"QLabel#secondaryText {{ color: {tk['text_secondary']}; line-height: 1.5; }}"
        f"QWidget#emptyState {{ background: {tk['surface']}; border: 1px solid {tk['divider']}; border-radius: 7px; }}"
        f"QLabel#emptyTitle {{ font: bold 33px 'Segoe UI'; }}"
        f"QToolButton#importButton {{"
        f" color: {tk['pill_text']}; background: {tk['pill_bg']};"
        f" border: 1px solid {tk['control_stroke']}; border-radius: 6px; padding: 6px 16px 6px 8px; }}"
        f"QToolButton#importButton:hover {{"
        f" background: {tk['accent_secondary_hover']}; border-color: {tk['accent']}; }}"
        f"QWidget#InfoCard {{ background: {tk['surface']}; border: 1px solid {tk['card_stroke']}; border-radius: 10px; }}"
        f"QToolButton, QToolBar QToolButton {{"
        f"  color: {tk['text']}; background: transparent; border: none;"
        f"  padding: 5px 8px; border-radius: 6px;"
        f"}}"
        f"QToolButton:hover, QToolBar QToolButton:hover {{"
        f"  background: {tk['surface_hover']};"
        f"}}"
        f"QToolBar QToolButton {{ padding: 4px 6px; }}"
        f"QToolButton:disabled {{ color: {tk['text_secondary']}; }}"
        f"QToolButton:checked, QToolBar QToolButton:checked {{"
        f"  background: {tk['accent_secondary']};"
        f"}}"
        f"QGroupBox {{"
        f"  color: {tk['text']}; border: 1px solid {tk['card_stroke']};"
        f"  border-radius: 6px; margin-top: 10px; padding-top: 8px;"
        f"}}"
        f"QGroupBox::title {{"
        f"  color: {tk['text']}; subcontrol-origin: margin;"
        f"  left: 10px; padding: 0 4px;"
        f"}}"
        f"QLabel {{ color: {tk['text']}; }}"
        f"QStatusBar {{ color: {tk['text_secondary']};"
        f"  background: {tk['surface_alt']}; border-top: 1px solid {tk['divider']}; padding: 2px 8px; }}"
        f"QStatusBar::item {{ border: none; }}"
        f"QToolTip {{ color: {tk['tooltip_text']};"
        f"  background: {tk['tooltip_bg']}; border: 1px solid {tk['tooltip_border']}; }}"
    )


def chart_tooltip_style() -> str:
    """Inline style for the floating click-tooltip labels on charts."""
    tk = theme_manager.tokens
    return (
        f"background-color: {tk['glass_bg']}; "
        f"color: {tk['text']}; "
        f"border: 1px solid {tk['glass_stroke']}; "
        f"border-top: 1px solid {tk['glass_shine']}; "
        f"border-radius: {_RADIUS}; padding: 5px 8px; "
        f"font-size: {FONT_SM}pt;"
    )


# ===========================================================================
# Per-recording styles
# ===========================================================================

def card_title_style(color: str) -> str:
    tk = theme_manager.tokens
    # The title TEXT always uses the theme text colour (readable in both light
    # and dark mode). The dataset colour is shown as a thin LEFT accent stripe
    # — large colour fills behind filename text made titles unreadable for
    # low-contrast palette colours (#0000FF on dark, #FFFF00 on light, etc.).
    return (
        f"color: {tk['text']}; "
        f"font-size: {FONT_MD}pt; font-weight: 600;"
        f" border-left: 3px solid {color or tk['accent']};"
        f" padding-left: 5px;"
    )


def card_meta_inline_style() -> str:
    tk = theme_manager.tokens
    return f"color: {tk['text_secondary']}; font-size: {FONT_SM}pt;"


def card_section_label_style() -> str:
    tk = theme_manager.tokens
    return (
        f"color: {tk['text_secondary']}; "
        f"font-size: {FONT_XS}pt; font-weight: 700; "
        f"letter-spacing: 0.6px;"
    )


def swatch_button_style(color: str) -> str:
    # Border reads the theme so it's visible in BOTH modes: the old fixed
    # rgba(0,0,0,...) was invisible against the dark surface in dark mode.
    tk = theme_manager.tokens
    border = tk["card_stroke"]
    border_hover = tk["text_secondary"]
    return (
        f"QPushButton {{ background-color: {color}; border: 1px solid {border};"
        f" border-radius: 7px; }}"
        f"QPushButton:hover {{ border: 1px solid {border_hover}; }}"
    )


def status_pill_style(text: str = "") -> str:
    tk = theme_manager.tokens
    base = (
        f"QLabel {{ background-color: {tk['pill_bg']}; "
        f"color: {tk['pill_text']}; "
        f"border: 1px solid {tk['pill_border']}; "
        f"border-radius: {_RADIUS_PILL}; padding: 2px 10px; "
        f"font-size: {FONT_SM}pt; }}"
    )
    return base
