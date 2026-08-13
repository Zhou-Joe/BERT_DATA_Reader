# coding: utf-8
"""
BERT Reader Theme — QFluentWidgets bridge.

This module used to be a hand-rolled Fluent design system (a 1200-line token
dict + a giant QSS string + a fake glassmorphism ``GlowBackground``). It has
been rewritten as a thin adapter over the ``qfluentwidgets`` library, which now
owns all widget styling and light/dark switching.

Two things survive for backward compatibility:

1.  **Constants** — ``DATASET_COLORS``, ``FONT_*``, ``FONT_FAMILY``,
    ``LINE_WIDTH_DEFAULT``, ``SIDEBAR_MIN_WIDTH`` and the legacy color aliases.
2.  **The token dict + chart helpers** — ``theme_manager.tokens``,
    ``apply_chart_theme()`` and ``chart_tooltip_style()`` are kept because the
    Qt Charts layer (``chart_*.py``) cannot be themed by qfluentwidgets (Qt
    Charts live on a ``QGraphicsScene`` and ignore QSS). The token values are
    now *derived* from qfluentwidgets' current theme (``isDarkTheme()``,
    ``themeColor()``, ``FluentSystemColor``) so the charts stay in sync with
    the rest of the UI automatically.

The old inline-style factories (``card_*_style``, ``status_pill_style``,
``swatch_button_style``, ``svg_icon``) are kept as thin shims so existing
callers keep importing, but the UI migration replaces them with real Fluent
widgets — they will be removed once sidebar.py / main_window.py no longer
reference them.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QWidget

import qfluentwidgets as qfw
from qfluentwidgets import (
    Theme,
    isDarkTheme,
    qconfig,
    setTheme,
    setThemeColor,
    themeColor,
)
from qfluentwidgets.common import FluentSystemColor as _SC

# ---------------------------------------------------------------------------
# 19-colour dataset palette (kept identical to the original)
# ---------------------------------------------------------------------------
DATASET_COLORS = [
    '#FF0000', '#0000FF', '#ADD8E6', '#800080', '#FFFF00',
    '#00FF00', '#FF00FF', '#FFC0CB', '#808080', '#000000',
    '#FFA500', '#A52A2A', '#800000', '#008000', '#808000',
    '#00008B', '#7FFFD4', '#C0C0C0', '#00FFFF',
]

# Backwards-compatible aliases (a few modules import these names)
PRIMARY = "#0078D4"
PRIMARY_HOVER = "#106EBE"
PRIMARY_PRESSED = "#005A9E"
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
LINE_WIDTH_DEFAULT = 0.6
# Thicker line for the fitting charts (GB + ASTM), where curves need to read
# clearly against the reference grid and overlapping rides.
LINE_WIDTH_FITTING = 2.0

# Measured-trace sign colours, shared by the GB and ASTM fitting charts so the
# +/− measured curves read consistently across both. These OVERRIDE the
# dataset's own colour for the measured (±) series — i.e. regardless of which
# dataset you plot, measured (+) is always bright blue, measured (−) bright
# green. Bright (not dark) so the traces stand out against the reference grid.
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
# Token derivation — build the legacy token dict from qfluentwidgets' theme.
#
# Charts can't read QSS, so they still read ``theme_manager.tokens["..."]``.
# We reconstruct the same keys the chart/inline helpers expect, sourcing the
# values from qfluentwidgets so they track the real Fluent theme. The exact
# hex values differ slightly from the old hand-picked palette, but the keys
# and semantics are identical.
# ---------------------------------------------------------------------------

def _pair(sc: _SC) -> tuple[str, str]:
    """Return ``(light_hex, dark_hex)`` for a FluentSystemColor."""
    return sc.value


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
    """Reconstruct the legacy token dict from the current Fluent theme.

    ``dark`` selects which side of every (light, dark) pair to use. Accent
    colours come from qfluentwidgets' live ``themeColor()`` so a custom
    accent propagates everywhere, including the charts.
    """
    accent = QColor(themeColor()).name()  # e.g. "#009faa"
    accent_hover = _darken(accent, 0.88) if not dark else _lighten(accent, 0.18)
    accent_pressed = _darken(accent, 0.76) if not dark else _lighten(accent, 0.32)
    accent_disabled = _lighten(accent, 0.55) if not dark else _darken(accent, 0.5)

    succ_fg_pair = _pair(_SC.SUCCESS_FOREGROUND)
    caut_fg_pair = _pair(_SC.CAUTION_FOREGROUND)
    crit_fg_pair = _pair(_SC.CRITICAL_FOREGROUND)
    succ_bg_pair = _pair(_SC.SUCCESS_BACKGROUND)
    caut_bg_pair = _pair(_SC.CAUTION_BACKGROUND)
    crit_bg_pair = _pair(_SC.CRITICAL_BACKGROUND)
    idx = 1 if dark else 0

    if dark:
        tk = {
            "mode": "dark",
            # chrome
            "window_bg": "#202020",
            "bg": "#1F1F1F",
            "surface": "#2B2B2B",
            "surface_alt": "#292929",
            "surface_hover": "#3A3A3A",
            "surface_pressed": "#454545",
            # text
            "text": "#FFFFFF",
            "text_secondary": "#C7C7C7",
            "text_disabled": "#6E6E6E",
            "text_on_accent": "#FFFFFF",
            # strokes
            "card_stroke": "#3D3D3D",
            "divider": "#3F3F3F",
            "control_stroke": "#404040",
            "control_stroke_strong": "#5A5A5A",
            # accent ramp (live from themeColor)
            "accent": accent,
            "accent_hover": accent_hover,
            "accent_pressed": accent_pressed,
            "accent_disabled": accent_disabled,
            "accent_secondary": _darken(accent, 0.45),
            "accent_secondary_hover": _darken(accent, 0.55),
            "subtle_fill": "#2D2D2D",
            "subtle_fill_hover": "#3A3A3A",
            # chart palette
            "chart_bg": "#1F1F1F",
            "chart_grid": "#333333",
            "chart_axis": "#B0B0B0",
            "chart_title": "#FFFFFF",
            "chart_text": "#D8D8D8",
            # status (from FluentSystemColor pairs)
            "danger": crit_fg_pair[idx],
            "danger_bg": crit_bg_pair[idx],
            "danger_rgb": _hex_to_rgb(crit_fg_pair[idx]),
            "success": succ_fg_pair[idx],
            "success_bg": succ_bg_pair[idx],
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
            "window_bg": "#F3F3F3",
            "bg": "#FAFAFA",
            "surface": "#FFFFFF",
            "surface_alt": "#F5F5F5",
            "surface_hover": "#EDEDED",
            "surface_pressed": "#E5E5E5",
            "text": "#1A1A1A",
            "text_secondary": "#5B5B5B",
            "text_disabled": "#A0A0A0",
            "text_on_accent": "#FFFFFF",
            "card_stroke": "#E5E5E5",
            "divider": "#E4E4E4",
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
            "chart_grid": "#ECECEC",
            "chart_axis": "#5B5B5B",
            "chart_title": "#1A1A1A",
            "chart_text": "#333333",
            "danger": crit_fg_pair[idx],
            "danger_bg": crit_bg_pair[idx],
            "danger_rgb": _hex_to_rgb(crit_fg_pair[idx]),
            "success": succ_fg_pair[idx],
            "success_bg": succ_bg_pair[idx],
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
# ThemeManager — thin adapter over qfluentwidgets
# ===========================================================================

class ThemeManager(QObject):
    """Holds the current theme mode and notifies listeners on change.

    All real work is delegated to ``qfluentwidgets.qconfig``: this class
    exists so existing callers (``theme_manager.set_mode/toggle/mode``,
    ``theme_manager.tokens``, the ``changed`` signal) keep working without
    edits.
    """

    changed = Signal(str)   # emits the new mode ("light" / "dark")

    def __init__(self) -> None:
        super().__init__()
        self._tokens_cache: dict[str, dict] = {}
        # Bridge qfluentwidgets' theme change into our legacy signal so the
        # chart layer (which connects to ``changed``) re-styles on switch.
        qconfig.themeChanged.connect(self._on_qfw_theme_changed)

    # -- qfluentwidgets bridge ------------------------------------------------
    def _on_qfw_theme_changed(self, theme: Theme) -> None:
        # Invalidate cached tokens; emit legacy mode string.
        self._tokens_cache.pop("light", None)
        self._tokens_cache.pop("dark", None)
        # Update the QApplication palette so NON-Fluent widgets (plain QLabel,
        # QToolBar text, navigation labels, QGroupBox titles) pick up readable
        # text/background colours in dark mode. qfluentwidgets only restyles
        # its own Fluent widget classes, leaving the app palette on light.
        # The per-widget QSS is applied by MainWindow.refresh_theme (app-level
        # QSS would be clobbered by qfluentwidgets' own app stylesheet).
        self._apply_app_palette()
        self.changed.emit("dark" if theme == Theme.DARK else "light")

    @property
    def app_qss(self) -> str:
        """Token-driven QSS for non-Fluent Qt widgets (applied on the window).

        Callers (MainWindow.refresh_theme) set this on the window widget so it
        cascades to QToolBar/QToolButton/QLabel/QGroupBox — which the Qt style
        engine draws directly and the app QPalette can't reach.
        """
        return _app_qss(self.tokens)

    def _apply_app_palette(self) -> None:
        """Set the QApplication palette from the current theme tokens.

        Fluent widgets restyle themselves, but plain Qt widgets (QLabel,
        QToolBar, nav-item labels) read QPalette — without this they stay
        dark-on-dark in night mode.
        """
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
        return "dark" if isDarkTheme() else "light"

    @property
    def is_dark(self) -> bool:
        return isDarkTheme()

    @property
    def tokens(self) -> dict:
        """Current token dict (rebuilt when the Fluent theme changes)."""
        dark = isDarkTheme()
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
        target = Theme.DARK if mode == "dark" else Theme.LIGHT
        if qconfig.theme == target:
            return
        setTheme(target, save=False, lazy=False)

    def toggle(self) -> str:
        new = Theme.DARK if not isDarkTheme() else Theme.LIGHT
        setTheme(new, save=False, lazy=False)
        return "dark" if new == Theme.DARK else "light"

    # -- application ----------------------------------------------------------
    def apply(self, app) -> None:
        """Apply the Fluent theme to the QApplication.

        qfluentwidgets owns the global stylesheet; we additionally pin the
        theme colour to Fluent Communication Blue to match the original
        app's identity.
        """
        setThemeColor(QColor(PRIMARY))
        # qfluentwidgets installs its stylesheet lazily when widgets are
        # created; force it now so the very first paint is correct.
        setTheme(Theme.LIGHT, save=False, lazy=False)
        # Set the app palette so non-Fluent widgets pick up theme colours.
        self._apply_app_palette()
        # NOTE: the QSS for non-Fluent widgets (QToolBar/QToolButton/QLabel/
        # QGroupBox) is applied on the MainWindow widget, NOT the QApplication —
        # qfluentwidgets owns the app-level stylesheet and would overwrite it.
        # MainWindow picks it up via the ``changed`` signal → refresh_theme.


theme_manager = ThemeManager()


# ===========================================================================
# Chart theming — QChart/QChartView live on a QGraphicsScene and ignore QSS,
# so this applies the palette directly. Values come from ``theme_manager.tokens``
# which now derives from the live Fluent theme.
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
    """Token-driven QSS for non-Fluent Qt widgets that QPalette can't reach.

    QToolBar/QToolButton (toolbar action buttons), QGroupBox titles, and
    QLabel tool-tip text are drawn by the Qt style engine, which ignores the
    app QPalette for some roles — so they stay dark-on-dark in night mode.
    This stylesheet forces their text colour from the live theme tokens.
    Applied globally via ``QApplication.setStyleSheet`` on every theme switch.
    """
    return (
        f"QToolBar {{ background: transparent; border: none; spacing: 2px; }}"
        f"QToolButton, QToolBar QToolButton {{"
        f"  color: {tk['text']}; background: transparent; border: none;"
        f"  padding: 2px 6px; border-radius: 4px;"
        f"}}"
        f"QToolButton:hover, QToolBar QToolButton:hover {{"
        f"  background: {tk['surface_hover']};"
        f"}}"
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
        f"  background: {tk['window_bg']}; }}"
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
# Legacy inline-style shims — kept so existing imports in sidebar.py /
# main_window.py don't break during the staged migration. The Fluent widget
# migration removes these callers; these functions will be deleted once
# nothing references them. They read the same token dict as before.
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


# ===========================================================================
# Legacy SVG icon shim — replaced by FluentIcon.VIEW / FluentIcon.HIDE in the
# sidebar. Kept temporarily for import compatibility.
# ===========================================================================

def svg_icon(name: str, color: str | None = None, size: int = 20):
    """Return a FluentIcon-based QIcon for 'eye_open'/'eye_closed'.

    Other names return an empty icon. New code should use FluentIcon directly.
    """
    from PySide6.QtGui import QIcon
    from qfluentwidgets import FluentIcon as FI
    mapping = {"eye_open": FI.VIEW, "eye_closed": FI.HIDE}
    fi = mapping.get(name)
    if fi is None:
        return QIcon()
    return fi.icon()


# ===========================================================================
# GlowBackground shim — the real implementation painted ambient glow blobs
# behind the (now-removed) glassmorphism layer. qfluentwidgets' Mica/Acrylic
# background replaces it. We keep a no-op placeholder so MainWindow can be
# migrated in stages without breaking the import.
# ===========================================================================

class GlowBackground(QWidget):
    """Deprecated no-op. qfluentwidgets owns the window background now.

    Instances paint nothing; kept only so ``from theme import GlowBackground``
    continues to work during the staged migration. Will be removed once
    main_window.py stops referencing it.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setEnabled(False)
        self.setFocusPolicy(Qt.FocusPolicy.NoFocus)

    def refresh_theme(self) -> None:  # no-op
        pass
