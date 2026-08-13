# coding: utf-8
"""
Custom, function-specific navigation icons for BERT Reader.

Each tab gets an SVG icon that reflects what its chart actually shows, instead
of the generic Fluent icons (HOME / ACCEPT / SPEED_HIGH). Icons are drawn from
inline SVG so no resource files are needed, and they recolour automatically
with the light/dark theme (the SVG fill is chosen per theme at render time).

Implemented as FluentIconBase subclasses so they drop straight into
``addSubInterface(..., icon=MyIcon.TREND, ...)`` and benefit from qfluentwidgets'
``drawIcon`` / FluentIconEngine machinery.
"""

from __future__ import annotations

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap, QImage
from PySide6.QtSvg import QSvgRenderer

from qfluentwidgets.common.icon import FluentIconBase
from qfluentwidgets.common.config import Theme, isDarkTheme

# Icon colour per theme. Drawn as a single fill so the whole glyph recolours.
_LIGHT = "#000000"   # near-black on the light Fluent surface
_DARK = "#FFFFFF"    # white on the dark Fluent surface


def _color(theme: Theme = Theme.AUTO) -> str:
    if theme == Theme.AUTO:
        return _DARK if isDarkTheme() else _LIGHT
    return _DARK if theme == Theme.DARK else _LIGHT


class SvgIcon(FluentIconBase):
    """A FluentIconBase backed by an inline SVG template.

    Subclasses set ``SVG`` — an SVG body using ``{color}`` as the fill
    placeholder. At render time the current theme's colour is substituted in,
    so the icon recolours correctly on a light/dark switch.
    """

    SVG: str = ""

    def path(self, theme: Theme = Theme.AUTO) -> str:
        # Return the themed SVG markup (tagged .svg so callers that special-case
        # the suffix still treat it as SVG content).
        return self.svg(theme)

    def svg(self, theme: Theme = Theme.AUTO) -> str:
        """Return the full SVG document with the theme colour applied."""
        return self.SVG.replace("{color}", _color(theme))

    def render(self, painter: QPainter, rect: QRectF,
               theme: Theme = Theme.AUTO, indexes=None, **attributes) -> None:
        renderer = QSvgRenderer(QByteArray(self.svg(theme).encode("utf-8")))
        renderer.render(painter, rect)

    def icon(self, theme: Theme = Theme.AUTO, color=None) -> QIcon:
        # Rasterise the SVG into a pixmap big enough to stay crisp at nav size.
        svg = self.svg(theme)
        renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
        pm = QPixmap(64, 64)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        renderer.render(p, QRectF(0, 0, 64, 64))
        p.end()
        return QIcon(pm)


# --------------------------------------------------------------------------
# SVG templates — a 24x24 viewBox, single {color} fill, stroke-free shapes.
# Each is designed to evoke the chart it represents.
# --------------------------------------------------------------------------

# Trend: three overlapping time-series lines (the 3-axis x/y/z traces).
_TREND_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
    'fill="none" stroke="{color}" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round">'
    # baseline time axis
    '<path d="M3 20 H21" stroke-width="1.4" opacity="0.55"/>'
    # three traces of different wavelength/phase
    '<path d="M3 14 Q6 9 9 14 T15 14 T21 14"/>'
    '<path d="M3 10 Q5 5 7 10 T11 10 T15 10 T21 10" opacity="0.8"/>'
    '<path d="M3 6 Q6.5 2.5 10 6 T17 6 T21 6" opacity="0.6"/>'
    '</svg>'
)

# ASTM Sustained: a horizontal time axis with a step-decaying limit line
# (the characteristic g-vs-dt limit that falls as dt grows).
_SUST_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
    'fill="none" stroke="{color}" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M3 20 H21" stroke-width="1.4" opacity="0.55"/>'   # dt axis
    '<path d="M3 5 H8 V9 H13 V12 H18 V15 H21"/>'                 # step-down limit
    '<path d="M3 16 Q9 13 15 16.5 T21 15.5" opacity="0.65"/>'    # a measured curve
    '</svg>'
)

# ASTM Eggplot: an egg/oval boundary loop (the 2D acceleration envelope).
_EGG_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
    'fill="none" stroke="{color}" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round">'
    # egg outline: rounded bottom, slightly pointier top
    '<path d="M12 3 C7 3 5 9 5 14 C5 18 8 21 12 21 C16 21 19 18 19 14 '
    'C19 9 17 3 12 3 Z"/>'
    # crosshair axes through the egg
    '<path d="M3 14 H21" stroke-width="1.1" opacity="0.5"/>'
    '<path d="M12 1 V23" stroke-width="1.1" opacity="0.5"/>'
    '</svg>'
)

# GB Contour: three nested closed loops (the 0.1s / 0.2s / 0.05s dt-contours).
_GB_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
    'fill="none" stroke="{color}" stroke-linecap="round" '
    'stroke-linejoin="round">'
    '<ellipse cx="12" cy="12" rx="9" ry="7" stroke-width="1.8"/>'
    '<ellipse cx="12" cy="12" rx="6" ry="4.6" stroke-width="1.5" opacity="0.75"/>'
    '<ellipse cx="12" cy="12" rx="3" ry="2.3" stroke-width="1.3" opacity="0.55"/>'
    '</svg>'
)

# Acc Zone: a 2x2 zone-classification grid (the four ride-intensity zones).
_ACCZONE_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
    'fill="none" stroke="{color}" stroke-width="1.7" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<rect x="3" y="3" width="18" height="18" rx="1.5"/>'
    '<path d="M12 3 V21"/>'   # vertical split
    '<path d="M3 12 H21"/>'   # horizontal split
    # filled corners hint at increasing zone intensity (denser = higher zone)
    '<rect x="3" y="3" width="9" height="9" fill="{color}" opacity="0.12"/>'
    '<rect x="12" y="3" width="9" height="9" fill="{color}" opacity="0.30"/>'
    '<rect x="3" y="12" width="9" height="9" fill="{color}" opacity="0.50"/>'
    '<rect x="12" y="12" width="9" height="9" fill="{color}" opacity="0.75"/>'
    '</svg>'
)

# Z Reversal: a vertical (z) sign-flipping waveform — sharp up/down spikes.
_ZREV_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
    'fill="none" stroke="{color}" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M3 12 H21" stroke-width="1.3" opacity="0.5"/>'   # zero line
    '<path d="M4 12 L7 4 L10 12 L13 20 L16 12 L19 5"/>'        # vertical zigzag
    '</svg>'
)

# X/Y Reversal: a horizontal sign-flip pair — a left/right zigzag above + below.
_XYREV_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
    'fill="none" stroke="{color}" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M3 12 H21" stroke-width="1.3" opacity="0.5"/>'   # zero line
    '<path d="M4 7 L9 7 L9 17 L14 17 L14 7 L19 7"/>'           # square-wave reversal
    '<path d="M4 17 L4 7" opacity="0.6"/><path d="M19 7 L19 17" opacity="0.6"/>'
    '</svg>'
)


class TrendIcon(SvgIcon):
    SVG = _TREND_SVG


class SustIcon(SvgIcon):
    SVG = _SUST_SVG


class EggIcon(SvgIcon):
    SVG = _EGG_SVG


class GBIcon(SvgIcon):
    SVG = _GB_SVG


class AccZoneIcon(SvgIcon):
    SVG = _ACCZONE_SVG


class ZReversalIcon(SvgIcon):
    SVG = _ZREV_SVG


class XYReversalIcon(SvgIcon):
    SVG = _XYREV_SVG


# Single instances — use these as ``icon=NavIcon.TREND`` in addSubInterface.
class NavIcon:
    TREND = TrendIcon()
    SUST = SustIcon()
    EGG = EggIcon()
    GB = GBIcon()
    ACC_ZONE = AccZoneIcon()
    Z_REVERSAL = ZReversalIcon()
    XY_REVERSAL = XYReversalIcon()
