"""Generate icon.ico from icon.svg — vector paths only (no <text>).

Qt's QSvgRenderer silently drops (or segfaults on) <text> elements in minimal
environments, which is why the previous icon showed empty rectangles. This SVG
uses vector <path> shapes for the B and X, so it renders identically everywhere.

Build pipeline: render each resolution to PNG via QSvgRenderer -> QImage.save,
then assemble the .ico with Pillow (avoids the in-process multi-render crash).
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtCore import QByteArray, QBuffer, QRectF, Qt
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtGui import QImage, QPainter

with open('icon.svg', 'rb') as f:
    svg_bytes = QByteArray(f.read())
renderer = QSvgRenderer(svg_bytes)
assert renderer.isValid(), "icon.svg is not valid SVG"


def render_png(size: int) -> bytes:
    img = QImage(size, size, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing, True)
    p.setRenderHint(QPainter.SmoothPixmapTransform, True)
    renderer.render(p, QRectF(0, 0, size, size))
    p.end()
    path = f'_icon_tmp_{size}.png'
    ok = img.save(path, 'PNG')
    assert ok, f'failed to save {path}'
    with open(path, 'rb') as f:
        return f.read()


# Preview + multi-resolution .ico
sizes = [16, 24, 32, 48, 64, 128, 256]

# Full-size preview
preview = render_png(256)
with open('icon_preview.png', 'wb') as f:
    f.write(preview)
print(f"Wrote icon_preview.png ({len(preview)} bytes)")

import io
from PIL import Image
# Pillow ICO: load ONE full-res frame and let it auto-generate all sizes from it.
master = Image.open(io.BytesIO(render_png(256))).convert('RGBA')
master.save('icon.ico', format='ICO', sizes=[(s, s) for s in sizes])
print(f"Wrote icon.ico with sizes {sizes}")

# cleanup temp pngs
for s in [16, 24, 32, 48, 64, 128, 256]:
    try:
        os.remove(f'_icon_tmp_{s}.png')
    except OSError:
        pass
