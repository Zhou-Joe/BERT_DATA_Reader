"""Print signed lateral measurements and their limits on one set of axes."""
import math

from PySide6.QtCharts import QScatterSeries
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPolygonF


def lateral_traces(chart, restraint):
    """Copy the actual plotted samples and limits; don't run the analysis again."""
    directions = {"+": {}, "-": {}}
    violations = {"+": [], "-": []}
    for series in chart.series():
        if not series.isVisible():
            continue
        if isinstance(series, QScatterSeries):
            for point in series.points():
                violations["-" if point.y() < 0 else "+"].append((point.x(), point.y()))
            continue
        for sign, traces in directions.items():
            names = {f"Measured ay ({sign})": "Measured", f"ASTM ay ({sign})": "ASTM limit"}
            if restraint != "None":
                names[f"{restraint} ({sign})"] = "Restraint limit"
            key = names.get(series.name())
            if key:
                traces[key] = [(p.x(), p.y()) for p in series.points()]
    for traces in directions.values():
        required = ["Measured", "ASTM limit"]
        if restraint != "None":
            required.append("Restraint limit")
        if any(not traces.get(k) for k in required):
            raise ValueError("The lateral chart is missing a measured trace or selected limit.")
    return directions, violations


def render_lateral_figure(chart, restraint, path):
    directions, violations = lateral_traces(chart, restraint)
    all_points = [p for traces in directions.values() for pts in traces.values() for p in pts]
    duration = max(t for t, _ in all_points)
    peak = max(abs(v) for _, v in all_points) * 1.04
    rough_step = max(0.5, peak / 4)
    decade = 10 ** math.floor(math.log10(rough_step))
    tick_step = next(k * decade for k in (1, 2, 2.5, 5, 10) if k * decade >= rough_step)
    ticks_per_side = max(1, math.ceil(peak / tick_step))
    amplitude = ticks_per_side * tick_step
    image = QImage(1080, 590, QImage.Format.Format_RGB32)
    image.fill(QColor("white"))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    left, right, bottom = 88, 1044, 510
    align = Qt.AlignmentFlag
    dashed = Qt.PenStyle.DashLine
    restraint_pen = QPen(QColor("#774391"), 3, Qt.PenStyle.DotLine)
    measured_colours = {
        series.name(): series.pen().color()
        for series in chart.series()
        if series.name().startswith("Measured ay ")
    }
    pens = {
        ("Measured", "+"): QPen(measured_colours["Measured ay (+)"], 3),
        ("Measured", "-"): QPen(measured_colours["Measured ay (-)"], 3, dashed),
        ("ASTM limit", "+"): QPen(QColor("#BF252A"), 3),
        ("ASTM limit", "-"): QPen(QColor("#BF252A"), 3, dashed),
        ("Restraint limit", "+"): restraint_pen,
        ("Restraint limit", "-"): restraint_pen,
    }
    legend = [
        ("Measured +ay (right)", pens["Measured", "+"]),
        ("Measured −ay (left)", pens["Measured", "-"]),
        ("ASTM upper limit", pens["ASTM limit", "+"]),
        ("ASTM lower limit", pens["ASTM limit", "-"]),
    ]
    if restraint != "None":
        legend.append(("Restraint limits (±)", restraint_pen))
    if any(violations.values()):
        legend.append(("Exceedance", None))

    def font(size):
        f = QFont("Segoe UI")
        f.setPixelSize(size)
        painter.setFont(f)

    def text(rect, value, flags, color="#233544"):
        painter.setPen(QColor(color))
        painter.drawText(QRectF(*rect), flags, value)

    def mark_exceedance(center):
        painter.setPen(QPen(QColor("white"), 1.2))
        painter.setBrush(QColor("#CC2834"))
        painter.drawEllipse(center, 5, 5)

    try:
        font(20)
        legend_x, legend_y = left, 24
        for label, pen in legend:
            label_width = painter.fontMetrics().horizontalAdvance(label)
            if legend_x + 40 + label_width > right:
                legend_x, legend_y = left, legend_y + 32
            if pen is None:
                mark_exceedance(QPointF(legend_x + 15, legend_y))
            else:
                painter.setPen(pen)
                painter.drawLine(QPointF(legend_x, legend_y), QPointF(legend_x + 32, legend_y))
            text((legend_x + 40, legend_y - 14, label_width + 3, 28), label,
                 align.AlignLeft | align.AlignVCenter)
            legend_x += 60 + label_width

        top = legend_y + 34

        def pos(seconds, acceleration):
            return QPointF(left + (right - left) * seconds / duration,
                           top + (bottom - top) * (amplitude - acceleration) / (2 * amplitude))

        font(19)
        for i in range(-ticks_per_side, ticks_per_side + 1):
            v = i * tick_step
            y = pos(0, v).y()
            painter.setPen(QPen(QColor("#CED8E0"), 1))
            painter.drawLine(QPointF(left, y), QPointF(right, y))
            text((0, y - 14, left - 12, 28), f"{v:g}", align.AlignRight | align.AlignVCenter)
        for i in range(8):
            t = i * duration / 7
            x = pos(t, 0).x()
            painter.setPen(QPen(QColor("#DEE5EA"), 1))
            painter.drawLine(QPointF(x, top), QPointF(x, bottom))
            text((x - 22, bottom + 4, 44, 28), f"{t:g}", align.AlignCenter)
        painter.setPen(QPen(QColor("#71818E"), 1.5))
        painter.drawLine(QPointF(left, top), QPointF(left, bottom))
        painter.drawLine(pos(0, 0), pos(duration, 0))
        painter.save()
        painter.translate(21, (top + bottom) / 2)
        painter.rotate(-90)
        text((-150, -15, 300, 30), "Lateral acceleration, ay (g)", align.AlignCenter)
        painter.restore()
        text((left, bottom + 37, right - left, 31), "Sustained duration (s)", align.AlignCenter)

        painter.setClipRect(QRectF(left - 5, top - 5, right - left + 10, bottom - top + 10))
        for kind in ("ASTM limit", "Restraint limit", "Measured"):
            for sign, traces in directions.items():
                if kind in traces:
                    painter.setPen(pens[kind, sign])
                    painter.drawPolyline(QPolygonF([pos(t, v) for t, v in traces[kind]]))
        for points in violations.values():
            for t, v in points:
                mark_exceedance(pos(t, v))
    finally:
        painter.end()
    if not image.save(path, "PNG"):
        raise OSError("Could not save the lateral sustained figure")
