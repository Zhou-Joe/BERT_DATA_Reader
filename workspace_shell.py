"""Layout and visual identity for the BERT signal workbench."""

from math import cos, sin, tau
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QMainWindow, QScrollArea, QSizePolicy,
    QStackedWidget, QToolButton, QVBoxLayout, QWidget,
)
from qfluentwidgets import Theme
from qfluentwidgets.window.stacked_widget import StackedWidget


def caption(text, name='caption'):
    label = QLabel(text)
    label.setObjectName(name)
    return label


class AtlasPlate(QWidget):
    """Decorative contour engraving, independent of measurement data."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(116)
        self.setMaximumHeight(250)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAccessibleName('Decorative contour illustration')
        self._contours = self._build_contours()

    @staticmethod
    def _build_contours():
        contours = []
        for ring in range(1, 37):
            path = QPainterPath()
            for sample in range(145):
                angle = tau * sample / 144
                radius = (5 + ring * 5.3) * (1 + .17 * cos(3 * angle + ring * .045) + .1 * sin(2 * angle))
                point = QPointF(1.58 * radius * cos(angle) - ring * .9, .70 * radius * sin(angle))
                if sample == 0:
                    path.moveTo(point)
                else:
                    path.lineTo(point)
            path.closeSubpath()
            contours.append(path)
        return contours

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width, height = self.width(), self.height()
        painter.fillRect(self.rect(), QColor('#123D39'))
        painter.setPen(QPen(QColor('#23504B'), 1))
        for column in range(14, width, 24):
            for row in range(12, height, 24):
                painter.drawPoint(column, row)
        scale = min(width / 400, height / 170)
        painter.save()
        painter.translate(width * .66, height * .47)
        painter.scale(scale, scale)
        painter.rotate(-19)
        for ring, path in enumerate(self._contours, 1):
            pen = QPen(QColor('#DBF58B' if ring in (7, 19) else '#5D9A89'), 1)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.drawPath(path)
        painter.restore()
        painter.setPen(QColor('#D6E8DF'))
        font = painter.font()
        font.setFamily('Consolas')
        font.setPixelSize(11)
        painter.setFont(font)
        painter.drawText(QRectF(18, 12, 210, 20), Qt.AlignmentFlag.AlignLeft, 'FIG. 01   /   SIGNAL ATLAS')
        painter.end()


class WorkspaceWindow(QMainWindow):
    GROUPS = {0: 'SIGNAL', 1: 'STANDARDS', 5: 'REVERSALS'}

    def __init__(self):
        super().__init__()
        root = QWidget()
        root.setObjectName('workbench')
        self.setCentralWidget(root)
        shell = QHBoxLayout(root)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)
        rail = QFrame()
        rail.setObjectName('rail')
        rail.setFixedWidth(204)
        shell.addWidget(rail)
        rail_layout = QVBoxLayout(rail)
        rail_layout.setContentsMargins(14, 23, 14, 16)
        rail_layout.setSpacing(8)
        identity = QHBoxLayout()
        identity.setSpacing(10)
        mark = QLabel()
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mark.setFixedSize(38, 38)
        logo = QIcon(str(Path(__file__).with_name('icon.svg')))
        mark.setPixmap(logo.pixmap(mark.size(), mark.devicePixelRatioF()))
        self.setWindowIcon(logo)
        identity.addWidget(mark)
        signature = QVBoxLayout()
        signature.setSpacing(0)
        signature.addWidget(caption('BERT', 'brandName'))
        signature.addWidget(caption('SIGNAL LAB', 'brandTag'))
        identity.addLayout(signature)
        identity.addStretch()
        rail_layout.addLayout(identity)
        rail_layout.addSpacing(17)
        self._rail_stack = QStackedWidget()
        self._nav_buttons = []
        nav = QWidget()
        self._nav_layout = QVBoxLayout(nav)
        self._nav_layout.setContentsMargins(0, 0, 0, 0)
        self._nav_layout.setSpacing(2)
        self._nav_layout.addStretch()
        self._rail_stack.addWidget(nav)
        self._ctrl_panel = QWidget()
        self._ctrl_panel.setObjectName('ctrlPanel')
        self._ctrl_panel_layout = QVBoxLayout(self._ctrl_panel)
        self._ctrl_panel_layout.setContentsMargins(9, 8, 9, 10)
        scroll = QScrollArea()
        scroll.setObjectName('analysisScroll')
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._ctrl_panel)
        self._rail_stack.addWidget(scroll)
        rail_layout.addWidget(self._rail_stack, 1)
        settings = QToolButton()
        settings.setObjectName('railSettings')
        settings.setText('Analysis settings  →')
        settings.setCheckable(True)
        settings.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        settings.setMinimumHeight(38)
        settings.toggled.connect(lambda checked: self._rail_stack.setCurrentIndex(int(checked)))
        settings.toggled.connect(lambda checked: settings.setText('←  Analysis views' if checked else 'Analysis settings  →'))
        rail_layout.addWidget(settings)

        desk = QVBoxLayout()
        desk.setContentsMargins(0, 0, 0, 0)
        desk.setSpacing(0)
        shell.addLayout(desk, 1)
        self._top_bar = QFrame()
        self._top_bar.setObjectName('topBar')
        self._top_bar_layout = QVBoxLayout(self._top_bar)
        self._top_bar_layout.setContentsMargins(20, 14, 20, 12)
        self._top_bar_layout.setSpacing(12)
        desk.addWidget(self._top_bar)
        self._work_area = QHBoxLayout()
        self._work_area.setContentsMargins(0, 0, 0, 0)
        self._work_area.setSpacing(0)
        desk.addLayout(self._work_area, 1)
        self.stackedWidget = StackedWidget()
        self.stackedWidget.setAnimationEnabled(False)
        self._work_area.addWidget(self.stackedWidget, 1)
        self.statusBar().setSizeGripEnabled(False)

    def _register_view(self, page, icon, title):
        index = len(self._nav_buttons)
        if index in self.GROUPS:
            self._nav_layout.insertWidget(self._nav_layout.count() - 1,
                                          caption(self.GROUPS[index], 'railCaption'))
        button = QToolButton()
        button.setObjectName('viewLink')
        button.setAccessibleName(title)
        row = QHBoxLayout(button)
        row.setContentsMargins(12, 0, 10, 0)
        row.setSpacing(10)
        glyph = QLabel()
        glyph.setPixmap(icon.icon(Theme.DARK).pixmap(QSize(18, 18), self.devicePixelRatioF()))
        row.addWidget(glyph)
        row.addWidget(caption(title, 'navTitle'), 1)
        row.addWidget(caption(f'{index + 1:02d}', 'navIndex'))
        for label in button.findChildren(QLabel):
            label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        button.setMinimumHeight(37)
        button.setCheckable(True)
        button.setAutoExclusive(True)
        button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(index))
        self._nav_buttons.append(button)
        self._nav_layout.insertWidget(self._nav_layout.count() - 1, button)
        self.stackedWidget.addWidget(page)
