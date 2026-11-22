"""Layout and visual identity for the BERT signal workbench."""

from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QMainWindow, QPushButton, QScrollArea, QSizePolicy,
    QStackedWidget, QVBoxLayout, QWidget,
)
from ui_controls import icon


def caption(text, name='caption'):
    label = QLabel(text)
    label.setObjectName(name)
    return label


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
        rail.setFixedWidth(210)
        shell.addWidget(rail)
        rail_layout = QVBoxLayout(rail)
        rail_layout.setContentsMargins(12, 18, 12, 12)
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
        rail_layout.addSpacing(5)
        self._nav_buttons = []
        nav = QWidget()
        self._nav_layout = QVBoxLayout(nav)
        self._nav_layout.setContentsMargins(0, 0, 0, 0)
        self._nav_layout.setSpacing(2)
        rail_layout.addWidget(nav)
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
        rail_layout.addWidget(scroll, 1)

        desk = QVBoxLayout()
        desk.setContentsMargins(0, 0, 0, 0)
        desk.setSpacing(0)
        shell.addLayout(desk, 1)
        self._top_bar = QFrame()
        self._top_bar.setObjectName('topBar')
        self._top_bar_layout = QVBoxLayout(self._top_bar)
        self._top_bar_layout.setContentsMargins(18, 10, 18, 10)
        self._top_bar_layout.setSpacing(7)
        desk.addWidget(self._top_bar)
        self.chart_workspace = QWidget()
        chart_layout = QVBoxLayout(self.chart_workspace)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        chart_layout.setSpacing(0)
        self._chart_tools_layout = QVBoxLayout()
        self._chart_tools_layout.setContentsMargins(16, 5, 16, 0)
        self._chart_tools_layout.setSpacing(4)
        chart_layout.addLayout(self._chart_tools_layout)
        self._work_area = QHBoxLayout()
        self._work_area.setContentsMargins(0, 0, 0, 0)
        self._work_area.setSpacing(0)
        chart_layout.addLayout(self._work_area, 1)
        desk.addWidget(self.chart_workspace, 1)
        self.stackedWidget = QStackedWidget()
        self._work_area.addWidget(self.stackedWidget, 1)
        self.statusBar().setSizeGripEnabled(False)

    def _register_view(self, page, glyph_name, title):
        index = len(self._nav_buttons)
        if index in self.GROUPS:
            self._nav_layout.addWidget(caption(self.GROUPS[index], 'railCaption'))
        button = QPushButton(title)
        button.setObjectName('viewLink')
        button.setIcon(icon(glyph_name, '#DBE4F9'))
        button.setIconSize(QSize(18, 18))
        button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        button.setMinimumHeight(33)
        button.setCheckable(True)
        button.setAutoExclusive(True)
        button.clicked.connect(lambda: self.stackedWidget.setCurrentIndex(index))
        self._nav_buttons.append(button)
        self._nav_layout.addWidget(button)
        self.stackedWidget.addWidget(page)
