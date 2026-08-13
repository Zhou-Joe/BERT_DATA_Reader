# coding: utf-8
"""
BERT_READER X — Application entry point (PySide6 + Qt Charts).

Usage:
    python main.py
"""

import sys
import os

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QIcon
from PySide6.QtWidgets import QApplication

from theme import theme_manager, FONT_FAMILY, FONT_MD
from main_window import MainWindow


def main() -> int:
    # PySide6 enables high-DPI scaling by default; ensure AA is on too.
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    # NOTE: deliberately not calling setStyle("Fusion") — qfluentwidgets owns
    # the global stylesheet and Fusion would clash with it.

    # App-wide font (Segoe UI Variable on Win11, graceful fallback elsewhere)
    font = QFont()
    for fam in ("Segoe UI Variable Text", "Segoe UI", "Microsoft YaHei UI"):
        font.setFamilies([fam, "Segoe UI", "Microsoft YaHei UI", "Verdana"])
        break
    font.setPointSize(FONT_MD)
    app.setFont(font)

    # Apply the Fluent theme (light by default).
    theme_manager.apply(app)

    window = MainWindow()

    # Load the app icon (SVG). PySide6's QIcon reads SVG natively.
    ico_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.svg")
    if os.path.exists(ico_path):
        window.setWindowIcon(QIcon(ico_path))

    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
