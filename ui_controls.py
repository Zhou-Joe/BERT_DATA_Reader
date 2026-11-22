"""Shared controls for the Qt 6 workspace."""

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QMessageBox, QPushButton
import qtawesome

from theme import theme_manager


def icon(name, color=None):
    return qtawesome.icon('ph.' + name, color=color or theme_manager.t('text'),
                          color_disabled=theme_manager.t('text_disabled'))


def icon_action(name, text, parent):
    action = QAction(icon(name), text, parent)
    action.setProperty('glyph', name)
    return action


class PrimaryButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setObjectName('primaryButton')


def notice(parent, title, content, duration, severity):
    message = QMessageBox(severity, title, content, QMessageBox.StandardButton.Ok, parent)
    message.setTextFormat(Qt.TextFormat.PlainText)
    message.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    message.setWindowModality(Qt.WindowModality.NonModal)
    message.show()
    if duration > 0:
        timer = QTimer(message)
        timer.setSingleShot(True)
        timer.timeout.connect(message.close)
        timer.start(duration)
    return message


def warning(*, parent, title, content, duration):
    return notice(parent, title, content, duration, QMessageBox.Icon.Warning)


def error(*, parent, title, content, duration):
    return notice(parent, title, content, duration, QMessageBox.Icon.Critical)
