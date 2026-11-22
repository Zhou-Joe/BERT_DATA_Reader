"""React desktop shell. Interactive chart widgets stay in-process with their data."""
from pathlib import Path

from PySide6.QtCore import QObject, QRect, Qt, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QColor
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QWidget

from main_window import MainWindow
from theme import theme_manager


class WorkspacePage(QWebEnginePage):
    def acceptNavigationRequest(self, url, kind, main_frame):
        return main_frame and url == self.parent().entry_url


class WorkspaceBridge(QObject):
    changed = Signal('QVariantMap')

    def __init__(self, window, surface):
        super().__init__(surface)
        self.window = window
        self.surface = surface
        self.previous = None
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.publish)
        model = window.select_data.model()
        for signal in (window.stackedWidget.currentChanged, window.select_data.currentIndexChanged,
                       model.rowsInserted, model.rowsRemoved, model.modelReset, model.dataChanged,
                       window.cutoff_spin.valueChanged, window.select_cond.currentIndexChanged,
                       window.select_restraint.currentIndexChanged, window.height_input.valueChanged,
                       window._analyze_window_spin.valueChanged, window._prone_btn.toggled,
                       window._analyze_btn.toggled, window._library_toggle.toggled,
                       window.statusBar().messageChanged, window._jobs.changed, theme_manager.changed):
            signal.connect(self.schedule)

    def schedule(self, *_):
        self.timer.start(0)

    @Slot(result='QVariantMap')
    def snapshot(self):
        w = self.window
        def options(combo):
            return [combo.itemText(i) for i in range(combo.count())]

        return dict(
            views=[w.stackedWidget.widget(i).windowTitle() for i in range(w.stackedWidget.count())],
            view=w.stackedWidget.currentIndex(), sources=options(w.select_data),
            source=w.select_data.currentIndex(), formats=[name for _, name, _ in w._import_formats],
            restraints=options(w.select_restraint), restraint=w.select_restraint.currentIndex(),
            conditions=options(w.select_cond), condition=w.select_cond.currentIndex(),
            height=w.height_input.value(), period=w._analyze_window_spin.value(),
            cutoff=w.cutoff_spin.value(), prone=w._prone_btn.isChecked(),
            library=w._library_toggle.isChecked(), dark=theme_manager.is_dark,
            state=w._processing_state, status=w._workspace_title.text(),
            update=w._generate_button.isEnabled(), canAnalyze=w._analyze_btn.isEnabled(),
            analyzing=w._analyze_btn.isChecked(), canImport=w._import_buttons[0].isEnabled(),
            files={name: action.isEnabled() for name, action in w._data_actions.items()})

    def publish(self):
        state = self.snapshot()
        if state != self.previous:
            self.previous = state
            self.changed.emit(state)

    @Slot(int)
    def navigate(self, index):
        stack = self.window.stackedWidget
        if 0 <= index < stack.count():
            stack.setCurrentIndex(index)
            self.publish()

    @Slot(int)
    def openFormat(self, index):
        w = self.window
        if 0 <= index < len(w._import_formats) and w._import_buttons[0].isEnabled():
            QTimer.singleShot(0, w._import_formats[index][2])

    @Slot(str)
    def command(self, name):
        w = self.window
        if name == 'theme':
            theme_manager.toggle()
        elif name == 'library':
            w._library_toggle.click()
        elif name == 'update':
            w._generate_button.click()
        elif name == 'analyze':
            w._analyze_btn.click()
        elif name in w._data_actions and w._data_actions[name].isEnabled():
            QTimer.singleShot(0, w._data_actions[name].trigger)

    @Slot(str, str)
    def setting(self, name, value):
        w = self.window
        choices = {'restraint': w.select_restraint, 'condition': w.select_cond, 'source': w.select_data}
        numbers = {'height': (w.height_input, int), 'cutoff': (w.cutoff_spin, int),
                   'period': (w._analyze_window_spin, float)}
        try:
            if name in choices:
                combo = choices[name]
                index = int(value)
                if 0 <= index < combo.count():
                    combo.setCurrentIndex(index)
            elif name in numbers:
                spin, convert = numbers[name]
                number = convert(value)
                if spin.minimum() <= number <= spin.maximum():
                    spin.setValue(number)
            elif name == 'prone' and value in ('true', 'false'):
                w._prone_btn.setChecked(value == 'true')
        except (ValueError, OverflowError):
            return

    @Slot(int, int, int, int, int, int)
    def chartArea(self, x, y, width, height, viewport_width, viewport_height):
        if min(width, height, viewport_width, viewport_height) <= 0:
            return
        surface = self.surface
        sx, sy = surface.width() / viewport_width, surface.height() / viewport_height
        rect = QRect(round(x * sx), round(y * sy), round(width * sx), round(height * sy))
        surface.deck.setGeometry(rect.intersected(surface.rect()))
        surface.deck.show()
        surface.deck.raise_()


class DesktopSurface(QWidget):
    def __init__(self, window):
        super().__init__(window)
        entry = Path(__file__).parent / 'frontend' / 'dist' / 'index.html'
        if not entry.is_file():
            raise RuntimeError('Build the desktop frontend in frontend/ with npm ci and npm run build.')
        self.entry_url = QUrl.fromLocalFile(str(entry.resolve()))
        self.web = QWebEngineView(self)
        self.web.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        page = WorkspacePage(self)
        self.web.setPage(page)
        page.setBackgroundColor(QColor('#f8f9f5'))
        page.settings().setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, False)
        self.deck = window.chart_workspace
        self.deck.setParent(self)
        self.deck.hide()
        self.bridge = WorkspaceBridge(window, self)
        self.channel = QWebChannel(self)
        self.channel.registerObject('workspace', self.bridge)
        page.setWebChannel(self.channel)
        self.web.load(self.entry_url)

    def resizeEvent(self, event):
        self.web.setGeometry(self.rect())
        super().resizeEvent(event)


class DesktopWindow(MainWindow):
    def __init__(self):
        super().__init__()
        # Retain the action and settings model owned by the original controls.
        controls = self.takeCentralWidget()
        controls.setParent(self)
        controls.hide()
        self.surface = DesktopSurface(self)
        self.setCentralWidget(self.surface)
        self.setWindowTitle('BERT Signal Lab')

    def _set_processing_state(self, state, message):
        super()._set_processing_state(state, message)
        if hasattr(self, 'surface'):
            self.surface.bridge.schedule()

    def _update_job_controls(self):
        super()._update_job_controls()
        if hasattr(self, 'surface'):
            self.surface.bridge.schedule()

    def _find_recording(self):
        if self.datalist:
            self.surface.bridge.navigate(self.stackedWidget.currentIndex())
        super()._find_recording()
