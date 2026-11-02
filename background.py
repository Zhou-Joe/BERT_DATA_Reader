"""Cancellable Qt workers. Only queued signals cross back to the GUI thread."""
from __future__ import annotations

from threading import Event

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot


class Cancelled(Exception):
    pass


class CancellationToken:
    def __init__(self):
        self._event = Event()

    def cancel(self):
        self._event.set()

    def check(self):
        if self._event.is_set():
            raise Cancelled()


class _Signals(QObject):
    progress = Signal(str, int, int, str)
    result = Signal(str, int, object)
    error = Signal(str, int, str)
    finished = Signal(str, int)


class _Worker(QRunnable):
    def __init__(self, kind, ticket, function, token):
        super().__init__()
        self.kind, self.ticket = kind, ticket
        self.function, self.token = function, token
        self.signals = _Signals()

    def run(self):
        try:
            self.token.check()
            result = self.function(self.token, self._progress)
            self.token.check()
            self.signals.result.emit(self.kind, self.ticket, result)
        except Cancelled:
            pass
        except Exception as error:
            self.signals.error.emit(self.kind, self.ticket, str(error))
        finally:
            self.signals.finished.emit(self.kind, self.ticket)

    def _progress(self, percent, message):
        self.token.check()
        self.signals.progress.emit(self.kind, self.ticket, percent, message)


class JobManager(QObject):
    result = Signal(str, object)
    error = Signal(str, str)
    progress = Signal(str, int, str)
    changed = Signal()
    drained = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(2)
        self._serial = 0
        self._current = {}
        self._workers = {}

    def start(self, kind, function):
        self.cancel(kind)
        self._serial += 1
        ticket = self._serial
        worker = _Worker(kind, ticket, function, CancellationToken())
        self._workers[ticket] = worker
        self._current[kind] = ticket
        worker.signals.result.connect(self._on_result)
        worker.signals.error.connect(self._on_error)
        worker.signals.progress.connect(self._on_progress)
        worker.signals.finished.connect(self._on_finished)
        self.pool.start(worker)
        self.changed.emit()
        return ticket

    def cancel(self, kind=None):
        kinds = list(self._current) if kind is None else [kind]
        for name in kinds:
            ticket = self._current.pop(name, None)
            if ticket in self._workers:
                self._workers[ticket].token.cancel()
        self.changed.emit()

    def busy(self, kind=None):
        return bool(self._current) if kind is None else kind in self._current

    @property
    def active_count(self):
        return len(self._workers)

    @Slot(str, int, object)
    def _on_result(self, kind, ticket, result):
        if self._current.get(kind) == ticket:
            self.result.emit(kind, result)

    @Slot(str, int, str)
    def _on_error(self, kind, ticket, message):
        if self._current.get(kind) == ticket:
            self.error.emit(kind, message)

    @Slot(str, int, int, str)
    def _on_progress(self, kind, ticket, percent, message):
        if self._current.get(kind) == ticket:
            self.progress.emit(kind, percent, message)

    @Slot(str, int)
    def _on_finished(self, kind, ticket):
        if self._current.get(kind) == ticket:
            self._current.pop(kind)
        self._workers.pop(ticket, None)
        self.changed.emit()
        if not self._workers:
            self.drained.emit()
