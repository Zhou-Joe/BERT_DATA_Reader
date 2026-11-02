"""Render all analysis views for visual review without opening a desktop window.

Example: python tools/review_gui.py --sample ../A3_BEFORE.csv --out ../gui_review
"""
import argparse
import os
from pathlib import Path
import sys
from time import perf_counter

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from main_window import MainWindow
from theme import theme_manager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    app = QApplication([])
    font = Path('C:/Windows/Fonts/segoeui.ttf')
    if font.exists():
        QFontDatabase.addApplicationFont(str(font))
    app.setFont(QFont('Segoe UI', 10))
    theme_manager.apply(app)
    window = MainWindow()
    window.stackedWidget.setAnimationEnabled(False)
    window.show()
    app.processEvents()
    def wait_for_jobs(timeout=30000):
        loop = QEventLoop()
        poll = QTimer(loop)
        poll.setInterval(10)
        def idle():
            return not (window._jobs.active_count or window._refresh_timer.isActive()
                        or window._cutoff_timer.isActive() or window._import_commit is not None)
        poll.timeout.connect(lambda: loop.quit() if idle() else None)
        deadline = QTimer(loop)
        deadline.setSingleShot(True)
        deadline.timeout.connect(loop.quit)
        poll.start()
        deadline.start(timeout)
        loop.exec()
        if not idle() or window._processing_state == 'error':
            raise RuntimeError(window.statusBar().currentMessage())
        app.processEvents()

    ticks = []
    heartbeat = QTimer()
    heartbeat.setInterval(10)
    heartbeat.timeout.connect(lambda: ticks.append(perf_counter()))
    heartbeat.start()
    start = perf_counter()
    window._start_import(args.sample, raw=True)
    wait_for_jobs()
    if len(window.datalist) != 2 * len(args.sample):
        raise RuntimeError('Not every selected raw file produced both signal groups')
    print(f'Imported {len(args.sample)} files into {len(window.datalist)} recordings')
    print(f'Background import, trend plot and preparation: {perf_counter() - start:.3f}s')
    print(f'GUI heartbeat callbacks during import/preparation: {len(ticks)}')
    window.stackedWidget.setCurrentIndex(1)
    window.cutoff_spin.setValue(6)
    app.processEvents()
    window.grab().save(str(args.out / 'outdated.png'))
    window.cancel_processing()
    app.processEvents()
    window.grab().save(str(args.out / 'cancelled.png'))
    window.cutoff_spin.setValue(5)
    window._generate_active_page()
    QTimer.singleShot(20, lambda: window.grab().save(str(args.out / 'working.png')))
    wait_for_jobs()
    heartbeat.stop()
    for mode in ['light', 'dark']:
        theme_manager.set_mode(mode)
        for index in range(window.stackedWidget.count()):
            window.stackedWidget.setCurrentIndex(index)
            app.processEvents()
            QTest.qWait(100)
            # Deferred equal-aspect and label layout callbacks need one more pass.
            app.processEvents()
            name = window.stackedWidget.currentWidget().objectName()
            window.grab().save(str(args.out / f'{mode}-{name}.png'))
        window.stackedWidget.setCurrentIndex(0)
        window.sidebar.grab().save(str(args.out / f'{mode}-recording-library.png'))
        for width, height in [(1100, 720), (1366, 768)]:
            window.resize(width, height)
            app.processEvents()
            QTest.qWait(100)
            window.grab().save(str(args.out / f'{mode}-{width}.png'))
        window.resize(1600, 900)
    window.cancel_processing()
    window._jobs.pool.waitForDone(5000)
    app.processEvents()
    window.close()
    app.processEvents()
    print(f'Visual review images: {os.path.abspath(args.out)}')


if __name__ == '__main__':
    main()
