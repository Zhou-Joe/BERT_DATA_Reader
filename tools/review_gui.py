"""Render all analysis views for visual review without opening a desktop window.

Example: python tools/review_gui.py --sample ../A3_BEFORE.csv --out ../gui_review
"""
import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
if os.environ['QT_QPA_PLATFORM'] == 'offscreen':
    os.environ.setdefault('QT_QUICK_BACKEND', 'software')
    os.environ.setdefault('QTWEBENGINE_CHROMIUM_FLAGS', '--disable-gpu')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton, QToolButton

from desktop_workspace import DesktopWindow
from theme import theme_manager


def web_value(window, expression):
    loop = QEventLoop()
    values = []
    def done(value):
        values.append(value)
        loop.quit()
    window.surface.web.page().runJavaScript(expression, 0, done)
    QTimer.singleShot(3000, loop, loop.quit)
    loop.exec()
    return values[0] if values else None


def wait_for_web(window, expression, timeout=12):
    deadline = perf_counter() + timeout
    while perf_counter() < deadline:
        if web_value(window, expression):
            return
        QTest.qWait(30)
    raise RuntimeError(f'Timed out waiting for {expression}')


def workspace_frame(window):
    # Offscreen WebEngine can return the previous compositor frame after a
    # resize. Wait for the appearance icon to paint at its new DOM location.
    wait_for_web(window, f"innerWidth === {window.surface.web.width()}")
    position = web_value(window, """JSON.stringify((() => {
        const r = document.querySelector('[aria-label="Toggle appearance"] svg').getBoundingClientRect();
        return [r.left, r.top, r.right, r.bottom];
    })())""")
    left, top, right, bottom = json.loads(position)
    deadline = perf_counter() + 5
    while perf_counter() < deadline:
        frame = window.grab()
        pixels = frame.toImage()
        scale = frame.devicePixelRatio()
        colors = {pixels.pixel(x, y)
                  for x in range(round(left * scale), round(right * scale))
                  for y in range(round(top * scale), round(bottom * scale))}
        if len(colors) >= 4:
            return frame
        QTest.qWait(50)
    raise RuntimeError('The workspace did not paint at the requested size')


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
    window = DesktopWindow()
    window.show()
    app.processEvents()
    window.resize(1366, 768)
    wait_for_web(window, "document.querySelectorAll('nav .view').length === 7")
    wait_for_web(window, "document.querySelectorAll('.format').length === 3")
    for mode in ('light', 'dark'):
        theme_manager.set_mode(mode)
        wait_for_web(window, f"!!document.querySelector('.radix-themes.{mode}')")
        QTest.qWait(100)
        workspace_frame(window).save(str(args.out / f'{mode}-welcome.png'))
    theme_manager.set_mode('light')
    window.resize(1600, 900)
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
    with patch('main_window.QFileDialog.getOpenFileNames',
               return_value=([str(p) for p in args.sample], '')):
        web_value(window, "document.querySelector('.format').click()")
        wait_for_web(window, """!document.querySelector('select[aria-label="Recording"]').disabled""",
                     timeout=30)
    wait_for_jobs()
    if len(window.datalist) != 2 * len(args.sample):
        raise RuntimeError('Not every selected raw file produced both signal groups')
    print(f'Imported {len(args.sample)} files into {len(window.datalist)} recordings')
    print(f'Background import, trend plot and preparation: {perf_counter() - start:.3f}s')
    print(f'GUI heartbeat callbacks during import/preparation: {len(ticks)}')
    if not window.surface.deck.isVisible():
        raise RuntimeError('The chart did not become visible after import')
    web_value(window, """const condition = document.querySelector('select[aria-label="Condition"]');
              condition.selectedIndex = 1; condition.dispatchEvent(new Event('change', {bubbles:true}))""")
    wait_for_web(window, """document.querySelector('select[aria-label="Condition"]').value === '1'""")
    if window.select_cond.currentText() != 'E-Stop':
        raise RuntimeError('The selected condition did not reach the analysis engine')
    window.select_cond.setCurrentIndex(0)
    wait_for_web(window, """document.querySelector('select[aria-label="Condition"]').value === '0'""")
    web_value(window, """document.querySelector('button[aria-label="Toggle recording library"]').click()""")
    wait_for_web(window, """document.querySelector('button[aria-label="Toggle recording library"]').getAttribute('aria-pressed') === 'false'""")
    if window.sidebar.isVisible():
        raise RuntimeError('The library toggle did not hide the native inspector')
    window._library_toggle.click()
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
            web_value(window, f"document.querySelectorAll('nav .view')[{index}].click()")
            wait_for_web(window, f"document.querySelectorAll('nav .view')[{index}].matches('[aria-current]')")
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
            frame = workspace_frame(window)
            visible = web_value(window, """[...document.querySelectorAll('main button')].every(button => {
                const r = button.getBoundingClientRect();
                return button.contains(document.elementFromPoint(r.x + r.width/2, r.y + r.height/2));
            })""")
            if not visible:
                raise RuntimeError(f'A workspace button is clipped at {width} × {height}')
            for bar in (window._trend_tools, window._shift_tools):
                if bar.findChild(QToolButton, 'qt_toolbar_ext_button').isVisible():
                    raise RuntimeError(f'{bar.windowTitle()} overflowed at {width} × {height}')
            for button in window._recording_tools.findChildren(QPushButton):
                point = button.mapTo(window, button.rect().center())
                if window.childAt(point) != button:
                    raise RuntimeError(f'{button.text()} is covered at {width} × {height}')
            frame.save(str(args.out / f'{mode}-{width}.png'))
        window.resize(1600, 900)
    web_value(window, """[...document.querySelectorAll('.file-actions button')]
              .find(button => button.textContent === 'Clear all').click()""")
    wait_for_web(window, """document.querySelector('select[aria-label="Recording"]').disabled""")
    if window.datalist or window._recording_tools.isEnabled():
        raise RuntimeError('Clear all did not reset the recording controls')
    window.cancel_processing()
    window._jobs.pool.waitForDone(5000)
    app.processEvents()
    window.close()
    app.processEvents()
    print(f'Visual review images: {os.path.abspath(args.out)}')


if __name__ == '__main__':
    main()
