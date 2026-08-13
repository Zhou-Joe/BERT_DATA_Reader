# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for BERT Reader — one-file build (clean venv).

Built against a minimal uv venv containing only:
  PySide6, PySide6-Fluent-Widgets (qfluentwidgets), pandas, numpy, scipy,
  python-docx, matplotlib, pyinstaller

The app only uses matplotlib.path.Path (for boundary hit-testing), NOT the
plotting backends — so the heavy GUI backends / toolkits are excluded.
"""

from PyInstaller.utils.hooks import (
    collect_data_files, collect_submodules, collect_dynamic_libs)

# --- qfluentwidgets: SVG icons, QSS, QRC resources, every submodule ----------
qfw_datas = collect_data_files('qfluentwidgets')
qfw_hidden = collect_submodules('qfluentwidgets')

# --- PySide6 plugins (platforms/styles/imageformats/svg) ---------------------
pyside6_datas = collect_data_files('PySide6')
pyside6_hidden = collect_submodules('PySide6')

# --- python-docx default template -------------------------------------------
docx_datas = collect_data_files('docx')

# --- App icon (SVG, loaded at runtime for the window icon) -------------------
icon_datas = [('icon.svg', '.')]

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=collect_dynamic_libs('PySide6'),
    datas=qfw_datas + pyside6_datas + docx_datas + icon_datas,
    hiddenimports=qfw_hidden + pyside6_hidden + [
        'qfluentwidgets._rc',
        'PySide6.QtSvg',
        'PySide6.QtSvgWidgets',
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[
        # matplotlib GUI backend modules — the app only uses matplotlib.path
        # (for boundary hit-testing), NOT any plotting backend. But keep
        # matplotlib.backends itself (its __init__ is lightweight and imported
        # by rcsetup at load time) and matplotlib.path.
        'matplotlib.backends.backend_agg',
        'matplotlib.backends.backend_qt', 'matplotlib.backends.backend_qtagg',
        'matplotlib.backends.backend_qtcairo',
        'matplotlib.backends.backend_tk', 'matplotlib.backends.backend_tkagg',
        'matplotlib.backends.backend_gtk3', 'matplotlib.backends.backend_gtk4',
        'matplotlib.backends.backend_pdf', 'matplotlib.backends.backend_svg',
        'matplotlib.backends.backend_cairo', 'matplotlib.backends.backend_ps',
        'matplotlib.backends.backend_pgf', 'matplotlib.backends.backend_web',
        'matplotlib.pyplot',
        'tkinter', '_tkinter',
        # Testing / notebook tooling (not in the clean venv, but defensive).
        'pytest', 'IPython', 'jupyter', 'notebook', 'ipykernel',
    ],
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='BERT_READER_X',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    icon='icon.ico',   # embedded in the exe (multi-resolution, generated from icon.svg)
)
