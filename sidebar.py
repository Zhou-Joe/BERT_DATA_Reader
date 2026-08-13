# coding: utf-8
"""
Dockable sidebar for BERT Reader — displays dataset info cards with stats,
angle info, and action buttons for each loaded dataset.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QSizePolicy, QColorDialog,
)

import qfluentwidgets as qfw
from qfluentwidgets import (
    FluentIcon as FI, TransparentToolButton, TransparentPushButton,
    SmoothScrollArea,
)

from theme import (
    DATASET_COLORS, FONT_SM, FONT_XS,
    card_title_style, card_meta_inline_style,
    card_section_label_style,
    swatch_button_style,
    theme_manager,
)


class InfoCard(QWidget):
    """Self-contained card widget for one loaded dataset.

    Layout (top to bottom):
      ┌──────────────────────────────────────────┐
      │ [swatch]  Filename (title, bold)         │  ← header row
      │           Pitch/Seatback/Roll/Yaw (meta) │
      ├──────────────────────────────────────────┤
      │ STATISTICS                                │
      │  X | tMax | tMin | Max | Min | Zero      │  ← compact table
      │  Y | ...                                  │
      │  Z | ...                                  │
      ├──────────────────────────────────────────┤
      │  [Truncate] [Edit Angle] [Modify]         │  ← actions row 1
      │  [Export]            [Remove]             │  ← actions row 2
      └──────────────────────────────────────────┘
    """

    def __init__(self, data, plottype: str, plot_order: int,
                 remove_callback, mainui, parent=None):
        super().__init__(parent)
        self.setObjectName("InfoCard")  # picks up QSS InfoCard{} rule
        self._data = data
        self._mainui = mainui
        self._plot_order = plot_order
        # Read the colour from the central map (single source of truth) so
        # the swatch always matches what's on the charts.
        self._color = self._resolve_color()

        # Bug fix: hold references to popup dialogs so they aren't GC'd
        # before being shown. Cleared on close (see _track_dialog).
        self._active_dialogs: list = []

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)
        # Keep a reference so update_plottype can swap the stats table in place
        # without rebuilding the whole card (a full rebuild is ~52ms/card and
        # dominates plot-type switch latency when many datasets are loaded).
        self._root_layout = root
        self._plottype = plottype
        # Allow the card to shrink to the sidebar viewport width. Without an
        # explicit maximumWidth the stats grid's natural size forces the card
        # wider than the viewport → horizontal scrolling.
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)

        # ---- Header: swatch + filename + inline angles, ONE row -----------
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)

        # 'Plot' visibility toggle — a Fluent eye icon (open / closed) using
        # FluentIcon.VIEW / FluentIcon.HIDE. State mirrors the central
        # dataset_visibility map; clicking toggles the dataset's trend curves.
        # TransparentToolButton is borderless and re-themes itself automatically.
        # Placed FIRST (left of the swatch) so show/hide is the leftmost control.
        self._plot_check = TransparentToolButton()
        self._plot_check.setCheckable(True)
        self._plot_check.setCursor(Qt.CursorShape.PointingHandCursor)
        self._plot_check.setIconSize(QSize(18, 18))
        self._plot_check.setToolTip(
            "Show/hide this dataset on the Trend chart")
        initial_visible = getattr(self._mainui, "dataset_visibility", {}).get(
            data.filename, True)
        self._plot_check.setChecked(initial_visible)
        self._style_visibility_icon()
        self._plot_check.toggled.connect(self._on_plot_toggled)
        header.addWidget(self._plot_check)

        # Colour swatch — small circle, sized to match the text line height.
        self._swatch = QPushButton()
        self._swatch.setFixedSize(14, 14)
        self._swatch.setCursor(Qt.CursorShape.PointingHandCursor)
        self._swatch.setToolTip("Click to change this dataset's colour")
        self._swatch.clicked.connect(self._pick_color)
        self._style_swatch()
        header.addWidget(self._swatch)

        # Filename (bold title, single line, elided if long).
        self._title_label = QLabel(self._short_title(data.filename))
        self._title_label.setStyleSheet(card_title_style(self._color))
        self._title_label.setWordWrap(False)
        self._title_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        header.addWidget(self._title_label)

        # Inline angle meta — small secondary text on the same row, no box.
        self._info_label = QLabel()
        self._info_label.setStyleSheet(card_meta_inline_style())
        self._info_label.setText(self._format_meta_inline(data))
        self._info_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Preferred
        )
        header.addWidget(self._info_label, 1)
        root.addLayout(header)

        # ---- Statistics section (collapsible) ---------------------------
        if plottype == "Standard":
            stats = data.get_data_stats(data.std_data)
        elif plottype == "Raw":
            stats = data.get_data_stats(data.data)
        elif plottype == "Filter":
            stats = data.get_data_stats(data.filtered_data)
        else:
            stats = data.get_data_stats(data.std_data)

        # Compact stats: merge tMax/tMin and Max/Min, add the 0.2s column.
        stats = self._compact_stats(stats, data, plottype)

        # Collapsible header: ▾/▸ STATISTICS — click to toggle.
        self._stats_expanded = True
        self._stats_header = self._make_section_label("▾ STATISTICS")
        self._stats_header.setCursor(Qt.CursorShape.PointingHandCursor)
        self._stats_header.mousePressEvent = self._toggle_stats
        root.addWidget(self._stats_header)

        self._table = self._make_stats_table(stats)
        # Map of row-label (e.g. "X"/"Y"/"Z") -> [value QLabel, value QLabel,
        # value QLabel] for the three data columns. Captured once so
        # update_plottype can setText in place on a plot-type switch instead
        # of rebuilding the whole table widget (a rebuild costs ~40ms of Qt
        # layout reflow per card; setText is ~0).
        self._stat_cells: dict[str, list[QLabel]] = self._collect_stat_cells()
        root.addWidget(self._table)

        # ---- Actions (single row, compact) ------------------------------
        # Fluent TransparentPushButton, text-only: qfluentwidgets' PushButton
        # paints its Fluent icon at a fixed left offset on top of Qt's own
        # centred text, so icon+text overlap on these compact 28px-tall
        # buttons. Text-only keeps them clean and readable.
        # Labels are shortened so all 5 fit even at the sidebar's MIN_WIDTH.
        self._action_buttons: list = []
        actions = QHBoxLayout()
        actions.setSpacing(2)
        actions.setContentsMargins(0, 0, 0, 0)
        for text, cb in (
            ("Trunc", self._popup_truncate),
            ("Ang", self._popup_angle),
            ("Edit", self._popup_edit),
            ("Exp", self._popup_export),
            ("Del", remove_callback),
        ):
            btn = self._make_action_btn(text, cb, danger=(text == "Del"))
            actions.addWidget(btn)
        root.addLayout(actions)

    def _toggle_stats(self, event) -> None:
        """Collapse/expand the statistics table on header click."""
        self._stats_expanded = not self._stats_expanded
        self._stats_header.setText("▾ STATISTICS" if self._stats_expanded
                                   else "▸ STATISTICS")
        self._table.setVisible(self._stats_expanded)

    def _collect_stat_cells(self) -> dict:
        """Map row label -> [value QLabel × 3] for the current stats table.

        The table layout is fixed across plot types: a header row, a divider
        row, then one row per axis (X/Y/Z). Column 0 is the row label; columns
        1-3 are the data cells. We capture references to those data-cell
        QLabels so update_plottype can update them in place (setText) instead
        of rebuilding the table (which costs ~40ms of Qt layout reflow per
        card via processEvents on every add/remove).
        """
        cells: dict[str, list] = {}
        if self._table is None:
            return cells
        grid = self._table.layout()
        # Walk every cell and remember the data cells (cols 1..) keyed by the
        # row label in col 0. Skips the header row and divider.
        for i in range(grid.count()):
            r, c, _rs, _cs = grid.getItemPosition(i)
            it = grid.itemAt(i)
            w = it.widget() if it is not None else None
            if w is None or c == 0 or r < 2:
                continue
            # Find this row's label (col 0) by scanning the grid for (r, 0).
            row_label = None
            for j in range(grid.count()):
                jr, jc, _r, _c = grid.getItemPosition(j)
                if jr == r and jc == 0:
                    jw = grid.itemAt(j).widget() if grid.itemAt(j) else None
                    row_label = jw.text() if jw is not None else None
                    break
            if row_label is None:
                continue
            cells.setdefault(row_label, [None, None, None])
            # Columns are contiguous from 1; index by column position.
            if 1 <= c <= 3:
                cells[row_label][c - 1] = w
        return cells

    def update_plottype(self, plottype: str) -> None:
        """Refresh this card's stats for a new plot type in place.

        A plot-type switch (Raw/Filter/Standard) only changes the numbers in
        the stats table — the headers, layout, swatch, eye toggle, title,
        meta, and action buttons are identical across plot types. So instead
        of rebuilding the table widget (which forces a ~40ms Qt layout reflow
        per card on the event loop), recompute the stats and just setText on
        the existing data-cell QLabels (~0ms). This is the main reason
        switching plot types no longer lags when many datasets are loaded.
        """
        self._plottype = plottype
        src = {"Raw": "data", "Filter": "filtered_data",
               "Standard": "std_data"}.get(plottype, "std_data")
        df = getattr(self._data, src)
        try:
            stats = self._data.get_data_stats(df)
            stats = self._compact_stats(stats, self._data, plottype)
        except Exception:
            return

        cols = list(stats.columns)
        # Update each data cell in place. cells[row_label] = [lbl, lbl, lbl]
        # in column order (matches the stats DataFrame's column order).
        for row_label, lbls in self._stat_cells.items():
            if row_label not in stats.index:
                continue
            for ci, col in enumerate(cols):
                if ci >= len(lbls) or lbls[ci] is None:
                    continue
                lbls[ci].setText(str(stats.loc[row_label, col]))

    # ==================================================================
    # Builders
    # ==================================================================

    @staticmethod
    def _short_title(filename: str) -> str:
        """Trim long filenames so the title stays on one line."""
        name = filename or "dataset"
        return name if len(name) <= 46 else name[:43] + "…"

    @staticmethod
    def _format_meta_inline(data) -> str:
        """Single-line angle summary, e.g. ``P:0° S:0° R:0° Y:0°``.

        Designed to sit on the header row beside the filename, so it stays
        compact and uses abbreviated keys.
        """
        try:
            return (
                f"P:{data.pitch_angle}° "
                f"S:{data.seatback_angle}° "
                f"R:{data.roll_angle}° "
                f"Y:{data.yaw_angle}°"
            )
        except Exception:
            return ""

    @staticmethod
    def _format_meta(data) -> str:
        """Two-line rich-text version (kept for backwards compatibility /
        any caller that still wants the boxed layout)."""
        try:
            return (
                f"<span>Pitch <b>{data.pitch_angle}°</b></span>"
                f"&nbsp;&nbsp;<span>Seatback <b>{data.seatback_angle}°</b></span>"
                f"<br>"
                f"<span>Roll <b>{data.roll_angle}°</b></span>"
                f"&nbsp;&nbsp;<span>Yaw <b>{data.yaw_angle}°</b></span>"
            )
        except Exception:
            return data.get_angle_info()

    @staticmethod
    def _make_section_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(card_section_label_style())
        return lbl

    def _restyle_section_labels(self) -> None:
        """Re-apply section/header inline styles after a theme switch.

        ``_stats_header`` is built once via ``card_section_label_style()``
        and would otherwise keep the old palette's colour forever."""
        if self._stats_header is not None:
            self._stats_header.setStyleSheet(card_section_label_style())

    @staticmethod
    def _compact_stats(stats, data, plottype: str):
        """Return a compact, card-friendly stats DataFrame.

        Merges the time/peak pairs into single ``Max/Min`` style cells so
        the table is narrower and reads more like the 0.2s column:

            t (Max/Min)   Max/Min      0.2s (+/−)   Zero Pos
            1.23/4.56s    +3.4/−1.2    +2.10/−0.50    0.0

        The source ``stats`` frame (from ``get_data_stats``) keeps its
        original columns; this is a presentation-layer transform only.
        """
        import pandas as pd
        try:
            src_attr = {"Standard": "std_data", "Raw": "data",
                        "Filter": "filtered_data"}.get(plottype, "std_data")
            frame = getattr(data, src_attr, None)
            if frame is None:
                frame = data.std_data
            fs = getattr(data, "fs", 500)
            from sci_calculation import ProcessASTM
            pos, neg = ProcessASTM(frame, max(1, int(round(0.2 * fs))))
        except Exception as e:
            print(f"0.2s column skipped: {e}")
            pos = neg = [0, 0, 0]

        rows = []
        for i, idx in enumerate(stats.index):
            mx = stats.iloc[i, stats.columns.get_loc("Max")]
            mn = stats.iloc[i, stats.columns.get_loc("Min")]
            zero = stats.iloc[i, stats.columns.get_loc("Zero Pos")]
            rows.append([
                f"{mx:+.3f}/{mn:+.3f}",
                f"{pos[i]:+.3f}/{neg[i]:+.3f}",
                f"{zero:.3f}",
            ])
        return pd.DataFrame(
            rows,
            index=stats.index,
            columns=["Max/Min", "0.2s (+/−)", "Zero Pos"],
        )

    def _make_stats_table(self, stats) -> QWidget:
        """Build the compact stats display as a hand-laid grid of QLabels.

        We deliberately avoid ``QTableView``/``QHeaderView`` here: the
        combination of section-resize modes + a tall InfoCard used to trigger
        a Qt layout feedback loop that froze the UI. A static QGridLayout of
        labels has no such risk and gives full control over the compact
        ``X | t (Max/Min) | Max/Min | 0.2s (+/−) | Zero Pos`` layout.

        Each label carries a ``statRole`` dynamic property (``header`` /
        ``row`` / ``cell``) so ``refresh_theme`` can re-apply its colour
        after a light/dark switch — otherwise the stats text stays frozen on
        whichever palette the card was built under.
        """
        from PySide6.QtWidgets import QGridLayout, QFrame

        host = QWidget()
        host.setObjectName("cardStatsGrid")
        # Allow the table to shrink below its sizeHint so the card fits the
        # sidebar width (otherwise the long header text forces ~700px and the
        # card needs horizontal scrolling).
        host.setMinimumWidth(0)
        host.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        grid = QGridLayout(host)
        grid.setContentsMargins(6, 4, 6, 4)
        grid.setHorizontalSpacing(4)
        grid.setVerticalSpacing(2)

        cols = list(stats.columns)
        # Shorten column headers so the grid can shrink to fit narrow sidebars.
        # The full names are restored via tooltip.
        short = {"Max/Min": "±g", "0.2s (+/−)": "0.2s±", "Zero Pos": "0"}
        # Layout (rows):
        #   0           header (column titles)
        #   1           divider line — its OWN row, so it no longer overlaps
        #               the X data row (the old "white line on the X row")
        #   2..         data rows (X / Y / Z)
        grid.addWidget(self._mk_stat_label("", "header"), 0, 0)
        for c, name in enumerate(cols, start=1):
            lbl = self._mk_stat_label(short.get(name, name), "header")
            lbl.setToolTip(name)
            grid.addWidget(lbl, 0, c)

        # Data rows start at row 2; the divider occupies row 1 on its own.
        for r, idx in enumerate(stats.index, start=2):
            grid.addWidget(self._mk_stat_label(str(idx), "row"), r, 0)
            for c, col in enumerate(cols, start=1):
                val = stats.loc[idx, col]
                grid.addWidget(self._mk_stat_label(str(val), "cell"), r, c)

        # Make value columns expand evenly; row-label column stays narrow.
        grid.setColumnStretch(0, 0)
        for c in range(1, len(cols) + 1):
            grid.setColumnStretch(c, 1)

        # A thin divider under the header row. Lives on its own grid row (1)
        # so it sits BETWEEN the header and the X data row instead of ON it.
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFixedHeight(1)
        line.setObjectName("cardStatsDivider")
        line.setStyleSheet(self._divider_style())
        grid.addWidget(line, 1, 0, 1, len(cols) + 1)
        return host

    # --- Stats-table styling helpers (token-aware, re-read on theme switch) -

    @staticmethod
    def _stat_role_style(role: str) -> str:
        """Inline stylesheet for one stats-table role. Reads the *current*
        palette so the same call works at build time and at refresh time.

        Uses a compact font so the grid fits narrow sidebars without forcing
        the card wider than the viewport (which would need horizontal scroll).
        """
        tk = theme_manager.tokens
        if role == "header":
            return (
                f"color: {tk['text_secondary']}; background: transparent; "
                f"font-size: {FONT_XS}pt; font-weight: 600; padding: 1px 0;"
            )
        if role == "row":
            return (
                f"color: {tk['text']}; background: transparent; "
                f"font-size: {FONT_XS}pt; font-weight: 600; padding: 1px 0;"
            )
        # cell
        return (
            f"color: {tk['text']}; background: transparent; "
            f"font-size: {FONT_XS}pt; padding: 1px 0;"
        )

    @staticmethod
    def _divider_style() -> str:
        tk = theme_manager.tokens
        # Use the neutral 'divider' token, not 'glass_stroke' — glass_stroke is
        # near-white in light mode (rgba(255,255,255,0.85)) since it's tuned
        # for translucent glass on the glow layer; on the opaque InfoCard that
        # read as a stark white line. 'divider' is a soft grey in both themes.
        return f"color: {tk['divider']}; background: transparent;"

    def _mk_stat_label(self, text: str, role: str) -> QLabel:
        """Create a stats-table label tagged with ``statRole`` so its colour
        can be re-applied on a theme switch."""
        lbl = QLabel(text)
        lbl.setProperty("statRole", role)  # tagged for refresh_theme
        lbl.setStyleSheet(self._stat_role_style(role))
        # Allow shrinking below sizeHint + word-wrap so the stats grid fits
        # narrow sidebars without forcing the card wider than the viewport.
        lbl.setMinimumWidth(0)
        lbl.setWordWrap(True)
        lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        return lbl

    def _make_action_btn(self, text, callback, danger: bool = False):
        """Build one compact Fluent action button (text-only) for the actions row.

        Text-only because qfluentwidgets' PushButton overlaps icon+text on
        these small buttons (see comment in __init__). The Remove button gets
        a danger-red text colour so it reads as destructive. Padding/font are
        shrunk so all 5 fit at the sidebar's MIN_WIDTH.
        """
        btn = TransparentPushButton(text)
        btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        btn.setMinimumHeight(20)
        btn.setMaximumHeight(22)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        # Compact: minimal padding + small font so 5 buttons fit at min width.
        btn.setStyleSheet(
            "TransparentPushButton, QPushButton {"
            "  padding: 1px 4px; margin: 0px;"
            "  min-height: 0px; min-width: 0px;"
            "  font-size: 8pt;"
            "}"
        )
        btn.clicked.connect(callback)
        if danger:
            # Danger tint via a per-button stylesheet. Reads the live token so
            # it stays correct after a light/dark switch (refresh_theme re-applies).
            self._apply_danger_style(btn)
        self._action_buttons.append(btn)
        return btn

    @staticmethod
    def _apply_danger_style(btn) -> None:
        """Tint a button with the danger token (red text + red hover fill).

        Preserves the compact padding/font from _make_action_btn.
        """
        tk = theme_manager.tokens
        btn.setStyleSheet(
            f"TransparentPushButton, QPushButton {{"
            f"  color: {tk['danger']};"
            f"  padding: 1px 4px; margin: 0px;"
            f"  min-height: 0px; min-width: 0px; font-size: 8pt;"
            f"}}"
            f"TransparentPushButton:hover, QPushButton:hover {{"
            f"  background-color: rgba({tk['danger_rgb']}, 0.12);"
            f"  color: {tk['danger']};"
            f"}}"
        )

    # --- Theme refresh hook (called by MainWindow.refresh_theme) ---------

    def refresh_theme(self) -> None:
        """Re-apply tokenized inline styles after a light/dark switch.

        Fluent widgets (TransparentPushButton, TransparentToolButton) restyle
        themselves via qfluentwidgets' themeChanged signal, so this only
        re-applies the token-driven bits: the title/meta labels, the stats
        table cells, the swatch, and the danger-tinted Remove button.
        """
        self._title_label.setStyleSheet(card_title_style(self._color))
        self._info_label.setStyleSheet(card_meta_inline_style())
        self._restyle_section_labels()
        self._restyle_stats_table()
        self._style_swatch()
        # Re-tint the Remove button (its danger style reads the live token).
        if self._action_buttons:
            self._apply_danger_style(self._action_buttons[-1])

    def _restyle_stats_table(self) -> None:
        """Re-apply the stats-table cell/row/header/divider colours.

        Each label was tagged with a ``statRole`` property at build time;
        we walk the table and re-emit its stylesheet from the *current*
        palette so the numbers don't stay frozen on the old theme."""
        from PySide6.QtWidgets import QFrame

        if self._table is None:
            return
        for child in self._table.findChildren(QLabel):
            role = child.property("statRole")
            if role:
                child.setStyleSheet(self._stat_role_style(role))
        for divider in self._table.findChildren(QFrame):
            if divider.objectName() == "cardStatsDivider":
                divider.setStyleSheet(self._divider_style())

    # --- Colour management ----------------------------------------------

    def _resolve_color(self) -> str:
        """Read this dataset's colour from the central map, falling back to
        the palette-derived index colour if the map isn't populated yet."""
        cmap = getattr(self._mainui, "dataset_colors", None)
        if cmap is not None:
            c = cmap.get(self._data.filename)
            if c:
                return c
        return DATASET_COLORS[(self._plot_order - 1) % len(DATASET_COLORS)]

    def _style_swatch(self) -> None:
        """Render the swatch using the tokenized swatch_button_style helper."""
        self._swatch.setStyleSheet(swatch_button_style(self._color))

    def _style_visibility_icon(self) -> None:
        """Set the eye icon based on checked state.

        Uses FluentIcon.VIEW (open eye) when visible and FluentIcon.HIDE
        (struck-through eye) when hidden. Fluent icons auto-recolor with the
        theme, so no manual tinting is needed — and they're far cleaner than
        the old hand-rolled SVG glyphs.
        """
        checked = self._plot_check.isChecked()
        self._plot_check.setIcon(FI.VIEW if checked else FI.HIDE)

    def _on_plot_toggled(self, checked: bool) -> None:
        """Show/hide this dataset's trend curves via the central map."""
        self._style_visibility_icon()   # swap eye open / closed glyph
        self._mainui.set_dataset_visible(self._data.filename, checked)

    def _pick_color(self) -> None:
        """Open a native colour dialog and apply the chosen colour."""
        initial = QColor(self._color)
        dlg = QColorDialog(initial, self)
        dlg.setWindowTitle("Choose Dataset Colour")
        # The main window sets a QSS that targets QLabel/QGroupBox/QStatusBar —
        # those rules cascade into QColorDialog and corrupt its native picker
        # (custom-colour swatches and the OK/Cancel buttons end up unreadable,
        # most visibly in light mode). Reparent to None and force the platform
        # style so the dialog renders with its OWN styling, not the window's.
        dlg.setParent(None)
        dlg.setOption(QColorDialog.ColorDialogOption.ShowAlphaChannel, False)
        try:
            from PySide6.QtWidgets import QStyleFactory
            dlg.setStyle(QStyleFactory.create("Windows") or dlg.style())
        except Exception:
            pass
        dlg.setStyleSheet("")  # block inherited QSS
        if dlg.exec() == QColorDialog.DialogCode.Accepted:
            chosen = dlg.currentColor()
            if chosen.isValid():
                self._mainui.set_dataset_color(self._data.filename, chosen.name())

    def apply_color(self, color_hex: str) -> None:
        """Public hook called by MainWindow.set_dataset_color to refresh the
        swatch + label stripe after the central map was updated."""
        self._color = color_hex
        self._title_label.setStyleSheet(card_title_style(self._color))
        self._info_label.setStyleSheet(card_meta_inline_style())
        self._style_swatch()

    # --- Popup launchers (refs held to prevent garbage collection) -------

    def _track_dialog(self, win) -> None:
        """Keep *win* alive until it closes, then drop the reference."""
        self._active_dialogs.append(win)
        if hasattr(win, "destroyed"):
            win.destroyed.connect(lambda *_: self._discard_dialog(win))
        if hasattr(win, "finished"):
            win.finished.connect(lambda *_: self._discard_dialog(win))

    def _discard_dialog(self, win) -> None:
        try:
            self._active_dialogs.remove(win)
        except ValueError:
            pass

    def _popup_truncate(self) -> None:
        from dialogs import TruncateTimeDialog
        try:
            win = TruncateTimeDialog(self._mainui, self._data)
            self._track_dialog(win)
            win.show()
        except Exception as e:
            print(e)

    def _popup_angle(self) -> None:
        from dialogs import EditAngleDialog
        try:
            win = EditAngleDialog(self._data, self._mainui)
            self._track_dialog(win)
            win.show()
        except Exception as e:
            print(e)

    def _popup_edit(self) -> None:
        from dialogs import EditDataDialog
        try:
            self._edit_win = EditDataDialog(self._mainui, self._data)
            self._track_dialog(self._edit_win)
            self._edit_win.show()
        except Exception as e:
            print(e)

    def _popup_export(self) -> None:
        from dialogs import ExportDialog
        try:
            win = ExportDialog(self._mainui, self._data)
            self._track_dialog(win)
            win.show()
        except Exception as e:
            print(e)


class _ResizeHandle(QWidget):
    """A thin vertical strip the user drags to resize its parent DataSidebar.

    Sits on the LEFT edge of the sidebar (next to the chart area). Captures
    mouse drag and calls ``parent.resize_width(delta)``.
    """

    def __init__(self, sidebar: "DataSidebar"):
        super().__init__(sidebar)
        self._sidebar = sidebar
        self.setFixedWidth(5)
        self.setCursor(Qt.CursorShape.SplitHCursor)
        self._press_x: int | None = None
        self._press_w: int = 0

    def paintEvent(self, event) -> None:
        from PySide6.QtGui import QPainter
        tk = theme_manager.tokens
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(tk["card_stroke"]))
        painter.end()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_x = int(event.globalPosition().x())
            self._press_w = self._sidebar.width()

    def mouseMoveEvent(self, event) -> None:
        if self._press_x is None:
            return
        # Dragging LEFT (negative delta) widens the sidebar (it's on the right
        # side of the window), dragging RIGHT narrows it.
        delta = self._press_x - int(event.globalPosition().x())
        self._sidebar.resize_width(self._press_w + delta)

    def mouseReleaseEvent(self, event) -> None:
        self._press_x = None


class DataSidebar(QWidget):
    """Panel showing loaded dataset info cards.

    Lives as a persistent right panel of the FluentWindow (added to hBoxLayout).
    Because it's a plain QWidget (not a Fluent widget), it doesn't inherit the
    window's Mica/dark background automatically — so ``paintEvent`` paints the
    current theme's surface colour, and ``refresh_theme`` re-paints on a
    light/dark switch. A thin drag handle on the left edge lets the user
    resize the panel width.
    """

    # Width floor/ceiling for the drag-resize. Wide enough that all 5 action
    # buttons always render at full text (Trunc/Angle/Edit/Export/Remove).
    MIN_WIDTH = 320
    MAX_WIDTH = 560

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(self.MIN_WIDTH)
        self.setMaximumWidth(self.MAX_WIDTH)
        # Painted background reads the live theme token, so it tracks dark mode.
        self.setAutoFillBackground(False)
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)

        # Outer layout: [drag handle | scroll area]. The handle is a 4px strip
        # the user drags to change this panel's width.
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._handle = _ResizeHandle(self)
        outer.addWidget(self._handle)

        # Scrollable content — Fluent SmoothScrollArea for buttery kinetic scroll.
        self._scroll = SmoothScrollArea()
        self._scroll.setObjectName("sidebarScroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._scroll.setFrameShape(SmoothScrollArea.Shape.NoFrame)
        # Transparent viewport + autoFillBackground off so the sidebar's own
        # paintEvent (theme-aware surface colour) shows through instead of the
        # default light widget background.
        self._scroll.viewport().setAutoFillBackground(False)
        self._scroll.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._scroll.setStyleSheet("QScrollArea, QScrollArea > QWidget > QWidget "
                                   "{ background: transparent; }")

        self._content = QWidget()
        self._content.setObjectName("sidebarContent")
        self._content.setStyleSheet("background: transparent;")
        self._layout = QVBoxLayout(self._content)
        self._layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._layout.setSpacing(8)
        self._layout.setContentsMargins(16, 12, 12, 12)
        self._layout.addStretch()

        self._scroll.setWidget(self._content)
        outer.addWidget(self._scroll, 1)

        self._cards: list[InfoCard] = []

    # --- Theme-aware background -------------------------------------------

    def paintEvent(self, event) -> None:
        """Paint the panel background from the current theme token.

        A plain QWidget would show the default window colour (light grey)
        even in dark mode; this keeps the sidebar in sync with the Fluent
        dark/light surface.
        """
        from PySide6.QtGui import QPainter, QColor as _QColor
        tk = theme_manager.tokens
        painter = QPainter(self)
        painter.fillRect(self.rect(), _QColor(tk["surface"]))
        # A hairline on the LEFT edge (next to the chart area) for definition.
        painter.setPen(_QColor(tk["card_stroke"]))
        painter.drawLine(0, 0, 0, self.height())
        painter.end()

    def refresh_theme(self) -> None:
        """Re-paint the background + every card after a light/dark switch."""
        self.update()
        for card in self._cards:
            if hasattr(card, "refresh_theme"):
                card.refresh_theme()
        self._apply_card_widths()

    def _apply_card_widths(self) -> None:
        """Cap each card's width to the scroll viewport so the stats grid is
        forced to shrink/wrap instead of making the card wider than the panel
        (which would require horizontal scrolling). Called after add/resize/
        theme-refresh. Deferred via QTimer because the viewport width isn't
        settled until the next layout pass.
        """
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, self._do_apply_card_widths)

    def _do_apply_card_widths(self) -> None:
        vp = self._scroll.viewport().width()
        if vp <= 0:
            return
        lm, _, rm, _ = self._layout.getContentsMargins()
        w = max(0, vp - lm - rm)
        for card in self._cards:
            card.setMaximumWidth(w)

    def add_card(self, card: InfoCard) -> None:
        """Add an info card to the sidebar."""
        # Insert before the stretch
        self._layout.insertWidget(self._layout.count() - 1, card)
        self._cards.append(card)
        self._apply_card_widths()

    def resize_width(self, width: int) -> None:
        """Set this panel's width, clamped to [MIN_WIDTH, MAX_WIDTH].

        Called by the ``_ResizeHandle`` drag. The MIN_WIDTH guarantees all 5
        action buttons (Trunc/Ang/Edit/Exp/Del) render at full text.
        """
        w = max(self.MIN_WIDTH, min(self.MAX_WIDTH, int(width)))
        self.setFixedWidth(w)
        self._apply_card_widths()

    def clear_all(self) -> None:
        """Remove all info cards."""
        for card in self._cards:
            card.deleteLater()
        self._cards.clear()

    def update_plottype(self, plottype: str) -> None:
        """Refresh every card's stats table for a new plot type in place.

        Cheaper than clear + re-add: only the stats table is rebuilt per card
        (~1-2ms each) instead of the whole card widget tree (~52ms each).
        """
        for card in self._cards:
            if hasattr(card, "update_plottype"):
                card.update_plottype(plottype)
