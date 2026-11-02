"""Main-window coordination for fresh charts and cancellable background work."""
from functools import partial
import os

from PySide6.QtCore import QTimer, Slot
from PySide6.QtWidgets import QProgressBar, QPushButton

from analysis_pipeline import (AnalysisRequest, ImportBatch, import_recording,
                               import_recordings, prepare_analysis, snapshot)
from background import JobManager
from chart_acczone import AUTO_ZONE_DUR
from sci_calculation import allowance_at_t0
from theme import theme_manager


class ProcessingController:
    def _init_processing(self):
        self._closing = False
        self._refresh_suspended = False
        self._deferred_results = []
        self._prepared = None
        self._rendered_keys = {}
        self._trend_dirty = False
        self._auto_paused = False
        self._pending_report_notice = None
        self._pending_import_notice = None
        self._import_commit = None
        self._last_import_summary = None
        self._import_commit_timer = QTimer(self)
        self._import_commit_timer.setSingleShot(True)
        self._import_commit_timer.timeout.connect(self._commit_import_file)
        self._jobs = JobManager(self)
        self._jobs.result.connect(self._background_result)
        self._jobs.error.connect(self._background_error)
        self._jobs.progress.connect(self._background_progress)
        self._jobs.changed.connect(self._update_job_controls)
        self._jobs.drained.connect(self._processing_drained)
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(180)
        self._refresh_timer.timeout.connect(self.request_chart_update)
        self._job_progress = QProgressBar()
        self._job_progress.setFixedWidth(105)
        self._job_progress.setRange(0, 100)
        self._cancel_job = QPushButton('Cancel')
        self._cancel_job.clicked.connect(self.cancel_processing)
        self.statusBar().addPermanentWidget(self._job_progress)
        self.statusBar().addPermanentWidget(self._cancel_job)
        self.select_data.currentTextChanged.connect(self._queue_chart_refresh)
        for combo in (self.select_restraint, self.select_cond,
                      self._zr_plottype, self._xr_plottype):
            combo.currentTextChanged.connect(self._queue_chart_refresh)
        self.height_input.valueChanged.connect(self._queue_chart_refresh)
        self.height_input.setKeyboardTracking(False)
        self._xr_hyst.valueChanged.connect(self._queue_chart_refresh)
        self._xr_hyst.setKeyboardTracking(False)
        self.stackedWidget.currentChanged.connect(self._processing_page_changed)
        self._set_processing_state('empty', 'Add a recording to begin')
        self._update_job_controls()

    def _current_analysis_key(self):
        sources = tuple((id(data), id(data.data), data.filename, data.fs,
                         data.pitch_angle, data.seatback_angle, data.roll_angle,
                         data.yaw_angle, data.x_avg, data.y_avg, data.z_avg,
                         data.offset_x, data.offset_y, data.offset_z)
                        for data in self.datalist)
        return (sources, self.select_data.currentText(), self.cutoff_spin.value(),
                self.plottype, self.select_restraint.currentText(),
                self.select_cond.currentText(), self.height_input.value(),
                self._prone_btn.isChecked(), self._zr_plottype.currentText(),
                self._xr_plottype.currentText(), self._xr_hyst.value())

    def _page_is_current(self, page=None):
        page = page or self.stackedWidget.currentWidget()
        if page is None:
            return False
        if page.objectName() == 'page_trend':
            return bool(self.datalist) and not self._trend_dirty
        return self._rendered_keys.get(page.objectName()) == self._current_analysis_key()

    def _set_processing_state(self, state, message):
        self._processing_state = state
        title = {'empty': 'Acceleration workspace', 'stale': 'Charts outdated',
                 'working': 'Updating charts...', 'ready': 'Charts up to date',
                 'cancelled': 'Update cancelled', 'error': 'Update failed',
                 'importing': 'Importing recording...'}[state]
        tk = theme_manager.tokens
        color = tk['danger'] if state == 'error' else (
            tk['warning'] if state in ('stale', 'cancelled') else tk['text'])
        self._workspace_title.setText(title)
        self._workspace_title.setToolTip(message)
        self._workspace_title.setStyleSheet(f'color: {color}; font-size: 12pt; font-weight: 600;')

    def _show_pending_page(self, page, message):
        page._pending_label.setText(message)
        page._pending_label.show()
        page._chart_content.hide()

    def _show_current_page(self, page):
        page._pending_label.hide()
        page._chart_content.show()

    def _mark_charts_outdated(self, trend=False):
        self._jobs.cancel('analysis')
        self._prepared = None
        self._rendered_keys.clear()
        self._trend_dirty |= trend
        self.sidebar.set_pending_details(self._trend_dirty)
        name = self.select_data.currentText()
        for index in range(self.stackedWidget.count()):
            page = self.stackedWidget.widget(index)
            if page.objectName() != 'page_trend' or self._trend_dirty:
                self._show_pending_page(page, f'Updating charts for {name}\nPreparing current inputs...')
        for host in self._summary_hosts():
            host.hide()
        self._job_progress.setValue(0)
        self.statusBar().showMessage('Inputs changed. Preparing updated results...')
        self._set_processing_state('stale', 'Inputs changed. Previous results are hidden until refreshed.')
        self._analyze_btn.setEnabled(self._page_is_current())

    def _summary_hosts(self):
        return [self._sust_summary_host, self._egg_summary_host, self._gb_summary_host,
                self._zr_summary_host, self._xr_summary_host]

    def _queue_chart_refresh(self, *_args, trend=False):
        if (not hasattr(self, '_jobs') or self._closing or self._refresh_suspended
                or self._import_commit is not None):
            return
        if not self.datalist:
            return
        self._auto_paused = False
        if self._prepared and self._prepared['key'] == self._current_analysis_key() and not trend:
            self._render_current_chart()
            return
        self._mark_charts_outdated(trend=trend)
        self._refresh_timer.start()
        self._update_job_controls()

    def request_chart_update(self, force=False):
        if self._closing or self._refresh_suspended or self._import_commit is not None:
            return
        self._refresh_timer.stop()
        self._cutoff_timer.stop()
        self._auto_paused = False
        if not self.datalist:
            self.statusBar().showMessage('Add a dataset before updating charts', 3000)
            self._update_job_controls()
            return
        self.set_cutoff(self.cutoff_spin.value())
        key = self._current_analysis_key()
        if self._prepared and self._prepared['key'] == key and not force:
            self._render_current_chart()
            return
        dirty = self._trend_dirty or any(data._filtered_cutoff != self.cutoff for data in self.datalist)
        self._mark_charts_outdated(trend=dirty)
        request = AnalysisRequest(
            key=key, datasets=[(id(data), snapshot(data)) for data in self.datalist],
            selected=self.select_data.currentText(), cutoff=self.cutoff,
            plot_type=self.plottype, z_type=self._zr_plottype.currentText(),
            xy_type=self._xr_plottype.currentText(), deadband=self._xr_hyst.value(),
            zone_polygons={zone: tuple(path.vertices.copy() for path in paths)
                           for zone, paths in self.canvasAccZone._zone_paths.items()},
            zone_duration=AUTO_ZONE_DUR)
        self._job_progress.setValue(0)
        self._set_processing_state('working', f'Calculating current inputs for {request.selected}')
        self._jobs.start('analysis', partial(prepare_analysis, request))

    @Slot(str, object)
    def _background_result(self, kind, payload):
        if self._closing:
            return
        if self._refresh_suspended:
            self._deferred_results.append((kind, payload))
            return
        try:
            self._apply_background_result(kind, payload)
        except Exception as error:
            self._background_error(kind, str(error))

    def _apply_background_result(self, kind, payload):
        if kind == 'import':
            if isinstance(payload, ImportBatch):
                self._accept_import_batch(payload)
            else:
                self._accept_import(payload)
            return
        if payload['key'] != self._current_analysis_key():
            self._queue_chart_refresh()
            return
        live = {id(data): data for data in self.datalist}
        changed = self._trend_dirty
        for identity, prepared in payload['datasets']:
            data = live[identity]
            changed |= data.std_data is not prepared.std_data
            data.filtered_data, data.std_data = prepared.filtered_data, prepared.std_data
            data.cutoff = data._filtered_cutoff = prepared._filtered_cutoff
        self._prepared = payload
        needs_rebuild = (len(self.trendplot._series_list) != 3 * len(self.datalist)
                         or set(self.trendplot._series_by_dataset) != set(range(1, len(self.datalist) + 1)))
        if needs_rebuild:
            self._refresh_suspended = True
            try:
                self.trendplot.initplotxyz()
                self.clear_sidebar()
                self.replot_all()
            finally:
                self._refresh_suspended = False
        elif changed:
            self.trendplot.update_plottype(self.datalist, self.plottype, self.dataset_colors)
            self.sidebar.update_plottype(self.plottype)
        self._trend_dirty = False
        self.sidebar.set_pending_details(False)
        self._show_current_page(self.stackedWidget.widget(0))
        self._render_current_chart()

    def _render_current_chart(self):
        if self._refresh_suspended or not self._prepared or self._closing:
            return
        result = self._prepared
        key = self._current_analysis_key()
        if result['key'] != key:
            self._queue_chart_refresh()
            return
        page = self.stackedWidget.currentWidget()
        name = page.objectName()
        if self._rendered_keys.get(name) == key:
            return
        data = result['selected']
        color = self.dataset_colors.get(data.filename)
        try:
            if name in ('page_astm_sust', 'page_astm_egg'):
                canvas = self.canvasASTMSust if name == 'page_astm_sust' else self.canvasASTMEgg
                canvas.addplotxyz(result['frame'], self.select_restraint.currentText(),
                                  self.select_cond.currentText(), self.height_input.value(),
                                  color=color, prone=self._prone_btn.isChecked(), sustained=result['astm'])
            elif name == 'page_gb':
                self.canvasGB.addplotxyz(result['frame'], color=color, sustained=result['gb'])
            elif name == 'page_acczone':
                self.canvasAccZone.addplotxyz(result['zone_frame'], color=color, classification=result['zone'])
            elif name == 'page_zreversal':
                self.canvasZReversal.plot_data(result['z_frame'], color, data.fs, reversals=result['z_events'])
            elif name == 'page_xyreversal':
                self.canvasXYReversal.set_allowance({
                    axis: allowance_at_t0(axis, self.select_restraint.currentText(),
                                          self.select_cond.currentText(), self.height_input.value())
                    for axis in ('x', 'y')})
                self.canvasXYReversal.set_deadband(self._xr_hyst.value())
                self.canvasXYReversal.plot_data(result['xy_frame'], color, data.fs, reversals=result['xy_events'])
            self._rendered_keys[name] = key
            self._show_current_page(page)
            for host, host_page in zip(self._summary_hosts(),
                    ('page_astm_sust', 'page_astm_egg', 'page_gb', 'page_zreversal', 'page_xyreversal')):
                host.setVisible(self._rendered_keys.get(host_page) == key)
            self._analyze_btn.setEnabled(True)
            self._on_page_changed_for_analyze()
            for refresh in (self._refresh_sust_summary, self._refresh_egg_summary,
                            self._refresh_gb_summary, self._refresh_zreversal_summary,
                            self._refresh_xyreversal_summary):
                refresh()
            self._set_processing_state('ready', f'Current inputs: {data.filename}. Other views render when opened.')
            if self._pending_report_notice:
                self.statusBar().showMessage(self._pending_report_notice, 5000)
                self._pending_report_notice = None
            elif self._pending_import_notice:
                self.statusBar().showMessage(self._pending_import_notice, 10000)
                self._pending_import_notice = None
            else:
                self.statusBar().showMessage(f'Charts ready: {data.filename}', 3000)
        except Exception as error:
            self._rendered_keys.pop(name, None)
            self._show_pending_page(page, f'Could not update this view\n{error}\nClick Update charts to retry.')
            self._background_error('analysis', str(error))

    def _processing_page_changed(self, *_args):
        if self._refresh_suspended or self._closing:
            return
        self._analyze_btn.setEnabled(self._page_is_current())
        if self._prepared:
            QTimer.singleShot(0, self._render_current_chart)
        elif self.datalist and not self._auto_paused and not self._jobs.busy('analysis'):
            self._queue_chart_refresh()

    @Slot(str, int, str)
    def _background_progress(self, kind, percent, message):
        if self._closing:
            return
        self._job_progress.setValue(percent)
        self._job_progress.setToolTip(message)
        if not self._pending_report_notice and not self._pending_import_notice:
            self.statusBar().showMessage(message)

    @Slot(str, str)
    def _background_error(self, kind, message):
        if self._closing:
            return
        self._auto_paused = True
        self._set_processing_state('error', message)
        self.statusBar().showMessage(f'{kind.capitalize()} failed: {message}', 10000)
        if kind == 'analysis':
            self.sidebar.set_pending_details(self._trend_dirty, 'Summary is outdated. Update charts to retry.')
            self._analyze_btn.setEnabled(self._page_is_current())
            for index in range(self.stackedWidget.count()):
                page = self.stackedWidget.widget(index)
                if not self._page_is_current(page):
                    self._show_pending_page(page, 'Charts are outdated\nUpdate failed. Click Update charts to retry.')

    def _update_job_controls(self):
        if not hasattr(self, '_cancel_job'):
            return
        importing = self._jobs.busy('import') or self._import_commit is not None
        busy = importing or self._jobs.busy() or self._refresh_timer.isActive() or self._cutoff_timer.isActive()
        self._job_progress.setVisible(busy)
        self._cancel_job.setVisible(busy)
        self.annot_label.setVisible(not busy)
        self.status_label.setVisible(not busy)
        for action in getattr(self, '_import_actions', []):
            action.setEnabled(not importing)
        self.sidebar._import_button.setEnabled(not importing)

    def cancel_processing(self):
        commit, self._import_commit = self._import_commit, None
        self._import_commit_timer.stop()
        self._refresh_timer.stop()
        self._cutoff_timer.stop()
        self._jobs.cancel()
        self._deferred_results.clear()
        self._auto_paused = True
        self._set_processing_state('cancelled', 'Cancelled. No unfinished results were applied.')
        self.sidebar.set_pending_details(self._trend_dirty, 'Summary is outdated. Update charts to refresh.')
        message = (f'Import cancelled. Kept {commit["completed"]} imported file(s); remaining files skipped.'
                   if commit else 'Cancelled. Click Update charts to retry.')
        self.statusBar().showMessage(message, 5000)
        self._pending_import_notice = None
        for index in range(self.stackedWidget.count()):
            page = self.stackedWidget.widget(index)
            if not self._page_is_current(page):
                self._show_pending_page(page, 'Update cancelled\nClick Update charts to refresh this view.')
        self._update_job_controls()

    def _start_import(self, paths, raw):
        if self._jobs.busy('import') or self._import_commit is not None:
            self.statusBar().showMessage('An import is already running. Wait or cancel it first.', 4000)
            return
        paths = [os.fspath(paths)] if isinstance(paths, (str, os.PathLike)) else [os.fspath(path) for path in paths]
        if not paths:
            return
        self._pending_import_notice = None
        self._set_processing_state('importing', f'Importing {len(paths)} file(s)')
        self._job_progress.setValue(0)
        function, argument = ((import_recording, paths[0]) if len(paths) == 1
                              else (import_recordings, paths))
        self._jobs.start('import', partial(function, argument, raw, self.cutoff_spin.value()))

    def _accept_import(self, datasets):
        self._refresh_suspended = True
        try:
            if self.trendplot._analysis_active:
                self.trendplot.toggle_analysis(self.datalist, self.plottype, self.dataset_colors)
            for data in datasets:
                self.set_datalist(data)
                self.add_trend_data(data, self.plottype, len(self.datalist))
        finally:
            self._refresh_suspended = False
        dirty = any(data._filtered_cutoff != self.cutoff_spin.value() for data in self.datalist)
        self._queue_chart_refresh(trend=dirty)

    def _accept_import_batch(self, result):
        # Diagnostics retain filenames/errors, never the large data buffers of
        # files which may later be cancelled before being added to the GUI.
        self._last_import_summary = dict(total=result.total,
                                        files=tuple(path for path, _ in result.recordings),
                                        errors=tuple(result.errors))
        self._jobs.cancel('analysis')
        self._refresh_timer.stop()
        self._import_commit = dict(result=result, completed=0, added=0)
        self._update_job_controls()
        self._import_commit_timer.start(0)

    def _commit_import_file(self):
        state = self._import_commit
        if state is None or self._closing or self._refresh_suspended:
            return
        result = state['result']
        index = state['completed']
        if index >= len(result.recordings):
            self._finish_import_batch(state)
            return
        path, datasets = result.recordings[index]
        try:
            self._accept_import(datasets)
            state['completed'] += 1
            state['added'] += len(datasets)
            dirty = any(data._filtered_cutoff != self.cutoff_spin.value() for data in self.datalist)
            self._mark_charts_outdated(trend=dirty)
            self._set_processing_state('importing', f'Adding {os.path.basename(path)}')
            self._job_progress.setValue(int(100 * state['completed'] / len(result.recordings)))
            self.statusBar().showMessage(
                f'Added file {state["completed"]}/{len(result.recordings)}: {os.path.basename(path)}')
            self._import_commit_timer.start(0)
        except Exception as error:
            self._import_commit = None
            self._background_error('import', str(error))
            self._update_job_controls()

    def _finish_import_batch(self, state):
        from qfluentwidgets import InfoBar
        result = state['result']
        self._import_commit = None
        summary = (f'Imported {state["completed"]}/{result.total} files; '
                   f'{state["added"]} recording(s) added')
        if result.errors:
            details = '\n'.join(f'{path}: {error}' for path, error in result.errors)
            visible = '\n'.join(f'{os.path.basename(path)}: {error[:180]}'
                                for path, error in result.errors[:3])
            if len(result.errors) > 3:
                visible += f'\n{len(result.errors) - 3} more failure(s). Hover for details.'
            notice = InfoBar.warning(title=summary, content=visible, duration=10000,
                                     isClosable=True, parent=self)
            notice.setToolTip(details)
        if state['completed']:
            self._pending_import_notice = summary
            dirty = any(data._filtered_cutoff != self.cutoff_spin.value() for data in self.datalist)
            self._queue_chart_refresh(trend=dirty)
        else:
            self._set_processing_state('error', summary)
        self.statusBar().showMessage(summary, 10000)
        self._update_job_controls()

    def _reset_processing(self):
        if not hasattr(self, '_jobs'):
            return
        self._refresh_timer.stop()
        self._cutoff_timer.stop()
        self._import_commit_timer.stop()
        self._import_commit = None
        self._pending_import_notice = None
        self._jobs.cancel()
        self._prepared = None
        self._rendered_keys.clear()
        self._trend_dirty = False
        self.sidebar.set_pending_details(False)
        self._deferred_results.clear()
        for index in range(self.stackedWidget.count()):
            self._show_current_page(self.stackedWidget.widget(index))
        self._analyze_btn.setEnabled(False)
        self._set_processing_state('empty', 'Add a recording to begin')
        self._update_job_controls()

    def _begin_report_processing(self):
        self._refresh_suspended = True
        self._refresh_timer.stop()
        self._cutoff_timer.stop()
        self._import_commit_timer.stop()
        self._jobs.cancel('analysis')
        self.set_cutoff(self.cutoff_spin.value())

    def _end_report_processing(self):
        message = self.statusBar().currentMessage()
        if message.startswith('Report generated:'):
            self._pending_report_notice = message
        self._refresh_suspended = False
        deferred, self._deferred_results = self._deferred_results, []
        for kind, payload in deferred:
            if kind == 'import':
                self._apply_background_result(kind, payload)
        if self._import_commit is not None:
            self._import_commit_timer.start(0)
        self._queue_chart_refresh(trend=True)

    def closeEvent(self, event):
        if hasattr(self, '_jobs'):
            self._closing = True
            self._refresh_timer.stop()
            self._cutoff_timer.stop()
            self._import_commit_timer.stop()
            self._import_commit = None
            self._jobs.cancel()
            if self._jobs.active_count:
                self.statusBar().showMessage('Cancelling background work before closing...')
                event.ignore()
                return
        super().closeEvent(event)

    def _processing_drained(self):
        if self._closing:
            QTimer.singleShot(0, self.close)
