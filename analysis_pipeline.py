"""Preparation on isolated numeric snapshots; no Qt widgets or chart objects."""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass

import numpy as np
from matplotlib.path import Path

from accdata import AccData, RawData
from background import Cancelled
from sci_calculation import (ProcessASTM, detect_z_acceleration_reversal,
                             detect_xy_acceleration_reversal, sampling_frequency)

ASTM_DURATIONS = (.01, .02, .05, .1, .2, .3, .4, .5, 1, 2, 4, 5, 6, 7, 8, 9, 14)
GB_DURATIONS = (.01, .05, .08, .1, .2, .3, .4, .5, 1, 2, 4)
FRAME_ATTR = {'Raw': 'data', 'Filter': 'filtered_data', 'Standard': 'std_data'}


def snapshot(data):
    result = copy.copy(data)
    # Filtering creates new frames; the existing filtered frames are read-only
    # and may be reused when their cutoff is already correct.
    result.data = data.data.copy(deep=True)
    return result


def classify_zone(data, polygons, duration):
    time = data.iloc[:, 0].to_numpy(dtype=float)
    if len(time) < 2:
        return None, None, None
    points = data.iloc[:, [1, 3]].to_numpy(dtype=float)
    inside = np.zeros(len(time), dtype=bool)
    for zone in (5, 4, 3, 2):
        for vertices in polygons.get(zone, ()):
            inside |= Path(vertices).contains_points(points)
        edges = np.diff(np.r_[False, inside, False].astype(np.int8))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1
        qualifying = np.flatnonzero(time[ends] - time[starts] > duration)
        if len(qualifying):
            index = qualifying[0]
            return zone, float(time[starts[index]]), float(time[ends[index]])
    return 1, None, None


@dataclass
class AnalysisRequest:
    key: tuple
    datasets: list  # (original object identity, isolated AccData snapshot)
    selected: str
    cutoff: int
    plot_type: str
    z_type: str
    xy_type: str
    deadband: float
    zone_polygons: dict
    zone_duration: float


def prepare_analysis(request, token, progress):
    filtered = []
    selected = None
    for index, (identity, data) in enumerate(request.datasets):
        token.check()
        data.filter_data(request.cutoff)
        filtered.append((identity, data))
        if data.filename == request.selected:
            selected = data
        progress(int(25 * (index + 1) / len(request.datasets)), 'Filtering recordings')
    if selected is None:
        raise ValueError('The selected recording is no longer available')
    frame = getattr(selected, FRAME_ATTR[request.plot_type])
    fs = sampling_frequency(frame)
    durations = sorted(set(ASTM_DURATIONS + GB_DURATIONS))
    sustained = {}
    for index, duration in enumerate(durations):
        token.check()
        sustained[duration] = ProcessASTM(frame, max(1, round(fs * duration)))
        progress(25 + int(40 * (index + 1) / len(durations)), 'Calculating sustained acceleration')
    gb = {duration: (sustained[duration][0][1:], sustained[duration][1][1:])
          for duration in GB_DURATIONS}
    token.check()
    z_frame = getattr(selected, FRAME_ATTR[request.z_type])
    z_events = {'z': detect_z_acceleration_reversal(
        z_frame.iloc[:, 3].to_numpy(), z_frame.iloc[:, 0].to_numpy(), selected.fs)}
    progress(75, 'Detecting Z reversals')
    xy_frame = getattr(selected, FRAME_ATTR[request.xy_type])
    xy_result = detect_xy_acceleration_reversal(
        xy_frame.iloc[:, 1].to_numpy(), xy_frame.iloc[:, 2].to_numpy(),
        xy_frame.iloc[:, 0].to_numpy(), selected.fs, deadband=request.deadband)
    progress(85, 'Detecting X/Y reversals')
    zone_frame = selected.get_1hz_standard_frame()
    zone = classify_zone(zone_frame, request.zone_polygons, request.zone_duration)
    progress(100, 'Analysis ready')
    return dict(key=request.key, datasets=filtered, selected=selected, frame=frame,
                astm=sustained, gb=gb, z_frame=z_frame, z_events=z_events,
                xy_frame=xy_frame, xy_events={'x': xy_result['x_reversals'],
                                             'y': xy_result['y_reversals']},
                zone_frame=zone_frame, zone=zone)


def import_recording(path, raw, cutoff, token, progress):
    token.check()
    progress(5, 'Reading recording')
    data = RawData(path).export_data() if raw else (AccData(path, cutoff=cutoff),)
    for index, item in enumerate(data):
        token.check()
        item.filter_data(cutoff)
        progress(50 + int(50 * (index + 1) / len(data)), 'Preparing imported signals')
    return list(data)


@dataclass
class ImportBatch:
    total: int
    recordings: list  # (source path, complete list of signals for that file)
    errors: list  # (source path, error message)


def import_recordings(paths, raw, cutoff, token, progress):
    """Parse a selection in order, retaining per-file errors and cancellation."""
    recordings, errors = [], []
    total = len(paths)
    for index, path in enumerate(paths):
        token.check()
        name = os.path.basename(path)
        def file_progress(percent, message):
            progress(int((index + percent / 100) * 100 / total),
                     f'File {index + 1}/{total}: {name} — {message}')
        try:
            datasets = import_recording(path, raw, cutoff, token, file_progress)
            token.check()
            recordings.append((os.fspath(path), datasets))
        except Cancelled:
            raise
        except Exception as error:
            errors.append((os.fspath(path), str(error)))
        progress(int((index + 1) * 100 / total), f'Checked file {index + 1}/{total}: {name}')
    token.check()
    return ImportBatch(total, recordings, errors)
