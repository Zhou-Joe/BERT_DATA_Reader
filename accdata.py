import pandas as pd
import re
import math
from sci_calculation import *
import numpy as np
import os
import io
class AccData:
    def __init__(self, rawdata, path=None, is_raw=False, cutoff=5, *args,
                 sampling_rate=None, **kwargs):
        self._sampling_frequency_hint = sampling_rate
        if is_raw:
            self.cutoff=cutoff
            self.rawdata=rawdata.copy()
            self.data=rawdata.copy()
            self.filename=path
            self.filtered_data = pd.DataFrame()
            self.std_data = pd.DataFrame()
            self._filtered_cutoff = None  # cache guard for filter_data
            self.fs = 500
            self.pitch_angle = 0
            self.seatback_angle = 0
            self.roll_angle = 0
            self.yaw_angle = 0
            self.zero_x = 0
            self.zero_y = 0
            self.zero_z = 1
            self.offset_x = 0
            self.offset_y = 0
            self.offset_z = 0
            self.x_avg = 0
            self.y_avg = 0
            self.z_avg = 0
            self.col_idx = 0
            self.start_idx = 0
            self.columns = list(rawdata.columns)
            self.column_row_content = ' '.join(self.columns)
            self.calculate_sampling_frequency()
            self.get_offset()
            self.filter_data(cutoff=cutoff)

        else:
            self.cutoff=cutoff
            self.rawdata = pd.read_csv(rawdata, header=None, index_col=False, sep='\t', low_memory=False)
            # If data collapsed into 1 column, the file likely uses whitespace separators instead of tabs
            if self.rawdata.shape[1] == 1:
                # Preprocess: convert space-separated data lines to tab-separated
                # while preserving header lines (which contain spaces in text)
                with open(rawdata, 'r') as f:
                    lines = f.readlines()
                # First pass: detect number of data columns from the first numeric line
                num_data_cols = 4  # default: Time, X, Y, Z
                for line in lines:
                    stripped = line.strip()
                    if stripped and re.match(r'^[\d.\-+eE\s]+$', stripped) and re.search(r'\d', stripped):
                        num_data_cols = len(stripped.split())
                        break
                # Second pass: convert lines
                converted_lines = []
                for line in lines:
                    stripped = line.strip()
                    if not stripped:
                        converted_lines.append(line)
                        continue
                    # Check if line is a numeric data line
                    # Must contain only digits, dots, minus, plus, e, E, and spaces
                    if re.match(r'^[\d.\-+eE\s]+$', stripped) and re.search(r'\d', stripped):
                        # Data line - split by whitespace and rejoin with tabs
                        parts = stripped.split()
                        converted_lines.append('\t'.join(parts) + '\n')
                    else:
                        # Header line - keep text in column 0, pad with empty tabs for remaining columns
                        converted_lines.append(stripped + '\t' * (num_data_cols - 1) + '\n')
                buffer = io.StringIO(''.join(converted_lines))
                self.rawdata = pd.read_csv(buffer, header=None, index_col=False, sep='\t', engine='python')
            self.filename = os.path.basename(rawdata)
            self.filtered_data = pd.DataFrame()
            self.std_data = pd.DataFrame()
            self._filtered_cutoff = None  # cache guard for filter_data
            self.fs = 500  # Default sampling frequency, will be updated based on actual data
            self.start_idx = 0
            self.col_idx = 0
            self.get_core_idx()
            self.init_angle()
            self.get_angle()
            self.get_colID()
            self.parsie_cols()
            if self._sampling_frequency_hint is None:
                for value in self.rawdata.iloc[:self.start_idx, 0]:
                    match = re.fullmatch(r'Sampling Rate \(Hz\):\s*([\d.]+)', str(value))
                    if match:
                        self._sampling_frequency_hint = float(match[1])
            self.calculate_sampling_frequency()  # Calculate actual sampling frequency
            self.get_offset()
            self.filter_data(cutoff=cutoff)
    def init_angle(self):
        self.pitch_angle = 0
        self.yaw_angle = 0
        self.roll_angle = 0
        self.seatback_angle = 0

    def get_angle(self):
        for i in range(min(10, len(self.rawdata))):
            raw_line = None
            try:
                raw_line = self.rawdata.iloc[i, 0]
                # 角度用 float 解析：int() 遇 '5.0' 等浮点格式会抛 ValueError
                # 被下面 except 吞掉，角度静默保持 0，零位点与 std_data 偏移
                # 随之全错。reformat 导出用 '{}' 格式化，本就兼容浮点。
                if "Initial Pitch Angle: " in str(raw_line):
                    self.pitch_angle = float(re.sub(r"Initial Pitch Angle: ", '', raw_line))
                elif "Seatback Angle: " in str(raw_line):
                    self.seatback_angle = float(re.sub(r"Seatback Angle: ", '', raw_line))

                elif "Initial Roll Angle: " in str(raw_line):
                    self.roll_angle = float(re.sub(r"Initial Roll Angle: ", '', raw_line))
                elif "Initial Yaw Angle: " in str(raw_line):
                    self.yaw_angle = float(re.sub(r"Initial Yaw Angle: ", '', raw_line))
            except Exception as e:
                # 附上原始行内容，便于定位是哪一行、什么格式导致解析失败。
                print(f"Angle parse error at header row {i} ({raw_line!r}): {e}")

    def get_offset(self):
        [self.zero_x, self.zero_y, self.zero_z] = coordTransform(self.pitch_angle, self.seatback_angle, self.roll_angle, self.yaw_angle)

        # Zero-point / averaging changed — these feed the filter offsets, so
        # the cached filter frames are now stale. Invalidate.
        self._filtered_cutoff = None

        # Calculate zero offset based on first 1 second of data
        # Use actual sampling frequency to determine number of points
        num_points = min(round(self.fs), len(self.data))  # First 1 second = fs points
        
        if num_points > 0:
            self.x_avg = self.data.iloc[:num_points, 1].mean()
            self.y_avg = self.data.iloc[:num_points, 2].mean()
            self.z_avg = self.data.iloc[:num_points, 3].mean()
            print(f"Zero offset calculated using first {num_points} points ({num_points/self.fs:.2f}s)")
        else:
            # Fallback if not enough data
            self.x_avg = self.data.iloc[:, 1].mean()
            self.y_avg = self.data.iloc[:, 2].mean()
            self.z_avg = self.data.iloc[:, 3].mean()
            print(f"Warning: Not enough data for 1 second offset, using all {len(self.data)} points")
        
        # get deviate value:
        self.offset_x = self.x_avg - self.zero_x
        self.offset_y = self.y_avg - self.zero_y
        self.offset_z = self.z_avg - self.zero_z

    def get_core_idx(self):
        # Locate the header independently of the numeric time origin. Looking
        # for a 0 -> positive transition dropped pre-trigger data and failed on
        # recordings whose first timestamp was nonzero.
        headers = self.rawdata.iloc[:, 0].astype(str).str.contains(
            "SARCcolumnIDs", regex=False)
        matches = np.flatnonzero(headers.to_numpy())
        if not len(matches):
            raise ValueError("Missing SARCcolumnIDs header in acceleration file")
        self.col_idx = int(matches[0])
        numeric = self.rawdata.iloc[self.col_idx + 1:, :4].apply(
            pd.to_numeric, errors="coerce")
        rows = np.flatnonzero(numeric.notna().all(axis=1).to_numpy())
        if not len(rows):
            raise ValueError("No numeric acceleration samples found")
        self.start_idx = self.col_idx + 1 + int(rows[0])

    def get_colID(self, column_str=None):
        try:
            self.column_row_content = re.sub(r'SARCcolumnIDs: ', "", self.rawdata.iloc[self.col_idx, 0])
            # 按任意空白拆分并自动去空项：旧 .split(' ') 遇 'Time  X'（连续
            # 空格）会产生空串列名，各通道取错列/首列被覆盖。
            self.columns = re.sub(r'SARCcolumnIDs: ', '', self.rawdata.iloc[self.col_idx, 0]).split()
        except Exception as e:
            print (e)

    def set_colID(self, column_str):
        self.column_row_content = column_str
        self.columns = self.column_row_content.split()
        self.data.columns = self.columns

    def parsie_cols(self):
        # 表头/数据列数一致性校验：列数不符或列名含空串时，旧代码会在下面
        # iloc 取列处抛晦涩的 IndexError；这里提前抛带数字的 ValueError，
        # 由 open_accel_data 的状态栏错误提示直接承接展示。
        if len(self.columns) > self.rawdata.shape[1] or "" in self.columns:
            raise ValueError(
                f"表头列数({len(self.columns)})与数据列数({self.rawdata.shape[1]})不符")
        if len(self.columns) != 4 or self.columns[0].lower() != "time":
            raise ValueError("Expected Time and three acceleration columns")
        self.columns[0] = "Time"
        self.data = self.rawdata.iloc[self.start_idx:, :4].copy()
        self.data.columns = self.columns
        self.data = self.data.apply(pd.to_numeric, errors="raise").astype(float)
        self.data.reset_index(drop=True, inplace=True)
        if self.data.empty or not np.isfinite(self.data.to_numpy()).all():
            raise ValueError("Acceleration data must contain finite numeric samples")
        if len(self.data) > 1 and (np.diff(self.data['Time']) <= 0).any():
            raise ValueError("Timestamps must be strictly increasing")
        # Source data changed — invalidate the filter cache so the next
        # filter_data call actually recomputes.
        self._filtered_cutoff = None
    
    def calculate_sampling_frequency(self):
        """Calculate actual sampling frequency from time data"""
        if self._sampling_frequency_hint is not None:
            rate = float(self._sampling_frequency_hint)
            if not np.isfinite(rate) or rate < 1:
                raise ValueError('Sampling rate must be finite and at least 1 Hz')
            self.fs = int(rate) if rate.is_integer() else rate
            self.data.attrs['sampling_rate'] = self.fs
            return
        try:
            if len(self.data) > 1:
                time_interval = float(np.median(np.diff(self.data.iloc[:, 0])))
                if time_interval > 0:
                    measured = 1.0 / time_interval
                    # Convert time interval to sampling frequency
                    self.fs = int(round(measured))
                    print(f"Detected sampling frequency: {self.fs} Hz (time interval: {time_interval:.6f}s)")
                else:
                    self.fs = 500  # Default fallback
                    print("Warning: Could not detect sampling frequency, using default 500 Hz")
            else:
                self.fs = 500  # Default fallback
                print("Warning: Not enough data points to detect sampling frequency, using default 500 Hz")
        except Exception as e:
            self.fs = 500  # Default fallback
            print(f"Error calculating sampling frequency: {e}, using default 500 Hz")

    def filter_data(self, cutoff=500):
        # Cache guard: if the data was already filtered at this exact cutoff
        # (and the source data hasn't changed since), recompute is pure waste —
        # it costs ~3.5 ms/dataset and runs on EVERY replot_all / plot-type
        # switch even though the frames are identical. Skip when unchanged.
        if (getattr(self, "_filtered_cutoff", None) == cutoff
                and not self.filtered_data.empty and not self.std_data.empty):
            return
        filtered = pd.DataFrame({'Time': self.data['Time']})
        standard = pd.DataFrame({'Time': self.data['Time']})
        i = 0
        zero_list = [self.x_avg, self.y_avg, self.z_avg]
        offset_list=[self.offset_x, self.offset_y, self.offset_z]
        for cols in list(self.data.columns):
            if not "Time" in cols:
                values = butter_lowpass_filter(self.data[cols], cutoff, fs=self.fs,
                                               order=4, zeropoint=zero_list[i])
                filtered[cols] = values
                standard[cols] = values - offset_list[i]
                i += 1
        filtered.iloc[:, 1:] = filtered.iloc[:, 1:].round(4)
        standard.iloc[:, 1:] = standard.iloc[:, 1:].round(4)
        self.filtered_data = filtered
        self.std_data = standard
        if self._sampling_frequency_hint is not None:
            filtered.attrs['sampling_rate'] = self.fs
            standard.attrs['sampling_rate'] = self.fs
        self.cutoff = cutoff
        # Remember the cutoff these frames were produced at so the next call
        # with the same cutoff is a no-op. Invalidated in reformat (data change)
        # and get_offset (zero-point change).
        self._filtered_cutoff = cutoff

    def get_1hz_standard_frame(self):
        """Return a FRESH 1 Hz standard-filtered frame for the Acc Zone page.

        Acc Zone analysis requires 1 Hz data independent of the global cutoff.
        This deliberately does NOT mutate self.std_data / self.filtered_data /
        self._filtered_cutoff, because those shared frames feed the Trend/GB/ASTM
        pages — overwriting them here would re-filter those pages at 1 Hz too.
        Mirrors how filter_data() builds std_data, just into a local frame.
        """
        zero_list = [self.x_avg, self.y_avg, self.z_avg]
        offset_list = [self.offset_x, self.offset_y, self.offset_z]
        frame = pd.DataFrame()
        frame['Time'] = self.data['Time']
        i = 0
        for cols in list(self.data.columns):
            if "Time" not in cols:
                frame[cols] = (butter_lowpass_filter(
                    self.data[cols], 1, fs=self.fs, order=4,
                    zeropoint=zero_list[i]) - offset_list[i])
                i += 1
        frame.iloc[:, 1:] = frame.iloc[:, 1:].round(4)
        if self._sampling_frequency_hint is not None:
            frame.attrs['sampling_rate'] = self.fs
        return frame

    def set_angle(self, pitch_angle, seatback_angle, roll_angle, yaw_angle):
        self.pitch_angle = pitch_angle
        self.seatback_angle = seatback_angle
        self.roll_angle = roll_angle
        self.yaw_angle = yaw_angle
        self.get_offset()

    def _export_frame(self):
        """Build the mixed header/body table without changing numeric frames."""
        rows = [
            f"Initial Pitch Angle: {self.pitch_angle}",
            f"Seatback Angle: {self.seatback_angle}",
            f"Initial Roll Angle: {self.roll_angle}",
            f"Initial Yaw Angle: {self.yaw_angle}",
            f"SARCcolumnIDs: {' '.join(self.columns)}",
            (f"Sampling Rate (Hz): {self.fs:g}"
             if self._sampling_frequency_hint is not None else ""),
        ]
        header = pd.DataFrame(np.nan, index=range(6), columns=self.columns,
                              dtype=object)
        header['Time'] = rows
        return pd.concat([header, self.data.astype(object)], ignore_index=True)

    def reformat(self, overwrite=False, setting_angle=False, pitch_angle=0,
                 seatback_angle=0, roll_angle=0, yaw_angle=0, cutoff=5):
        if setting_angle:
            self.set_angle(pitch_angle, seatback_angle, roll_angle, yaw_angle)
        self.new_data = self._export_frame()
        if overwrite:
            self.rawdata = self.new_data.copy()
            self.get_core_idx()
            self.get_colID()
            self.parsie_cols()
            self.calculate_sampling_frequency()
            self.get_offset()
            self.filter_data(cutoff=cutoff)

    def edit_data(self, method, value_array, cutoff):
        values = np.asarray(value_array, dtype=float)
        if values.shape != (3,) or not np.isfinite(values).all():
            raise ValueError("Enter three finite acceleration values")
        if method not in ('add', 'multiply'):
            raise ValueError(f"Unknown edit method: {method}")
        frame = self.data.copy()
        axes = frame.iloc[:, 1:].to_numpy()
        frame.iloc[:, 1:] = axes + values if method == 'add' else axes * values
        self.data = frame.round(4)
        self.reformat(overwrite=True, cutoff=cutoff)

    def truncate_data(self, starttime, endtime, cutoff):
        """Keep [start, end) by timestamp, preserving fractional/pretrigger time."""
        if not np.isfinite([starttime, endtime]).all() or endtime <= starttime:
            raise ValueError("End time must be greater than start time")
        time = self.data['Time'].to_numpy()
        lo, hi = np.searchsorted(time, [starttime, endtime], side='left')
        frame = self.data.iloc[lo:hi].copy().reset_index(drop=True)
        if len(frame) < 2:
            raise ValueError("The selected interval must contain at least two samples")
        frame['Time'] = frame['Time'] - starttime
        self.data = frame
        self.reformat(overwrite=True, cutoff=cutoff)

    def reset_data(self):
        self.parsie_cols()
        self.calculate_sampling_frequency()
        self.get_offset()
        self.filter_data(cutoff=self.cutoff)

    def get_data_stats(self, data):
        rows = []
        for column, zero in zip(data.columns[1:],
                                (self.zero_x, self.zero_y, self.zero_z)):
            series = data[column]
            rows.append([data.loc[series.idxmax(), 'Time'],
                         data.loc[series.idxmin(), 'Time'],
                         series.max(), series.min(), zero])
        return pd.DataFrame(rows, index=['X', 'Y', 'Z'],
                            columns=['t (Max)', 't (Min)', 'Max', 'Min', 'Zero Pos'])

    def get_angle_info(self):  # output string value
        str_info = ("{}\nPitch: {}°  Seatback: {}°  Roll: {}°  Yaw: {}°".format
                    (self.filename, self.pitch_angle, self.seatback_angle, self.roll_angle,
                     self.yaw_angle))
        return str_info

    def export_to_file(self, path):
        # 不再用裸 except 吞掉写盘异常：调用方（ExportDialog /
        # _export_all_to_sup）都在 try 内接住并计数/提示，让失败可见。
        self._export_frame().to_csv(path, sep='\t', header=False, index=False)

    def export_filter(self, path, plottype):
        # 同上：异常向上抛。plottype=='Raw'（顶栏 Raw Data 视图）此前两个分支
        # 都不命中、什么都不写却报成功，现在导出原始 self.data 帧。
        if plottype == 'Standard':
            self.std_data.to_csv(path, sep='\t', header=False, index=False)

        elif plottype == 'Filter':
            self.filtered_data.to_csv(path, sep='\t', header=False, index=False)

        elif plottype == 'Raw':
            self.data.to_csv(path, sep='\t', header=False, index=False)

def _csv_first_data_row(path: str, max_scan: int = 200) -> int:
    """Return the 0-based index of the first all-numeric CSV row.

    CSV exports from different DAQ tools have varying metadata header lengths
    (this project's reference file has 23 rows; others may have more or fewer).
    Hardcoding ``skiprows=23`` breaks on non-conforming files. This scans up to
    ``max_scan`` rows and returns the index of the first row whose every
    non-empty comma-field parses as a float — i.e. the first real data row.

    Falls back to 23 (the known-good count) if no numeric row is found within
    the scan window (e.g. a malformed/truncated file).
    """
    try:
        with open(path, 'r', encoding='utf-8-sig', errors='replace') as f:
            for i, line in enumerate(f):
                if i >= max_scan:
                    break
                parts = [p.strip() for p in line.split(',') if p.strip()]
                if not parts:
                    continue
                try:
                    [float(p) for p in parts]
                    return i
                except ValueError:
                    continue
    except Exception:
        pass
    return 23


class SensorConnectData:
    """SensorConnect 软件导出的单传感器 CSV（三通道 X/Y/Z 加速度）。

    实测文件结构（2.9M 行 / 217MB 样例）：
      - FILE_INFO / SESSION_INFO / DEVICE_INFO / CHANNEL_INFO 元数据块；
        CHANNEL_INFO 表的 SampleRate 列给出采样率（如 ``512Hz``）；
      - ``DATA_START`` 行之后是列头行 ``Time,<addr>:ch1,<addr>:ch2,<addr>:ch3``，
        再往后为「相对秒时间(9 位小数) + 三通道加速度(g)」逗号分隔数据。

    元数据用逐行扫描只读到 DATA_START 为止（文件可能数百 MB）；数据块
    一次 read_csv 载入。输出单个 AccData（列 Time/accel-x/y/z，fs 显式
    覆盖 —— is_raw 构造默认 fs=500，512Hz 这类非整百采样率必须纠正，
    否则滤波与持续窗的时长换算全部偏差 ~2.4%）。
    """

    def __init__(self, path, *args, **kwargs):
        self.filename = path
        data_header_row = None
        fs_hint = None
        try:
            with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
                row = 0
                in_channels = False
                for line in f:
                    row += 1
                    stripped = line.strip()
                    if stripped == "CHANNEL_INFO":
                        in_channels = True
                        continue
                    if in_channels and fs_hint is None:
                        # 通道行形如 "40305:ch1,Timeseries,512Hz,,,,"
                        parts = [p.strip() for p in stripped.split(",")]
                        if len(parts) >= 3 and "Hz" in parts[2]:
                            try:
                                fs_hint = float(re.sub(r"[^\d.]", "", parts[2]))
                            except ValueError:
                                fs_hint = None
                    if stripped == "DATA_START":
                        data_header_row = row  # 下一行是 Time,ch1..ch3 列头
                        break
        except Exception as e:
            raise RuntimeError(f"无法读取 SensorConnect 文件 {path}: {e}") from e
        if data_header_row is None:
            raise RuntimeError(
                f"不是 SensorConnect 导出格式（未找到 DATA_START）: {path}")

        raw = pd.read_csv(path, skiprows=data_header_row, header=0,
                          index_col=False, sep=",", low_memory=False)
        if raw.shape[1] < 4 or len(raw) < 2:
            raise RuntimeError(
                f"SensorConnect 数据不完整：需要 Time+3 通道，"
                f"实际 {raw.shape[1]} 列 / {len(raw)} 行")

        frame = pd.DataFrame()
        # Preserve source precision so 512 Hz spacing is not rounded to 500 Hz.
        frame["Time"] = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
        # 该传感器的安装方向与应用约定相反：Y、Z 通道取反（静止时 Z 由
        # ≈-0.99g 翻成 ≈+0.99g，符合 +1g 重力约定；Y 同步翻转保持手性）。
        axis_signs = (1.0, -1.0, -1.0)
        for i, (axis, sign) in enumerate(
                zip(("accel-x", "accel-y", "accel-z"), axis_signs)):
            frame[axis] = sign * pd.to_numeric(raw.iloc[:, i + 1],
                                               errors="coerce")
        frame = frame.dropna().reset_index(drop=True)
        if len(frame) < 2:
            raise RuntimeError("SensorConnect 数据为空（有效行不足）")
        if not np.isfinite(frame.to_numpy()).all():
            raise ValueError("SensorConnect data must contain finite numeric samples")

        # Keep the first increasing timestamp; skip duplicates and backwards
        # jitter until the sensor clock catches up.
        tt = frame["Time"].to_numpy()
        tt = np.maximum.accumulate(tt)
        keep = np.concatenate(([True], np.diff(tt) > 0))
        if not keep.all():
            frame = frame.loc[keep].reset_index(drop=True)
        frame["Time"] = tt[keep] if not keep.all() else tt
        if len(frame) < 2:
            raise ValueError("SensorConnect needs at least two increasing timestamps")

        # Prefer metadata; otherwise infer from median valid spacing.
        fs = fs_hint
        if not fs or not np.isfinite(fs) or fs <= 0:
            fs = round(1.0 / float(np.median(np.diff(frame['Time']))))

        cols_str = "Time accel-x accel-y accel-z"
        self.data = AccData(frame, is_raw=True, path=os.path.basename(path),
                            sampling_rate=int(round(fs)))
        self.data.set_colID(cols_str)
        self.fs = self.data.fs

    def export_data(self):
        return (self.data,)


class RawData:
    """DAQ CSV or six-channel tab data; read the numeric body only once."""

    def __init__(self, path, *args, **kwargs):
        import csv
        try:
            self.filename = path
            with open(path, encoding='utf-8-sig', errors='replace') as stream:
                header_lines = []
                for line in stream:
                    header_lines.append(line)
                    if len(header_lines) >= 200:
                        break
            self.is_csv_format = any(',' in line for line in header_lines[:5])
            delimiter = ',' if self.is_csv_format else '\t'
            labels = None
            sample_rate = None
            start = None
            for index, line in enumerate(header_lines):
                fields = next(csv.reader([line], delimiter=delimiter))
                fields = [field.strip() for field in fields]
                if fields and fields[0].lower().startswith('sample rate'):
                    sample_rate = float(fields[1])
                if any('GB_' in field.upper() or 'ASTM_' in field.upper()
                       or '60CM' in field.upper() or '13INCH' in field.upper()
                       for field in fields):
                    labels = fields
                nonempty = [field for field in fields if field]
                try:
                    if len(nonempty) >= 6:
                        [float(field) for field in nonempty]
                        start = index
                        break
                except ValueError:
                    pass
            if start is None:
                raise ValueError('No numeric data found in the first 200 rows')
            body = pd.read_csv(path, sep=delimiter, skiprows=start, header=None,
                               encoding='utf-8-sig').dropna(axis=1, how='all')
            body = body.apply(pd.to_numeric, errors='raise').astype(float)
            if body.empty or not np.isfinite(body.to_numpy()).all():
                raise ValueError('Data must contain finite numeric samples')
            # CSV DAQ exports include Time as the first column. Legacy tab
            # files contain six channels; seven-column tab files include Time.
            has_time = self.is_csv_format or body.shape[1] == 7
            if has_time:
                if body.shape[1] < 7:
                    raise ValueError('Expected Time and six acceleration channels')
                self.time = body.iloc[:, 0].to_numpy(copy=True)
                self.rawdata = body.iloc[:, 1:].reset_index(drop=True)
            else:
                if body.shape[1] != 6:
                    raise ValueError('Expected six acceleration channels')
                rate = sample_rate or 500.0
                if not np.isfinite(rate) or rate <= 0:
                    raise ValueError('Sample rate must be positive')
                self.time = np.arange(len(body), dtype=float) / rate
                self.rawdata = body.reset_index(drop=True)
            if len(self.time) < 2 or (np.diff(self.time) <= 0).any():
                raise ValueError('Timestamps must be strictly increasing')
            self.sample_rate = 1.0 / float(np.median(np.diff(self.time)))
            self.shape = self.rawdata.shape
            self.gb_col_indices = [0, 1, 2]
            self.astm_col_indices = [3, 4, 5]
            if labels:
                # Metadata begins with "Channel Description"; data headings
                # begin with "Time". Both precede the six channel labels.
                if len(labels) > 6 and (has_time or not any(
                        token in labels[0].upper() for token in ('GB_', 'ASTM_', '60CM', '13INCH'))):
                    labels = labels[1:]
                mappings = {'gb': {}, 'astm': {}}
                for idx, label in enumerate(labels):
                    upper = label.upper()
                    group = ('gb' if ('GB_' in upper or '60CM' in upper) else
                             'astm' if ('ASTM_' in upper or '13INCH' in upper)
                             else None)
                    if group:
                        axis = re.search(r'[-_ ]([XYZ])(?:\b|$)', upper)
                        if axis:
                            mappings[group][axis.group(1)] = idx
                for group, attr in [('gb', 'gb_col_indices'),
                                    ('astm', 'astm_col_indices')]:
                    if set(mappings[group]) != {'X', 'Y', 'Z'}:
                        raise ValueError(f'Cannot identify all three {group.upper()} axes')
                    setattr(self, attr, [mappings[group][axis] for axis in 'XYZ'])
            self.init_data()
        except Exception as error:
            raise RuntimeError(f'Cannot parse raw data file {path}: {error}') from error

    def init_data(self):
        frames = []
        for group, indices, prefix in [
                ('GB', self.gb_col_indices, '60cm'),
                ('ASTM', self.astm_col_indices, '13inch')]:
            columns = ['Time'] + [f'{prefix}_accel-{axis}' for axis in 'xyz']
            frame = pd.DataFrame(np.column_stack([
                self.time, self.rawdata.iloc[:, indices].to_numpy()]),
                columns=columns)
            frames.append(AccData(frame, is_raw=True,
                                  path=f'({group}){os.path.basename(self.filename)}'))
        self.GB_data, self.ASTM_data = frames

    def export_data(self):
        return self.GB_data, self.ASTM_data


class StingData: ##obselete
    def __init__(self, path, *args, **kwargs):
        self.filename = path
        try:
            self.rawdata = pd.read_csv(path, index_col=False)
            self.rawdata.dropna(inplace=True, axis=1)
            self.shape = self.rawdata.shape
            self.init_data()
        except:
            self.rawdata = pd.read_csv(path, index_col=False, header=None, skiprows=np.arange(0, 8))
            self.title= pd.read_csv(path, index_col=False, header=None, skiprows=np.arange(0, 2)).iloc[0,:]
            self.rawdata.dropna(inplace=True, axis=1)
            self.shape = self.rawdata.shape
            self.init_data2()

    def init_data(self):
        ASTM_cols_str = "Time 13inch_accel-x 13inch_accel-y 13inch_accel-z"
        ASTM_cols = ASTM_cols_str.split(' ')
        col_time = pd.DataFrame(np.arange(0, self.shape[0] * 0.002, 0.002), columns=['Time'])
        ASTM_data = pd.DataFrame()
        ASTM_data['Time'] = col_time['Time']
        for i in range(3):
            ASTM_data[ASTM_cols[i + 1]] = self.rawdata.iloc[:, i + 2]
        self.ASTM_data = AccData(ASTM_data, is_raw=True, path="(ASTM)" + os.path.basename(self.filename))
        self.ASTM_data.set_colID(ASTM_cols_str)

        self.datalist = [self.ASTM_data]

    def init_data2(self):
        ASTM_cols_str = "Time 13inch_accel-x 13inch_accel-y 13inch_accel-z"
        ASTM_cols = ASTM_cols_str.split(' ')
        col_time = pd.DataFrame(np.arange(0, self.shape[0] * 0.002, 0.002), columns=['Time'])
        ASTM_16g_data = pd.DataFrame()
        ASTM_16g_data['Time'] = col_time['Time']
        ASTM_10HZ_data = pd.DataFrame()
        ASTM_10HZ_data['Time'] = col_time['Time']
        ASTM_5HZ_data = pd.DataFrame()
        ASTM_5HZ_data['Time'] = col_time['Time']

        ASTM_16g_data[ASTM_cols[1]] = self.rawdata.iloc[:, 4]
        ASTM_16g_data[ASTM_cols[2]] = self.rawdata.iloc[:, 2]
        ASTM_16g_data[ASTM_cols[3]] = self.rawdata.iloc[:, 3]

        ASTM_10HZ_data[ASTM_cols[1]] = self.rawdata.iloc[:, 7]
        ASTM_10HZ_data[ASTM_cols[2]] = self.rawdata.iloc[:, 5]
        ASTM_10HZ_data[ASTM_cols[3]] = self.rawdata.iloc[:, 6]

        ASTM_5HZ_data[ASTM_cols[1]] = self.rawdata.iloc[:, 10]
        ASTM_5HZ_data[ASTM_cols[2]] = self.rawdata.iloc[:, 8]
        ASTM_5HZ_data[ASTM_cols[3]] = self.rawdata.iloc[:, 9]

        self.ASTM_16g_data = AccData(ASTM_16g_data, is_raw=True, path="({})".format(self.title[2])[6:] + os.path.basename(self.filename))
        self.ASTM_16g_data.set_colID(ASTM_cols_str)

        self.ASTM_10HZ_data = AccData(ASTM_10HZ_data, is_raw=True, path="({})".format(self.title[5])[6:] + os.path.basename(self.filename))
        self.ASTM_10HZ_data.set_colID(ASTM_cols_str)

        self.ASTM_5HZ_data = AccData(ASTM_5HZ_data, is_raw=True, path="({})".format(self.title[8])[6:] + os.path.basename(self.filename))
        self.ASTM_5HZ_data.set_colID(ASTM_cols_str)

        self.datalist = [self.ASTM_16g_data, self.ASTM_10HZ_data, self.ASTM_5HZ_data]

    def export_data(self):
        if len(self.datalist) == 1:
            return self.ASTM_data
        else:
            return self.ASTM_16g_data, self.ASTM_10HZ_data, self.ASTM_5HZ_data


# ---------------------------------------------------------------------------
# Auto test-boundary detection
# ---------------------------------------------------------------------------

def detect_test_boundaries(acc_data, quiet_thresh: float = 0.10,
                           win_s: float = 0.5, min_gap_s: float = 10.0,
                           min_test_s: float = 3.0, max_lead_s: float = 15.0,
                           min_active_frac: float = 0.25,
                           quiet_stub_s: float = 60.0):
    """Return proposed test windows as half-open source timestamps [start, end).

    A sustained stop separates rides; a logger time gap separates acquisitions.
    Qualify motion before adding quiet lead-in/out, so padding cannot create a
    test or disqualify a short one. Between rides share the stop at its clock
    midpoint, retaining at most max_lead_s of quiet on either end.

    min_active_frac is the required fraction of moving readings in a core.
    The denominator is at least min_test_s (reject isolated impulses) and
    at most quiet_stub_s (allow intermittent motion in long rides).
    """
    df = getattr(acc_data, "data", None)
    fs = getattr(acc_data, "fs", 500) or 500
    if df is None or len(df) < 2 or df.shape[1] < 4:
        return []
    samples = df.iloc[:, :4].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    samples = samples[np.isfinite(samples).all(axis=1)]
    if len(samples) < 2:
        return []
    t, axes = samples[:, 0], samples[:, 1:]
    dt = np.diff(t)
    if np.any(dt <= 0):
        # Overlapping source timestamps can't be represented by time windows.
        return []
    win = max(1, round(win_s * fs))
    # A gap contains no samples; never smooth across it, allocate padding in
    # it, or count the missing wall time towards a test or a quiet run.
    cuts = np.r_[0, np.flatnonzero(dt > min_gap_s) + 1, len(t)]
    blocks = list(zip(cuts[:-1], cuts[1:]))

    # Prefer a stop-length stationary window for the rest vector. A brief
    # flat acceleration plateau can look quieter than actual sensor noise.
    # If no such window is quiet, fall back to the flattest short window.
    # Sharing this reference lets a logger restart in the middle of a ride.
    longest_block = max(end - start for start, end in blocks)
    short_win = min(win, longest_block)
    stop_win = max(short_win, round(min_gap_s * fs))
    windows = [stop_win, short_win] if short_win < stop_win <= longest_block else [short_win]
    for baseline_win in windows:
        best_noise = np.inf
        for start, end in blocks:
            if end - start < baseline_win:
                continue
            block = axes[start:end]
            noise = (pd.DataFrame(block).rolling(baseline_win).std(ddof=0)
                     .to_numpy().sum(axis=1))
            i = int(np.nanargmin(noise))
            if noise[i] < best_noise:
                best_noise = noise[i]
                rest = block[i + 1 - baseline_win:i + 1].mean(axis=0)
        if best_noise < quiet_thresh:
            break
    motion = np.linalg.norm(axes - rest, axis=1)

    out = []
    for start, end in blocks:
        # Include the nominal exclusive end of the last acquired sample.
        edges = np.r_[t[start:end], t[end - 1] + 1 / fs]
        raw = motion[start:end]
        smoothed = (pd.Series(raw).rolling(win, center=True, min_periods=1)
                    .mean().to_numpy())
        moving = smoothed >= quiet_thresh
        # Require actual moving samples: a single large spike can otherwise
        # make an entire smoothing window look active.
        evidence = moving & (raw >= quiet_thresh)
        transitions = np.diff(np.r_[False, ~moving, False].astype(int))
        quiet_runs = zip(np.flatnonzero(transitions == 1),
                         np.flatnonzero(transitions == -1))

        # Each long stop ends a core and starts the next. If a stop touches a
        # file/block edge, there is no candidate on that side of the stop.
        core_start = padded_start = 0

        def add_core(core_end, padded_end):
            if padded_end - padded_start < 2:
                return  # Not enough acquired samples to trim or analyze.
            active = np.flatnonzero(evidence[core_start:core_end])
            if active.size == 0:
                return
            span = (active[-1] - active[0] + 1) / fs
            required = min_active_frac * max(min_test_s, min(span, quiet_stub_s))
            if active.size / fs >= required:
                out.append((float(edges[padded_start]), float(edges[padded_end])))

        for gs, ge in quiet_runs:
            stop_start, stop_end = edges[gs], edges[ge]
            if stop_end - stop_start < min_gap_s - 1e-9:
                continue
            midpoint = (stop_start + stop_end) / 2
            # At the file edges the single neighboring ride gets the full
            # allowed lead. Otherwise adjacent windows meet at one boundary.
            tail = min(stop_start + max(0, max_lead_s),
                       stop_end if ge == len(raw) else midpoint)
            head = max(stop_end - max(0, max_lead_s),
                       stop_start if gs == 0 else midpoint)
            add_core(gs, np.searchsorted(edges, tail))
            core_start, padded_start = ge, np.searchsorted(edges, head)
        add_core(len(raw), len(raw))
    return out
