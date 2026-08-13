import pandas as pd
import re
import math
from sci_calculation import *
import numpy as np
import os
import io
class AccData:
    def __init__(self, rawdata, path=None, is_raw=False, cutoff=5, *args, **kwargs):
        if is_raw:
            self.cutoff=cutoff
            self.rawdata=rawdata
            self.data=rawdata
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
            self.calculate_sampling_frequency()  # Calculate actual sampling frequency
            self.get_offset()
            self.filter_data(cutoff=cutoff)
    def init_angle(self):
        self.pitch_angle = 0
        self.yaw_angle = 0
        self.roll_angle = 0
        self.seatback_angle = 0

    def get_angle(self):
        for i in range(10):
            try:
                if "Initial Pitch Angle: " in str(self.rawdata.iloc[i, 0]):
                    self.pitch_angle = int(re.sub(r"Initial Pitch Angle: ", '', self.rawdata.iloc[i, 0]))
                elif "Seatback Angle: " in str(self.rawdata.iloc[i, 0]):
                    self.seatback_angle = int(re.sub(r"Seatback Angle: ", '', self.rawdata.iloc[i, 0]))

                elif "Initial Roll Angle: " in str(self.rawdata.iloc[i, 0]):
                    self.roll_angle = int(re.sub(r"Initial Roll Angle: ", '', self.rawdata.iloc[i, 0]))
                elif "Initial Yaw Angle: " in str(self.rawdata.iloc[i, 0]):
                    self.yaw_angle = int(re.sub(r"Initial Yaw Angle: ", '', self.rawdata.iloc[i, 0]))
            except Exception as e:
                print (e)

    def get_offset(self):
        [self.zero_x, self.zero_y, self.zero_z] = coordTransform(self.pitch_angle, self.seatback_angle, self.roll_angle, self.yaw_angle)

        # Zero-point / averaging changed — these feed the filter offsets, so
        # the cached filter frames are now stale. Invalidate.
        self._filtered_cutoff = None

        # Calculate zero offset based on first 1 second of data
        # Use actual sampling frequency to determine number of points
        num_points = min(self.fs, len(self.data))  # First 1 second = fs points
        
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
        for i in range(20):
            try:
                if "SARCcolumnIDs" in str(self.rawdata.iloc[i, 0]):
                    self.col_idx = i
            except Exception as e:
                print(e)

            try:
                m1 = round(float(self.rawdata.iloc[i, 0]), 3)
                m2 = round(float(self.rawdata.iloc[i + 1, 0]), 3)
            except:
                m1 = 0
                m2 = 0
            if m1 == 0 and m2 > m1:
                self.start_idx = i

        # Fallback: if start_idx is still 0 and col_idx is set, find first numeric row after header
        # Only apply fallback if start_idx wasn't successfully set (check if col_idx was found)
        if self.start_idx == 0 and self.col_idx > 0:
            # Try multiple columns and larger search range
            for col_check in [0, 1, 2]:  # Check Time column, then X, then Y
                for i in range(self.col_idx + 1, min(self.col_idx + 100, len(self.rawdata))):
                    try:
                        # Try to convert to float
                        val = float(self.rawdata.iloc[i, col_check])
                        # Check if it's a valid data point (reasonable range)
                        if -1000 <= val <= 1000:
                            self.start_idx = i
                            print(f"Found start_idx={i} at column {col_check} with value {val}")
                            break
                    except (ValueError, TypeError):
                        continue
                if self.start_idx > 0:
                    break

        try:
            re.sub(r'time', 'Time', self.rawdata.iloc[self.col_idx, 0])
        except Exception as e:
            print(e)

    def get_colID(self, column_str=None):
        try:
            self.column_row_content = re.sub(r'SARCcolumnIDs: ', "", self.rawdata.iloc[self.col_idx, 0])
            self.columns = re.sub(r'SARCcolumnIDs: ', '', self.rawdata.iloc[self.col_idx, 0]).split(' ')
        except Exception as e:
            print (e)

    def set_colID(self, column_str):
        self.column_row_content = column_str
        self.columns = self.column_row_content.split(' ')
        self.data.columns = self.columns

    def parsie_cols(self):
        self.data = pd.DataFrame()
        for i in range(len(self.columns)):
            self.data[self.columns[i]] = self.rawdata.iloc[self.start_idx:, i]
        self.data.reset_index(drop=True, inplace=True)
        if "Time" in self.columns:
            # pandas 3.0 removed errors='ignore'; coerce turns any stray
            # non-numeric row (e.g. a header that slipped past start_idx) into
            # NaN, which we then drop. By here data is past the header so this
            # is just defensive.
            self.data['Time'] = pd.to_numeric(self.data['Time'], errors='coerce')
            self.data = self.data.dropna(subset=['Time'])
            self.data['Time'] = self.data['Time'].round(decimals=4)
        self.data = self.data.round(decimals=4)
        # Source data changed — invalidate the filter cache so the next
        # filter_data call actually recomputes.
        self._filtered_cutoff = None
    
    def calculate_sampling_frequency(self):
        """Calculate actual sampling frequency from time data"""
        try:
            if len(self.data) > 1:
                # Calculate time interval from first two data points
                time_interval = self.data.iloc[1, 0] - self.data.iloc[0, 0]
                if time_interval > 0:
                    # Convert time interval to sampling frequency
                    self.fs = int(round(1.0 / time_interval))
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
        self.filtered_data['Time'] = self.data['Time']
        self.std_data['Time'] = self.data['Time']
        cutoff = cutoff
        i = 0
        zero_list = [self.x_avg, self.y_avg, self.z_avg]
        offset_list=[self.offset_x, self.offset_y, self.offset_z]
        for cols in list(self.data.columns):
            if not "Time" in cols:
                self.std_data[cols] = butter_lowpass_filter(self.data[cols], cutoff, fs=self.fs, order=4,
                                                                 zeropoint=zero_list[i])-offset_list[i]
                self.filtered_data[cols] = butter_lowpass_filter(self.data[cols], cutoff, fs=self.fs, order=4,
                                                                 zeropoint=zero_list[i])
                i += 1
        self.filtered_data = self.filtered_data.round(decimals=4)
        self.std_data = self.std_data.round(decimals=4)
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
        return frame.round(decimals=4)

    def set_angle(self, pitch_angle, seatback_angle, roll_angle, yaw_angle):
        self.pitch_angle = pitch_angle
        self.seatback_angle = seatback_angle
        self.roll_angle = roll_angle
        self.yaw_angle = yaw_angle
        self.get_offset()

    def reformat(self, overwrite=False, setting_angle=False, pitch_angle=0, seatback_angle=0, roll_angle=0, yaw_angle=0, cutoff=5):
        if setting_angle:
            self.set_angle(pitch_angle, seatback_angle, roll_angle, yaw_angle)
        self.new_data = pd.DataFrame()
        for i in self.columns:
            self.new_data[i] = self.data[i]
        self.new_data.loc[-6, 'Time'] = 'Initial Pitch Angle: {}'.format(self.pitch_angle)
        self.new_data.loc[-5, 'Time'] = 'Seatback Angle: {}'.format(self.seatback_angle)
        self.new_data.loc[-4, 'Time'] = 'Initial Roll Angle: {}'.format(self.roll_angle)
        self.new_data.loc[-3, 'Time'] = 'Initial Yaw Angle: {}'.format(self.yaw_angle)
        self.new_data.loc[-2, 'Time'] = 'SARCcolumnIDs: {}'.format(self.column_row_content)
        self.new_data.loc[-1, 'Time'] = np.nan
        self.new_data.sort_index(inplace=True)
        self.new_data.reset_index(drop=True, inplace=True) #only for export if not overwrite
        if overwrite == True:
            self.rawdata = self.new_data
            self.get_core_idx()
            self.get_colID()
            self.parsie_cols()
            self.get_offset()
            self.filter_data(cutoff=cutoff)

    def edit_data(self, method, value_array, cutoff):

        self.raw_data=pd.DataFrame()
        for i in self.columns:
            self.raw_data[i] = self.data[i]
        self.raw_data.loc[-6, 'Time'] = 'Initial Pitch Angle: {}'.format(self.pitch_angle)
        self.raw_data.loc[-5, 'Time'] = 'Seatback Angle: {}'.format(self.seatback_angle)
        self.raw_data.loc[-4, 'Time'] = 'Initial Roll Angle: {}'.format(self.roll_angle)
        self.raw_data.loc[-3, 'Time'] = 'Initial Yaw Angle: {}'.format(self.yaw_angle)
        self.raw_data.loc[-2, 'Time'] = 'SARCcolumnIDs: {}'.format(self.column_row_content)
        self.raw_data.loc[-1, 'Time'] = np.nan
        self.raw_data.sort_index(inplace=True)
        self.raw_data.reset_index(drop=True, inplace=True)  # only for export if not overwrite
        self.rawdata=self.raw_data
        if method == 'add':
            for i in range(3):
                self.rawdata.iloc[:,i+1] = self.rawdata.iloc[:,i+1].values + value_array[i]
        if method == 'multiply':
            for i in range(3):
                self.rawdata.iloc[:, i + 1] = self.rawdata.iloc[:, i + 1].values * value_array[i]


        self.rawdata=self.rawdata.round(decimals=4)

        self.get_core_idx()
        self.get_colID()
        self.parsie_cols()
        self.get_offset()
        self.filter_data(cutoff=cutoff)



    def truncate_data(self, starttime, endtime, cutoff):
        # Negative start_row would slice from the END of the series (Python
        # negative indexing), so clamp to 0 — a negative starttime (pre-trigger
        # data) means "start from the beginning".
        start_row = max(0, int(starttime) * self.fs)
        end_row = int(endtime) * self.fs
        self.new_data=pd.DataFrame()
        for i in self.columns:
            self.new_data[i] = self.data[i][start_row:end_row]
        self.new_data['Time']=self.new_data['Time'].values-starttime
        self.new_data.loc[-6, 'Time'] = 'Initial Pitch Angle: {}'.format(self.pitch_angle)
        self.new_data.loc[-5, 'Time'] = 'Seatback Angle: {}'.format(self.seatback_angle)
        self.new_data.loc[-4, 'Time'] = 'Initial Roll Angle: {}'.format(self.roll_angle)
        self.new_data.loc[-3, 'Time'] = 'Initial Yaw Angle: {}'.format(self.yaw_angle)
        self.new_data.loc[-2, 'Time'] = 'SARCcolumnIDs: {}'.format(self.column_row_content)
        self.new_data.loc[-1, 'Time'] = np.nan
        self.new_data.sort_index(inplace=True)
        self.new_data.reset_index(drop=True, inplace=True) #only for export if not overwrite
        self.rawdata = self.new_data
        self.get_core_idx()
        self.get_colID()
        self.parsie_cols()
        self.get_offset()
        self.filtered_data=pd.DataFrame()
        self.std_data=pd.DataFrame()
        self.filter_data(cutoff=cutoff)
    def reset_data(self):
        self.parsie_cols()
    def get_data_stats(self, data):  # output dataframe value
        minmax = [[data.iloc[:, 1].idxmax() / self.fs, data.iloc[:, 1].idxmin() / self.fs, data.iloc[:, 1].max(),
                   data.iloc[:, 1].min(), self.zero_x],
                  [data.iloc[:, 2].idxmax() / self.fs, data.iloc[:, 2].idxmin() / self.fs, data.iloc[:, 2].max(),
                   data.iloc[:, 2].min(), self.zero_y],
                  [data.iloc[:, 3].idxmax() / self.fs, data.iloc[:, 3].idxmin() / self.fs, data.iloc[:, 3].max(),
                   data.iloc[:, 3].min(), self.zero_z]]
        return pd.DataFrame(minmax, index=['X', 'Y', 'Z'], columns=['t (Max)', 't (Min)','Max',  'Min', 'Zero Pos'])

    def get_angle_info(self):  # output string value
        str_info = ("{}\nPitch: {}°  Seatback: {}°  Roll: {}°  Yaw: {}°".format
                    (self.filename, self.pitch_angle, self.seatback_angle, self.roll_angle,
                     self.yaw_angle))
        return str_info

    def export_to_file(self, path):
        try:
            self.reformat(overwrite=True, setting_angle=False)
            self.rawdata.to_csv(path, sep='\t', header=False, index=False)
        except:
            pass

    def export_filter(self, path, plottype):
        try:
            if plottype == 'Standard':
                self.std_data.to_csv(path, sep='\t', header=False, index=False)

            elif plottype == 'Filter':
                self.filtered_data.to_csv(path, sep='\t', header=False, index=False)
        except:
            pass

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


class RawData:
    def __init__(self, path, *args, **kwargs):
        try:
            self.filename = path
            self.is_csv_format = False
            self.sample_rate = None  # Will be detected from header
            # Default column mapping (fallback)
            self.gb_col_indices = [0, 1, 2]
            self.astm_col_indices = [3, 4, 5]

            self.rawdata = pd.read_csv(path, header=None, index_col=False, sep='\t', low_memory=False)
            if self.rawdata.shape[1]>5:
                self.rawdata.dropna(inplace=True, axis=1)
                self.shape = self.rawdata.shape
            else:
                self.is_csv_format = True
                # Don't hardcode the header row count — different export tools
                # produce different header lengths (this file has 23 metadata
                # rows before the first numeric data line, others may differ).
                # Scan the file for the first all-numeric row and skip to there.
                skip_n = _csv_first_data_row(path)
                self.rawdata = pd.read_csv(path, index_col=False, sep=',',
                                          skiprows=skip_n, header=None).iloc[:, 1:]
                self.rawdata.dropna(inplace=True, axis=1)
                self.shape = self.rawdata.shape

            self.detect_columns(path)
            self.init_data()
        except Exception as e:
            print (e)

    def detect_columns(self, path):
        """Detect which columns belong to GB and ASTM by reading the file header."""
        try:
            if self.is_csv_format:
                # CSV format: read the header lines to find Channel Description (row 10) or data header (row 23)
                with open(path, 'r') as f:
                    lines = f.readlines()

                # Search for "Channel Description" line or data header line with GB/ASTM labels
                header_line = None
                for line in lines:
                    stripped = line.strip()
                    # Look for the data header line (e.g., "Time,Chan 0:GB_X,Chan 1:GB_Y,...")
                    if 'GB_' in stripped.upper() or 'ASTM_' in stripped.upper():
                        header_line = stripped
                        break
                    # Also look for Channel Description line
                    if 'Channel Description' in stripped:
                        header_line = stripped
                        break

                if header_line:
                    # Parse column labels from the header.
                    parts = [p.strip() for p in header_line.split(',')]
                    # The Time column is dropped by .iloc[:,1:] after read, so a
                    # channel's rawdata index = its CSV column index minus the
                    # number of dropped columns that PRECEDE it. Normally Time
                    # is column 0 (one dropped column before every channel), so
                    # the offset is 1. Detect it explicitly instead of assuming.
                    time_col = next(
                        (i for i, p in enumerate(parts)
                         if p.upper() == 'TIME' or p.upper().startswith('TIME')),
                        0)
                    drop_offset = 1 if time_col == 0 else 0
                    gb_indices = []
                    astm_indices = []
                    for idx, part in enumerate(parts):
                        part_upper = part.upper()
                        if 'GB_' in part_upper or '60CM' in part_upper:
                            gb_indices.append(idx - drop_offset)
                        elif 'ASTM_' in part_upper or '13INCH' in part_upper:
                            astm_indices.append(idx - drop_offset)

                    if len(gb_indices) == 3 and len(astm_indices) == 3:
                        self.gb_col_indices = gb_indices
                        self.astm_col_indices = astm_indices
                        print(f"Detected columns from header - GB cols: {gb_indices}, ASTM cols: {astm_indices}")
                        return

                # Fallback: if header detection failed, use default (GB=0,1,2 ASTM=3,4,5)
                print("Warning: Could not detect GB/ASTM columns from header, using default mapping (GB=cols 0-2, ASTM=cols 3-5)")
                self.gb_col_indices = [0, 1, 2]
                self.astm_col_indices = [3, 4, 5]
            else:
                # Tab-separated format: try to read header for column labels
                # Default: GB=0,1,2 ASTM=3,4,5
                print("Tab format: using default mapping (GB=cols 0-2, ASTM=cols 3-5)")
                self.gb_col_indices = [0, 1, 2]
                self.astm_col_indices = [3, 4, 5]
        except Exception as e:
            print(f"Error detecting columns: {e}, using defaults")

    def init_data(self):
        ASTM_cols_str = "Time 13inch_accel-x 13inch_accel-y 13inch_accel-z"
        GB_cols_str = "Time 60cm_accel-x 60cm_accel-y 60cm_accel-z"
        ASTM_cols = ASTM_cols_str.split(' ')
        GB_cols = GB_cols_str.split(' ')
        col_time = pd.DataFrame(np.arange(0, self.shape[0] * 0.002, 0.002), columns=['Time'])
        ASTM_data = pd.DataFrame()
        ASTM_data['Time'] = col_time['Time']
        GB_data = pd.DataFrame()
        GB_data['Time'] = col_time['Time']

        # Use detected column indices for both CSV and Tab formats
        for i in range(3):
            GB_data[GB_cols[i + 1]] = self.rawdata.iloc[:, self.gb_col_indices[i]]
            ASTM_data[ASTM_cols[i + 1]] = self.rawdata.iloc[:, self.astm_col_indices[i]]

        self.GB_data = AccData(GB_data, is_raw=True, path="(GB)"+os.path.basename(self.filename))
        self.GB_data.set_colID(GB_cols_str)

        self.ASTM_data = AccData(ASTM_data,is_raw=True, path="(ASTM)"+os.path.basename(self.filename))
        self.ASTM_data.set_colID(ASTM_cols_str)
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
                           min_active_frac: float = 0.15):
    """Detect sub-test boundaries in a (possibly concatenated) recording.

    A single imported file may hold several measured rides back-to-back,
    separated by quiet "stop" periods where the accelerometer reads only
    gravity (X≈0, Y≈0, Z≈1g). This finds those stops and returns the tests.

    Split rule (data-driven, no fixed lead time):
      * The boundary BETWEEN two tests is the MIDPOINT of the long quiet run
        separating them — so each side gets half the stop as a static lead-in
        (the previous lap's tail-quiet + the next lap's head-quiet meet there).
      * The FIRST lap starts at the file start (sample 0). If a long quiet
        pre-roll precedes it, the lead-in is naturally the part of that run
        up to the motion onset, capped at ``max_lead_s``.
      * The LAST lap runs to the file end; a long trailing quiet run is capped
        at ``max_lead_s`` of lead-in before it ends.
      * Any static lead-in derived this way is capped at ``max_lead_s`` so a
        very long stop doesn't pad a segment with tens of seconds of dead air.

    Algorithm (validated on real GB/ASTM files):
      1. motion = sqrt(x² + y² + (z−1)²)            — deviation from rest
      2. rolling-mean smooth over ``win_s`` seconds — kills sample noise
      3. is_quiet = smoothed < quiet_thresh
      4. contiguous quiet runs >= min_gap_s are the inter-test stops
         (intra-test pauses are short — a few seconds — while real test
         separators are tens of seconds, so a duration filter is the robust
         discriminator and makes the exact threshold non-critical).
      5. split points = the midpoint of each long quiet run; tests are the
         spans between consecutive split points (file start → split → … → end).
      6. drop short QUIET fragments — a candidate that is both shorter than
         ``min_test_s`` AND mostly below ``quiet_thresh`` (active fraction <
         ``min_active_frac``) is dead air carved out by the split, not a test.
         A short-but-genuinely-active test is kept.
      7. return each test's [start_s, end_s] as INTEGER seconds, matching
         truncate_data's ``int(time) * fs`` convention.

    Parameters
    ----------
    acc_data : AccData
        The loaded dataset (reads ``.data`` columns + ``.fs``).
    quiet_thresh : float
        Motion (g) below which a sample counts as "quiet". 0.10 separates
        idle from active on measured GB/ASTM data.
    win_s : float
        Rolling-mean window in seconds. 0.5 smooths sample noise without
        erasing real short stops.
    min_gap_s : float
        A quiet run must last at least this long to count as a test boundary.
        10s cleanly separates real separators (≥30s) from intra-test pauses
        (≤3s) — there is ~10× separation between the two clusters.
    min_test_s : float
        Tests shorter than this (after capping) are dropped (noise blips).
    max_lead_s : float
        Cap on how much static lead-in a test may carry at either end. A
        long stop split at its midpoint already balances both sides, but the
        first lap's head and last lap's tail (which touch the file edges)
        and any asymmetric midpoint can otherwise pad a segment with a very
        long quiet run. 15s keeps each lead-in ample for calibration without
        wasting data.
    min_active_frac : float
        Fraction of a candidate segment's samples that must exceed
        ``quiet_thresh`` (i.e. be genuinely moving) for it to count as a real
        test. Used together with ``min_test_s`` to DROP short quiet fragments:
        a region is dropped only if it is BOTH shorter than ``min_test_s``
        AND has an active fraction below this. This protects brief-but-real
        tests (a 6s ride is ~60% active) while removing the short dead-air
        slivers that gap-splitting can carve out. 0.15 is well below any real
        test (~0.5–0.7) and well above pure quiet (~0).

    Returns
    -------
    list[tuple[int, int]]
        One (start_seconds, end_seconds) pair per detected test. Empty list
        if the recording is too short to contain a test. Times are taken
        directly from the ``Time`` column (may be negative) and rounded to
        integers.
    """
    df = getattr(acc_data, "data", None)
    fs = getattr(acc_data, "fs", 500) or 500
    if df is None or len(df) < 2:
        return []

    # Columns: iloc[:,0]=Time, iloc[:,1]=X, iloc[:,2]=Y, iloc[:,3]=Z.
    try:
        t = pd.to_numeric(df.iloc[:, 0], errors="coerce").values
        x = pd.to_numeric(df.iloc[:, 1], errors="coerce").values
        y = pd.to_numeric(df.iloc[:, 2], errors="coerce").values
        z = pd.to_numeric(df.iloc[:, 3], errors="coerce").values
    except Exception:
        return []
    # Drop any leading/trailing non-numeric rows defensively.
    mask = ~(np.isnan(t) | np.isnan(x) | np.isnan(y) | np.isnan(z))
    if not mask.all():
        t, x, y, z = t[mask], x[mask], y[mask], z[mask]
    if len(t) < 2:
        return []

    # 1-2. Motion magnitude + rolling-mean smoothing.
    motion = np.sqrt(x ** 2 + y ** 2 + (z - 1.0) ** 2)
    win = max(1, int(round(win_s * fs)))
    smoothed = (pd.Series(motion).rolling(win, center=True).mean()
                .bfill().ffill().values)

    # 3-4. Contiguous quiet runs; keep the long ones as test boundaries.
    is_quiet = smoothed < quiet_thresh
    edges = np.diff(np.concatenate([[0], is_quiet.astype(int), [0]]))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1
    long_gaps = [(s, e) for s, e in zip(starts, ends)
                 if (e - s) / fs >= min_gap_s]

    # 5-6. Build tests from the ACTIVE regions (the spans between long quiet
    #      gaps). A long gap between two tests splits at its MIDPOINT so each
    #      side gets half the stop as static lead-in/out; if that half exceeds
    #      max_lead_s, cap each side's quiet to max_lead_s (don't pad with more
    #      dead air). A leading gap (pre-roll) trims the first test's head to
    #      max_lead_s of lead-in; a trailing gap (tail) trims the last test's
    #      tail the same way. The first lap's head and last lap's tail fall
    #      back to the file start / end when there is no gap to cap against.
    max_lead = max(0, int(round(max_lead_s * fs)))
    last = len(t) - 1

    # For each long gap compute the capped prev-tail and next-head sample
    # indices: prev_test ends at tail_end, next_test starts at head_start.
    gap_cuts = []
    for gs, ge in long_gaps:
        midpoint = (gs + ge) // 2
        half = (ge - gs) // 2
        if half <= max_lead:
            tail_end, head_start = midpoint, midpoint + 1
        else:
            tail_end, head_start = gs + max_lead, ge - max_lead + 1
        gap_cuts.append((tail_end, head_start))

    # Walk the gaps: each gap separates two tests. The test BEFORE a gap ends
    # at the gap's tail_end; the test AFTER starts at the gap's head_start.
    # The very first test starts at the file start (or, if a leading gap
    # exists, max_lead into it); the last test runs to the file end.
    regions = []
    test_start = 0
    # If the file opens with a long gap (pre-roll), the first test's head is
    # capped to max_lead of lead-in past the gap start.
    has_leading = long_gaps and long_gaps[0][0] <= max_lead
    if has_leading:
        first_head = long_gaps[0][1] - max_lead + 1
        test_start = max(long_gaps[0][0], min(first_head, long_gaps[0][1]))
    for (tail_end, head_start) in gap_cuts:
        # Guard against a single gap that is BOTH leading and trailing (the
        # whole file is one dead-air gap): skip a region whose end is at/before
        # its start to avoid a negative-length or duplicate segment.
        if tail_end > test_start and (tail_end - test_start) / fs >= min_test_s:
            regions.append((test_start, tail_end))
        test_start = max(test_start, head_start)
    # If the file ends with a long gap (tail), cap the last test's tail.
    # Only treat it as a tail when it's a DIFFERENT gap from the leading one
    # (a single gap can't be both the head cap and the tail cap sensibly).
    has_trailing = (long_gaps
                    and long_gaps[-1][1] >= last - max_lead
                    and (not has_leading or long_gaps[0] is not long_gaps[-1]))
    if has_trailing:
        last_end = long_gaps[-1][0] + max_lead
        last_end = min(last_end, last)
    else:
        last_end = last
    if last_end > test_start and (last_end - test_start) / fs >= min_test_s:
        regions.append((test_start, last_end))

    # 7. Drop short QUIET fragments. A candidate that is BOTH shorter than
    #    min_test_s AND mostly quiet (active fraction < min_active_frac) is
    #    dead air that the midpoint split carved out, not a real test. A short
    #    but genuinely moving segment (a brief ride) is kept — its active
    #    fraction is high, so it survives this filter.
    def _is_short_quiet(s, e):
        dur = (e - s) / fs
        if dur >= min_test_s:
            return False                     # long enough — keep regardless
        active_frac = float(np.mean(smoothed[s:e + 1] >= quiet_thresh))
        return active_frac < min_active_frac

    regions = [(s, e) for s, e in regions if not _is_short_quiet(s, e)]

    # 8. Sample indices -> integer seconds (from the Time column).
    out = []
    for s, e in regions:
        out.append((int(round(float(t[s]))), int(round(float(t[e])))))
    return out