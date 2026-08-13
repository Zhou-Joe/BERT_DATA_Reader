import matplotlib
import numpy as np
import pandas as pd
import math
from scipy.signal import butter, lfilter_zi, lfilter
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    zi = lfilter_zi(b, a)
    return b, a, zi

def butter_lowpass_filter(data, cutoff, fs, zeropoint=0, order=4):
    # Cutoff at/above the Nyquist frequency (fs/2) is invalid for a digital
    # filter — scipy.signal.butter raises ValueError for Wn >= 1. The old
    # `if cutoff==500` guard was value-based and only worked for fs=1000; a
    # 500 Hz file with cutoff=250 (or any fs where cutoff==fs/2) crashed.
    # Treat cutoff >= Nyquist as "no filtering": return the data unchanged.
    nyq = 0.5 * fs
    if cutoff >= nyq:
        return data
    b, a, zi = butter_lowpass(cutoff, fs, order=order)
    y = np.round(lfilter(b, a, data, zi=zeropoint*zi)[0], decimals=4)
    return y


def ProcessASTM(data, dt):
    """Sustained-peak (+) / (-) for the three ASTM axes at one window width.

    Returns ``(pos, neg)`` where ``pos = [ax+, ay+, az+]`` and
    ``neg = [ax−, ay−, az−]`` (lists of 3 floats).

    The sustained value over a window is its min (for +peak) / max (for −peak).
    The reported peak is the max/min of those per-window stats. The crucial
    detail is the FLOOR clamp before the outer reduce:

      • ax+/ay+ : ``max(min(window), 0)``  — a window that dips below 0
        contributes 0, so a large window with no all-positive span reports 0
        (NOT a negative number).
      • ax−/ay− : ``min(max(window), 0)``  — symmetric: an all-positive window
        contributes 0.
      • az+/az− : same idea but the gravity offset is 1 g, so the floor/ceiling
        is 1 instead of 0, and results are reported in the 1 g-referenced frame.

    Without these clamps, ``np.max(rolling.min())`` returns a negative number
    for a big window whose every span crosses zero, which is wrong: the
    sustained +acceleration is then reported as negative (e.g. measured ax(+)
    dipping below 0 at a 12 s window). The original code had the clamps in a
    commented-out list-comprehension; this restores them with rolling().

    Performance: the per-window floor/ceiling clamp is done VECTORISED
    (``np.maximum`` / ``np.minimum``) rather than via a Python
    ``[max(v,0) for v in rmin]`` list-comprehension. The list-comprehension is
    O(n) in pure Python and dominated ProcessASTM (~34ms/axis for 27k points);
    the vectorised form is ~7× faster. This matters because ProcessASTM is
    called 17× per ASTM Sustained Generate (one per dt column) and once per
    InfoCard stats rebuild on every plot-type switch.
    """
    # numpy arrays directly — avoids the pandas-Series overhead for the source
    # columns (rolling still needs a Series, but we build it once per axis).
    x_array = data.iloc[:, 1].values
    y_array = data.iloc[:, 2].values
    z_array = data.iloc[:, 3].values

    # ---- +sustained: per-window min, clamped to the axis floor, then max ----
    # ax / ay floor at 0; az floor at 1 (1 g gravity offset for vertical axis).
    # np.maximum(arr, floor) is the vectorised clamp; np.max of that is the peak.
    # Guard against short recordings: when len(data) < dt the rolling window is
    # empty after dropna(), and np.max(np.maximum([], floor)) raises ValueError
    # ("zero-size array"). Fall back to the floor value (no +sustained evidence).
    rmin_x = pd.Series(x_array).rolling(dt).min().dropna().values
    lstx1 = round(float(np.max(np.maximum(rmin_x, 0.0))), 4) if rmin_x.size else 0.0

    rmin_y = pd.Series(y_array).rolling(dt).min().dropna().values
    lsty1 = round(float(np.max(np.maximum(rmin_y, 0.0))), 4) if rmin_y.size else 0.0

    rmin_z = pd.Series(z_array).rolling(dt).min().dropna().values
    lstz1 = round(float(np.max(np.maximum(rmin_z, 1.0))), 4) if rmin_z.size else 1.0

    # ---- −sustained: per-window max, clamped to the axis ceiling, then min ---
    # ax / ay ceiling at 0; az ceiling at 1 (gravity). az result is returned in
    # the 1 g-referenced frame (consistent with lstz1).
    rmax_x = pd.Series(x_array).rolling(dt).max().dropna().values
    xn = round(float(np.min(np.minimum(rmax_x, 0.0))), 4) if rmax_x.size else 0.0

    rmax_y = pd.Series(y_array).rolling(dt).max().dropna().values
    yn = round(float(np.min(np.minimum(rmax_y, 0.0))), 4) if rmax_y.size else 0.0

    rmax_z = pd.Series(z_array).rolling(dt).max().dropna().values
    zn = round(float(np.min(np.minimum(rmax_z, 1.0))), 4) if rmax_z.size else 1.0

    return [lstx1, lsty1, lstz1], [xn, yn, zn]





def ProcessGB(data, dt):
    # if dt > 999:
    #     x = min(int(dt / 50),90)
    # else:


    y_array = list(data.iloc[:, 2].values)
    z_array = list(data.iloc[:, 3].values)
    m = len(y_array)

   # maxp = np.max(list(map(lambda i: [max(min(y_array[i:i + dt]), 0), max(min(z_array[i:i + dt]), 1)], range(m - dt))),
      #            axis=0)
   # maxn = np.min(list(map(lambda i: [min(max(y_array[i:i + dt]), 0), min(max(z_array[i:i + dt]), 1)], range(m - dt))),
          #        axis=0)

    # Guard against short recordings: when len(data) < dt, rolling().dropna()
    # is empty and np.max([]) raises ValueError. Fall back to the axis floor
    # (y floor 0, z floor 1).
    _yp_vals = pd.Series(y_array).rolling(dt).min().dropna().tolist()
    yp = round(np.max(_yp_vals), 4) if _yp_vals else 0


    lsty2 = pd.Series(y_array).rolling(dt).max().dropna()

    lsty2 = lsty2[(lsty2 < 0)]

    if len(lsty2) > 0:
        yn = round(np.min(lsty2),4)
    else:
        yn = 0

    _zp_vals = pd.Series(z_array).rolling(dt).min().dropna().tolist()
    zp = round(np.max(_zp_vals), 4) if _zp_vals else 1
    lstz2 = (pd.Series(z_array) - 1).rolling(dt).max().dropna()
    lstz2 = lstz2[(lstz2 < 0)]
    if len(lstz2) > 0:

        zn = round(np.min(lstz2) + 1, 4)
    else:
        zn = 1


   # print (maxp,maxn, [yp, zp],[yn,zn])
    return [yp, zp],[yn,zn]

def append4(x1,x2,x3,x4):
    x=np.append(x1,x2)
    x=np.append(x,x3)
    x=np.append(x,x4)
    return x

def eggXY(x1, x2, y_max):
    x1 = min(x1, 2)
    y_max = min(y_max, 3)

    x11 = np.cos(np.arange(0, math.pi * 0.5, 0.01)) * x2
    x22 = np.cos(np.arange(math.pi * 0.5, math.pi, 0.01)) * x1
    x33 = np.cos(np.arange(math.pi, math.pi * 1.5, 0.01)) * x1
    x44 = np.cos(np.arange(math.pi * 1.5, math.pi * 2 + 0.01, 0.01)) * x2
    y11 = np.sin(np.arange(0, math.pi * 0.5, 0.01)) * y_max
    y22 = np.sin(np.arange(math.pi * 0.5, math.pi, 0.01)) * y_max
    y33 = np.sin(np.arange(math.pi, math.pi * 1.5, 0.01)) * y_max
    y44 = np.sin(np.arange(math.pi * 1.5, math.pi * 2 + 0.01, 0.01)) * y_max
    return append4(x11, x22, x33, x44), append4(y11, y22, y33, y44)


def eggXZ(x1, x2, y1, y2):
    x1 = min(x1, 2)
    y1 = min(y1, 2)
    y2 = min(y2, 6)
    x11 = np.cos(np.arange(0, math.pi * 0.5, 0.01)) * x2
    x22 = np.cos(np.arange(math.pi * 0.5, math.pi, 0.01)) * x1
    x33 = np.cos(np.arange(math.pi, math.pi * 1.5, 0.01)) * x1
    x44 = np.cos(np.arange(math.pi * 1.5, math.pi * 2 + 0.01, 0.01)) * x2
    y11 = np.sin(np.arange(0, math.pi * 0.5, 0.01)) * y2
    y22 = np.sin(np.arange(math.pi * 0.5, math.pi, 0.01)) * y2
    y33 = np.sin(np.arange(math.pi, math.pi * 1.5, 0.01)) * y1
    y44 = np.sin(np.arange(math.pi * 1.5, math.pi * 2 + 0.01, 0.01)) * y1

    return append4(x11, x22, x33, x44), append4(y11, y22, y33, y44)


def eggYZ(x, y1, y2):
    y1 = min(2, y1)
    y2 = min(6, y2)
    x = min(3, x)
    x11 = np.cos(np.arange(0, math.pi * 0.5, 0.01)) * x
    x22 = np.cos(np.arange(math.pi * 0.5, math.pi, 0.01)) * x
    x33 = np.cos(np.arange(math.pi, math.pi * 1.5, 0.01)) * x
    x44 = np.cos(np.arange(math.pi * 1.5, math.pi * 2 + 0.01, 0.01)) * x
    y11 = np.sin(np.arange(0, math.pi * 0.5, 0.01)) * y2
    y22 = np.sin(np.arange(math.pi * 0.5, math.pi, 0.01)) * y2
    y33 = np.sin(np.arange(math.pi, math.pi * 1.5, 0.01)) * y1
    y44 = np.sin(np.arange(math.pi * 1.5, math.pi * 2 + 0.01, 0.01)) * y1
    return append4(x11, x22, x33, x44), append4(y11, y22, y33, y44)


def eggXY_unclamped(x1, x2, y_max):
    """Like eggXY but WITHOUT the min(x1,2)/min(y_max,3) radius clamps.

    Used for the prone-posture egg, whose left (negative-x) extent (3.5·coef)
    exceeds the 2g clamp that the standard ASTM egg enforces. Each half is an
    independent ellipse: left half radius ``x1``, right half radius ``x2``.
    """
    x11 = np.cos(np.arange(0, math.pi * 0.5, 0.01)) * x2
    x22 = np.cos(np.arange(math.pi * 0.5, math.pi, 0.01)) * x1
    x33 = np.cos(np.arange(math.pi, math.pi * 1.5, 0.01)) * x1
    x44 = np.cos(np.arange(math.pi * 1.5, math.pi * 2 + 0.01, 0.01)) * x2
    y11 = np.sin(np.arange(0, math.pi * 0.5, 0.01)) * y_max
    y22 = np.sin(np.arange(math.pi * 0.5, math.pi, 0.01)) * y_max
    y33 = np.sin(np.arange(math.pi, math.pi * 1.5, 0.01)) * y_max
    y44 = np.sin(np.arange(math.pi * 1.5, math.pi * 2 + 0.01, 0.01)) * y_max
    return append4(x11, x22, x33, x44), append4(y11, y22, y33, y44)


def eggXZ_unclamped(x1, x2, y1, y2):
    """Like eggXZ but WITHOUT the radius clamps (x1≤2, y1≤2, y2≤6).

    Used for the prone-posture egg, whose left (negative-x) extent exceeds the
    2g clamp that the standard ASTM egg enforces. Four independent quadrant
    radii: +x=x2, −x=x1, up(+y)=y2, down(−y)=y1.
    """
    x11 = np.cos(np.arange(0, math.pi * 0.5, 0.01)) * x2
    x22 = np.cos(np.arange(math.pi * 0.5, math.pi, 0.01)) * x1
    x33 = np.cos(np.arange(math.pi, math.pi * 1.5, 0.01)) * x1
    x44 = np.cos(np.arange(math.pi * 1.5, math.pi * 2 + 0.01, 0.01)) * x2
    y11 = np.sin(np.arange(0, math.pi * 0.5, 0.01)) * y2
    y22 = np.sin(np.arange(math.pi * 0.5, math.pi, 0.01)) * y2
    y33 = np.sin(np.arange(math.pi, math.pi * 1.5, 0.01)) * y1
    y44 = np.sin(np.arange(math.pi * 1.5, math.pi * 2 + 0.01, 0.01)) * y1
    return append4(x11, x22, x33, x44), append4(y11, y22, y33, y44)


def coef(height, condition):
    if condition == 'x' or condition == 'y':
        if height <= 32:
            return 0.6
        elif height <= 48:
            return height * 0.025 - 0.2
        else:
            return 1

    if condition == 'z':
        if height <= 32:
            return 0.6
        elif height <= 48:
            return height * 0.025 - 0.2
        else:
            return 1


def coordTransform(pitch, seatback, roll, yaw):
    pitch = pitch/180*np.pi
    seatback=seatback/180*np.pi
    roll=roll/180*np.pi
    yaw=yaw/180*np.pi

    P0 = np.array([0, 0, 1])
    R0A = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    RAB = np.array([[1, 0, 0], [0, math.cos(roll), math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
    RBC = np.array(
        [[math.cos(seatback), 0, math.sin(seatback)], [0, 1, 0], [-math.sin(seatback), 0, math.cos(seatback)]])
    RC0 = (np.dot(np.dot(R0A, RAB), RBC)).T

    return np.round(np.dot(RC0, P0), decimals=4)


# ===========================================================================
# Acceleration reversal detection (ported verbatim from the archived app).
# Used by the "Z Reversal" and "X/Y Reversal" tabs. Algorithms are unchanged
# from archived/sci_calculation.py; only the module home moved.
# ===========================================================================

def detect_z_acceleration_reversal(z_data, time_data, fs, deadband=0.05):
    """
    Detect excessive Z-acceleration transitions (reversals), BIDIRECTIONALLY.

    A reversal is a large swing crossing from one sustained state to the other
    inside a 0.133s window (min < 0 AND max > 2 must both occur). Two directions
    are detected:
      • UP   — the minimum occurs BEFORE the maximum (negative → +2g+), AND the
               0.2s pre-window is in a sustained NEGATIVE state.
      • DOWN — the maximum occurs BEFORE the minimum (+2g+ → negative), AND the
               0.2s pre-window is in a sustained POSITIVE state.

    The pre-window "sustained state" uses a hysteresis dead-band (same idea as
    detect_xy_acceleration_reversal): a sample commits to + only above
    +deadband and to − only below −deadband; sub-threshold wobbles near zero
    keep the previous sign. So the UP pre-window qualifies if no sample exceeds
    +deadband, and the DOWN pre-window qualifies if no sample drops below
    −deadband. This tolerates small noise/filtering glitches that the old strict
    test (any sample >= 0 fails) rejected, while still filtering out a pre-window
    that genuinely sat on the wrong side.

    Each reversal carries ``direction`` ('up' or 'down') plus the existing keys.
    Overlapping windows are deduplicated, keeping the one with the largest
    max−min swing.

    ``deadband`` (g, default 0.05) is the hysteresis threshold. Pass 0 to
    disable hysteresis (plain strict-sign behaviour).

    Returns a list of dicts, each with keys: direction, window_start, window_end,
    min_time, max_time, min_value, max_value, pre_window_start, pre_window_end,
    window_start_idx, window_end_idx, min_idx, max_idx, value_diff.
    """
    z_data = np.asarray(z_data)
    time_data = np.asarray(time_data)

    # Window sizes in samples
    window_duration = 0.133  # seconds
    pre_window_duration = 0.2  # seconds
    window_size = int(window_duration * fs)
    pre_window_size = int(pre_window_duration * fs)

    all_reversals = []
    n = len(z_data)

    # Slide through the data with 0.133s window
    for i in range(n - window_size + 1):
        # Get the current window
        window_start_idx = i
        window_end_idx = i + window_size - 1

        window_data = z_data[window_start_idx:window_end_idx + 1]
        window_time = time_data[window_start_idx:window_end_idx + 1]

        # Condition 1: the window must span both below 0 and above 2.
        min_val = np.min(window_data)
        max_val = np.max(window_data)
        if min_val >= 0 or max_val <= 2:
            continue

        # Find indices of min and max in the window
        min_idx_in_window = int(np.argmin(window_data))
        max_idx_in_window = int(np.argmax(window_data))
        min_time = window_time[min_idx_in_window]
        max_time = window_time[max_idx_in_window]

        # Condition 2: 0.2s pre-window must exist.
        pre_window_start_idx = max(0, window_start_idx - pre_window_size)
        pre_window_end_idx = window_start_idx - 1
        if pre_window_end_idx < pre_window_start_idx:
            continue   # not enough data before the window
        pre_window_data = z_data[pre_window_start_idx:pre_window_end_idx + 1]

        # Determine direction from the order of min/max, then verify the
        # pre-window is in the matching sustained state.
        if min_time < max_time:
            # UP transition (negative → +2g+): pre-window must be sustained NEG.
            direction = 'up'
            if deadband > 0:
                if np.any(pre_window_data > deadband):
                    continue
            else:
                if np.any(pre_window_data >= 0):
                    continue
        else:
            # DOWN transition (+2g+ → negative): pre-window must be sustained POS.
            direction = 'down'
            if deadband > 0:
                if np.any(pre_window_data < -deadband):
                    continue
            else:
                if np.any(pre_window_data <= 0):
                    continue

        # All conditions satisfied - record this reversal.
        reversal = {
            'direction': direction,
            'window_start': window_time[0],
            'window_end': window_time[-1],
            'min_time': min_time,
            'max_time': max_time,
            'min_value': min_val,
            'max_value': max_val,
            'pre_window_start': time_data[pre_window_start_idx],
            'pre_window_end': window_time[0],
            'window_start_idx': window_start_idx,
            'window_end_idx': window_end_idx,
            'min_idx': window_start_idx + min_idx_in_window,
            'max_idx': window_start_idx + max_idx_in_window,
            'value_diff': max_val - min_val
        }
        all_reversals.append(reversal)

    # Deduplicate: keep only the one with max value difference for consecutive windows
    if len(all_reversals) == 0:
        return []

    deduplicated = []
    current_group = [all_reversals[0]]

    for i in range(1, len(all_reversals)):
        prev = current_group[-1]
        curr = all_reversals[i]

        # Check if windows overlap (consecutive)
        if curr['window_start_idx'] <= prev['window_end_idx']:
            # Overlapping - add to current group
            current_group.append(curr)
        else:
            # Non-overlapping - finalize current group and start new one
            # Keep the one with maximum value difference
            best = max(current_group, key=lambda x: x['value_diff'])
            deduplicated.append(best)
            current_group = [curr]

    # Don't forget the last group
    if len(current_group) > 0:
        best = max(current_group, key=lambda x: x['value_diff'])
        deduplicated.append(best)

    return deduplicated


def detect_xy_acceleration_reversal(x_data, y_data, time_data, fs,
                                    deadband=0.05):
    """
    Detect X and Y acceleration reversals under the ASTM F2291
    **peak-to-peak sustained-event** rule:

      "The peak-to-peak transition time between consecutive sustained events in
       X and Y accelerations shall be greater than 200 ms, as measured by the
       time between the peaks of the consecutive events. When the elapsed time
       between consecutive sustained events is less than 200 ms, the limit for
       the peak values shall be reduced by 50%."

    For EACH axis independently:
      1. Compute a per-sample sign with a hysteresis dead-band (so a tiny
         noise/filter glitch poking across zero doesn't create a spurious sign
         flip — a macro single transition still reads as ONE sign change).
      2. Segment the signal into **sustained runs**: maximal consecutive
         regions of constant committed sign. Each run carries start/end index,
         sign, and a **peak** (the index of the largest |value| in the run).
      3. A pair of **adjacent runs of opposite sign** (A then B) is a reversal
         candidate only if BOTH sustained durations are > 0.2 s. A run that
         flips back too quickly (< 0.2 s) is not a real reversal (transient /
         glitch) and is rejected.
      4. The candidate's acceptance multiplier is set by the **peak-to-peak
         time** Δt = peakB_time − peakA_time:
           • Δt > 0.2 s → ``mult = 1.0`` (full allowance)
           • Δt ≤ 0.2 s → ``mult = 0.5`` (allowance reduced by 50%)
      5. Overlapping candidate pairs are merged, keeping the one with the
         larger |peakB − peakA| (so one long transition doesn't fire twice).

    ``deadband`` (g, default 0.05) is the hysteresis threshold: a sample only
    commits to the + or − sign once the value exceeds ±deadband, and keeps its
    previous sign inside the dead-band. Pass 0 to disable hysteresis (plain
    sign behaviour).

    Each recorded event carries:
      ``peakA_idx`` / ``peakA_time`` / ``peakA_value`` — prior run's peak,
      ``peakB_idx`` / ``peakB_time`` / ``peakB_value`` — new run's peak,
      ``signA`` / ``signB`` — +1 / −1 for the two runs,
      ``seg_a_start_idx`` / ``seg_a_end_idx`` — run A index bounds,
      ``seg_b_start_idx`` / ``seg_b_end_idx`` — run B index bounds,
      ``window_start`` / ``window_end`` — times spanning run A start to run B
         end (the highlight band bounds),
      ``window_start_idx`` / ``window_end_idx`` — the integer index bounds
         backing window_start/window_end,
      ``delta_t`` — peak-to-peak time (s),
      ``mult`` — 1.0 or 0.5 (the allowance multiplier for this pair),
      ``axis`` — 'x' or 'y'.

    Returns a dict ``{'x_reversals': [...], 'y_reversals': [...]}``.
    """
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)
    time_data = np.asarray(time_data, dtype=float)

    # Minimum sustained-run duration for a run to take part in a reversal
    # (both sides of the direction change must last longer than this).
    min_sustained = 0.2  # seconds
    peak_to_peak_threshold = 0.2  # Δt above this → full allowance, below → 0.5×

    def hysteresis_sign(arr):
        """Per-sample sign with a dead-band.

        A sample commits to +1 once arr > deadband, −1 once arr < −deadband,
        and otherwise holds its previous committed sign. So a sub-threshold
        blip near zero never flips the sign — it's absorbed by the dead-band,
        which prevents noise/filtering glitches from creating spurious extra
        sign changes inside an otherwise single-crossing window.
        """
        out = np.zeros(len(arr), dtype=int)
        cur = 0
        for k in range(len(arr)):
            v = arr[k]
            if v > deadband:
                cur = 1
            elif v < -deadband:
                cur = -1
            # else: inside the dead-band → keep cur
            out[k] = cur
        return out

    def sustained_runs(axis_data):
        """Split the signal into maximal constant-signed runs.

        Each run is ``(start_idx, end_idx, sign)`` where ``start_idx`` is the
        FIRST sample of the committed sign and ``end_idx`` the LAST (inclusive).
        Runs are bounded by sign flips (cur changing from +1↔−1). Samples still
        in the dead-band (sign 0) at the very start are skipped.
        """
        signs = hysteresis_sign(axis_data)
        n = len(signs)
        runs = []
        i = 0
        # Skip any leading dead-band samples (sign == 0).
        while i < n and signs[i] == 0:
            i += 1
        while i < n:
            s = int(signs[i])
            start = i
            # Extend the run while the sign stays committed (a dead-band
            # sample that didn't flip keeps cur, so it stays in the run).
            while i < n and signs[i] == s:
                i += 1
            runs.append((start, i - 1, s))
        return runs

    def run_peak(axis_data, start_idx, end_idx, sign):
        """Index/value of the run's peak (extreme in the run's sign direction).

        For a +run the peak is the MAX value (the +peak); for a −run the MIN
        value (the −peak). Falls back to the start sample on an empty slice.
        """
        seg = axis_data[start_idx:end_idx + 1]
        if seg.size == 0:
            return start_idx, float(axis_data[start_idx])
        if sign > 0:
            off = int(np.argmax(seg))
        else:
            off = int(np.argmin(seg))
        idx = start_idx + off
        return idx, float(axis_data[idx])

    def find_reversals_for_axis(axis_data, axis_name):
        n = len(axis_data)
        if n < 2:
            return []
        runs = sustained_runs(axis_data)
        if len(runs) < 2:
            return []

        candidates = []
        for (sA, eA, sgA), (sB, eB, sgB) in zip(runs, runs[1:]):
            # Only ADJACENT runs of OPPOSITE sign form a reversal pair.
            if sgB == sgA:
                continue
            # Both sustained durations must be > 0.2 s. A run that flips back
            # too quickly (< 0.2 s) is a transient, not a reversal.
            dur_a = float(time_data[eA] - time_data[sA])
            dur_b = float(time_data[eB] - time_data[sB])
            if dur_a <= min_sustained or dur_b <= min_sustained:
                continue

            pa_idx, pa_val = run_peak(axis_data, sA, eA, sgA)
            pb_idx, pb_val = run_peak(axis_data, sB, eB, sgB)
            delta_t = float(time_data[pb_idx] - time_data[pa_idx])
            mult = 1.0 if delta_t > peak_to_peak_threshold else 0.5

            candidates.append({
                'peakA_idx': pa_idx,
                'peakA_time': float(time_data[pa_idx]),
                'peakA_value': pa_val,
                'peakB_idx': pb_idx,
                'peakB_time': float(time_data[pb_idx]),
                'peakB_value': pb_val,
                'signA': int(sgA),
                'signB': int(sgB),
                'seg_a_start_idx': int(sA),
                'seg_a_end_idx': int(eA),
                'seg_b_start_idx': int(sB),
                'seg_b_end_idx': int(eB),
                'window_start': float(time_data[sA]),
                'window_end': float(time_data[eB]),
                'window_start_idx': int(sA),
                'window_end_idx': int(eB),
                'delta_t': delta_t,
                'mult': mult,
                'axis': axis_name,
            })

        if not candidates:
            return []

        # Merge overlapping candidate pairs: keep the one with the largest
        # |peakB − peakA| so a long transition doesn't fire twice (same spirit
        # as the Z-reversal dedup). Two pairs overlap when their bands' index
        # ranges intersect.
        candidates.sort(key=lambda c: c['window_start_idx'])
        merged = []
        cur = candidates[0]
        for nxt in candidates[1:]:
            if nxt['window_start_idx'] <= cur['window_end_idx']:
                # Overlap — keep the stronger pair.
                cur_amp = abs(cur['peakB_value'] - cur['peakA_value'])
                nxt_amp = abs(nxt['peakB_value'] - nxt['peakA_value'])
                cur = nxt if nxt_amp > cur_amp else cur
            else:
                merged.append(cur)
                cur = nxt
        merged.append(cur)
        return merged

    return {
        'x_reversals': find_reversals_for_axis(x_data, 'x'),
        'y_reversals': find_reversals_for_axis(y_data, 'y'),
    }


# ===========================================================================
# t=0 allowance per ASTM F2291 restraint type (for the X/Y Reversal tab).
#
# Ported from chart_astm.ASTMChartView._add_restraint_curves: each restraint
# type clamps a restraint-specific coefficient array against the ASTM base
# limits (frontASTM/backASTM/lrASTM) using coef = cond_mult * coef(height).
# At t=0 the x-axis array's first element is 0, so the t=0 allowance is the
# FIRST element of the clamped array (a31=ax-, a32=ax+, a33=ay).
# ===========================================================================

# ASTM base limit first-elements (frontASTM/backASTM/lrASTM at index 0).
_AX_NEG_BASE = -2.0   # frontASTM[0]
_AX_POS_BASE = 6.0    # backASTM[0]
_AY_BASE = 3.0        # lrASTM[0]

# Restraint-specific t=0 coefficients R (the multiplier applied to coef before
# clamping). (R_ax_neg, R_ax_pos, R_ay) per restraint, mirroring the a31/a32/a33
# first elements in _add_restraint_curves.
_RESTR_COEF_T0 = {
    'Upper Body':            (-2.0, 3.6, 3.0),
    'Group Lower Body':      (-2.0, 2.5, 2.4),
    'Individual Lower Body': (-1.8, 2.5, 2.6),
    'No Restraint':          (-1.5, 2.5, 1.8),
    'Convenience Restraint': (-1.5, 2.5, 1.8),
    'None':                  (_AX_NEG_BASE, _AX_POS_BASE, _AY_BASE),
}


def allowance_at_t0(axis, restraint, condition, height):
    """Return the ASTM F2291 allowance at t=0 for one axis.

    ``axis`` is 'x' or 'y' (the X/Y Reversal axes). ``restraint`` is one of the
    restraint-type strings (matches chart_astm). ``condition`` is 'Normal',
    'E-Stop', or 'Expected/Permitted Bumping'. ``height`` is in inches.

    Returns ``(neg, pos)`` in g where:
      • For 'x': neg = ax− allowance (max(-2, R*coef)), pos = ax+ allowance
        (min(6, R*coef)) — these can differ in magnitude.
      • For 'y': the egg's lateral half is symmetric, so pos = min(3, R*coef)
        and neg = -pos.

    Matches chart_astm._add_restraint_curves' a31/a32/a33 at index 0 exactly.
    """
    # Condition multiplier (chart_astm: E-Stop / Expected-Permitted → 1.25).
    cond_mult = 1.25 if condition in ('E-Stop', 'Expected/Permitted Bumping') else 1.0
    c = coef(float(height), 'x') * cond_mult   # coef(height,'x') == coef(height,'z')
    r_neg, r_pos, r_ay = _RESTR_COEF_T0.get(restraint, _RESTR_COEF_T0['None'])
    if axis == 'x':
        return (max(_AX_NEG_BASE, r_neg * c), min(_AX_POS_BASE, r_pos * c))
    # Y: symmetric lateral allowance.
    pos = min(_AY_BASE, r_ay * c)
    return (-pos, pos)
