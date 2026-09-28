"""Zero-phase Butterworth filters (after the SciPy Cookbook recipes)."""

import os
import numpy as np
from scipy.signal import butter, filtfilt, lfilter
from multiprocessing import Pool


def bandstop_filter(data, dt, lowcut, highcut, order=4):
    """Apply a zero-phase Butterworth band-stop filter.

    Parameters
    ----------
    data : np.ndarray
        Input data; filtered along the last axis.
    dt : float
        Sampling interval (s).
    lowcut : float
        Lower edge of the stop band (Hz).
    highcut : float
        Upper edge of the stop band (Hz).
    order : int, optional
        Filter order. Default is 4.

    Returns
    -------
    np.ndarray
        Filtered data with the dtype of ``data``.
    """

    fs = 1.0 / dt
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(order, [low, high], btype='bandstop')

    # forward-backward filtering for zero phase
    data_bs = filtfilt(b, a, data)

    return np.asarray(data_bs, dtype=data.dtype)



def bandpass_filter(data, dt, lowcut, highcut, order=4):
    """Apply a zero-phase Butterworth band-pass filter.

    Parameters
    ----------
    data : np.ndarray
        Input data; filtered along the last axis.
    dt : float
        Sampling interval (s).
    lowcut : float
        Low corner frequency (Hz).
    highcut : float
        High corner frequency (Hz).
    order : int, optional
        Filter order. Default is 4.

    Returns
    -------
    np.ndarray
        Filtered data with the dtype of ``data``.
    """

    fs = 1.0 / dt
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    data_bp = filtfilt(b, a, data)

    return np.asarray(data_bp, dtype=data.dtype)


def lowpass_filter(data, dt,  highcut, order=4):
    """Apply a zero-phase Butterworth low-pass filter.

    Parameters
    ----------
    data : np.ndarray
        Input data; filtered along the last axis.
    dt : float
        Sampling interval (s).
    highcut : float
        Corner frequency (Hz).
    order : int, optional
        Filter order. Default is 4.

    Returns
    -------
    np.ndarray
        Filtered data with the dtype of ``data``.
    """

    fs = 1.0 / dt
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='low')
    data_lp = filtfilt(b, a, data)

    return np.asarray(data_lp,  dtype=data.dtype)


def highpass_filter(data, dt, lowcut, order=4):
    """Apply a zero-phase Butterworth high-pass filter.

    Parameters
    ----------
    data : np.ndarray
        Input data; filtered along the last axis.
    dt : float
        Sampling interval (s).
    lowcut : float
        Corner frequency (Hz).
    order : int, optional
        Filter order. Default is 4.

    Returns
    -------
    np.ndarray
        Filtered data with the dtype of ``data``.
    """

    fs = 1.0 / dt
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='hp')
    data_hp = filtfilt(b, a, data)

    return np.asarray(data_hp, dtype=data.dtype)
