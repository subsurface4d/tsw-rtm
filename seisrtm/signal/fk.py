"""Frequency-wavenumber transforms and filters for 2-D space-time data.

Adapted from the Earth Model code of the ETH Seismology and Wave Physics
Group (www.swp.ethz.ch).
"""

import numpy as np
from scipy import interpolate
import math

def fk(data, dt, dx, distance_scaling=0.0, taper=0.05):
    """Compute the f-k transform of 2-D space-time data.

    Parameters
    ----------
    data : np.ndarray
        Data with shape (nx, nt). When ``taper > 0`` and no distance scaling
        is applied, the taper modifies ``data`` in place.
    dt : float
        Time sampling interval (s).
    dx : float
        Spatial sampling interval (m).
    distance_scaling : float, optional
        If greater than 0, each trace is multiplied by ``sqrt(r)``, with
        ``r = dx * ix`` the distance (m). Default is 0 (no scaling).
    taper : float, optional
        Fraction of the time and space lengths tapered linearly at both ends
        before the transform. Default is 0.05.

    Returns
    -------
    f : np.ndarray
        Frequency axis (Hz), length nt.
    k : np.ndarray
        Angular wavenumber axis (rad/m), length nx.
    d_fk_r : np.ndarray
        Complex f-k spectrum with shape (nx, nt), scaled by ``dt * dx`` and
        rolled so that zero frequency and wavenumber lie at the centre.
    """

    nx, nt = np.shape(data)

    if distance_scaling > 0.0:
        scaling_factors = (dx * np.arange(nx))**0.5
        data = scaling_factors[:, np.newaxis] * data

    # temporal taper
    if taper > 0.0:
        width = int(nt * taper)
        taper_values = np.linspace(1.0 / (width + 1), 1, width)
        data[:, :width] *= taper_values
        data[:, -width:] *= taper_values[::-1]

    # spatial taper
    if taper > 0.0:
        width = int(nx * taper)
        taper_values = np.linspace(1.0 / (width + 1), 1, width)
        data[:width, :] *= taper_values[:, np.newaxis]
        data[-width:, :] *= taper_values[::-1, np.newaxis]

    d_fk = np.fft.fft2(data) * dt * dx

    # centre the spectrum on zero frequency and wavenumber
    d_fk_r = np.roll(np.roll(d_fk, int(
        (nt-1)/2), axis=1), int((nx-1)/2), axis=0)

    f = np.linspace(-0.5/dt, 0.5/dt, nt)
    k = np.linspace(-np.pi/dx, np.pi/dx, nx)

    return f, k, d_fk_r


def fk_filter_dispersion(d_fk_r, f, k, dt, dx, f_pick, k_pick, n_smooth=10):
    """Filter data in the f-k domain along a picked dispersion curve.

    A mask equal to 1 near the interpolated curve and 0 elsewhere is
    smoothed, applied to the spectrum, and the result is transformed back.

    Parameters
    ----------
    d_fk_r : np.ndarray
        Rolled f-k spectrum with shape (nx, nt), as returned by :func:`fk`.
    f : np.ndarray
        Frequency axis (Hz).
    k : np.ndarray
        Angular wavenumber axis (rad/m).
    dt : float
        Time sampling interval (s).
    dx : float
        Spatial sampling interval (m).
    f_pick : np.ndarray
        Frequencies of the dispersion-curve picks (Hz).
    k_pick : np.ndarray
        Wavenumbers of the dispersion-curve picks (rad/m).
    n_smooth : int, optional
        Number of 5-point averaging passes applied to the mask. Default is 10.

    Returns
    -------
    np.ndarray
        Filtered data in the space-time domain, shape (nx, nt).

    Notes
    -----
    The cubic interpolation/extrapolation of the picks is not reliable, so
    this filter may not give the expected result.
    """

    mask = np.zeros_like(d_fk_r)

    nk, nf = np.shape(d_fk_r)

    # cubic interpolation of the picked curve k(f)
    interpolate_k = interpolate.interp1d(
        f_pick, k_pick, kind='cubic', fill_value="extrapolate")

    # frequency indices of the first and last picks
    if_start = np.where(
        np.min(np.abs(f-f_pick[0])) == np.abs(f-f_pick[0]))[0][0]
    if_end = np.where(
        np.min(np.abs(f-f_pick[-1])) == np.abs(f-f_pick[-1]))[0][0]

    # set the mask to 1 along the curve
    for i in np.arange(if_start, if_end+1):

        k_interp = interpolate_k(f[i])

        ik = np.where(np.min(np.abs(-k-k_interp)) == np.abs(-k-k_interp))[0][0]

        mask[ik-1:ik+1, i] = 1.0

    # smooth the mask by averaging over neighbouring samples
    for l in range(n_smooth):
        mask[1:nk-1, 1:nf-1] = (mask[1:nk-1, 1:nf-1]+mask[0:nk-2, 1:nf-1] +
                                mask[2:nk, 1:nf-1]+mask[1:nk-1, 0:nf-2]+mask[1:nk-1, 2:nf])/5.0

    d_fk_r_filtered = d_fk_r*mask

    # undo the roll applied in fk()
    nt = len(f)
    nx = len(k)
    d_fk_filtered = np.roll(
        np.roll(d_fk_r_filtered, -int((nt-1)/2), axis=1), -int((nx-1)/2), axis=0)

    d_filtered = np.real(np.fft.ifft2(d_fk_filtered))/(dt*dx)

    return d_filtered


def fk_filter(data, dt, dx, vmin=300.0, vmax=4000.0,
              fmin=5.0, fmax=80.0, n_smooth=10, mask_fk = None):
    """Filter data in the f-k domain by phase-velocity and frequency limits.

    Parameters
    ----------
    data : np.ndarray
        Data with shape (nx, nt). It is tapered in place by :func:`fk`.
    dt : float
        Time sampling interval (s).
    dx : float
        Spatial sampling interval (m).
    vmin, vmax : float, optional
        Minimum and maximum absolute phase velocity passed (m/s).
        Defaults are 300 and 4000.
    fmin, fmax : float, optional
        Minimum and maximum frequency passed (Hz). Defaults are 5 and 80.
    n_smooth : int, optional
        Number of 5-point averaging passes applied to the mask. Default is 10.
    mask_fk : np.ndarray, optional
        Precomputed f-k mask with shape (nx, nt). If given, it replaces the
        computed mask.

    Returns
    -------
    d_filtered : np.ndarray
        Filtered data in the space-time domain, shape (nx, nt).
    mask : np.ndarray
        Complex f-k mask that was applied.
    """

    f, k, d_fk_r = fk(data, dt, dx, distance_scaling=0.0, taper=0.05)

    mask = np.ones(np.shape(d_fk_r), dtype='complex64')

    nx = np.shape(d_fk_r)[0]
    nt = np.shape(d_fk_r)[1]

    # frequency indices closest to fmin and fmax
    ifmin = np.where(np.abs(f-fmin) == np.min(np.abs(f-fmin)))[0][0]
    ifmax = np.where(np.abs(f-fmax) == np.min(np.abs(f-fmax)))[0][0]

    mask[:, 0:ifmin] = 0.0
    mask[:, ifmax:-1] = 0.0

    for i in range(nx):
        for j in np.arange(ifmin, ifmax):

            # phase velocity c = omega / k
            if np.abs(k[i]):
                c = 2.0*np.pi*f[j]/k[i]
            else:
                c = 1.0e9

            if (np.abs(c) > vmax) or (np.abs(c) < vmin):
                mask[i, j] = 0.0
            if np.abs(f[j]) > fmax:
                mask[i, j] = 0.0
            if np.abs(f[j]) < fmin:
                mask[i, j] = 0.0

    # smooth the mask by averaging over neighbouring samples
    for _ in range(n_smooth):
        mask[1:nx-1, 1:nt-1] = (mask[1:nx-1, 1:nt-1]+mask[0:nx-2, 1:nt-1] +
                                mask[2:nx, 1:nt-1]+mask[1:nx-1, 0:nt-2]+mask[1:nx-1, 2:nt])/5.0

    if mask_fk is not None:
        mask = mask_fk

    d_fk_r_filtered = d_fk_r*mask

    # undo the roll applied in fk()
    nt = len(f)
    nx = len(k)
    d_fk_filtered = np.roll(
        np.roll(d_fk_r_filtered, -int((nt-1)/2), axis=1), -int((nx-1)/2), axis=0)

    d_filtered = np.real(np.fft.ifft2(d_fk_filtered))/(dt*dx)

    return d_filtered, mask


def fs(d_fk_r, f, k, smin=0.0002, smax=0.003):
    """Compute the frequency-slowness transform from an f-k spectrum.

    The f-k amplitude at each frequency is interpolated (cubic) onto
    ``k = -2 * pi * s * f``.

    Parameters
    ----------
    d_fk_r : np.ndarray
        Rolled f-k spectrum with shape (nx, nt), as returned by :func:`fk`.
    f : np.ndarray
        Frequency axis (Hz).
    k : np.ndarray
        Angular wavenumber axis (rad/m).
    smin : float, optional
        Minimum slowness (s/m). Default is 0.0002.
    smax : float, optional
        Maximum slowness (s/m). Default is 0.003.

    Returns
    -------
    f : np.ndarray
        Frequency axis (Hz), unchanged.
    s : np.ndarray
        Slowness axis (s/m), length nx.
    d_fs : np.ndarray
        Amplitude spectrum with shape (nx, nt), indexed by (slowness,
        frequency). Only columns from ``nt // 2`` onward are filled; the
        rest are zero.
    """

    nt = np.shape(d_fk_r)[1]

    s = np.linspace(smin, smax, len(k))
    d_fs = np.zeros(np.shape(d_fk_r))

    for i in np.arange(int(nt/2), nt):
        f_interp = interpolate.interp1d(
            k, np.abs(d_fk_r[:, i]), kind='cubic', bounds_error=False, fill_value=0.0)
        # wavenumbers corresponding to the slowness samples
        k_interp = -2.0*np.pi*s*f[i]
        d_fs[:, i] = f_interp(k_interp)

    return f, s, d_fs


def fc(d_fk_r, f, k, c_min=200.0, c_max=4000.0):
    """Compute the frequency-phase velocity transform from an f-k spectrum.

    The transform is evaluated on slowness samples uniformly spaced between
    ``1 / c_max`` and ``1 / c_min``, so the returned velocity axis is not
    uniform.

    Parameters
    ----------
    d_fk_r : np.ndarray
        Rolled f-k spectrum with shape (nx, nt), as returned by :func:`fk`.
    f : np.ndarray
        Frequency axis (Hz).
    k : np.ndarray
        Angular wavenumber axis (rad/m).
    c_min : float, optional
        Minimum phase velocity (m/s). Default is 200.
    c_max : float, optional
        Maximum phase velocity (m/s). Default is 4000.

    Returns
    -------
    f : np.ndarray
        Frequency axis (Hz), unchanged.
    c : np.ndarray
        Phase-velocity axis (m/s), length nx, decreasing from ``c_max`` to
        ``c_min``.
    d_fc : np.ndarray
        Amplitude spectrum with shape (nx, nt), indexed by (velocity,
        frequency). Only columns from ``nt // 2`` onward are filled.
    """

    nt = np.shape(d_fk_r)[1]

    s = np.linspace(1.0/c_max, 1.0/c_min, len(k))
    d_fs = np.zeros(np.shape(d_fk_r))

    for i in np.arange(int(nt/2), nt):
        f_interp = interpolate.interp1d(
            k, np.abs(d_fk_r[:, i]), kind='cubic', bounds_error=False, fill_value=0.0)
        # wavenumbers corresponding to the slowness samples
        k_interp = -2.0*np.pi*s*f[i]
        d_fs[:, i] = f_interp(k_interp)

    c = 1.0/s

    return f, c, d_fs
