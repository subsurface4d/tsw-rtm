"""Source time functions and moment tensors for wave-equation modelling."""

import numpy as np

def wavelet(nt, dt, f0, amp0 = 1, t0 = None, type = 'Ricker'):
    """Generate a source time function.

    Parameters
    ----------
    nt : int
        Number of time samples.
    dt : float
        Time sampling interval (s).
    f0 : float
        Dominant frequency (Hz).
    amp0 : float, optional
        Amplitude scale. Default is 1.
    t0 : float, optional
        Time shift (s). Default is ``1.2 / f0``. For ``'ramp'`` it is the
        rise-time constant instead of a delay.
    type : str, optional
        Wavelet type (case-insensitive):

        - ``'ricker'``: Ricker wavelet centred at ``t0``.
        - ``'gaussian'``: the Ricker wavelet integrated twice by cumulative
          summation (not normalised).
        - ``'ramp'``: ``amp0 * 0.5 * (1 + tanh(t / t0))``.

        Default is ``'Ricker'``.

    Returns
    -------
    np.ndarray
        Wavelet of length ``nt``.

    Raises
    ------
    ValueError
        If ``type`` is not supported.
    """
    t = np.arange(nt) * dt + 0.0
    wavelet = np.zeros_like(t)
    t0 = t0 if t0 is not None else 1.2 / f0

    if type.lower() in ['ricker']:
        temp = (np.pi*f0) ** 2
        wavelet = amp0 * (1 - 2 * temp * (t - t0) ** 2) * np.exp(- temp * (t - t0) ** 2)

    elif type.lower() in ['gaussian']:
        temp = (np.pi*f0) ** 2
        wavelet = amp0 * (1 - 2 * temp * (t - t0) ** 2) * np.exp(- temp * (t - t0) ** 2)

        # integrate the Ricker wavelet twice
        wavelet = np.cumsum(wavelet)
        wavelet = np.cumsum(wavelet)

    elif type.lower() in ['ramp']:
        wavelet = amp0 * 0.5 * (1. + np.tanh(t / t0))

    else:
        msg = 'Support source types: Rikcer, Guassian, Ramp. \n'
        err = 'Unknown source type: {}'.format(type)
        raise ValueError(msg + '\n' + err)

    return wavelet


def moment_tensor(strike: float, dip: float, rake: float) -> np.ndarray:
    """Compute the unit moment tensor of a double-couple fault mechanism.

    Uses ``M_ij = s_i n_j + s_j n_i`` with slip vector ``s`` and fault normal
    ``n`` in the Aki & Richards convention (x north, y east, z down).

    Parameters
    ----------
    strike : float
        Fault strike (degrees).
    dip : float
        Fault dip (degrees).
    rake : float
        Slip rake (degrees).

    Returns
    -------
    np.ndarray
        Symmetric 3x3 moment tensor for unit scalar moment.
    """

    pi180 = np.pi/180
    CS=np.cos(strike*pi180)
    SS=np.sin(strike*pi180)
    CDI=np.cos(dip*pi180)
    SDI=np.sin(dip*pi180)
    CR=np.cos(rake*pi180)
    SR=np.sin(rake*pi180)
    # slip vector
    AS1=CR*CS+SR*CDI*SS
    AS2=CR*SS-SR*CDI*CS
    AS3=-SR*SDI
    # fault normal
    AN1=-SDI*SS
    AN2=SDI*CS
    AN3=-CDI
    CM11=2.*AS1*AN1
    CM22=2.*AS2*AN2
    CM33=2.*AS3*AN3
    CM12=(AS1*AN2+AS2*AN1)
    CM13=(AS1*AN3+AS3*AN1)
    CM23=(AS2*AN3+AS3*AN2)

    CM_FD = np.zeros((3,3))
    CM_FD[0,0] = CM11
    CM_FD[0,1] = CM12
    CM_FD[0,2] = CM13
    CM_FD[1,0] = CM12
    CM_FD[1,1] = CM22
    CM_FD[1,2] = CM23
    CM_FD[2,0] = CM13
    CM_FD[2,1] = CM23
    CM_FD[2,2] = CM33

    return CM_FD



def cosin_tapered_wavelet(nt, dt, f1, f2, f3, f4, t0=None):
    """Generate a band-limited wavelet with a cosine-squared tapered spectrum.

    The amplitude spectrum is 0 below ``f1``, rises as a cosine-squared taper
    to 1 at ``f2``, is flat up to ``f3``, and falls to 0 at ``f4``. A linear
    phase delays the wavelet by ``t0``. Adapted from code by Ali and Ettore.

    Parameters
    ----------
    nt : int
        Number of time samples.
    dt : float
        Time sampling interval (s).
    f1 : float
        Low cut-off frequency (Hz).
    f2 : float
        Start of the flat pass band (Hz).
    f3 : float
        End of the flat pass band (Hz).
    f4 : float
        High cut-off frequency (Hz); must not exceed the Nyquist frequency.
    t0 : float, optional
        Time delay (s). Default is ``1.2 / f1``.

    Returns
    -------
    np.ndarray
        float32 wavelet of length ``nt``, normalised to unit peak amplitude.

    Raises
    ------
    ValueError
        If ``f1 < f2 <= f3 < f4`` does not hold, or ``f4`` exceeds the
        Nyquist frequency.
    """
    if t0 is None:
        t0 = 1.2 / f1

    nts = nt
    dts = dt
    ots = 0.0
    timeDelay = t0

    waveletNd = np.zeros(nts, dtype=np.float32)
    # pad odd lengths to an even number of samples for the spectrum
    odd = False
    if (nts % 2 != 0):
        nts += 1
        odd = True
    waveletFftNd = np.zeros(nts, dtype=np.complex64)
    t = np.arange(nts) * dts

    if (not (f1 < f2 <= f3 < f4)):
        raise ValueError(
            "**** ERROR: Corner frequencies values must be increasing ****\n")

    fNyquist = 1 / (2 * dts)
    if (f4 > fNyquist):
        raise ValueError("**** ERROR: f4 > fNyquist ****\n")

    df = 1.0 / ((nts) * dts)

    # positive-frequency spectrum with a linear phase shift for the delay
    for iFreq in range(nts // 2):
        f = iFreq * df
        if (f < f1):
            waveletFftNd[iFreq] = 0
        elif (f1 <= f < f2):
            waveletFftNd[iFreq] = np.cos(np.pi / 2.0 * (f2 - f) /
                                            (f2 - f1)) * np.cos(np.pi / 2.0 *
                                                                (f2 - f) / (f2 - f1))
            waveletFftNd[iFreq] = waveletFftNd[iFreq] * np.exp(
                -1j * 2.0 * np.pi * f * timeDelay)
        elif (f2 <= f < f3):
            waveletFftNd[iFreq] = 1.0
            waveletFftNd[iFreq] = waveletFftNd[iFreq] * np.exp(
                -1j * 2.0 * np.pi * f * timeDelay)
        elif (f3 <= f < f4):
            waveletFftNd[iFreq] = np.cos(np.pi / 2.0 * (f - f3) /
                                            (f4 - f3)) * np.cos(np.pi / 2.0 *
                                                                (f - f3) / (f4 - f3))
            waveletFftNd[iFreq] = waveletFftNd[iFreq] * np.exp(
                -1j * 2.0 * np.pi * f * timeDelay)
        elif (f >= f4):
            waveletFftNd[iFreq] = 0

    # Hermitian symmetry for a real-valued wavelet
    waveletFftNd[nts // 2 + 1:] = np.flip(waveletFftNd[1:nts // 2].conj(),
                                            axis=0)

    if (odd):
        waveletNd[:] = np.fft.ifft(waveletFftNd[:-1]).real
    else:
        waveletNd[:] = np.fft.ifft(waveletFftNd[:]).real

    waveletNd = waveletNd / np.max(np.abs(waveletNd))

    return waveletNd
