"""Model smoothing and wavefield animation helpers."""

from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def smooth2d(data, span_x, span_z):
    """Smooth a 2-D grid with a Gaussian filter.

    Parameters
    ----------
    data : np.ndarray
        2-D array with shape (nz, nx).
    span_x : float
        Gaussian standard deviation along x (grid points).
    span_z : float
        Gaussian standard deviation along z (grid points).

    Returns
    -------
    np.ndarray
        Smoothed array with the shape of ``data``.
    """

    return gaussian_filter(data, sigma=(span_z, span_x))


def plot_wavefield(wavefield):
    """Animate wavefield snapshots.

    The colour scale is fixed to +/- 80 % of the global maximum.

    Parameters
    ----------
    wavefield : np.ndarray
        Snapshots with shape (nt, nz, nx); each ``wavefield[i]`` is drawn as
        one frame.

    Returns
    -------
    matplotlib.animation.ArtistAnimation
        Looping animation with 500 ms between frames.
    """

    fig = plt.figure()
    ims = []

    for i in range(wavefield.shape[0]):
        caxis = wavefield.max() * 0.8
        im = plt.imshow(wavefield[i], vmin = -caxis, vmax = caxis, aspect = 'equal', cmap='seismic', animated=False)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True,repeat=True, repeat_delay=0)
    plt.close()

    return ani
