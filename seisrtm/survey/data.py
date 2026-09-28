"""Shot-gather container for simulated seismic data, with I/O, simple
processing (windowing, band-pass filtering) and plotting utilities."""

import numpy as np
import matplotlib.pyplot as plt

from typing import Optional, List, Union
import torch

from seisrtm.survey import Survey
from seisrtm.signal import fk, fc, bandpass_filter

class SeismicData():
    """Seismic data stored as shot gathers.

    Data are held in ``self.data`` as a dictionary keyed by receiver
    component (e.g. ``'vx'``, ``'vz'``, ``'pr'``, ``'das'``); each entry has
    shape (nshots, nrec, nt).

    Parameters
    ----------
    survey : Survey
        Survey providing source/receiver geometry and time sampling.
    """

    def __init__(self, survey: Survey):

        # acquisition geometry and time axis from the survey
        self.src_num = survey.source.num
        self.rec_num = survey.receiver.num
        self.src_loc = survey.source.get_loc()
        self.rec_loc = survey.receiver.get_loc_dict()
        self.rec_type = survey.receiver.get_type()
        self.src_type = survey.source.get_type()
        self.nt = survey.receiver.nt
        self.dt = survey.receiver.dt
        self.t = np.arange(self.nt) * self.dt  # starts at 0 s

        self.first_break = None  # in s

        # filled by record_data
        self.data = None

    def __repr__(self):
        """Return a short summary of the data dimensions."""

        info = f"Seismic Data:\n"
        info += f"  Source number : {self.src_num}\n"
        info += f"  Receiver number : {self.rec_num}\n"
        info += f"  Time samples : {self.nt} samples at {self.dt * 1000:.2f} ms\n"

        return info

    @classmethod
    def load_from_segy_files(self):
        """Load shot gathers from SEG-Y files (not implemented).

        Raises
        ------
        NotImplementedError
            Always.
        """

        raise NotImplementedError('This function is not implemented yet')

    @classmethod
    def load_from_sep_files(self):
        """Load shot gathers from SEP files (not implemented).

        Raises
        ------
        NotImplementedError
            Always.
        """

        raise NotImplementedError('This function is not implemented yet')


    def select_time_window(self, tmin: Optional[float] = 0.0, tmax: Optional[float] = None, nt: Optional[int] = None):
        """Keep only the samples within a time window (in place).

        Parameters
        ----------
        tmin : float, optional
            Start time of the window (s), by default 0.0.
        tmax : float, optional
            End time of the window (s); defaults to the last sample time.
        nt : int, optional
            If given, overrides ``tmax`` with the time of sample ``nt - 1``.
        """

        if tmax is None:
            tmax = self.t[-1]

        if nt is not None:
            tmax = self.t[nt-1]

        # indices of samples inside the window
        index = np.arange(self.nt)
        index = index[(self.t >= tmin) & (self.t <= tmax)]

        self.t = self.t[index]
        self.nt = len(self.t)

        print(f'Select time window: {tmin * 1000:.2f} ms - {tmax * 1000:.2f} ms')
        print(f'New time samples: {self.nt} samples at {self.dt * 1000:.2f} ms')

        for comp in set(self.rec_type):
            self.data[comp] = self.data[comp][:, :, index]

    def resample(self, dt: float):
        """Resample the data to a new time interval (not implemented).

        Returns immediately if ``dt`` equals the current interval.

        Parameters
        ----------
        dt : float
            New time sampling interval (s).

        Raises
        ------
        NotImplementedError
            If ``dt`` differs from the current interval.
        """

        if dt == self.dt:
            print(f'The data is already sampled at {dt * 1000:.2f} ms')
            return

        raise NotImplementedError('This function is not implemented yet')


        print(f'Resample the data: {self.dt * 1000:.2f} ms -> {dt * 1000:.2f} ms')
        print(f'New time samples: {self.nt} samples at {dt * 1000:.2f} ms')


    def record_data(self, data: dict):
        """Attach shot-gather data.

        Parameters
        ----------
        data : dict
            Data keyed by component, each of shape (nshots, nrec, nt).
        """

        self.data = data

    def save(self, path: str, save_seperate: Optional[bool] = False):
        """Save data and geometry to a single ``.npz`` file.

        Parameters
        ----------
        path : str
            Output file path.
        save_seperate : bool, optional
            Reserved for saving each shot/component separately; currently
            ignored.
        """

        data_save = {'data': self.data,
                    'src_loc': self.src_loc,
                    'rec_loc': self.rec_loc,
                    'src_num': self.src_num,
                    'rec_num': self.rec_num,
                    'rec_type': self.rec_type,
                    'src_type': self.src_type,
                    't': self.t,
                    'nt': self.nt,
                    'dt': self.dt}

        np.savez(path, **data_save)


    @classmethod
    def load(cls, path: str):
        """Load data previously written by :meth:`save`.

        Parameters
        ----------
        path : str
            Path to the ``.npz`` file.

        Returns
        -------
        SeismicData
            Restored object (``first_break`` is not set).
        """

        data = np.load(path, allow_pickle=True)

        # bypass __init__, which requires a Survey
        seismic_data = cls.__new__(cls)

        seismic_data.data = data['data'].item()
        seismic_data.src_loc = data['src_loc']
        seismic_data.rec_loc = data['rec_loc'].item()
        seismic_data.src_num = data['src_num']
        seismic_data.rec_num = data['rec_num']
        seismic_data.rec_type = data['rec_type']
        seismic_data.src_type = data['src_type']
        seismic_data.t = data['t']
        seismic_data.nt = data['nt']
        seismic_data.dt = data['dt']

        return seismic_data


    def check_same(self, other: 'SeismicData'):
        """Check that another dataset has the same geometry and time sampling.

        Parameters
        ----------
        other : SeismicData
            Dataset to compare with.

        Raises
        ------
        TypeError
            If ``other`` is not a SeismicData object.
        ValueError
            If source locations, receiver components/locations, ``nt`` or
            ``dt`` differ.
        """

        if not isinstance(other, SeismicData):
            raise TypeError('Input must be a SeismicData object.')

        if not np.all(self.src_loc == other.src_loc):
            raise ValueError('The source locations are not the same')
        if not self.rec_loc.keys() == other.rec_loc.keys():
            raise ValueError('The receiver components are not the same')
        for comp in self.rec_loc.keys():
            if not np.all(np.array(self.rec_loc[comp]) == np.array((other.rec_loc[comp]))):
                raise ValueError(f'The receiver locations for component {comp} are not the same')
        if not self.nt == other.nt:
            raise ValueError('The time samples are not the same')
        if not self.dt == other.dt:
            raise ValueError('The time sampling interval are not the same')

    def get_data(self, shotid: Optional[int] = 0, comp: Optional[str] = 'vz'):
        """Return one shot gather as a NumPy array.

        Parameters
        ----------
        shotid : int, optional
            Shot index, by default 0.
        comp : str, optional
            Receiver component, by default 'vz'.

        Returns
        -------
        np.ndarray
            Shot gather of shape (nrec, nt).

        Raises
        ------
        ValueError
            If no data are recorded, the component is unavailable, or
            ``shotid`` is out of range.
        """

        if self.data is None:
            raise ValueError('The shot gather data is not available, forward modeling is needed')

        if comp not in self.rec_type:
            raise ValueError(f'The component {comp} is not available, available components are {set(self.rec_type)}')

        if shotid > self.src_num - 1 :
            raise ValueError(f'The shot id {shotid} is not available, available shot ids are 0 - {self.src_num-1}')

        sg_data = self.data[comp][shotid]

        # convert torch tensors to NumPy
        if not isinstance(sg_data, np.ndarray):
            try:
                sg_data = sg_data.cpu().detach().numpy()
            except AttributeError:
                pass

        return sg_data


    def plot(self, shotid: Optional[int] = 0, comp: Optional[Union[str, List[str]]] = 'vz',
             clip: Optional[float] = 99.9, cmap: Optional[str] = 'gray',
             aspect: Optional[Union[str, float]] = 'auto',
             figsize: Optional[tuple] = (4,6),
             show_colorbar: Optional[bool] = False,
             show_first_break: Optional[bool] = False,
             save_path: Optional[str] = None):
        """Plot a shot gather, one panel per component.

        Parameters
        ----------
        shotid : int, optional
            Shot index, by default 0.
        comp : str or list of str, optional
            Component(s) to plot, by default 'vz'.
        clip : float, optional
            Percentile of the amplitudes used as the color limit, by default 99.9.
        cmap : str, optional
            Colormap, by default 'gray'.
        aspect : str or float, optional
            Image aspect ratio, by default 'auto'.
        figsize : tuple, optional
            Size of a single panel; width is scaled by the number of
            components. By default (4, 6).
        show_colorbar : bool, optional
            Add a colorbar to each panel, by default False.
        show_first_break : bool, optional
            Overlay picked first breaks, if available, by default False.
        save_path : str, optional
            If given, save the figure to this path.
        """

        if isinstance(comp, str):
            comp = [comp]

        for c in comp:
            if c not in set(self.rec_type):
                print(f'The component {comp} is not available, available components are {set(self.rec_type)}')
                return None


        figsize = (figsize[0] * len(comp), figsize[1])

        fig = plt.figure(figsize=figsize)

        for i, c in enumerate(comp):

            data = self.get_data(shotid, c)

            channel = np.arange(data.shape[0])

            vmax = np.percentile(data, clip)

            parms = {'cmap':cmap,
                    'aspect':aspect,
                    'vmin':-vmax,
                    'vmax':vmax,
                    'extent': [channel[0], channel[-1], self.t[-1], self.t[0]]}

            ax = fig.add_subplot(1,len(comp),i+1)
            ax.imshow(data.T, **parms)
            ax.set_xlabel('Channel#')
            if i == 0:
                ax.set_ylabel('Time (s)')
            ax.set_title(f' Shot #{shotid} ({c.upper()})')
            ax.grid(True, axis='y', alpha=0.5)

            if show_colorbar:
                fig.colorbar(ax.images[0], ax=ax, orientation='vertical')

            if show_first_break:
                if self.first_break is None:
                    print('The first break is not picked, please run the first break picking algorithm first')
                else:
                    ax.plot(channel, self.first_break[c][shotid], 'r-', linewidth=1.5)

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()


    def plot_compare(self, shotdata: 'SeismicData',
                        shotid: Optional[int] = 0,
                        comp: Optional[str] = 'vz',
                        clip: Optional[float] = 99.9,
                        ratio: Optional[float] = 1.0,
                        cmap: Optional[str] = 'gray',
                        aspect: Optional[Union[str, float]] = 'auto',
                        figsize: Optional[tuple] = (12,6),
                        show_colorbar: Optional[bool] = False,
                        title: Optional[str] = ['Data 1', 'Data 2', 'Difference'],
                        save_path: Optional[str] = None):
        """Plot two shot gathers and their difference side by side.

        Parameters
        ----------
        shotdata : SeismicData
            Dataset to compare with; must pass :meth:`check_same`.
        shotid : int, optional
            Shot index, by default 0.
        comp : str, optional
            Component to plot, by default 'vz'.
        clip : float, optional
            Percentile of this dataset's amplitudes used as the color limit,
            by default 99.9.
        ratio : float, optional
            Amplification factor for the difference panel, by default 1.0.
        cmap : str, optional
            Colormap, by default 'gray'.
        aspect : str or float, optional
            Image aspect ratio, by default 'auto'.
        figsize : tuple, optional
            Figure size, by default (12, 6).
        show_colorbar : bool, optional
            Add a colorbar to each panel, by default False.
        title : list of str, optional
            Titles of the three panels.
        save_path : str, optional
            If given, save the figure to this path.
        """

        self.check_same(shotdata)

        data1 = self.get_data(shotid, comp)
        data2 = shotdata.get_data(shotid, comp)

        data_diff = data1 - data2

        channel = np.arange(data1.shape[0])
        vmax = np.percentile(data1, clip)

        vmaxs = [vmax, vmax, vmax / ratio]
        title = [title[0], title[1], title[2] + f' (x{ratio})']

        fig = plt.figure(figsize=figsize)

        for i, data in enumerate([data1, data2, data_diff]):

            parms = {'cmap':cmap,
                    'aspect':aspect,
                    'vmin':-vmaxs[i],
                    'vmax':vmaxs[i],
                    'extent': [channel[0], channel[-1], self.t[-1], self.t[0]]}

            ax = fig.add_subplot(1,3,i+1)
            ax.imshow(data.T, **parms)
            ax.set_xlabel('Channel')
            ax.set_ylabel('Time (s)')
            ax.set_title(title[i])
            ax.grid(True, axis='y', alpha=0.5)

            if show_colorbar:
                fig.colorbar(ax.images[0], ax=ax, orientation='vertical')

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_dispersion(self, shotid: Optional[int] = 0,
                        comp: Optional[str] = 'vz',
                        cmin: Optional[float] = 200.0,
                        cmax: Optional[float] = 4000.0,
                        fmin: Optional[float] = 0.0,
                        fmax: Optional[float] = 50.0,
                        normalize: Optional[bool] = False,
                        clip: Optional[float] = 99.9,
                        cmap: Optional[str] = 'gray',
                        aspect: Optional[Union[str, float]] = 'auto',
                        figsize: Optional[tuple] = (8,6),
                        show_colorbar: Optional[bool] = False,
                        save_path: Optional[str] = None,
                        return_data: Optional[bool] = False):
        """Plot the frequency--phase-velocity dispersion image of a shot gather.

        The image is obtained from an f-k transform followed by mapping to
        phase velocity. The receiver spacing is taken as the distance between
        the first two receivers of ``comp``.

        Parameters
        ----------
        shotid : int, optional
            Shot index, by default 0.
        comp : str, optional
            Component to analyse, by default 'vz'.
        cmin, cmax : float, optional
            Phase-velocity range (m/s), by default 200.0 and 4000.0.
        fmin, fmax : float, optional
            Frequency range (Hz), by default 0.0 and 50.0.
        normalize : bool, optional
            Normalize each frequency column by its maximum, by default False.
        clip : float, optional
            Percentile used as the upper color limit, by default 99.9.
        cmap : str, optional
            Colormap, by default 'gray'.
        aspect : str or float, optional
            Axes aspect ratio, by default 'auto'.
        figsize : tuple, optional
            Figure size, by default (8, 6).
        show_colorbar : bool, optional
            Add a colorbar, by default False.
        save_path : str, optional
            If given, save the figure to this path.
        return_data : bool, optional
            Return the plotted arrays, by default False.

        Returns
        -------
        f_plot : np.ndarray
            Frequencies (Hz). Returned only if ``return_data`` is True.
        c_plot : np.ndarray
            Phase velocities (m/s).
        d_fs_plot : np.ndarray
            Dispersion image of shape (nc, nf).
        """


        data = self.get_data(shotid, comp)

        # receiver spacing from the first two receivers
        rec1 = self.rec_loc[comp][0]
        rec2 = self.rec_loc[comp][1]
        dx = np.sqrt((rec2[0] - rec1[0])**2 + (rec2[1] - rec1[1])**2)

        dt = self.dt

        print(f' dx = {dx:.2f} m')
        print(f' dt = {dt * 1000:.2f} ms')

        f, k, d_fk_r = fk(data, dt, dx, )
        f, c, d_fs = fc(d_fk_r, f, k, c_min=cmin, c_max=cmax)

        # nearest indices to the requested frequency/velocity limits
        ifmin = np.where(np.abs(f-fmin)==np.min(np.abs(f-fmin)))[0][0]
        ifmax = np.where(np.abs(f-fmax)==np.min(np.abs(f-fmax)))[0][0]
        icmax = np.where(np.abs(c-cmin)==np.min(np.abs(c-cmin)))[0][0]
        icmin = np.where(np.abs(c-cmax)==np.min(np.abs(c-cmax)))[0][0]

        f_plot = f[ifmin:ifmax]
        c_plot = c[icmin:icmax]
        d_fs_plot = d_fs[icmin:icmax, ifmin:ifmax]

        if normalize:
            d_fs_plot = d_fs_plot / np.max(d_fs_plot, axis=0)

        vmax = np.percentile(d_fs_plot, clip)

        fig = plt.figure(figsize=figsize)

        ax = fig.add_subplot(1,1,1)
        plt.pcolor(f_plot, c_plot, d_fs_plot, cmap=cmap, clim=(0.0, vmax), shading='auto')
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Phase Velocity (m/s)')
        ax.set_title(f' Dispersion Curve of Shot #{shotid} ({comp.upper()})')
        ax.grid(True, axis='y', alpha=0.5)
        plt.gca().set_aspect(aspect, adjustable='box')

        if show_colorbar:
            plt.colorbar(orientation='horizontal')

        plt.tight_layout()
        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()


        if return_data:
            return f_plot, c_plot, d_fs_plot



    def plot_trace(self, shotid: Optional[int] = 0,
             traceid: Optional[int] = 0,
             comp: Optional[Union[str, List[str]]] = 'vz',
             figsize: Optional[tuple] = (8,3),
             save_path: Optional[str] = None):
        """Plot a single trace, one panel per component.

        Parameters
        ----------
        shotid : int, optional
            Shot index, by default 0.
        traceid : int, optional
            Trace (receiver) index, by default 0.
        comp : str or list of str, optional
            Component(s) to plot, by default 'vz'.
        figsize : tuple, optional
            Size of a single panel; width is scaled by the number of
            components. By default (8, 3).
        save_path : str, optional
            If given, save the figure to this path.
        """

        if isinstance(comp, str):
            comp = [comp]

        figsize = (figsize[0] * len(comp), figsize[1])

        fig = plt.figure(figsize=figsize)

        for i, c in enumerate(comp):

                data = self.get_data(shotid, c)

                ax = fig.add_subplot(1,len(comp),i+1)
                ax.plot(self.t, data[traceid], 'k-')
                ax.set_ylabel('Amplitude')
                if i == 0:
                    ax.set_xlabel('Time (s)')
                ax.set_title(f' Shot #{shotid} ({c.upper()})')
                ax.grid(True, axis='y', alpha=0.5)

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()


    def plot_compare_trace(self, shotdata: 'SeismicData',
                            shotid: Optional[int] = 0,
                            traceid: Optional[int] = 0,
                            comp: Optional[str] = 'vz',
                            linewidth: Optional[float] = 1.5,
                            linestyle: Optional[List[str]] = ['k-', 'r-'],
                            figsize: Optional[tuple] = (8,3),
                            title: Optional[str] = ['Data 1', 'Data 2'],
                            normlize: Optional[bool] = False,
                            save_path: Optional[str] = None):
        """Overlay the same trace from two datasets.

        Parameters
        ----------
        shotdata : SeismicData
            Dataset to compare with; must pass :meth:`check_same`.
        shotid : int, optional
            Shot index, by default 0.
        traceid : int, optional
            Trace (receiver) index, by default 0.
        comp : str, optional
            Component to plot, by default 'vz'.
        linewidth : float, optional
            Line width, by default 1.5.
        linestyle : list of str, optional
            Matplotlib format strings for the two traces.
        figsize : tuple, optional
            Figure size, by default (8, 3).
        title : list of str, optional
            Legend labels of the two traces; the first is also the axes title.
        normlize : bool, optional
            Normalize each trace by its maximum absolute value, by default False.
        save_path : str, optional
            If given, save the figure to this path.
        """

        self.check_same(shotdata)

        data1 = self.get_data(shotid, comp)
        data2 = shotdata.get_data(shotid, comp)

        trace1 = data1[traceid]
        trace2 = data2[traceid]

        if normlize:
            trace1 = trace1 / (np.max(abs(trace1)) + 1e-10)
            trace2 = trace2 / (np.max(abs(trace2)) + 1e-10)

        fig = plt.figure(figsize=figsize)

        ax = fig.add_subplot(1,1,1)
        ax.plot(self.t, trace1, linestyle[0], linewidth=linewidth, label=title[0])
        ax.plot(self.t, trace2, linestyle[1], linewidth=linewidth, label=title[1])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(title[0])
        ax.grid(True, axis='y', alpha=0.5)
        ax.legend()

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()


    def filter_bandpass(self, lowcut = 2, highcut = 10, order = 4):
        """Band-pass filter all shot gathers in place.

        Each gather is filtered in NumPy/SciPy and copied back as a tensor
        with the original dtype and device.

        Parameters
        ----------
        lowcut : float, optional
            Low corner frequency (Hz), by default 2.
        highcut : float, optional
            High corner frequency (Hz), by default 10.
        order : int, optional
            Filter order, by default 4.
        """

        dt = self.dt
        dtype = self.data[self.rec_type[0]].dtype
        device = self.data[self.rec_type[0]].device

        for comp in self.data.keys():
            for isrc in range(self.src_num):
                data = self.get_data(isrc, comp)
                data_bp = bandpass_filter(data, dt, lowcut, highcut, order=order)

                self.data[comp][isrc] = torch.tensor(data_bp.copy(), dtype=dtype, device=device)

        print(f'Bandpass filter the data: {lowcut} - {highcut} Hz')
