"""2-D acquisition geometry: combines sources and receivers, maps them onto
the finite-difference grid, and assembles recorded data by component."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
import torch

from .receiver import Receiver
from .source import Source


class Survey(object):
    """2-D seismic acquisition geometry.

    All sources share the same receivers, number of time samples and time
    interval.

    Parameters
    ----------
    source : Source
        Source object.
    receiver : Receiver
        Receiver object.
    device : str, optional
        Computation device, 'cpu' or 'cuda[:N]', by default 'cpu'.
    cpu_num : int, optional
        Maximum number of CPU cores (used when device is 'cpu'), by default 1.
    gpu_num : int, optional
        Maximum number of GPUs (used when device is 'cuda'), by default 1.
    reciprocity : bool, optional
        Swap sources and receivers using reciprocity, by default False.
    simultaneous : bool, optional
        Fire all sources simultaneously as a single shot, by default False.
    interpolation : bool, optional
        Use Kaiser-windowed sinc interpolation (Hicks, 2002) for off-grid
        sources and receivers, by default False.

    Notes
    -----
    1. Receiver locations are not shot-dependent; see :class:`Receiver`.
    2. Only 2-D (x, z) simulation is supported.
    3. Parallelization is shot by shot, not by domain decomposition. Several
       shots can share one GPU as long as memory suffices to cache the
       wavefields needed for the gradient.
    4. Interpolation is costly but required for DAS acquisition.
    """

    def __init__(
        self,
        source: Source,
        receiver: Receiver,
        device: Optional[str] = "cpu",
        cpu_num: Optional[int] = 1,
        gpu_num: Optional[int] = 1,
        reciprocity: Optional[bool] = False,
        simultaneous: Optional[bool] = False,
        interpolation: Optional[bool] = False,
    ) -> None:

        # acquisition
        self.source = source
        self.receiver = receiver

        # computation
        self.device = device
        self.cpu_num = cpu_num
        self.gpu_num = gpu_num

        # acquisition options
        self.reciprocity = reciprocity
        self.simultaneous = simultaneous
        self.interpolation = interpolation

        self.__check__()


    def __repr__(self):
        """Return a summary of the survey, sources, receivers and cable."""

        info = f"Survey Information:\n"
        info += f"  Device   : {self.device}\n"
        info += f"  CPU num  : {self.cpu_num}\n" if self.device == "cpu" else ""
        info += f"  GPU num  : {self.gpu_num}\n" if self.device.startswith("cuda") else ""
        info += f"  Apply reciprocity: {self.reciprocity}\n"
        info += f"  Simultaneous source: {self.simultaneous}\n"
        info += f"  Apply interpolation: {self.interpolation}\n"
        info += "\n"
        info += repr(self.source)
        info += "\n"
        info += repr(self.receiver)
        info += "\n"
        if self.receiver.cable is not None:
            info += repr(self.receiver.cable)

        return info

    def __check__(self):
        """Validate the survey and adjust device settings.

        Falls back to CPU if no GPU is available, and caps ``gpu_num`` by the
        number of available GPUs and by the number of shots.

        Raises
        ------
        AssertionError
            If source and receiver ``nt``/``dt`` differ, the device is not
            'cpu' or 'cuda*', or ``cpu_num``/``gpu_num`` is not positive.
        """

        assert (
            self.source.nt == self.receiver.nt
        ), "nt should be the same for source and receiver"
        assert (
            self.source.dt == self.receiver.dt
        ), "dt should be the same for source and receiver"
        assert self.device == "cpu" or self.device.startswith("cuda"),  "device should be either cpu or cuda"
        assert (
            self.cpu_num > 0 and self.gpu_num > 0
        ), "cpu_num and gpu_num should be larger than 0"

        gpu_num_avai = torch.cuda.device_count()

        if gpu_num_avai < 1 and self.device.startswith("cuda"):
            print(f"Warning: no GPUs available, set to use cpu.")
            self.device = "cpu"

        if gpu_num_avai < self.gpu_num and self.device.startswith("cuda"):
            print(
                f"Warning: requested {gpu_num_avai} GPUs, but only {self.gpu_num} are available."
            )
            self.gpu_num = gpu_num_avai

        if self.gpu_num > self.source.num and self.device.startswith("cuda"):
            print(
                f"Warning: requested {self.gpu_num} GPUs, but only {self.source.num} shots."
            )
            self.gpu_num = self.source.num



        if self.receiver.cable is not None and self.interpolation is False:
            print("Warning: For DAS, survey interpolation is recommended")


    def initialize(self, dx: float, dz: float) -> None:
        """Prepare the survey for simulation on a grid.

        Builds the DAS cable operator (if any), applies reciprocity (if
        enabled), and sets the grid-level source and receiver arrays. Must be
        called before running the simulation.

        Parameters
        ----------
        dx : float
            Grid spacing in x (m).
        dz : float
            Grid spacing in z (m); must equal ``dx``.

        Raises
        ------
        AssertionError
            If ``dx != dz``.
        """

        assert dx == dz, "dx and dz should be the same for this survey"

        self.dx = dx
        self.dz = dz

        if self.receiver.cable is not None:
            self.cable_operator = self.receiver.cable.get_operator(device=self.device)

        if self.reciprocity:
            self.__apply_reciprocity()

        self.__set_source()
        self.__set_receiver()


    def __apply_reciprocity(self) -> None:
        """Swap sources and receivers using reciprocity.

        Each pressure source becomes a pair of vx and a pair of vz receivers
        (a staggered dipole); each original receiver becomes a force source
        driven by the time-integrated wavelet of the first source. All
        sources are assumed to share that wavelet.

        Raises
        ------
        ValueError
            If sources are not all pressure sources, or the receiver type
            combination is unsupported.
        """

        print("Applying reciprocity to the survey ...")
        print("  assuming the same source wavelet for all sources!")

        # --- new receivers from the original sources ---

        src_type = self.source.get_type(unique=True)
        if len(src_type) > 1:
            raise ValueError(
                "Reciprocity Error: source type should be the same for all sources"
            )

        src_type = src_type[0]
        if src_type != "pr":
            raise ValueError("Reciprocity Error: source type should be pressure source")

        receiver = Receiver(nt=self.receiver.nt, dt=self.receiver.dt)
        src_loc = self.source.get_loc(type=src_type)

        for isrc in range(src_loc.shape[0]):
            receiver.add_receiver(src_loc[isrc] + np.array([0.0,      0.0]), "vx")
            receiver.add_receiver(src_loc[isrc] + np.array([self.dx,  0.0]), "vx")
            receiver.add_receiver(src_loc[isrc] + np.array([0.0, -self.dz]), "vz")
            receiver.add_receiver(src_loc[isrc] + np.array([0.0,      0.0]), "vz")

        # --- new sources from the original receivers ---

        source = Source(nt=self.source.nt, dt=self.source.dt, f0=self.source.f0)

        src_amp = self.source.get_wavelet(0)

        # integrate the wavelet to match the pressure-source scaling in __set_source
        src_amp_int = integrate.cumtrapz(
            src_amp, axis=-1, initial=0, dx=self.source.dt
        ) / (-2.0 * self.dx)

        rec_type = self.receiver.get_type(unique=True)

        if rec_type == ["vx"]:
            self.recip_type = "pr-vx"

            rec_loc = self.receiver.get_loc("vx")
            for irec in range(rec_loc.shape[0]):
                source.add_source(rec_loc[irec], src_amp_int, "vx")

        elif rec_type == ["vz"]:
            self.recip_type = "pr-vz"

            rec_loc = self.receiver.get_loc("vz")
            for irec in range(rec_loc.shape[0]):
                source.add_source(rec_loc[irec], src_amp_int, "vz")

        elif rec_type == ["vx", "vz"] or rec_type == ["vz", "vx"]:
            self.recip_type = "pr-vx-vz"

            rec_loc_vx = self.receiver.get_loc("vx")
            for irec in range(rec_loc_vx.shape[0]):
                source.add_source(rec_loc_vx[irec], src_amp_int, "vx")

            rec_loc_vz = self.receiver.get_loc("vz")
            for irec in range(rec_loc_vz.shape[0]):
                source.add_source(rec_loc_vz[irec], src_amp_int, "vz")

        elif rec_type == ["das"]:
            self.recip_type = "pr-das"

            gauge_len = self.receiver.cable.gauge_len
            chann_num = self.receiver.cable.chann_num
            rec_loc = self.receiver.cable.get_rec_loc()[:, [0, 2]]
            tangent = self.receiver.cable.get_tangent()[:, [0, 2]]

            # each channel becomes two opposite force pairs at its gauge ends,
            # projected onto the cable tangent
            for ic in range(chann_num):
                channl_neg = 2 * ic
                channl_pos = 2 * ic + 1
                source.add_source(
                    rec_loc[channl_pos],
                    src_amp_int * tangent[channl_pos, 0] / gauge_len,
                    "vx",
                )
                source.add_source(
                    rec_loc[channl_pos],
                    src_amp_int * tangent[channl_pos, 1] / gauge_len,
                    "vz",
                )
                source.add_source(
                    rec_loc[channl_neg],
                    -src_amp_int * tangent[channl_neg, 0] / gauge_len,
                    "vx",
                )
                source.add_source(
                    rec_loc[channl_neg],
                    -src_amp_int * tangent[channl_neg, 1] / gauge_len,
                    "vz",
                )

        else:
            raise ValueError("Reciprocity Error: unsupported receiver type:", rec_type)

        self.source = source
        self.receiver = receiver

    def __set_source(self) -> None:
        """Build grid-level force-source locations and amplitudes.

        Every source is expanded into 6 point forces per shot in x and z
        (unused slots have zero amplitude):

        - 'vx' / 'vz': a single force in x / z.
        - 'pr': explosive source as x and z dipoles of the integrated
          wavelet, scaled by ``-1 / (2 dx)``.
        - 'mt': moment tensor as force couples, scaled by ``1 / dx**3``.

        Sets ``source_fx_loc``, ``source_fz_loc`` of shape (nshots, nsrc, 2)
        and ``source_fx_amp``, ``source_fz_amp`` of shape (nshots, nsrc, nt).

        Raises
        ------
        RuntimeError
            If a source type is unknown.
        """

        nt = self.source.nt
        dt = self.source.dt
        dx = self.dx
        src_num = self.source.num
        src_num_per_shot = 6

        source_fx_loc = np.zeros((src_num, src_num_per_shot, 2))
        source_fz_loc = np.zeros((src_num, src_num_per_shot, 2))
        source_fx_amp = np.zeros((src_num, src_num_per_shot, nt))
        source_fz_amp = np.zeros((src_num, src_num_per_shot, nt))

        for isrc in range(src_num):
            # start all point forces at the source location, then offset below
            source_fx_loc[isrc, :, :] = np.tile(
                self.source.get_loc(isrc), (src_num_per_shot, 1)
            )
            source_fz_loc[isrc, :, :] = np.tile(
                self.source.get_loc(isrc), (src_num_per_shot, 1)
            )

            src_type = self.source.get_type(isrc)
            src_amp = self.source.get_wavelet(isrc)
            src_amp_int = integrate.cumtrapz(src_amp, axis=-1, initial=0, dx=dt) / (
                -2.0 * dx
            )

            if src_type == "vx":
                source_fx_amp[isrc, 0, :] = src_amp


            elif src_type == "vz":
                source_fz_amp[isrc, 0, :] = src_amp

            elif src_type == "pr":
                # dipoles in x and z
                source_fx_loc[isrc, 0, 0] += 0
                source_fx_loc[isrc, 0, 1] += 0

                source_fx_loc[isrc, 1, 0] += dx
                source_fx_loc[isrc, 1, 1] += 0

                source_fz_loc[isrc, 0, 0] += 0
                source_fz_loc[isrc, 0, 1] -= dx
                source_fz_loc[isrc, 1, 0] += 0
                source_fz_loc[isrc, 1, 1] += 0

                source_fx_amp[isrc, 0, :] = -1.0 * src_amp_int
                source_fx_amp[isrc, 1, :] =        src_amp_int
                source_fz_amp[isrc, 0, :] = -1.0 * src_amp_int
                source_fz_amp[isrc, 1, :] =        src_amp_int

            elif src_type == "mt":
                # moment tensor as force couples
                sm = self.source.get_moment_tensor(isrc)
                sm = sm / (self.dx * self.dx * self.dx)

                source_fx_loc[isrc, 0, 0] += 0
                source_fx_loc[isrc, 0, 1] += 0
                source_fx_loc[isrc, 1, 0] += dx
                source_fx_loc[isrc, 1, 1] += 0
                source_fx_loc[isrc, 2, 0] += 0
                source_fx_loc[isrc, 2, 1] += dx
                source_fx_loc[isrc, 3, 0] += dx
                source_fx_loc[isrc, 3, 1] += dx
                source_fx_loc[isrc, 4, 0] += 0
                source_fx_loc[isrc, 4, 1] -= dx
                source_fx_loc[isrc, 5, 0] += dx
                source_fx_loc[isrc, 5, 1] -= dx

                source_fz_loc[isrc, 0, 0] += dx
                source_fz_loc[isrc, 0, 1] += 0
                source_fz_loc[isrc, 1, 0] += dx
                source_fz_loc[isrc, 1, 1] -= dx
                source_fz_loc[isrc, 2, 0] -= dx
                source_fz_loc[isrc, 2, 1] += 0
                source_fz_loc[isrc, 3, 0] -= dx
                source_fz_loc[isrc, 3, 1] -= dx
                source_fz_loc[isrc, 4, 0] += 0
                source_fz_loc[isrc, 4, 1] -= dx
                source_fz_loc[isrc, 5, 0] += 0
                source_fz_loc[isrc, 5, 1] += 0

                # Mxx
                source_fx_amp[isrc, 0, :] = -sm[0, 0] * src_amp
                source_fx_amp[isrc, 1, :] =  sm[0, 0] * src_amp
                # Mxz (fx)
                source_fx_amp[isrc, 2, :] =  sm[0, 1] * src_amp / 4.0
                source_fx_amp[isrc, 3, :] =  sm[0, 1] * src_amp / 4.0
                source_fx_amp[isrc, 4, :] = -sm[0, 1] * src_amp / 4.0
                source_fx_amp[isrc, 5, :] = -sm[0, 1] * src_amp / 4.0
                # Mxz (fz)
                source_fz_amp[isrc, 0, :] =  sm[2, 1] * src_amp / 4.0
                source_fz_amp[isrc, 1, :] =  sm[2, 1] * src_amp / 4.0
                source_fz_amp[isrc, 2, :] = -sm[2, 1] * src_amp / 4.0
                source_fz_amp[isrc, 3, :] = -sm[2, 1] * src_amp / 4.0
                # Mzz
                source_fz_amp[isrc, 4, :] = -sm[2, 2] * src_amp
                source_fz_amp[isrc, 5, :] =  sm[2, 2] * src_amp

            else:
                raise RuntimeError("Unknown source type: %s" % src_type)

        if self.simultaneous:
            # merge all sources into one shot
            self.source_fx_loc = source_fx_loc.reshape(
                (1, src_num_per_shot * src_num, 2)
            )
            self.source_fz_loc = source_fz_loc.reshape(
                (1, src_num_per_shot * src_num, 2)
            )
            self.source_fx_amp = source_fx_amp.reshape(
                (1, src_num_per_shot * src_num, nt)
            )
            self.source_fz_amp = source_fz_amp.reshape(
                (1, src_num_per_shot * src_num, nt)
            )

        elif self.reciprocity and self.recip_type == "pr-das":
            # reciprocal DAS: 4 sources per channel form one shot
            self.source_fx_loc = source_fx_loc.reshape((-1, src_num_per_shot * 4, 2))
            self.source_fz_loc = source_fz_loc.reshape((-1, src_num_per_shot * 4, 2))
            self.source_fx_amp = source_fx_amp.reshape((-1, src_num_per_shot * 4, nt))
            self.source_fz_amp = source_fz_amp.reshape((-1, src_num_per_shot * 4, nt))

        else:
            self.source_fx_loc = source_fx_loc
            self.source_fz_loc = source_fz_loc
            self.source_fx_amp = source_fx_amp
            self.source_fz_amp = source_fz_amp

    def __set_receiver(self) -> None:
        """Build grid-level receiver locations for every shot.

        DAS channel end points are appended to both the vx and vz receiver
        lists. Sets the per-shot receiver counts and ``receiver_pr_loc``,
        ``receiver_vx_loc``, ``receiver_vz_loc`` of shape (nshots, nrec, 2).
        """
        receiver_pr_loc = self.receiver.get_loc("pr")
        receiver_vx_loc = self.receiver.get_loc("vx")
        receiver_vz_loc = self.receiver.get_loc("vz")
        receiver_das_loc = self.receiver.get_loc("das")

        n_receivers_pr_per_shot = receiver_pr_loc.shape[0]
        n_receivers_vx_per_shot = receiver_vx_loc.shape[0]
        n_receivers_vz_per_shot = receiver_vz_loc.shape[0]
        n_receivers_das_per_shot = receiver_das_loc.shape[0]

        # DAS requires both vx and vz at the channel end points
        receiver_vx_loc = np.concatenate((receiver_vx_loc, receiver_das_loc), axis=0)
        receiver_vz_loc = np.concatenate((receiver_vz_loc, receiver_das_loc), axis=0)

        # number of shots actually simulated
        if self.simultaneous:
            src_num = 1

        elif self.reciprocity and self.recip_type == "pr-das":
            # reciprocal DAS: 4 sources per channel form one shot
            src_num = self.source.num // 4

        else:
            src_num = self.source.num

        self.n_receivers_pr_per_shot = n_receivers_pr_per_shot
        self.n_receivers_vx_per_shot = n_receivers_vx_per_shot
        self.n_receivers_vz_per_shot = n_receivers_vz_per_shot
        self.n_receivers_das_per_shot = n_receivers_das_per_shot
        self.receiver_pr_loc = np.tile(receiver_pr_loc, (src_num, 1, 1))
        self.receiver_vx_loc = np.tile(receiver_vx_loc, (src_num, 1, 1))
        self.receiver_vz_loc = np.tile(receiver_vz_loc, (src_num, 1, 1))


    def record_data(self, rec_pr: torch.Tensor,
                    rec_vz: torch.Tensor,
                    rec_vx: torch.Tensor) -> dict:
        """Assemble recorded wavefield samples into data by component.

        Without reciprocity, pressure is sign-flipped, vx/vz are taken
        directly, and DAS is computed from the appended vx/vz samples via the
        cable operator. With reciprocity, data are formed from differences of
        the paired dipole receivers.

        Parameters
        ----------
        rec_pr : torch.Tensor
            Pressure samples, shape (nshots, nrec_pr, nt).
        rec_vz : torch.Tensor
            Vz samples, shape (nshots, nrec_vz, nt).
        rec_vx : torch.Tensor
            Vx samples, shape (nshots, nrec_vx, nt).

        Returns
        -------
        dict
            Data keyed by component ('pr', 'vx', 'vz', 'das'), each of shape
            (nshots, nrec, nt).

        Raises
        ------
        ValueError
            If the reciprocity type is unsupported.
        """

        data = {}


        if not self.reciprocity:
            if self.n_receivers_pr_per_shot > 0:
                data["pr"] = -1.0 * rec_pr

            if self.n_receivers_vx_per_shot > 0:
                data["vx"] = rec_vx[:, : self.n_receivers_vx_per_shot, :]

            if self.n_receivers_vz_per_shot > 0:
                data["vz"] = rec_vz[:, : self.n_receivers_vz_per_shot, :]

            # DAS samples are appended after the vx/vz receivers
            if self.n_receivers_das_per_shot:
                das_vx = rec_vx[:, -self.n_receivers_das_per_shot :, :]
                das_vz = rec_vz[:, -self.n_receivers_das_per_shot :, :]
                data["das"] = self.cable_operator.forward(das_vx, das_vz)

        else:
            if self.recip_type == "pr-vx":
                indices1 = np.arange(0, self.n_receivers_vx_per_shot, 2)
                indices2 = np.arange(1, self.n_receivers_vx_per_shot, 2)
                data["vx"] = (
                    rec_vx[:, indices2, :]
                    - rec_vx[:, indices1, :]
                    + rec_vz[:, indices2, :]
                    - rec_vz[:, indices1, :]
                )

            elif self.recip_type == "pr-vz":
                indices1 = np.arange(0, self.n_receivers_vz_per_shot, 2)
                indices2 = np.arange(1, self.n_receivers_vz_per_shot, 2)
                data["vz"] = (
                    rec_vx[:, indices2, :]
                    - rec_vx[:, indices1, :]
                    + rec_vz[:, indices2, :]
                    - rec_vz[:, indices1, :]
                )

            elif self.recip_type == "pr-vx-vz":
                # vx sources come first, then vz sources
                nshots_vx = self.source.get_loc(type="vx").shape[0]
                indices1 = np.arange(0, self.n_receivers_vx_per_shot, 2)
                indices2 = np.arange(1, self.n_receivers_vx_per_shot, 2)
                data["vx"] = (
                    rec_vx[:nshots_vx, indices2, :]
                    - rec_vx[:nshots_vx, indices1, :]
                    + rec_vz[:nshots_vx, indices2, :]
                    - rec_vz[:nshots_vx, indices1, :]
                )
                data["vz"] = (
                    rec_vx[nshots_vx:, indices2, :]
                    - rec_vx[nshots_vx:, indices1, :]
                    + rec_vz[nshots_vx:, indices2, :]
                    - rec_vz[nshots_vx:, indices1, :]
                )

            elif self.recip_type == "pr-das":
                indices1 = np.arange(0, self.n_receivers_vx_per_shot, 2)
                indices2 = np.arange(1, self.n_receivers_vx_per_shot, 2)
                data["das"] = (
                    rec_vx[:, indices2, :]
                    - rec_vx[:, indices1, :]
                    + rec_vz[:, indices2, :]
                    - rec_vz[:, indices1, :]
                )

            else:
                raise ValueError(
                    "Reciprocity Error: unsupported receiver types:", self.reciq_type
                )

        return data


    def plot(self, figsize=(8, 8)) -> None:
        """Plot source and receiver locations.

        Parameters
        ----------
        figsize : tuple, optional
            Accepted for API compatibility; the figure is drawn at (6, 6).
        """
        src_type = ["Src-Pr", "Src-Vx", "Src-Vz", "Src-MT"]
        src_pr = self.source.get_loc(type="pr")
        src_vx = self.source.get_loc(type="vx")
        src_vz = self.source.get_loc(type="vz")
        src_mt = self.source.get_loc(type="moment_tensor")

        rec_type = ["Rec-Pr", "Rec-Vx", "Rec-Vz", "Rec-DAS"]
        rec_pr = self.receiver.get_loc(type="pr")
        rec_vx = self.receiver.get_loc(type="vx")
        rec_vz = self.receiver.get_loc(type="vz")
        rec_das = self.receiver.get_loc(type="das")

        fontsize = 14
        plt.rcParams.update(
            {
                "axes.labelsize": fontsize,
                "xtick.labelsize": fontsize,
                "ytick.labelsize": fontsize,
                "legend.fontsize": fontsize,
                "figure.titlesize": fontsize,
            }
        )

        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111)

        for type, loc in zip(rec_type, [rec_pr, rec_vx, rec_vz, rec_das]):
            if loc.shape[0] > 0:
                ax.scatter(loc[:, 0], loc[:, 1], marker="^", s=20, label=type)

        for type, loc in zip(src_type, [src_pr, src_vx, src_vz, src_mt]):
            if loc.shape[0] > 0:
                ax.scatter(loc[:, 0], loc[:, 1], marker="*", s=30, label=type)

        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Depth (m)")
        ax.set_title("Seismic Survey (2D)")
        ax.invert_yaxis()
        ax.grid(linewidth=1.0, color="gray", alpha=0.3)
        ax.legend(
            loc="lower left", bbox_to_anchor=(1.05, 0.0), ncol=1, borderaxespad=0.0
        )
        plt.show()
