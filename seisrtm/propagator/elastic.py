"""Isotropic elastic wave-equation propagator built on Deepwave.

Solves the 2D stress-velocity elastic equations by finite differences, with
optional Hicks interpolation of off-grid sources and receivers.
"""

from typing import Optional

import numpy as np
import torch
import deepwave
from deepwave import elastic
from deepwave.common import downsample
from seisrtm.model import AbstractModel
from seisrtm.survey import Survey, SeismicData
from .hicks import Hicks


class ElasticPropagator(torch.nn.Module):
    """Finite-difference propagator for the isotropic elastic wave equation.

    Sets up source and receiver grid locations from the survey (with Hicks
    interpolation if ``survey.interpolation`` is True) and the absorbing
    boundaries from the model. Sources are body forces in x and z; receivers
    record pressure, vx and vz.

    Parameters
    ----------
    model : AbstractModel
        Elastic model providing the grid, origin, PML width, free-surface
        flag and, when called, the (lambda, mu, buoyancy) tensors.
    survey : Survey
        Acquisition geometry, source wavelets and time sampling. It is
        initialised on the model grid by this constructor.

    Raises
    ------
    ValueError
        If `model` is not an AbstractModel or `survey` is not a Survey.
    RuntimeError
        If a free surface is combined with ``survey.reciprocity``.

    Notes
    -----
    Physical locations (m) are converted to grid indices by dividing by
    ``dx`` in both directions, so a square grid (dx == dz) is assumed.
    """

    def __init__(self, model: AbstractModel, survey: Survey):

        super().__init__()

        if not isinstance(model, AbstractModel):
            raise ValueError("model is not AbstractModel")

        if not isinstance(survey, Survey):
            raise ValueError("survey is not Survey")

        # --- model and survey ---

        model.analyze_survey(survey)

        # must be called before any other survey method
        dx, dz = model.get_grid()
        survey.initialize(dx, dz)

        if model.get_free_surface() and survey.reciprocity:
            raise RuntimeError(
                "free surface and reciprocity is not supported!"
            )

        self.model = model
        self.survey = survey

        # --- grid and time parameters ---
        self.dtype = torch.float32
        self.device = survey.device
        self.ox, self.oz = model.get_origin()
        self.dx, self.dz = model.get_grid()
        self.nx, self.nz = model.get_shape()
        self.nt = survey.receiver.nt
        self.dt = survey.receiver.dt
        self.f0 = survey.source.f0

        # --- boundaries, ordered [top, bottom, left, right] ---
        npml = model.get_nabc()
        self.free_surface = model.get_free_surface()
        print(f"free surface: {self.free_surface}")
        self.pml_width = (
            [0, npml, npml, npml] if self.free_surface else [npml, npml, npml, npml]
        )
        free_surfaces = (
            [True, False, False, False]
            if self.free_surface
            else [False, False, False, False]
        )

        # --- sources ---

        # reorder to (z, x) and shift relative to the model origin
        source_fx_loc = survey.source_fx_loc[:, :, [1, 0]] - np.array(
            [self.ox, self.oz]
        )
        source_fz_loc = survey.source_fz_loc[:, :, [1, 0]] - np.array(
            [self.ox, self.oz]
        )

        if self.survey.interpolation:
            # fractional grid indices
            source_fx_ind = torch.tensor(
                source_fx_loc / self.dx, dtype=torch.float, device=self.device
            )
            source_fz_ind = torch.tensor(
                source_fz_loc / self.dx, dtype=torch.float, device=self.device
            )
            source_fx_amp = torch.tensor(
                survey.source_fx_amp, dtype=self.dtype, device=self.device
            )
            source_fz_amp = torch.tensor(
                survey.source_fz_amp, dtype=self.dtype, device=self.device
            )

            source_fx_hicks = Hicks(
                source_fx_ind,
                free_surfaces=free_surfaces,
                model_shape=[self.nz, self.nx],
            )
            source_fz_hicks = Hicks(
                source_fz_ind,
                free_surfaces=free_surfaces,
                model_shape=[self.nz, self.nx],
            )

            self.source_fx_ind = source_fx_hicks.get_locations()
            self.source_fz_ind = source_fz_hicks.get_locations()
            self.source_fx_amp = source_fx_hicks.source(source_fx_amp)
            self.source_fz_amp = source_fz_hicks.source(source_fz_amp)

        else:
            # integer grid indices
            source_fx_ind = torch.tensor(
                source_fx_loc / self.dx, dtype=torch.long, device=self.device
            )
            source_fz_ind = torch.tensor(
                source_fz_loc / self.dx, dtype=torch.long, device=self.device
            )
            source_fx_amp = torch.tensor(
                survey.source_fx_amp, dtype=self.dtype, device=self.device
            )
            source_fz_amp = torch.tensor(
                survey.source_fz_amp, dtype=self.dtype, device=self.device
            )

            self.source_fx_ind = source_fx_ind
            self.source_fz_ind = source_fz_ind
            self.source_fx_amp = source_fx_amp
            self.source_fz_amp = source_fz_amp

        # --- receivers ---

        # reorder to (z, x) and shift relative to the model origin
        receiver_pr_loc = survey.receiver_pr_loc[:, :, [1, 0]] - np.array(
            [self.ox, self.oz]
        )
        receiver_vx_loc = survey.receiver_vx_loc[:, :, [1, 0]] - np.array(
            [self.ox, self.oz]
        )
        receiver_vz_loc = survey.receiver_vz_loc[:, :, [1, 0]] - np.array(
            [self.ox, self.oz]
        )

        if self.survey.interpolation:
            # fractional grid indices
            receiver_pr_ind = torch.tensor(
                receiver_pr_loc / self.dx, dtype=torch.float, device=self.device
            )
            receiver_vx_ind = torch.tensor(
                receiver_vx_loc / self.dx, dtype=torch.float, device=self.device
            )
            receiver_vz_ind = torch.tensor(
                receiver_vz_loc / self.dx, dtype=torch.float, device=self.device
            )

            # kept for mapping recorded data back to the true locations
            self.receiver_pr_hicks = Hicks(
                receiver_pr_ind,
                free_surfaces=free_surfaces,
                model_shape=[self.nz, self.nx],
            )
            self.receiver_vx_hicks = Hicks(
                receiver_vx_ind,
                free_surfaces=free_surfaces,
                model_shape=[self.nz, self.nx],
            )
            self.receiver_vz_hicks = Hicks(
                receiver_vz_ind,
                free_surfaces=free_surfaces,
                model_shape=[self.nz, self.nx],
            )

            self.receiver_pr_ind = self.receiver_pr_hicks.get_locations()
            self.receiver_vx_ind = self.receiver_vx_hicks.get_locations()
            self.receiver_vz_ind = self.receiver_vz_hicks.get_locations()

        else:
            # integer grid indices
            receiver_pr_ind = torch.tensor(
                receiver_pr_loc / self.dx, dtype=torch.long, device=self.device
            )
            receiver_vx_ind = torch.tensor(
                receiver_vx_loc / self.dx, dtype=torch.long, device=self.device
            )
            receiver_vz_ind = torch.tensor(
                receiver_vz_loc / self.dx, dtype=torch.long, device=self.device
            )

            self.receiver_pr_ind = receiver_pr_ind
            self.receiver_vx_ind = receiver_vx_ind
            self.receiver_vz_ind = receiver_vz_ind

        # --- mark empty sources/receivers as None ---
        # source shape: (nsrc, nsrc_per_shot, 2)
        if self.source_fx_ind.shape[0] == 0:
            self.source_fx_ind = None
            self.source_fx_amp = None
        if self.source_fz_ind.shape[0] == 0:
            self.source_fz_ind = None
            self.source_fz_amp = None

        # receiver shape: (nsrc, nrec_per_shot, 2)
        if self.receiver_pr_ind.shape[1] == 0:
            self.receiver_pr_ind = None
        if self.receiver_vx_ind.shape[1] == 0:
            self.receiver_vx_ind = None
        if self.receiver_vz_ind.shape[1] == 0:
            self.receiver_vz_ind = None

    def forward(
        self,
        model: Optional[AbstractModel] = None,
        shot_indx = None,
        grad_interval = 1,
        data_mask = None,
        save_wavefield = False,
    ):
        """Model data d = F(m) for the configured sources and receivers.

        Parameters
        ----------
        model : AbstractModel, optional
            Model to propagate through. Default None uses the model given
            at initialisation.
        shot_indx : slice, optional
            Shots to propagate. Default None propagates all shots.
        grad_interval : int, optional
            Time-sample interval at which wavefields are stored for the
            model gradient (Deepwave ``model_gradient_sampling_interval``).
            Default 1.
        data_mask : torch.Tensor, optional
            Mask multiplied into every data component, indexed by
            `shot_indx`; broadcastable to (nshots, nrec, nt). Default None
            applies no mask.
        save_wavefield : bool, optional
            If True, also store vz and vx wavefield snapshots at every
            time sample. Default False.

        Returns
        -------
        SeismicData
            Modelled data. ``data.data`` holds one entry per receiver
            component present in the survey (``'pr'``, ``'vx'``, ``'vz'``,
            ``'das'``), each (nshots, nrec, nt). If `save_wavefield` is True
            it also holds ``'wavefields_vz'`` and ``'wavefields_vx'``, each
            (nt, nshots, nz, nx) with the absorbing boundaries removed.
        """

        if data_mask is not None:
            if data_mask.device != self.device:
                data_mask = data_mask.to(self.device)

        lamb, mu, buoyancy = self.model() if model is None else model()

        propagator = Elastic(
            lamb, mu, buoyancy, self.dx, self.dt, self.f0, self.pml_width,
            grad_interval=grad_interval,
            save_wavefield=save_wavefield,
        )

        n_shots = self.survey.source.num
        n_gpus = self.survey.gpu_num

        shot_indx = slice(0, n_shots) if shot_indx is None else shot_indx

        # split shots across GPUs with DataParallel
        if self.device == "cuda":
            propagator = torch.nn.DataParallel(propagator, device_ids=range(n_gpus)).to(
                self.device
            )

        # placeholder location for receiver types that are not present
        if self.survey.simultaneous:
            fake_receiver = torch.ones(
                (1, 1, 2), dtype=torch.long, device=self.device
            )

        else:
            fake_receiver = torch.ones(
                (n_shots, 1, 2), dtype=torch.long, device=self.device
            )

        out = propagator(
            self.source_fx_amp[shot_indx] if self.source_fx_amp is not None else None,
            self.source_fx_ind[shot_indx] if self.source_fx_ind is not None else None,
            self.source_fz_amp[shot_indx] if self.source_fz_amp is not None else None,
            self.source_fz_ind[shot_indx] if self.source_fz_ind is not None else None,
            self.receiver_vz_ind[shot_indx] if self.receiver_vz_ind is not None else fake_receiver[shot_indx],
            self.receiver_vx_ind[shot_indx] if self.receiver_vx_ind is not None else fake_receiver[shot_indx],
            self.receiver_pr_ind[shot_indx] if self.receiver_pr_ind is not None else fake_receiver[shot_indx],
            self.device,
        )

        if not save_wavefield:
            rec_pr, rec_vz, rec_vx = out
            wavefields_vy = None
            wavefields_vx = None
        else:
            rec_pr, rec_vz, rec_vx, wavefields_vy, wavefields_vx = out

        # map interpolated receivers back to their true locations
        if self.survey.interpolation:
            if self.receiver_pr_ind is not None:
                rec_pr = self.receiver_pr_hicks.receiver(rec_pr)
            if self.receiver_vx_ind is not None:
                rec_vx = self.receiver_vx_hicks.receiver(rec_vx)
            if self.receiver_vz_ind is not None:
                rec_vz = self.receiver_vz_hicks.receiver(rec_vz)

        data_dict = self.survey.record_data(rec_pr, rec_vz, rec_vx)

        if data_mask is not None:
            for key in data_dict:
                data_dict[key] *= data_mask[shot_indx]

        data = SeismicData(self.survey)

        data.record_data(data_dict)

        if save_wavefield:
            data.data["wavefields_vz"] = wavefields_vy
            data.data["wavefields_vx"] = wavefields_vx

        return data


class Elastic(torch.nn.Module):
    """Thin ``torch.nn.Module`` wrapper around ``deepwave.elastic``.

    Wrapping the call in a module allows shots to be split across GPUs
    with ``torch.nn.DataParallel``. Uses 4th-order spatial accuracy.

    Parameters
    ----------
    lamb : torch.Tensor
        First Lamé parameter (nz, nx), in Pa.
    mu : torch.Tensor
        Shear modulus (nz, nx), in Pa.
    buoyancy : torch.Tensor
        Buoyancy, 1 / density (nz, nx), in m^3/kg.
    dx : float
        Grid spacing, in m (used for both directions).
    dt : float
        Time sampling of sources and receivers, in s.
    f0 : float
        Dominant frequency used to tune the PML, in Hz.
    pml_width : list of int
        PML width in cells, ordered [top, bottom, left, right].
    grad_interval : int, optional
        Time-sample interval for storing wavefields for the model gradient.
        Default 1.
    save_wavefield : bool, optional
        If True, `forward` also returns vz and vx wavefield snapshots.
        Default False.
    """

    def __init__(self, lamb, mu, buoyancy, dx, dt, f0, pml_width,
                 grad_interval=1, save_wavefield=False):
        super().__init__()
        self.lamb = lamb
        self.mu = mu
        self.buoyancy = buoyancy
        self.dx = dx
        self.dt = dt
        self.f0 = f0
        self.pml_width = pml_width
        self.accuracy = 4
        self.grad_interval = grad_interval
        self.freq_taper_frac = 0.1
        self.time_pad_frac = 0.1
        self.save_wavefield = save_wavefield

    def forward(
        self,
        source_fx_amp,
        source_fx_ind,
        source_fz_amp,
        source_fz_ind,
        receiver_vz_ind,
        receiver_vx_ind,
        receiver_pr_ind,
        device,
    ):
        """Run the elastic simulation.

        Parameters
        ----------
        source_fx_amp : torch.Tensor
            Horizontal force amplitudes (nshots, nsrc_per_shot, nt).
        source_fx_ind : torch.Tensor
            Horizontal force grid locations (nshots, nsrc_per_shot, 2),
            ordered (z, x).
        source_fz_amp : torch.Tensor
            Vertical force amplitudes (nshots, nsrc_per_shot, nt).
        source_fz_ind : torch.Tensor
            Vertical force grid locations (nshots, nsrc_per_shot, 2).
        receiver_vz_ind : torch.Tensor
            vz receiver grid locations (nshots, nrec_per_shot, 2).
        receiver_vx_ind : torch.Tensor
            vx receiver grid locations (nshots, nrec_per_shot, 2).
        receiver_pr_ind : torch.Tensor
            Pressure receiver grid locations (nshots, nrec_per_shot, 2).
        device : str or torch.device
            Device on which to run.

        Returns
        -------
        rec_pr, rec_vz, rec_vx : torch.Tensor
            Recorded pressure, vz and vx, each (nshots, nrec_per_shot, nt).
        wavefields_vy, wavefields_vx : torch.Tensor
            Only if ``save_wavefield`` is True: vz and vx snapshots at each
            source time sample, (nt, nshots, nz, nx), without the PML.
        """

        if not self.save_wavefield:
            rec_pr, rec_vz, rec_vx = elastic(
                self.lamb.to(device),
                self.mu.to(device),
                self.buoyancy.to(device),
                self.dx,
                self.dt,
                source_amplitudes_y=source_fz_amp.to(device),
                source_amplitudes_x=source_fx_amp.to(device),
                source_locations_y=source_fz_ind,
                source_locations_x=source_fx_ind,
                receiver_locations_y=receiver_vz_ind,
                receiver_locations_x=receiver_vx_ind,
                receiver_locations_p=receiver_pr_ind,
                accuracy=self.accuracy,
                pml_freq=self.f0,
                pml_width=self.pml_width,
                model_gradient_sampling_interval=self.grad_interval,
                freq_taper_frac=self.freq_taper_frac,
                time_pad_frac=self.time_pad_frac,
            )[-3:]

            return rec_pr, rec_vz, rec_vx

        else:
            # Propagate one source time sample at a time on the internal
            # (CFL-stable) time step, carrying the full wavefield state
            # between calls so that snapshots can be taken.

            dx = self.dx
            dt = self.dt
            vp = torch.sqrt((self.lamb + 2 * self.mu) * self.buoyancy)
            nx = vp.shape[-1]
            ny = vp.shape[-2]
            max_vel = torch.max(vp)
            pml_width = self.pml_width
            n_shots = source_fz_amp.shape[0]
            n_receivers_per_shot_pr = receiver_pr_ind.shape[1]
            n_receivers_per_shot_vz = receiver_vz_ind.shape[1]
            n_receivers_per_shot_vx = receiver_vx_ind.shape[1]

            # upsample the sources to the internal time step
            n_segments = source_fz_amp.shape[-1]
            dt_up, step_ratio = deepwave.common.cfl_condition(dx, dx, dt, max_vel)

            source_fx_amp_up = deepwave.common.upsample(source_fx_amp, step_ratio)
            source_fz_amp_up = deepwave.common.upsample(source_fz_amp, step_ratio)

            def wrap(chunk_fx, chunk_fy, vy, vx, sigmayy, sigmaxy, sigmaxx,
                     m_vyy, m_vyx, m_vxy, m_vxx, m_sigmayyy, m_sigmaxyy,
                     m_sigmaxyx, m_sigmaxxx):
                # propagate one chunk from the given initial wavefield state
                return  elastic(
                            self.lamb.to(device),
                            self.mu.to(device),
                            self.buoyancy.to(device),
                            self.dx,
                            dt_up,
                            source_amplitudes_y=chunk_fy,
                            source_amplitudes_x=chunk_fx,
                            source_locations_y=source_fz_ind,
                            source_locations_x=source_fx_ind,
                            receiver_locations_y=receiver_vz_ind,
                            receiver_locations_x=receiver_vx_ind,
                            receiver_locations_p=receiver_pr_ind,
                            accuracy=self.accuracy,
                            pml_freq=self.f0,
                            pml_width=self.pml_width,
                            model_gradient_sampling_interval=self.grad_interval,
                            freq_taper_frac=self.freq_taper_frac,
                            time_pad_frac=self.time_pad_frac,
                            vy_0 = vy,
                            vx_0 = vx,
                            sigmayy_0 = sigmayy,
                            sigmaxy_0 = sigmaxy,
                            sigmaxx_0 = sigmaxx,
                            m_vyy_0 = m_vyy,
                            m_vyx_0 = m_vyx,
                            m_vxy_0 = m_vxy,
                            m_vxx_0 = m_vxx,
                            m_sigmayyy_0 = m_sigmayyy,
                            m_sigmaxyy_0 = m_sigmaxyy,
                            m_sigmaxyx_0 = m_sigmaxyx,
                            m_sigmaxxx_0 = m_sigmaxxx,
                        )

            # wavefield state, including the PML region
            wavefield_size = [n_shots, ny + pml_width[0] + pml_width[1],
                        nx + pml_width[2] + pml_width[3]]

            vy = torch.zeros(*wavefield_size, device=device)
            vx = torch.zeros(*wavefield_size, device=device)
            sigmayy = torch.zeros(*wavefield_size, device=device)
            sigmaxy = torch.zeros(*wavefield_size, device=device)
            sigmaxx = torch.zeros(*wavefield_size, device=device)
            m_vyy = torch.zeros(*wavefield_size, device=device)
            m_vyx = torch.zeros(*wavefield_size, device=device)
            m_vxy = torch.zeros(*wavefield_size, device=device)
            m_vxx = torch.zeros(*wavefield_size, device=device)
            m_sigmayyy = torch.zeros(*wavefield_size, device=device)
            m_sigmaxyy = torch.zeros(*wavefield_size, device=device)
            m_sigmaxyx = torch.zeros(*wavefield_size, device=device)
            m_sigmaxxx = torch.zeros(*wavefield_size, device=device)

            wavefields_vy = torch.zeros(n_segments, n_shots, ny, nx, device=device)
            wavefields_vx = torch.zeros(n_segments, n_shots, ny, nx, device=device)

            # receiver data on the upsampled time axis
            rec_pr = torch.zeros(n_shots, n_receivers_per_shot_pr, n_segments * step_ratio,
                                    device=device)
            rec_vz = torch.zeros(n_shots, n_receivers_per_shot_vz, n_segments * step_ratio,
                                    device=device)
            rec_vx = torch.zeros(n_shots, n_receivers_per_shot_vx, n_segments * step_ratio,
                                    device=device)

            k = 0

            for i, chunk in enumerate(zip(
                torch.chunk(source_fx_amp_up, n_segments, dim=-1),
                torch.chunk(source_fz_amp_up, n_segments, dim=-1))):

                chunk_fx, chunk_fz = chunk

                # propagate up to the last time step in the chunk
                if chunk_fx.shape[-1] > 1:
                    (vy, vx, sigmayy, sigmaxy, sigmaxx,
                    m_vyy, m_vyx, m_vxy, m_vxx,
                    m_sigmayyy, m_sigmaxyy, m_sigmaxyx, m_sigmaxxx,
                    rec_pr_chunk, rec_vz_chunk, rec_vx_chunk) \
                        = wrap(chunk_fx[...,:-1], chunk_fz[...,:-1],
                                        vy, vx, sigmayy, sigmaxy, sigmaxx,
                                        m_vyy, m_vyx, m_vxy, m_vxx, m_sigmayyy, m_sigmaxyy,
                                        m_sigmaxyx, m_sigmaxxx)

                    wavefields_vy[i] = vy.detach()[:, pml_width[1]:-pml_width[0] or None, pml_width[2]:-pml_width[3]]
                    wavefields_vx[i] = vx.detach()[:, pml_width[1]:-pml_width[0] or None, pml_width[2]:-pml_width[3]]
                    rec_pr[..., k:k + chunk_fx.shape[-1] -1] = rec_pr_chunk
                    rec_vz[..., k:k + chunk_fx.shape[-1] -1] = rec_vz_chunk
                    rec_vx[..., k:k + chunk_fx.shape[-1] -1] = rec_vx_chunk

                # then propagate the last time step in the chunk
                (vy, vx, sigmayy, sigmaxy, sigmaxx,
                    m_vyy, m_vyx, m_vxy, m_vxx,
                    m_sigmayyy, m_sigmaxyy, m_sigmaxyx, m_sigmaxxx,
                    rec_pr_chunk, rec_vz_chunk, rec_vx_chunk) \
                        = wrap(chunk_fx[...,-1:], chunk_fz[...,-1:],
                                        vy, vx, sigmayy, sigmaxy, sigmaxx,
                                        m_vyy, m_vyx, m_vxy, m_vxx, m_sigmayyy, m_sigmaxyy,
                                        m_sigmaxyx, m_sigmaxxx)

                wavefields_vy[i] = vy.detach()[:, pml_width[1]:-pml_width[0] or None, pml_width[2]:-pml_width[3]]
                wavefields_vx[i] = vx.detach()[:, pml_width[1]:-pml_width[0] or None, pml_width[2]:-pml_width[3]]
                rec_pr[..., k + chunk_fx.shape[-1] - 1:k + chunk_fx.shape[-1]] = rec_pr_chunk
                rec_vz[..., k + chunk_fx.shape[-1] - 1:k + chunk_fx.shape[-1]] = rec_vz_chunk
                rec_vx[..., k + chunk_fx.shape[-1] - 1:k + chunk_fx.shape[-1]] = rec_vx_chunk
                k += chunk_fx.shape[-1]

            def average_adjacent(receiver_amplitudes):
                # average neighbouring time samples of receiver data
                if receiver_amplitudes.numel() == 0:
                    return receiver_amplitudes
                if receiver_amplitudes.device == torch.device('cpu'):
                    return (receiver_amplitudes[:, 1:] +
                            receiver_amplitudes[:, :-1]) / 2
                else:
                    return (receiver_amplitudes[1:] + receiver_amplitudes[:-1]) / 2

            # resample to the original time step, keeping all dimensions
            rec_pr = downsample(rec_pr, step_ratio, self.freq_taper_frac,
                                                   self.time_pad_frac, False)
            rec_vz = downsample(rec_vz, step_ratio, self.freq_taper_frac,
                                                   self.time_pad_frac, False)
            rec_vx = downsample(rec_vx, step_ratio, self.freq_taper_frac,
                                                    self.time_pad_frac, False)

            return rec_pr, rec_vz, rec_vx, wavefields_vy, wavefields_vx
