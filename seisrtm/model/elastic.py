"""Isotropic elastic model parameterized by P-wave velocity, S-wave velocity
and density.
"""

from typing import Optional, Tuple, Union
import numpy as np
import torch
from torch import Tensor
from seisrtm.model import AbstractModel


eps = 1e-7

class ElasticModel(AbstractModel):
    """Isotropic elastic model with vp, vs and rho as trainable parameters.

    Parameters
    ----------
    ox : float
        Origin of the model in the x-direction (m).
    oz : float
        Origin of the model in the z-direction (m).
    dx : float
        Grid spacing in the x-direction (m).
    dz : float
        Grid spacing in the z-direction (m). Must equal ``dx``.
    nx : int
        Number of grid points in the x-direction.
    nz : int
        Number of grid points in the z-direction.
    vp : np.ndarray or torch.Tensor
        P-wave velocity (m/s) with shape (nz, nx).
    vs : np.ndarray or torch.Tensor
        S-wave velocity (m/s) with shape (nz, nx).
    rho : np.ndarray or torch.Tensor
        Density (kg/m^3) with shape (nz, nx).
    vp_bound : tuple of float, optional
        ``(min, max)`` bounds on vp (m/s). Default is None (unbounded).
    vs_bound : tuple of float, optional
        ``(min, max)`` bounds on vs (m/s). Default is None (unbounded).
    rho_bound : tuple of float, optional
        ``(min, max)`` bounds on rho (kg/m^3). Default is None (unbounded).
    vp_grad : bool, optional
        Whether vp requires a gradient. Default is False.
    vs_grad : bool, optional
        Whether vs requires a gradient. Default is False.
    rho_grad : bool, optional
        Whether rho requires a gradient. Default is False.
    free_surface : bool, optional
        Whether the top boundary is a free surface. Default is False.
    nabc : int, optional
        Number of absorbing boundary cells. Default is 20.
    mask : np.ndarray, optional
        Gradient mask with shape (nz, nx). Accepted but not stored by this
        class. Default is None.
    smooth_size : optional
        Unused.

    Raises
    ------
    AssertionError
        If ``dx != dz``, a lower bound is not below its upper bound, or a
        parameter does not have shape (nz, nx).
    """

    def __init__(
        self,
        ox: float,
        oz: float,
        dx: float,
        dz: float,
        nx: int,
        nz: int,
        vp: Optional[Union[np.ndarray, Tensor]] = None,
        vs: Optional[Union[np.ndarray, Tensor]] = None,
        rho: Optional[Union[np.ndarray, Tensor]] = None,
        vp_bound: Optional[Tuple[float, float]] = None,
        vs_bound: Optional[Tuple[float, float]] = None,
        rho_bound: Optional[Tuple[float, float]] = None,
        vp_grad: Optional[bool] = False,
        vs_grad: Optional[bool] = False,
        rho_grad: Optional[bool] = False,
        free_surface: Optional[bool] = False,
        nabc: Optional[int] = 20,
        mask: Optional[np.ndarray] = None,
        smooth_size=None,
        ) -> None:

        super().__init__(ox, oz, dx, dz, nx, nz, free_surface, nabc)

        self.pars = ["vp", "vs", "rho"]

        # copy the inputs and convert numpy arrays to torch tensors
        vp = vp.copy()
        vs = vs.copy()
        rho = rho.copy()
        vp = torch.tensor(vp) if isinstance(vp, np.ndarray) else vp
        vs = torch.tensor(vs) if isinstance(vs, np.ndarray) else vs
        rho = torch.tensor(rho) if isinstance(rho, np.ndarray) else rho

        self.vp = torch.nn.Parameter(vp.to(torch.float32), requires_grad=vp_grad)
        self.vs = torch.nn.Parameter(vs.to(torch.float32), requires_grad=vs_grad)
        self.rho = torch.nn.Parameter(rho.to(torch.float32), requires_grad=rho_grad)

        self.lower_bound["vp"] = vp_bound[0] if vp_bound is not None else None
        self.lower_bound["vs"] = vs_bound[0] if vs_bound is not None else None
        self.lower_bound["rho"] = rho_bound[0] if rho_bound is not None else None
        self.upper_bound["vp"] = vp_bound[1] if vp_bound is not None else None
        self.upper_bound["vs"] = vs_bound[1] if vs_bound is not None else None
        self.upper_bound["rho"] = rho_bound[1] if rho_bound is not None else None

        self.requires_grad["vp"] = vp_grad
        self.requires_grad["vs"] = vs_grad
        self.requires_grad["rho"] = rho_grad

        self.check_bounds()
        self.check_dims()


    def forward(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Convert (vp, vs, rho) to the elastic parameters used by the propagator.

        The parameters are first clamped in place to their bounds, then

        ``lam = (vp^2 - 2 vs^2) rho``, ``mu = vs^2 rho``, ``buoyancy = 1 / rho``.

        Returns
        -------
        lam : torch.Tensor
            Lame parameter lambda (Pa) with shape (nz, nx).
        mu : torch.Tensor
            Shear modulus (Pa) with shape (nz, nx).
        buoyancy : torch.Tensor
            Buoyancy (m^3/kg) with shape (nz, nx).

        Raises
        ------
        ValueError
            If a bounded parameter contains inf, negative values or NaN after
            clamping.
        """
        self.clip_params()

        vp  = self.constrain_range(self.vp, self.lower_bound["vp"], self.upper_bound["vp"])
        vs  = self.constrain_range(self.vs, self.lower_bound["vs"], self.upper_bound["vs"])
        rho = self.constrain_range(self.rho, self.lower_bound["rho"], self.upper_bound["rho"])

        lam = (vp**2 - 2 * vs**2) * rho
        mu = vs**2 * rho
        # rho / (rho^2 + eps) keeps 1/rho stable near zero density
        buoyancy = 1 / (rho**2 + eps) * rho

        return lam, mu, buoyancy
