"""Interpolation of off-grid source and receiver locations onto the grid.

Implements the Kaiser-windowed sinc method of Hicks (2002), "Arbitrary source
and receiver positioning in finite-difference schemes using Kaiser windowed
sinc functions", Geophysics, 67(1), 156-165. Adapted from Deepwave:
https://github.com/ar4/deepwave/blob/master/src/deepwave/location_interpolation.py
"""


import math
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

DEFAULT_EPS = 1e-5

def _get_hicks_for_one_location_dim(
        hicks_weight_cache: Dict[Tuple[int, int, int], Tensor],
        location: float,
        halfwidth: int,
        beta: Tensor,
        free_surface: List[bool],
        size: int,
        monopole: bool = True,
        eps: float = DEFAULT_EPS) -> Tuple[Tensor, Tensor]:
    """Compute Hicks grid indices and weights along one dimension.

    Parameters
    ----------
    hicks_weight_cache : dict
        Cache of weights keyed by (fractional offset, halfwidth, monopole).
    location : float
        Location along this dimension, in grid cells.
    halfwidth : int
        Half-width of the interpolation window, in grid cells.
    beta : Tensor
        Kaiser window shape parameter.
    free_surface : list of bool
        Whether the [beginning, end] of this dimension is a free surface.
        Weights beyond a free surface are mirrored back with opposite sign.
    size : int
        Number of grid cells along this dimension.
    monopole : bool, optional
        If True, use a monopole (sinc) kernel; otherwise use a dipole
        (derivative of sinc) kernel. Default True.
    eps : float, optional
        Tolerance for treating a location as lying on a grid point.

    Returns
    -------
    locations : Tensor
        Integer grid indices covered by the window.
    weights : Tensor
        Interpolation weight for each index.
    """
    if monopole and abs(location - round(location)) < eps:
        locations = torch.tensor([location]).round().long().to(beta.device)
        weights = torch.ones(1, dtype=beta.dtype, device=beta.device)
    else:
        key = (int((location - int(location)) / eps), halfwidth, int(monopole))
        x = (torch.arange(-halfwidth + 1,
                          halfwidth + 1,
                          dtype=beta.dtype,
                          device=beta.device) - location + int(location))
        locations = (location + x).long()

        if key in hicks_weight_cache:
            weights = hicks_weight_cache[key]
        else:
            if monopole:
                weights = (torch.sinc(x) *
                           torch.i0(beta * (1 - (x / halfwidth)**2).sqrt()) /
                           torch.i0(beta))
            else:
                weights = ((torch.cos(math.pi * x) - torch.sinc(x)) /
                           (x**2 + eps) * x *
                           torch.i0(beta * (1 - (x / halfwidth)**2).sqrt()) /
                           torch.i0(beta))
            hicks_weight_cache[key] = weights

        # Mirror weights across a free surface at the start of the axis
        if free_surface[0] and locations[0].item() < 0:
            idx0 = int(-locations[0].item())
            locations = locations[idx0:]
            flipped_weights = weights[:idx0 - 1].flip(0)
            weights[idx0:idx0 + len(flipped_weights)] -= flipped_weights
            weights = weights[idx0:]

        # Mirror weights across a free surface at the end of the axis
        if free_surface[1] and locations[-1].item() >= size:
            idxe = size - int(locations[-1].item()) - 1
            locations = locations[:idxe]
            flipped_weights = weights[idxe + 1:].flip(0)
            weights[idxe - len(flipped_weights):idxe] -= flipped_weights
            weights = weights[:idxe]

    return locations, weights


def _get_hicks_locations_and_weights(
    locations: Tensor,
    halfwidth: int,
    beta: Tensor,
    free_surfaces: List[bool],
    model_shape: List[int],
    monopole: Union[Tensor, bool] = True,
    dipole_dim: Union[Tensor, int] = 0,
    eps: float = DEFAULT_EPS
) -> Tuple[Tensor, List[List[List[int]]], List[List[List[Tensor]]]]:
    """Compute Hicks grid locations and weights for all points.

    Parameters
    ----------
    locations : Tensor
        Point locations [shot, per_shot, 2], in grid cells.
    halfwidth : int
        Half-width of the interpolation window, in grid cells.
    beta : Tensor
        Kaiser window shape parameter.
    free_surfaces : list of bool
        Free-surface flags for the four grid edges (see `Hicks`).
    model_shape : list of int
        Grid size [n0, n1].
    monopole : bool or Tensor, optional
        Monopole flag, scalar or per point [shot, per_shot]. Default True.
    dipole_dim : int or Tensor, optional
        Dipole orientation (0 or 1), scalar or per point. Default 0.
    eps : float, optional
        Tolerance for treating a location as lying on a grid point.

    Returns
    -------
    hicks_locations : Tensor
        Unique grid locations per shot [shot, n_per_shot_hicks, 2] (long).
        Unused entries are filled with 1.
    hicks_idxs : list
        For each shot and point, indices into `hicks_locations` of the
        grid cells that the point is spread over.
    weights : list
        For each shot and point, the pair [weights0, weights1] of 1D
        weights along each dimension.
    """

    hicks_weight_cache: Dict[Tuple[int, int, int], Tensor] = {}
    n_shots, n_per_shot, _ = locations.shape
    hicks_locations_list: List[List[Tuple[int, int]]] = []
    hicks_idxs: List[List[List[int]]] = []
    weights: List[List[List[Tensor]]] = []
    n_per_shot_hicks = 0

    for shotidx in range(n_shots):
        shot_location_idxs: List[List[int]] = []
        shot_weights: List[List[Tensor]] = []
        n_hicks_locations = 0
        locations_dict: Dict[Tuple[int, int], int] = {}

        for i in range(n_per_shot):
            if isinstance(monopole, Tensor):
                monopole_i = bool(monopole[shotidx, i].item())
            else:
                monopole_i = monopole
            if isinstance(dipole_dim, Tensor):
                dipole_dim_i = int(dipole_dim[shotidx, i].item())
            else:
                dipole_dim_i = dipole_dim
            locations0, weights0 = \
                _get_hicks_for_one_location_dim(
                    hicks_weight_cache,
                    float(locations[shotidx, i, 0].item()), halfwidth, beta,
                    free_surfaces[:2], model_shape[0],
                    monopole_i or (not monopole_i and dipole_dim_i == 1),
                    eps=eps
                )
            locations1, weights1 = \
                _get_hicks_for_one_location_dim(
                    hicks_weight_cache,
                    float(locations[shotidx, i, 1].item()), halfwidth, beta,
                    free_surfaces[2:], model_shape[1],
                    monopole_i or (not monopole_i and dipole_dim_i == 0),
                    eps=eps
                )
            shot_weights.append([weights0, weights1])
            # Assign each grid cell a unique index within the shot
            i_idxs: List[int] = []
            for loc in torch.cartesian_prod(locations0, locations1):
                loc_tuple = (loc[0].item(), loc[1].item())
                if loc_tuple in locations_dict:
                    i_idxs.append(locations_dict[loc_tuple])
                else:
                    locations_dict[loc_tuple] = n_hicks_locations
                    i_idxs.append(n_hicks_locations)
                    n_hicks_locations += 1
            shot_location_idxs.append(i_idxs)
        hicks_idxs.append(shot_location_idxs)
        weights.append(shot_weights)
        n_per_shot_hicks = max(n_per_shot_hicks, n_hicks_locations)
        hicks_locations_list.append(list(locations_dict.keys()))

    # Fill unused entries with 1 rather than 0 so that padded locations of
    # empty source wavelets do not sit at the grid origin
    hicks_locations = torch.zeros(n_shots,
                                  n_per_shot_hicks,
                                  2,
                                  dtype=torch.long,
                                  device=locations.device) + 1

    for shotidx in range(n_shots):
        for i, loc in enumerate(hicks_locations_list[shotidx]):
            hicks_locations[shotidx,i] = (torch.tensor(loc).to(locations.device))

    return hicks_locations, hicks_idxs, weights


def _check_shot_idxs(amplitudes: Tensor,
                     shot_idxs: Optional[Tensor] = None) -> None:
    """Raise if `shot_idxs` does not match the number of shots in `amplitudes`."""
    if (shot_idxs is not None and shot_idxs.shape != (len(amplitudes), )):
        raise RuntimeError("shot_idxs must have the same length "
                           "as amplitudes")


class Hicks:
    """Interpolate off-grid points onto the grid using the Hicks method.

    Each point is spread over a (2 * halfwidth)^2 neighbourhood of grid
    cells with Kaiser-windowed sinc weights (Hicks, 2002,
    https://doi.org/10.1190/1.1451454). Dipole sources and receivers are
    also supported.

    Parameters
    ----------
    locations : Tensor
        Point locations [shot, per_shot, 2] as float values in grid cells
        relative to the grid origin, e.g. [[[0.5, 0.5]]] is half a cell from
        the origin in both dimensions.
    halfwidth : int, optional
        Half-width of the interpolation window, in [1, 10]. Default 4
        (an 8x8 window).
    free_surfaces : list of bool, optional
        Whether each grid edge is a free surface, in the order [start of
        dim 0, end of dim 0, start of dim 1, end of dim 1]. Default: no
        free surfaces.
    model_shape : list of int, optional
        Grid size [n0, n1]. Required only when there are free surfaces.
    monopole : bool or Tensor, optional
        Whether each point is a monopole (True) or dipole (False); scalar
        or Tensor of shape [shot, per_shot]. Default True.
    dipole_dim : int or Tensor, optional
        Dimension (0 or 1) along which dipoles are oriented; scalar or
        Tensor of shape [shot, per_shot]. Ignored for monopoles. Default 0.
    dtype : torch.dtype, optional
        Data type of the weights. Default torch.float.
    eps : float, optional
        Small value preventing division by zero; points closer than this to
        a grid point are snapped to it. Default 1e-5.

    Raises
    ------
    RuntimeError
        If `locations` is not 3D, `halfwidth` is not an integer in [1, 10],
        free surfaces are given without `model_shape`, or `monopole` /
        `dipole_dim` Tensors do not have shape [shot, per_shot].
    """

    def __init__(self,
                 locations: Tensor,
                 halfwidth: int = 4,
                 free_surfaces: Optional[List[bool]] = None,
                 model_shape: Optional[List[int]] = None,
                 monopole: Union[Tensor, bool] = True,
                 dipole_dim: Union[Tensor, int] = 0,
                 dtype: torch.dtype = torch.float,
                 eps: float = DEFAULT_EPS):

        if locations.ndim != 3:
            raise RuntimeError("Locations should have three dimensions")
        if not isinstance(halfwidth, int):
            raise RuntimeError("halfwidth must be an integer")
        if halfwidth < 1 or halfwidth > 10:
            raise RuntimeError("halfwidth must be in [1, 10]")
        if free_surfaces is None:
            free_surfaces = [False, False, False, False]
        if any(free_surfaces) and model_shape is None:
            raise RuntimeError("If there are free surfaces then model_shape "
                               "must be specified")
        if model_shape is None:
            model_shape = [-1, -1]
        if (isinstance(monopole, Tensor)
                and (monopole.shape[0] != locations.shape[0]
                     or monopole.shape[1] != locations.shape[1])):
            raise RuntimeError("monopole must have dimensions "
                               "[shot, per_shot]")
        if (isinstance(dipole_dim, Tensor)
                and (dipole_dim.shape[0] != locations.shape[0]
                     or dipole_dim.shape[1] != locations.shape[1])):
            raise RuntimeError("dipole_dim must have dimensions "
                               "[shot, per_shot]")
        # Kaiser window beta for each halfwidth (Hicks, 2002)
        betas = [0.0, 1.84, 3.04, 4.14, 5.26, 6.40, 7.51, 8.56, 9.56, 10.64]
        beta = (torch.tensor(betas[halfwidth - 1]).to(dtype).to(
            locations.device))
        self.locations = locations
        self.hicks_locations, self.idxs, self.weights = \
            _get_hicks_locations_and_weights(
                locations, halfwidth, beta, free_surfaces, model_shape,
                monopole, dipole_dim, eps
            )

    def get_locations(self, shot_idxs: Optional[Tensor] = None) -> Tensor:
        """Return the interpolated grid locations.

        These can be passed to a Deepwave propagator as source or receiver
        locations.

        Parameters
        ----------
        shot_idxs : Tensor, optional
            1D indices of the shots to return, relative to the locations
            given at initialisation (useful for shot batching). Default
            None returns all shots.

        Returns
        -------
        Tensor
            Integer grid locations [shot, n_per_shot_hicks, 2].
        """
        if shot_idxs is not None:
            return self.hicks_locations[shot_idxs]
        return self.hicks_locations

    def source(self,
               amplitudes: Tensor,
               shot_idxs: Optional[Tensor] = None) -> Tensor:
        """Spread source amplitudes onto the interpolated locations.

        Parameters
        ----------
        amplitudes : Tensor
            Source amplitudes at the original locations [shot, per_shot, nt].
        shot_idxs : Tensor, optional
            1D indices of the shots in `amplitudes`, relative to the
            locations given at initialisation. Default None means all shots.

        Returns
        -------
        Tensor
            Amplitudes at the interpolated locations
            [shot, n_per_shot_hicks, nt], on the device of `amplitudes`.

        Raises
        ------
        RuntimeError
            If `shot_idxs` does not have the same length as `amplitudes`.
        """

        _check_shot_idxs(amplitudes, shot_idxs)
        n_shots, n_per_shot, nt = amplitudes.shape
        n_per_shot_hicks = self.hicks_locations.shape[1]
        # Accumulate on the CPU to work around an apparent PyTorch issue
        # that otherwise gives incorrect results
        out = torch.zeros(n_shots,
                          n_per_shot_hicks,
                          nt,
                          dtype=amplitudes.dtype,
                          device=torch.device('cpu'))
        for shotidx in range(n_shots):
            if shot_idxs is not None:
                hicks_shotidx = int(shot_idxs[shotidx])
            else:
                hicks_shotidx = shotidx
            for i in range(n_per_shot):
                out[shotidx, self.idxs[hicks_shotidx][i], :] += (
                    amplitudes[shotidx, i][None] *
                    (self.weights[hicks_shotidx][i][0].reshape(-1, 1) *
                     self.weights[hicks_shotidx][i][1].reshape(
                         1, -1)).reshape(-1)[..., None]).cpu()
        return out.to(amplitudes.device)

    def receiver(self,
                 amplitudes: Tensor,
                 shot_idxs: Optional[Tensor] = None) -> Tensor:
        """Gather recorded amplitudes back to the original receiver locations.

        Parameters
        ----------
        amplitudes : Tensor
            Amplitudes recorded at the interpolated locations
            [shot, n_per_shot_hicks, nt].
        shot_idxs : Tensor, optional
            1D indices of the shots in `amplitudes`, relative to the
            locations given at initialisation. Default None means all shots.

        Returns
        -------
        Tensor
            Weighted sum of amplitudes at the original receiver locations
            [shot, per_shot, nt].

        Raises
        ------
        RuntimeError
            If `shot_idxs` does not have the same length as `amplitudes`.
        """
        _check_shot_idxs(amplitudes, shot_idxs)
        n_shots, _, nt = amplitudes.shape
        n_per_shot = self.locations.shape[1]
        out = torch.zeros(n_shots,
                          n_per_shot,
                          nt,
                          dtype=amplitudes.dtype,
                          device=amplitudes.device)
        for shotidx in range(n_shots):
            if shot_idxs is not None:
                hicks_shotidx = int(shot_idxs[shotidx])
            else:
                hicks_shotidx = shotidx
            for i in range(n_per_shot):
                out[shotidx, i, :] = (
                    amplitudes[shotidx, self.idxs[hicks_shotidx][i]] *
                    (self.weights[hicks_shotidx][i][0].reshape(-1, 1) *
                     self.weights[hicks_shotidx][i][1].reshape(
                         1, -1)).reshape(-1)[..., None]).sum(dim=0)
        return out
