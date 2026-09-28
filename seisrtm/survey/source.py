"""Seismic source definition for 2-D elastic wave-equation modelling."""

import numpy as np
from typing import List, Optional


class Source(object):
    """Collection of seismic sources sharing one time axis.

    Parameters
    ----------
    nt : int
        Number of time samples of each source wavelet.
    dt : float
        Time sampling interval of the wavelets (s).
    f0 : float
        Dominant frequency of the wavelets (Hz). Used only to check numerical
        dispersion before running the propagator.

    Notes
    -----
    Wavelets are attached with :meth:`add_source`. Coordinates are 2-D,
    given as ``[x, z]`` in metres.
    """

    def __init__(self, nt: int, dt: float, f0: float) -> None:
        self.nt = nt
        self.dt = dt
        self.f0 = f0

        self.locs = []
        self.type = []
        self.wavelet = []
        self.moment_tensor = []
        self.num = 0

    def __add__(self, other):
        """Merge two source collections.

        Parameters
        ----------
        other : Source
            Source collection to append.

        Returns
        -------
        Source
            New collection containing the sources of both operands, with the
            ``nt``, ``dt`` and ``f0`` of ``self``.

        Raises
        ------
        TypeError
            If ``other`` is not a ``Source``.
        ValueError
            If ``nt`` or ``dt`` differ between the two collections.
        """

        if not isinstance(other, Source):
            raise TypeError(
                "Source Error: the other source must be an instance of Source"
            )

        if self.nt != other.nt or self.dt != other.dt:
            raise ValueError(
                "Source Error: the number of time samples and time interval must be the same"
            )

        new_source = Source(self.nt, self.dt, self.f0)
        new_source.locs = self.locs + other.locs
        new_source.type = self.type + other.type
        new_source.wavelet = self.wavelet + other.wavelet
        new_source.moment_tensor = self.moment_tensor + other.moment_tensor
        new_source.num = self.num + other.num

        return new_source

    def __repr__(self):
        """Return a summary of the wavelet sampling, source count, types and extent."""

        try:
            locs = np.array(self.locs)
            xmin = locs[:, 0].min()
            xmax = locs[:, 0].max()
            zmin = locs[:, 1].min()
            zmax = locs[:, 1].max()

            info = f"Seismic Source:\n"
            info += f"  Source wavelet: {self.nt} samples at {self.dt * 1000:.2f} ms\n"
            info += f"  Source number : {self.num}\n"
            info += f"  Source types  : {self.get_type(unique = True)}\n"
            info += f"  Source x range: {xmin:6.2f} - {xmax:6.2f} m\n"
            info += f"  Source z range: {zmin:6.2f} - {zmax:6.2f} m\n"
        except:
            info = f"Seismic Source:\n"
            info += f"  empty\n"

        return info

    def add_source(
        self,
        loc: List[float],
        wavelet: np.ndarray,
        type: str,
        mt: Optional[np.ndarray] = None, ) -> None:
        """Append a source.

        Parameters
        ----------
        loc : list of float
            Source location ``[x, z]`` (m).
        wavelet : np.ndarray
            Source time function of length ``nt``.
        type : str
            Source type (case-insensitive): ``'pr'`` (pressure), ``'vx'``,
            ``'vz'`` (velocity components) or ``'mt'`` (moment tensor).
        mt : np.ndarray, optional
            3x3 moment tensor. Required when ``type`` is ``'mt'``.

        Raises
        ------
        ValueError
            If ``loc`` does not have two entries, ``type`` is not supported,
            the wavelet length differs from ``nt``, ``mt`` is not 3x3, or
            ``mt`` is missing for an ``'mt'`` source.
        """

        if len(loc) != 2:
            raise ValueError(
                "Source location must be a list in the format of [x, z] (m)"
            )

        if type.lower() not in ["pr", "vx", "vz", "mt"]:
            raise ValueError(
                "Source type must be either pr, vx, vz, or mt"
            )

        if wavelet.shape[0] != self.nt:
            raise ValueError(
                "Source wavelet must have the same length as the number of time samples"
            )

        if mt is not None and mt.shape != (3, 3):
            raise ValueError("Moment tensor must be a 3x3 matrix")

        if type.lower() == "mt" and mt is None:
            raise ValueError("Moment tensor must be provided for mt source")

        self.locs.append(loc)
        self.type.append(type)
        self.wavelet.append(wavelet)
        self.moment_tensor.append(mt)
        self.num += 1

    def get_wavelet(self, isrc: int) -> np.ndarray:
        """Return the wavelet of one source.

        Parameters
        ----------
        isrc : int
            Source index, from 0 to ``num - 1``.

        Returns
        -------
        np.ndarray
            Source time function of length ``nt``.
        """

        return self.wavelet[isrc]

    def get_type(self, isrc=None, unique=False) -> List[str]:
        """Return source types.

        Parameters
        ----------
        isrc : int, optional
            Source index. If None, the types of all sources are returned.
        unique : bool, optional
            If True, return the list of distinct types over all sources
            (``isrc`` is then ignored). Default is False.

        Returns
        -------
        list of str or str
            Types of all sources, the type of source ``isrc``, or the distinct
            types when ``unique`` is True.
        """

        if isrc is None:
            type = self.type
        else:
            type = self.type[isrc]

        if unique:
            type = list(set(self.type))

        return type

    def get_moment_tensor(self, isrc: int) -> np.ndarray:
        """Return the moment tensor of one source.

        Parameters
        ----------
        isrc : int
            Source index, from 0 to ``num - 1``.

        Returns
        -------
        np.ndarray
            3x3 moment tensor.

        Raises
        ------
        RuntimeError
            If no moment tensor was given for this source.
        """

        mt = self.moment_tensor[isrc]

        if mt is None:
            raise RuntimeError(f"Moment tensor is not defined for source {isrc}")

        return mt

    def get_loc(self, isrc=None, type=None) -> np.ndarray:
        """Return source locations, optionally for one index or one type.

        Parameters
        ----------
        isrc : int, optional
            Source index. Cannot be combined with ``type``.
        type : str, optional
            Source type (``'pr'``, ``'vx'``, ``'vz'`` or ``'mt'``,
            case-insensitive). Cannot be combined with ``isrc``.

        Returns
        -------
        np.ndarray
            Locations ``[x, z]`` (m): shape (n, 2) for all sources or a given
            type, shape (2,) for a single index. An empty (0, 2) array is
            returned when no source matches.

        Raises
        ------
        ValueError
            If both ``isrc`` and ``type`` are given.
        """

        if isrc is None and type is None:
            locs = np.array(self.locs)

        elif isrc is not None and type is None:
            locs = np.array(self.locs[isrc])

        elif isrc is None and type is not None:
            locs = np.array(
                [
                    loc
                    for loc, t in zip(self.locs, self.type)
                    if t.lower() == type.lower()
                ]
            )

        else:
            raise ValueError("Cannot specify both isrc and type")

        if locs.shape[0] == 0:
            locs = np.empty((0, 2), dtype=int)

        return locs
