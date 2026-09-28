"""Seismic receiver definition for 2-D elastic wave-equation modelling."""

from typing import List
import numpy as np

# Placeholder for the DAS cable type used by ``Receiver.add_cable``.
GenericCable = None


class Receiver(object):
    """Collection of seismic receivers sharing one time axis.

    Parameters
    ----------
    nt : int
        Number of time samples of the recorded data.
    dt : float
        Time sampling interval of the recorded data (s).

    Notes
    -----
    - Recording starts at time 0.
    - Receivers are added with :meth:`add_receiver`; a DAS cable with
      :meth:`add_cable`.
    - The same receivers are used for every shot, so data can be stored as a
      regular (nshot, nrec, nt) array, which is efficient on GPUs. Streamer
      geometries with shot-dependent receivers can be mimicked by masking
      offsets in the regular data.
    """

    def __init__(self, nt: int, dt: float) -> None:
        self.nt = nt
        self.dt = dt

        self.locs = []
        self.type = []
        self.num = 0
        self.cable = None

    def __add__(self, other: "Receiver") -> "Receiver":
        """Merge two receiver collections.

        Parameters
        ----------
        other : Receiver
            Receiver collection to append.

        Returns
        -------
        Receiver
            New collection containing the receivers of both operands. Any
            attached DAS cable is not carried over.

        Raises
        ------
        TypeError
            If ``other`` is not a ``Receiver``.
        ValueError
            If ``nt`` or ``dt`` differ between the two collections.
        """

        if not isinstance(other, Receiver):
            raise TypeError(
                "Receiver Error: the other receiver must be an instance of Receiver"
            )

        if self.nt != other.nt or self.dt != other.dt:
            raise ValueError(
                "Receiver Error: the number of time samples and time interval must be the same"
            )

        new_receiver = Receiver(self.nt, self.dt)
        new_receiver.locs = self.locs + other.locs
        new_receiver.type = self.type + other.type
        new_receiver.num = self.num + other.num

        return new_receiver

    def __repr__(self):
        """Return a summary of the data sampling, receiver count, types and extent."""

        try:
            locs = np.array(self.locs)
            xmin = locs[:, 0].min()
            xmax = locs[:, 0].max()
            zmin = locs[:, 1].min()
            zmax = locs[:, 1].max()

            info = f"Seismic Receiver:\n"
            info += (
                f"  Receiver data   : {self.nt} samples at {self.dt * 1000:.2f} ms\n"
            )
            info += f"  Receiver number : {self.num}\n"
            info += f"  Receiver types  : {self.get_type(unique = True)}\n"
            info += f"  Receiver x range: {xmin:6.2f} - {xmax:6.2f} m\n"
            info += f"  Receiver z range: {zmin:6.2f} - {zmax:6.2f} m\n"
        except:
            info = f"Seismic Receiver:\n"
            info += f"  empty\n"

        return info

    def add_receiver(self, loc: List[float], type: str) -> None:
        """Append a receiver.

        Parameters
        ----------
        loc : list of float
            Receiver location ``[x, z]`` (m).
        type : str
            Receiver type (case-insensitive): ``'pr'`` (pressure), ``'vx'``,
            ``'vz'`` (velocity components) or ``'das'``.

        Raises
        ------
        ValueError
            If ``loc`` does not have two entries or ``type`` is not supported.
        """

        if len(loc) != 2:
            raise ValueError(
                "Receiver Error: the location must be in the format of [x, z]"
            )

        if type.lower() not in ["pr", "vx", "vz", "das"]:
            raise ValueError("Receiver type must be either pr, vx, vz, or das")

        self.locs.append(loc)
        self.type.append(type)
        self.num += 1
        self.cable_num = 0

    def add_cable(self, cable: GenericCable) -> None:
        """Attach a DAS cable and add its unique channel locations as receivers.

        Parameters
        ----------
        cable : GenericCable
            DAS cable object providing ``get_rec_loc_unique()``, which returns
            (n, 3) coordinates; the x and z columns are used.

        Raises
        ------
        TypeError
            If ``cable`` is not a ``GenericCable``.
        ValueError
            If more than one cable is attached.
        """

        if not isinstance(cable, GenericCable):
            raise TypeError(
                "Receiver Error: the cable must be an instance of GenericCable"
            )

        self.cable = cable

        # unique channel locations, keeping the x and z columns
        rec_loc = cable.get_rec_loc_unique()[:, [0, 2]]
        rec_num = rec_loc.shape[0]

        for irec in range(rec_num):
            self.add_receiver(rec_loc[irec], "das")

        print(f"  DAS cable is added to the survey with {rec_num} unique receivers")

        self.cable_num += 1

        if self.cable_num > 1:
            raise ValueError("Only one DAS cable is allowed for now, future work \
                              will be done to support multiple DAS cables")


    def get_loc(self, type: str) -> np.ndarray:
        """Return the locations of all receivers of a given type.

        Parameters
        ----------
        type : str
            Receiver type: ``'pr'``, ``'vx'``, ``'vz'`` or ``'das'``
            (case-insensitive).

        Returns
        -------
        np.ndarray
            Locations ``[x, z]`` (m) with shape (n, 2); an empty (0, 2) array
            if no receiver matches.

        Raises
        ------
        ValueError
            If ``type`` is not supported.
        """

        if type.lower() not in ["pr", "vx", "vz", "das"]:
            raise ValueError("Receiver type must be either pr, vx, vz, or das")

        rec_locs = np.array(
            [loc for loc, t in zip(self.locs, self.type) if t.lower() == type.lower()]
        )

        if rec_locs.shape[0] == 0:
            rec_locs = np.empty((0, 2), dtype=int)

        return rec_locs

    def get_type(self, unique=False) -> List[str]:
        """Return receiver types.

        Parameters
        ----------
        unique : bool, optional
            If True, return only the distinct types. Default is False.

        Returns
        -------
        list of str
            Type of every receiver, or the distinct types.
        """

        if unique:
            return list(set(self.type))
        else:
            return self.type

    def get_loc_dict(self) -> dict:
        """Return receiver locations grouped by type.

        Returns
        -------
        dict
            Mapping from lower-case receiver type to a list of ``[x, z]``
            locations (m).
        """

        loc_dict = {}
        for loc, t in zip(self.locs, self.type):
            if t.lower() not in loc_dict.keys():
                loc_dict[t.lower()] = []
            loc_dict[t.lower()].append(loc)

        return loc_dict
