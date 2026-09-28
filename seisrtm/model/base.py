"""Abstract base class for 2D earth models used by the wave-equation propagators,
with shared utilities for bounds, I/O, cloning, sub-model selection and plotting.
"""

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy

import torch
from torch import Tensor


# physical units of the supported model parameters
units = {
    "vp": "m/s",
    "vs": "m/s",
    "rho": "kg/m^3",
    "lam": "Pa",
    "mu": "Pa",
    "Ip": "kg/m^2/s",
    "Is": "kg/m^2/s",
    "vs_vp": "none",
    "ref": "kg/m^2/s",
}
eps = 1e-7


class AbstractModel(torch.nn.Module):
    """Abstract 2D earth model on a regular grid.

    Subclasses define the model parameters (e.g. vp, vs, rho) as
    ``torch.nn.Parameter`` attributes and implement :meth:`forward`.

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
    free_surface : bool, optional
        Whether the top boundary is a free surface. Default is False.
    nabc : int, optional
        Number of absorbing boundary cells. Default is 20.
    """

    def __init__(self,
        ox: float,
        oz: float,
        dx: float,
        dz: float,
        nx: int,
        nz: int,
        free_surface: Optional[bool] = False,
        nabc: Optional[int] = 20) -> None:

        super().__init__()

        # common grid parameters
        self.ox = ox
        self.oz = oz
        self.dx = dx
        self.dz = dz
        self.nx = nx
        self.nz = nz
        self.free_surface = free_surface
        self.nabc = nabc
        assert self.dx == self.dz, "Model grid size dx and dz must be the same"

        # grid coordinates (m)
        self.x = np.arange(self.nx) * self.dx + self.ox
        self.z = np.arange(self.nz) * self.dz + self.oz

        # parameter names and per-parameter settings, filled by subclasses
        self.pars = []
        self.model = {}
        self.upper_bound = {}
        self.lower_bound = {}
        self.requires_grad = {}

    def forward(self, *args, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        """Return the elastic parameters used by the wave-equation propagator.

        Returns
        -------
        lam : torch.Tensor
            Lame parameter lambda (Pa).
        mu : torch.Tensor
            Shear modulus (Pa).
        buoyancy : torch.Tensor
            Buoyancy, 1/rho (m^3/kg).

        Raises
        ------
        NotImplementedError
            Always; must be implemented by the subclass.
        """
        raise NotImplementedError("Forward method must be implemented by the subclass")

    def __repr__(self) -> str:
        """Return a text summary of the model parameters, bounds and grid.

        Returns
        -------
        info : str
            Multi-line summary of the model.
        """

        info = f"Earth model with parameters {self.pars}:\n"
        for par in self.pars:
            par_min = self.get_model(par).min()
            par_max = self.get_model(par).max()
            requires_grad = self.requires_grad[par]
            lower_bound = self.lower_bound[par]
            upper_bound = self.upper_bound[par]
            info += (
                f"  Model {par:4s}: {par_min:8.2f} - {par_max:8.2f} {units[par]:6s}, "
                f"requires_grad = {requires_grad}, "
                f"constrain bound: {lower_bound} - {upper_bound}\n"
            )

        info += f"  Model orig: ox = {self.ox:6.2f}, oz = {self.oz:6.2f} m\n"
        info += f"  Model grid: dx = {self.dx:6.2f}, dz = {self.dz:6.2f} m\n"
        info += f"  Model dims: nx = {self.nx:6d}, nz = {self.nz:6d}\n"
        info += f"  Model size: {self.nx * self.nz * len(self.pars)}\n"
        info += f"  Free surface: {self.free_surface}\n"
        info += f"  Absorbing layers: {self.nabc}\n"

        return info

    def __sub__(self, other):
        """Subtract another model parameter-by-parameter.

        Parameters
        ----------
        other : AbstractModel
            Model defined on the same grid with the same parameters.

        Returns
        -------
        model : AbstractModel
            New model holding ``self - other`` for every parameter.
        """

        self.check_same_space(other)

        model = self.clone(other)

        for par in self.pars:
            model.set_model(par, self.get_model(par) - other.get_model(par))

        return model

    def __add__(self, other):
        """Add another model parameter-by-parameter.

        Parameters
        ----------
        other : AbstractModel
            Model defined on the same grid with the same parameters.

        Returns
        -------
        model : AbstractModel
            New model holding ``self + other`` for every parameter.
        """
        self.check_same_space(other)

        model = self.clone(other)

        for par in self.pars:
            model.set_model(par, self.get_model(par) + other.get_model(par))

        return model


    def forward(self, *args, **kwargs) -> Tuple[Tensor, Tensor, Tensor]:
        """Return the elastic parameters used by the wave-equation propagator.

        Returns
        -------
        lam : torch.Tensor
            Lame parameter lambda (Pa).
        mu : torch.Tensor
            Shear modulus (Pa).
        buoyancy : torch.Tensor
            Buoyancy, 1/rho (m^3/kg).

        Raises
        ------
        NotImplementedError
            Always; must be implemented by the subclass.
        """
        raise NotImplementedError("Forward method must be implemented by the subclass")

    def set_model_vector(self, model_flatten: np.ndarray) -> None:
        """Set the model parameters from a 1D array.

        Parameters
        ----------
        model_flatten : np.ndarray
            Flattened model parameters.

        Raises
        ------
        NotImplementedError
            Not implemented.
        """
        raise NotImplementedError("This method is not implemented yet")

    def get_model_vector(self) -> np.ndarray:
        """Return the model parameters as a 1D array.

        Returns
        -------
        model_flatten : np.ndarray
            Flattened model parameters.

        Raises
        ------
        NotImplementedError
            Not implemented.
        """
        raise NotImplementedError("This method is not implemented yet")

    def get_gradient_vector(self, par: str) -> torch.Tensor:
        """Return the gradient of a model parameter as a 1D tensor.

        Parameters
        ----------
        par : str
            Model parameter name.

        Returns
        -------
        grad : torch.Tensor
            Flattened gradient.

        Raises
        ------
        NotImplementedError
            Not implemented.
        """
        raise NotImplementedError("This method is not implemented yet")


    def clip_params(self) -> None:
        """Clamp each bounded model parameter in place to its bounds.

        Parameters without both a lower and an upper bound are left unchanged.
        """

        for par in self.pars:
            if self.lower_bound[par] is not None and self.upper_bound[par] is not None:
                m = getattr(self, par)
                min_value = self.lower_bound[par]
                max_value = self.upper_bound[par]
                m.data.clamp_(min_value, max_value)


    def constrain_range(self, value, min_value, max_value) -> Tensor:
        """Clamp a tensor to ``[min_value, max_value]``.

        NaN entries are first replaced in place by ``max_value``. If either
        bound is None the input is returned unchanged.

        Parameters
        ----------
        value : torch.Tensor
            Tensor to constrain.
        min_value : float or None
            Lower bound.
        max_value : float or None
            Upper bound.

        Returns
        -------
        value : torch.Tensor
            Constrained tensor.

        Raises
        ------
        ValueError
            If the clamped tensor contains inf, negative values or NaN.
        """

        if min_value is not None and max_value is not None:

            if torch.isnan(value).any():
                # replace nan with the upper bound
                value[torch.isnan(value)] = max_value

            # a sigmoid/logit reparameterization was found to produce NaN, so a hard clamp is used
            value = torch.clamp(value, min_value, max_value)

            if torch.isinf(value).any():
                raise ValueError("Value is inf")

            if (value < 0.0).any():
                raise ValueError("Value is negative")

            if torch.isnan(value).any():
                raise ValueError("Value is nan")

            return value

        else:
            return value

    def check_same_space(self, other):
        """Check that another model shares the same parameters and grid.

        Parameters
        ----------
        other : AbstractModel
            Model to compare against.

        Raises
        ------
        TypeError
            If ``other`` is not an ``AbstractModel``.
        AssertionError
            If parameters, origin, spacing, dimensions, free-surface flag or
            number of absorbing cells differ.
        """

        if not isinstance(other, AbstractModel):
            raise TypeError("Subtraction is only supported between two ElasticModel objects")

        assert self.pars == other.pars, "Model parameters must be the same"
        assert self.ox == other.ox, "Model origin in x-direction must be the same"
        assert self.oz == other.oz, "Model origin in z-direction must be the same"
        assert self.dx == other.dx, "Model grid size in x-direction must be the same"
        assert self.dz == other.dz, "Model grid size in z-direction must be the same"
        assert self.nx == other.nx, "Model dimensions in x-direction must be the same"
        assert self.nz == other.nz, "Model dimensions in z-direction must be the same"
        assert self.free_surface == other.free_surface, "Model free surface flag must be the same"
        assert self.nabc == other.nabc, "Model absorbing boundary cells must be the same"


    def check_bounds(self) -> None:
        """Validate the parameter bounds and widen them to enclose the model.

        A lower (upper) bound that does not lie strictly below (above) the
        model minimum (maximum) is reset to the model minimum minus ``eps``
        (maximum plus ``eps``).

        Raises
        ------
        AssertionError
            If a lower bound is not smaller than the corresponding upper bound.
        """

        for par in self.pars:
            if self.lower_bound[par] is not None and self.upper_bound[par] is not None:
                assert (
                    self.lower_bound[par] < self.upper_bound[par]
                ), "Lower bound must be smaller than upper bound"

            if self.lower_bound[par] is not None:
                if self.lower_bound[par] + eps > self.get_model(par).min():
                    Warning(f"Lower bound must be larger than minimum value, set to {self.get_model(par).min()}")
                    self.lower_bound[par] = self.get_model(par).min() - eps

            if self.upper_bound[par] is not None:
                if self.upper_bound[par] - eps < self.get_model(par).max():
                    Warning(f"Upper bound must be smaller than maximum value, set to {self.get_model(par).max()}")
                    self.upper_bound[par] = self.get_model(par).max() + eps


    def check_dims(self) -> None:
        """Check that every model parameter has shape (nz, nx).

        Raises
        ------
        AssertionError
            If any parameter has a different shape.
        """

        for par in self.pars:
            assert (
                self.get_model(par).shape == (self.nz, self.nx)
            ), "Model dimensions must be (nz, nx)"


    def get_origin(self) -> Tuple[float, float]:
        """Return the model origin.

        Returns
        -------
        ox : float
            Origin in the x-direction (m).
        oz : float
            Origin in the z-direction (m).
        """

        return self.ox, self.oz

    def get_shape(self) -> Tuple[int, int]:
        """Return the number of grid points.

        Returns
        -------
        nx : int
            Number of grid points in the x-direction.
        nz : int
            Number of grid points in the z-direction.
        """

        return self.nx, self.nz

    def get_grid(self) -> Tuple[float, float]:
        """Return the grid spacing.

        Returns
        -------
        dx : float
            Grid spacing in the x-direction (m).
        dz : float
            Grid spacing in the z-direction (m).
        """

        return self.dx, self.dz

    def get_pars(self) -> List[str]:
        """Return the model parameter names.

        Returns
        -------
        pars : list of str
            Names of the model parameters.
        """

        return self.pars

    def get_ndim(self) -> int:
        """Return the total number of model unknowns.

        Returns
        -------
        ndim : int
            Number of model unknowns.

        Raises
        ------
        NotImplementedError
            Not implemented.
        """

        raise NotImplementedError("This method is not implemented yet")


    def get_model(self, par: str) -> np.ndarray:
        """Return a copy of a model parameter as a numpy array.

        Parameters
        ----------
        par : str
            Model parameter name.

        Returns
        -------
        model : np.ndarray
            Model array with shape (nz, nx).

        Raises
        ------
        ValueError
            If ``par`` is not a parameter of the model.
        """
        if par not in self.pars:
            raise ValueError(f"Parameter {par} not in model")

        model = getattr(self, par)

        return model.clone().cpu().detach().numpy()

    def set_model(self, par: str, model: np.ndarray) -> None:
        """Replace a model parameter with a new float32 ``nn.Parameter``.

        The parameter keeps its current ``requires_grad`` setting.

        Parameters
        ----------
        par : str
            Model parameter name.
        model : np.ndarray or torch.Tensor
            New values with shape (nz, nx).

        Raises
        ------
        ValueError
            If ``par`` is not a parameter of the model or the shape is not
            (nz, nx).
        """

        if par not in self.pars:
            raise ValueError(f"Parameter {par} not in model")

        if model.shape != (self.nz, self.nx):
            raise ValueError("Model dimensions must be (nz, nx)")

        model = torch.tensor(model) if isinstance(model, np.ndarray) else model
        setattr(self, par,
                torch.nn.Parameter(model.to(torch.float32),
                requires_grad=self.requires_grad[par]))

    def get_bound(self, par: str) -> Tuple[float, float]:
        """Return the bounds of a model parameter.

        Parameters
        ----------
        par : str
            Model parameter name.

        Returns
        -------
        bound : list
            ``[min_value, max_value]``, or ``[None, None]`` if either bound
            is unset.

        Raises
        ------
        ValueError
            If ``par`` is not a parameter of the model.
        """

        if par not in self.pars:
            raise ValueError("Parameter {} not in model".format(par))

        m_min = self.lower_bound[par]
        m_max = self.upper_bound[par]

        if m_min is None or m_max is None:
            return [None, None]

        return [m_min, m_max]

    def get_requires_grad(self, par: str) -> bool:
        """Return whether a model parameter requires a gradient.

        Parameters
        ----------
        par : str
            Model parameter name.

        Returns
        -------
        requires_grad : bool
            True if the parameter is updated during inversion.

        Raises
        ------
        ValueError
            If ``par`` is not a parameter of the model.
        """

        if par not in self.pars:
            raise ValueError("Parameter {} not in model".format(par))

        return self.requires_grad[par]

    def get_grad(self, par: str) -> np.ndarray:
        """Return the gradient of a model parameter as a numpy array.

        Parameters
        ----------
        par : str
            Model parameter name.

        Returns
        -------
        grad : np.ndarray
            Gradient with shape (nz, nx); zeros if no gradient is available.

        Raises
        ------
        ValueError
            If ``par`` is not a parameter of the model.
        """

        if par not in self.pars:
            raise ValueError("Parameter {} not in model".format(par))

        m = getattr(self, par)

        if m.grad is None:
            return np.zeros(m.shape)

        else:
            return m.grad.cpu().detach().numpy()

    def get_free_surface(self) -> bool:
        """Return the free-surface flag.

        Returns
        -------
        free_surface : bool
            True if the top boundary is a free surface.
        """

        return self.free_surface

    def get_nabc(self) -> int:
        """Return the number of absorbing boundary cells.

        Returns
        -------
        nabc : int
            Number of absorbing boundary cells.
        """

        # nabc may be a 0-d array after loading from file
        try:
            return self.nabc.item()
        except:
            return self.nabc

    def get_clone_data(self) -> Tuple:
        """Return the constructor arguments needed to rebuild the model.

        Returns
        -------
        args : tuple
            Positional arguments ``(ox, oz, dx, dz, nx, nz)``.
        kwargs : dict
            Keyword arguments: each parameter array and its ``<par>_bound``
            and ``<par>_grad`` entries, plus ``free_surface``, ``nabc`` and
            ``mask``.
        """

        args = (
            self.ox,
            self.oz,
            self.dx,
            self.dz,
            self.nx,
            self.nz,
        )

        kwargs = {}
        for par in self.pars:
            kwargs[par] = self.get_model(par)
            kwargs[par + "_bound"] = self.get_bound(par)
            kwargs[par + "_grad"] = self.get_requires_grad(par)

        kwargs["free_surface"] = self.free_surface
        kwargs["nabc"] = self.nabc
        kwargs["mask"] = self.mask

        return args, kwargs


    def save(self, filename: str) -> None:
        """Save the model to an ``.npz`` file.

        Parameters
        ----------
        filename : str
            Output file name.
        """

        args, kwargs = self.get_clone_data()

        # positional args are stored as arr_0 ... arr_5
        np.savez(filename, *args, **kwargs)


    @classmethod
    def from_file(cls, filename: str, grad: Optional[bool] = False, verbose: Optional[bool] = False):
        """Load a model from an ``.npz`` file written by :meth:`save`.

        Parameters
        ----------
        filename : str
            Input file name.
        grad : bool, optional
            ``requires_grad`` flag applied to all parameters, overriding the
            saved values. Default is False.
        verbose : bool, optional
            Print a message after loading. Default is False.

        Returns
        -------
        model : AbstractModel
            Model of type ``cls``.
        """
        data = np.load(filename, allow_pickle=True)

        # the first six arrays are the positional args (ox, oz, dx, dz, nx, nz)
        num_file = 6
        args = tuple(data[f"arr_{i}"].item() for i in range(num_file))
        kwargs = {key: data[key] for key in data.files if not key.startswith("arr_")}

        for par in kwargs:
            if par.endswith("_grad"):
                kwargs[par] = grad

        if verbose:
            print(f"Model loaded from file: {filename}")

        return cls(*args, **kwargs)


    @classmethod
    def clone(cls, instance, grad: Optional[bool] = False):
        """Create a copy of a model.

        Parameters
        ----------
        instance : AbstractModel
            Model to copy.
        grad : bool, optional
            ``requires_grad`` flag applied to all parameters of the copy.
            Default is False.

        Returns
        -------
        model : AbstractModel
            Copied model of type ``cls``.

        Raises
        ------
        TypeError
            If ``instance`` is not an ``AbstractModel``.
        """
        if not isinstance(instance, AbstractModel):
            raise TypeError("instance must be an instance of AbstractModel")

        args, kwargs = instance.get_clone_data()

        for par in instance.pars:
            kwargs[par + "_grad"] = grad

        return cls(*args, **kwargs)


    @classmethod
    def resample(cls, instance, dh: float):
        """Resample a model to a new grid spacing.

        Parameters
        ----------
        instance : AbstractModel
            Model to resample.
        dh : float
            New grid spacing in both x and z (m).

        Returns
        -------
        model : AbstractModel
            Resampled model of type ``cls``.

        Raises
        ------
        ValueError
            Always raised; the interpolation is not yet reliable.
        """

        args, kwargs = instance.get_clone_data()

        # linear interpolation with scipy
        for par in instance.pars:
            m = instance.get_model(par)
            m_new = scipy.ndimage.zoom(m, instance.dx / dh, order=1)
            kwargs[par] = m_new

            raise ValueError("The interpolation is not working properly, need to fix it.")

        if instance.mask is not None:
            kwargs["mask"] = scipy.ndimage.zoom(instance.mask, instance.dx / dh, order=1)

        ox = args[0]
        oz = args[1]
        nx = kwargs["vp"].shape[1]
        nz = kwargs["vp"].shape[0]
        args_new = (ox, oz, dh, dh, nx, nz)

        return cls(*args_new, **kwargs)

    @classmethod
    def select(cls, instance, x0: Optional[float] = None,
                   x1: Optional[float] = None,
                   z0: Optional[float] = None,
                   z1: Optional[float] = None):
        """Extract a rectangular sub-model.

        Parameters
        ----------
        instance : AbstractModel
            Model to extract from.
        x0 : float, optional
            Minimum x-coordinate (m). Default is the model origin.
        x1 : float, optional
            Maximum x-coordinate (m). Default is the model end.
        z0 : float, optional
            Minimum z-coordinate (m). Default is the model origin.
        z1 : float, optional
            Maximum z-coordinate (m). Default is the model end.

        Returns
        -------
        model : AbstractModel
            Sub-model of type ``cls`` with origin ``(x0, z0)``.

        Raises
        ------
        AssertionError
            If the requested range lies outside the model.
        """

        args, kwargs = instance.get_clone_data()

        # default to the full model extent
        x0 = instance.ox                             if x0 is None else x0
        x1 = instance.ox + instance.nx * instance.dx if x1 is None else x1
        z0 = instance.oz                             if z0 is None else z0
        z1 = instance.oz + instance.nz * instance.dz if z1 is None else z1

        # convert coordinates to grid indices
        ix0 = int((x0 - instance.ox) / instance.dx)
        ix1 = int((x1 - instance.ox) / instance.dx)
        iz0 = int((z0 - instance.oz) / instance.dz)
        iz1 = int((z1 - instance.oz) / instance.dz)

        assert ix0 >= 0 and ix1 <= instance.nx, "x-coordinate is out of model range"
        assert iz0 >= 0 and iz1 <= instance.nz, "z-coordinate is out of model range"

        for par in instance.pars:
            m = instance.get_model(par)
            m = m[iz0:iz1, ix0:ix1]
            kwargs[par] = m

        dx = args[2]
        dz = args[3]
        args_new = (x0, z0, dx, dz, ix1-ix0, iz1-iz0)

        return cls(*args_new, **kwargs)

    # placeholder used only in type annotations
    Survey = None
    def plot(self, survey: Optional[Survey] = None,
            pars: Optional[List] = None,
            grad: Optional[bool] = False,
            orientation: Optional[str] = "horizontal",
            sym_clip: Optional[bool] = False,
            grid_on: Optional[bool] = False,
            cmap_range: Optional[dict] = None,
            clip: Optional[float] = 99.99,
            save_path: Optional[str] = None,
            add_label: Optional[bool] = False,
            xlim: List[Optional[float]] = None,
            zlim: List[Optional[float]] = None,
            axhline: Optional[float] = None,
            **kwargs) -> None:

        """Plot the model parameters or their gradients.

        Parameters
        ----------
        survey : Survey, optional
            If given, source and receiver locations are overlaid.
        pars : list of str, optional
            Parameters to plot. Default is all.
        grad : bool, optional
            Plot gradients instead of model values. Default is False.
        orientation : {'horizontal', 'vertical'}, optional
            Subplot and colorbar layout. Default is 'horizontal'.
        sym_clip : bool, optional
            Use a symmetric color range at the ``clip`` percentile of
            ``|m|``. Default is False.
        grid_on : bool, optional
            Draw grid lines. Default is False.
        cmap_range : dict, optional
            Mapping ``par -> (vmin, vmax)``. Default is the data range.
        clip : float, optional
            Percentile used for symmetric clipping. Default is 99.99.
        save_path : str, optional
            If given, save the figure to this path at 300 dpi.
        add_label : bool, optional
            Add panel labels a), b), c). Default is False.
        xlim : list of float, optional
            x-axis limits (m).
        zlim : list of float, optional
            Depth limits ``[zmin, zmax]`` (m).
        axhline : float, optional
            Depth (m) at which to draw a dashed horizontal line.
        **kwargs
            ``cmap`` (default 'jet'), ``fontsize`` (default 14) and
            ``aspect`` (default 'equal').

        Raises
        ------
        ValueError
            If ``orientation`` is not supported.
        """

        pars = self.get_pars() if pars is None else pars

        if not isinstance(pars, list):
            pars = [pars]

        cmap = kwargs.get("cmap", "jet")
        fontsize = kwargs.get("fontsize", 14)
        aspect = kwargs.get("aspect", "equal")
        labels = ['a)', 'b)', 'c)']

        plt.rcParams.update(
            {
                "axes.labelsize": fontsize,
                "xtick.labelsize": fontsize,
                "ytick.labelsize": fontsize,
                "legend.fontsize": fontsize,
                "figure.titlesize": fontsize,
            }
        )

        extent = [self.x[0], self.x[-1], self.z[-1], self.z[0]]

        pars = self._par if pars is None else pars
        if orientation == "vertical":
            fig = plt.figure(figsize=(12, 8))
        else:
            fig = plt.figure(figsize=(16, 8))

        for i, par in enumerate(pars):
            if orientation == "vertical":
                ax = fig.add_subplot(len(pars), 1, i + 1)
            elif orientation == "horizontal":
                ax = fig.add_subplot(1, len(pars), i + 1)
            else:
                raise ValueError(f"Orientation {orientation} not supported. Use 'vertical' or 'horizontal'")

            if not grad:

                if sym_clip:
                    data = self.get_model(par)
                    vmax = np.percentile(np.abs(data), clip)
                    ax.imshow(data, cmap=cmap, extent=extent, vmin=-vmax, vmax=vmax)
                    ax.set_title(par.upper(), fontsize=fontsize)
                else:
                    if cmap_range is not None:
                        vmin, vmax = cmap_range[par]
                    else:
                        vmin = self.get_model(par).min()
                        vmax = self.get_model(par).max()

                    ax.imshow(self.get_model(par), cmap=cmap, extent=extent, vmin=vmin, vmax=vmax)
                    ax.set_title(par.upper(), fontsize=fontsize)

            else:
                data = self.get_grad(par)
                vmax = np.percentile(np.abs(data), clip)
                ax.imshow(data, cmap=cmap, extent=extent, vmin=-vmax, vmax=vmax)
                ax.set_title("GRAD-" + par.upper(), fontsize=fontsize)

            if axhline is not None:
                ax.axhline(axhline, color='k', linestyle='--', linewidth=1.0)

            if xlim is not None:
                ax.set_xlim(xlim)
            if zlim is not None:
                ax.set_ylim(zlim[1], zlim[0])

            if grid_on:
                ax.grid()
            ax.set_xlabel("Distance (m)")
            ax.set_ylabel("Depth (m)")
            ax.set_aspect(aspect)

            # panel label a), b), c)
            if add_label:
                ax.text(-0.17, 1.1, labels[i], transform=ax.transAxes, fontsize=fontsize, va='top', fontweight='bold')

            # colorbar with the parameter unit
            if orientation == "vertical":
                pad = 0.025
            elif  orientation == "horizontal":
                pad = 0.08
            cbar = fig.colorbar(
                ax.images[0], ax=ax, orientation=orientation, pad=pad
            )
            cbar.ax.set_ylabel(f'${units[par]}$', rotation=0, fontsize=fontsize, labelpad=30,)
            cbar.ax.yaxis.set_label_position("right")

            if survey is not None:
                src_type = ['Src-Pr', 'Src-Vx', 'Src-Vz', 'Src-MT']
                src_pr = survey.source.get_loc(type='pr')
                src_vx = survey.source.get_loc(type='vx')
                src_vz = survey.source.get_loc(type='vz')
                src_mt = survey.source.get_loc(type='mt')

                rec_type = ['Rec-Pr', 'Rec-Vx', 'Rec-Vz', 'Rec-DAS']
                rec_pr = survey.receiver.get_loc(type='pr')
                rec_vx = survey.receiver.get_loc(type='vx')
                rec_vz = survey.receiver.get_loc(type='vz')
                rec_das = survey.receiver.get_loc(type='das')

                # point receivers
                for type, loc in zip(rec_type, [rec_pr, rec_vx, rec_vz]):
                    if loc.shape[0] > 0:
                        ax.scatter(loc[:, 0], loc[:, 1], marker='^', s=20,  c='k', label=type)

                # DAS fiber drawn as a line
                if rec_das.shape[0] > 0:
                    ax.plot(rec_das[:, 0], rec_das[:, 1], 'k-', label='Rec-DAS', linewidth=2)

                # sources
                for type, loc in zip(src_type, [src_pr, src_vx, src_vz, src_mt]):
                    if loc.shape[0] > 0:
                        ax.scatter(loc[:, 0], loc[:, 1], marker='*', s=30, c='r', label=type)

                ax.legend(loc='upper right', fontsize=fontsize-4, framealpha=0.5)

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()



    def plot_well_log(self, x: Optional[float] = None, save_path = None,
                      add_label: Optional[bool] = False, **kwargs) -> None:
        """Plot vertical profiles of all model parameters at one x position.

        Parameters
        ----------
        x : float, optional
            x-coordinate of the profile (m). Default is the model center.
        save_path : str, optional
            If given, save the figure to this path at 300 dpi.
        add_label : bool, optional
            Add panel labels a), b), c). Default is False.
        **kwargs
            ``color`` (default 'k'), ``linewidth`` (default 2),
            ``linestyle`` (default '-') and ``fontsize`` (default 14).
        """


        labels = ['a)', 'b)', 'c)']

        parms ={
            "color": kwargs.get("color", "k"),
            "linewidth":  kwargs.get("linewidth", 2),
            "linestyle": kwargs.get("linestyle", "-"),
        }

        fontsize = kwargs.get("fontsize", 14)
        plt.rcParams.update(
            {
                "axes.labelsize": fontsize,
                "xtick.labelsize": fontsize,
                "ytick.labelsize": fontsize,
                "legend.fontsize": fontsize,
                "figure.titlesize": fontsize,
            }
        )

        if x is None:
            x = self.ox + self.dx * self.nx / 2

        idx = int((x - self.ox) / self.dx)

        fig = plt.figure(figsize=(10, 6))
        for i, par in enumerate(self.pars):

            m = self.get_model(par)
            ax = fig.add_subplot(1,3,i+1)
            plt.plot(m[:, idx], self.z, **parms)
            plt.xlabel(par.upper() + f' (${units[par]}$)')
            plt.ylabel("Depth (m)")
            plt.gca().invert_yaxis()
            plt.title(f'At x = {x} m')

            if add_label:
                ax.text(-0.17, 1.1, labels[i], transform=ax.transAxes,
                        fontsize=fontsize, va='top', fontweight='bold')

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_mask(self, save_path: Optional[str] = None,
                  fontsize: Optional[int] = 14,
                  cmap: Optional[str] = "binary",
                  aspect: Optional[str] = "equal",
                  orientation="horizontal",
                  figsize: Optional[Tuple[int, int]] = (10, 6),
                  ) -> None:
        """Plot the gradient mask.

        Parameters
        ----------
        save_path : str, optional
            If given, save the figure to this path at 300 dpi.
        fontsize : int, optional
            Font size. Default is 14.
        cmap : str, optional
            Colormap. Default is 'binary'.
        aspect : str, optional
            Axes aspect ratio. Default is 'equal'.
        orientation : {'horizontal', 'vertical'}, optional
            Colorbar orientation. Default is 'horizontal'.
        figsize : tuple of int, optional
            Figure size in inches. Default is (10, 6).
        """

        plt.rcParams.update(
            {
                "axes.labelsize": fontsize,
                "xtick.labelsize": fontsize,
                "ytick.labelsize": fontsize,
                "legend.fontsize": fontsize,
                "figure.titlesize": fontsize,
            }
        )

        parms ={
                "cmap": cmap,
                "aspect":  aspect,
                "extent": [self.x[0], self.x[-1], self.z[-1], self.z[0]]
        }


        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(self.mask, **parms)
        ax.set_title('Mask Applied to Gradient', fontsize=fontsize)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Depth (m)")

        cbar = fig.colorbar(
            ax.images[0], ax=ax, orientation=orientation,
        )

        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()


    def analyze_survey(self, survey: Survey) -> None:
        """Check that all sources and receivers lie strictly inside the model.

        Parameters
        ----------
        survey : Survey
            Survey with ``source`` and ``receiver`` geometry.

        Raises
        ------
        RuntimeError
            If any source or receiver lies on or outside the model boundary.
        """

        src_type = set(survey.source.get_type())
        for type in src_type:
            src_loc = survey.source.get_loc(type = type)

            if (src_loc[:,0].min() <= self.x.min() or
                src_loc[:,0].max() >= self.x.max() or
                src_loc[:,1].min() <= self.z.min() or
                src_loc[:,1].max() >= self.z.max()):
                raise RuntimeError('Survey Error: source location is out of model range')

        rec_type = set(survey.receiver.get_type())

        for type in rec_type:
            rec_loc = survey.receiver.get_loc(type = type)

            if (rec_loc[:,0].min() <= self.x.min() or
                rec_loc[:,0].max() >= self.x.max() or
                rec_loc[:,1].min() <= self.z.min() or
                rec_loc[:,1].max() >= self.z.max()):
                raise RuntimeError(f'Survey Error: {type} receiver location is out of model range')

        print("Survey analysis completed: legal survey")
