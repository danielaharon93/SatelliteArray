"""Write a SMART-G 3D cloud file in the format of `cumulus.dat` from a cloud
built by cloud_microphysics.py (the BOMEX dataset).

The format (SMART-G's `read_i3rc_cloud()` reads it; see
theory/cumulus_dat_format.pdf):

    line 1     free-text comment
    line 2     Nx Ny Nz flag
    line 3     dx dy z_0 z_1 ... z_Nz         [km; Nz + 1 z boundaries]
    line 4...  ix iy iz ext reff, one line per cloudy cell
               (1-based indices, ix slowest and iz fastest; ext in 1/km at
               the reference wavelength, reff in um)

Usage, in the order the functions are meant to be called:

    grid = grid_from_optics(optics, crop=...)       # the file's grid
    cf = create_cumulus_file(path, grid, comment)   # writes lines 1-3
    set_extinction(cf, optics.beta[crop])           # fill the fields ...
    set_effective_radius(cf, optics.r_eff[crop])
    write_cells(cf)                                 # ... and write the cell lines

`build_cumulus_file()` does all of that in one call, and
`verify_cumulus_file()` reads the result back with SMART-G.

Only the cloud's own layers carry cell lines. The grid is padded above the
data with empty layers up to 30 km, as cumulus.dat does: SMART-G models the
background atmosphere (molecules, ozone) only on the grid's layers, so
without them the atmosphere above the cloud would be missing.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cloud_microphysics import CloudOptics, cell_bounds_m

FLAG = 2  # line 2's last value, as in cumulus.dat (not read by SMART-G)
# Boundaries of the empty layers added above the data [km], as in cumulus.dat
# (which reaches 30 km); only those more than TOP_PAD_MIN_KM above it are used.
TOP_PAD_EDGES_KM = (3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0, 25.0, 30.0)
TOP_PAD_MIN_KM = 0.25
UNIFORM_TOL = 1e-4  # relative; the horizontal spacing must be uniform
# Digits written: cumulus.dat has <= 3 decimals on ext, which would round thin
# voxels to zero; %g keeps significant digits and drops trailing zeros.
EXT_FORMAT = "%.5g"
REFF_FORMAT = "%.4g"


@dataclass
class CumulusGrid:
    """The grid of the file: horizontal cell count and size, and the z
    boundaries (including the padding above the data)."""

    nx: int
    ny: int
    dx_km: float
    dy_km: float
    z_bounds_km: np.ndarray
    nz_data: int  # layers that hold data; the layers above are padding

    @property
    def nz(self) -> int:
        return len(self.z_bounds_km) - 1


@dataclass
class CumulusFile:
    """A cumulus file being built: its path, its grid, and the fields to write."""

    path: Path
    grid: CumulusGrid
    ext: np.ndarray | None = field(default=None, repr=False)  # (nz_data, ny, nx), 1/km
    reff: np.ndarray | None = field(default=None, repr=False)  # (nz_data, ny, nx), um
    n_cells: int | None = None  # set once the cell lines are written


def crop_around_thickest_column(optics: CloudOptics, width_cells: int, z_cells: int | None = None):
    """(z, y, x) slices of a `width_cells`-wide square column around the
    column with the largest optical depth, from the ground up to `z_cells`
    layers (default: all). The square is moved to stay inside the field."""
    from cloud_microphysics import optical_depth

    tau = optical_depth(optics)
    iy, ix = np.unravel_index(tau.argmax(), tau.shape)
    ny, nx = tau.shape

    def window(centre: int, n: int) -> slice:
        start = int(np.clip(centre - width_cells // 2, 0, max(n - width_cells, 0)))
        return slice(start, start + min(width_cells, n))

    return slice(0, z_cells), window(iy, ny), window(ix, nx)


def grid_from_optics(
    optics: CloudOptics,
    crop: tuple[slice, slice, slice] | None = None,
    pad_top: bool = True,
) -> CumulusGrid:
    """The file's grid for the (cropped) cloud.

    The horizontal cell size is a single number per axis, so the grid must be
    uniform there. The first z boundary must be the ground (0 km), so a z crop
    has to start at the lowest layer. With `pad_top`, empty layers are added
    above the data up to 30 km (`TOP_PAD_EDGES_KM`).
    """
    zs, ys, xs = crop if crop is not None else (slice(None),) * 3
    x_m, y_m, z_m = optics.x_m[xs], optics.y_m[ys], optics.z_m[zs]

    steps = []
    for name, coord in (("x", x_m), ("y", y_m)):
        d = np.diff(coord)
        if len(coord) < 2 or np.ptp(d) > UNIFORM_TOL * d.mean():
            raise ValueError(f"the {name} grid must be uniform with at least 2 cells")
        steps.append(float(d.mean()) * 1e-3)

    z_bounds = cell_bounds_m(z_m) * 1e-3
    if abs(z_bounds[0]) > 1e-6:
        raise ValueError(
            f"the lowest z boundary is {z_bounds[0]:.4f} km, not the ground (0 km): the "
            "z crop must start at the lowest layer, since the atmosphere below the grid is not modelled"
        )
    z_bounds[0] = 0.0
    nz_data = len(z_bounds) - 1
    if pad_top:
        pad = [e for e in TOP_PAD_EDGES_KM if e > z_bounds[-1] + TOP_PAD_MIN_KM]
        z_bounds = np.concatenate([z_bounds, pad])
    return CumulusGrid(len(x_m), len(y_m), steps[0], steps[1], z_bounds, nz_data)


def create_cumulus_file(path: str | Path, grid: CumulusGrid, comment: str = "") -> CumulusFile:
    """Create the file with its three header lines (comment; `Nx Ny Nz flag`;
    `dx dy` and the z boundaries). The cell lines are added later by
    `write_cells()`. Returns the object that the fill functions work on."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    comment = " ".join(comment.split())  # one line
    comment = comment if comment.startswith("#") else "# " + comment
    with open(path, "w", newline="\n") as f:
        f.write(comment + "\n")
        f.write(f"{grid.nx} {grid.ny} {grid.nz} {FLAG} \n")
        f.write(" ".join(f"{v:.6g}" for v in (grid.dx_km, grid.dy_km, *grid.z_bounds_km)) + "\n")
    return CumulusFile(path, grid)


def _check_field(cf: CumulusFile, values: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(values)
    expected = (cf.grid.nz_data, cf.grid.ny, cf.grid.nx)
    if values.shape != expected:
        raise ValueError(f"{name} has shape {values.shape}, the file's grid needs (nz, ny, nx) = {expected}")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"{name} must be finite and not negative")
    return values


def set_extinction(cf: CumulusFile, ext_km: np.ndarray) -> None:
    """Set the extinction coefficient field [1/km, at the reference wavelength],
    shape (nz, ny, nx) of the cropped cloud; 0 where there is no cloud."""
    cf.ext = _check_field(cf, ext_km, "extinction")


def set_effective_radius(cf: CumulusFile, reff_um: np.ndarray) -> None:
    """Set the droplet effective radius field [um], shape (nz, ny, nx); 0 where
    there is no cloud."""
    cf.reff = _check_field(cf, reff_um, "effective radius")


def write_cells(cf: CumulusFile) -> int:
    """Append the cell lines `ix iy iz ext reff` to the file: the cells where
    both fields are positive, with 1-based indices, ix slowest and iz fastest.
    Returns the number of cells written."""
    if cf.ext is None or cf.reff is None:
        raise RuntimeError("set_extinction() and set_effective_radius() must be called first")
    if cf.n_cells is not None:
        raise RuntimeError(f"the cells of {cf.path} have already been written")

    # (z, y, x) -> (x, y, z): np.nonzero then lists the cells in C order of the
    # transposed array, i.e. ix slowest, iy next, iz fastest
    ext_xyz = np.transpose(cf.ext, (2, 1, 0))
    reff_xyz = np.transpose(cf.reff, (2, 1, 0))
    ix, iy, iz = np.nonzero((ext_xyz > 0) & (reff_xyz > 0))
    table = np.column_stack([ix + 1, iy + 1, iz + 1,
                             ext_xyz[ix, iy, iz].astype(np.float64),
                             reff_xyz[ix, iy, iz].astype(np.float64)])
    with open(cf.path, "a", newline="\n") as f:
        np.savetxt(f, table, fmt=("%d", "%d", "%d", EXT_FORMAT, REFF_FORMAT))
    cf.n_cells = len(table)
    return cf.n_cells


def build_cumulus_file(
    optics: CloudOptics,
    path: str | Path,
    crop: tuple[slice, slice, slice] | None = None,
    comment: str = "",
    pad_top: bool = True,
) -> CumulusFile:
    """Create, fill and write a cumulus file for `optics` (optionally cropped)
    in one call. `comment` is stored on line 1; say there how the field was
    made (source file, extinction method, reference wavelength), since the
    file itself cannot carry it."""
    zs, ys, xs = crop if crop is not None else (slice(None),) * 3
    cf = create_cumulus_file(path, grid_from_optics(optics, crop, pad_top), comment)
    set_extinction(cf, optics.beta[zs, ys, xs])
    set_effective_radius(cf, optics.r_eff[zs, ys, xs])
    write_cells(cf)
    return cf


def verify_cumulus_file(cf: CumulusFile) -> dict[str, float]:
    """Read the file back with SMART-G's `read_i3rc_cloud()` and compare it with
    the fields that were written. Returns the largest relative differences
    (from the rounding of the digits written) and the cell counts; raises if
    the grid, the cell set or the values do not match."""
    from smartg.atmosphere import read_i3rc_cloud

    ds = read_i3rc_cloud(cf.path)
    g = cf.grid
    if ds["ext"].shape != (g.nz, g.ny, g.nx):
        raise AssertionError(f"read-back shape {ds['ext'].shape} != {(g.nz, g.ny, g.nx)}")
    if not np.allclose(ds["z_bounds"].values, g.z_bounds_km, atol=1e-5):
        raise AssertionError("z bounds differ after the round trip")

    ext = ds["ext"].values[: g.nz_data]
    reff = ds["reff"].values[: g.nz_data]
    written = (cf.ext > 0) & (cf.reff > 0)
    if not np.array_equal(ext > 0, written):
        raise AssertionError("the set of cloudy cells differs after the round trip")
    if ds["ext"].values[g.nz_data :].any():
        raise AssertionError("the padding layers are not empty")
    rel = lambda a, b: float(np.max(np.abs(a[written] - b[written]) / b[written])) if written.any() else 0.0
    return {
        "cells": int(written.sum()),
        "max_rel_diff_ext": rel(ext, cf.ext.astype(np.float64)),
        "max_rel_diff_reff": rel(reff, cf.reff.astype(np.float64)),
        "grid_cells": g.nx * g.ny * g.nz,
    }
