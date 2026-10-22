"""Optical parameters of a cloud field from its droplet-size-distribution
microphysics.

Reads a netCDF cloud-model snapshot such as
BOMEX_512x512x170_500CCN_20m_micro_256_0000021600.com3D_int_2.nc, whose
variable `FNCD` holds the droplet spectrum - the number concentration N_i
[#/cm^3] of droplets in each of the 33 radius bins `rd` [um] - and derives,
per voxel, the fields SMART-G's 3D cloud needs.

The equations are those of Tzabari et al., "Settings for Spaceborne 3-D
Scattering Tomography of Liquid-Phase Clouds by the CloudCT Mission"
(IEEE TGRS 60, 2022), Section II-A. With n(r) the droplet size distribution
(the paper models it as a gamma distribution) and n(r) dr -> N_i per bin:

    LWC = 4/3 pi rho_w  int r^3 n(r) dr                       (1)  [g/m^3]
    r_e = int pi r^2 r n(r) dr / int pi r^2 n(r) dr           (2)  [um]
    beta = int n(r) Q pi r^2 dr,  Q ~ 2 (Mie, visible)        (3)  [1/km]
    beta = 3/2 LWC / (r_e rho_w)                              (4)  [1/km]

(3) and (4) are the same quantity for a constant Q = 2. `beta` is computed
either that way (`extinction="q2"`, the default) or with the full integral,
using the Mie extinction efficiency Q(r) of each radius bin at 670 nm
(`extinction="mie"`); Eq. (4) is always returned too as `beta_from_lwc_re`.
The optical depth along a path is the integral of beta along it
(`optical_depth()`).

For SMART-G, `make_cloud3d()` builds the 3D cloud with either a
Henyey-Greenstein phase function of a chosen asymmetry parameter g
(`phase_function="hg"`) or the phase functions of SMART-G's water-cloud
table (`phase_function="mie"`).

Beyond the paper, the effective variance v_e is returned too, since the
droplet distribution's width is needed for the scattering phase function.
It uses the standard definition

    v_e = int (r - r_e)^2 pi r^2 n(r) dr / (r_e^2 int pi r^2 n(r) dr)

Not an equation of the paper: the voxel filtering (`lwc_min`, `n_min`),
which follows the processing notebook this module was written from.

Note the netCDF variable `rho` is the *air* density; the water density rho_w
is the constant 1 g/cm^3.

Arrays are (nz, ny, nx), the file's own order.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

# Physical constants and defaults.
RHO_W_G_CM3 = 1.0  # liquid water density, g/cm^3
Q_EXT = 2.0  # extinction efficiency, Mie at visible wavelengths
SMARTG_W_REF_NM = 670.0  # wavelength the extinction is quoted at for SMART-G
M_WATER = 1.331  # refractive index of liquid water at 670 nm (absorption ~1e-8, ignored)
EXTINCTION_Q2 = "q2"  # beta with the paper's Q = 2
EXTINCTION_MIE = "mie"  # beta with the Mie Q of each radius bin
MIE_SUBSAMPLES = 16  # radii averaged per bin in mie_q_ext_bins()
PHASE_HG = "hg"  # Henyey-Greenstein phase function with a chosen g
PHASE_MIE = "mie"  # the Mie phase functions of SMART-G's water-cloud table
LWC_MIN_G_M3 = 0.01  # voxels with less liquid water are treated as clear
N_MIN_CM3 = 1.0  # voxels with fewer droplets are treated as clear

SPECTRUM_VAR = "FNCD"
RADIUS_VAR = "rd"
Z_CHUNK = 4  # levels read at a time; keeps the (33, z, y, x) block small

# Unit conversions, with r in um and N in #/cm^3:
#   um^3 * #/cm^3 = 1e-12 cm^3 * #/cm^3 -> cm^3 of water per cm^3 of air
#   * rho_w [g/cm^3] * 1e6 [cm^3/m^3] -> g/m^3
_LWC_FACTOR = 4.0 / 3.0 * np.pi * RHO_W_G_CM3 * 1e-12 * 1e6
#   um^2 * #/cm^3 = 1e-12 m^2 * 1e6 #/m^3 = 1e-6 1/m = 1e-3 1/km
_BETA_FACTOR = 1e-3


@dataclass
class CloudOptics:
    """Per-voxel optical parameters; all 3D fields are (nz, ny, nx)."""

    x_m: np.ndarray
    y_m: np.ndarray
    z_m: np.ndarray
    lwc: np.ndarray  # liquid water content, g/m^3, Eq. (1)
    r_eff: np.ndarray  # effective radius, um, Eq. (2)
    v_eff: np.ndarray  # effective variance, dimensionless
    n_total: np.ndarray  # droplet number concentration, #/cm^3
    beta: np.ndarray  # extinction coefficient, 1/km, Eq. (3)
    beta_from_lwc_re: np.ndarray  # extinction coefficient, 1/km, Eq. (4)

    @property
    def cloud_mask(self) -> np.ndarray:
        return self.lwc > 0.0

    @property
    def voxel_size_m(self) -> tuple[float, float, float]:
        """(dx, dy, dz) from the grid coordinates; dz is the *smallest* layer
        thickness, since the vertical grid is stretched near the top."""
        return tuple(float(np.diff(a).min()) for a in (self.x_m, self.y_m, self.z_m))

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, **self.__dict__)

    @classmethod
    def load(cls, path: str | Path) -> "CloudOptics":
        with np.load(path) as data:
            return cls(**{k: data[k] for k in data.files})


def _mie_qext_single(x: float, m: complex) -> float:
    """Extinction efficiency of one sphere of size parameter x = 2 pi r / lambda
    and relative refractive index m (Bohren & Huffman's BHMIE recursions)."""
    y = m * x
    nstop = int(x + 4.0 * x ** (1.0 / 3.0) + 2.0)
    nmx = int(max(nstop, abs(y)) + 15)
    d = np.zeros(nmx + 1, np.complex128)  # logarithmic derivative, by downward recurrence
    for n in range(nmx, 0, -1):
        d[n - 1] = n / y - 1.0 / (d[n] + n / y)
    psi0, psi1 = np.cos(x), np.sin(x)
    chi0, chi1 = -np.sin(x), np.cos(x)
    xi1 = complex(psi1, -chi1)
    total = 0.0
    for n in range(1, nstop + 1):
        psi = (2 * n - 1) * psi1 / x - psi0
        chi = (2 * n - 1) * chi1 / x - chi0
        xi = complex(psi, -chi)
        a = ((d[n] / m + n / x) * psi - psi1) / ((d[n] / m + n / x) * xi - xi1)
        b = ((m * d[n] + n / x) * psi - psi1) / ((m * d[n] + n / x) * xi - xi1)
        total += (2 * n + 1) * (a.real + b.real)
        psi0, psi1, chi0, chi1 = psi1, psi, chi1, chi
        xi1 = complex(psi1, -chi1)
    return 2.0 / x**2 * total


def mie_q_ext_bins(
    r_um: np.ndarray,
    wavelength_nm: float = SMARTG_W_REF_NM,
    m_water: complex = M_WATER,
) -> np.ndarray:
    """Mie extinction efficiency Q of each radius bin, at `wavelength_nm`.

    Q(r) ripples strongly with r for droplets of a few wavelengths, so each
    bin is represented by the r^2-weighted mean of Q over the bin's radius
    range (edges halfway between neighbouring radii on a log scale), not by
    the single value at its nominal radius. Above x = 400 the ripple is
    below ~1% and the bin centre is used (this also keeps the recursion
    short for the mm-sized rain bins).
    """
    r = np.asarray(r_um, np.float64)
    log_mid = (np.log(r[1:]) + np.log(r[:-1])) / 2
    log_edges = np.concatenate([[2 * np.log(r[0]) - log_mid[0]], log_mid,
                                [2 * np.log(r[-1]) - log_mid[-1]]])
    edges = np.exp(log_edges)
    q = np.empty_like(r)
    for i, ri in enumerate(r):
        x_of = lambda rad: 2.0 * np.pi * rad / (wavelength_nm * 1e-3)
        if x_of(ri) > 400.0:
            q[i] = _mie_qext_single(x_of(ri), m_water)
            continue
        sub = np.linspace(edges[i], edges[i + 1], MIE_SUBSAMPLES)
        qs = np.array([_mie_qext_single(x_of(s), m_water) for s in sub])
        q[i] = np.sum(qs * sub**2) / np.sum(sub**2)
    return q


def spectrum_moments(
    r_um: np.ndarray,
    counts: np.ndarray,
    r_max_um: float | None = None,
    q_bins: np.ndarray | float = Q_EXT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sums S_k = sum_i r_i^k N_i for k = 0, 2, 3, 4, and S_2q = sum_i Q_i
    r_i^2 N_i, over the bin axis (axis 0).

    These discretise the integrals int r^k n(r) dr of the paper. Bins above
    `r_max_um` are left out (e.g. to exclude rain drops); by default every
    bin is used. `q_bins` is the extinction efficiency, one value or one per
    bin.
    """
    shape = counts.shape[1:]
    s0, s2, s3, s4, s2q = (np.zeros(shape, np.float64) for _ in range(5))
    q_bins = np.broadcast_to(np.asarray(q_bins, np.float64), np.shape(r_um))
    for r, q, n in zip(np.asarray(r_um, np.float64), q_bins, counts):
        if r_max_um is not None and r > r_max_um:
            continue
        n = np.nan_to_num(n, nan=0.0).astype(np.float64)
        s0 += n
        s2 += r**2 * n
        s3 += r**3 * n
        s4 += r**4 * n
        s2q += q * r**2 * n
    return s0, s2, s3, s4, s2q


def optics_from_spectrum(
    r_um: np.ndarray,
    counts: np.ndarray,
    r_max_um: float | None = None,
    lwc_min: float = LWC_MIN_G_M3,
    n_min: float = N_MIN_CM3,
    q_bins: np.ndarray | float = Q_EXT,
) -> dict[str, np.ndarray]:
    """LWC, r_e, v_e, N and beta for droplet spectra.

    `counts` is (n_bins, ...) in #/cm^3 per bin, `r_um` the (n_bins,) bin
    radii in um. `q_bins` is the extinction efficiency in Eq. (3): the
    paper's constant 2 by default, or one Mie value per bin (see
    `mie_q_ext_bins`). Voxels below `lwc_min` [g/m^3] or `n_min` [#/cm^3] are
    set to zero in every output. `beta_from_lwc_re` is Eq. (4), which
    assumes Q = 2, whatever `q_bins` is.
    """
    s0, s2, s3, s4, s2q = spectrum_moments(r_um, counts, r_max_um, q_bins)

    lwc = _LWC_FACTOR * s3  # Eq. (1)
    beta = np.pi * _BETA_FACTOR * s2q  # Eq. (3)

    has_drops = s2 > 0.0
    safe_s2 = np.where(has_drops, s2, 1.0)
    safe_s3 = np.where(has_drops, s3, 1.0)
    r_eff = np.where(has_drops, s3 / safe_s2, 0.0)  # Eq. (2)
    # v_e = S4 S2 / S3^2 - 1, expanding (r - r_e)^2 in the definition
    v_eff = np.where(has_drops, s4 * s2 / safe_s3**2 - 1.0, 0.0)

    # Eq. (4); r_e in um -> m, rho_w in g/m^3, result 1/m -> 1/km
    rho_w_g_m3 = RHO_W_G_CM3 * 1e6
    safe_re = np.where(has_drops, r_eff, 1.0)
    beta_4 = np.where(has_drops, 1.5 * lwc / (safe_re * 1e-6 * rho_w_g_m3) * 1e3, 0.0)

    clear = (lwc < lwc_min) | (s0 < n_min)
    out = dict(
        lwc=lwc, r_eff=r_eff, v_eff=v_eff, n_total=s0, beta=beta, beta_from_lwc_re=beta_4
    )
    for field in out.values():
        field[clear] = 0.0
    return {k: v.astype(np.float32) for k, v in out.items()}


def load_cloud_optics(
    path: str | Path,
    time_index: int = 0,
    r_max_um: float | None = None,
    lwc_min: float = LWC_MIN_G_M3,
    n_min: float = N_MIN_CM3,
    extinction: str = EXTINCTION_Q2,
    w_ref_nm: float = SMARTG_W_REF_NM,
    m_water: complex = M_WATER,
) -> CloudOptics:
    """Read a netCDF snapshot and derive the cloud's optical parameters.

    The spectrum is read `Z_CHUNK` levels at a time, so the multi-GB file is
    never fully in memory. `r_max_um` optionally drops large (rain) bins.

    `extinction` picks how beta is computed from Eq. (3):
    - "q2": the paper's approximation Q = 2 (Mie, visible), beta = 2 pi S_2.
    - "mie": the full integral, with the Mie extinction efficiency of each
      radius bin at `w_ref_nm` for water of refractive index `m_water`.
    """
    if extinction == EXTINCTION_Q2:
        q_bins = Q_EXT
    elif extinction == EXTINCTION_MIE:
        q_bins = None  # needs the radius axis, read below
    else:
        raise ValueError(f"extinction must be {EXTINCTION_Q2!r} or {EXTINCTION_MIE!r}")

    with Dataset(path) as ds:
        ds.set_auto_mask(False)
        spectrum = ds.variables[SPECTRUM_VAR]  # (time, bin, z, y, x)
        r_um = np.asarray(ds.variables[RADIUS_VAR][:], np.float64)
        x_m, y_m, z_m = (np.asarray(ds.variables[k][:], np.float64) for k in "xyz")

        if q_bins is None:
            q_bins = mie_q_ext_bins(r_um, w_ref_nm, m_water)

        nz = spectrum.shape[2]
        chunks = []
        for z0 in range(0, nz, Z_CHUNK):
            block = spectrum[time_index, :, z0 : z0 + Z_CHUNK]
            chunks.append(
                optics_from_spectrum(r_um, block, r_max_um, lwc_min, n_min, q_bins)
            )

    fields = {k: np.concatenate([c[k] for c in chunks], axis=0) for k in chunks[0]}
    return CloudOptics(x_m=x_m, y_m=y_m, z_m=z_m, **fields)


def optical_depth(
    optics: CloudOptics, axis: str = "z", beta: np.ndarray | None = None
) -> np.ndarray:
    """Optical depth, integral of beta along a grid axis (default: vertical).

    The result is 2D. The vertical value is the cloud's optical thickness
    seen at nadir; use `axis="x"` or `"y"` for horizontal paths.
    """
    beta = optics.beta if beta is None else beta
    ax, coord = {"z": (0, optics.z_m), "y": (1, optics.y_m), "x": (2, optics.x_m)}[axis]
    shape = [1, 1, 1]
    shape[ax] = -1
    thickness_m = np.diff(cell_bounds_m(coord)).reshape(shape)
    return (beta * thickness_m * 1e-3).sum(axis=ax, dtype=np.float64)  # beta in 1/km


def cell_bounds_m(centres_m: np.ndarray) -> np.ndarray:
    """Cell boundaries for cell-centre coordinates, at the midpoints between
    neighbours. The BOMEX files' vertical grid stretches from 20 m to 50 m
    near the top, so cells are not assumed to be equally thick."""
    c = np.asarray(centres_m, np.float64)
    return np.concatenate([[1.5 * c[0] - 0.5 * c[1]], (c[1:] + c[:-1]) / 2,
                           [1.5 * c[-1] - 0.5 * c[-2]]])


def to_smartg_dataset(
    optics: CloudOptics,
    w_ref_nm: float = SMARTG_W_REF_NM,
    crop: tuple[slice, slice, slice] | None = None,
):
    """The cloud as the dense 3D dataset that `smartg.atmosphere.Cloud3D(ds=...)`
    takes (the same schema `read_i3rc_cloud()` returns):

    - ext(z, y, x): extinction coefficient [1/km] at `w_ref_nm`, 0 where clear
    - reff(z, y, x): effective radius [um]
    - x_bounds, y_bounds, z_bounds: cell boundaries [km]; x and y are centred
      on 0 (as `read_i3rc_cloud` does by default), z starts at the ground

    `ext` is the field's `beta` (Eq. 3, whichever Q was chosen when loading),
    quoted at `w_ref_nm`; SMART-G scales it to other wavelengths with its
    water-cloud table (`wc_sol.nc`), a change of ~1% between 440 and 670 nm.
    `crop` = (z, y, x) slices selects a sub-volume, e.g. around one cloud.
    """
    import xarray as xr

    zs, ys, xs = crop if crop is not None else (slice(None),) * 3

    def bounds_km(centres_m: np.ndarray, sl: slice, centre_on_zero: bool) -> np.ndarray:
        lo, hi, _ = sl.indices(len(centres_m))
        edges = cell_bounds_m(centres_m)[lo : hi + 1] * 1e-3
        return edges - edges.mean() if centre_on_zero else edges

    return xr.Dataset(
        {
            "ext": (("z", "y", "x"), optics.beta[zs, ys, xs].astype(np.float64)),
            "reff": (("z", "y", "x"), optics.r_eff[zs, ys, xs].astype(np.float64)),
        },
        coords={
            "x_bounds": ("x_b", bounds_km(optics.x_m, xs, True)),
            "y_bounds": ("y_b", bounds_km(optics.y_m, ys, True)),
            "z_bounds": ("z_b", bounds_km(optics.z_m, zs, False)),
        },
        attrs={"w_ref": float(w_ref_nm)},
    )


def henyey_greenstein(g: float, theta_deg: np.ndarray) -> np.ndarray:
    """Henyey-Greenstein phase function, normalised as SMART-G expects:
    the integral of P(theta) sin(theta) d(theta) over 0..180 deg is 2.

        P(theta) = (1 - g^2) / (1 + g^2 - 2 g cos(theta))^(3/2)

    g is the asymmetry parameter <cos theta>; liquid-water clouds in the
    visible have g ~ 0.85.
    """
    if not -1.0 < g < 1.0:
        raise ValueError(f"asymmetry parameter g must be in (-1, 1), got {g}")
    cos_t = np.cos(np.radians(theta_deg))
    return (1.0 - g**2) / (1.0 + g**2 - 2.0 * g * cos_t) ** 1.5


def henyey_greenstein_phase(g: float, n_theta: int = 3601):
    """`phase=` argument of `Cloud3D`: the same Henyey-Greenstein matrix for
    every effective radius and wavelength.

    Only the intensity term is meaningful: F11 = F22 = F33 = F44 = HG and
    F12 = F34 = 0 give the correct I, but Q, U, V are not physical (HG is a
    scalar phase function). The axes only need to span the cloud's `reff`
    (SMART-G's water-cloud table covers 5-30 um) and the wavelengths rendered.
    """
    import xarray as xr

    theta = np.linspace(0.0, 180.0, n_theta)
    hg = henyey_greenstein(g, theta)
    terms = np.stack([hg, np.zeros_like(hg), hg, np.zeros_like(hg)])  # F11, F21, F33, F34
    data = np.broadcast_to(terms, (2, 2, 4, n_theta)).copy()
    return xr.DataArray(
        data,
        coords={
            "wavelength_phase": [100.0, 6000.0],
            "reff": [1.0, 100.0],
            "nphamat": np.arange(4),
            "theta_atm": theta,
        },
        dims=["wavelength_phase", "reff", "nphamat", "theta_atm"],
        name="phase_atm",
    )


def make_cloud3d(
    optics: CloudOptics,
    phase_function: str = PHASE_HG,
    g: float = 0.85,
    crop: tuple[slice, slice, slice] | None = None,
    ssa: float | None = None,
    w_ref_nm: float = SMARTG_W_REF_NM,
    reff_min_um: float = 5.0,
):
    """Build SMART-G's `Cloud3D` for this cloud.

    `phase_function` picks the scattering phase function:
    - "hg": Henyey-Greenstein with asymmetry parameter `g` (ignored by "mie").
    - "mie": the phase matrices of SMART-G's water-cloud table (`wc_sol.nc`),
      taken to be Mie phase functions of spherical droplets (effective
      variance 0.1) and indexed by each voxel's `reff`; the table covers
      5-30 um. This is the table's own content, not independently checked.

    `ssa` forces the single-scattering albedo; by default it comes from
    the same table (~0.99999 in the visible). `reff_min_um` raises smaller
    effective radii to it, since the table starts at 5 um.
    Returns (cloud3d, dataset); the dataset's x/y/z bounds give the
    `Grid3D`: Grid3D(ds.x_bounds.values, ds.y_bounds.values, ds.z_bounds.values, ...).
    """
    from smartg.atmosphere import Cloud3D

    if phase_function == PHASE_HG:
        phase = henyey_greenstein_phase(g)
    elif phase_function == PHASE_MIE:
        phase = None  # Cloud3D then reads the phase matrices from the table
    else:
        raise ValueError(f"phase_function must be {PHASE_HG!r} or {PHASE_MIE!r}")

    ds = to_smartg_dataset(optics, w_ref_nm, crop)
    cloud = Cloud3D(
        "wc",
        w_ref=w_ref_nm,
        ds=ds,
        reff_acc=1,
        reff_min=reff_min_um,
        phase=phase,
        ssa_cst=ssa,
    )
    return cloud, ds


def check_eq3_eq4_consistency(optics: CloudOptics) -> float:
    """Largest relative difference between beta from Eq. (3) and Eq. (4);
    only ~0 when the field was loaded with extinction="q2", since Eq. (4)
    assumes Q = 2."""
    m = optics.cloud_mask
    if not m.any():
        return 0.0
    return float(np.max(np.abs(optics.beta[m] - optics.beta_from_lwc_re[m]) / optics.beta[m]))


def summary(optics: CloudOptics) -> str:
    m = optics.cloud_mask
    lines = [
        f"grid (nz, ny, nx) = {optics.lwc.shape}, voxel (dx, dy, dz) = "
        f"{tuple(round(v, 3) for v in optics.voxel_size_m)} m",
        f"cloudy voxels: {int(m.sum())} of {m.size} ({100 * m.mean():.3f}%)",
    ]
    if m.any():
        zc = optics.z_m[m.any(axis=(1, 2))]
        lines.append(f"cloud heights: {zc.min():.0f} - {zc.max():.0f} m")
        for name, field, unit in (
            ("LWC", optics.lwc, "g/m^3"),
            ("r_eff", optics.r_eff, "um"),
            ("v_eff", optics.v_eff, ""),
            ("N", optics.n_total, "#/cm^3"),
            ("beta", optics.beta, "1/km"),
        ):
            p = np.percentile(field[m], [5, 50, 95, 100])
            lines.append(
                f"{name:6s} p5/p50/p95/max = "
                + " / ".join(f"{v:.4g}" for v in p)
                + f" {unit}"
            )
        tau = optical_depth(optics)
        lines.append(f"vertical optical depth: max {tau.max():.2f}, mean over cloudy columns "
                     f"{tau[tau > 0].mean():.2f}")
        ratio = optics.beta[m] / optics.beta_from_lwc_re[m]
        lines.append(f"beta / beta(Eq.4, Q=2): p5/p50/p95 = "
                     + " / ".join(f"{v:.4f}" for v in np.percentile(ratio, [5, 50, 95])))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("netcdf", type=Path, help="cloud snapshot with an FNCD spectrum")
    parser.add_argument("--out", type=Path, help="save the result to this .npz file")
    parser.add_argument("--r-max-um", type=float, default=None,
                        help="ignore drops larger than this radius (rain cut-off)")
    parser.add_argument("--extinction", choices=[EXTINCTION_Q2, EXTINCTION_MIE],
                        default=EXTINCTION_Q2,
                        help="beta from Eq. (3) with Q = 2 (default) or the full Mie Q per radius bin")
    args = parser.parse_args()

    optics = load_cloud_optics(args.netcdf, r_max_um=args.r_max_um, extinction=args.extinction)
    print(summary(optics))
    if args.out:
        optics.save(args.out)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
