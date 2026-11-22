"""Process a raw SMART-G result into an image, interactively.

Open the raw reflectance array (<name>.npy) that run_bomex_headless.py,
run_from_config.py or the playground saves, then change the processing in the
window and watch the image change - to compare processing approaches on the
same simulation, without rendering again.

    python process_reflectance_gui.py [<name>.npy]

What can be set (applied top to bottom):

  1. What the numbers are: reflectance (the sun's spectrum divided out), radiance
     (reflectance x solar irradiance x cos(sun zenith) / pi, in W m-2 um-1 sr-1, the
     unit of MODIS L1B radiances, so the sun's colour stays in), or, with the CloudCT camera
     response, the camera's digital number DN = G_eff x L_eff x exposure time.
  2. How the channels are made. "As simulated": the saved channels. When the render
     was made with the grid band set, <name>_spectral.npz holds the reflectance at every
     simulated wavelength, and the channels can be rebuilt with another camera response:
     overlapping Gaussian filters, curves from a CSV file (columns: nm, blue, green, red),
     MODIS Terra / Aqua, or the CloudCT camera (configs/cloudct/: one gray channel, whose
     radiance is the integral L_eff = sum of L(lambda) x eta_hat(lambda) x bin width, in
     W m-2 sr-1, not divided by the response's area; G_eff = 18.3 DN / (W m-2 sr-1 s)).
  3. Colour: white balance (none, divide out the sun, gray world), per-channel gains,
     normalisation (the chosen percentile becomes 1), a gain, saturation.
  4. Display: highlight handling (clip or roll off) and the tone curve (gamma, sRGB, linear).
  5. What to show: the colour image or one channel.
  6. Spectrum plot: reflectance (or, in radiance mode, spectral radiance) against wavelength
     for one pixel, chosen by clicking it in the image (it is outlined there; the pixel's x and y
     can also be typed). With the grid band set it
     shows every simulated wavelength, the camera response behind it, and each channel's value
     over the wavelengths it covers; for other band sets only the channels, at their wavelengths.

Polarization. A render made with smartg_playground.py or the headless scripts also has <name>_stokes.npy,
the Stokes parameters I, Q, U, V of every channel (shape (4, height, width, channels)), and, for the
grid band set, <name>_stokes_spectral.npz (the Stokes parameters at every simulated wavelength). When
<name>_stokes.npy sits beside the file, group 6 of the window can show the degree of linear
polarization, DoLP = sqrt(Q^2 + U^2) / I, or its angle AoLP = atan2(U, Q) / 2 as a map (of the channel
you choose), and plot DoLP or AoLP against wavelength for the clicked pixel. The wavelength options of
group 2 rebuild the Stokes channels too. DoLP and AoLP do not depend on reflectance or radiance.

Files next to <name>.npy that are used when they exist: <name>_radiance.json (the solar
irradiance per channel and the sun zenith angle, written by run_bomex_headless.py with
SAVE_RADIANCE), <name>_spectral.npz (grid band set) and the two polarization files above. Radiance needs the solar irradiance:
from the first, from the second, or, when neither exists, the values the playground uses for
the band set that the file name gives (_modis, or none for 440 / 550 / 670 nm). The sun zenith
angle comes from the json, or from the file name (rounded to whole degrees: edit it in the
window). The channel arrays are in the order blue, green, red.

The processing is plain numpy (process() and what it calls, no Qt), so it can also be
used from a script. Needs numpy, pillow and PyQt6 (all in requirements.txt); no GPU and
no SMART-G.
"""

import json
import re
import sys
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from PIL import Image

from cloudct_response import load_cloudct_response  # the CloudCT camera's response (configs/cloudct/)

CHANNEL_NAMES = ("blue", "green", "red")  # the order of the arrays' last axis
LUMINANCE_BGR = np.array([0.0722, 0.7152, 0.2126])  # Rec. 709, in the arrays' B, G, R order

QUANTITIES = ("reflectance", "radiance", "dn")
SPECTRAL_CHOICES = ("saved", "gaussian", "csv", "modis_terra", "modis_aqua", "cloudct")
# The CloudCT camera's normalised spectral responsivity (one curve: the camera is monochrome); see
# configs/cloudct/README.md. Its radiance is the integral of L(lambda) x response, in W m-2 sr-1, and its
# signal in digital numbers is DN = G_eff x that x the exposure time.
CLOUDCT_G_EFF = 18.3  # DN / (W m-2 sr-1 s), from the camera's radiometric calibration
MW_TO_W = 1e-3  # the solar irradiance is in mW m-2 nm-1
# The measured MODIS relative spectral responses of bands 3, 4 and 1 (blue, green, red), detector-averaged,
# from NASA's MODIS Characterization Support Team (see configs/modis_rsr/README.md).
MODIS_RESPONSE_FILES = {
    "modis_terra": Path(__file__).with_name("configs") / "modis_rsr" / "modis_terra_rsr_b03_b04_b01.csv",
    "modis_aqua": Path(__file__).with_name("configs") / "modis_rsr" / "modis_aqua_rsr_b03_b04_b01.csv",
}
WHITE_BALANCES = ("none", "sun", "gray")
NORMALISATIONS = ("none", "percentile")
HIGHLIGHTS = ("clip", "rolloff")
TONES = ("gamma", "srgb", "linear")
VIEWS = ("colour", *CHANNEL_NAMES)
MAPS = ("image", "dolp", "aolp")  # what the image pane shows: the colour image, or a polarization map
PLOTS = ("spectrum", "dolp", "aolp")  # what the plot shows against wavelength


# ----------------------------------------------------------------------------
# what is loaded
# ----------------------------------------------------------------------------


@dataclass
class Loaded:
    """A raw result and what sits next to it."""

    path: Path
    reflectance: np.ndarray  # (height, width, n_channels), last axis blue, green, red
    sza_deg: float | None = None
    irradiance: np.ndarray | None = None  # mW m-2 nm-1, per saved channel
    irradiance_source: str = ""  # where `irradiance` came from, for the window's text
    wavelength_nm: np.ndarray | None = None  # (n_wavelengths,)
    spectral: np.ndarray | None = None  # (height, width, n_wavelengths)
    solar: np.ndarray | None = None  # mW m-2 nm-1 at each wavelength
    bin_width_nm: np.ndarray | None = None
    group: np.ndarray | None = None  # the saved channel each wavelength belongs to
    channel_labels: tuple = ()  # the saved channels' names, when the files give them
    description: dict | None = None  # the contents of <name>_radiance.json, when it exists
    stokes: np.ndarray | None = None  # (4, height, width, n_channels): I, Q, U, V, when polarization was saved
    stokes_spectral: np.ndarray | None = None  # (4, height or 1, width or 1, n_wavelengths)
    stokes_wavelength_nm: np.ndarray | None = None  # the wavelengths of stokes_spectral

    @property
    def has_spectral(self) -> bool:
        return self.spectral is not None

    @property
    def single_index(self) -> int | None:
        """0 blue, 1 green, 2 red when the result is a single-channel run (one saved channel) whose
        channel is known, from the _radiance.json's scenario or the file name (_blue, _green, _red);
        else None. A rebuilt single channel uses only that channel's camera response."""
        if self.reflectance.shape[-1] != 1:
            return None
        if "cloudct" in str((self.description or {}).get("band_set", "")):
            return None  # the CloudCT band sets are one channel of their own, whatever "channel" says
        name = ((self.description or {}).get("scenario") or {}).get("channel")
        if name in CHANNEL_NAMES:
            return CHANNEL_NAMES.index(name)
        return name_band_info(self.path)[1]

    @property
    def has_polarization(self) -> bool:
        return self.stokes is not None


# The solar irradiance (mW m-2 nm-1, blue, green, red) that run_bomex_headless.py and the
# playground use for a render without <name>_radiance.json: SMART-G's solar REPTRAN values at the
# MODIS b03, b04, b01 representative wavelengths, and at the "rgb" band set's 440, 550, 670 nm.
ASSUMED_IRRADIANCE = {
    "modis": (2049.0884, 1869.5815, 1576.4119),
    "rgb": (1847.8285, 1891.2274, 1531.7266),
}


def name_band_info(path: Path) -> tuple[str | None, int | None]:
    """What a file's name says: the band set ("modis", "rgb" for the 440 / 550 / 670 nm set, or None
    for "grid", whose wavelengths are in <name>_spectral.npz) and, for a single-channel render, the
    channel's index (0 blue, 1 green, 2 red; else None). The band set is in the name (_modis, _grid,
    or nothing for rgb; a single channel adds _blue, _green or _red); a formation's satellite_NN.npy
    has it in its folder's name."""
    path = Path(path)
    text = f"{path.parent.name}/{path.name}" if path.stem.startswith("satellite_") else path.name
    single = re.search(r"_(blue|green|red)(?=_noise|/|\.|$)", text)
    index = ["blue", "green", "red"].index(single.group(1)) if single else None
    if "_grid" in text or "_cloudct" in text:
        return None, index
    return ("modis" if "_modis" in text else "rgb"), index


def assumed_irradiance(path: Path, n_channels: int) -> tuple[np.ndarray, str] | None:
    """The irradiance a file's name implies (see name_band_info); None for the grid band set,
    whose irradiance is in <name>_spectral.npz."""
    band_set, index = name_band_info(path)
    if band_set is None:
        return None
    table = np.array(ASSUMED_IRRADIANCE[band_set])
    if n_channels == 1 and index is not None:
        return table[index][None], f"assumed from the file name ({band_set} band set)"
    if n_channels == 3 and index is None:
        return table, f"assumed from the file name ({band_set} band set)"
    return None


def sun_zenith_from_name(name: str) -> float | None:
    """The sun zenith angle (deg, rounded) in a file name like ..._sza39_..."""
    match = re.search(r"_sza(\d+)", name)
    return float(match.group(1)) if match else None


def _bin_widths(wavelength_nm: np.ndarray, group: np.ndarray) -> np.ndarray:
    """The width of each wavelength's bin, from the spacing of the bin centres in its channel
    (a channel with one bin uses the spacing of the others)."""
    width = np.full(len(wavelength_nm), np.nan)
    for k in np.unique(group):
        mine = np.where(group == k)[0]
        if len(mine) > 1:
            width[mine] = np.median(np.diff(np.sort(wavelength_nm[mine])))
    width[np.isnan(width)] = np.nanmedian(width) if np.isfinite(width).any() else 10.0
    return width


def load(path: str | Path) -> Loaded:
    """Load <name>.npy and the files next to it (a "_radiance", "_stokes" or "_stokes_spectral" file
    name opens its reflectance)."""
    path = Path(path)
    for suffix in ("_radiance", "_stokes_spectral", "_stokes"):
        if path.stem.endswith(suffix):
            path = path.with_name(path.stem[: -len(suffix)] + ".npy")
            break
    reflectance = np.load(path)
    if reflectance.ndim != 3:
        raise ValueError(f"{path.name} has shape {reflectance.shape}; expected (height, width, channels)")
    loaded = Loaded(path, reflectance.astype(float), sza_deg=sun_zenith_from_name(path.name) or sun_zenith_from_name(path.parent.name))

    meta = path.with_name(f"{path.stem}_radiance.json")
    if meta.is_file():
        description = json.loads(meta.read_text())
        loaded.sza_deg = float(description["sza_deg"])
        loaded.irradiance = np.array(description["channel_irradiance_mW_m2_nm"], float)
        loaded.irradiance_source = "from the _radiance.json"
        loaded.channel_labels = tuple(description.get("channels", ()))
        loaded.description = description
    elif (guess := assumed_irradiance(path, reflectance.shape[-1])) is not None:
        loaded.irradiance, loaded.irradiance_source = guess

    spectral = path.with_name(f"{path.stem}_spectral.npz")
    if spectral.is_file():
        with np.load(spectral) as data:
            cube = data["reflectance"].astype(float)
            wavelength = data["wavelength_nm"].astype(float)
            if cube.shape[:2] == (1, 1):  # the clear sky: one pixel stands for the whole image
                cube = np.broadcast_to(cube, reflectance.shape[:2] + cube.shape[2:]).copy()
            if cube.shape[:2] == reflectance.shape[:2]:
                loaded.spectral, loaded.wavelength_nm = cube, wavelength
                loaded.group = data["channel_of_wavelength"].astype(int)
                if not loaded.channel_labels and "channel_labels" in data:
                    loaded.channel_labels = tuple(str(x) for x in data["channel_labels"])
                if "solar_irradiance_mW_m2_nm" in data and "bin_width_nm" in data:
                    loaded.solar = data["solar_irradiance_mW_m2_nm"].astype(float)
                    loaded.bin_width_nm = data["bin_width_nm"].astype(float)
                else:  # an older file: channel_weight = response (1) x solar x bin width
                    group = data["channel_of_wavelength"]
                    loaded.bin_width_nm = _bin_widths(wavelength, group)
                    loaded.solar = data["channel_weight"].astype(float) / loaded.bin_width_nm
                if loaded.irradiance is None:  # the saved channels' irradiance, from the wavelengths:
                    group = data["channel_of_wavelength"]  # sum(weight) / sum(width) per channel
                    if group.max() + 1 == reflectance.shape[-1]:
                        weight = data["channel_weight"].astype(float)
                        loaded.irradiance = np.array(
                            [weight[group == k].sum() / loaded.bin_width_nm[group == k].sum() for k in range(group.max() + 1)]
                        )
                        loaded.irradiance_source = "from the per-wavelength file"

    stokes_file = path.with_name(f"{path.stem}_stokes.npy")
    if stokes_file.is_file():
        cube = np.load(stokes_file).astype(float)
        if cube.shape == (4, *reflectance.shape):
            loaded.stokes = cube
    stokes_spectral = path.with_name(f"{path.stem}_stokes_spectral.npz")
    if loaded.stokes is not None and stokes_spectral.is_file():
        with np.load(stokes_spectral) as data:
            cube, wavelength = data["stokes"].astype(float), data["wavelength_nm"].astype(float)
        if cube.ndim == 4 and cube.shape[0] == 4 and cube.shape[-1] == len(wavelength) and cube.shape[1:3] in ((1, 1), reflectance.shape[:2]):
            loaded.stokes_spectral, loaded.stokes_wavelength_nm = cube, wavelength
    return loaded


# ----------------------------------------------------------------------------
# the processing
# ----------------------------------------------------------------------------


@dataclass
class Settings:
    quantity: str = "reflectance"
    spectral: str = "saved"
    centres_nm: tuple = (460.0, 540.0, 610.0)  # blue, green, red filters (Gaussian)
    fwhm_nm: tuple = (70.0, 80.0, 90.0)
    response_csv: str = ""  # columns: nm, blue, green, red
    sza_deg: float = 0.0
    responsivity: float = CLOUDCT_G_EFF  # G_eff, DN / (W m-2 sr-1 s): for quantity "dn"
    exposure_s: float = 1.0  # the exposure time, for quantity "dn"
    white_balance: str = "none"
    gains: tuple = (1.0, 1.0, 1.0)  # blue, green, red
    normalisation: str = "none"
    percentile: float = 99.5
    gain: float = 1.0
    saturation: float = 1.0
    highlights: str = "clip"
    tone: str = "gamma"
    gamma: float = 1.8
    view: str = "colour"
    map: str = "image"  # "dolp" or "aolp": show that polarization map instead of the image
    pol_channel: str = "green"  # the channel the map is of
    dolp_max_pct: float = 0.0  # the DoLP map's full scale, in percent; 0: the largest value
    plot: str = "spectrum"  # what the plot shows against wavelength: "spectrum", "dolp" or "aolp"


def load_response_csv(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """A camera response file: columns nm, blue, green, red (a header line is skipped)."""
    table = np.genfromtxt(path, delimiter=",", comments="#")
    table = np.atleast_2d(table)
    table = table[~np.isnan(table).any(axis=1)]
    if table.shape[1] < 4 or len(table) < 2:
        raise ValueError("the response file needs the columns nm, blue, green, red and at least two rows")
    order = np.argsort(table[:, 0])
    return table[order, 0], table[order, 1:4]


def cloudct_coverage(loaded: Loaded) -> float:
    """The share of the CloudCT response (its integral) that lies in the simulated wavelengths: below 1
    the integrated radiance misses the rest of the camera's band."""
    nm, eta = load_cloudct_response()
    whole = float(np.sum(np.diff(nm) * (eta[1:] + eta[:-1]) / 2.0))
    simulated = float((np.interp(loaded.wavelength_nm, nm, eta, left=0.0, right=0.0) * loaded.bin_width_nm).sum())
    return simulated / whole


def n_made_channels(loaded: Loaded, s: Settings) -> int:
    """How many channels the options of group 2 make: one for the (monochrome) CloudCT camera, else three,
    or the saved number for "As simulated" and for a single-channel run."""
    if s.spectral == "cloudct":
        return 1
    return 3 if s.spectral != "saved" and loaded.single_index is None else loaded.reflectance.shape[-1]


def response_matrix(loaded: Loaded, s: Settings) -> np.ndarray:
    """(n_wavelengths, 3): the camera's response at each simulated wavelength; (n_wavelengths, 1), that of
    the one channel, for a single-channel run (see Loaded.single_index) and for the CloudCT camera."""
    full = _full_response(loaded, s)
    k = loaded.single_index
    return full if k is None or full.shape[1] == 1 else full[:, [k]]


def _full_response(loaded: Loaded, s: Settings) -> np.ndarray:
    lam = loaded.wavelength_nm
    if s.spectral == "cloudct":
        nm, eta = load_cloudct_response()
        return np.interp(lam, nm, eta, left=0.0, right=0.0)[:, None]
    if s.spectral == "gaussian":
        centre, fwhm = np.array(s.centres_nm, float), np.array(s.fwhm_nm, float)
        if (fwhm <= 0).any():
            raise ValueError("the filter widths must be positive")
        return np.exp(-4.0 * np.log(2.0) * ((lam[:, None] - centre) / fwhm) ** 2)
    if s.spectral == "csv" or s.spectral in MODIS_RESPONSE_FILES:
        if s.spectral in MODIS_RESPONSE_FILES:
            csv_file = MODIS_RESPONSE_FILES[s.spectral]
        elif not s.response_csv:
            raise ValueError("choose a response file")
        else:
            csv_file = s.response_csv
        nm, columns = load_response_csv(csv_file)
        return np.stack([np.interp(lam, nm, columns[:, k], left=0.0, right=0.0) for k in range(3)], axis=1)
    raise ValueError(f"no response for {s.spectral!r}")


def channels_from_wavelengths(loaded: Loaded, response: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The three channels rebuilt from the per-wavelength reflectance with `response`:
    reflectance = sum(R E rho dl) / sum(R E dl), the same definition the simulation uses.
    Also the channels' mean solar irradiance, sum(R E dl) / sum(R dl). Returns (h, w, 3), (3,)."""
    weight, total, dl = channel_weights(loaded, response)
    return weighted_channels(loaded.spectral, weight), total / (response * dl).sum(axis=0)


def weighted_channels(values: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """sum(value x weight) / sum(weight) over the last axis of `values` (..., n_wavelengths) for each column
    of `weight` (n_wavelengths, n_channels) -> (..., n_channels). A wavelength that got no photon in a pixel
    is NaN in the render's data (a very low sampling probability, few photons): it is left out of that
    pixel's weighted mean."""
    valid = np.isfinite(values)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(valid, values, 0.0) @ weight / (valid.astype(float) @ weight)


def channel_weights(loaded: Loaded, response: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(weight (n_wavelengths, 3), its sum per channel, the bin widths (n_wavelengths, 1)): a channel's
    value is sum(value at each wavelength x weight) / sum(weight), weight = response x solar x bin width."""
    dl = loaded.bin_width_nm[:, None]
    weight = response * loaded.solar[:, None] * dl
    total = weight.sum(axis=0)
    if (total <= 0).any():
        one = CHANNEL_NAMES[loaded.single_index] if loaded.single_index is not None else "camera"
        names = CHANNEL_NAMES if len(total) == 3 else (one,)
        empty = [names[k] for k in np.where(total <= 0)[0]]
        raise ValueError(f"the {', '.join(empty)} response is zero over all the simulated wavelengths")
    return weight, total, dl


def quantity_unit(s: Settings) -> str:
    """The unit of the values base_channels() gives for `s`."""
    if s.quantity == "reflectance":
        return "reflectance"
    if s.quantity == "dn":
        return "DN"
    return "W m-2 sr-1" if s.spectral == "cloudct" else "W m-2 um-1 sr-1"


def base_channels(loaded: Loaded, s: Settings, integrated: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
    """Step 1 and 2: the channels as reflectance, radiance or DN, (h, w, 3) in blue, green, red
    order, and the channels' mean solar irradiance (None when it is not known).

    The CloudCT camera's radiance is the INTEGRAL of L(lambda) x response over the wavelengths, in W m-2 sr-1
    (the response is normalised to its peak, not to its area), and its DN is G_eff x that x the exposure
    time; the other responses give the response-weighted MEAN radiance, in W m-2 um-1 sr-1. With
    `integrated` False the CloudCT camera also gives that mean (the curve the spectrum plot's bars need)."""
    cloudct = s.spectral == "cloudct"
    if s.quantity == "dn" and not cloudct:
        raise ValueError(
            "DN is the CloudCT camera's signal: choose 'CloudCT camera response' in group 2 "
            "(it needs the per-wavelength data of the grid band set)"
        )
    if s.quantity == "dn" and not (s.responsivity > 0 and s.exposure_s > 0):
        raise ValueError("the responsivity and the exposure time must be positive")
    if s.spectral == "saved":
        reflectance, irradiance = loaded.reflectance, loaded.irradiance
        if reflectance.shape[-1] == 1:  # a single wavelength: gray
            reflectance = np.repeat(reflectance, 3, axis=-1)
            irradiance = None if irradiance is None else np.repeat(irradiance[:1], 3)
        elif reflectance.shape[-1] != 3:
            raise ValueError(f"expected 1 or 3 channels, the file has {reflectance.shape[-1]}")
    else:
        if not loaded.has_spectral:
            raise ValueError(
                "this result has no <name>_spectral.npz (it holds the reflectance at every wavelength "
                "and is written only for the grid band set): choose 'As simulated'"
            )
        response = response_matrix(loaded, s)
        reflectance, irradiance = channels_from_wavelengths(loaded, response)
        if cloudct and integrated:  # the irradiance over the response: sum(R E dl), in W m-2
            irradiance = irradiance * (response * loaded.bin_width_nm[:, None]).sum(axis=0) * MW_TO_W
        if reflectance.shape[-1] == 1:  # a single-channel run, or the monochrome CloudCT camera: gray
            reflectance, irradiance = np.repeat(reflectance, 3, axis=-1), np.repeat(irradiance, 3)
    if s.quantity == "reflectance":
        return reflectance, irradiance
    if s.quantity not in ("radiance", "dn"):
        raise ValueError(f"unknown quantity {s.quantity!r}")
    if irradiance is None:
        raise ValueError(
            "radiance needs the solar irradiance: <name>_radiance.json (run_bomex_headless.py writes it), "
            "<name>_spectral.npz next to the file, or a file name that gives the band set"
        )
    if not 0.0 <= s.sza_deg < 90.0:
        raise ValueError("the sun zenith angle must be in [0, 90) degrees for radiance")
    radiance = reflectance * irradiance * np.cos(np.radians(s.sza_deg)) / np.pi
    if s.quantity == "dn":
        radiance = radiance * s.responsivity * s.exposure_s
    return radiance, irradiance


def to_linear(base: np.ndarray, irradiance: np.ndarray | None, s: Settings) -> np.ndarray:
    """Step 3: white balance, gains, normalisation, gain and saturation. Still linear."""
    x = base.copy()
    if s.white_balance == "sun" and s.quantity != "reflectance" and irradiance is not None:
        x = x / irradiance * irradiance.mean()  # the sun's colour divided out
    elif s.white_balance == "gray":
        mean = x.reshape(-1, 3).mean(axis=0)
        x = x * np.where(mean > 0, mean.mean() / np.maximum(mean, 1e-30), 1.0)
    x = x * np.array(s.gains, float)
    if s.normalisation == "percentile":
        level = np.percentile(x, s.percentile)
        if level > 0:
            x = x / level
    x = x * s.gain
    if s.saturation != 1.0:
        luminance = (x * LUMINANCE_BGR).sum(axis=-1, keepdims=True)
        x = luminance + s.saturation * (x - luminance)
    return np.clip(x, 0.0, None)


def to_display(linear: np.ndarray, s: Settings) -> np.ndarray:
    """Step 4 and 5: the channel to show, highlight handling and the tone curve; an
    (h, w, 3) uint8 image in red, green, blue order."""
    x = linear
    if s.view != "colour":
        x = np.repeat(x[..., CHANNEL_NAMES.index(s.view)][..., None], 3, axis=-1)
    if s.highlights == "rolloff":
        x = x / (1.0 + x)
    x = np.clip(x, 0.0, 1.0)
    if s.tone == "gamma":
        x = x ** (1.0 / s.gamma)
    elif s.tone == "srgb":
        x = np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1.0 / 2.4) - 0.055)
    return (np.clip(x, 0.0, 1.0)[..., ::-1] * 255.0 + 0.5).astype(np.uint8)  # B, G, R -> R, G, B


@dataclass
class Legend:
    """The colour bar under the image: `ramp` (n, 3) uint8 RGB from `low` to `high`, and its caption."""

    ramp: np.ndarray
    low: float
    high: float
    label: str


def image_legend(s: Settings) -> Legend:
    """The grey ramp the image's values go through: the linear value (after the processing) from 0 to
    the white level, shown with the highlight handling and tone curve in `s`. Values above the right
    end look like it (clip) or approach white (rolloff)."""
    top = 1.0 if s.highlights == "clip" else 4.0
    x = np.linspace(0.0, top, 256)
    ramp = to_display(np.repeat(x[None, :, None], 3, axis=-1), replace(s, view="colour"))[0]
    unit = quantity_unit(s)
    after = ", after the processing" if (s.normalisation == "percentile" or s.gain != 1.0 or s.saturation != 1.0 or s.white_balance != "none") else ""
    where = "" if s.view == "colour" else f" ({s.view} channel)"
    return Legend(ramp, 0.0, top, f"Linear value, {unit}{after}{where}: grey = displayed brightness")


def statistics(linear: np.ndarray, unit: str) -> str:
    """Per channel: the linear values the display curve is applied to."""
    lines = [f"Linear values before the tone curve ({unit}):"]
    for k, name in enumerate(CHANNEL_NAMES):
        v = linear[..., k]
        lines.append(
            f"  {name:5s} min {v.min():.4g}  median {np.median(v):.4g}  p99 {np.percentile(v, 99):.4g}  "
            f"max {v.max():.4g}  above 1: {100.0 * (v > 1.0).mean():.1f}%"
        )
    return "\n".join(lines)


@dataclass
class Result:
    base: np.ndarray  # the channels as reflectance or radiance
    linear: np.ndarray  # after white balance, gains, normalisation, gain, saturation
    image: np.ndarray  # the displayed (h, w, 3) uint8, red, green, blue
    text: str
    legend: "Legend | None" = None


def process(loaded: Loaded, s: Settings, base: tuple | None = None) -> Result:
    """The whole chain. `base` is the result of base_channels() when the caller already has it."""
    base_values, irradiance = base if base is not None else base_channels(loaded, s)
    linear = to_linear(base_values, irradiance, s)
    unit = quantity_unit(s)
    if s.normalisation == "percentile" or s.gain != 1.0 or s.saturation != 1.0 or s.white_balance != "none":
        unit += ", after the processing"
    text = statistics(linear, unit)
    if s.spectral == "cloudct" and s.quantity != "reflectance" and loaded.has_spectral:
        text += (
            "\nCloudCT camera: L_eff = integral of L x eta_hat, in W m-2 sr-1"
            + (f"; DN = {s.responsivity:g} x L_eff x {s.exposure_s:g} s" if s.quantity == "dn" else "")
            + f". {100.0 * cloudct_coverage(loaded):.0f} % of the camera's response lies in the simulated wavelengths"
            + (" (the rest is missing from L_eff)." if cloudct_coverage(loaded) < 0.99 else ".")
        )
    return Result(base_values, linear, to_display(linear, s), text, image_legend(s))


# ----------------------------------------------------------------------------
# polarization
# ----------------------------------------------------------------------------

NO_POLARIZATION = (
    "this result has no polarization data: <name>_stokes.npy beside the file "
    "(every render of the current programs writes it; an older render may not have it)"
)


def stokes_to_polarization(stokes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(4, ...) Stokes parameters I, Q, U, V -> (DoLP, AoLP in degrees), each of shape (...).
    DoLP = sqrt(Q^2 + U^2) / I (0 to 1); AoLP = atan2(U, Q) / 2 in [-90, 90), measured from the
    meridian plane of the viewing direction (the same as playground_core.stokes_to_polarization)."""
    intensity, q, u = stokes[0], stokes[1], stokes[2]
    dolp = np.where(intensity > 0, np.hypot(q, u) / np.where(intensity > 0, intensity, 1.0), 0.0)
    aolp = (np.degrees(0.5 * np.arctan2(u, q)) + 90.0) % 180.0 - 90.0
    return dolp, aolp


def channel_stokes(loaded: Loaded, s: Settings) -> np.ndarray:
    """The Stokes parameters I, Q, U, V of the three channels, (4, h, w, 3) in blue, green, red order:
    as saved, or (s.spectral gaussian / csv) rebuilt from <name>_stokes_spectral.npz with the same
    weights as the reflectance channels (the Stokes parameters are linear, so they combine alike)."""
    if loaded.stokes is None:
        raise ValueError(NO_POLARIZATION)
    if s.spectral == "saved":
        stokes = loaded.stokes
        if stokes.shape[-1] == 1:  # a single wavelength: gray
            stokes = np.repeat(stokes, 3, axis=-1)
        elif stokes.shape[-1] != 3:
            raise ValueError(f"expected 1 or 3 channels, the file has {stokes.shape[-1]}")
        return stokes
    if (
        loaded.stokes_spectral is None
        or not loaded.has_spectral
        or len(loaded.stokes_wavelength_nm) != len(loaded.wavelength_nm)
    ):
        raise ValueError(
            "the polarization of rebuilt channels needs <name>_stokes_spectral.npz and <name>_spectral.npz "
            "(grid or grid_modis band set): choose 'As simulated'"
        )
    weight, total, _ = channel_weights(loaded, response_matrix(loaded, s))
    stokes = weighted_channels(loaded.stokes_spectral, weight)  # (4, 1 or h, 1 or w, channels)
    if stokes.shape[-1] == 1:  # a single-channel run: gray
        stokes = np.repeat(stokes, 3, axis=-1)
    if stokes.shape[1:3] != loaded.reflectance.shape[:2]:  # the clear sky: one pixel stands for the image
        stokes = np.broadcast_to(stokes, (4, *loaded.reflectance.shape[:2], 3)).copy()
    return stokes


def polarization_statistics(stokes: np.ndarray, what: str, channel: str) -> str:
    """Per channel: the range of the DoLP over the image, and the AoLP of the mean Stokes parameters."""
    dolp, _ = stokes_to_polarization(stokes)
    _, aolp_mean = stokes_to_polarization(stokes.reshape(4, -1, stokes.shape[-1]).mean(axis=1))
    lines = [f"Linear polarization ({'DoLP' if what == 'dolp' else 'AoLP'} map of the {channel} channel):"]
    for k, name in enumerate(CHANNEL_NAMES):
        v = 100.0 * dolp[..., k]
        lines.append(
            f"  {name:5s} DoLP min {v.min():.3g} %  median {np.median(v):.3g} %  max {v.max():.3g} %   "
            f"AoLP of the mean {aolp_mean[k]:+.1f} deg"
        )
    return "\n".join(lines)


def polarization_image(loaded: Loaded, s: Settings) -> tuple[np.ndarray, str, Legend]:
    """The map s.map ("dolp": viridis, 0 to the full scale; "aolp": twilight, -90 to 90 deg, cyclic) of
    the channel s.pol_channel, as an (h, w, 3) uint8 image in red, green, blue order, and a text."""
    from matplotlib import colormaps

    stokes = channel_stokes(loaded, s)
    dolp, aolp = stokes_to_polarization(stokes)
    k = CHANNEL_NAMES.index(s.pol_channel)
    if s.map == "dolp":
        top = s.dolp_max_pct / 100.0 if s.dolp_max_pct > 0 else max(float(dolp[..., k].max()), 1e-9)
        rgba = colormaps["viridis"](np.clip(dolp[..., k] / top, 0.0, 1.0))
        scale = f"viridis: dark = 0 %, bright = {100.0 * top:.3g} %"
        bar = np.linspace(0.0, 1.0, 256)
        legend = Legend((colormaps["viridis"](bar)[:, :3] * 255.0 + 0.5).astype(np.uint8), 0.0, 100.0 * top,
                        f"DoLP (%), {s.pol_channel} channel")
    elif s.map == "aolp":
        rgba = colormaps["twilight"]((aolp[..., k] + 90.0) / 180.0)
        scale = "twilight, cyclic: -90 and +90 deg look the same"
        bar = np.linspace(0.0, 1.0, 256)
        legend = Legend((colormaps["twilight"](bar)[:, :3] * 255.0 + 0.5).astype(np.uint8), -90.0, 90.0,
                        f"AoLP (deg), {s.pol_channel} channel (cyclic)")
    else:
        raise ValueError(f"unknown map {s.map!r}")
    image = (rgba[..., :3] * 255.0 + 0.5).astype(np.uint8)
    return image, polarization_statistics(stokes, s.map, s.pol_channel) + f"\n  {scale}", legend


def polarization_array(loaded: Loaded, s: Settings) -> np.ndarray:
    """(2, h, w, 3): DoLP (0 to 1) and AoLP (degrees) of the three channels, blue, green, red."""
    return np.stack(stokes_to_polarization(channel_stokes(loaded, s)))


# ----------------------------------------------------------------------------
# reflectance (or radiance, or polarization) against wavelength
# ----------------------------------------------------------------------------

# The wavelengths of the channels, for a result without per-wavelength data.
BAND_RANGES_NM = {
    "modis": ((456.0, 474.0), (545.0, 565.0), (625.0, 667.0)),  # MODIS b03, b04, b01
    "rgb": ((440.0, 440.0), (550.0, 550.0), (670.0, 670.0)),
}
CHANNEL_COLOURS = ("tab:blue", "tab:green", "tab:red")


def channel_ranges(loaded: Loaded) -> list[tuple[float, float]] | None:
    """(low, high) wavelengths of each saved channel: from the channel names the files give
    ("MODIS b03 (456-474 nm)", "440 nm"), else from the band set in the file name."""
    n = loaded.reflectance.shape[-1]
    found = []
    for label in loaded.channel_labels[:n]:
        match = re.search(r"(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*nm", label)
        if match:
            found.append((float(match.group(1)), float(match.group(2))))
            continue
        match = re.search(r"(\d+(?:\.\d+)?)\s*nm", label)
        if not match:
            break
        found.append((float(match.group(1)), float(match.group(1))))
    if len(found) == n:
        return found
    band_set, index = name_band_info(loaded.path)
    if band_set is None:
        return None
    ranges = BAND_RANGES_NM[band_set]
    if n == 3 and index is None:
        return list(ranges)
    if n == 1 and index is not None:
        return [ranges[index]]
    return None


@dataclass
class SpectrumData:
    """What the spectrum plot shows."""

    wavelength_nm: np.ndarray | None  # the simulated wavelengths (None: no per-wavelength data)
    values: np.ndarray | None  # reflectance or spectral radiance at each of them
    response: np.ndarray | None  # (n_wavelengths, n_channels): the camera response, when known
    bars: list  # per channel: (low nm, high nm, value, name, colour)
    ylabel: str
    where: str
    ylim: tuple | None = None
    bin_width_nm: np.ndarray | None = None  # the width of each wavelength's bin, to draw the response as blocks


def _pick(loaded: Loaded, pixel: tuple[int, int] | None):
    """(index into a (h, w, ...) array, text): the whole image, or the pixel (x, y)."""
    height, width = loaded.reflectance.shape[:2]
    if pixel is None:
        return (slice(None), slice(None)), "mean over the image"
    x, y = min(max(int(pixel[0]), 0), width - 1), min(max(int(pixel[1]), 0), height - 1)
    return (y, x), f"pixel x={x}, y={y}"


def spectrum_plot_data(
    loaded: Loaded, s: Settings, pixel: tuple[int, int] | None = None, base: tuple | None = None
) -> SpectrumData:
    """The reflectance (or, with s.quantity radiance, the spectral radiance in W m-2 um-1 sr-1) at
    every simulated wavelength, averaged over the image or at one pixel (x, y), and each channel's
    value over the wavelengths it covers. With s.plot "dolp" or "aolp" the degree (%) or angle (deg) of
    linear polarization instead. `base` is base_channels() when the caller has it."""
    if s.plot in ("dolp", "aolp"):
        return polarization_plot_data(loaded, s, pixel)
    channels, _ = base if base is not None else base_channels(loaded, s)
    pick, where = _pick(loaded, pixel)
    radiance = s.quantity != "reflectance"
    n_bars = n_made_channels(loaded, s)
    values = channels[pick].reshape(-1, channels.shape[-1]).mean(axis=0)[:n_bars]
    if s.spectral == "cloudct" and radiance:
        # The curve is the spectral radiance, so the bar is the response-weighted mean of it (W m-2 um-1 sr-1);
        # the camera's own L_eff or DN is said in the title.
        mean_radiance, _ = base_channels(loaded, replace(s, quantity="radiance"), integrated=False)
        bar = mean_radiance[pick].reshape(-1, 3).mean(axis=0)[:n_bars]
        where += f"; CloudCT camera: {values[0]:.4g} {quantity_unit(s)}"
        values = bar
    cos_sza_over_pi = np.cos(np.radians(s.sza_deg)) / np.pi

    lam = vals = None
    if loaded.has_spectral:
        lam = loaded.wavelength_nm
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")  # a wavelength with no photon in any selected pixel stays NaN
            vals = np.nanmean(loaded.spectral[pick].reshape(-1, len(lam)), axis=0)
        if radiance:
            vals = vals * loaded.solar * cos_sza_over_pi
    ylabel = "Spectral radiance\n(W m-2 um-1 sr-1)" if radiance else "Reflectance"
    return _spectrum_layout(loaded, s, n_bars, values, lam, vals, ylabel, where)


def polarization_plot_data(loaded: Loaded, s: Settings, pixel: tuple[int, int] | None = None) -> SpectrumData:
    """DoLP (%) or AoLP (deg) against wavelength at one pixel (or the image mean), from the Stokes
    parameters at every simulated wavelength when the result has them, and of each channel."""
    stokes = channel_stokes(loaded, s)
    pick, where = _pick(loaded, pixel)
    n_bars = n_made_channels(loaded, s)
    dolp_pct = s.plot == "dolp"

    def value(stokes_mean):  # (4, n) mean Stokes parameters -> DoLP in % or AoLP in degrees
        dolp, aolp = stokes_to_polarization(stokes_mean)
        return 100.0 * dolp if dolp_pct else aolp

    # the polarization of the mean Stokes parameters (the mean of AoLP itself would be wrong: it is cyclic)
    values = value(stokes[:, pick[0], pick[1]].reshape(4, -1, stokes.shape[-1]).mean(axis=1))[:n_bars]
    lam = vals = None
    if loaded.stokes_spectral is not None:
        lam = loaded.stokes_wavelength_nm
        cube = loaded.stokes_spectral
        if cube.shape[1:3] == (1, 1):
            vals = value(cube[:, 0, 0, :])
        else:
            vals = value(cube[:, pick[0], pick[1]].reshape(4, -1, cube.shape[-1]).mean(axis=1))
    ylabel = "DoLP (%)" if dolp_pct else "AoLP (deg)"
    data = _spectrum_layout(loaded, s, n_bars, values, lam, vals, ylabel, where)
    data.ylim = (0.0, None) if dolp_pct else (-95.0, 95.0)
    return data


def _spectrum_layout(loaded, s, n_bars, values, lam, vals, ylabel, where) -> SpectrumData:
    """What the plot shows around the values: each channel's wavelength range (a bar of its value) and,
    where the per-wavelength data allow, the camera response behind the curve."""
    response = None
    if lam is not None and loaded.has_spectral and len(lam) == len(loaded.wavelength_nm):
        if s.spectral != "saved":
            response = response_matrix(loaded, s)
        elif loaded.group is not None and loaded.group.max() + 1 == n_bars:
            response = np.stack([(loaded.group == k).astype(float) for k in range(n_bars)], axis=1)

    ranges = None
    if response is None:
        ranges = channel_ranges(loaded) if s.spectral == "saved" else None
    names = CHANNEL_NAMES if n_bars == 3 else (tuple(loaded.channel_labels[:n_bars]) or ("channel",) * n_bars)
    if s.spectral == "cloudct":
        names = ("CloudCT camera",)
    colours = CHANNEL_COLOURS if n_bars == 3 else ("black",) * n_bars
    bars = []
    for k in range(n_bars):
        if response is not None:  # where the channel responds at least half as much as at its peak
            inside = response[:, k] >= 0.5 * response[:, k].max()
            half = 0.5 * loaded.bin_width_nm[inside]
            low, high = (lam[inside] - half).min(), (lam[inside] + half).max()
        elif ranges is not None:
            low, high = ranges[k]
        else:
            continue
        bars.append((float(low), float(high), float(values[k]), names[k], colours[k]))
    return SpectrumData(lam, vals, response, bars, ylabel, where, bin_width_nm=loaded.bin_width_nm)


def draw_spectrum(figure, data: SpectrumData) -> None:
    """Draw `data` on a matplotlib Figure (no pyplot, so it can sit in a window)."""
    figure.clear()
    ax = figure.add_subplot(111)
    if data.wavelength_nm is not None:
        if data.response is not None:  # the camera response behind the spectrum, on its own axis
            ax2 = ax.twinx()
            for k in range(data.response.shape[1]):
                colour = CHANNEL_COLOURS[k] if data.response.shape[1] == 3 else "black"
                height = data.response[:, k] / data.response.max()
                if data.bin_width_nm is not None:  # one flat block per bin, with its true edges
                    ax2.bar(data.wavelength_nm, height, width=data.bin_width_nm, color=colour, alpha=0.12, lw=0)
                else:
                    ax2.fill_between(data.wavelength_nm, height, color=colour, alpha=0.12, lw=0)
            ax2.set_ylim(0, 3.0)  # the response fills the lower third
            ax2.set_yticks([])
            ax2.set_zorder(0)
            ax.set_zorder(1)
            ax.patch.set_visible(False)
        ax.plot(data.wavelength_nm, data.values, "o-", color="0.35", ms=3, lw=1, label="simulated wavelengths")
    for low, high, value, name, colour in data.bars:
        if high > low:
            ax.hlines(value, low, high, colors=colour, lw=4, label=f"{name}: {value:.4g}")
        else:
            ax.plot([low], [value], "s", color=colour, ms=8, label=f"{name}: {value:.4g}")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel(data.ylabel)
    ax.set_title(data.where, fontsize=9)
    if data.ylim is not None:
        ax.set_ylim(*data.ylim)
    ax.grid(alpha=0.3)
    if data.wavelength_nm is not None or data.bars:
        ax.legend(fontsize=8, loc="best")
    figure.tight_layout()


# ----------------------------------------------------------------------------
# the window
# ----------------------------------------------------------------------------


def run_window(initial: str | None = None, run: bool = True):
    from PyQt6 import QtCore, QtGui, QtWidgets  # before matplotlib's Qt backend, which follows it

    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
    from matplotlib.figure import Figure

    class ImageView(QtWidgets.QLabel):
        """The image, scaled to the widget with hard pixel edges; a click gives the pixel."""

        clicked = QtCore.pyqtSignal(int, int)  # the pixel's x (column) and y (row from the top)

        def mousePressEvent(self, event):
            pixmap = self.pixmap()
            if self._image is None or pixmap is None or pixmap.isNull():
                return
            x = (event.position().x() - (self.width() - pixmap.width()) / 2) / pixmap.width()
            y = (event.position().y() - (self.height() - pixmap.height()) / 2) / pixmap.height()
            if 0 <= x < 1 and 0 <= y < 1:
                self.clicked.emit(int(x * self._image.width()), int(y * self._image.height()))

        def __init__(self):
            super().__init__()
            self.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            self.setMinimumSize(320, 320)
            self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Ignored)
            self._image = None
            self._marker = None

        def show_array(self, rgb: np.ndarray):
            rgb = np.ascontiguousarray(rgb)
            h, w, _ = rgb.shape
            self._image = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format.Format_RGB888).copy()
            self._refresh()

        def set_marker(self, pixel: tuple[int, int] | None):
            """Outline this pixel (x, y) on the image, e.g. the one a plot is for; None: no outline."""
            self._marker = pixel
            self._refresh()

        def _refresh(self):
            if self._image is None:
                return
            pixmap = QtGui.QPixmap.fromImage(self._image).scaled(
                self.size(),
                QtCore.Qt.AspectRatioMode.KeepAspectRatio,
                QtCore.Qt.TransformationMode.FastTransformation,
            )
            if self._marker is not None:
                x, y = self._marker
                if 0 <= x < self._image.width() and 0 <= y < self._image.height():
                    scale = pixmap.width() / self._image.width()
                    size = max(scale, 9.0)  # a pixel drawn smaller than this would be hard to see
                    left, top = (x + 0.5) * scale - size / 2, (y + 0.5) * scale - size / 2
                    painter = QtGui.QPainter(pixmap)
                    for colour, width in ((QtCore.Qt.GlobalColor.black, 4), (QtCore.Qt.GlobalColor.white, 2)):
                        painter.setPen(QtGui.QPen(colour, width))
                        painter.drawRect(QtCore.QRectF(left, top, size, size))
                    painter.end()
            self.setPixmap(pixmap)

        def resizeEvent(self, event):
            super().resizeEvent(event)
            self._refresh()

    class Window(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("Process a raw SMART-G result")
            self.loaded: Loaded | None = None
            self.result: Result | None = None
            self._cache_key = None
            self._cache = None
            self._building = True

            self.view = ImageView()
            self.info = QtWidgets.QLabel("Open a raw result (<name>.npy).")
            self.info.setWordWrap(True)
            self.message = QtWidgets.QLabel("")
            self.message.setWordWrap(True)
            self.message.setStyleSheet("color: #b00020;")
            self.stats = QtWidgets.QPlainTextEdit()
            self.stats.setReadOnly(True)
            self.stats.setMaximumHeight(110)
            self.stats.setFont(QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.SystemFont.FixedFont))
            self.stats.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)

            controls = QtWidgets.QWidget()
            column = QtWidgets.QVBoxLayout(controls)
            column.addLayout(self._buttons())
            column.addWidget(self.info)
            column.addWidget(self._group_values())
            column.addWidget(self._group_channels())
            column.addWidget(self._group_colour())
            column.addWidget(self._group_display())
            column.addWidget(self._group_plot())
            column.addWidget(self._group_polarization())
            column.addWidget(self.message)
            column.addStretch(1)
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(controls)
            natural = controls.sizeHint().width()  # the controls' own width, so nothing is clipped
            scroll.setMinimumWidth(natural + 40)  # room for the scroll bar and the frame
            scroll.setMaximumWidth(natural + 100)

            self.figure = Figure(figsize=(6, 3), layout=None)
            self.canvas = FigureCanvasQTAgg(self.figure)
            self.plot_box = QtWidgets.QWidget()
            plot_layout = QtWidgets.QVBoxLayout(self.plot_box)
            plot_layout.setContentsMargins(0, 0, 0, 0)
            plot_layout.addWidget(NavigationToolbar2QT(self.canvas, self.plot_box))
            plot_layout.addWidget(self.canvas, 1)
            self.plot_box.setMinimumHeight(220)
            self.view.clicked.connect(self._pixel_clicked)

            right = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
            self.legend_figure = Figure(figsize=(6, 0.8), layout=None)
            self.legend_canvas = FigureCanvasQTAgg(self.legend_figure)
            self.legend_canvas.setFixedHeight(64)
            view_box = QtWidgets.QWidget()
            view_layout = QtWidgets.QVBoxLayout(view_box)
            view_layout.setContentsMargins(0, 0, 0, 0)
            view_layout.setSpacing(0)
            view_layout.addWidget(self.view, 1)
            view_layout.addWidget(self.legend_canvas)
            right.addWidget(view_box)
            right.addWidget(self.plot_box)
            right.addWidget(self.stats)
            right.setStretchFactor(0, 3)
            right.setStretchFactor(1, 2)
            right.setCollapsible(0, False)
            self.show_plot.setChecked(True)  # the plot is on from the start; click a pixel to choose it
            splitter = QtWidgets.QSplitter()
            splitter.addWidget(scroll)
            splitter.addWidget(right)
            splitter.setStretchFactor(1, 1)
            self.setCentralWidget(splitter)
            self.resize(1280, 780)
            self._building = False
            self._sync_enabled()

        # ---- building the controls
        def _buttons(self):
            row = QtWidgets.QHBoxLayout()
            for text, slot in (
                ("Open...", self.open_dialog),
                ("Save image...", self.save_image),
                ("Save linear data...", self.save_linear),
                ("Save settings...", self.save_settings),
            ):
                button = QtWidgets.QPushButton(text)
                button.clicked.connect(slot)
                row.addWidget(button)
            return row

        def _combo(self, items, tooltip=""):
            box = QtWidgets.QComboBox()
            for value, label in items:
                box.addItem(label, value)
            box.setToolTip(tooltip)
            box.currentIndexChanged.connect(self._changed)
            return box

        def _spin(self, low, high, value, step, decimals=2, suffix="", tooltip=""):
            box = QtWidgets.QDoubleSpinBox()
            box.setRange(low, high)
            box.setDecimals(decimals)
            box.setSingleStep(step)
            box.setValue(value)
            box.setSuffix(suffix)
            box.setToolTip(tooltip)
            box.valueChanged.connect(self._changed)
            return box

        def _group_values(self):
            group = QtWidgets.QGroupBox("1  What the numbers are")
            form = QtWidgets.QFormLayout(group)
            self.quantity = self._combo(
                [
                    ("reflectance", "Reflectance (the sun's spectrum divided out)"),
                    ("radiance", "Radiance (the sun's spectrum kept)"),
                    ("dn", "Digital number (CloudCT camera: G_eff x L_eff x exposure)"),
                ],
                "Radiance = reflectance x solar irradiance x cos(sun zenith) / pi, in W m-2 um-1 sr-1 (with the "
                "CloudCT camera response: the integral of L x eta_hat, in W m-2 sr-1). DN needs the CloudCT camera "
                "response: DN = G_eff x L_eff x exposure time.",
            )
            self.sza = self._spin(0, 89.9, 0, 1, 2, " deg", "Sun zenith angle, for the radiance")
            self.responsivity = self._spin(
                0.001, 1e6, CLOUDCT_G_EFF, 1, 3, " DN/(W m-2 sr-1 s)",
                "The CloudCT camera's radiometric responsivity G_eff, from its radiometric calibration",
            )
            self.exposure = self._spin(1e-6, 1e4, Settings.exposure_s, 0.01, 6, " s", "The exposure time the simulation assumes")
            form.addRow("Quantity", self.quantity)
            form.addRow("Sun zenith", self.sza)
            form.addRow("Responsivity G_eff", self.responsivity)
            form.addRow("Exposure time", self.exposure)
            return group

        def _group_channels(self):
            group = QtWidgets.QGroupBox("2  How the channels are made")
            form = QtWidgets.QFormLayout(group)
            self.spectral = self._combo(
                [
                    ("saved", "As simulated"),
                    ("gaussian", "Gaussian filters (needs the wavelengths)"),
                    ("csv", "Response file (needs the wavelengths)"),
                    ("modis_terra", "MODIS Terra response, b03 b04 b01 (needs the wavelengths)"),
                    ("modis_aqua", "MODIS Aqua response, b03 b04 b01 (needs the wavelengths)"),
                    ("cloudct", "CloudCT camera response, one gray channel (needs the wavelengths)"),
                ],
                "The wavelength options rebuild the channels from <name>_spectral.npz (grid band set only)",
            )
            form.addRow("Channels", self.spectral)
            self.centre, self.fwhm = [], []
            for k, name in enumerate(CHANNEL_NAMES):
                centre = self._spin(300, 1000, Settings.centres_nm[k], 5, 0, " nm")
                fwhm = self._spin(5, 500, Settings.fwhm_nm[k], 5, 0, " nm")
                self.centre.append(centre)
                self.fwhm.append(fwhm)
                row = QtWidgets.QHBoxLayout()
                row.addWidget(QtWidgets.QLabel("centre"))
                row.addWidget(centre)
                row.addWidget(QtWidgets.QLabel("width (FWHM)"))
                row.addWidget(fwhm)
                form.addRow(f"Filter {name}", row)
            self.csv_label = QtWidgets.QLabel("(none)")
            self.csv_button = QtWidgets.QPushButton("Choose...")
            self.csv_button.clicked.connect(self.choose_csv)
            self.csv_button.setToolTip("A CSV with the columns nm, blue, green, red (a header line is skipped)")
            row = QtWidgets.QHBoxLayout()
            row.addWidget(self.csv_button)
            row.addWidget(self.csv_label, 1)
            form.addRow("Response file", row)
            self.csv_path = ""
            return group

        def _group_colour(self):
            group = QtWidgets.QGroupBox("3  Colour")
            form = QtWidgets.QFormLayout(group)
            self.white_balance = self._combo(
                [
                    ("none", "None"),
                    ("sun", "Divide out the sun's colour (radiance only)"),
                    ("gray", "Gray world (equal channel means)"),
                ]
            )
            form.addRow("White balance", self.white_balance)
            self.gains = []
            row = QtWidgets.QHBoxLayout()
            for name in CHANNEL_NAMES:
                gain = self._spin(0.0, 10.0, 1.0, 0.05, 2)
                self.gains.append(gain)
                row.addWidget(QtWidgets.QLabel(name))
                row.addWidget(gain)
            form.addRow("Channel gains", row)
            self.normalisation = self._combo(
                [("none", "None"), ("percentile", "The percentile below becomes 1")],
                "Switching to radiance turns this on, since radiances are far above 1",
            )
            self.percentile = self._spin(50, 100, 99.5, 0.5, 1, " %")
            form.addRow("Normalise", self.normalisation)
            form.addRow("Percentile", self.percentile)
            self.gain = self._spin(0.001, 1000, 1.0, 0.1, 3)
            self.saturation = self._spin(0.0, 3.0, 1.0, 0.1, 2, tooltip="0: gray, 1: as is, above 1: more colour")
            form.addRow("Gain", self.gain)
            form.addRow("Saturation", self.saturation)
            return group

        def _group_display(self):
            group = QtWidgets.QGroupBox("4  Display")
            form = QtWidgets.QFormLayout(group)
            self.highlights = self._combo(
                [("clip", "Clip at 1"), ("rolloff", "Roll off, x / (1 + x)")],
                "Roll-off keeps bright clouds from saturating; it also darkens the midtones",
            )
            self.tone = self._combo([("gamma", "Gamma"), ("srgb", "sRGB"), ("linear", "Linear")])
            self.gamma = self._spin(0.5, 4.0, 1.8, 0.1, 2, tooltip="The playground's PNGs use 1.8")
            self.show_channel = self._combo([("colour", "Colour")] + [(n, n.capitalize() + " only") for n in CHANNEL_NAMES])
            form.addRow("Highlights", self.highlights)
            form.addRow("Tone curve", self.tone)
            form.addRow("Gamma", self.gamma)
            form.addRow("Show", self.show_channel)
            return group

        def _group_plot(self):
            group = QtWidgets.QGroupBox("5  Spectrum plot")
            form = QtWidgets.QFormLayout(group)
            self.show_plot = QtWidgets.QCheckBox("Plot reflectance (or radiance) against wavelength")
            self.show_plot.setToolTip(
                "For one pixel: click it in the image. The values at the simulated wavelengths (grid band "
                "set), with each channel's value over the wavelengths it covers. A wavelength that got few "
                "photons is noisy."
            )
            self.show_plot.toggled.connect(self._plot_toggled)
            self.pixel_x = QtWidgets.QSpinBox()
            self.pixel_y = QtWidgets.QSpinBox()
            for box in (self.pixel_x, self.pixel_y):
                box.setRange(0, 0)
                box.valueChanged.connect(self._changed)
            row = QtWidgets.QHBoxLayout()
            row.addWidget(QtWidgets.QLabel("x"))
            row.addWidget(self.pixel_x)
            row.addWidget(QtWidgets.QLabel("y"))
            row.addWidget(self.pixel_y)
            form.addRow(self.show_plot)
            form.addRow("Pixel (or click the image)", row)
            return group

        def _group_polarization(self):
            group = QtWidgets.QGroupBox("6  Polarization")
            form = QtWidgets.QFormLayout(group)
            self.pol_note = QtWidgets.QLabel("")
            self.pol_note.setWordWrap(True)
            self.pol_map = self._combo(
                [
                    ("image", "The image above"),
                    ("dolp", "Degree of linear polarization (DoLP)"),
                    ("aolp", "Angle of linear polarization (AoLP)"),
                ],
                "DoLP = sqrt(Q^2 + U^2) / I, 0 to 100 %. AoLP = atan2(U, Q) / 2, -90 to 90 deg, from the "
                "meridian plane of the viewing direction",
            )
            self.pol_channel = self._combo([(n, n.capitalize()) for n in CHANNEL_NAMES], "The channel the map is of")
            self.pol_channel.setCurrentIndex(CHANNEL_NAMES.index(Settings.pol_channel))
            self.dolp_max = self._spin(0, 100, 0.0, 5, 1, " %", "The DoLP map's full scale; 0: the largest value in the image")
            self.pol_plot = self._combo(
                [
                    ("spectrum", "Reflectance (or radiance)"),
                    ("dolp", "DoLP (%)"),
                    ("aolp", "AoLP (deg)"),
                ],
                "What the plot of group 5 shows against wavelength, for the clicked pixel",
            )
            form.addRow(self.pol_note)
            form.addRow("Show", self.pol_map)
            form.addRow("Of the channel", self.pol_channel)
            form.addRow("DoLP full scale", self.dolp_max)
            form.addRow("Plot (group 5)", self.pol_plot)
            self.pol_group = group
            return group

        def _plot_toggled(self, shown):
            self.plot_box.setVisible(shown)
            self._changed()

        def _pixel_clicked(self, x, y):
            if not self.show_plot.isChecked():
                return
            self._building = True
            self.pixel_x.setValue(x)
            self.pixel_y.setValue(y)
            self._building = False
            self._changed()

        def update_plot(self, s: Settings):
            """Redraw the spectrum plot; a problem is said in the message, the plot stays."""
            if not self.show_plot.isChecked() or self.loaded is None:
                self.view.set_marker(None)
                return
            pixel = (self.pixel_x.value(), self.pixel_y.value())
            self.view.set_marker(pixel)  # outline the pixel on the image
            try:
                data = spectrum_plot_data(self.loaded, s, pixel, self._cache)
            except Exception as exc:
                self.say(f"Spectrum plot: {exc}")
                return
            draw_spectrum(self.figure, data)
            self.canvas.draw_idle()

        # ---- reading the controls
        def settings(self) -> Settings:
            return Settings(
                quantity=self.quantity.currentData(),
                spectral=self.spectral.currentData(),
                centres_nm=tuple(b.value() for b in self.centre),
                fwhm_nm=tuple(b.value() for b in self.fwhm),
                response_csv=self.csv_path,
                sza_deg=self.sza.value(),
                responsivity=self.responsivity.value(),
                exposure_s=self.exposure.value(),
                white_balance=self.white_balance.currentData(),
                gains=tuple(b.value() for b in self.gains),
                normalisation=self.normalisation.currentData(),
                percentile=self.percentile.value(),
                gain=self.gain.value(),
                saturation=self.saturation.value(),
                highlights=self.highlights.currentData(),
                tone=self.tone.currentData(),
                gamma=self.gamma.value(),
                view=self.show_channel.currentData(),
                map=self.pol_map.currentData(),
                pol_channel=self.pol_channel.currentData(),
                dolp_max_pct=self.dolp_max.value(),
                plot=self.pol_plot.currentData(),
            )

        def _sync_enabled(self):
            spectral = self.spectral.currentData()
            for box in (*self.centre, *self.fwhm):
                box.setEnabled(spectral == "gaussian")
            self.csv_button.setEnabled(spectral == "csv")
            self.percentile.setEnabled(self.normalisation.currentData() == "percentile")
            self.gamma.setEnabled(self.tone.currentData() == "gamma")
            self.sza.setEnabled(self.quantity.currentData() != "reflectance")
            for box in (self.responsivity, self.exposure):
                box.setEnabled(self.quantity.currentData() == "dn")
            self.white_balance.model().item(1).setEnabled(self.quantity.currentData() != "reflectance")
            has_polarization = self.loaded is not None and self.loaded.has_polarization
            for box in (self.pol_map, self.pol_channel, self.pol_plot):
                box.setEnabled(has_polarization)
            self.dolp_max.setEnabled(has_polarization and self.pol_map.currentData() == "dolp")
            self.pol_channel.setEnabled(has_polarization and self.pol_map.currentData() != "image")

        def _changed(self, *_):
            if self._building:
                return
            sender = self.sender()
            if sender is self.quantity:  # radiances are far above 1: normalise them, reflectances not
                self._building = True
                not_reflectance = self.quantity.currentData() != "reflectance"
                self.normalisation.setCurrentIndex(self.normalisation.findData("percentile" if not_reflectance else "none"))
                if not not_reflectance and self.white_balance.currentData() == "sun":
                    self.white_balance.setCurrentIndex(0)
                if self.quantity.currentData() == "dn" and self.loaded is not None and self.loaded.has_spectral:
                    self.spectral.setCurrentIndex(self.spectral.findData("cloudct"))  # DN is that camera's signal
                self._building = False
            self._sync_enabled()
            self.refresh()

        # ---- loading and saving
        def choose_csv(self):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Camera response", "", "CSV (*.csv *.txt);;All files (*)")
            if path:
                self.csv_path = path
                self.csv_label.setText(Path(path).name)
                self.refresh()

        def open_dialog(self):
            start = str(self.loaded.path.parent) if self.loaded else ""
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Raw result", start, "NumPy arrays (*.npy)")
            if path:
                self.open_file(path)

        def open_file(self, path):
            try:
                loaded = load(path)
            except Exception as exc:  # a bad file should not close the window
                self.say(f"Could not open {path}: {exc}")
                return
            self.loaded = loaded
            self._cache_key = None
            parts = [f"{loaded.path.name.replace('_', '_' + chr(0x200b))}: {loaded.reflectance.shape[1]} x {loaded.reflectance.shape[0]} pixels, "
                     f"{loaded.reflectance.shape[2]} channel(s)."]
            parts.append("Sun zenith " + (f"{loaded.sza_deg:g} deg." if loaded.sza_deg is not None else "unknown."))
            if loaded.description is not None:
                d = loaded.description
                parts.append(
                    f"From the _radiance.json: band set {d.get('band_set', '?')}, sun azimuth {d.get('saa_deg', '?')} deg, "
                    f"time {d.get('time_utc', '?')}."
                )
            else:
                parts.append(
                    "WARNING: no _radiance.json beside the file, so the sun zenith (from the file name, rounded) "
                    "and the solar irradiance (a built-in table) are guesses; check them before trusting a radiance."
                )
            parts.append(
                "Solar irradiance per channel: " + ", ".join(f"{e:.0f}" for e in loaded.irradiance)
                + f" mW m-2 nm-1 ({loaded.irradiance_source})."
                if loaded.irradiance is not None
                else "No solar irradiance for the saved channels (no _radiance.json); the per-wavelength data may give it."
            )
            parts.append(
                f"Per-wavelength data: {len(loaded.wavelength_nm)} wavelengths, "
                f"{loaded.wavelength_nm.min():.0f}-{loaded.wavelength_nm.max():.0f} nm."
                if loaded.has_spectral
                else "No per-wavelength data (only the saved channels can be used)."
            )
            parts.append(
                "Polarization: Stokes I, Q, U, V of the channels"
                + (" and at every wavelength." if loaded.stokes_spectral is not None else ".")
                if loaded.has_polarization
                else "No polarization data (no _stokes.npy beside the file)."
            )
            self.info.setText("\n".join(parts))
            self.pol_note.setText(
                "Needs <name>_stokes.npy (written by the current renders)." if not loaded.has_polarization
                else "The map replaces the image above; DoLP and AoLP do not depend on the quantity or the gains."
            )
            self._building = True
            if not loaded.has_polarization:  # nothing to show: back to the image and the spectrum
                self.pol_map.setCurrentIndex(0)
                self.pol_plot.setCurrentIndex(0)
            if loaded.sza_deg is not None:
                self.sza.setValue(loaded.sza_deg)
            if not loaded.has_spectral:  # the wavelength options need <name>_spectral.npz
                self.spectral.setCurrentIndex(0)
            height, width = loaded.reflectance.shape[:2]
            self.pixel_x.setRange(0, width - 1)
            self.pixel_y.setRange(0, height - 1)
            self.pixel_x.setValue(width // 2)
            self.pixel_y.setValue(height // 2)
            self._sync_enabled()
            self._building = False
            self.setWindowTitle(f"Process a raw SMART-G result - {loaded.path.name}")
            self.refresh()

        def refresh(self):
            if self.loaded is None:
                return
            s = self.settings()
            try:
                key = (s.quantity, s.spectral, s.centres_nm, s.fwhm_nm, s.response_csv, s.sza_deg, s.responsivity, s.exposure_s)
                if key != self._cache_key:
                    self._cache, self._cache_key = base_channels(self.loaded, s), key
                self.result = process(self.loaded, s, self._cache)
            except Exception as exc:  # keep the last image and say what is wrong
                self._cache_key = None
                self.say(str(exc))
                return
            if s.map != "image":
                try:
                    self.result.image, self.result.text, self.result.legend = polarization_image(self.loaded, s)
                except Exception as exc:  # no polarization data, or rebuilt channels it cannot make
                    self.say(str(exc))
                    return
            self.say("")
            self.view.show_array(self.result.image)
            self.draw_legend(self.result.legend)
            self.stats.setPlainText(self.result.text)
            self.update_plot(s)

        def draw_legend(self, legend: Legend | None):
            """The colour bar under the image: what value each colour or brightness stands for."""
            fig = self.legend_figure
            fig.clear()
            if legend is not None:
                ax = fig.add_axes((0.04, 0.58, 0.92, 0.32))
                ax.imshow(legend.ramp[None, :, :], aspect="auto", extent=(legend.low, legend.high, 0, 1), interpolation="nearest")
                ax.set_yticks([])
                ax.tick_params(axis="x", labelsize=8)
                ax.set_xlabel(legend.label, fontsize=8)
            self.legend_canvas.draw_idle()

        def say(self, text: str, ok: bool = False):
            self.message.setStyleSheet("color: #1b5e20;" if ok else "color: #b00020;")
            self.message.setText(text)

        def _stem(self):
            return self.loaded.path.stem if self.loaded else "result"

        def save_image(self):
            if self.result is None:
                return
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save the image", str(self.loaded.path.with_name(f"{self._stem()}_processed.png")), "PNG (*.png)"
            )
            if path:
                Image.fromarray(self.result.image, mode="RGB").save(path)
                Path(path).with_suffix(".json").write_text(self._settings_json())
                self.say(f"Saved {Path(path).name} and its settings (.json).", ok=True)

        def save_linear(self):
            if self.result is None:
                return
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save the linear data", str(self.loaded.path.with_name(f"{self._stem()}_processed.npy")), "NumPy (*.npy)"
            )
            if path:
                np.save(path, self.result.linear)  # blue, green, red, like the input
                Path(path).with_suffix(".json").write_text(self._settings_json())
                text = f"Saved {Path(path).name} (blue, green, red) and its settings (.json)."
                if self.loaded.has_polarization:
                    try:
                        pol_path = Path(path).with_name(f"{Path(path).stem}_polarization.npy")
                        np.save(pol_path, polarization_array(self.loaded, self.settings()))
                        text += f" And {pol_path.name}: DoLP (0-1) and AoLP (deg), shape (2, height, width, 3)."
                    except Exception as exc:
                        text += f" The polarization was not saved: {exc}"
                self.say(text, ok=True)

        def _settings_json(self) -> str:
            return json.dumps({"source": str(self.loaded.path), **asdict(self.settings())}, indent=2)

        def save_settings(self):
            if self.loaded is None:
                return
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "Save the settings", str(self.loaded.path.with_name(f"{self._stem()}_settings.json")), "JSON (*.json)"
            )
            if path:
                Path(path).write_text(self._settings_json())

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = Window()
    window.show()
    if initial:
        window.open_file(initial)
    return app.exec() if run else (app, window)  # run=False: for tests, which drive the window themselves


if __name__ == "__main__":
    sys.exit(run_window(sys.argv[1] if len(sys.argv) > 1 else None))
