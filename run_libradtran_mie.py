"""Make a water-cloud Mie phase-function table with libRadtran's `mie` tool.

For each effective radius `r_eff` of a grid, the tool computes the scattering
phase matrix of a gamma distribution of spherical water droplets
(Wiscombe's MIEV0 code) and writes it as a netCDF file in libRadtran's
`wc.mie.cdf` layout, which SMART-G can read (`smartg.phase.read_phase_cdf`).
With the defaults this redoes the table behind SMART-G's `wc_sol.nc`: effective
radii 5-30 um and alpha = 7, i.e. effective variance 1/(alpha+3) = 0.1.

This only drives libRadtran; it needs a compiled libRadtran (Linux, see
https://www.libradtran.org/doku.php?id=download), pointed to by
--libradtran-dir or $LIBRADTRAN_DIR. It writes no files outside --workdir.

The `mie` keywords follow the libRadtran user guide (chapter 4 and
section 6.2). Try `--quick` first - a single radius and wavelength - to see
the run time and the output file name before the full table: the number of
Mie evaluations grows with the size parameter, so the 5-30 um table at
440-670 nm can take a long time.

Examples:
    python run_libradtran_mie.py --libradtran-dir ~/libRadtran-2.0.5 --quick
    python run_libradtran_mie.py --libradtran-dir ~/libRadtran-2.0.5 --dry-run
    python run_libradtran_mie.py --libradtran-dir ~/libRadtran-2.0.5 --veff 0.08
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

DEFAULT_ALPHA = 7.0  # gamma distribution parameter; effective variance = 1/(alpha+3)


def find_mie_binary(libradtran_dir: str | None, mie_bin: str | None) -> Path:
    """The `mie` executable: --mie-bin, else <libradtran dir>/bin/mie, else
    whatever is on PATH."""
    candidates = []
    if mie_bin:
        candidates.append(Path(mie_bin))
    if libradtran_dir:
        candidates.append(Path(libradtran_dir).expanduser() / "bin" / "mie")
    on_path = shutil.which("mie")
    if on_path:
        candidates.append(Path(on_path))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise FileNotFoundError(
        "libRadtran's `mie` executable was not found. Build libRadtran "
        "(./configure && make) and pass --libradtran-dir <install dir> "
        "(or set $LIBRADTRAN_DIR), or --mie-bin <path>. Looked at: "
        + ", ".join(map(str, candidates or ["(nothing given)"]))
    )


def build_input(args: argparse.Namespace) -> str:
    """The `mie` input file: one `keyword value` line each."""
    lines = [
        "mie_program MIEV0            # Wiscombe's code; the only one with phase matrices",
        "refrac water                 # refractive index of liquid water",
        f"r_eff {args.reff[0]:g} {args.reff[1]:g} {args.reff[2]:g}"
        "            # effective radius grid: start stop step [um]",
        f"distribution gamma {args.alpha:g}      # n(r) ~ r^alpha exp(-b r), r_eff = (alpha+3)/b",
        f"wavelength {args.wavelength[0]:g} {args.wavelength[1]:g}",
    ]
    if args.wavelength[0] != args.wavelength[1]:
        lines.append(f"wavelength_step {args.wavelength_step:g}")
    lines += [
        f"nstokes {args.nstokes}                # 4 = all phase matrix elements",
        f"nmom {args.nmom}                # Legendre terms computed",
        f"nmom_netcdf {args.nmom}         # ... and stored",
        f"nthetamax {args.nthetamax}            # max scattering angles per phase function",
        "output_user netcdf           # write the netCDF file uvspec/SMART-G read",
    ]
    lines += [extra for extra in args.extra]
    if args.verbose:
        lines.append("verbose")
    return "\n".join(lines) + "\n"


def run_mie(binary: Path, input_text: str, workdir: Path) -> tuple[list[Path], float]:
    """Run `mie < input` in `workdir`; return the netCDF files it created."""
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "mie.inp").write_text(input_text)
    before = {p: p.stat().st_mtime for p in workdir.iterdir()}

    start = time.time()
    with open(workdir / "mie.inp") as stdin, \
            open(workdir / "mie_stdout.txt", "w") as stdout, \
            open(workdir / "mie_stderr.txt", "w") as stderr:
        result = subprocess.run([str(binary)], stdin=stdin, stdout=stdout, stderr=stderr, cwd=workdir)
    elapsed = time.time() - start

    if result.returncode != 0:
        tail = (workdir / "mie_stderr.txt").read_text()[-2000:]
        raise RuntimeError(f"mie exited with status {result.returncode} after {elapsed:.0f} s:\n{tail}")

    new = [p for p in workdir.iterdir()
           if p.suffix in (".cdf", ".nc") and (p not in before or p.stat().st_mtime > before[p])]
    return sorted(new), elapsed


def verify(cdf_path: Path) -> None:
    """Print what is in the table and check the phase functions.

    Checks, for each (wavelength, r_eff), that P11 is normalised the way
    SMART-G expects (integral of P11 sin(theta) d(theta) = 2) and prints the
    asymmetry parameter g. The variable names are those SMART-G's
    `read_phase_cdf` documents; a mismatch is reported, not fatal.
    """
    from netCDF4 import Dataset

    with Dataset(cdf_path) as ds:
        print(f"\n{cdf_path.name}: dimensions {dict((k, len(v)) for k, v in ds.dimensions.items())}")
        for name, var in ds.variables.items():
            print(f"  {name:10s} {str(var.dimensions):45s} {var.shape}")
        try:
            phase = np.asarray(ds["phase"][:], np.float64)  # [wavelength, reff, nphamat, theta]
            wav_um = np.asarray(ds["wavelen"][:], np.float64)
            reff = np.asarray(ds["reff"][:], np.float64)
            theta_all = np.asarray(ds["theta"][:], np.float64)
            ntheta = np.asarray(ds["ntheta"][:])
        except (KeyError, IndexError) as exc:
            print(f"  (could not run the checks: variable {exc} missing)")
            return

    def pick(array: np.ndarray, iw: int, ir: int):
        """Entry (iw, ir), whatever extra axes the variable carries."""
        out = array[iw, ir]
        while out.ndim > (1 if array is theta_all else 0):
            out = out[0]
        return out

    print("\n  wavelength[nm]  r_eff[um]  norm(want 2)     g")
    worst = 0.0
    for iw, w in enumerate(wav_um):
        for ir, r in enumerate(reff):
            n = int(pick(ntheta, iw, ir))
            theta = np.radians(pick(theta_all, iw, ir)[:n])
            p11 = phase[iw, ir, 0, :n]
            norm = np.trapezoid(p11 * np.sin(theta), theta)
            g = np.trapezoid(p11 * np.cos(theta) * np.sin(theta), theta) / norm
            worst = max(worst, abs(norm - 2.0))
            if len(wav_um) * len(reff) <= 40 or ir in (0, len(reff) - 1):
                print(f"  {w * 1e3:13.1f} {r:10.2f} {norm:12.4f} {g:9.4f}")
    print(f"  worst |norm - 2| = {worst:.3g}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--libradtran-dir", default=os.environ.get("LIBRADTRAN_DIR"),
                   help="libRadtran install/build directory (default: $LIBRADTRAN_DIR)")
    p.add_argument("--mie-bin", help="path to the `mie` executable, if not <dir>/bin/mie")
    p.add_argument("--workdir", type=Path, default=Path("libradtran_mie_run"),
                   help="where the input, logs and the netCDF table are written")
    p.add_argument("--reff", type=float, nargs=3, default=[5.0, 30.0, 1.0],
                   metavar=("MIN", "MAX", "STEP"), help="effective radius grid [um]")
    width = p.add_mutually_exclusive_group()
    width.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                       help="gamma distribution parameter (default 7)")
    width.add_argument("--veff", type=float,
                       help="effective variance instead of alpha: alpha = 1/veff - 3")
    p.add_argument("--wavelength", type=float, nargs=2, default=[440.0, 670.0],
                   metavar=("MIN", "MAX"), help="wavelength range [nm]")
    p.add_argument("--wavelength-step", type=float, default=115.0,
                   help="wavelength step [nm]; the default gives 440, 555, 670")
    p.add_argument("--nstokes", type=int, default=4, choices=(1, 4))
    p.add_argument("--nmom", type=int, default=2000,
                   help="Legendre terms; the tool warns if too few for the phase function")
    p.add_argument("--nthetamax", type=int, default=1000)
    p.add_argument("--extra", action="append", default=[], metavar='"KEY VALUE"',
                   help='extra `mie` input line, e.g. --extra "dx_max 0.01" (repeatable)')
    p.add_argument("--verbose", action="store_true", help="pass `verbose` to mie")
    p.add_argument("--quick", action="store_true",
                   help="one radius (10 um), one wavelength (670 nm), nmom 1000: a fast test")
    p.add_argument("--dry-run", action="store_true",
                   help="write the input file and print what would run, without running mie")
    args = p.parse_args()

    if args.veff is not None:
        if not 0.0 < args.veff < 1.0 / 3.0:
            p.error("--veff must be in (0, 1/3)")
        args.alpha = 1.0 / args.veff - 3.0
    if args.quick:
        args.reff, args.wavelength, args.nmom = [10.0, 10.0, 1.0], [670.0, 670.0], 1000

    input_text = build_input(args)
    print(f"gamma distribution: alpha = {args.alpha:g}, effective variance = {1 / (args.alpha + 3):.4g}")
    print("mie input:\n" + "".join("    " + line + "\n" for line in input_text.splitlines()))

    if args.dry_run:
        args.workdir.mkdir(parents=True, exist_ok=True)
        (args.workdir / "mie.inp").write_text(input_text)
        print(f"dry run: wrote {args.workdir / 'mie.inp'}; run it with `mie < mie.inp`")
        return

    try:
        binary = find_mie_binary(args.libradtran_dir, args.mie_bin)
        print(f"using {binary}")
        files, elapsed = run_mie(binary, input_text, args.workdir)
    except (FileNotFoundError, RuntimeError) as exc:
        sys.exit(str(exc))

    print(f"mie finished in {elapsed:.0f} s; logs in {args.workdir}")
    if not files:
        sys.exit(f"mie ran but wrote no .cdf/.nc file in {args.workdir}; see mie_stdout.txt / mie_stderr.txt")
    for f in files:
        verify(f)
    print(f"\nTable: {files[0]}")
    print("SMART-G can read it with smartg.phase.read_phase_cdf(path, output_sg_ready=False).")


if __name__ == "__main__":
    main()
