# CloudCT camera spectral response

`eta_hat_spectral_responsivity.csv`: the CloudCT camera's normalised spectral responsivity (supplied by the
CloudCT team). Used by `process_reflectance_gui.py` ("CloudCT camera response").

    eta_hat(lambda) = eta(lambda) / eta_peak,   eta(lambda) = filter(lambda) * QE_pix(lambda) * lambda / (h c)

`filter` is the band filter, `QE_pix` the datasheet quantum efficiency multiplied by the micro-polarizer
transmission, and `lambda / (h c)` converts photons to energy.

Columns: `wavelength_nm`, `eta_hat`, `filter_transmission`, `qe_pixel_e_per_photon`. Only the first two are
needed; the last two are kept for reference.

The integrated radiance of the camera is

    L_eff = integral of L(lambda) * eta_hat(lambda) d(lambda)      [W m-2 sr-1]

with L(lambda) in W m-2 sr-1 nm-1. `eta_hat` is normalised, so L_eff is not divided by its integral. The
signal in digital numbers is

    DN = G_eff * L_eff * exposure time,    G_eff = 18.3 DN / (W m-2 sr-1 s)

where G_eff is the radiometric responsivity from the real radiometric calibration. The exposure time is
whatever the simulation assumes.
