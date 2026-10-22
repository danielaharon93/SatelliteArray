# Theoretical background

Notes on the physics and mathematics behind the CloudCT-style satellite
formation simulation in this repository. They are written to be read in
order, but each file stands alone.

| # | Topic | Used in |
|---|---|---|
| 1 | [Orbital mechanics, TLEs and SGP4](01_orbital_mechanics.md) | `main.py` (`build_satellite`, `formation_positions_km`) |
| 2 | [Coordinate frames and viewing geometry](02_coordinate_frames_and_geometry.md) | `main.py` (`enu_basis_km`, `zenith_azimuth_deg`, `ray_sphere_intersection_km`) |
| 3 | [Radiative transfer and Monte Carlo (SMART-G)](03_radiative_transfer.md) | `playground_core.py` (`render_view`) |
| 4 | [Reflectance, clouds and surfaces](04_reflectance_clouds_surfaces.md) | `playground_core.py`, `main.py` (`toa_reflectance_at_angles`) |
| 5 | [Multi-angle imaging and cloud tomography](05_multi_angle_imaging.md) | formation design, view-angle selection |
| 6 | [Pointing-error model](06_pointing_error.md) | `pointing_noise.py` |
| 7 | [What SMART-G accepts: the input data format](07_smartg_input_format.md) | `cloud_microphysics.py` (`to_smartg_dataset`, `make_cloud3d`), `playground_core.py` (`build_scene_3d`) |

[references.md](references.md) lists the sources.

## Conventions

- Distances in km unless stated; angles in degrees in the UI and radians in
  formulas.
- Zenith angle is measured from the local vertical (0° = straight up/down).
- Azimuth is measured clockwise from north unless a section says otherwise.
- Vectors are column vectors; `‖v‖` is the Euclidean norm.

These notes are general background, not a description of the code's exact
behaviour. Where they say what the code does, check the named function.
