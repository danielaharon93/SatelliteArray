"""SMART-G playground: an interactive sandbox for trying out radiative-
transfer techniques against the real 3D cumulus field before deciding
whether/how to wire them into main.py.

Standalone from the orbit-formation simulation: it reuses main.py's
TLE/geometry/SMART-G helpers (imported as a module, not copied), with
every viewing/illumination angle under direct interactive control
instead of being derived from real ephemeris - the point is to freely
explore geometries main.py's own pipeline doesn't (yet) expose.

Two sandbox modes (the "Sandbox mode" control)
-----------------------------------------------
"Single satellite": a lone satellite at its real current TLE position,
with a freely chosen viewing direction. The user picks it as an
(off-nadir angle, azimuth) pair relative to the satellite's OWN nadir
(straight down, toward Earth's center). Casting a ray from the satellite
in that direction and intersecting it with Earth's surface gives a
ground point, "point a". The 3D cloud field (see
main.build_smartg_scene_3d()) is centered on the axis from Earth's
center through point a - see main.viewing_angles_deg(), which
independently recovers the correct ground-based viewing (zenith,
azimuth) angles at point a (they are *not* simply the satellite's own
off-nadir angle, because of Earth's curvature).

"Formation (frozen)": the CloudCT formation exactly as main.py defines
it (main.formation_positions_km(): satellite index
main.NADIR_REFERENCE_INDEX looks at its own nadir, every other satellite
looks at that same shared ground target - see
main.render_formation_images_3d()), frozen at a single moment - no
orbital motion, unlike main.py's animated simulation. The user picks
which of the formation's satellites to render.

Both modes let the sun's direction be chosen directly - as a (zenith,
azimuth) pair relative to the target point's own local zenith/nadir -
rather than computed from real ephemeris, and let the domain's
periodic/non-periodic boundary (see build_scene_3d()) be switched at
will.

Saving and running scenarios
----------------------------
"Save configuration" writes every setting - plus the moment in time and the
TLE the satellite geometry was computed from - to a JSON file (see
playground_core.PlaygroundConfig). run_from_config.py renders such a file
without any GUI or display, e.g. on a remote server with a GPU. The rendering
itself lives in playground_core.py, shared by both front ends.

Run: python smartg_playground.py [config.json]
(with a config file, opens the GUI with that scenario's settings and
satellite geometry restored)
"""

import argparse
from pathlib import Path

import numpy as np
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor
from pyface.api import GUI, OK, FileDialog
from skyfield.api import load
from traits.api import Bool, Button, Enum, HasTraits, Instance, Int, List, Range, Str, observe
from traitsui.api import CheckListEditor, HGroup, Item, VGroup, View

import main
import pointing_noise
from playground_core import (
    CONFIG_DIR,
    DEFAULT_NOISE_PARAMS,
    FORMATION_MODE,
    MAX_OFF_NADIR_DEG,
    RESOLUTION_PRESETS,
    SINGLE_MODE,
    PlaygroundConfig,
    PlaygroundSimulation,
    parse_time_utc,
    viewing_direction_km,
)

MARKER_SCALE_KM = main.SATELLITE_MARKER_SCALE_KM
POINT_A_MARKER_SCALE_KM = MARKER_SCALE_KM * 0.6
# Length of the target -> sun arrow; the sun is effectively infinitely far
# away, so only its direction means anything, this just sets how visible it is.
SUN_ARROW_LENGTH_KM = 1500.0


class SmartgPlayground(HasTraits):
    """3D scene plus controls for two sandbox modes (see the module
    docstring), the sun direction, the domain's periodic/non-periodic
    boundary, and a Render button that runs SMART-G and saves the
    resulting image(s).

    "Single satellite": the satellite, and the chosen viewing ray down to
    point a. "Formation (frozen)": the CloudCT formation's 10 satellites,
    their pointing arrows, and the shared nadir target they all point at.
    """

    scene = Instance(MlabSceneModel, ())

    mode = Enum("Single satellite", "Formation (frozen)")

    off_nadir_deg = Range(0.0, MAX_OFF_NADIR_DEG, 0.0)
    view_azimuth_deg = Range(0.0, 360.0, 0.0)
    # Formation mode: which satellites to render - all 10 checked would
    # mean 10 sequential Smartg.run() calls, so this defaults to just the
    # nadir reference satellite for a fast first render.
    satellites_to_render = List(Str, [f"Sat {main.NADIR_REFERENCE_INDEX}"])

    sza_deg = Range(0.0, 100.0, 30.0)
    saa_deg = Range(0.0, 360.0, 90.0)
    periodic = Bool(False)
    resolution = Enum(list(RESOLUTION_PRESETS))
    # Default matches what render_view()/render_formation_frozen() assume
    # when n_photons isn't given; the total sent to SMART-G is this times
    # the pixel count.
    photons_per_pixel = Range(
        1.0, 1.0e6, main.SMARTG_3D_PHOTONS_PER_PIXEL, mode="text"
    )

    # Pointing error (see pointing_noise.py). Sigmas are 1 sigma per
    # cross-boresight axis; picking a preset fills them in, editing any of
    # them switches the preset to "Custom".
    pointing_noise_enabled = Bool(False)
    noise_preset = Enum(
        pointing_noise.DEFAULT_PRESET, [*pointing_noise.PRESETS, "Custom"]
    )
    sigma_bias_deg = Range(0.0, 10.0, DEFAULT_NOISE_PARAMS["sigma_bias_deg"], mode="text")
    sigma_acquisition_deg = Range(
        0.0, 10.0, DEFAULT_NOISE_PARAMS["sigma_acquisition_deg"], mode="text"
    )
    sigma_drift_deg = Range(0.0, 10.0, DEFAULT_NOISE_PARAMS["sigma_drift_deg"], mode="text")
    drift_tau_s = Range(1.0, 1.0e7, DEFAULT_NOISE_PARAMS["drift_tau_s"], mode="text")
    sigma_jitter_deg = Range(0.0, 10.0, DEFAULT_NOISE_PARAMS["sigma_jitter_deg"], mode="text")
    yaw_factor = Range(0.0, 10.0, pointing_noise.DEFAULT_YAW_FACTOR, mode="text")
    acquisition_interval_s = Range(
        0.0, 1.0e7, pointing_noise.DEFAULT_ACQUISITION_INTERVAL_S, mode="text"
    )
    noise_seed = Int(0)
    new_run_button = Button("New Monte-Carlo run")
    noise_budget_text = Str()

    # Saved scenarios (see playground_core.PlaygroundConfig).
    headless_renders = Range(1, 100000, 1, mode="text")
    save_config_button = Button("Save configuration...")
    load_config_button = Button("Load configuration...")

    render_button = Button("Render")
    status_text = Str()

    view = View(
        VGroup(
            Item(
                "scene",
                editor=SceneEditor(scene_class=MayaviScene),
                height=600,
                width=900,
                show_label=False,
            ),
            Item("mode", label="Sandbox mode"),
            HGroup(
                Item("off_nadir_deg", label="Off-nadir angle (deg)"),
                Item("view_azimuth_deg", label="View azimuth (deg)"),
                visible_when="mode == 'Single satellite'",
            ),
            Item(
                "satellites_to_render",
                label="Satellites to render",
                style="custom",
                editor=CheckListEditor(
                    values=[f"Sat {i}" for i in range(main.NUM_SATELLITES)],
                    cols=main.NUM_SATELLITES,
                ),
                visible_when="mode == 'Formation (frozen)'",
            ),
            HGroup(
                Item("sza_deg", label="Sun zenith angle (deg)"),
                Item("saa_deg", label="Sun azimuth (deg)"),
            ),
            HGroup(
                Item("periodic", label="Periodic domain"),
                Item("resolution", label="Image resolution"),
                Item("photons_per_pixel", label="Photons per pixel"),
                Item("render_button", show_label=False),
            ),
            VGroup(
                HGroup(
                    Item("pointing_noise_enabled", label="Enable"),
                    Item("noise_preset", label="Preset"),
                    Item("noise_seed", label="Random seed"),
                    Item("new_run_button", show_label=False),
                ),
                HGroup(
                    Item("sigma_bias_deg", label="Static bias sigma (deg)"),
                    Item("sigma_acquisition_deg", label="Per-acquisition sigma (deg)"),
                    Item("sigma_drift_deg", label="Drift sigma (deg)"),
                    Item("drift_tau_s", label="Drift time constant (s)"),
                    enabled_when="pointing_noise_enabled",
                ),
                HGroup(
                    Item("sigma_jitter_deg", label="Jitter sigma (deg)"),
                    Item("yaw_factor", label="Yaw sigma factor"),
                    Item("acquisition_interval_s", label="Time between renders (s)"),
                    enabled_when="pointing_noise_enabled",
                ),
                Item("noise_budget_text", show_label=False, style="readonly"),
                label="Pointing error",
                show_border=True,
            ),
            HGroup(
                Item("save_config_button", show_label=False),
                Item("load_config_button", show_label=False),
                Item(
                    "headless_renders",
                    label="Renders in a headless run",
                    tooltip="Saved with the configuration; each render draws fresh pointing errors",
                ),
                label="Scenario file",
                show_border=True,
            ),
            Item("status_text", show_label=False, style="readonly"),
        ),
        resizable=True,
        title="SMART-G Playground",
    )

    def __init__(
        self,
        simulation: PlaygroundSimulation,
        config: PlaygroundConfig | None = None,
        **traits,
    ) -> None:
        super().__init__(**traits)
        self._sim = simulation
        self.position_km = simulation.position_km
        # The formation's positions, frozen at the simulation's single
        # moment - see main.formation_positions_km().
        self._formation_positions_km = simulation.formation_positions_km
        self._formation_target_km = simulation.formation_target_km

        self._satellite_marker = None
        self._direction_line = None
        self._point_a_marker = None
        self.point_a_km = None

        self._formation_points = None
        self._formation_arrows = None
        self._formation_target_marker = None
        self._sun_arrow = None

        self._applying_preset = False
        if config is not None:
            self._apply_config(config)
        self._recompute_point_a()
        self._update_noise_budget()

    @observe("scene.activated")
    def _setup_scene(self, event=None) -> None:
        self.scene.background = (0.0, 0.0, 0.0)
        texture_path = main.ensure_earth_texture()
        main.add_earth(self.scene, texture_path)

        self._satellite_marker = self.scene.mlab.points3d(
            [self.position_km[0]],
            [self.position_km[1]],
            [self.position_km[2]],
            color=(1.0, 0.1, 0.1),
            scale_factor=MARKER_SCALE_KM,
        )
        self._redraw_direction()

        self._setup_formation_actors()
        self._apply_mode_visibility()
        self._center_view_on_target()

    def _setup_formation_actors(self) -> None:
        positions = self._formation_positions_km
        pointing_vectors = main.formation_pointing_vectors_km(positions)
        self._formation_points = self.scene.mlab.points3d(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            color=(1.0, 0.1, 0.1),
            scale_factor=MARKER_SCALE_KM,
        )
        self._formation_arrows = self.scene.mlab.quiver3d(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            pointing_vectors[:, 0],
            pointing_vectors[:, 1],
            pointing_vectors[:, 2],
            color=(0.2, 1.0, 1.0),
            mode="arrow",
            scale_factor=1.0,
            scale_mode="vector",
        )
        target = self._formation_target_km
        self._formation_target_marker = self.scene.mlab.points3d(
            [target[0]],
            [target[1]],
            [target[2]],
            color=(1.0, 0.85, 0.0),
            scale_factor=POINT_A_MARKER_SCALE_KM,
        )

    def _apply_mode_visibility(self) -> None:
        if self._satellite_marker is None or self._formation_points is None:
            return  # scene not yet activated
        single = self.mode == "Single satellite"
        self._satellite_marker.visible = single
        self._direction_line.visible = single
        self._point_a_marker.visible = single
        self._formation_points.visible = not single
        self._formation_arrows.visible = not single
        self._formation_target_marker.visible = not single

    def _current_target_km(self) -> np.ndarray:
        """The spot on Earth's surface the satellite(s) look at."""
        if self.mode == "Single satellite":
            return self.point_a_km
        return self._formation_target_km

    def _center_view_on_target(self) -> None:
        """Make the camera orbit/zoom about the spot the satellite(s) look
        at (point a, or the formation's shared nadir target) instead of the
        Earth's centre. Only the focal point moves; the viewing direction
        and distance are kept. Also keeps the sun arrow anchored there."""
        if self._satellite_marker is None:
            return  # scene not yet activated
        # Arrow first: creating a new actor can make Mayavi re-fit the
        # camera to the whole scene (i.e. back onto Earth's centre), which
        # would undo the focal point set below.
        self._redraw_sun_arrow()
        target = self._current_target_km()
        self.scene.mlab.view(focalpoint=[float(c) for c in target])

    def _redraw_sun_arrow(self) -> None:
        """Arrow from the target spot toward the sun, per the GUI's SZA/SAA
        (zenith from local up, azimuth counterclockwise from local east -
        same convention as viewing_direction_km() and main.enu_basis_km())."""
        if self._satellite_marker is None:
            return  # scene not yet activated
        target = self._current_target_km()
        east, north, up = main.enu_basis_km(target)
        zenith = np.radians(self.sza_deg)
        azimuth = np.radians(self.saa_deg)
        horizontal = east * np.cos(azimuth) + north * np.sin(azimuth)
        direction = up * np.cos(zenith) + horizontal * np.sin(zenith)
        direction = direction / np.linalg.norm(direction)
        x, y, z = ([float(c)] for c in target)
        u, v, w = ([float(c)] for c in direction)
        if self._sun_arrow is None:
            self._sun_arrow = self.scene.mlab.quiver3d(
                x,
                y,
                z,
                u,
                v,
                w,
                color=(1.0, 0.5, 0.0),
                mode="arrow",
                scale_factor=SUN_ARROW_LENGTH_KM,
                scale_mode="vector",
            )
        else:
            self._sun_arrow.mlab_source.reset(x=x, y=y, z=z, u=u, v=v, w=w)

    def _sza_deg_changed(self) -> None:
        self._redraw_sun_arrow()

    def _saa_deg_changed(self) -> None:
        self._redraw_sun_arrow()

    def _mode_changed(self) -> None:
        self._apply_mode_visibility()
        self._center_view_on_target()
        self.status_text = f"Switched to {self.mode} mode."

    def _recompute_point_a(self) -> None:
        direction = viewing_direction_km(
            self.position_km, self.off_nadir_deg, self.view_azimuth_deg
        )
        self.point_a_km = main.ray_sphere_intersection_km(
            self.position_km, direction, main.EARTH_RADIUS_KM
        )

    def _redraw_direction(self) -> None:
        if self._satellite_marker is None:
            return  # scene not yet activated
        self._recompute_point_a()
        xs = [self.position_km[0], self.point_a_km[0]]
        ys = [self.position_km[1], self.point_a_km[1]]
        zs = [self.position_km[2], self.point_a_km[2]]
        if self._direction_line is None:
            self._direction_line = self.scene.mlab.plot3d(
                xs, ys, zs, color=(0.2, 1.0, 0.2), tube_radius=None, line_width=2.0
            )
            self._point_a_marker = self.scene.mlab.points3d(
                [self.point_a_km[0]],
                [self.point_a_km[1]],
                [self.point_a_km[2]],
                color=(1.0, 0.85, 0.0),
                scale_factor=POINT_A_MARKER_SCALE_KM,
            )
        else:
            self._direction_line.mlab_source.reset(x=xs, y=ys, z=zs)
            self._point_a_marker.mlab_source.reset(
                x=[self.point_a_km[0]], y=[self.point_a_km[1]], z=[self.point_a_km[2]]
            )

    def _off_nadir_deg_changed(self) -> None:
        self._redraw_direction()
        self._center_view_on_target()

    def _view_azimuth_deg_changed(self) -> None:
        self._redraw_direction()
        self._center_view_on_target()

    def _noise_params(self) -> pointing_noise.PointingNoiseParams:
        return pointing_noise.PointingNoiseParams(
            sigma_bias_deg=self.sigma_bias_deg,
            sigma_acquisition_deg=self.sigma_acquisition_deg,
            sigma_drift_deg=self.sigma_drift_deg,
            drift_tau_s=self.drift_tau_s,
            sigma_jitter_deg=self.sigma_jitter_deg,
            yaw_factor=self.yaw_factor,
            acquisition_interval_s=self.acquisition_interval_s,
        )

    def _update_noise_budget(self) -> None:
        """Budget check (the document's step 7): APE 1 sigma per axis by
        RSS, its cone CE90 by Rayleigh statistics, and roughly what that
        means on the ground at nadir."""
        sigma_deg = self._noise_params().ape_sigma_deg()
        ce90_deg = pointing_noise.CE90_PER_SIGMA * sigma_deg
        altitude_km = np.linalg.norm(self.position_km) - main.EARTH_RADIUS_KM
        ce90_ground_km = altitude_km * np.radians(ce90_deg)
        self.noise_budget_text = (
            f"APE 1-sigma per axis = {sigma_deg:.4f} deg  |  "
            f"cone CE90 = {ce90_deg:.4f} deg = ~{ce90_ground_km:.2f} km on the ground at nadir  |  "
            f"Monte-Carlo run seed {self.noise_seed}, renders so far: {self._sim.noise.acquisition}"
        )

    @observe(
        [
            "sigma_bias_deg",
            "sigma_acquisition_deg",
            "sigma_drift_deg",
            "drift_tau_s",
            "sigma_jitter_deg",
        ]
    )
    def _noise_sigma_changed(self, event) -> None:
        if not self._applying_preset:
            self.noise_preset = self._matching_noise_preset()
        self._update_noise_budget()

    @observe(["yaw_factor", "acquisition_interval_s"])
    def _noise_other_changed(self, event) -> None:
        self._update_noise_budget()

    def _noise_preset_changed(self) -> None:
        if self.noise_preset == "Custom":
            return
        self._applying_preset = True
        try:
            for name, value in pointing_noise.PRESETS[self.noise_preset].items():
                setattr(self, name, value)
        finally:
            self._applying_preset = False
        self._update_noise_budget()

    def _noise_seed_changed(self) -> None:
        self._sim.reset_noise(self.noise_seed)
        self._update_noise_budget()

    def _new_run_button_fired(self) -> None:
        # A new seed = new static biases and drift states for every satellite.
        self.noise_seed += 1

    def _noise_values(self) -> dict:
        return {name: getattr(self, name) for name in DEFAULT_NOISE_PARAMS}

    def _matching_noise_preset(self) -> str:
        values = self._noise_values()
        return next(
            (name for name, preset in pointing_noise.PRESETS.items() if preset == values),
            "Custom",
        )

    def _current_config(self) -> PlaygroundConfig:
        """Every setting as currently shown in the GUI, plus the geometry
        moment/TLE this window was opened with."""
        return PlaygroundConfig(
            tle=self._sim.tle_text,
            time_utc=self._sim.time_utc,
            mode=self.mode,
            off_nadir_deg=self.off_nadir_deg,
            view_azimuth_deg=self.view_azimuth_deg,
            satellites_to_render=sorted(
                int(label.split()[1]) for label in self.satellites_to_render
            ),
            sza_deg=self.sza_deg,
            saa_deg=self.saa_deg,
            periodic=self.periodic,
            resolution=self.resolution,
            photons_per_pixel=self.photons_per_pixel,
            pointing_noise_enabled=self.pointing_noise_enabled,
            yaw_factor=self.yaw_factor,
            acquisition_interval_s=self.acquisition_interval_s,
            noise_seed=self.noise_seed,
            renders=self.headless_renders,
            **self._noise_values(),
        )

    def _apply_config(self, config: PlaygroundConfig) -> None:
        """Set the GUI's controls from a config (its geometry moment/TLE
        are not applied - see PlaygroundSimulation.from_config())."""
        self.mode = config.mode
        self.off_nadir_deg = config.off_nadir_deg
        self.view_azimuth_deg = config.view_azimuth_deg
        self.satellites_to_render = [f"Sat {i}" for i in config.satellites_to_render]
        self.sza_deg = config.sza_deg
        self.saa_deg = config.saa_deg
        self.periodic = config.periodic
        self.resolution = config.resolution
        self.photons_per_pixel = config.photons_per_pixel
        self.pointing_noise_enabled = config.pointing_noise_enabled
        self.yaw_factor = config.yaw_factor
        self.acquisition_interval_s = config.acquisition_interval_s
        self.headless_renders = config.renders
        # Set the sigmas without the preset logic reacting to each one, then
        # name the resulting set (a preset if it matches one, else Custom).
        self._applying_preset = True
        try:
            for name in DEFAULT_NOISE_PARAMS:
                setattr(self, name, getattr(config, name))
        finally:
            self._applying_preset = False
        self.noise_preset = self._matching_noise_preset()
        self.noise_seed = config.noise_seed
        self._sim.reset_noise(config.noise_seed)

    def _show_status(self, text: str) -> None:
        self.status_text = text
        GUI.process_events()

    def _ask_config_path(self, action: str) -> Path | None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        dialog = FileDialog(
            action=action,
            default_directory=str(CONFIG_DIR),
            default_filename="scenario.json" if action == "save as" else "",
            wildcard="JSON files (*.json)|*.json|",
        )
        if dialog.open() != OK:
            return None
        path = Path(dialog.path)
        if action == "save as" and path.suffix.lower() != ".json":
            path = path.with_suffix(".json")
        return path

    def _save_config_button_fired(self) -> None:
        path = self._ask_config_path("save as")
        if path is None:
            return
        try:
            config = self._current_config()
            config.validate()
            config.save(path)
        except Exception as exc:
            self.status_text = f"Could not save configuration: {exc}"
        else:
            self.status_text = (
                f"Saved configuration to {path} (geometry at {config.time_utc}) - "
                f"run it headlessly with: python run_from_config.py {path.name}"
            )

    def _load_config_button_fired(self) -> None:
        path = self._ask_config_path("open")
        if path is None:
            return
        try:
            config = PlaygroundConfig.load(path)
        except Exception as exc:
            self.status_text = f"Could not load configuration: {exc}"
            return
        self._apply_config(config)
        self._recompute_point_a()
        self._update_noise_budget()
        same_geometry = (
            parse_time_utc(config.time_utc) == parse_time_utc(self._sim.time_utc)
            and config.tle.strip() == self._sim.tle_text.strip()
        )
        self.status_text = f"Loaded settings from {path}." + (
            ""
            if same_geometry
            else (
                f" Note: its satellite geometry (time {config.time_utc}) differs from "
                f"this window's ({self._sim.time_utc}) and was not applied - open it "
                f"with 'python smartg_playground.py {path.name}' to restore it."
            )
        )

    def _render_button_fired(self) -> None:
        if self.mode == FORMATION_MODE and not self.satellites_to_render:
            self.status_text = "Select at least one satellite first."
            return
        try:
            self.status_text = self._sim.render(
                self._current_config(), on_status=self._show_status
            )
        except Exception as exc:  # a GPU/auxdata/SMART-G failure shouldn't crash the GUI
            self.status_text = f"Render failed: {exc}"
        self._update_noise_budget()


def run_playground(config_path: Path | None = None) -> None:
    """Entry point. Named to avoid colliding with the `main` module this
    file imports everything from.

    Without `config_path`, places the satellites at the current time from
    a live TLE. With one, restores that saved scenario's settings *and* its
    exact satellite geometry (time + TLE)."""
    if config_path is None:
        tle_text = main.get_tle()
        simulation = PlaygroundSimulation(
            main.build_satellite(tle_text), load.timescale().now(), tle_text
        )
        config = None
    else:
        config = PlaygroundConfig.load(config_path)
        simulation = PlaygroundSimulation.from_config(config)
    app = SmartgPlayground(simulation, config)
    app.configure_traits()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive SMART-G playground.")
    parser.add_argument(
        "config",
        nargs="?",
        type=Path,
        help="optional saved scenario (.json) to open, settings and geometry included",
    )
    run_playground(parser.parse_args().config)
