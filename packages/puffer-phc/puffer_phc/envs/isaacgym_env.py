import sys

from isaacgym import gymapi


def _sim_params(sim_timestep, device_type):
    # Sim params: keep these hardcoded here for now
    sim_params = gymapi.SimParams()

    sim_params.dt = sim_timestep

    sim_params.use_gpu_pipeline = device_type == "cuda"
    sim_params.num_client_threads = 0

    sim_params.physx.num_threads = 4
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 4
    sim_params.physx.num_velocity_iterations = 0
    sim_params.physx.contact_offset = 0.02
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.bounce_threshold_velocity = 0.2
    sim_params.physx.max_depenetration_velocity = 10.0
    sim_params.physx.default_buffer_size_multiplier = 10.0

    sim_params.physx.use_gpu = device_type == "cuda"
    sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024
    sim_params.physx.num_subscenes = 0

    # Set gravity based on up axis and return axis index
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity.x = 0
    sim_params.gravity.y = 0
    sim_params.gravity.z = -9.81

    return sim_params


class IsaacGymBase:
    def __init__(self, device_type, device_id, headless, sim_timestep=1.0 / 60.0, control_freq_inv=2):
        self.control_freq_inv = control_freq_inv
        self.dt = control_freq_inv * sim_timestep
        self.sim_params = _sim_params(sim_timestep, device_type)

        compute_device = -1 if device_type != "cuda" else device_id
        graphics_device = -1 if headless else compute_device

        # Create sim and viewer
        self.gym = gymapi.acquire_gym()
        self.sim = self.gym.create_sim(compute_device, graphics_device, gymapi.SIM_PHYSX, self.sim_params)
        assert self.sim is not None, "Failed to create sim"

        self.enable_viewer_sync = True
        self.viewer = None

        if not headless:  # Set up a minimal viewer
            # Subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

            # Set the camera position (Z axis up)
            cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
            cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def reset(self):
        pass

    def step(self, actions):
        for _ in range(self.control_freq_inv):
            self.gym.simulate(self.sim)

        self.gym.fetch_results(self.sim, True)

    def render(self):
        if not self.viewer:
            return

        # Check for window closed
        if self.gym.query_viewer_has_closed(self.viewer):
            sys.exit()

        # Check for keyboard events
        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.action == "QUIT" and evt.value > 0:
                sys.exit()
            elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                self.enable_viewer_sync = not self.enable_viewer_sync

        # Step graphics
        if self.enable_viewer_sync:
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
        else:
            self.gym.poll_viewer_events(self.viewer)

    def close(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
