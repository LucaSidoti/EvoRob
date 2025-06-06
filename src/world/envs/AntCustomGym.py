from os import path
from typing import Dict, Union

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from src.utils.geometry import quat2rot
import mujoco

DEFAULT_CAMERA_CONFIG = {
    "distance": 5,
}


class AntCustomEnv(MujocoEnv, utils.EzPickle):
    r"""In this environment a Passive Dynamic Walker is tasked to locomote.
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        robot_path: str,
        frame_skip: int = 5,
        default_camera_config: Dict[str, float] = DEFAULT_CAMERA_CONFIG,
        forward_reward_weight: float = 1,
        ctrl_cost_weight: float = 0.5,
        cfrc_cost_weight: float = 5e-4,
        main_body: Union[int, str] = 1,
        reset_noise_scale: float = 0.1,
        exclude_current_positions_from_observation: bool = True,
        include_cfrc_ext_in_observation: bool = False,
        pert_force=None,
        **kwargs,
    ):
        xml_file_path = path.join(
            path.dirname(path.realpath(__file__)),
            robot_path,
        )

        utils.EzPickle.__init__(
            self,
            xml_file_path,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            cfrc_cost_weight,
            main_body,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            pert_force,
            **kwargs,
        )
        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._cfrc_cost_weight = cfrc_cost_weight

        self._main_body = main_body

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        MujocoEnv.__init__(
            self,
            xml_file_path,
            frame_skip,
            observation_space=None,  # needs to be defined after
            default_camera_config=default_camera_config,
            width=832,
            height=496,
            camera_name="track",
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = self.data.qpos.size + self.data.qvel.size
        obs_size -= 2 * exclude_current_positions_from_observation
        obs_size += (
            self.data.cfrc_ext[1:].size * include_cfrc_ext_in_observation
        )

        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "skipped_qpos": 2 * exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size
            - 2 * exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
        }
        self.body_ids = None
        self.force = None
        self.previous_state = None
        self.stuck = 0
        if pert_force is not None:
            self.body_ids , self.force = pert_force


    def step(self, action):
        # --- Store previous position ---
        position = self.previous_position

        # --- Apply external perturbation, if any ---
        # if self.body_ids is not None:
        #     self.apply_force()

        # --- Simulate one step ---
        self.do_simulation(action, self.frame_skip)

        # --- Get new position and radius ---
        new_position = self.data.qpos[:2].copy()
        radius = np.linalg.norm(new_position)
        angle = np.arctan2(new_position[1], new_position[0])

        # --- Update cumulative angle for full-circle bonus ---
        dtheta = np.mod(angle - self.previous_angle + np.pi, 2 * np.pi) - np.pi
        self.cumulative_angle += dtheta
        self.previous_angle = angle
        circle_bonus = 0.0
        if abs(self.cumulative_angle) >= 2 * np.pi:
            circle_bonus = 50.0
            self.cumulative_angle = 0.0

        # --- Compute upright alignment ---
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Base")
        rotmat = self.data.xmat[body_id].reshape(3, 3)
        upright_alignment = np.dot(rotmat[:, 2], np.array([0, 0, 1]))

        # --- Reward function ---
        reward_radius = -5.0 * (radius - 2.0) ** 2
        reward_upright = 5.0 * upright_alignment
        reward = reward_radius + reward_upright + circle_bonus

        # --- Observation ---
        observation = self._get_obs()

        # --- Termination ---
        z = self.data.qpos[2]
        terminated = (
            z < 0.2 or z > 2.8 or radius > 2.5 or upright_alignment < 0.3 or
            np.isinf(observation).any() or
            np.any(np.isnan(self.data.qacc)) or
            np.any(np.isinf(self.data.qacc)) or
            np.any(np.abs(self.data.qacc) > 1e6)
        )
        terminated = False
        if terminated and (
            np.any(np.isnan(self.data.qacc)) or
            np.any(np.isinf(self.data.qacc)) or
            np.any(np.abs(self.data.qacc) > 1e6)
        ):
            bad_dof = np.argwhere(
                np.isnan(self.data.qacc) | np.isinf(self.data.qacc) | (np.abs(self.data.qacc) > 1e6)
            ).squeeze()[0]
            print(ValueError(f"MuJoCo Warning: NaN, Inf or huge value in QACC at DOF {bad_dof}"))

        # --- Info ---
        info = {
            "reward_radius": reward_radius,
            "reward_upright": reward_upright,
            "circle_bonus": circle_bonus,
            "radius": radius,
            "upright_alignment": upright_alignment,
        }

        # --- Update state ---
        self.previous_state = observation
        self.previous_position = new_position

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, info



    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        return np.concatenate((position, velocity))


    def apply_force(self):
        body_id = self.body_ids
        force = self.force
        pert = self.np_random.uniform(
            low=-0.1, high=0.1, size=3)
        rot = quat2rot([1, *pert])
        force = np.dot(rot, force.reshape(2, 3).T).T.flatten()
        self.data.xfrc_applied[body_id] = force


    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = (
            self.init_qvel
            + self._reset_noise_scale
            * self.np_random.standard_normal(self.model.nv)
        )
        self.set_state(qpos, qvel)
        self.previous_position = self.data.qpos[:2].copy()
        self.previous_angle = np.arctan2(self.data.qpos[1], self.data.qpos[0])
        self.cumulative_angle = 0.0
        observation = self._get_obs()
        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
        }
