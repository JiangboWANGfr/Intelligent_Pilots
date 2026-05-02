import gymnasium as gym
from gymnasium import spaces
import numpy as np
from copy import deepcopy
from typing import Dict, Tuple, Optional, List
from src.model.gmm_model import GMMVolcanicAshModel
from src.model.irregular_ash_generator import IrregularAshGenerator
from src.config.volcanic_ash_config import VolcanicAshConfig


class VolcanicAshEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, config: VolcanicAshConfig,
                 render_mode: Optional[str] = None,
                 scene_configs: Optional[List[VolcanicAshConfig]] = None,
                 concentration_map: Optional[np.ndarray] = None):
        super().__init__()

        self.base_config = VolcanicAshConfig.from_dict(config.to_dict())
        self.config = VolcanicAshConfig.from_dict(config.to_dict())
        self.render_mode = render_mode
        self.scene_configs = [
            VolcanicAshConfig.from_dict(scene_config.to_dict())
            for scene_config in (scene_configs or [config])
        ]
        self.scene_cursor = -1
        self.external_concentration_map = None
        self.scene_name = self.config.scene_name or self.config.model_type
        self.ash_model = GMMVolcanicAshModel(self.config)
        self.concentration_map = self.ash_model.generate_concentration_map()

        self.height, self.width = self.config.image_size

        self.dt = 1.0
        self.min_speed = 3.0
        self.max_speed = 15.0
        self.cruise_speed = 9.0
        self.max_turn_rate = np.deg2rad(12.0)
        self.max_accel = 2.0
        self.ray_angles = np.deg2rad(np.linspace(-90.0, 90.0, 13)).astype(np.float32)
        self.sensor_distances = np.array([10, 20, 40, 80, 120], dtype=np.float32)
        self.sensor_dim = len(self.ray_angles) * len(self.sensor_distances)

        self.observation_space = self._build_observation_space()
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.safety_threshold = self.config.concentration_threshold
        self.danger_threshold = self.config.concentration_threshold * 1.5
        self.success_threshold = self.config.success_threshold

        self.heading = 0.0
        self.speed = self.cruise_speed
        self.prev_action = np.zeros(2, dtype=np.float32)

        self.aircraft_pos = None
        self.target_pos = None
        self.step_count = 0
        self.max_steps = 500
        self.trajectory = []
        self.total_fuel_consumption = 0.0
        self.max_concentration_exposure = 0.0
        self.prev_distance_to_target = 0.0

        if concentration_map is not None:
            self.set_external_concentration_map(
                concentration_map,
                config=self.config,
                scene_name=self.scene_name
            )

    def _build_observation_space(self):
        return spaces.Dict({
            'aircraft_pos': spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32),
            'goal_vector': spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            'heading_vec': spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            'speed': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'distance_to_target': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'current_concentration': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'forward_concentration': spaces.Box(low=0.0, high=1.0,
                                                shape=(self.sensor_dim,), dtype=np.float32)
        })

    def _update_environment_shape(self):
        self.height, self.width = self.config.image_size
        self.observation_space = self._build_observation_space()
        self.safety_threshold = self.config.concentration_threshold
        self.danger_threshold = self.config.concentration_threshold * 1.5
        self.success_threshold = self.config.success_threshold

    def _wrap_heading(self):
        self.heading = (self.heading + np.pi) % (2.0 * np.pi) - np.pi

    def _get_concentration_at_pos(self, pos: np.ndarray) -> float:
        y = int(np.clip(pos[0], 0, self.height - 1))
        x = int(np.clip(pos[1], 0, self.width - 1))
        return float(self.concentration_map[y, x])

    def _get_concentration_at(self, pos: np.ndarray) -> float:
        return self._get_concentration_at_pos(pos)

    def _sample_safe_point(self, margin: int = 50, max_tries: int = 2000) -> np.ndarray:
        safe_limit = self.safety_threshold * 0.8
        margin_y = min(margin, max(0, (self.height - 2) // 2))
        margin_x = min(margin, max(0, (self.width - 2) // 2))
        low_y, high_y = margin_y, max(margin_y + 1, self.height - margin_y)
        low_x, high_x = margin_x, max(margin_x + 1, self.width - margin_x)

        for _ in range(max_tries):
            pos = np.array([
                self.np_random.integers(low_y, high_y),
                self.np_random.integers(low_x, high_x)
            ], dtype=np.float32)
            if self._get_concentration_at_pos(pos) < safe_limit:
                return pos

        safe_indices = np.argwhere(self.concentration_map < safe_limit)
        if len(safe_indices) > 0:
            idx = safe_indices[self.np_random.integers(0, len(safe_indices))]
            return np.array([idx[0], idx[1]], dtype=np.float32)

        return np.array([self.height // 2, self.width // 2], dtype=np.float32)

    def _get_forward_concentration_sensor(self) -> np.ndarray:
        values = []
        for relative_angle in self.ray_angles:
            angle = self.heading + relative_angle
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            for distance in self.sensor_distances:
                sample_x = self.aircraft_pos[1] + distance * cos_a
                sample_y = self.aircraft_pos[0] - distance * sin_a
                values.append(self._get_concentration_at_pos(
                    np.array([sample_y, sample_x], dtype=np.float32)
                ))
        return np.array(values, dtype=np.float32)

    def _segment_risk(self, old_pos: np.ndarray, new_pos: np.ndarray,
                      num_samples: int = 10) -> Dict[str, float]:
        samples = []
        for t in np.linspace(0.0, 1.0, max(num_samples, 2)):
            pos = old_pos * (1.0 - t) + new_pos * t
            samples.append(self._get_concentration_at_pos(pos))
        concentrations = np.array(samples, dtype=np.float32)
        return {
            'mean': float(np.mean(concentrations)),
            'max': float(np.max(concentrations)),
            'end': float(concentrations[-1])
        }

    def _compute_fuel_cost(self, speed: float, dv: float, turn_rate: float) -> float:
        speed_ratio = speed / max(self.cruise_speed, 1e-6)
        accel_ratio = abs(dv) / (self.max_accel * self.dt + 1e-6)
        turn_ratio = abs(turn_rate) / (self.max_turn_rate + 1e-6)

        base_fuel = 0.03 * self.dt
        speed_fuel = 0.02 * (speed_ratio ** 3) * self.dt
        accel_fuel = 0.015 * (accel_ratio ** 2) * self.dt
        turn_fuel = 0.015 * (turn_ratio ** 2) * self.dt
        return float(base_fuel + speed_fuel + accel_fuel + turn_fuel)

    def _compute_reward(self,
                        old_pos: np.ndarray,
                        new_pos: np.ndarray,
                        action: np.ndarray,
                        dv: float,
                        turn_rate: float,
                        out_of_bounds: bool = False):
        reward = 0.0

        distance = float(np.linalg.norm(self.target_pos - self.aircraft_pos))
        progress = self.prev_distance_to_target - distance
        progress_norm = progress / (self.max_speed * self.dt + 1e-6)
        reward += 5.0 * progress_norm
        self.prev_distance_to_target = distance

        risk = self._segment_risk(old_pos, new_pos, num_samples=10)
        mean_conc = risk['mean']
        max_conc = risk['max']

        reward -= 10.0 * mean_conc

        if max_conc > self.safety_threshold:
            excess = (max_conc - self.safety_threshold) / (
                self.danger_threshold - self.safety_threshold + 1e-6
            )
            reward -= 30.0 * excess

        if max_conc > self.danger_threshold:
            reward -= 80.0

        lethal = False
        if max_conc > 0.9:
            reward -= 200.0
            lethal = True

        fuel_cost = self._compute_fuel_cost(self.speed, dv, turn_rate)
        self.total_fuel_consumption += fuel_cost
        reward -= 2.0 * fuel_cost

        action_change = float(np.linalg.norm(action - self.prev_action))
        reward -= 0.1 * action_change
        self.prev_action = action.copy()

        success = False
        if distance < self.success_threshold:
            reward += 200.0
            success = True

        if out_of_bounds:
            reward -= 200.0

        reward -= 0.05
        return reward, success, lethal, fuel_cost, max_conc, mean_conc

    def set_external_concentration_map(self, concentration_map: np.ndarray,
                                       config: Optional[VolcanicAshConfig] = None,
                                       scene_name: Optional[str] = None):
        map_array = np.asarray(concentration_map, dtype=np.float32)
        if map_array.ndim != 2:
            raise ValueError('External concentration map must be a 2D array')

        runtime_config = VolcanicAshConfig.from_dict((config or self.config).to_dict())
        runtime_config.image_size = tuple(map_array.shape)
        if scene_name:
            runtime_config.scene_name = scene_name

        self.config = runtime_config
        self.scene_name = runtime_config.scene_name or runtime_config.model_type
        self.ash_model = GMMVolcanicAshModel(runtime_config)
        self.external_concentration_map = np.clip(map_array, 0.0, 1.0)
        self.scene_configs = [VolcanicAshConfig.from_dict(runtime_config.to_dict())]
        self.scene_cursor = -1
        self._update_environment_shape()
        self.concentration_map = np.array(self.external_concentration_map, copy=True)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.scene_cursor = (self.scene_cursor + 1) % len(self.scene_configs)
        self.config = deepcopy(self.scene_configs[self.scene_cursor])
        self.scene_name = self.config.scene_name or self.config.model_type

        if self.external_concentration_map is not None:
            self._update_environment_shape()
            self.ash_model = GMMVolcanicAshModel(self.config)
            self.concentration_map = np.array(self.external_concentration_map, copy=True)
        else:
            self._update_environment_shape()
            self.ash_model = GMMVolcanicAshModel(self.config)
            if self.config.enable_irregular:
                episode_seed = int(self.np_random.integers(0, 99999))
                self.ash_model.irregular_generator = IrregularAshGenerator(seed=episode_seed)
                self.ash_model.config.wind_direction = float(self.np_random.integers(0, 360))
                self.ash_model.config.turbulence_scale = float(0.08 + self.np_random.random() * 0.14)
                self.ash_model.config.wind_strength = float(0.15 + self.np_random.random() * 0.25)
            self.concentration_map = self.ash_model.generate_concentration_map()

        margin = 50
        self.aircraft_pos = self._sample_safe_point(margin)

        min_distance = min(self.width, self.height) * 0.4
        for _ in range(2000):
            self.target_pos = self._sample_safe_point(margin)
            dist = np.linalg.norm(self.target_pos - self.aircraft_pos)
            if dist > min_distance:
                break

        delta = self.target_pos - self.aircraft_pos
        dy = -float(delta[0])
        dx = float(delta[1])
        self.heading = float(np.arctan2(dy, dx))
        self.speed = self.cruise_speed
        self.prev_action = np.zeros(2, dtype=np.float32)

        self.step_count = 0
        self.trajectory = [self.aircraft_pos.copy()]
        self.total_fuel_consumption = 0.0
        self.max_concentration_exposure = 0.0
        self.prev_distance_to_target = float(np.linalg.norm(delta))

        observation = self._get_observation()
        info = self._get_info()
        return observation, info

    def _get_observation(self) -> Dict:
        delta = self.target_pos - self.aircraft_pos
        distance_to_target = float(np.linalg.norm(delta))
        max_distance = np.sqrt(self.height ** 2 + self.width ** 2)
        current_conc = self._get_concentration_at_pos(self.aircraft_pos)

        return {
            'aircraft_pos': np.array([
                self.aircraft_pos[0] / self.height,
                self.aircraft_pos[1] / self.width
            ], dtype=np.float32),
            'goal_vector': np.array([
                delta[0] / self.height,
                delta[1] / self.width
            ], dtype=np.float32),
            'heading_vec': np.array([
                np.cos(self.heading),
                np.sin(self.heading)
            ], dtype=np.float32),
            'speed': np.array([self.speed / self.max_speed], dtype=np.float32),
            'distance_to_target': np.array([distance_to_target / max_distance],
                                           dtype=np.float32),
            'current_concentration': np.array([current_conc], dtype=np.float32),
            'forward_concentration': self._get_forward_concentration_sensor()
        }

    def _get_info(self) -> Dict:
        current_conc = self._get_concentration_at_pos(self.aircraft_pos)
        return {
            'current_concentration': current_conc,
            'distance_to_target': float(np.linalg.norm(
                self.target_pos - self.aircraft_pos)),
            'fuel_consumed': self.total_fuel_consumption,
            'step_count': self.step_count,
            'is_in_danger_zone': current_conc > self.danger_threshold,
            'trajectory_length': len(self.trajectory),
            'scene_name': self.scene_name,
            'speed': float(self.speed),
            'heading': float(self.heading),
            'max_concentration_exposure': float(self.max_concentration_exposure)
        }

    def step(self, action):
        self.step_count += 1
        action = np.clip(action, -1.0, 1.0).astype(np.float32)

        old_pos = self.aircraft_pos.copy()

        turn_cmd = float(action[0])
        speed_cmd = float(action[1])

        turn_rate = turn_cmd * self.max_turn_rate
        self.heading += turn_rate * self.dt
        self._wrap_heading()

        target_speed = self.min_speed + (speed_cmd + 1.0) * 0.5 * (
            self.max_speed - self.min_speed
        )
        dv = float(np.clip(
            target_speed - self.speed,
            -self.max_accel * self.dt,
            self.max_accel * self.dt
        ))

        self.speed = float(np.clip(self.speed + dv, self.min_speed, self.max_speed))

        dx = self.speed * np.cos(self.heading) * self.dt
        dy = -self.speed * np.sin(self.heading) * self.dt
        new_pos = old_pos + np.array([dy, dx], dtype=np.float32)

        out_of_bounds = (
            new_pos[0] < 0 or new_pos[0] >= self.height or
            new_pos[1] < 0 or new_pos[1] >= self.width
        )
        self.aircraft_pos = np.array([
            np.clip(new_pos[0], 0, self.height - 1),
            np.clip(new_pos[1], 0, self.width - 1)
        ], dtype=np.float32)

        reward, success, lethal, fuel_cost, max_conc, mean_conc = self._compute_reward(
            old_pos=old_pos,
            new_pos=self.aircraft_pos,
            action=action,
            dv=dv,
            turn_rate=turn_rate,
            out_of_bounds=out_of_bounds
        )

        terminated = bool(success or lethal or out_of_bounds)
        truncated = bool(self.step_count >= self.max_steps)
        if truncated and not terminated:
            reward -= 100.0

        self.max_concentration_exposure = max(self.max_concentration_exposure, max_conc)
        self.trajectory.append(self.aircraft_pos.copy())

        observation = self._get_observation()
        info = self._get_info()
        info.update({
            'success': success,
            'lethal': lethal,
            'out_of_bounds': out_of_bounds,
            'fuel_cost': fuel_cost,
            'segment_max_concentration': max_conc,
            'segment_mean_concentration': mean_conc,
            'turn_rate': float(turn_rate),
            'dv': float(dv)
        })
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'rgb_array':
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.imshow(self.concentration_map, cmap='gray', alpha=0.7)

            if len(self.trajectory) > 1:
                trajectory = np.array(self.trajectory)
                ax.plot(trajectory[:, 1], trajectory[:, 0], 'b-', linewidth=2,
                        label='Trajectory')

            ax.plot(self.aircraft_pos[1], self.aircraft_pos[0], 'g^',
                    markersize=15, label='Aircraft')
            ax.plot(self.target_pos[1], self.target_pos[0], 'r*',
                    markersize=20, label='Target')

            heading_dx = np.cos(self.heading) * 20
            heading_dy = -np.sin(self.heading) * 20
            ax.arrow(self.aircraft_pos[1], self.aircraft_pos[0],
                     heading_dx, heading_dy,
                     color='cyan', width=1.5, head_width=8)

            ax.legend()
            ax.set_title(f'{self.scene_name} | Step: {self.step_count}')

            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close(fig)
            return image

        return None

    def close(self):
        pass
