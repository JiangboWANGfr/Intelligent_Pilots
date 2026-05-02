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
        
        if concentration_map is not None:
            self.set_external_concentration_map(
                concentration_map,
                config=self.config,
                scene_name=self.scene_name
            )
        
        self.height, self.width = self.config.image_size
        self.observation_space = spaces.Dict({
            'aircraft_pos': spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
            'target_pos': spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
            'goal_vector': spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            'local_concentration': spaces.Box(low=0, high=1, shape=(9,),
                                             dtype=np.float32),
            'velocity': spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            'distance_to_target': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        })
        self.safety_threshold = self.config.concentration_threshold
        self.danger_threshold = self.config.concentration_threshold * 1.5
        self.success_threshold = self.config.success_threshold
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,),
                                       dtype=np.float32)
        
        self.max_speed = 15.0
        self.perception_range = 50
        
        self.aircraft_pos = None
        self.target_pos = None
        self.velocity = None
        self.step_count = 0
        self.max_steps = 500
        self.trajectory = []
        self.total_fuel_consumption = 0.0
        self.max_concentration_exposure = 0.0
        self.prev_distance_to_target = 0.0
    
    def _update_environment_shape(self):
        self.height, self.width = self.config.image_size
        self.observation_space = spaces.Dict({
            'aircraft_pos': spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
            'target_pos': spaces.Box(low=0, high=1, shape=(2,), dtype=np.float32),
            'goal_vector': spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            'local_concentration': spaces.Box(low=0, high=1, shape=(9,),
                                             dtype=np.float32),
            'velocity': spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
            'distance_to_target': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        })
        self.safety_threshold = self.config.concentration_threshold
        self.danger_threshold = self.config.concentration_threshold * 1.5
        self.success_threshold = self.config.success_threshold

    def _get_concentration_at(self, pos: np.ndarray) -> float:
        y = int(np.clip(pos[0], 0, self.height - 1))
        x = int(np.clip(pos[1], 0, self.width - 1))
        return float(self.concentration_map[y, x])

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
            if self._get_concentration_at(pos) < safe_limit:
                return pos

        safe_indices = np.argwhere(self.concentration_map < safe_limit)
        if len(safe_indices) > 0:
            idx = safe_indices[self.np_random.integers(0, len(safe_indices))]
            return np.array([idx[0], idx[1]], dtype=np.float32)

        return np.array([self.height // 2, self.width // 2], dtype=np.float32)
    
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

        self.velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.step_count = 0
        self.trajectory = [self.aircraft_pos.copy()]
        self.total_fuel_consumption = 0.0
        self.max_concentration_exposure = 0.0
        self.prev_distance_to_target = float(np.linalg.norm(
            self.target_pos - self.aircraft_pos
        ))
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def _get_local_concentration(self) -> np.ndarray:
        y, x = int(self.aircraft_pos[0]), int(self.aircraft_pos[1])
        local_conc = []
        
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny, nx = y + dy * self.perception_range // 3, \
                        x + dx * self.perception_range // 3
                ny = np.clip(ny, 0, self.height - 1)
                nx = np.clip(nx, 0, self.width - 1)
                local_conc.append(self.concentration_map[ny, nx])
        
        return np.array(local_conc, dtype=np.float32)
    
    def _get_observation(self) -> Dict:
        local_conc = self._get_local_concentration()
        delta = self.target_pos - self.aircraft_pos
        distance_to_target = np.linalg.norm(delta)
        max_distance = np.sqrt(self.height ** 2 + self.width ** 2)
        
        return {
            'aircraft_pos': np.array([
                self.aircraft_pos[0] / self.height,
                self.aircraft_pos[1] / self.width
            ], dtype=np.float32),
            'target_pos': np.array([
                self.target_pos[0] / self.height,
                self.target_pos[1] / self.width
            ], dtype=np.float32),
            'goal_vector': np.array([
                delta[0] / self.height,
                delta[1] / self.width
            ], dtype=np.float32),
            'local_concentration': local_conc,
            'velocity': (self.velocity / self.max_speed).astype(np.float32),
            'distance_to_target': np.array([distance_to_target / max_distance],
                                          dtype=np.float32)
        }
    
    def _get_info(self) -> Dict:
        y, x = int(self.aircraft_pos[0]), int(self.aircraft_pos[1])
        current_conc = self.concentration_map[
            np.clip(y, 0, self.height-1),
            np.clip(x, 0, self.width-1)
        ]
        
        return {
            'current_concentration': current_conc,
            'distance_to_target': float(np.linalg.norm(
                self.target_pos - self.aircraft_pos)),
            'fuel_consumed': self.total_fuel_consumption,
            'step_count': self.step_count,
            'is_in_danger_zone': current_conc > self.danger_threshold,
            'trajectory_length': len(self.trajectory),
            'scene_name': self.scene_name
        }
    
    def step(self, action):
        action = np.clip(action, -1, 1)
        
        acceleration = action * 2.0
        self.velocity += acceleration
        speed = np.linalg.norm(self.velocity)
        if speed > self.max_speed:
            self.velocity = self.velocity / speed * self.max_speed
        
        new_pos = self.aircraft_pos + self.velocity
        new_pos[0] = np.clip(new_pos[0], 0, self.height - 1)
        new_pos[1] = np.clip(new_pos[1], 0, self.width - 1)
        
        self.aircraft_pos = new_pos
        self.step_count += 1
        
        fuel_cost = 0.1 + 0.01 * speed + 0.05 * np.linalg.norm(acceleration)
        self.total_fuel_consumption += fuel_cost
        
        y, x = int(self.aircraft_pos[0]), int(self.aircraft_pos[1])
        current_conc = self.concentration_map[y, x]
        self.max_concentration_exposure = max(self.max_concentration_exposure,
                                              current_conc)
        
        self.trajectory.append(self.aircraft_pos.copy())
        
        distance_to_target = np.linalg.norm(self.target_pos - self.aircraft_pos)
        progress = self.prev_distance_to_target - distance_to_target
        self.prev_distance_to_target = float(distance_to_target)
        
        terminated = False
        truncated = False
        reward = 0.0

        reward += 2.0 * progress
        reward -= 20.0 * current_conc

        if distance_to_target < self.success_threshold:
            reward += 200.0
            terminated = True

        if current_conc > self.danger_threshold:
            reward -= 80.0

        if current_conc > 0.9:
            reward -= 200.0
            terminated = True

        reward -= fuel_cost * 0.2
        reward -= 0.05 * np.linalg.norm(acceleration)
        
        if self.step_count >= self.max_steps:
            truncated = True
            reward -= 100.0
        
        observation = self._get_observation()
        info = self._get_info()
        
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
