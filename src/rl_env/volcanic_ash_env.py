import gymnasium as gym
from gymnasium import spaces
import numpy as np
from copy import deepcopy
from typing import Dict, Tuple, Optional, List
from scipy.ndimage import distance_transform_edt
from src.model.gmm_model import GMMVolcanicAshModel
from src.model.irregular_ash_generator import IrregularAshGenerator
from src.model.random_ash_scene_generator import RandomAshSceneGenerator
from src.path_planning.fallback_planner import FallbackPlanner
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
        self.scene_map_cache = {}
        self.scene_name = self.config.scene_name or self.config.model_type
        self.random_scene_generator = None
        self.random_scene_counter = 0
        if bool(getattr(self.base_config, 'use_random_ash_scenes', False)):
            self.random_scene_generator = RandomAshSceneGenerator(self.base_config)
        self.ash_model = GMMVolcanicAshModel(self.config)
        self.concentration_map = self.ash_model.generate_concentration_map()

        self.height, self.width = self.config.image_size
        self.dt = 1.0
        self.min_cruise_speed = 7.0
        self.max_cruise_speed = 13.0
        self.fixed_cruise_speed = 9.0
        self.cruise_speed = 9.0
        self.speed = self.cruise_speed
        self.max_turn_rate = np.deg2rad(12.0)
        self.corridor_radius = 30.0
        self.lookahead_distance = 45.0
        self.reference_path_points = 160
        self.path_planning_threshold_ratio = 0.8
        self.path_risk_inflation_radius = 8.0
        self.path_boundary_margin = 45.0
        self.ash_avoidance_gain = 0.0
        self.ash_avoidance_activation_ratio = 0.6
        self.airport_safety_threshold_ratio = 0.35
        self.airport_clearance_radius = 35.0
        self.safe_airport_mask = None
        self.cruise_speed_mode = 'fixed'
        self._refresh_runtime_parameters()

        self.ray_angles = np.deg2rad(np.linspace(-90.0, 90.0, 13)).astype(np.float32)
        self.sensor_distances = np.array([10, 20, 40, 80, 120], dtype=np.float32)
        self.sensor_dim = len(self.ray_angles) * len(self.sensor_distances)

        self.observation_space = self._build_observation_space()
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32
        )

        self.safety_threshold = self.config.concentration_threshold
        self.danger_threshold = self.config.concentration_threshold * 1.5
        self.success_threshold = self.config.success_threshold

        self.heading = 0.0
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.prev_turn_cmd = 0.0

        self.aircraft_pos = None
        self.target_pos = None
        self.step_count = 0
        self.max_steps = 500
        self.trajectory = []
        self.total_fuel_consumption = 0.0
        self.ash_exposure = 0.0
        self.path_length_travelled = 0.0
        self.max_concentration_exposure = 0.0
        self.prev_distance_to_target = 0.0
        self.prev_path_s = 0.0
        self.best_path_s = 0.0

        self.reference_path = None
        self.path_segment_vectors = None
        self.path_segment_lengths = None
        self.path_cumulative_lengths = None
        self.path_total_length = 0.0

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
            'cruise_speed': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'distance_to_target': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'current_concentration': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            'forward_concentration': spaces.Box(low=0.0, high=1.0,
                                                shape=(self.sensor_dim,), dtype=np.float32),
            'lookahead_vector': spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            'lookahead_heading_error': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            'reference_turn_cmd': spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            'cross_track_error': spaces.Box(low=0.0, high=3.0, shape=(1,), dtype=np.float32),
            'path_progress_ratio': spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        })

    def _refresh_runtime_parameters(self):
        self.height, self.width = self.config.image_size
        self.min_cruise_speed = float(getattr(self.config, 'min_cruise_speed', 7.0))
        self.max_cruise_speed = float(getattr(self.config, 'max_cruise_speed', 13.0))
        if self.min_cruise_speed > self.max_cruise_speed:
            self.min_cruise_speed, self.max_cruise_speed = (
                self.max_cruise_speed,
                self.min_cruise_speed
            )
        self.fixed_cruise_speed = float(getattr(self.config, 'fixed_cruise_speed', 9.0))
        self.fixed_cruise_speed = float(np.clip(
            self.fixed_cruise_speed,
            self.min_cruise_speed,
            self.max_cruise_speed
        ))
        self.cruise_speed = self.fixed_cruise_speed
        self.speed = self.cruise_speed
        self.cruise_speed_mode = str(getattr(self.config, 'cruise_speed_mode', 'fixed')).lower()
        self.corridor_radius = float(getattr(self.config, 'path_corridor_radius', 30.0))
        self.lookahead_distance = float(getattr(self.config, 'path_lookahead_distance', 45.0))
        self.reference_path_points = int(getattr(self.config, 'reference_path_points', 160))
        self.path_planning_threshold_ratio = float(getattr(
            self.config,
            'path_planning_threshold_ratio',
            0.8
        ))
        self.path_planning_threshold_ratio = float(np.clip(
            self.path_planning_threshold_ratio,
            0.05,
            1.0
        ))
        self.path_risk_inflation_radius = float(getattr(
            self.config,
            'path_risk_inflation_radius',
            8.0
        ))
        self.path_boundary_margin = float(getattr(
            self.config,
            'path_boundary_margin',
            45.0
        ))
        self.ash_avoidance_gain = float(getattr(self.config, 'ash_avoidance_gain', 0.0))
        self.ash_avoidance_activation_ratio = float(getattr(
            self.config,
            'ash_avoidance_activation_ratio',
            0.6
        ))
        self.airport_safety_threshold_ratio = float(getattr(
            self.config,
            'airport_safety_threshold_ratio',
            0.35
        ))
        self.airport_clearance_radius = float(getattr(
            self.config,
            'airport_clearance_radius',
            35.0
        ))

    def _select_episode_cruise_speed(self) -> float:
        if self.cruise_speed_mode == 'random':
            return float(self.np_random.uniform(self.min_cruise_speed, self.max_cruise_speed))
        return self.fixed_cruise_speed

    def _update_environment_shape(self):
        self._refresh_runtime_parameters()
        self.observation_space = self._build_observation_space()
        self.safety_threshold = self.config.concentration_threshold
        self.danger_threshold = self.config.concentration_threshold * 1.5
        self.success_threshold = self.config.success_threshold

    def _wrap_heading(self):
        self.heading = (self.heading + np.pi) % (2.0 * np.pi) - np.pi

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return float((angle + np.pi) % (2.0 * np.pi) - np.pi)

    def _heading_to_velocity_dir(self) -> np.ndarray:
        return np.array([-np.sin(self.heading), np.cos(self.heading)], dtype=np.float32)

    @staticmethod
    def _heading_from_tangent(tangent: np.ndarray) -> float:
        return float(np.arctan2(-float(tangent[0]), float(tangent[1])))

    def _clip_position(self, pos: np.ndarray) -> np.ndarray:
        return np.array([
            np.clip(pos[0], 0, self.height - 1),
            np.clip(pos[1], 0, self.width - 1)
        ], dtype=np.float32)

    def _get_concentration_at_pos(self, pos: np.ndarray) -> float:
        y = int(np.clip(pos[0], 0, self.height - 1))
        x = int(np.clip(pos[1], 0, self.width - 1))
        return float(self.concentration_map[y, x])

    def _get_concentration_at(self, pos: np.ndarray) -> float:
        return self._get_concentration_at_pos(pos)

    def _sample_safe_point(self, margin: int = 50, max_tries: int = 2000) -> np.ndarray:
        safe_limit = self.safety_threshold * self.airport_safety_threshold_ratio
        margin_y = min(margin, max(0, (self.height - 2) // 2))
        margin_x = min(margin, max(0, (self.width - 2) // 2))
        low_y, high_y = margin_y, max(margin_y + 1, self.height - margin_y)
        low_x, high_x = margin_x, max(margin_x + 1, self.width - margin_x)

        for _ in range(max_tries):
            pos = np.array([
                self.np_random.integers(low_y, high_y),
                self.np_random.integers(low_x, high_x)
            ], dtype=np.float32)
            y, x = int(pos[0]), int(pos[1])
            if self._get_concentration_at_pos(pos) < safe_limit and self.safe_airport_mask[y, x]:
                return pos

        safe_indices = np.argwhere(self.safe_airport_mask)
        if len(safe_indices) > 0:
            idx = safe_indices[self.np_random.integers(0, len(safe_indices))]
            return np.array([idx[0], idx[1]], dtype=np.float32)

        return np.array([self.height // 2, self.width // 2], dtype=np.float32)

    def _build_safe_airport_mask(self, margin: int = 50) -> np.ndarray:
        safe_limit = self.safety_threshold * self.airport_safety_threshold_ratio
        near_cloud_limit = self.safety_threshold * 0.5
        base_safe = self.concentration_map < safe_limit
        distance_to_cloud = distance_transform_edt(self.concentration_map < near_cloud_limit)
        clearance_safe = distance_to_cloud >= self.airport_clearance_radius
        mask = base_safe & clearance_safe

        margin_y = min(margin, max(0, (self.height - 2) // 2))
        margin_x = min(margin, max(0, (self.width - 2) // 2))
        if margin_y > 0:
            mask[:margin_y, :] = False
            mask[self.height - margin_y:, :] = False
        if margin_x > 0:
            mask[:, :margin_x] = False
            mask[:, self.width - margin_x:] = False

        if np.any(mask):
            return mask

        fallback = base_safe.copy()
        if margin_y > 0:
            fallback[:margin_y, :] = False
            fallback[self.height - margin_y:, :] = False
        if margin_x > 0:
            fallback[:, :margin_x] = False
            fallback[:, self.width - margin_x:] = False
        return fallback if np.any(fallback) else base_safe

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

    def _build_reference_path(self,
                              start_pos: np.ndarray,
                              target_pos: np.ndarray) -> List[Tuple[float, float]]:
        planner = FallbackPlanner(self.config)
        return planner.plan(
            self.concentration_map,
            tuple(map(float, start_pos)),
            tuple(map(float, target_pos)),
            max_concentration=self.safety_threshold * self.path_planning_threshold_ratio,
            risk_inflation_radius=self.path_risk_inflation_radius,
            boundary_margin=self.path_boundary_margin,
            desired_points=max(self.reference_path_points, 2)
        )

    def set_reference_path(self, path_points: List[Tuple[float, float]]):
        points = np.asarray(path_points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError('reference_path must have shape [N, 2] in [y, x] coordinates')
        if len(points) == 1:
            points = np.vstack([points[0], points[0]])

        points[:, 0] = np.clip(points[:, 0], 0, self.height - 1)
        points[:, 1] = np.clip(points[:, 1], 0, self.width - 1)

        segment_vectors = points[1:] - points[:-1]
        segment_lengths = np.linalg.norm(segment_vectors, axis=1)
        valid = segment_lengths > 1e-6
        if not np.any(valid):
            target = points[-1].copy()
            target[1] = np.clip(target[1] + 1.0, 0, self.width - 1)
            points = np.vstack([points[0], target]).astype(np.float32)
            segment_vectors = points[1:] - points[:-1]
            segment_lengths = np.linalg.norm(segment_vectors, axis=1)

        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths))).astype(np.float32)

        self.reference_path = points.astype(np.float32)
        self.path_segment_vectors = segment_vectors.astype(np.float32)
        self.path_segment_lengths = segment_lengths.astype(np.float32)
        self.path_cumulative_lengths = cumulative
        self.path_total_length = float(cumulative[-1])

    def _project_to_reference_path(self, pos: np.ndarray) -> Dict[str, object]:
        if self.reference_path is None:
            raise RuntimeError('reference_path is required before observations or rewards can be computed')

        starts = self.reference_path[:-1]
        vectors = self.path_segment_vectors
        lengths = self.path_segment_lengths
        length_sq = np.maximum(lengths ** 2, 1e-6)
        rel = pos.astype(np.float32) - starts
        ratios = np.clip(np.sum(rel * vectors, axis=1) / length_sq, 0.0, 1.0)
        projections = starts + ratios[:, None] * vectors
        distances = np.linalg.norm(projections - pos.astype(np.float32), axis=1)
        index = int(np.argmin(distances))

        segment_length = max(float(lengths[index]), 1e-6)
        tangent = vectors[index] / segment_length
        s_value = float(self.path_cumulative_lengths[index] + ratios[index] * lengths[index])
        progress_ratio = 0.0
        if self.path_total_length > 1e-6:
            progress_ratio = float(np.clip(s_value / self.path_total_length, 0.0, 1.0))

        return {
            's': s_value,
            'cross_track_error': float(distances[index]),
            'nearest_point': projections[index].astype(np.float32),
            'tangent': tangent.astype(np.float32),
            'segment_index': index,
            'progress_ratio': progress_ratio
        }

    def _point_at_path_s(self, s_value: float) -> np.ndarray:
        if self.reference_path is None:
            raise RuntimeError('reference_path is required before lookahead can be computed')
        if self.path_total_length <= 1e-6:
            return self.reference_path[-1].copy()

        s_value = float(np.clip(s_value, 0.0, self.path_total_length))
        index = int(np.searchsorted(self.path_cumulative_lengths, s_value, side='right') - 1)
        index = min(max(index, 0), len(self.path_segment_lengths) - 1)
        local_length = max(float(self.path_segment_lengths[index]), 1e-6)
        ratio = (s_value - float(self.path_cumulative_lengths[index])) / local_length
        return (
            self.reference_path[index] * (1.0 - ratio) +
            self.reference_path[index + 1] * ratio
        ).astype(np.float32)

    def _get_path_tracking(self, pos: Optional[np.ndarray] = None) -> Dict[str, object]:
        position = self.aircraft_pos if pos is None else pos
        metrics = self._project_to_reference_path(position)
        lookahead_s = float(metrics['s']) + self.lookahead_distance
        lookahead_point = self._point_at_path_s(lookahead_s)
        lookahead_delta = lookahead_point - position
        metrics['lookahead_point'] = lookahead_point
        metrics['lookahead_delta'] = lookahead_delta.astype(np.float32)
        return metrics

    def get_reference_turn_command(self, gain: float = 1.0) -> float:
        heading_error = self.get_lookahead_heading_error()
        path_cmd = -float(gain) * heading_error / (self.max_turn_rate + 1e-6)
        ash_bias = self._get_ash_avoidance_turn_bias()
        turn_cmd = path_cmd + self.ash_avoidance_gain * ash_bias
        return float(np.clip(turn_cmd, -1.0, 1.0))

    def _get_ash_avoidance_turn_bias(self) -> float:
        if self.ash_avoidance_gain <= 0.0:
            return 0.0

        sensor_grid = self._get_forward_concentration_sensor().reshape(
            len(self.ray_angles),
            len(self.sensor_distances)
        )
        distance_weights = 1.0 / np.sqrt(np.maximum(self.sensor_distances, 1.0) / 10.0)
        risk_by_angle = np.max(sensor_grid * distance_weights[None, :], axis=1)

        left_mask = self.ray_angles > np.deg2rad(10.0)
        right_mask = self.ray_angles < np.deg2rad(-10.0)
        center_mask = np.abs(self.ray_angles) <= np.deg2rad(35.0)

        left_risk = float(np.max(risk_by_angle[left_mask])) if np.any(left_mask) else 0.0
        right_risk = float(np.max(risk_by_angle[right_mask])) if np.any(right_mask) else 0.0
        center_risk = float(np.max(risk_by_angle[center_mask])) if np.any(center_mask) else 0.0
        trigger_risk = max(left_risk, right_risk, center_risk)

        activation = self.safety_threshold * self.ash_avoidance_activation_ratio
        if trigger_risk <= activation:
            return 0.0

        pressure = (trigger_risk - activation) / max(self.safety_threshold - activation, 1e-6)
        side_bias = (left_risk - right_risk) / max(self.safety_threshold, 1e-6)
        return float(np.clip(pressure * side_bias, -1.0, 1.0))

    def get_lookahead_heading_error(self) -> float:
        metrics = self._get_path_tracking(self.aircraft_pos)
        delta = metrics['lookahead_point'] - self.aircraft_pos
        desired_heading = float(np.arctan2(-float(delta[0]), float(delta[1])))
        return self._wrap_angle(desired_heading - self.heading)

    def _compute_reward(self,
                        old_pos: np.ndarray,
                        new_pos: np.ndarray,
                        action: np.ndarray,
                        turn_cmd: float,
                        turn_rate: float,
                        reference_turn_cmd: float,
                        out_of_bounds: bool = False):
        reward = 0.0

        path_metrics = self._get_path_tracking(new_pos)
        current_path_s = float(path_metrics['s'])
        path_progress = current_path_s - self.prev_path_s
        path_progress_norm = path_progress / (self.cruise_speed * self.dt + 1e-6)
        reward += 6.0 * np.clip(path_progress_norm, -1.0, 1.0)
        self.prev_path_s = current_path_s
        self.best_path_s = max(self.best_path_s, current_path_s)

        cross_track_error = float(path_metrics['cross_track_error'])
        cte_norm = min(cross_track_error / max(self.corridor_radius, 1e-6), 3.0)
        if cross_track_error <= self.corridor_radius:
            reward -= 0.2 * (cte_norm ** 2)
        else:
            reward -= 1.5 * (cte_norm ** 2)

        distance = float(np.linalg.norm(self.target_pos - self.aircraft_pos))
        goal_progress = self.prev_distance_to_target - distance
        goal_progress_norm = goal_progress / (self.cruise_speed * self.dt + 1e-6)
        reward += 2.0 * np.clip(goal_progress_norm, -1.0, 1.0)
        self.prev_distance_to_target = distance

        heading_alignment = float(np.dot(self._heading_to_velocity_dir(), path_metrics['tangent']))
        reward += 0.8 * max(heading_alignment, 0.0)
        if heading_alignment < -0.2:
            reward -= 2.0

        risk = self._segment_risk(old_pos, new_pos, num_samples=10)
        mean_conc = risk['mean']
        max_conc = risk['max']
        segment_length = float(np.linalg.norm(new_pos - old_pos))
        self.ash_exposure += mean_conc * segment_length
        self.path_length_travelled += segment_length

        reward -= 10.0 * mean_conc

        if max_conc > self.safety_threshold:
            excess = max_conc - self.safety_threshold
            reward -= 30.0 * excess

        if max_conc > self.danger_threshold:
            reward -= 80.0

        lethal = False
        if max_conc > 0.9:
            reward -= 200.0
            lethal = True

        reward -= 0.05 * (turn_cmd ** 2)
        reward -= 0.5 * ((turn_cmd - reference_turn_cmd) ** 2)
        action_change = float(abs(turn_cmd - self.prev_turn_cmd))
        reward -= 0.1 * action_change
        self.prev_turn_cmd = turn_cmd
        self.prev_action = action.copy()

        success = False
        if distance < self.success_threshold:
            reward += 200.0
            success = True

        if out_of_bounds:
            reward -= 200.0

        reward -= 0.05
        fuel_cost = 0.0
        return (
            float(reward),
            success,
            lethal,
            fuel_cost,
            max_conc,
            mean_conc,
            path_metrics,
            segment_length,
            heading_alignment,
            path_progress
        )

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
        self.scene_map_cache = {}
        self._update_environment_shape()
        self.concentration_map = np.array(self.external_concentration_map, copy=True)

    def initialize_flight(self,
                          start_pos: Tuple[float, float],
                          target_pos: Tuple[float, float],
                          reference_path: Optional[List[Tuple[float, float]]] = None):
        self.aircraft_pos = self._clip_position(np.array(start_pos, dtype=np.float32))
        self.target_pos = self._clip_position(np.array(target_pos, dtype=np.float32))

        path_points = reference_path
        if path_points is None:
            path_points = self._build_reference_path(self.aircraft_pos, self.target_pos)
        self.set_reference_path(path_points)

        start_metrics = self._get_path_tracking(self.aircraft_pos)
        self.heading = self._heading_from_tangent(start_metrics['tangent'])
        self.speed = self.cruise_speed
        self.prev_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.prev_turn_cmd = 0.0

        self.step_count = 0
        self.trajectory = [self.aircraft_pos.copy()]
        self.total_fuel_consumption = 0.0
        self.ash_exposure = 0.0
        self.path_length_travelled = 0.0
        self.max_concentration_exposure = 0.0
        self.prev_distance_to_target = float(np.linalg.norm(self.target_pos - self.aircraft_pos))
        self.prev_path_s = float(start_metrics['s'])
        self.best_path_s = self.prev_path_s

        return self._get_observation(), self._get_info()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.random_scene_generator is not None:
            base_seed = getattr(self.base_config, 'random_scene_seed', None)
            if base_seed is None:
                scene_seed = int(self.np_random.integers(0, 2**31 - 1))
            else:
                scene_seed = int(base_seed) + self.random_scene_counter
            self.random_scene_counter += 1
            self.config = self.random_scene_generator.sample_config(
                seed=scene_seed,
                rng=np.random.default_rng(scene_seed)
            )
            self.scene_cursor = -1
        else:
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
            randomize_irregular = bool(getattr(
                self.config,
                'randomize_irregular_each_episode',
                True
            ))
            cache_key = self.scene_cursor
            if (not randomize_irregular) and cache_key in self.scene_map_cache:
                self.concentration_map = np.array(self.scene_map_cache[cache_key], copy=True)
            else:
                if self.config.enable_irregular and randomize_irregular:
                    episode_seed = int(self.np_random.integers(0, 99999))
                    self.ash_model.irregular_generator = IrregularAshGenerator(seed=episode_seed)
                    self.ash_model.config.wind_direction = float(self.np_random.integers(0, 360))
                    self.ash_model.config.turbulence_scale = float(0.08 + self.np_random.random() * 0.14)
                    self.ash_model.config.wind_strength = float(0.15 + self.np_random.random() * 0.25)
                self.concentration_map = self.ash_model.generate_concentration_map()
                if not randomize_irregular:
                    self.scene_map_cache[cache_key] = np.array(self.concentration_map, copy=True)

        self.cruise_speed = self._select_episode_cruise_speed()
        self.speed = self.cruise_speed

        margin = 50
        self.safe_airport_mask = self._build_safe_airport_mask(margin)
        aircraft_pos = self._sample_safe_point(margin)

        min_distance = min(self.width, self.height) * 0.4
        target_pos = aircraft_pos.copy()
        for _ in range(2000):
            target_pos = self._sample_safe_point(margin)
            dist = np.linalg.norm(target_pos - aircraft_pos)
            if dist > min_distance:
                break

        return self.initialize_flight(aircraft_pos, target_pos)

    def _get_observation(self) -> Dict:
        delta = self.target_pos - self.aircraft_pos
        distance_to_target = float(np.linalg.norm(delta))
        max_distance = np.sqrt(self.height ** 2 + self.width ** 2)
        current_conc = self._get_concentration_at_pos(self.aircraft_pos)
        path_metrics = self._get_path_tracking(self.aircraft_pos)
        lookahead_delta = path_metrics['lookahead_delta']
        heading_error = self.get_lookahead_heading_error()
        reference_turn_cmd = self.get_reference_turn_command()

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
            'cruise_speed': np.array([
                self.cruise_speed / max(self.max_cruise_speed, 1e-6)
            ], dtype=np.float32),
            'distance_to_target': np.array([distance_to_target / max_distance],
                                           dtype=np.float32),
            'current_concentration': np.array([current_conc], dtype=np.float32),
            'forward_concentration': self._get_forward_concentration_sensor(),
            'lookahead_vector': np.array([
                lookahead_delta[0] / self.height,
                lookahead_delta[1] / self.width
            ], dtype=np.float32),
            'lookahead_heading_error': np.array([
                heading_error / np.pi
            ], dtype=np.float32),
            'reference_turn_cmd': np.array([reference_turn_cmd], dtype=np.float32),
            'cross_track_error': np.array([
                min(float(path_metrics['cross_track_error']) / max(self.corridor_radius, 1e-6), 3.0)
            ], dtype=np.float32),
            'path_progress_ratio': np.array([
                float(path_metrics['progress_ratio'])
            ], dtype=np.float32)
        }

    def _get_info(self) -> Dict:
        current_conc = self._get_concentration_at_pos(self.aircraft_pos)
        path_metrics = self._get_path_tracking(self.aircraft_pos)
        heading_alignment = float(np.dot(self._heading_to_velocity_dir(), path_metrics['tangent']))
        return {
            'current_concentration': current_conc,
            'distance_to_target': float(np.linalg.norm(
                self.target_pos - self.aircraft_pos)),
            'fuel_consumed': self.total_fuel_consumption,
            'ash_exposure': float(self.ash_exposure),
            'path_length_travelled': float(self.path_length_travelled),
            'step_count': self.step_count,
            'is_in_danger_zone': current_conc > self.danger_threshold,
            'trajectory_length': len(self.trajectory),
            'scene_name': self.scene_name,
            'speed': float(self.speed),
            'cruise_speed': float(self.cruise_speed),
            'heading': float(self.heading),
            'max_concentration_exposure': float(self.max_concentration_exposure),
            'path_s': float(path_metrics['s']),
            'path_progress_ratio': float(path_metrics['progress_ratio']),
            'cross_track_error': float(path_metrics['cross_track_error']),
            'heading_alignment': heading_alignment,
            'lookahead_point': np.asarray(path_metrics['lookahead_point'], dtype=np.float32)
        }

    def step(self, action):
        self.step_count += 1
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size < 1:
            raise ValueError('Action must contain a turn command')
        action = np.clip(action[:1], -1.0, 1.0).astype(np.float32)

        old_pos = self.aircraft_pos.copy()

        turn_cmd = float(action[0])
        reference_turn_cmd = self.get_reference_turn_command()
        turn_rate = turn_cmd * self.max_turn_rate
        self.heading -= turn_rate * self.dt
        self._wrap_heading()
        self.speed = self.cruise_speed

        velocity_dir = self._heading_to_velocity_dir()
        new_pos_unclipped = old_pos + velocity_dir * (self.cruise_speed * self.dt)

        out_of_bounds = (
            new_pos_unclipped[0] < 0 or new_pos_unclipped[0] >= self.height or
            new_pos_unclipped[1] < 0 or new_pos_unclipped[1] >= self.width
        )
        self.aircraft_pos = self._clip_position(new_pos_unclipped)

        (
            reward,
            success,
            lethal,
            fuel_cost,
            max_conc,
            mean_conc,
            path_metrics,
            segment_length,
            heading_alignment,
            path_progress
        ) = self._compute_reward(
            old_pos=old_pos,
            new_pos=self.aircraft_pos,
            action=action,
            turn_cmd=turn_cmd,
            turn_rate=turn_rate,
            reference_turn_cmd=reference_turn_cmd,
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
            'segment_length': float(segment_length),
            'segment_max_concentration': max_conc,
            'segment_mean_concentration': mean_conc,
            'turn_rate': float(turn_rate),
            'turn_cmd': float(turn_cmd),
            'reference_turn_cmd': float(reference_turn_cmd),
            'path_progress': float(path_progress),
            'heading_alignment': float(heading_alignment),
            'cross_track_error': float(path_metrics['cross_track_error'])
        })
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == 'rgb_array':
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
            ax.imshow(self.concentration_map, cmap='gray', alpha=0.7)

            if self.reference_path is not None and len(self.reference_path) > 1:
                ax.plot(self.reference_path[:, 1], self.reference_path[:, 0],
                        color='white', linestyle='--', linewidth=1.5,
                        label='Reference Path')

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
