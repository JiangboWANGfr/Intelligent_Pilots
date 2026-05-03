from copy import deepcopy
from typing import Dict, Optional, Tuple

import numpy as np

from src.config.volcanic_ash_config import VolcanicAshConfig
from src.model.gmm_model import GMMVolcanicAshModel


class RandomAshSceneGenerator:
    """Sample bounded random rotated-GMM ash scenes for training episodes."""

    def __init__(self, base_config: VolcanicAshConfig):
        self.base_config = VolcanicAshConfig.from_dict(base_config.to_dict())

    @staticmethod
    def _get_range(config: VolcanicAshConfig, min_name: str, max_name: str) -> Tuple[float, float]:
        lower = float(getattr(config, min_name))
        upper = float(getattr(config, max_name))
        if lower > upper:
            lower, upper = upper, lower
        return lower, upper

    @staticmethod
    def summarize_area(concentration_map: np.ndarray,
                       config: VolcanicAshConfig) -> Dict[str, float]:
        threshold = float(config.concentration_threshold)
        return {
            'low_risk_area_percent': float(np.mean(concentration_map >= threshold * 0.45) * 100.0),
            'medium_risk_area_percent': float(np.mean(concentration_map >= threshold) * 100.0),
            'high_risk_area_percent': float(np.mean(concentration_map >= threshold * 1.45) * 100.0)
        }

    def _sample_centers(self,
                        config: VolcanicAshConfig,
                        rng: np.random.Generator):
        min_centers = int(getattr(config, 'random_scene_min_centers', 1))
        max_centers = int(getattr(config, 'random_scene_max_centers', 6))
        min_centers = max(1, min_centers)
        max_centers = max(min_centers, max_centers)
        num_centers = int(rng.integers(min_centers, max_centers + 1))

        height, width = tuple(config.image_size)
        margin = float(getattr(config, 'random_scene_position_margin', 90.0))
        margin = float(np.clip(margin, 0.0, max(1.0, min(height, width) / 2.0 - 1.0)))
        std_min, std_max = self._get_range(config, 'random_scene_min_std', 'random_scene_max_std')
        anisotropy_min, anisotropy_max = self._get_range(
            config,
            'random_scene_min_anisotropy',
            'random_scene_max_anisotropy'
        )
        weight_min, weight_max = self._get_range(
            config,
            'random_scene_min_weight',
            'random_scene_max_weight'
        )

        raw_weights = rng.uniform(weight_min, weight_max, size=num_centers)
        weights = raw_weights / max(float(np.sum(raw_weights)), 1e-6)
        centers = []

        for index in range(num_centers):
            base_std = float(rng.uniform(std_min, std_max))
            anisotropy = float(rng.uniform(anisotropy_min, anisotropy_max))
            if rng.random() < 0.5:
                std_x = base_std * anisotropy
                std_y = base_std
            else:
                std_x = base_std
                std_y = base_std * anisotropy

            centers.append({
                'x': float(rng.uniform(margin, width - margin)),
                'y': float(rng.uniform(margin, height - margin)),
                'weight': float(weights[index]),
                'std_x': float(std_x),
                'std_y': float(std_y),
                'theta': float(rng.uniform(0.0, np.pi))
            })

        return centers

    def sample_config(self,
                      seed: Optional[int] = None,
                      rng: Optional[np.random.Generator] = None) -> VolcanicAshConfig:
        if rng is None:
            rng = np.random.default_rng(seed)
        scene_seed = int(seed if seed is not None else rng.integers(0, 2**31 - 1))
        last_config = None

        max_attempts = int(getattr(self.base_config, 'random_scene_max_attempts', 120))
        for _ in range(max(1, max_attempts)):
            config = VolcanicAshConfig.from_dict(deepcopy(self.base_config.to_dict()))
            config.model_type = 'random_rotated_gmm'
            config.enable_irregular = True
            config.random_seed = int(rng.integers(0, 2**31 - 1))
            config.randomize_irregular_each_episode = bool(
                getattr(self.base_config, 'randomize_irregular_each_episode', True)
            )

            cloud_min, cloud_max = self._get_range(
                config,
                'random_scene_cloud_size_min',
                'random_scene_cloud_size_max'
            )
            threshold_min, threshold_max = self._get_range(
                config,
                'random_scene_threshold_min',
                'random_scene_threshold_max'
            )
            config.cloud_size = float(rng.uniform(cloud_min, cloud_max))
            config.concentration_threshold = float(rng.uniform(threshold_min, threshold_max))
            config.wind_direction = float(rng.uniform(0.0, 360.0))
            config.wind_strength = float(rng.uniform(0.15, 0.45))
            config.turbulence_scale = float(rng.uniform(0.08, 0.22))
            config.fractal_dimension = float(rng.uniform(1.35, 1.8))
            config.num_filaments = int(rng.integers(3, 13))
            config.centers = self._sample_centers(config, rng)
            config.num_centers = len(config.centers)
            config.scene_name = (
                f'随机旋转GMM_{config.num_centers}中心_seed{scene_seed}_{config.random_seed % 100000}'
            )

            model = GMMVolcanicAshModel(config)
            concentration_map = model.generate_concentration_map(irregular=False)
            area = self.summarize_area(concentration_map, config)
            last_config = config

            if (
                area['medium_risk_area_percent'] >= float(config.random_scene_min_medium_area)
                and area['medium_risk_area_percent'] <= float(config.random_scene_max_medium_area)
                and area['low_risk_area_percent'] <= float(config.random_scene_max_low_area)
                and area['high_risk_area_percent'] <= float(config.random_scene_max_high_area)
            ):
                return config

        return last_config
