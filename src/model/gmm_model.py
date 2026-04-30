import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
from src.config.volcanic_ash_config import VolcanicAshConfig
from src.model.irregular_ash_generator import IrregularAshGenerator


class GMMVolcanicAshModel:
    def __init__(self, config: VolcanicAshConfig):
        self.config = config
        self.height, self.width = config.image_size
        self.irregular_generator = None
        if config.enable_irregular:
            self.irregular_generator = IrregularAshGenerator(seed=config.random_seed)
        
    def _generate_single_gaussian(self, center: Dict) -> np.ndarray:
        x = np.arange(self.width)
        y = np.arange(self.height)
        X, Y = np.meshgrid(x, y)
        
        cx, cy = center['x'], center['y']
        std_x, std_y = center['std_x'], center['std_y']
        
        gaussian = np.exp(-((X - cx)**2 / (2 * std_x**2) + (Y - cy)**2 / (2 * std_y**2)))
        return gaussian * center['weight']
    
    def generate_concentration_map(self, irregular: Optional[bool] = None) -> np.ndarray:
        """
        生成浓度图

        Args:
            irregular: 是否生成不规则形状，None则使用配置中的设置

        Returns:
            浓度图数组
        """
        # 生成基础高斯混合浓度图
        concentration_map = np.zeros((self.height, self.width), dtype=np.float64)

        for center in self.config.centers:
            gaussian = self._generate_single_gaussian(center)
            concentration_map += gaussian

        if len(self.config.centers) > 0:
            total_weight = sum(c['weight'] for c in self.config.centers)
            if total_weight > 0:
                concentration_map /= total_weight

        concentration_map = np.clip(concentration_map, 0, 1)

        scale_factor = self.config.cloud_size / 100.0
        concentration_map = np.power(concentration_map, 1.0 / scale_factor)

        # 应用不规则形状转换
        use_irregular = irregular if irregular is not None else self.config.enable_irregular

        if use_irregular and self.irregular_generator is not None:
            irregular_config = {
                'turbulence_scale': self.config.turbulence_scale,
                'wind_direction': self.config.wind_direction,
                'wind_strength': self.config.wind_strength,
                'add_fractal': self.config.add_fractal_boundary,
                'fractal_dimension': self.config.fractal_dimension,
                'add_filaments': self.config.add_filaments,
                'num_filaments': self.config.num_filaments
            }

            concentration_map = self.irregular_generator.generate_irregular_ash_cloud(
                concentration_map,
                config=irregular_config
            )

        return concentration_map
    
    def generate_grayscale_image(self, concentration_map: np.ndarray) -> np.ndarray:
        grayscale = (concentration_map * 255).astype(np.uint8)
        return grayscale
    
    def generate_danger_zone_image(self, concentration_map: np.ndarray) -> np.ndarray:
        danger_zones = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        safe_mask = concentration_map < self.config.concentration_threshold * 0.5
        low_risk_mask = (concentration_map >= self.config.concentration_threshold * 0.5) & \
                       (concentration_map < self.config.concentration_threshold)
        medium_risk_mask = (concentration_map >= self.config.concentration_threshold) & \
                         (concentration_map < self.config.concentration_threshold * 1.5)
        high_risk_mask = concentration_map >= self.config.concentration_threshold * 1.5
        
        danger_zones[safe_mask] = [0, 255, 0]
        danger_zones[low_risk_mask] = [255, 255, 0]
        danger_zones[medium_risk_mask] = [255, 165, 0]
        danger_zones[high_risk_mask] = [255, 0, 0]
        
        return danger_zones
    
    def is_valid_image(self, concentration_map: np.ndarray) -> bool:
        max_conc = np.max(concentration_map)
        mean_conc = np.mean(concentration_map)
        
        if max_conc < 0.1:
            return False
        
        high_conc_ratio = np.sum(concentration_map > self.config.concentration_threshold) / \
                         concentration_map.size
        if high_conc_ratio < 0.01 or high_conc_ratio > 0.95:
            return False
        
        return True
    
    def pixel_to_geo(self, px: int, py: int) -> Tuple[float, float]:
        lat = self.config.geo_center_lat + (0.5 - py / self.height) * self.config.geo_span_lat
        lon = self.config.geo_center_lon + (px / self.width - 0.5) * self.config.geo_span_lon
        return lat, lon
    
    def generate_geo_data(self, concentration_map: np.ndarray, step: int = 10) -> Dict:
        geo_data = {
            'type': 'FeatureCollection',
            'features': [],
            'metadata': {
                'center_lat': self.config.geo_center_lat,
                'center_lon': self.config.geo_center_lon,
                'span_lat': self.config.geo_span_lat,
                'span_lon': self.config.geo_span_lon,
                'resolution': step
            }
        }
        
        for y in range(0, self.height, step):
            for x in range(0, self.width, step):
                conc = concentration_map[y, x]
                if conc > 0.01:
                    lat, lon = self.pixel_to_geo(x, y)
                    feature = {
                        'type': 'Feature',
                        'geometry': {
                            'type': 'Point',
                            'coordinates': [lon, lat]
                        },
                        'properties': {
                            'concentration': float(conc),
                            'pixel_x': x,
                            'pixel_y': y
                        }
                    }
                    geo_data['features'].append(feature)
        
        return geo_data
