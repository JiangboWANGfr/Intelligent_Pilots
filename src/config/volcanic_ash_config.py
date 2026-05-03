import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from enum import Enum


class CloudModelType(Enum):
    SINGLE_CENTER = "single_center"
    DOUBLE_CENTER = "double_center"
    TRIPLE_CENTER = "triple_center"
    RING = "ring"


@dataclass
class GMMCenter:
    x: float
    y: float
    weight: float
    std_x: float
    std_y: float
    theta: Optional[float] = None


@dataclass
class VolcanicAshConfig:
    model_type: str = "single_center"
    num_centers: int = 1
    cloud_size: float = 100.0
    concentration_threshold: float = 0.3
    mass_ratio: float = 0.8
    image_size: Tuple[int, int] = (512, 512)
    geo_center_lat: float = 35.0
    geo_center_lon: float = 120.0
    geo_span_lat: float = 2.0
    geo_span_lon: float = 2.0
    success_threshold: float = 20.0
    fixed_cruise_speed: float = 9.0
    min_cruise_speed: float = 7.0
    max_cruise_speed: float = 13.0
    cruise_speed_mode: str = "fixed"
    path_corridor_radius: float = 30.0
    path_lookahead_distance: float = 45.0
    reference_path_points: int = 160
    path_planning_threshold_ratio: float = 0.8
    path_risk_inflation_radius: float = 8.0
    path_boundary_margin: float = 45.0
    ash_avoidance_gain: float = 0.0
    ash_avoidance_activation_ratio: float = 0.6
    centers: List[Dict] = None
    scene_name: str = ""
    training_scene_names: List[str] = None

    # 不规则形状参数
    enable_irregular: bool = True  # 是否启用不规则形状生成
    turbulence_scale: float = 0.15  # 湍流扰动强度 [0, 1]
    wind_direction: float = 45.0  # 风向（度，0为向右，逆时针）
    wind_strength: float = 0.3  # 风力强度 [0, 1]
    add_fractal_boundary: bool = True  # 是否添加分形边界
    fractal_dimension: float = 1.5  # 分形维度 [1.0, 2.0]
    add_filaments: bool = True  # 是否添加细丝结构
    num_filaments: int = 5  # 细丝数量
    random_seed: Optional[int] = None  # 随机种子，用于可重复生成
    randomize_irregular_each_episode: bool = True
    
    def __post_init__(self):
        if self.centers is None:
            self.centers = []
        if self.training_scene_names is None:
            self.training_scene_names = []
    
    def to_dict(self) -> Dict:
        return {
            'model_type': self.model_type,
            'num_centers': self.num_centers,
            'cloud_size': self.cloud_size,
            'concentration_threshold': self.concentration_threshold,
            'mass_ratio': self.mass_ratio,
            'image_size': list(self.image_size),
            'geo_center_lat': self.geo_center_lat,
            'geo_center_lon': self.geo_center_lon,
            'geo_span_lat': self.geo_span_lat,
            'geo_span_lon': self.geo_span_lon,
            'success_threshold': self.success_threshold,
            'fixed_cruise_speed': self.fixed_cruise_speed,
            'min_cruise_speed': self.min_cruise_speed,
            'max_cruise_speed': self.max_cruise_speed,
            'cruise_speed_mode': self.cruise_speed_mode,
            'path_corridor_radius': self.path_corridor_radius,
            'path_lookahead_distance': self.path_lookahead_distance,
            'reference_path_points': self.reference_path_points,
            'path_planning_threshold_ratio': self.path_planning_threshold_ratio,
            'path_risk_inflation_radius': self.path_risk_inflation_radius,
            'path_boundary_margin': self.path_boundary_margin,
            'ash_avoidance_gain': self.ash_avoidance_gain,
            'ash_avoidance_activation_ratio': self.ash_avoidance_activation_ratio,
            'centers': self.centers,
            'scene_name': self.scene_name,
            'training_scene_names': self.training_scene_names,
            'enable_irregular': self.enable_irregular,
            'turbulence_scale': self.turbulence_scale,
            'wind_direction': self.wind_direction,
            'wind_strength': self.wind_strength,
            'add_fractal_boundary': self.add_fractal_boundary,
            'fractal_dimension': self.fractal_dimension,
            'add_filaments': self.add_filaments,
            'num_filaments': self.num_filaments,
            'random_seed': self.random_seed,
            'randomize_irregular_each_episode': self.randomize_irregular_each_episode
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'VolcanicAshConfig':
        config = cls()
        for key, value in data.items():
            if key == 'image_size' and isinstance(value, list):
                setattr(config, key, tuple(value))
            else:
                setattr(config, key, value)
        return config
    
    def save(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'VolcanicAshConfig':
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)


def get_preset_configs() -> Dict[str, VolcanicAshConfig]:
    """获取预设配置，包括规则和不规则形状的多种变体"""
    presets = {}

    # 原有规则模型（不启用不规则形状）
    single_config = VolcanicAshConfig(
        model_type="single_center",
        num_centers=1,
        cloud_size=80.0,
        concentration_threshold=0.3,
        mass_ratio=0.85,
        centers=[{'x': 256, 'y': 256, 'weight': 1.0, 'std_x': 60, 'std_y': 60}],
        enable_irregular=False
    )
    presets["单中心模型_规则"] = single_config

    # 不规则单中心模型 - 轻度扰动
    irregular_single_light = VolcanicAshConfig(
        model_type="single_center",
        num_centers=1,
        cloud_size=80.0,
        concentration_threshold=0.3,
        mass_ratio=0.85,
        centers=[{'x': 256, 'y': 256, 'weight': 1.0, 'std_x': 60, 'std_y': 60}],
        enable_irregular=True,
        turbulence_scale=0.1,
        wind_direction=45.0,
        wind_strength=0.2,
        fractal_dimension=1.3,
        num_filaments=3,
        random_seed=42
    )
    presets["单中心_轻度扰动"] = irregular_single_light

    # 不规则单中心模型 - 强风拉伸
    irregular_single_wind = VolcanicAshConfig(
        model_type="single_center",
        num_centers=1,
        cloud_size=80.0,
        concentration_threshold=0.3,
        mass_ratio=0.85,
        centers=[{'x': 256, 'y': 256, 'weight': 1.0, 'std_x': 60, 'std_y': 60}],
        enable_irregular=True,
        turbulence_scale=0.15,
        wind_direction=90.0,
        wind_strength=0.45,
        fractal_dimension=1.5,
        num_filaments=6,
        random_seed=123
    )
    presets["单中心_强风拉伸"] = irregular_single_wind

    # 不规则双中心 - 复杂扩散
    irregular_double = VolcanicAshConfig(
        model_type="double_center",
        num_centers=2,
        cloud_size=120.0,
        concentration_threshold=0.25,
        mass_ratio=0.8,
        centers=[
            {'x': 200, 'y': 220, 'weight': 0.6, 'std_x': 50, 'std_y': 45},
            {'x': 320, 'y': 290, 'weight': 0.4, 'std_x': 55, 'std_y': 50}
        ],
        enable_irregular=True,
        turbulence_scale=0.18,
        wind_direction=135.0,
        wind_strength=0.35,
        fractal_dimension=1.6,
        num_filaments=8,
        random_seed=456
    )
    presets["双中心_复杂扩散"] = irregular_double

    # 不规则三中心 - 多细丝
    irregular_triple = VolcanicAshConfig(
        model_type="triple_center",
        num_centers=3,
        cloud_size=150.0,
        concentration_threshold=0.2,
        mass_ratio=0.75,
        centers=[
            {'x': 180, 'y': 200, 'weight': 0.4, 'std_x': 40, 'std_y': 35},
            {'x': 280, 'y': 150, 'weight': 0.35, 'std_x': 45, 'std_y': 40},
            {'x': 330, 'y': 300, 'weight': 0.25, 'std_x': 38, 'std_y': 42}
        ],
        enable_irregular=True,
        turbulence_scale=0.2,
        wind_direction=225.0,
        wind_strength=0.3,
        fractal_dimension=1.7,
        num_filaments=12,
        random_seed=789
    )
    presets["三中心_多细丝"] = irregular_triple

    # 不规则环形 - 高分形维度
    irregular_ring = VolcanicAshConfig(
        model_type="ring",
        num_centers=6,
        cloud_size=140.0,
        concentration_threshold=0.22,
        mass_ratio=0.78,
        centers=[
            {'x': 256, 'y': 156, 'weight': 0.18, 'std_x': 30, 'std_y': 28},
            {'x': 332, 'y': 195, 'weight': 0.17, 'std_x': 28, 'std_y': 32},
            {'x': 356, 'y': 278, 'weight': 0.16, 'std_x': 32, 'std_y': 30},
            {'x': 308, 'y': 350, 'weight': 0.15, 'std_x': 29, 'std_y': 31},
            {'x': 204, 'y': 340, 'weight': 0.17, 'std_x': 31, 'std_y': 29},
            {'x': 160, 'y': 255, 'weight': 0.17, 'std_x': 30, 'std_y': 30}
        ],
        enable_irregular=True,
        turbulence_scale=0.16,
        wind_direction=0.0,
        wind_strength=0.25,
        fractal_dimension=1.8,
        num_filaments=10,
        random_seed=101
    )
    presets["环形_高分形"] = irregular_ring

    # 偏离火山口的火山灰云 - 位于左上角
    offset_topleft = VolcanicAshConfig(
        model_type="offset_topleft",
        num_centers=2,
        cloud_size=100.0,
        concentration_threshold=0.28,
        mass_ratio=0.82,
        centers=[
            {'x': 128, 'y': 128, 'weight': 0.65, 'std_x': 45, 'std_y': 40},
            {'x': 180, 'y': 180, 'weight': 0.35, 'std_x': 38, 'std_y': 42}
        ],
        enable_irregular=True,
        turbulence_scale=0.17,
        wind_direction=315.0,
        wind_strength=0.38,
        fractal_dimension=1.55,
        num_filaments=7,
        random_seed=202
    )
    presets["偏离位置_左上"] = offset_topleft

    # 偏离火山口的火山灰云 - 位于右下角
    offset_bottomright = VolcanicAshConfig(
        model_type="offset_bottomright",
        num_centers=3,
        cloud_size=110.0,
        concentration_threshold=0.26,
        mass_ratio=0.79,
        centers=[
            {'x': 350, 'y': 350, 'weight': 0.5, 'std_x': 42, 'std_y': 38},
            {'x': 400, 'y': 400, 'weight': 0.3, 'std_x': 35, 'std_y': 40},
            {'x': 380, 'y': 420, 'weight': 0.2, 'std_x': 30, 'std_y': 32}
        ],
        enable_irregular=True,
        turbulence_scale=0.14,
        wind_direction=180.0,
        wind_strength=0.32,
        fractal_dimension=1.45,
        num_filaments=5,
        random_seed=303
    )
    presets["偏离位置_右下"] = offset_bottomright

    # 极度不规则 - 模拟真实复杂场景
    extreme_irregular = VolcanicAshConfig(
        model_type="extreme_irregular",
        num_centers=4,
        cloud_size=130.0,
        concentration_threshold=0.24,
        mass_ratio=0.76,
        centers=[
            {'x': 200, 'y': 180, 'weight': 0.35, 'std_x': 48, 'std_y': 42},
            {'x': 280, 'y': 220, 'weight': 0.3, 'std_x': 40, 'std_y': 45},
            {'x': 320, 'y': 280, 'weight': 0.2, 'std_x': 35, 'std_y': 38},
            {'x': 240, 'y': 320, 'weight': 0.15, 'std_x': 32, 'std_y': 35}
        ],
        enable_irregular=True,
        turbulence_scale=0.22,
        wind_direction=60.0,
        wind_strength=0.42,
        fractal_dimension=1.75,
        num_filaments=15,
        random_seed=404
    )
    presets["极度不规则"] = extreme_irregular

    for name, config in presets.items():
        config.scene_name = name

    return presets


def get_training_scene_configs(scene_names: Optional[List[str]] = None) -> List[VolcanicAshConfig]:
    presets = get_preset_configs()
    if not scene_names:
        return [VolcanicAshConfig.from_dict(config.to_dict()) for config in presets.values()]

    missing_names = [name for name in scene_names if name not in presets]
    if missing_names:
        raise KeyError(f"Unknown preset scene names: {missing_names}")

    return [VolcanicAshConfig.from_dict(presets[name].to_dict()) for name in scene_names]
