import json
import os
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple

@dataclass
class GMMCenter:
    x: float = 0.0
    y: float = 0.0
    weight: float = 1.0
    sigma_x: float = 50.0
    sigma_y: float = 50.0

@dataclass
class VolcanicAshConfig:
    model_type: str = "single_center"
    num_centers: int = 1
    cloud_size: Tuple[float, float] = (500, 500)
    concentration_threshold: float = 0.3
    quality_ratio: float = 0.8
    centers: List[Dict] = field(default_factory=list)
    dynamic_params: Dict = field(default_factory=lambda: {
        "displacement_speed": 10.0,
        "deformation_rate": 0.1,
        "decay_rate": 0.05,
        "time_steps": 100
    })

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

def get_preset_configs():
    presets = {
        "single_center": VolcanicAshConfig(
            model_type="single_center",
            num_centers=1,
            centers=[{"x": 250, "y": 250, "weight": 1.0, "sigma_x": 60, "sigma_y": 60}]
        ),
        "double_center": VolcanicAshConfig(
            model_type="double_center",
            num_centers=2,
            centers=[
                {"x": 180, "y": 200, "weight": 0.6, "sigma_x": 50, "sigma_y": 50},
                {"x": 320, "y": 300, "weight": 0.4, "sigma_x": 45, "sigma_y": 45}
            ]
        ),
        "triple_center": VolcanicAshConfig(
            model_type="triple_center",
            num_centers=3,
            centers=[
                {"x": 200, "y": 200, "weight": 0.4, "sigma_x": 40, "sigma_y": 40},
                {"x": 300, "y": 180, "weight": 0.35, "sigma_x": 45, "sigma_y": 45},
                {"x": 250, "y": 320, "weight": 0.25, "sigma_x": 35, "sigma_y": 35}
            ]
        ),
        "ring": VolcanicAshConfig(
            model_type="ring",
            num_centers=6,
            centers=[
                {"x": 350, "y": 250, "weight": 0.18, "sigma_x": 30, "sigma_y": 30},
                {"x": 325, "y": 316, "weight": 0.18, "sigma_x": 30, "sigma_y": 30},
                {"x": 250, "y": 350, "weight": 0.18, "sigma_x": 30, "sigma_y": 30},
                {"x": 175, "y": 316, "weight": 0.18, "sigma_x": 30, "sigma_y": 30},
                {"x": 150, "y": 250, "weight": 0.14, "sigma_x": 30, "sigma_y": 30},
                {"x": 225, "y": 184, "weight": 0.14, "sigma_x": 30, "sigma_y": 30}
            ]
        )
    }
    return presets
