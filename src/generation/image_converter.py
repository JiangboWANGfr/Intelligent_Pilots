import os
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np

from src.config.volcanic_ash_config import VolcanicAshConfig


class AshImageConverter:
    def __init__(self, base_config: VolcanicAshConfig):
        self.base_config = VolcanicAshConfig.from_dict(base_config.to_dict())

    def load_image(self, image_source: Union[str, np.ndarray]) -> np.ndarray:
        if isinstance(image_source, np.ndarray):
            return np.array(image_source, copy=True)

        if not isinstance(image_source, str):
            raise TypeError('image_source must be a file path or numpy array')

        if not os.path.exists(image_source):
            raise FileNotFoundError(f'Image file not found: {image_source}')

        image = cv2.imread(image_source, cv2.IMREAD_UNCHANGED)
        if image is None:
            file_bytes = np.fromfile(image_source, dtype=np.uint8)
            if file_bytes.size > 0:
                image = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f'Unable to read image: {image_source}')

        return image

    def image_to_concentration_map(self,
                                   image_source: Union[str, np.ndarray],
                                   invert: Union[bool, str] = 'auto',
                                   blur_kernel: int = 5,
                                   clip_percentiles: Tuple[float, float] = (2.0, 98.0)) -> np.ndarray:
        image = self.load_image(image_source)
        grayscale = self._to_grayscale(image)
        resized = cv2.resize(
            grayscale,
            (self.base_config.image_size[1], self.base_config.image_size[0]),
            interpolation=cv2.INTER_AREA
        )

        if blur_kernel > 1:
            kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
            resized = cv2.GaussianBlur(resized, (kernel, kernel), 0)

        normalized = resized.astype(np.float32) / 255.0
        low, high = np.percentile(normalized, clip_percentiles)
        if high - low > 1e-6:
            normalized = np.clip((normalized - low) / (high - low), 0.0, 1.0)
        else:
            normalized = np.clip(normalized, 0.0, 1.0)

        if invert is True or (invert == 'auto' and float(np.mean(normalized)) > 0.55):
            normalized = 1.0 - normalized

        concentration_map = np.power(np.clip(normalized, 0.0, 1.0), 1.15)
        return concentration_map.astype(np.float32)

    def estimate_scene_config(self,
                              concentration_map: np.ndarray,
                              scene_name: str = 'image_derived_scene',
                              max_centers: int = 4) -> VolcanicAshConfig:
        map_array = np.asarray(concentration_map, dtype=np.float32)
        if map_array.ndim != 2:
            raise ValueError('concentration_map must be a 2D array')

        working = VolcanicAshConfig.from_dict(self.base_config.to_dict())
        working.scene_name = scene_name
        working.enable_irregular = False
        working.image_size = tuple(map_array.shape)

        threshold = max(0.08, working.concentration_threshold * 0.45)
        binary = (map_array >= threshold).astype(np.uint8)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

        components = []
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < max(6, map_array.size // 8000):
                continue
            mask = labels == label
            ys, xs = np.where(mask)
            weights = map_array[mask]
            if weights.size == 0:
                continue

            cy = float(np.average(ys, weights=weights))
            cx = float(np.average(xs, weights=weights))
            std_y = float(max(np.std(ys), 8.0))
            std_x = float(max(np.std(xs), 8.0))
            weight = float(np.max(weights))
            components.append({
                'x': cx,
                'y': cy,
                'weight': weight,
                'std_x': std_x,
                'std_y': std_y,
                'area': area
            })

        if not components:
            ys, xs = np.indices(map_array.shape)
            total_weight = float(np.sum(map_array))
            if total_weight <= 1e-6:
                cx = (map_array.shape[1] - 1) / 2.0
                cy = (map_array.shape[0] - 1) / 2.0
            else:
                cy = float(np.sum(ys * map_array) / total_weight)
                cx = float(np.sum(xs * map_array) / total_weight)

            components = [{
                'x': cx,
                'y': cy,
                'weight': float(np.max(map_array)),
                'std_x': float(max(map_array.shape[1] / 8.0, 12.0)),
                'std_y': float(max(map_array.shape[0] / 8.0, 12.0)),
                'area': int(np.sum(map_array > 0.05))
            }]

        components.sort(key=lambda item: (item['weight'], item['area']), reverse=True)
        selected = components[:max_centers]
        total_weight = sum(max(component['weight'], 1e-6) for component in selected)
        working.centers = [
            {
                'x': round(component['x'], 2),
                'y': round(component['y'], 2),
                'weight': round(component['weight'] / total_weight, 4),
                'std_x': round(component['std_x'], 2),
                'std_y': round(component['std_y'], 2)
            }
            for component in selected
        ]
        working.num_centers = len(working.centers)
        return working

    def summarize_map(self, concentration_map: np.ndarray) -> Dict:
        map_array = np.asarray(concentration_map, dtype=np.float32)
        return {
            'shape': [int(map_array.shape[0]), int(map_array.shape[1])],
            'max_concentration': float(np.max(map_array)),
            'mean_concentration': float(np.mean(map_array)),
            'active_area_ratio': float(np.mean(map_array >= self.base_config.concentration_threshold))
        }

    def convert_to_scene(self,
                         image_source: Union[str, np.ndarray],
                         scene_name: str = 'image_derived_scene',
                         invert: Union[bool, str] = 'auto') -> Dict:
        concentration_map = self.image_to_concentration_map(image_source, invert=invert)
        config = self.estimate_scene_config(concentration_map, scene_name=scene_name)
        return {
            'concentration_map': concentration_map,
            'config': config,
            'summary': self.summarize_map(concentration_map)
        }

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image.astype(np.uint8)

        if image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        raise ValueError(f'Unsupported image shape: {image.shape}')
