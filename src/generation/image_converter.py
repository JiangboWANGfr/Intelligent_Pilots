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
                                   clip_percentiles: Tuple[float, float] = (2.0, 98.0),
                                   mode: str = 'auto',
                                   plume_scale: float = 1.0) -> np.ndarray:
        image = self.load_image(image_source)
        conversion_mode = str(mode or 'auto').lower()
        if conversion_mode == 'auto':
            conversion_mode = self._choose_conversion_mode(image)
        if conversion_mode == 'color':
            concentration_map = self._image_to_concentration_map_color(
                image,
                blur_kernel=blur_kernel
            )
            return self._scale_plume(concentration_map, plume_scale)
        if conversion_mode not in {'grayscale', 'gray'}:
            raise ValueError(f'Unsupported conversion mode: {mode}')

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
        return self._scale_plume(concentration_map.astype(np.float32), plume_scale)

    def _choose_conversion_mode(self, image: np.ndarray) -> str:
        if image.ndim != 3 or image.shape[2] < 3:
            return 'grayscale'
        bgr = image[:, :, :3]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].astype(np.float32) / 255.0
        value = hsv[:, :, 2].astype(np.float32) / 255.0
        colorful_ratio = float(np.mean((saturation > 0.18) & (value > 0.18)))
        return 'color' if colorful_ratio > 0.01 else 'grayscale'

    def _image_to_concentration_map_color(self,
                                          image: np.ndarray,
                                          blur_kernel: int = 3) -> np.ndarray:
        if image.ndim == 2:
            return self.image_to_concentration_map(
                image,
                mode='grayscale',
                blur_kernel=blur_kernel
            )
        if image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if image.ndim != 3 or image.shape[2] < 3:
            raise ValueError(f'Unsupported image shape for color conversion: {image.shape}')

        bgr = cv2.resize(
            image[:, :, :3],
            (self.base_config.image_size[1], self.base_config.image_size[0]),
            interpolation=cv2.INTER_AREA
        )
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0].astype(np.float32)
        saturation = hsv[:, :, 1].astype(np.float32) / 255.0
        value = hsv[:, :, 2].astype(np.float32) / 255.0

        concentration = np.zeros(hue.shape, dtype=np.float32)
        colorful = (saturation > 0.18) & (value > 0.15)
        dark_linework = value < 0.18
        bright_background = (value > 0.86) & (saturation < 0.18)
        plume_mask = colorful & ~dark_linework & ~bright_background

        red = (hue <= 10) | (hue >= 170)
        orange = (hue > 10) & (hue <= 25)
        yellow = (hue > 25) & (hue <= 40)
        green = (hue > 40) & (hue <= 85)
        cyan_blue = (hue > 85) & (hue <= 135)
        violet = (hue > 135) & (hue < 170)

        concentration[plume_mask & red] = 1.0
        concentration[plume_mask & orange] = 0.78
        concentration[plume_mask & yellow] = 0.55
        concentration[plume_mask & green] = 0.30
        concentration[plume_mask & cyan_blue] = 0.24
        concentration[plume_mask & violet] = 0.42

        # Preserve faint colored plume edges without treating map grid/coastline as ash.
        edge_strength = np.clip((saturation - 0.18) / 0.55, 0.0, 1.0)
        concentration = np.maximum(concentration, plume_mask.astype(np.float32) * edge_strength * 0.22)

        if blur_kernel > 1:
            kernel = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
            concentration = cv2.GaussianBlur(concentration, (kernel, kernel), 0)

        max_value = float(np.max(concentration))
        if max_value > 1e-6:
            concentration = concentration / max_value
        return np.clip(concentration, 0.0, 1.0).astype(np.float32)

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
                         invert: Union[bool, str] = 'auto',
                         mode: str = 'auto',
                         blur_kernel: int = 5,
                         plume_scale: float = 1.0) -> Dict:
        concentration_map = self.image_to_concentration_map(
            image_source,
            invert=invert,
            mode=mode,
            blur_kernel=blur_kernel,
            plume_scale=plume_scale
        )
        config = self.estimate_scene_config(concentration_map, scene_name=scene_name)
        return {
            'concentration_map': concentration_map,
            'config': config,
            'summary': self.summarize_map(concentration_map)
        }

    def _scale_plume(self, concentration_map: np.ndarray, plume_scale: float) -> np.ndarray:
        scale = float(plume_scale or 1.0)
        if abs(scale - 1.0) <= 1e-6:
            return np.asarray(concentration_map, dtype=np.float32)

        map_array = np.clip(np.asarray(concentration_map, dtype=np.float32), 0.0, 1.0)
        height, width = map_array.shape
        weights = np.clip(map_array, 0.0, 1.0)
        if float(np.sum(weights)) <= 1e-6:
            center_x = (width - 1) / 2.0
            center_y = (height - 1) / 2.0
        else:
            ys, xs = np.indices(map_array.shape)
            total_weight = float(np.sum(weights))
            center_x = float(np.sum(xs * weights) / total_weight)
            center_y = float(np.sum(ys * weights) / total_weight)

        matrix = cv2.getRotationMatrix2D((center_x, center_y), 0.0, scale)
        scaled = cv2.warpAffine(
            map_array,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0
        )
        max_value = float(np.max(scaled))
        if max_value > 1e-6:
            scaled = scaled / max_value
        return np.clip(scaled, 0.0, 1.0).astype(np.float32)

    def save_standard_outputs(self,
                              concentration_map: np.ndarray,
                              output_dir: str,
                              prefix: str = 'real_ash') -> Dict[str, str]:
        os.makedirs(output_dir, exist_ok=True)
        map_array = np.clip(np.asarray(concentration_map, dtype=np.float32), 0.0, 1.0)
        npy_path = os.path.join(output_dir, f'{prefix}_concentration.npy')
        grayscale_path = os.path.join(output_dir, f'{prefix}_concentration.png')
        preview_path = os.path.join(output_dir, f'{prefix}_classified_preview.png')

        np.save(npy_path, map_array)
        cv2.imwrite(grayscale_path, (map_array * 255.0).astype(np.uint8))
        cv2.imwrite(preview_path, cv2.cvtColor(self.build_classified_preview(map_array), cv2.COLOR_RGB2BGR))
        return {
            'npy_path': npy_path,
            'grayscale_path': grayscale_path,
            'preview_path': preview_path
        }

    def build_classified_preview(self, concentration_map: np.ndarray) -> np.ndarray:
        map_array = np.clip(np.asarray(concentration_map, dtype=np.float32), 0.0, 1.0)
        threshold = float(getattr(self.base_config, 'concentration_threshold', 0.3))
        rgb = np.full((map_array.shape[0], map_array.shape[1], 3), 245, dtype=np.uint8)
        rgb[(map_array >= threshold * 0.15) & (map_array < threshold * 0.5)] = (0, 255, 0)
        rgb[(map_array >= threshold * 0.5) & (map_array < threshold)] = (255, 255, 0)
        rgb[(map_array >= threshold) & (map_array < threshold * 1.5)] = (255, 165, 0)
        rgb[map_array >= threshold * 1.5] = (255, 0, 0)
        return rgb

    def _to_grayscale(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return image.astype(np.uint8)

        if image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        raise ValueError(f'Unsupported image shape: {image.shape}')
