import json
import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional

import cv2
import numpy as np
from werkzeug.utils import secure_filename

from src.config.volcanic_ash_config import VolcanicAshConfig
from src.generation.image_converter import AshImageConverter


class WebCaseManager:
    def __init__(self, output_dir: str):
        self.output_dir = os.path.abspath(output_dir)
        self.scenes_dir = os.path.join(self.output_dir, 'web_scenes')
        self.cases_dir = os.path.join(self.output_dir, 'web_cases')
        os.makedirs(self.scenes_dir, exist_ok=True)
        os.makedirs(self.cases_dir, exist_ok=True)

    def create_scene_id(self, scene_name: str) -> str:
        slug = secure_filename(scene_name) or 'ash_scene'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        return f'{slug}_{timestamp}'

    def create_case_id(self, scene_name: str) -> str:
        slug = secure_filename(scene_name) or 'planning_case'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        return f'{slug}_{timestamp}'

    def scene_dir(self, scene_id: str) -> str:
        return os.path.join(self.scenes_dir, secure_filename(scene_id))

    def case_dir(self, case_id: str) -> str:
        return os.path.join(self.cases_dir, secure_filename(case_id))

    def save_imported_scene(self,
                            image_path: str,
                            config: VolcanicAshConfig,
                            scene_name: str,
                            conversion_mode: str = 'auto',
                            invert='auto',
                            blur_kernel: int = 5) -> Dict:
        converter = AshImageConverter(config)
        converted = converter.convert_to_scene(
            image_path,
            scene_name=scene_name,
            mode=conversion_mode,
            invert=invert,
            blur_kernel=blur_kernel
        )
        scene_id = self.create_scene_id(scene_name)
        scene_dir = self.scene_dir(scene_id)
        os.makedirs(scene_dir, exist_ok=True)

        source_ext = os.path.splitext(image_path)[1] or '.png'
        source_path = os.path.join(scene_dir, f'source_image{source_ext}')
        shutil.copy2(image_path, source_path)
        outputs = converter.save_standard_outputs(
            converted['concentration_map'],
            scene_dir,
            prefix='concentration'
        )
        overlay_path = os.path.join(scene_dir, 'concentration_overlay.png')
        self.save_overlay_png(converted['concentration_map'], overlay_path)
        config_path = os.path.join(scene_dir, 'scene_config.json')
        converted['config'].save(config_path)

        manifest = self._build_scene_manifest(
            scene_id=scene_id,
            scene_name=scene_name,
            scene_type='imported_image',
            config=converted['config'],
            summary=converted['summary'],
            source_image=source_path,
            concentration_map=outputs['npy_path'],
            concentration_png=outputs['grayscale_path'],
            preview_image=outputs['preview_path'],
            overlay_image=overlay_path,
            config_path=config_path,
            extra={
                'conversion_mode': conversion_mode,
                'blur_kernel': blur_kernel
            }
        )
        self._write_json(os.path.join(scene_dir, 'scene.json'), manifest)
        return manifest

    def save_generated_scene(self,
                             concentration_map: np.ndarray,
                             config: VolcanicAshConfig,
                             scene_name: str,
                             scene_type: str = 'generated_random',
                             extra: Optional[Dict] = None) -> Dict:
        scene_id = self.create_scene_id(scene_name)
        scene_dir = self.scene_dir(scene_id)
        os.makedirs(scene_dir, exist_ok=True)
        converter = AshImageConverter(config)
        map_array = np.clip(np.asarray(concentration_map, dtype=np.float32), 0.0, 1.0)
        outputs = converter.save_standard_outputs(map_array, scene_dir, prefix='concentration')
        overlay_path = os.path.join(scene_dir, 'concentration_overlay.png')
        self.save_overlay_png(map_array, overlay_path)
        config_path = os.path.join(scene_dir, 'scene_config.json')
        config.save(config_path)
        manifest = self._build_scene_manifest(
            scene_id=scene_id,
            scene_name=scene_name,
            scene_type=scene_type,
            config=config,
            summary=converter.summarize_map(map_array),
            source_image=None,
            concentration_map=outputs['npy_path'],
            concentration_png=outputs['grayscale_path'],
            preview_image=outputs['preview_path'],
            overlay_image=overlay_path,
            config_path=config_path,
            extra=extra or {}
        )
        self._write_json(os.path.join(scene_dir, 'scene.json'), manifest)
        return manifest

    def _build_scene_manifest(self,
                              scene_id: str,
                              scene_name: str,
                              scene_type: str,
                              config: VolcanicAshConfig,
                              summary: Dict,
                              source_image: Optional[str],
                              concentration_map: str,
                              concentration_png: str,
                              preview_image: str,
                              overlay_image: str,
                              config_path: str,
                              extra: Dict) -> Dict:
        return {
            'scene_id': scene_id,
            'scene_name': scene_name,
            'scene_type': scene_type,
            'created_at': datetime.now().isoformat(timespec='seconds'),
            'config': config.to_dict(),
            'summary': summary,
            'cloud_info': self.build_cloud_info_from_map(np.load(concentration_map), config),
            'files': {
                'source_image': source_image,
                'concentration_map': concentration_map,
                'concentration_png': concentration_png,
                'preview_image': preview_image,
                'overlay_image': overlay_image,
                'config': config_path
            },
            'extra': extra
        }

    @staticmethod
    def save_overlay_png(concentration_map: np.ndarray, output_path: str) -> str:
        map_array = np.clip(np.asarray(concentration_map, dtype=np.float32), 0.0, 1.0)
        rgba = np.zeros((*map_array.shape, 4), dtype=np.uint8)

        safe = (map_array > 0.02) & (map_array < 0.20)
        low = (map_array >= 0.20) & (map_array < 0.50)
        medium = (map_array >= 0.50) & (map_array < 0.80)
        high = map_array >= 0.80

        rgba[safe] = [0, 255, 0, 78]
        rgba[low] = [255, 255, 0, 120]
        rgba[medium] = [255, 165, 0, 155]
        rgba[high] = [255, 0, 0, 190]
        rgba[:, :, 3] = np.maximum(rgba[:, :, 3], np.clip(map_array * 165, 0, 190).astype(np.uint8))

        bgra = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(output_path, bgra)
        return output_path

    @staticmethod
    def build_cloud_info_from_map(concentration_map: np.ndarray,
                                  config: VolcanicAshConfig) -> Dict:
        map_array = np.asarray(concentration_map, dtype=np.float32)
        weights = np.clip(map_array, 0.0, 1.0)
        height, width = map_array.shape[:2]

        if float(np.sum(weights)) <= 1e-6:
            center_y = (height - 1) / 2.0
            center_x = (width - 1) / 2.0
        else:
            ys, xs = np.indices((height, width))
            total_weight = float(np.sum(weights))
            center_y = float(np.sum(ys * weights) / total_weight)
            center_x = float(np.sum(xs * weights) / total_weight)

        center_lon = config.geo_center_lon + (center_x / max(width, 1) - 0.5) * config.geo_span_lon
        center_lat = config.geo_center_lat + (0.5 - center_y / max(height, 1)) * config.geo_span_lat
        pixel_degree = ((config.geo_span_lat / max(height, 1)) +
                        (config.geo_span_lon / max(width, 1))) / 2.0
        thresholds = {
            'high': min(0.95, config.concentration_threshold * 1.5),
            'medium': config.concentration_threshold,
            'low': max(0.05, config.concentration_threshold * 0.5),
            'safe': max(0.02, config.concentration_threshold * 0.2)
        }
        radii = {}
        previous_radius = 0.0
        for level in ['high', 'medium', 'low', 'safe']:
            area = float(np.sum(map_array >= thresholds[level]))
            radius = float(np.sqrt(area / np.pi) * pixel_degree) if area > 0 else previous_radius
            radii[level] = round(max(radius, previous_radius), 4)
            previous_radius = radii[level]

        return {
            'center_lat': float(center_lat),
            'center_lon': float(center_lon),
            'zone_radii': {
                'high': radii['high'],
                'medium': radii['medium'],
                'low': radii['low'],
                'safe': radii['safe']
            }
        }

    def load_scene(self, scene_id: str) -> Dict:
        path = os.path.join(self.scene_dir(scene_id), 'scene.json')
        if not os.path.exists(path):
            raise FileNotFoundError(f'Scene not found: {scene_id}')
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def load_scene_map(self, scene_manifest: Dict) -> np.ndarray:
        map_path = scene_manifest.get('files', {}).get('concentration_map')
        if not map_path or not os.path.exists(map_path):
            raise FileNotFoundError('Scene concentration map is missing')
        return np.load(map_path).astype(np.float32)

    def list_scenes(self) -> List[Dict]:
        scenes = []
        for name in sorted(os.listdir(self.scenes_dir), reverse=True):
            path = os.path.join(self.scenes_dir, name, 'scene.json')
            if not os.path.exists(path):
                continue
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    scenes.append(json.load(f))
            except (OSError, json.JSONDecodeError):
                continue
        return scenes

    def create_case_dir(self, scene_name: str) -> Dict[str, str]:
        case_id = self.create_case_id(scene_name)
        case_dir = self.case_dir(case_id)
        os.makedirs(case_dir, exist_ok=True)
        return {'case_id': case_id, 'case_dir': case_dir}

    def save_case(self, case_id: str, manifest: Dict) -> Dict:
        case_dir = self.case_dir(case_id)
        os.makedirs(case_dir, exist_ok=True)
        manifest_path = os.path.join(case_dir, 'case.json')
        self._write_json(manifest_path, manifest)
        return manifest

    def load_case(self, case_id: str) -> Dict:
        path = os.path.join(self.case_dir(case_id), 'case.json')
        if not os.path.exists(path):
            raise FileNotFoundError(f'Case not found: {case_id}')
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def list_cases(self) -> List[Dict]:
        cases = []
        for name in sorted(os.listdir(self.cases_dir), reverse=True):
            path = os.path.join(self.cases_dir, name, 'case.json')
            if not os.path.exists(path):
                continue
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    cases.append(json.load(f))
            except (OSError, json.JSONDecodeError):
                continue
        return cases

    @staticmethod
    def copy_scene_files_to_case(scene_manifest: Dict, case_dir: str) -> Dict[str, str]:
        copied = {}
        for key, source in scene_manifest.get('files', {}).items():
            if not source or not os.path.exists(source):
                continue
            filename = f'scene_{key}{os.path.splitext(source)[1]}'
            dest = os.path.join(case_dir, filename)
            shutil.copy2(source, dest)
            copied[key] = dest
        return copied

    @staticmethod
    def save_dynamic_map_preview(concentration_map: np.ndarray,
                                 config: VolcanicAshConfig,
                                 output_path: str) -> str:
        converter = AshImageConverter(config)
        preview = converter.build_classified_preview(concentration_map)
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        cv2.imwrite(output_path, cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
        return output_path

    @staticmethod
    def _write_json(path: str, data: Dict):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
