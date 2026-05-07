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
                              config_path: str,
                              extra: Dict) -> Dict:
        return {
            'scene_id': scene_id,
            'scene_name': scene_name,
            'scene_type': scene_type,
            'created_at': datetime.now().isoformat(timespec='seconds'),
            'config': config.to_dict(),
            'summary': summary,
            'files': {
                'source_image': source_image,
                'concentration_map': concentration_map,
                'concentration_png': concentration_png,
                'preview_image': preview_image,
                'config': config_path
            },
            'extra': extra
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
