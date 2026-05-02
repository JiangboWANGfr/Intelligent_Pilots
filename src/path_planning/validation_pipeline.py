import os
from typing import Dict, Optional, Tuple, Union

import numpy as np

from src.config.volcanic_ash_config import VolcanicAshConfig
from src.generation.image_converter import AshImageConverter
from src.path_planning.animation_exporter import ValidationAnimationExporter
from src.path_planning.fallback_planner import FallbackPlanner
from src.path_planning.planner import PathPlanner
from src.rl_env.volcanic_ash_env import VolcanicAshEnv
from src.rl_training.ddpg_agent import DDPGAgent, create_agent, infer_checkpoint_algorithm


class ValidationPipeline:
    def __init__(self,
                 config: VolcanicAshConfig,
                 model_path: Optional[str] = None,
                 agent: Optional[DDPGAgent] = None):
        self.base_config = VolcanicAshConfig.from_dict(config.to_dict())
        self.model_path = model_path
        self.converter = AshImageConverter(self.base_config)
        self.fallback_planner = FallbackPlanner(self.base_config)
        self.agent = agent
        self.last_validation_context: Optional[Dict] = None

    def ensure_agent(self) -> Optional[DDPGAgent]:
        if self.agent is not None:
            return self.agent

        if not self.model_path or not os.path.exists(self.model_path):
            return None

        temp_env = VolcanicAshEnv(self.base_config)
        state_dim = len(DDPGAgent.flatten_state(temp_env.reset()[0]))
        action_dim = int(np.prod(temp_env.action_space.shape))
        algorithm = infer_checkpoint_algorithm(self.model_path)
        agent = create_agent(algorithm, state_dim=state_dim, action_dim=action_dim)
        agent.load_model(self.model_path)
        self.agent = agent
        return self.agent

    def validate_image(self,
                       image_source: Union[str, np.ndarray],
                       start_geo: Tuple[float, float],
                       target_geo: Tuple[float, float],
                       output_json_path: Optional[str] = None,
                       output_plot_path: Optional[str] = None,
                       scene_name: str = 'image_validation_scene',
                       fallback_concentration_limit: Optional[float] = None,
                       animation_output_dir: Optional[str] = None,
                       animation_gif_path: Optional[str] = None,
                       animation_video_path: Optional[str] = None,
                       animation_fps: int = 12,
                       animation_save_frames: bool = False,
                       animation_max_frames: int = 180) -> Dict:
        converted = self.converter.convert_to_scene(image_source, scene_name=scene_name)
        derived_config = converted['config']
        concentration_map = converted['concentration_map']

        planner = PathPlanner(derived_config, self.ensure_agent())
        planner.set_external_concentration_map(concentration_map, scene_name=scene_name)

        start_pixel, target_pixel = planner.convert_geo_input(start_geo, target_geo)
        max_allowed = float(
            fallback_concentration_limit
            if fallback_concentration_limit is not None
            else derived_config.concentration_threshold
        )

        rl_result = None
        used_fallback = False
        fallback_reason = ''
        if planner.agent is not None:
            try:
                rl_result = planner.plan_path_geo(start_geo, target_geo, max_steps=500)
            except Exception as exc:
                fallback_reason = f'rl_error:{exc}'

        if rl_result is None:
            used_fallback = True
            if not fallback_reason:
                fallback_reason = 'rl_agent_unavailable'
        elif (not rl_result.get('success', False) or
              rl_result.get('max_concentration', 1.0) > max_allowed):
            used_fallback = True
            fallback_reason = 'rl_path_rejected'

        if used_fallback:
            fallback_points = self.fallback_planner.plan(
                concentration_map,
                start_pixel,
                target_pixel,
                max_concentration=max_allowed,
                desired_points=160
            )
            path_result = planner.build_path_data_from_pixel_path(
                fallback_points,
                planning_method='fallback',
                start_geo=start_geo,
                target_geo=target_geo,
                start_pixel=start_pixel,
                target_pixel=target_pixel
            )
            fallback_summary = self.fallback_planner.summarize_path(concentration_map, fallback_points)
            path_result['validation_info'] = {
                'used_fallback': True,
                'fallback_reason': fallback_reason,
                'fallback_summary': fallback_summary,
                'source_image': image_source if isinstance(image_source, str) else 'array_input',
                'conversion_summary': converted['summary'],
                'scene_name': scene_name
            }
            if rl_result is not None:
                path_result['validation_info']['rl_attempt'] = {
                    'success': rl_result.get('success', False),
                    'max_concentration': rl_result.get('max_concentration', 0.0),
                    'steps_taken': rl_result.get('steps_taken', 0)
                }
        else:
            path_result = rl_result
            path_result['validation_info'] = {
                'used_fallback': False,
                'fallback_reason': '',
                'source_image': image_source if isinstance(image_source, str) else 'array_input',
                'conversion_summary': converted['summary'],
                'scene_name': scene_name
            }

        path_result['cloud_info'] = path_result.get('cloud_info') or planner.build_cloud_info(concentration_map)
        path_result['scene_name'] = scene_name
        path_result['planning_method'] = path_result.get('planning_method', 'rl')

        self.last_validation_context = {
            'config': VolcanicAshConfig.from_dict(derived_config.to_dict()),
            'concentration_map': np.array(concentration_map, copy=True),
            'scene_name': scene_name
        }

        if output_json_path:
            os.makedirs(os.path.dirname(output_json_path) or '.', exist_ok=True)
            planner.export_path_json(path_result, output_json_path)

        if output_plot_path:
            os.makedirs(os.path.dirname(output_plot_path) or '.', exist_ok=True)
            planner.visualize_path(path_result, save_path=output_plot_path)

        if animation_output_dir or animation_gif_path or animation_video_path or animation_save_frames:
            animation_export = self.export_validation_animation(
                path_result,
                output_dir=animation_output_dir or 'output/validation_animation',
                gif_path=animation_gif_path,
                video_path=animation_video_path,
                fps=animation_fps,
                save_frames=animation_save_frames,
                max_frames=animation_max_frames
            )
            path_result['validation_info']['animation_export'] = animation_export

        return path_result

    def export_validation_animation(self,
                                  path_result: Dict,
                                  output_dir: str = 'output/validation_animation',
                                  gif_path: Optional[str] = None,
                                  video_path: Optional[str] = None,
                                  fps: int = 12,
                                  save_frames: bool = False,
                                  max_frames: int = 180) -> Dict:
        if self.last_validation_context is None:
            raise ValueError('No validated scene is available for animation export')

        exporter = ValidationAnimationExporter(
            config=self.last_validation_context['config'],
            concentration_map=self.last_validation_context['concentration_map']
        )
        return exporter.export(
            path_result=path_result,
            output_dir=output_dir,
            gif_path=gif_path,
            video_path=video_path,
            fps=fps,
            save_frames=save_frames,
            max_frames=max_frames
        )

    def build_animation_export_manifest(self,
                                        path_result: Dict,
                                        output_dir: str = 'output/validation_frames') -> Dict:
        os.makedirs(output_dir, exist_ok=True)
        frame_manifest = []
        for index, waypoint in enumerate(path_result.get('waypoints', [])):
            frame_manifest.append({
                'frame_index': index,
                'pixel_x': waypoint.get('pixel_x'),
                'pixel_y': waypoint.get('pixel_y'),
                'longitude': waypoint.get('longitude'),
                'latitude': waypoint.get('latitude'),
                'concentration': waypoint.get('concentration')
            })

        return {
            'output_dir': output_dir,
            'frame_count': len(frame_manifest),
            'planning_method': path_result.get('planning_method', 'rl'),
            'scene_name': path_result.get('scene_name', self.base_config.scene_name or self.base_config.model_type),
            'frames': frame_manifest
        }
