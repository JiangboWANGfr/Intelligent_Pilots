import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from src.rl_env.volcanic_ash_env import VolcanicAshEnv
from src.path_planning.fallback_planner import FallbackPlanner
from src.rl_training.ddpg_agent import DDPGAgent
from src.config.volcanic_ash_config import VolcanicAshConfig


class PathPlanner:
    def __init__(self, config: VolcanicAshConfig, agent: Optional[DDPGAgent] = None):
        self.config = config
        self.agent = agent
        self.env = VolcanicAshEnv(config)
        self._sync_environment()
         
    def set_agent(self, agent: DDPGAgent):
        self.agent = agent
    
    def _sync_environment(self):
        self.config = self.env.config
        self.ash_model = self.env.ash_model
        self.concentration_map = self.env.concentration_map
        self.fallback_planner = FallbackPlanner(self.config)
    
    def set_external_concentration_map(self,
                                       concentration_map: np.ndarray,
                                       scene_name: Optional[str] = None):
        self.env.set_external_concentration_map(
            concentration_map,
            config=self.config,
            scene_name=scene_name or self.config.scene_name or 'external_image'
        )
        self._sync_environment()
    
    def build_cloud_info(self, concentration_map: Optional[np.ndarray] = None) -> Dict:
        map_array = np.asarray(concentration_map if concentration_map is not None else self.concentration_map,
                               dtype=np.float32)
        weights = np.clip(map_array, 0.0, 1.0)
        if float(np.sum(weights)) <= 1e-6:
            center_y = (map_array.shape[0] - 1) / 2.0
            center_x = (map_array.shape[1] - 1) / 2.0
        else:
            ys, xs = np.indices(map_array.shape)
            total_weight = float(np.sum(weights))
            center_y = float(np.sum(ys * weights) / total_weight)
            center_x = float(np.sum(xs * weights) / total_weight)

        center_lat, center_lon = self.ash_model.pixel_to_geo(int(round(center_x)), int(round(center_y)))
        pixel_degree = ((self.config.geo_span_lat / self.config.image_size[0]) +
                        (self.config.geo_span_lon / self.config.image_size[1])) / 2.0
        thresholds = {
            'high': min(0.95, self.config.concentration_threshold * 1.5),
            'medium': self.config.concentration_threshold,
            'low': max(0.05, self.config.concentration_threshold * 0.5),
            'safe': max(0.02, self.config.concentration_threshold * 0.2)
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
    
    def build_path_data_from_pixel_path(self,
                                        path_points: List[Tuple[float, float]],
                                        planning_method: str = 'fallback',
                                        start_geo: Optional[Tuple[float, float]] = None,
                                        target_geo: Optional[Tuple[float, float]] = None,
                                        start_pixel: Optional[Tuple[int, int]] = None,
                                        target_pixel: Optional[Tuple[int, int]] = None) -> Dict:
        path_data = {
            'waypoints': [],
            'total_reward': 0.0,
            'total_fuel': 0.0,
            'max_concentration': 0.0,
            'success': False,
            'steps_taken': 0,
            'path_coordinates': [],
            'geo_coordinates': [],
            'planning_method': planning_method,
            'scene_name': self.env.scene_name,
            'cloud_info': self.build_cloud_info()
        }

        if not path_points:
            return path_data

        cumulative_reward = 0.0
        cumulative_fuel = 0.0
        previous_point = None
        for step, point in enumerate(path_points):
            py = float(np.clip(point[0], 0, self.config.image_size[0] - 1))
            px = float(np.clip(point[1], 0, self.config.image_size[1] - 1))
            iy = int(round(py))
            ix = int(round(px))
            concentration = float(self.concentration_map[iy, ix])
            if previous_point is None:
                dy = 0.0
                dx = 0.0
                step_distance = 0.0
            else:
                dy = py - previous_point[0]
                dx = px - previous_point[1]
                step_distance = float(np.hypot(dy, dx))
                cumulative_fuel += 0.12 + step_distance * 0.012

            reward_delta = -step_distance * 0.18 - concentration * (35.0 if concentration > self.config.concentration_threshold else 10.0)
            cumulative_reward += reward_delta
            lat, lon = self.ash_model.pixel_to_geo(ix, iy)
            waypoint = {
                'step': step,
                'pixel_x': px,
                'pixel_y': py,
                'latitude': lat,
                'longitude': lon,
                'velocity_x': dx,
                'velocity_y': dy,
                'concentration': concentration,
                'cumulative_reward': cumulative_reward,
                'cumulative_fuel': cumulative_fuel
            }
            path_data['waypoints'].append(waypoint)
            path_data['path_coordinates'].append([px, py])
            path_data['geo_coordinates'].append([lon, lat])
            path_data['max_concentration'] = max(path_data['max_concentration'], concentration)
            previous_point = (py, px)

        final_point = np.array(path_points[-1], dtype=np.float32)
        target_reference = np.array(target_pixel if target_pixel is not None else path_points[-1], dtype=np.float32)
        if float(np.linalg.norm(final_point - target_reference)) <= self.config.success_threshold:
            path_data['success'] = True
            cumulative_reward += 100.0

        path_data['total_reward'] = float(cumulative_reward)
        path_data['total_fuel'] = float(cumulative_fuel)
        path_data['steps_taken'] = len(path_data['waypoints'])
        if start_geo is not None:
            path_data['start_geo'] = list(start_geo)
        if target_geo is not None:
            path_data['target_geo'] = list(target_geo)
        if start_pixel is not None:
            path_data['start_pixel'] = list(start_pixel)
        if target_pixel is not None:
            path_data['target_pixel'] = list(target_pixel)
        return path_data
    
    def plan_path(self, start_pos: Tuple[float, float],
                  target_pos: Tuple[float, float],
                  max_steps: int = 500) -> Dict:
        
        if self.agent is None:
            raise ValueError("No trained agent loaded. Please load a model first.")
        
        self.env.aircraft_pos = np.array(start_pos, dtype=np.float32)
        self.env.target_pos = np.array(target_pos, dtype=np.float32)
        self.env.velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.env.step_count = 0
        self.env.trajectory = [self.env.aircraft_pos.copy()]
        self.env.total_fuel_consumption = 0.0
        self.env.max_concentration_exposure = 0.0
        
        state, _ = self.env._get_observation(), {}
        path_data = {
            'waypoints': [],
            'total_reward': 0,
            'total_fuel': 0,
            'max_concentration': 0,
            'success': False,
            'steps_taken': 0,
            'path_coordinates': [],
            'geo_coordinates': [],
            'planning_method': 'rl',
            'scene_name': self.env.scene_name
        }
        
        for step in range(max_steps):
            action = self.agent.select_action(state, evaluate=True)
            
            next_state, reward, terminated, truncated, info = self.env.step(action)
            
            path_data['total_reward'] += reward
            path_data['total_fuel'] = info['fuel_consumed']
            path_data['max_concentration'] = max(path_data['max_concentration'],
                                                  info['current_concentration'])
            
            current_pos = self.env.aircraft_pos.copy()
            lat, lon = self.ash_model.pixel_to_geo(int(current_pos[1]),
                                                   int(current_pos[0]))
            
            waypoint = {
                'step': step,
                'pixel_x': float(current_pos[1]),
                'pixel_y': float(current_pos[0]),
                'latitude': lat,
                'longitude': lon,
                'velocity_x': float(self.env.velocity[0]),
                'velocity_y': float(self.env.velocity[1]),
                'concentration': float(info['current_concentration']),
                'cumulative_reward': path_data['total_reward'],
                'cumulative_fuel': path_data['total_fuel']
            }
            
            path_data['waypoints'].append(waypoint)
            path_data['path_coordinates'].append([float(current_pos[1]),
                                                  float(current_pos[0])])
            path_data['geo_coordinates'].append([lon, lat])
            
            state = next_state

            if terminated:
                if info['distance_to_target'] < self.config.success_threshold:
                    path_data['success'] = True
                break
            
            if truncated:
                break
        
        path_data['steps_taken'] = len(path_data['waypoints'])
        
        return path_data
    
    def convert_geo_input(self, start_geo: Tuple[float, float],
                         target_geo: Tuple[float, float]) -> Tuple[Tuple[int, int],
                                                                   Tuple[int, int]]:
        def geo_to_pixel(lat, lon):
            img_h, img_w = self.config.image_size
            px = int((lon - self.config.geo_center_lon) / self.config.geo_span_lon *
                     img_w + img_w / 2)
            py = int((0.5 - (lat - self.config.geo_center_lat) / self.config.geo_span_lat) *
                     img_h)
            return (py, px)
        
        start_pixel = geo_to_pixel(*start_geo)
        target_pixel = geo_to_pixel(*target_geo)
        
        return start_pixel, target_pixel
    
    def plan_path_geo(self, start_geo: Tuple[float, float],
                     target_geo: Tuple[float, float],
                     max_steps: int = 500) -> Dict:
        
        start_pixel, target_pixel = self.convert_geo_input(start_geo, target_geo)
        
        result = self.plan_path(tuple(start_pixel), tuple(target_pixel), max_steps)
        
        result['start_geo'] = list(start_geo)
        result['target_geo'] = list(target_geo)
        result['start_pixel'] = list(start_pixel)
        result['target_pixel'] = list(target_pixel)
        result['cloud_info'] = result.get('cloud_info') or self.build_cloud_info()
        
        return result
    
    def plan_path_geo_with_fallback(self,
                                    start_geo: Tuple[float, float],
                                    target_geo: Tuple[float, float],
                                    max_steps: int = 500,
                                    max_concentration: Optional[float] = None) -> Dict:
        fallback_limit = float(max_concentration if max_concentration is not None
                               else self.config.concentration_threshold)
        start_pixel, target_pixel = self.convert_geo_input(start_geo, target_geo)

        rl_error = None
        rl_result = None
        if self.agent is not None:
            try:
                rl_result = self.plan_path_geo(start_geo, target_geo, max_steps=max_steps)
            except Exception as exc:
                rl_error = str(exc)

        if rl_result is not None and rl_result.get('success', False) and \
                rl_result.get('max_concentration', 1.0) <= fallback_limit:
            rl_result['validation_info'] = {
                'used_fallback': False,
                'fallback_reason': '',
                'fallback_limit': fallback_limit
            }
            return rl_result

        fallback_points = self.fallback_planner.plan(
            self.concentration_map,
            tuple(start_pixel),
            tuple(target_pixel),
            max_concentration=fallback_limit,
            desired_points=160
        )
        fallback_result = self.build_path_data_from_pixel_path(
            fallback_points,
            planning_method='fallback',
            start_geo=start_geo,
            target_geo=target_geo,
            start_pixel=start_pixel,
            target_pixel=target_pixel
        )
        fallback_result['validation_info'] = {
            'used_fallback': True,
            'fallback_reason': 'rl_agent_unavailable' if self.agent is None else ('rl_error' if rl_error else 'rl_rejected'),
            'fallback_limit': fallback_limit,
            'fallback_summary': self.fallback_planner.summarize_path(self.concentration_map, fallback_points)
        }
        if rl_result is not None:
            fallback_result['validation_info']['rl_attempt'] = {
                'success': rl_result.get('success', False),
                'max_concentration': rl_result.get('max_concentration', 0.0),
                'steps_taken': rl_result.get('steps_taken', 0)
            }
        if rl_error:
            fallback_result['validation_info']['rl_error'] = rl_error
        return fallback_result

    def export_path_json(self, path_data: Dict, filepath: str):
        import numpy as np
        
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                elif isinstance(obj, (np.floating,)):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)
        
        output_data = {
            'planning_info': {
                'timestamp': str(datetime.now()),
                'success': path_data['success'],
                'planning_method': path_data.get('planning_method', 'rl'),
                'scene_name': path_data.get('scene_name', self.config.scene_name or self.config.model_type),
                'total_waypoints': len(path_data['waypoints']),
                'total_steps': path_data['steps_taken'],
                'summary': {
                    'total_reward': float(path_data['total_reward']),
                    'total_fuel_consumption': float(path_data['total_fuel']),
                    'max_concentration_exposure': float(path_data.get('max_concentration', 0))
                }
            },
            'path_data': path_data,
            'cloud_info': path_data.get('cloud_info'),
            'validation_info': path_data.get('validation_info')
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
    
    def visualize_path(self, path_data: Dict, save_path: Optional[str] = None):
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        axes[0].imshow(self.concentration_map, cmap='gray', alpha=0.8)
        axes[0].set_title('Concentration Map with Planned Path')
        
        if len(path_data['path_coordinates']) > 0:
            coords = np.array(path_data['path_coordinates'])
            axes[0].plot(coords[:, 0], coords[:, 1], 'b-', linewidth=2,
                        label='Planned Path')
            axes[0].plot(coords[0, 0], coords[0, 1], 'go', markersize=12,
                        label='Start')
            axes[0].plot(coords[-1, 0], coords[-1, 1], 'r*', markersize=15,
                        label='Target')
        
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        if len(path_data['waypoints']) > 0:
            steps = [w['step'] for w in path_data['waypoints']]
            concentrations = [w['concentration'] for w in path_data['waypoints']]
            rewards = [w['cumulative_reward'] for w in path_data['waypoints']]
            fuels = [w['cumulative_fuel'] for w in path_data['waypoints']]
            
            ax2 = axes[1]
            color = 'tab:blue'
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Concentration', color=color)
            ax2.plot(steps, concentrations, color=color, label='Concentration')
            ax2.tick_params(axis='y', labelcolor=color)
            ax2.set_ylim(0, 1)
            
            ax3 = ax2.twinx()
            color = 'tab:red'
            ax3.set_ylabel('Cumulative Reward/Fuel', color=color)
            ax3.plot(steps, rewards, color='tab:red', linestyle='-',
                    label='Reward', alpha=0.7)
            ax3.plot(steps, fuels, color='tab:green', linestyle='--',
                    label='Fuel', alpha=0.7)
            ax3.tick_params(axis='y', labelcolor=color)
            
            lines1, labels1 = ax2.get_legend_handles_labels()
            lines2, labels2 = ax3.get_legend_handles_labels()
            ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
            
            axes[1].set_title('Path Metrics Over Time')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Path visualization saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()


from datetime import datetime
