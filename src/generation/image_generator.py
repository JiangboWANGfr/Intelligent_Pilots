import numpy as np
import cv2
import json
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from src.model.gmm_model import GMMVolcanicAshModel
from src.config.volcanic_ash_config import VolcanicAshConfig


class DynamicSimulation:
    def __init__(self, base_model: GMMVolcanicAshModel):
        self.base_model = base_model
        self.config = base_model.config
        
    def simulate_displacement(self, frame_index: int, total_frames: int,
                             drift_direction: float = 45.0,
                             drift_speed: float = 5.0) -> List[Dict]:
        displaced_centers = []
        progress = frame_index / max(total_frames - 1, 1)
        
        drift_rad = np.radians(drift_direction)
        dx = drift_speed * progress * np.cos(drift_rad)
        dy = drift_speed * progress * np.sin(drift_rad)
        
        for center in self.config.centers:
            new_center = center.copy()
            new_center['x'] += dx
            new_center['y'] += dy
            
            deform_scale = 1.0 + 0.2 * progress
            new_center['std_x'] *= deform_scale
            new_center['std_y'] *= deform_scale
            
            decay_factor = np.exp(-0.3 * progress)
            new_center['weight'] *= decay_factor
            
            displaced_centers.append(new_center)
        
        return displaced_centers
    
    def generate_dynamic_sequence(self, num_frames: int = 30,
                                  drift_direction: float = 45.0,
                                  drift_speed: float = 5.0,
                                  output_dir: str = 'output/dynamic') -> Dict:
        os.makedirs(output_dir, exist_ok=True)
        
        dynamic_data = {
            'simulation_info': {
                'total_frames': num_frames,
                'drift_direction': drift_direction,
                'drift_speed': drift_speed,
                'timestamp': datetime.now().isoformat(),
                'base_config': self.config.to_dict()
            },
            'frames': []
        }
        
        for i in range(num_frames):
            displaced_centers = self.simulate_displacement(i, num_frames,
                                                          drift_direction, drift_speed)
            
            frame_config = VolcanicAshConfig(
                model_type=self.config.model_type,
                num_centers=self.config.num_centers,
                cloud_size=self.config.cloud_size,
                concentration_threshold=self.config.concentration_threshold,
                mass_ratio=self.config.mass_ratio,
                image_size=self.config.image_size,
                geo_center_lat=self.config.geo_center_lat,
                geo_center_lon=self.config.geo_center_lon,
                geo_span_lat=self.config.geo_span_lat,
                geo_span_lon=self.config.geo_span_lon,
                centers=displaced_centers
            )
            
            frame_model = GMMVolcanicAshModel(frame_config)
            conc_map = frame_model.generate_concentration_map()
            
            if not frame_model.is_valid_image(conc_map):
                continue
            
            grayscale = frame_model.generate_grayscale_image(conc_map)
            danger_zone = frame_model.generate_danger_zone_image(conc_map)
            
            grayscale_path = os.path.join(output_dir, f'frame_{i:03d}_grayscale.png')
            danger_path = os.path.join(output_dir, f'frame_{i:03d}_danger.png')
            
            cv2.imwrite(grayscale_path, grayscale)
            cv2.imwrite(danger_path, cv2.cvtColor(danger_zone, cv2.COLOR_RGB2BGR))
            
            geo_data = frame_model.generate_geo_data(conc_map)
            
            frame_info = {
                'frame_id': i,
                'grayscale_path': grayscale_path,
                'danger_zone_path': danger_path,
                'geo_data': geo_data,
                'centers': displaced_centers
            }
            dynamic_data['frames'].append(frame_info)
        
        json_output_path = os.path.join(output_dir, 'dynamic_data.json')
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump(dynamic_data, f, ensure_ascii=False, indent=2)
        
        dynamic_data['json_path'] = json_output_path
        return dynamic_data


class StaticImageGenerator:
    def __init__(self, config: VolcanicAshConfig):
        self.config = config
        self.model = GMMVolcanicAshModel(config)
        
    def generate_static_images(self, output_dir: str = 'output/static',
                               num_images: int = 10) -> Dict:
        os.makedirs(output_dir, exist_ok=True)
        
        results = {
            'generated_images': [],
            'failed_count': 0,
            'output_directory': output_dir
        }
        
        for i in range(num_images):
            noisy_centers = []
            for center in self.config.centers:
                noisy_center = center.copy()
                noise_scale = min(center['std_x'], center['std_y']) * 0.1
                noisy_center['x'] += np.random.normal(0, noise_scale)
                noisy_center['y'] += np.random.normal(0, noise_scale)
                noisy_center['std_x'] *= np.random.uniform(0.9, 1.1)
                noisy_center['std_y'] *= np.random.uniform(0.9, 1.1)
                noisy_centers.append(noisy_center)
            
            temp_config = VolcanicAshConfig(
                model_type=self.config.model_type,
                num_centers=self.config.num_centers,
                cloud_size=self.config.cloud_size,
                concentration_threshold=self.config.concentration_threshold,
                mass_ratio=self.config.mass_ratio,
                image_size=self.config.image_size,
                geo_center_lat=self.config.geo_center_lat,
                geo_center_lon=self.config.geo_center_lon,
                geo_span_lat=self.config.geo_span_lat,
                geo_span_lon=self.config.geo_span_lon,
                centers=noisy_centers
            )
            
            temp_model = GMMVolcanicAshModel(temp_config)
            conc_map = temp_model.generate_concentration_map()
            
            if not temp_model.is_valid_image(conc_map):
                results['failed_count'] += 1
                continue
            
            grayscale = temp_model.generate_grayscale_image(conc_map)
            danger_zone = temp_model.generate_danger_zone_image(conc_map)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            grayscale_path = os.path.join(output_dir, f'{timestamp}_{i:03d}_grayscale.png')
            danger_path = os.path.join(output_dir, f'{timestamp}_{i:03d}_danger.png')
            
            cv2.imwrite(grayscale_path, grayscale)
            cv2.imwrite(danger_path, cv2.cvtColor(danger_zone, cv2.COLOR_RGB2BGR))
            
            geo_data = temp_model.generate_geo_data(conc_map)
            geo_json_path = os.path.join(output_dir, f'{timestamp}_{i:03d}_geo.json')
            with open(geo_json_path, 'w', encoding='utf-8') as f:
                json.dump(geo_data, f, ensure_ascii=False, indent=2)
            
            image_info = {
                'id': i,
                'grayscale_path': grayscale_path,
                'danger_zone_path': danger_path,
                'geo_json_path': geo_json_path,
                'max_concentration': float(np.max(conc_map)),
                'mean_concentration': float(np.mean(conc_map)),
                'high_risk_area_percent': float(np.sum(conc_map > self.config.concentration_threshold) / conc_map.size * 100)
            }
            results['generated_images'].append(image_info)
        
        summary_path = os.path.join(output_dir, 'generation_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        return results
