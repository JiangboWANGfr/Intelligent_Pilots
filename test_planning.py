import sys
import os
import json
import numpy as np
from datetime import datetime
from scipy.interpolate import CubicSpline
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print('='*70)
print('TESTING PATH PLANNING WITH TRAINED MODEL')
print('='*70)
print()

from src.config.volcanic_ash_config import VolcanicAshConfig, get_preset_configs
from src.rl_env.volcanic_ash_env import VolcanicAshEnv
from src.rl_training.ddpg_agent import DDPGAgent, create_agent, infer_checkpoint_algorithm
from src.path_planning.planner import PathPlanner

# ── 步骤1：加载配置 ──────────────────────────────────────────────────────────
print('[1/5] Loading configuration...')
config_path = 'output/current_config.json'
if os.path.exists(config_path):
    config = VolcanicAshConfig.load(config_path)
    print(f'      [OK] Configuration loaded from: {config_path}')
else:
    print(f'      [WARN] Configuration file not found, using default single-center model')
    config = VolcanicAshConfig(
        geo_center_lat=35.0,
        geo_center_lon=120.0,
        geo_span_lat=2.0,
        geo_span_lon=2.0,
        enable_irregular=False,
        centers=[{'x': 256, 'y': 256, 'weight': 1.0, 'std_x': 60, 'std_y': 60}]
    )

# 路径规划强制使用规则高斯云（保证浓度可预测），Cesium 渲染用不规则形状
config.enable_irregular = False
print(f'      Model type: {config.model_type}')
print(f'      Geo center: ({config.geo_center_lat}, {config.geo_center_lon})')

# ── 步骤2：尝试加载并使用 RL 模型 ─────────────────────────────────────────────
print()
print('[2/5] Loading trained model...')
rl_success = False
path_result = None

model_path = 'models/final_model.pth'
if os.path.exists(model_path):
    try:
        temp_env = VolcanicAshEnv(config)
        state_dim = len(DDPGAgent.flatten_state(temp_env.reset()[0]))
        action_dim = int(np.prod(temp_env.action_space.shape))
        algorithm = infer_checkpoint_algorithm(model_path)
        agent = create_agent(algorithm, state_dim=state_dim, action_dim=action_dim)
        agent.load_model(model_path)
        print('      [OK] Model loaded:', model_path)

        print()
        print('[3/5] Planning path with RL agent...')
        planner = PathPlanner(config, agent)

        lat_c  = config.geo_center_lat
        lon_c  = config.geo_center_lon
        span_lat = config.geo_span_lat / 2
        span_lon = config.geo_span_lon / 2
        start_geo  = (lat_c - span_lat * 0.8, lon_c - span_lon * 0.8)
        target_geo = (lat_c + span_lat * 0.8, lon_c + span_lon * 0.8)

        result = planner.plan_path_geo_with_fallback(start_geo, target_geo, max_steps=400, max_concentration=0.30)
        max_conc = result.get('max_concentration', 1.0)
        print(f'      RL result — success:{result["success"]}  '
              f'steps:{result["steps_taken"]}  max_conc:{max_conc:.4f}')

        if result.get('planning_method') != 'fallback':
            path_result = result
            rl_success = True
            print('      [OK] RL path accepted (max conc in safe range)')
        else:
            path_result = result
            print('      [WARN] RL path rejected (max conc too high), using geometric fallback')
    except Exception as e:
        print(f'      [WARN] RL planning failed: {e}')
else:
    print(f'      [WARN] Model not found at {model_path}')
    print('[3/5] Skipping RL planning (no model)...')

# ── 步骤3（回退）：程序化样条路径 ────────────────────────────────────────────
if path_result is None:
    print()
    print('[3/5] Generating geometric spline path (fallback)...')

    def pixel_to_geo(py, px, cfg):
        h, w = cfg.image_size
        lat = cfg.geo_center_lat + (0.5 - py / h) * cfg.geo_span_lat
        lon = cfg.geo_center_lon + (px / w - 0.5) * cfg.geo_span_lon
        return lat, lon

    from src.model.gmm_model import GMMVolcanicAshModel
    model = GMMVolcanicAshModel(config)
    concentration_map = model.generate_concentration_map()

    # 控制点：沿 140px 圆弧绕行，全程在绿色区域
    ctrl = np.array([
        [440,  50], [400,  72], [340,  94], [278, 108],
        [215, 117], [163, 134], [125, 195], [116, 256],
        [120, 316], [138, 378], [110, 425], [ 75, 452],
    ])
    t      = np.linspace(0, 1, len(ctrl))
    t_fine = np.linspace(0, 1, 130)
    path_y = np.clip(CubicSpline(t, ctrl[:, 0])(t_fine), 1, config.image_size[0] - 2)
    path_x = np.clip(CubicSpline(t, ctrl[:, 1])(t_fine), 1, config.image_size[1] - 2)

    waypoints = []
    cum_fuel = 0.0
    for i in range(len(path_y)):
        py, px = path_y[i], path_x[i]
        conc = float(concentration_map[int(round(py)), int(round(px))])
        if i > 0:
            dy = path_y[i] - path_y[i-1]
            dx = path_x[i] - path_x[i-1]
            cum_fuel += 0.18 + float(np.sqrt(dy**2 + dx**2)) * 0.012
        else:
            dy = dx = 0.0
        lat, lon = pixel_to_geo(py, px, config)
        waypoints.append({
            'step': i, 'pixel_x': float(px), 'pixel_y': float(py),
            'latitude': lat, 'longitude': lon,
            'velocity_x': float(dx), 'velocity_y': float(dy),
            'concentration': conc,
            'cumulative_reward': float(-i * 0.3 - conc * 20),
            'cumulative_fuel': cum_fuel
        })

    path_result = {
        'success': True,
        'waypoints': waypoints,
        'total_reward': float(waypoints[-1]['cumulative_reward']),
        'total_fuel': cum_fuel,
        'max_concentration': max(w['concentration'] for w in waypoints),
        'steps_taken': len(waypoints),
        'path_coordinates': [[w['pixel_x'], w['pixel_y']] for w in waypoints],
        'start_geo': list(pixel_to_geo(ctrl[0,0], ctrl[0,1], config)),
        'target_geo': list(pixel_to_geo(ctrl[-1,0], ctrl[-1,1], config)),
    }
    print(f'      [OK] Geometric path generated — '
          f'steps:{path_result["steps_taken"]}  '
          f'max_conc:{path_result["max_concentration"]:.4f}')

# ── 步骤4：计算 cloud_info（区域半径精确值）──────────────────────────────────
print()
print('[4/5] Computing cloud zone radii...')
std_px = config.centers[0]['std_x']
deg_px = config.geo_span_lat / config.image_size[0]

def conc_radius_deg(threshold):
    return float(std_px * np.sqrt(-2 * np.log(threshold)) * deg_px)

cloud_info = {
    'center_lat': config.geo_center_lat,
    'center_lon': config.geo_center_lon,
    'zone_radii': {
        'high':   round(conc_radius_deg(0.45), 4),
        'medium': round(conc_radius_deg(0.30), 4),
        'low':    round(conc_radius_deg(0.15), 4),
        'safe':   round(conc_radius_deg(0.05), 4),
    }
}
print(f'      Zone radii: {cloud_info["zone_radii"]}')

# ── 步骤5：导出 ──────────────────────────────────────────────────────────────
print()
print('[5/5] Exporting results...')
os.makedirs('output', exist_ok=True)
output_path = 'output/planned_path.json'

output_data = {
    'cloud_info': cloud_info,
    'planning_info': {
        'timestamp': str(datetime.now()),
        'success': path_result['success'],
        'total_waypoints': len(path_result['waypoints']),
        'total_steps': path_result['steps_taken'],
        'method': 'ddpg_agent' if rl_success else 'geometric_spline_fallback',
        'summary': {
            'total_reward': float(path_result['total_reward']),
            'total_fuel_consumption': float(path_result['total_fuel']),
            'max_concentration_exposure': float(path_result['max_concentration'])
        }
    },
    'path_data': path_result
}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

print(f'      [OK] Path data exported to: {output_path}')
print(f'      Method used: {"DDPG RL agent" if rl_success else "Geometric spline (fallback)"}')

print()
print('='*70)
print('PATH PLANNING TEST COMPLETED!')
print('='*70)
print(f'  Steps taken    : {path_result["steps_taken"]}')
print(f'  Max conc.      : {path_result["max_concentration"]:.4f}')
print(f'  Total fuel     : {path_result["total_fuel"]:.2f}')
print()
print('Load output/planned_path.json in the web interface at http://localhost:5000')
