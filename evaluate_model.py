import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config.volcanic_ash_config import (
    VolcanicAshConfig,
    get_training_scene_configs,
    resize_config_canvas,
)
from src.rl_env.volcanic_ash_env import VolcanicAshEnv
from src.rl_training.ddpg_agent import DDPGAgent, create_agent, infer_checkpoint_algorithm


def parse_scene_names(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    return [name.strip() for name in raw.split(',') if name.strip()]


def parse_size_pair(raw: str):
    parts = [part.strip() for part in raw.split(',')]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError('Expected two comma-separated integers, e.g. 768,768.')
    first, second = int(parts[0]), int(parts[1])
    if first < 1 or second < 1:
        raise argparse.ArgumentTypeError('Image size values must be positive.')
    return first, second


def build_scene_configs(config: VolcanicAshConfig,
                        scene_names: Optional[List[str]]) -> List[VolcanicAshConfig]:
    if scene_names:
        return get_training_scene_configs(scene_names)
    return [VolcanicAshConfig.from_dict(config.to_dict())]


def apply_aircraft_runtime_config(config: VolcanicAshConfig,
                                  scene_configs: List[VolcanicAshConfig],
                                  cruise_speed: Optional[float],
                                  cruise_speed_mode: str,
                                  fixed_scene_maps: bool,
                                  dynamic_ash: bool = False) -> None:
    if cruise_speed is not None:
        config.fixed_cruise_speed = cruise_speed
    config.cruise_speed_mode = cruise_speed_mode
    if fixed_scene_maps:
        config.randomize_irregular_each_episode = False
    if dynamic_ash:
        config.enable_dynamic_ash = True

    for scene in scene_configs:
        if tuple(scene.image_size) != tuple(config.image_size):
            resize_config_canvas(scene, tuple(config.image_size))
        else:
            scene.image_size = tuple(config.image_size)
        scene.fixed_cruise_speed = config.fixed_cruise_speed
        scene.min_cruise_speed = config.min_cruise_speed
        scene.max_cruise_speed = config.max_cruise_speed
        scene.cruise_speed_mode = config.cruise_speed_mode
        scene.randomize_irregular_each_episode = config.randomize_irregular_each_episode
        scene.path_corridor_radius = config.path_corridor_radius
        scene.path_lookahead_distance = config.path_lookahead_distance
        scene.reference_path_points = config.reference_path_points
        scene.path_planning_threshold_ratio = config.path_planning_threshold_ratio
        scene.path_risk_inflation_radius = config.path_risk_inflation_radius
        scene.path_boundary_margin = config.path_boundary_margin
        scene.ash_avoidance_gain = config.ash_avoidance_gain
        scene.ash_avoidance_activation_ratio = config.ash_avoidance_activation_ratio
        scene.airport_safety_threshold_ratio = config.airport_safety_threshold_ratio
        scene.airport_clearance_radius = config.airport_clearance_radius
        scene.departure_cloud_clearance_radius = config.departure_cloud_clearance_radius
        scene.arrival_cloud_clearance_radius = config.arrival_cloud_clearance_radius
        scene.initial_clear_path_distance = config.initial_clear_path_distance
        scene.initial_clear_concentration_ratio = config.initial_clear_concentration_ratio
        scene.safety_factor_mode = config.safety_factor_mode
        scene.fixed_safety_factor = config.fixed_safety_factor
        scene.min_safety_factor = config.min_safety_factor
        scene.max_safety_factor = config.max_safety_factor
        scene.enable_dynamic_ash = config.enable_dynamic_ash
        scene.ash_advection_speed = config.ash_advection_speed
        scene.ash_diffusion_sigma = config.ash_diffusion_sigma
        scene.ash_decay_rate = config.ash_decay_rate
        scene.ash_turbulence_drift = config.ash_turbulence_drift
        scene.ash_dynamic_update_interval = config.ash_dynamic_update_interval
        scene.ash_dynamic_renormalize = config.ash_dynamic_renormalize
        scene.ash_advection_speed_min = config.ash_advection_speed_min
        scene.ash_advection_speed_max = config.ash_advection_speed_max
        scene.ash_wind_direction_jitter_deg = config.ash_wind_direction_jitter_deg
        scene.ash_wind_speed_jitter_ratio = config.ash_wind_speed_jitter_ratio
        scene.ash_wind_smoothness = config.ash_wind_smoothness
        scene.ash_rotation_enabled = config.ash_rotation_enabled
        scene.ash_rotation_rate_deg = config.ash_rotation_rate_deg
        scene.ash_rotation_jitter_deg = config.ash_rotation_jitter_deg
        scene.ash_local_deformation_strength = config.ash_local_deformation_strength
        scene.ash_local_flow_scale = config.ash_local_flow_scale
        scene.ash_local_flow_smoothness = config.ash_local_flow_smoothness
        scene.ash_local_flow_update_interval = config.ash_local_flow_update_interval
        scene.ash_shear_strength = config.ash_shear_strength


def infer_algorithm(model_path: str, requested: str) -> str:
    if requested != 'auto':
        return requested
    return infer_checkpoint_algorithm(model_path)


def evaluate_episode(env: VolcanicAshEnv,
                     agent,
                     seed: int,
                     max_steps: int) -> Dict:
    state, info = env.reset(seed=seed)
    previous_pos = env.aircraft_pos.copy()
    total_reward = 0.0
    path_length = 0.0
    concentrations = [float(info.get('current_concentration', 0.0))]
    cross_track_errors = [float(info.get('cross_track_error', 0.0))]
    danger_violation = bool(info.get('is_in_danger_zone', False))
    terminated = False
    truncated = False
    step = -1

    for step in range(max_steps):
        action = agent.select_action(state, evaluate=True)
        next_state, reward, terminated, truncated, info = env.step(action)

        current_pos = env.aircraft_pos.copy()
        path_length += float(np.linalg.norm(current_pos - previous_pos))
        previous_pos = current_pos

        concentration = float(info.get('current_concentration', 0.0))
        concentrations.append(concentration)
        cross_track_errors.append(float(info.get('cross_track_error', 0.0)))
        danger_violation = danger_violation or bool(info.get('is_in_danger_zone', False))
        total_reward += float(reward)
        state = next_state

        if terminated or truncated:
            break

    success = bool(terminated and info['distance_to_target'] < env.success_threshold)
    timeout = bool(truncated or (not terminated and (step + 1) >= max_steps))
    if success:
        termination_reason = 'success'
    elif timeout:
        termination_reason = 'timeout'
    elif max(concentrations) > 0.9:
        termination_reason = 'extreme_concentration'
    else:
        termination_reason = 'terminated'

    return {
        'seed': seed,
        'scene_name': info.get('scene_name', env.scene_name),
        'success': success,
        'timeout': timeout,
        'danger_violation': danger_violation,
        'termination_reason': termination_reason,
        'steps': step + 1,
        'total_reward': total_reward,
        'fuel_consumed': float(info.get('fuel_consumed', 0.0)),
        'ash_exposure': float(info.get('ash_exposure', 0.0)),
        'path_progress_ratio': float(info.get('path_progress_ratio', 0.0)),
        'avg_cross_track_error': float(np.mean(cross_track_errors)),
        'final_distance': float(info.get('distance_to_target', 0.0)),
        'path_length': path_length,
        'max_concentration': float(max(concentrations)),
        'avg_concentration': float(np.mean(concentrations))
    }


def summarize(results: List[Dict]) -> Dict:
    count = max(len(results), 1)
    return {
        'episodes': len(results),
        'success_rate': sum(r['success'] for r in results) / count * 100.0,
        'timeout_rate': sum(r['timeout'] for r in results) / count * 100.0,
        'danger_violation_rate': sum(r['danger_violation'] for r in results) / count * 100.0,
        'avg_reward': float(np.mean([r['total_reward'] for r in results])) if results else 0.0,
        'avg_steps': float(np.mean([r['steps'] for r in results])) if results else 0.0,
        'avg_fuel': float(np.mean([r['fuel_consumed'] for r in results])) if results else 0.0,
        'avg_ash_exposure': float(np.mean([r['ash_exposure'] for r in results])) if results else 0.0,
        'avg_path_progress_ratio': float(np.mean([r['path_progress_ratio'] for r in results])) if results else 0.0,
        'avg_cross_track_error': float(np.mean([r['avg_cross_track_error'] for r in results])) if results else 0.0,
        'avg_path_length': float(np.mean([r['path_length'] for r in results])) if results else 0.0,
        'avg_max_concentration': float(np.mean([r['max_concentration'] for r in results])) if results else 0.0,
        'avg_concentration': float(np.mean([r['avg_concentration'] for r in results])) if results else 0.0,
        'avg_final_distance': float(np.mean([r['final_distance'] for r in results])) if results else 0.0
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate a trained volcanic ash avoidance model.')
    parser.add_argument('--model', default='models/final_model.pth',
                        help='Path to a trained RL checkpoint.')
    parser.add_argument('--algorithm', choices=['auto', 'td3', 'ddpg'], default='auto',
                        help='Model algorithm. auto inspects the checkpoint.')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu', 'mps'], default='auto',
                        help='Torch device for model inference.')
    parser.add_argument('--config', default='output/current_config.json',
                        help='Path to the volcanic ash config JSON.')
    parser.add_argument('--image-size', type=parse_size_pair, default=None,
                        help='Map size as height,width, e.g. 768,768. Centers are repositioned proportionally.')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of deterministic evaluation tasks.')
    parser.add_argument('--max-steps', type=int, default=400,
                        help='Maximum steps per evaluation task.')
    parser.add_argument('--seed', type=int, default=2026,
                        help='Base random seed. Episode i uses seed + i.')
    parser.add_argument('--scenes', default=None,
                        help='Comma-separated preset scene names. Defaults to the config itself.')
    parser.add_argument('--cruise-speed', type=float, default=None,
                        help='Fixed per-step cruise speed used during evaluation.')
    parser.add_argument('--cruise-speed-mode', choices=['fixed', 'random'], default='fixed',
                        help='fixed uses --cruise-speed/config speed; random samples once per episode.')
    parser.add_argument('--fixed-scene-maps', action='store_true',
                        help='Reuse one deterministic ash map per scene instead of randomizing irregular maps every episode.')
    parser.add_argument('--dynamic-ash', action='store_true',
                        help='Move ash clouds during evaluation using wind advection, diffusion and decay.')
    parser.add_argument('--ash-advection-speed', type=float, default=None)
    parser.add_argument('--ash-diffusion-sigma', type=float, default=None)
    parser.add_argument('--ash-decay-rate', type=float, default=None)
    parser.add_argument('--ash-turbulence-drift', type=float, default=None)
    parser.add_argument('--ash-local-deformation-strength', type=float, default=None)
    parser.add_argument('--ash-local-flow-scale', type=float, default=None)
    parser.add_argument('--ash-local-flow-smoothness', type=float, default=None)
    parser.add_argument('--ash-local-flow-update-interval', type=int, default=None)
    parser.add_argument('--ash-shear-strength', type=float, default=None)
    parser.add_argument('--output', default='output/evaluation_results.json',
                        help='Where to write detailed JSON results.')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f'Config not found: {args.config}')
    if not os.path.exists(args.model):
        raise FileNotFoundError(f'Model not found: {args.model}')

    config = VolcanicAshConfig.load(args.config)
    if args.image_size is not None:
        resize_config_canvas(config, args.image_size)
    if args.dynamic_ash:
        config.enable_dynamic_ash = True
    if args.ash_advection_speed is not None:
        config.ash_advection_speed = args.ash_advection_speed
    if args.ash_diffusion_sigma is not None:
        config.ash_diffusion_sigma = args.ash_diffusion_sigma
    if args.ash_decay_rate is not None:
        config.ash_decay_rate = args.ash_decay_rate
    if args.ash_turbulence_drift is not None:
        config.ash_turbulence_drift = args.ash_turbulence_drift
    if args.ash_local_deformation_strength is not None:
        config.ash_local_deformation_strength = args.ash_local_deformation_strength
    if args.ash_local_flow_scale is not None:
        config.ash_local_flow_scale = args.ash_local_flow_scale
    if args.ash_local_flow_smoothness is not None:
        config.ash_local_flow_smoothness = args.ash_local_flow_smoothness
    if args.ash_local_flow_update_interval is not None:
        config.ash_local_flow_update_interval = args.ash_local_flow_update_interval
    if args.ash_shear_strength is not None:
        config.ash_shear_strength = args.ash_shear_strength
    scene_configs = build_scene_configs(config, parse_scene_names(args.scenes))
    apply_aircraft_runtime_config(
        config,
        scene_configs,
        cruise_speed=args.cruise_speed,
        cruise_speed_mode=args.cruise_speed_mode,
        fixed_scene_maps=args.fixed_scene_maps,
        dynamic_ash=args.dynamic_ash
    )
    env = VolcanicAshEnv(config, scene_configs=scene_configs)
    env.max_steps = args.max_steps

    obs, _ = env.reset(seed=args.seed)
    state_dim = len(DDPGAgent.flatten_state(obs))
    action_dim = int(np.prod(env.action_space.shape))
    env.scene_cursor = -1

    algorithm = infer_algorithm(args.model, args.algorithm)
    agent = create_agent(algorithm, state_dim=state_dim, action_dim=action_dim, device=args.device)
    agent.load_model(args.model)

    results = [
        evaluate_episode(env, agent, seed=args.seed + i, max_steps=args.max_steps)
        for i in range(args.episodes)
    ]
    summary = summarize(results)

    output = {
        'model_path': args.model,
        'algorithm': algorithm,
        'device': str(agent.device),
        'config_path': args.config,
        'fixed_cruise_speed': config.fixed_cruise_speed,
        'cruise_speed_mode': config.cruise_speed_mode,
        'randomize_irregular_each_episode': config.randomize_irregular_each_episode,
        'dynamic_ash': config.enable_dynamic_ash,
        'scene_names': [scene.scene_name or scene.model_type for scene in scene_configs],
        'seed': args.seed,
        'max_steps': args.max_steps,
        'summary': summary,
        'episodes': results
    }

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f'Evaluation results saved to: {args.output}')


if __name__ == '__main__':
    main()
