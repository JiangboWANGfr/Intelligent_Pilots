import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config.volcanic_ash_config import VolcanicAshConfig, get_training_scene_configs
from src.rl_env.volcanic_ash_env import VolcanicAshEnv
from src.rl_training.ddpg_agent import DDPGAgent, create_agent, infer_checkpoint_algorithm


def parse_scene_names(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    return [name.strip() for name in raw.split(',') if name.strip()]


def build_scene_configs(config: VolcanicAshConfig,
                        scene_names: Optional[List[str]]) -> List[VolcanicAshConfig]:
    if scene_names:
        return get_training_scene_configs(scene_names)
    return [VolcanicAshConfig.from_dict(config.to_dict())]


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
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of deterministic evaluation tasks.')
    parser.add_argument('--max-steps', type=int, default=400,
                        help='Maximum steps per evaluation task.')
    parser.add_argument('--seed', type=int, default=2026,
                        help='Base random seed. Episode i uses seed + i.')
    parser.add_argument('--scenes', default=None,
                        help='Comma-separated preset scene names. Defaults to the config itself.')
    parser.add_argument('--output', default='output/evaluation_results.json',
                        help='Where to write detailed JSON results.')
    args = parser.parse_args()

    if not os.path.exists(args.config):
        raise FileNotFoundError(f'Config not found: {args.config}')
    if not os.path.exists(args.model):
        raise FileNotFoundError(f'Model not found: {args.model}')

    config = VolcanicAshConfig.load(args.config)
    scene_configs = build_scene_configs(config, parse_scene_names(args.scenes))
    env = VolcanicAshEnv(config, scene_configs=scene_configs)
    env.max_steps = args.max_steps

    obs, _ = env.reset(seed=args.seed)
    state_dim = len(DDPGAgent.flatten_state(obs))
    env.scene_cursor = -1

    algorithm = infer_algorithm(args.model, args.algorithm)
    agent = create_agent(algorithm, state_dim=state_dim, action_dim=2, device=args.device)
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
