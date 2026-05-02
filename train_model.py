import argparse
import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.makedirs('output/mpl_cache', exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', 'output/mpl_cache')

from src.config.volcanic_ash_config import VolcanicAshConfig, get_training_scene_configs, get_preset_configs
from src.rl_training.trainer import Trainer


def parse_scene_names(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    return [name.strip() for name in raw.split(',') if name.strip()]


def load_config(config_path: str, fallback_preset: str) -> VolcanicAshConfig:
    if os.path.exists(config_path):
        config = VolcanicAshConfig.load(config_path)
        print(f'Loaded configuration from: {config_path}')
        return config

    presets = get_preset_configs()
    if fallback_preset not in presets:
        raise KeyError(f'Unknown fallback preset: {fallback_preset}')

    print(f'Configuration file not found: {config_path}')
    print(f'Using fallback preset: {fallback_preset}')
    return VolcanicAshConfig.from_dict(presets[fallback_preset].to_dict())


def build_scene_configs(config: VolcanicAshConfig,
                        scene_names: Optional[List[str]],
                        use_all_scenes: bool) -> List[VolcanicAshConfig]:
    if scene_names:
        return get_training_scene_configs(scene_names)
    if config.training_scene_names:
        return get_training_scene_configs(config.training_scene_names)
    if use_all_scenes:
        return get_training_scene_configs()
    return [VolcanicAshConfig.from_dict(config.to_dict())]


def main():
    parser = argparse.ArgumentParser(description='Train the volcanic ash avoidance RL model.')
    parser.add_argument('--config', default='output/current_config.json',
                        help='Path to config JSON.')
    parser.add_argument('--fallback-preset', default='双中心_复杂扩散',
                        help='Preset used when --config does not exist.')
    parser.add_argument('--scenes', default=None,
                        help='Comma-separated preset scene names. Overrides config.training_scene_names.')
    parser.add_argument('--all-scenes', action='store_true',
                        help='Train on all preset scenes when no explicit scenes are provided.')
    parser.add_argument('--episodes', type=int, default=3000)
    parser.add_argument('--max-steps', type=int, default=400)
    parser.add_argument('--algorithm', choices=['td3', 'ddpg'], default='td3')
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--buffer-size', type=int, default=300000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--noise-decay', type=float, default=0.999)
    parser.add_argument('--min-noise', type=float, default=0.05)
    parser.add_argument('--policy-noise', type=float, default=0.2,
                        help='TD3 target policy smoothing noise.')
    parser.add_argument('--noise-clip', type=float, default=0.5,
                        help='TD3 target policy smoothing noise clip.')
    parser.add_argument('--policy-delay', type=int, default=2,
                        help='TD3 delayed actor update interval.')
    parser.add_argument('--update-every', type=int, default=10,
                        help='Run one gradient update every N environment steps.')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu', 'mps'], default='auto',
                        help='Torch device for neural network training.')
    parser.add_argument('--save-dir', default='models')
    parser.add_argument('--load-model', default=None,
                        help='Optional checkpoint to continue training from.')
    parser.add_argument('--log-interval', type=int, default=50)
    args = parser.parse_args()

    print('=' * 70)
    print('STARTING REINFORCEMENT LEARNING TRAINING')
    print('=' * 70)
    print()
    print('Configuration:')
    print(f'  - Algorithm: {args.algorithm.upper()}')
    print(f'  - Episodes: {args.episodes}')
    print(f'  - Max steps per episode: {args.max_steps}')
    print(f'  - Learning rate: {args.learning_rate}')
    print(f'  - Batch size: {args.batch_size}')
    print(f'  - Buffer size: {args.buffer_size}')
    print(f'  - Update every: {args.update_every} steps')
    print(f'  - Device: {args.device}')
    print('  - Environment: VolcanicAshEnv (Gymnasium)')
    print()

    config = load_config(args.config, args.fallback_preset)
    scene_configs = build_scene_configs(
        config,
        scene_names=parse_scene_names(args.scenes),
        use_all_scenes=args.all_scenes
    )
    config.training_scene_names = [scene.scene_name for scene in scene_configs]

    print(f'Volcanic Ash Model: {config.model_type}')
    print(f'  Centers: {len(config.centers)}')
    print(f'  Cloud size: {config.cloud_size}')
    print(f'  Threshold: {config.concentration_threshold}')
    print(f'  Geo position: ({config.geo_center_lat}, {config.geo_center_lon})')
    print(f'  Training scenes: {len(scene_configs)}')
    for scene in scene_configs:
        print(f'    - {scene.scene_name or scene.model_type}')
    print()

    trainer = Trainer(
        config=config,
        num_episodes=args.episodes,
        max_steps_per_episode=args.max_steps,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        noise_decay=args.noise_decay,
        min_noise=args.min_noise,
        algorithm=args.algorithm,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_delay=args.policy_delay,
        device=args.device,
        save_dir=args.save_dir,
        scene_configs=scene_configs
    )
    print(f'Using torch device: {trainer.agent.device}')

    if args.load_model:
        trainer.agent.load_model(args.load_model)
        print(f'Loaded model checkpoint: {args.load_model}')

    print('Starting training...')
    print('-' * 70)

    agent, history = trainer.train(update_every=args.update_every,
                                   log_interval=args.log_interval)

    print()
    print('=' * 70)
    print('TRAINING COMPLETED SUCCESSFULLY!')
    print('=' * 70)
    final_reward = history['rewards'][-1] if history['rewards'] else 0
    print(f'Model saved to: {os.path.join(args.save_dir, "final_model.pth")}')
    print(f'Training curves: {os.path.join(args.save_dir, "training_curves.png")}')
    print(f'Final reward: {final_reward:.2f}')

    return agent, history


if __name__ == '__main__':
    main()
