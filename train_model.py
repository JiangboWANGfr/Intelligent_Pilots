import argparse
import os
import sys
from typing import List, Optional

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.makedirs('output/mpl_cache', exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', 'output/mpl_cache')

from src.config.volcanic_ash_config import VolcanicAshConfig, get_training_scene_configs, get_preset_configs
from src.rl_env.volcanic_ash_env import VolcanicAshEnv
from src.rl_training.trainer import Trainer


def parse_scene_names(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    return [name.strip() for name in raw.split(',') if name.strip()]


def parse_int_pair(raw: str):
    parts = [part.strip() for part in raw.split(',')]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError('Expected two comma-separated integers, e.g. 1,6.')
    lower, upper = int(parts[0]), int(parts[1])
    if lower < 1 or upper < lower:
        raise argparse.ArgumentTypeError('Expected a valid range with 1 <= min <= max.')
    return lower, upper


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
    if bool(getattr(config, 'use_random_ash_scenes', False)):
        random_config = VolcanicAshConfig.from_dict(config.to_dict())
        random_config.scene_name = '随机旋转GMM_每回合生成'
        random_config.model_type = 'random_rotated_gmm'
        return [random_config]
    if scene_names:
        return get_training_scene_configs(scene_names)
    if config.training_scene_names:
        return get_training_scene_configs(config.training_scene_names)
    if use_all_scenes:
        return get_training_scene_configs()
    return [VolcanicAshConfig.from_dict(config.to_dict())]


def apply_aircraft_runtime_config(config: VolcanicAshConfig,
                                  scene_configs: List[VolcanicAshConfig],
                                  cruise_speed: Optional[float],
                                  cruise_speed_mode: str,
                                  fixed_scene_maps: bool) -> None:
    if cruise_speed is not None:
        config.fixed_cruise_speed = cruise_speed
    config.cruise_speed_mode = cruise_speed_mode
    if fixed_scene_maps:
        config.randomize_irregular_each_episode = False

    for scene in scene_configs:
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


def safe_filename(name: str) -> str:
    safe_chars = []
    for char in name:
        if char.isalnum() or char in ('-', '_'):
            safe_chars.append(char)
        else:
            safe_chars.append('_')
    return ''.join(safe_chars).strip('_') or 'scene'


def save_pretraining_ash_images(config: VolcanicAshConfig,
                                scene_configs: List[VolcanicAshConfig],
                                save_dir: str,
                                seed: int = 2026,
                                preview_count: Optional[int] = None) -> None:
    preview_dir = os.path.join(save_dir, 'pre_training_ash_maps')
    os.makedirs(preview_dir, exist_ok=True)

    preview_env = VolcanicAshEnv(config, scene_configs=scene_configs)
    preview_env.max_steps = 1
    summary = []

    count = preview_count or len(scene_configs)
    for index in range(count):
        _, info = preview_env.reset(seed=seed + index)
        scene_name = info.get('scene_name', preview_env.scene_name)
        filename_prefix = f'{index + 1:02d}_{safe_filename(scene_name)}'

        concentration_map = np.asarray(preview_env.concentration_map, dtype=np.float32)
        grayscale = preview_env.ash_model.generate_grayscale_image(concentration_map)
        danger_rgb = preview_env.ash_model.generate_danger_zone_image(concentration_map)

        grayscale_path = os.path.join(preview_dir, f'{filename_prefix}_concentration.png')
        danger_path = os.path.join(preview_dir, f'{filename_prefix}_danger.png')
        overlay_path = os.path.join(preview_dir, f'{filename_prefix}_path_overlay.png')

        cv2.imwrite(grayscale_path, grayscale)
        cv2.imwrite(danger_path, cv2.cvtColor(danger_rgb, cv2.COLOR_RGB2BGR))

        overlay_rgb = cv2.applyColorMap(grayscale, cv2.COLORMAP_INFERNO)
        overlay_rgb = cv2.cvtColor(overlay_rgb, cv2.COLOR_BGR2RGB)
        danger_mask = concentration_map >= preview_env.danger_threshold
        safety_mask = concentration_map >= preview_env.safety_threshold
        overlay_rgb[safety_mask] = (
            0.65 * overlay_rgb[safety_mask] + 0.35 * np.array([255, 165, 0])
        ).astype(np.uint8)
        overlay_rgb[danger_mask] = (
            0.45 * overlay_rgb[danger_mask] + 0.55 * np.array([255, 0, 0])
        ).astype(np.uint8)

        if preview_env.reference_path is not None and len(preview_env.reference_path) > 1:
            path_points = np.round(preview_env.reference_path[:, [1, 0]]).astype(np.int32)
            cv2.polylines(
                overlay_rgb,
                [path_points],
                isClosed=False,
                color=(0, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA
            )

        start_xy = tuple(np.round(preview_env.aircraft_pos[[1, 0]]).astype(int).tolist())
        target_xy = tuple(np.round(preview_env.target_pos[[1, 0]]).astype(int).tolist())
        cv2.circle(overlay_rgb, start_xy, 5, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(overlay_rgb, target_xy, 5, (0, 128, 255), -1, lineType=cv2.LINE_AA)
        cv2.imwrite(overlay_path, cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))

        summary.append({
            'scene_name': scene_name,
            'concentration_path': grayscale_path,
            'danger_path': danger_path,
            'path_overlay_path': overlay_path,
            'max_concentration': float(np.max(concentration_map)),
            'mean_concentration': float(np.mean(concentration_map)),
            'safety_threshold': float(preview_env.safety_threshold),
            'danger_threshold': float(preview_env.danger_threshold)
        })

    summary_path = os.path.join(preview_dir, 'pre_training_ash_maps.json')
    import json
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f'Pre-training ash cloud images saved to: {preview_dir}')


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
    parser.add_argument('--random-ash-scenes', action='store_true',
                        help='Generate a new random 1-6 center rotated-GMM ash scene each episode.')
    parser.add_argument('--random-scene-seed', type=int, default=None,
                        help='Base seed metadata for random ash scenes.')
    parser.add_argument('--random-centers-range', type=parse_int_pair, default=(1, 6),
                        help='Random Gaussian center count range as min,max.')
    parser.add_argument('--random-preview-scenes', type=int, default=6,
                        help='Number of random ash previews saved before random-scene training.')
    parser.add_argument('--random-scene-max-attempts', type=int, default=None,
                        help='Rejection-sampling attempts per random ash scene.')
    parser.add_argument('--episodes', type=int, default=3000)
    parser.add_argument('--max-steps', type=int, default=400)
    parser.add_argument('--algorithm', choices=['td3', 'ddpg'], default='td3')
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--buffer-size', type=int, default=300000)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--noise-decay', type=float, default=0.999)
    parser.add_argument('--min-noise', type=float, default=0.05)
    parser.add_argument('--initial-noise', type=float, default=0.3,
                        help='Initial Gaussian exploration noise scale.')
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
    parser.add_argument('--cruise-speed', type=float, default=None,
                        help='Fixed per-step cruise speed used by the aircraft model.')
    parser.add_argument('--cruise-speed-mode', choices=['fixed', 'random'], default='fixed',
                        help='fixed uses --cruise-speed/config speed; random samples once per episode.')
    parser.add_argument('--fixed-scene-maps', action='store_true',
                        help='Reuse one deterministic ash map per scene instead of randomizing irregular maps every episode.')
    parser.add_argument('--expert-warmup-episodes', type=int, default=0,
                        help='Collect this many pure-pursuit expert episodes before RL training.')
    parser.add_argument('--behavior-clone-steps', type=int, default=0,
                        help='Supervised actor pretraining steps using expert warmup states.')
    parser.add_argument('--bc-regularization-steps', type=int, default=0,
                        help='Extra behavior-cloning actor updates after each RL episode.')
    parser.add_argument('--expert-gain', type=float, default=1.0,
                        help='Pure-pursuit expert turn gain used for warmup.')
    parser.add_argument('--imitation-only', action='store_true',
                        help='Only run expert warmup and behavior cloning, then save the actor.')
    parser.add_argument('--skip-pretraining-ash-images', action='store_true',
                        help='Do not save pre-training ash cloud preview images.')
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
    print(f'  - Expert warmup episodes: {args.expert_warmup_episodes}')
    print(f'  - Behavior clone steps: {args.behavior_clone_steps}')
    print(f'  - BC regularization steps: {args.bc_regularization_steps}')
    print(f'  - Imitation only: {args.imitation_only}')
    print('  - Environment: VolcanicAshEnv (Gymnasium)')
    print()

    config = load_config(args.config, args.fallback_preset)
    if args.random_ash_scenes:
        config.use_random_ash_scenes = True
        config.random_scene_seed = args.random_scene_seed
        config.random_scene_min_centers = args.random_centers_range[0]
        config.random_scene_max_centers = args.random_centers_range[1]
        if args.random_scene_max_attempts is not None:
            config.random_scene_max_attempts = args.random_scene_max_attempts
        config.training_scene_names = []
    scene_configs = build_scene_configs(
        config,
        scene_names=None if args.random_ash_scenes else parse_scene_names(args.scenes),
        use_all_scenes=args.all_scenes
    )
    apply_aircraft_runtime_config(
        config,
        scene_configs,
        cruise_speed=args.cruise_speed,
        cruise_speed_mode=args.cruise_speed_mode,
        fixed_scene_maps=args.fixed_scene_maps
    )
    config.training_scene_names = [scene.scene_name for scene in scene_configs]

    print(f'Volcanic Ash Model: {config.model_type}')
    print(f'  Centers: {len(config.centers)}')
    print(f'  Random ash scenes: {config.use_random_ash_scenes}')
    if config.use_random_ash_scenes:
        print(
            '  Random centers per episode: '
            f'{config.random_scene_min_centers}-{config.random_scene_max_centers}'
        )
    print(f'  Cloud size: {config.cloud_size}')
    print(f'  Threshold: {config.concentration_threshold}')
    print(f'  Geo position: ({config.geo_center_lat}, {config.geo_center_lon})')
    print(f'  Cruise speed: {config.fixed_cruise_speed} ({config.cruise_speed_mode})')
    print(f'  Randomize irregular maps: {config.randomize_irregular_each_episode}')
    print(f'  Path corridor radius: {config.path_corridor_radius}')
    print(f'  Path planning threshold ratio: {config.path_planning_threshold_ratio}')
    print(f'  Path risk inflation radius: {config.path_risk_inflation_radius}')
    print(f'  Path boundary margin: {config.path_boundary_margin}')
    print(f'  Ash avoidance gain: {config.ash_avoidance_gain}')
    print(f'  Training scenes: {len(scene_configs)}')
    for scene in scene_configs:
        print(f'    - {scene.scene_name or scene.model_type}')
    print()

    if not args.skip_pretraining_ash_images:
        save_pretraining_ash_images(
            config,
            scene_configs,
            args.save_dir,
            preview_count=args.random_preview_scenes if config.use_random_ash_scenes else None
        )
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
        initial_noise=args.initial_noise,
        algorithm=args.algorithm,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_delay=args.policy_delay,
        expert_warmup_episodes=args.expert_warmup_episodes,
        behavior_clone_steps=args.behavior_clone_steps,
        bc_regularization_steps=args.bc_regularization_steps,
        expert_gain=args.expert_gain,
        imitation_only=args.imitation_only,
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
