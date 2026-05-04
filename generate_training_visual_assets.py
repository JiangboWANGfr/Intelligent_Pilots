import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from evaluate_model import apply_aircraft_runtime_config, parse_scene_names
from src.config.volcanic_ash_config import VolcanicAshConfig, get_training_scene_configs
from src.path_planning.animation_exporter import ValidationAnimationExporter
from src.rl_env.volcanic_ash_env import VolcanicAshEnv
from src.rl_training.ddpg_agent import DDPGAgent, create_agent, infer_checkpoint_algorithm


def moving_average(values: List[float], window: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if len(array) == 0:
        return array
    window = max(1, min(window, len(array)))
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(array, kernel, mode='same')


def rolling_success_rate(history: Dict, window: int = 25) -> List[float]:
    flags = history.get('success_flags', [])
    if not flags:
        return history.get('success_rates', [])
    values = [100.0 if flag else 0.0 for flag in flags]
    return moving_average(values, window).tolist()


def load_history(model_dir: str) -> Optional[Dict]:
    history_path = os.path.join(model_dir, 'training_history.json')
    if not os.path.exists(history_path):
        return None
    with open(history_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_training_metric_plots(history: Dict, output_dir: str) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    episodes = np.asarray(history.get('episodes', []), dtype=np.float32)
    if len(episodes) == 0:
        return []

    outputs = []
    plot_specs = [
        ('reward_curve.png', 'Episode Reward', 'Reward', history.get('rewards', []), True),
        ('learning_rate_curve.png', 'Learning Rate', 'Learning Rate', history.get('learning_rates', []), False),
        ('loss_curve.png', 'Training Loss', 'Loss', history.get('losses', []), True),
        ('actor_critic_loss_curve.png', 'Actor/Critic Loss', 'Loss', None, False),
        ('success_rate_curve.png', 'Rolling Success Rate', 'Success %', rolling_success_rate(history), False),
        ('ash_exposure_curve.png', 'Ash Exposure', 'Exposure', history.get('ash_exposures', []), True),
        ('cross_track_error_curve.png', 'Cross Track Error', 'Pixels', history.get('cross_track_errors', []), True),
        ('final_distance_curve.png', 'Final Distance', 'Pixels', history.get('final_distances', []), True),
        ('path_progress_curve.png', 'Path Progress Ratio', 'Ratio', history.get('path_progress_ratios', []), True),
    ]

    for filename, title, ylabel, values, smooth in plot_specs:
        fig, ax = plt.subplots(figsize=(10, 5))
        if filename == 'actor_critic_loss_curve.png':
            actor = history.get('actor_losses', [])
            critic = history.get('critic_losses', [])
            if actor:
                ax.plot(episodes[:len(actor)], actor, alpha=0.35, label='Actor loss')
                ax.plot(episodes[:len(actor)], moving_average(actor, 25), linewidth=2, label='Actor MA')
            if critic:
                ax.plot(episodes[:len(critic)], critic, alpha=0.35, label='Critic loss')
                ax.plot(episodes[:len(critic)], moving_average(critic, 25), linewidth=2, label='Critic MA')
        elif values:
            xs = episodes[:len(values)]
            ax.plot(xs, values, alpha=0.35, label='Raw')
            if smooth:
                ax.plot(xs, moving_average(values, 25), linewidth=2, label='Moving average')
            else:
                ax.plot(xs, values, linewidth=2, label='Value')

        ax.set_title(title)
        ax.set_xlabel('Episode')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = os.path.join(output_dir, filename)
        fig.savefig(path, dpi=160)
        plt.close(fig)
        outputs.append(path)

    dashboard_path = os.path.join(output_dir, 'training_dashboard.png')
    fig, axes = plt.subplots(3, 2, figsize=(14, 13))
    dashboard_items = [
        ('rewards', 'Reward'),
        ('__rolling_success__', 'Rolling Success Rate (%)'),
        ('losses', 'Loss'),
        ('ash_exposures', 'Ash Exposure'),
        ('cross_track_errors', 'Cross Track Error'),
        ('final_distances', 'Final Distance'),
    ]
    for ax, (key, title) in zip(axes.flat, dashboard_items):
        values = rolling_success_rate(history) if key == '__rolling_success__' else history.get(key, [])
        if values:
            xs = episodes[:len(values)]
            ax.plot(xs, values, alpha=0.28)
            ax.plot(xs, moving_average(values, 25), linewidth=2)
        ax.set_title(title)
        ax.set_xlabel('Episode')
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(dashboard_path, dpi=160)
    plt.close(fig)
    outputs.append(dashboard_path)
    return outputs


def checkpoint_episode(path: str) -> int:
    match = re.search(r'checkpoint_ep(\d+)\.pth$', os.path.basename(path))
    if not match:
        return 10**9
    return int(match.group(1))


def safe_name(value: str) -> str:
    sanitized = ''.join(char if char.isalnum() or char in ('-', '_') else '_' for char in value.strip())
    return sanitized.strip('_') or 'scene'


def parse_int_pair(raw: str) -> Tuple[int, int]:
    parts = [part.strip() for part in raw.split(',')]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError('Expected two comma-separated integers, e.g. 1,6.')
    lower, upper = int(parts[0]), int(parts[1])
    if lower < 1 or upper < lower:
        raise argparse.ArgumentTypeError('Expected a valid range with 1 <= min <= max.')
    return lower, upper


def parse_episode_list(raw: Optional[str]) -> Optional[List[int]]:
    if not raw:
        return None
    episodes = []
    for part in raw.split(','):
        value = part.strip()
        if not value:
            continue
        episode = int(value)
        if episode < 1:
            raise argparse.ArgumentTypeError('Checkpoint episodes must be >= 1.')
        episodes.append(episode)
    return sorted(set(episodes))


def find_milestone_checkpoints(model_dir: str,
                               max_checkpoints: int,
                               include_final: bool,
                               checkpoint_episodes: Optional[List[int]] = None) -> List[Tuple[str, str]]:
    checkpoint_paths = sorted(
        [
            os.path.join(model_dir, name)
            for name in os.listdir(model_dir)
            if re.match(r'checkpoint_ep\d+\.pth$', name)
        ],
        key=checkpoint_episode
    )
    selected = []
    if checkpoint_episodes:
        by_episode = {checkpoint_episode(path): path for path in checkpoint_paths}
        available_episodes = sorted(by_episode.keys())
        for requested_episode in checkpoint_episodes:
            path = by_episode.get(requested_episode)
            if path is None and available_episodes:
                nearest_episode = min(
                    available_episodes,
                    key=lambda episode: abs(episode - requested_episode)
                )
                path = by_episode[nearest_episode]
            if path is not None:
                selected.append((f'ep{checkpoint_episode(path):04d}', path))
        seen_paths = set()
        selected = [
            item for item in selected
            if not (item[1] in seen_paths or seen_paths.add(item[1]))
        ]
    elif max_checkpoints <= 0:
        selected = []
    elif checkpoint_paths:
        indices = np.linspace(0, len(checkpoint_paths) - 1, min(max_checkpoints, len(checkpoint_paths)))
        for index in sorted({int(round(i)) for i in indices}):
            path = checkpoint_paths[index]
            selected.append((f'ep{checkpoint_episode(path):04d}', path))

    final_path = os.path.join(model_dir, 'final_model.pth')
    if include_final and os.path.exists(final_path):
        selected.append(('final', final_path))
    return selected


def find_font_path() -> Optional[str]:
    candidates = [
        '/System/Library/Fonts/PingFang.ttc',
        '/System/Library/Fonts/STHeiti Light.ttc',
        '/Library/Fonts/Arial Unicode.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def get_font(size: int):
    font_path = find_font_path()
    if font_path:
        try:
            return ImageFont.truetype(font_path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def make_title_card(size: Tuple[int, int],
                    label: str,
                    subtitle: str,
                    duration_ms: int) -> Image.Image:
    width, height = size
    card = Image.new('RGB', size, (18, 21, 27))
    draw = ImageDraw.Draw(card)
    title_font = get_font(max(28, width // 34))
    subtitle_font = get_font(max(18, width // 56))
    title = f'Training Milestone: {label}'
    text_color = (245, 247, 250)
    accent_color = (0, 255, 255)

    title_box = draw.textbbox((0, 0), title, font=title_font)
    subtitle_box = draw.textbbox((0, 0), subtitle, font=subtitle_font)
    title_x = (width - (title_box[2] - title_box[0])) // 2
    subtitle_x = (width - (subtitle_box[2] - subtitle_box[0])) // 2
    center_y = height // 2
    draw.text((title_x, center_y - 48), title, font=title_font, fill=text_color)
    draw.text((subtitle_x, center_y + 8), subtitle, font=subtitle_font, fill=(190, 196, 208))
    draw.rectangle((width // 3, center_y + 54, 2 * width // 3, center_y + 60), fill=accent_color)
    card.info['duration'] = duration_ms
    return card


def read_gif_frames(gif_path: str, target_size: Optional[Tuple[int, int]] = None) -> List[Image.Image]:
    frames = []
    with Image.open(gif_path) as image:
        for frame_index in range(getattr(image, 'n_frames', 1)):
            image.seek(frame_index)
            frame = image.convert('RGB')
            if target_size is not None and frame.size != target_size:
                frame = frame.resize(target_size, Image.Resampling.BILINEAR)
            frame.info['duration'] = image.info.get('duration', 83)
            frames.append(frame)
    return frames


def stitch_progress_gif(animation_outputs: List[Dict],
                        output_dir: str,
                        filename: str = 'training_progress.gif',
                        title_duration_ms: int = 900,
                        max_frames_per_clip: int = 90) -> Optional[str]:
    clips = [
        item for item in animation_outputs
        if item.get('gif_path') and os.path.exists(str(item.get('gif_path')))
    ]
    if not clips:
        return None

    progress_dir = os.path.join(output_dir, 'animations')
    os.makedirs(progress_dir, exist_ok=True)
    output_path = os.path.join(progress_dir, filename)

    stitched_frames: List[Image.Image] = []
    target_size = None
    for clip in clips:
        gif_path = str(clip['gif_path'])
        clip_frames = read_gif_frames(gif_path, target_size=target_size)
        if not clip_frames:
            continue
        if target_size is None:
            target_size = clip_frames[0].size

        if len(clip_frames) > max_frames_per_clip:
            indices = np.linspace(0, len(clip_frames) - 1, max_frames_per_clip)
            clip_frames = [clip_frames[int(round(index))] for index in indices]

        label = str(clip.get('label', 'checkpoint'))
        success_text = 'SUCCESS' if clip.get('success') else 'NOT SUCCESSFUL'
        reward = float(clip.get('total_reward', 0.0))
        distance = float(clip.get('final_distance', 0.0))
        subtitle = f'{success_text} | reward {reward:.1f} | final distance {distance:.1f}'
        stitched_frames.append(make_title_card(target_size, label, subtitle, title_duration_ms))
        stitched_frames.extend(clip_frames)

    if not stitched_frames:
        return None

    durations = [int(frame.info.get('duration', 83)) for frame in stitched_frames]
    stitched_frames[0].save(
        output_path,
        save_all=True,
        append_images=stitched_frames[1:],
        duration=durations,
        loop=0,
        optimize=False
    )
    return output_path


def simulate_episode(env: VolcanicAshEnv,
                     agent,
                     seed: int,
                     max_steps: int,
                     scene_label: str,
                     milestone_label: str) -> Dict:
    state, info = env.reset(seed=seed)
    concentration_maps = []
    if getattr(env.config, 'enable_dynamic_ash', False):
        concentration_maps.append((env.concentration_map * 255.0).astype(np.uint8))
    positions = [env.aircraft_pos.copy()]
    waypoints = [{
        'x': float(env.aircraft_pos[1]),
        'y': float(env.aircraft_pos[0]),
        'concentration': float(info.get('current_concentration', 0.0))
    }]
    total_reward = 0.0
    max_concentration = float(info.get('current_concentration', 0.0))
    terminated = False
    truncated = False

    for _ in range(max_steps):
        action = agent.select_action(state, evaluate=True)
        state, reward, terminated, truncated, info = env.step(action)
        current_conc = float(info.get('current_concentration', 0.0))
        total_reward += float(reward)
        max_concentration = max(max_concentration, current_conc)
        positions.append(env.aircraft_pos.copy())
        if getattr(env.config, 'enable_dynamic_ash', False):
            concentration_maps.append((env.concentration_map * 255.0).astype(np.uint8))
        waypoints.append({
            'x': float(env.aircraft_pos[1]),
            'y': float(env.aircraft_pos[0]),
            'concentration': current_conc
        })
        if terminated or truncated:
            break

    success = bool(terminated and info.get('distance_to_target', float('inf')) < env.success_threshold)
    result = {
        'scene_name': f'{scene_label}_{milestone_label}',
        'planning_method': 'rl_checkpoint',
        'waypoints': waypoints,
        'path_coordinates': [[float(pos[1]), float(pos[0])] for pos in positions],
        'max_concentration': float(max_concentration),
        'total_fuel': float(info.get('fuel_consumed', 0.0)),
        'total_reward': float(total_reward),
        'success': success,
        'validation_info': {
            'used_fallback': False,
            'fallback_reason': '',
            'termination_reason': (
                'success' if success else 'timeout' if truncated else 'terminated'
            ),
            'distance_to_target': float(info.get('distance_to_target', 0.0)),
            'ash_exposure': float(info.get('ash_exposure', 0.0)),
            'path_progress_ratio': float(info.get('path_progress_ratio', 0.0))
        }
    }
    if concentration_maps:
        result['concentration_maps'] = concentration_maps
    return result


def export_checkpoint_animations(model_dir: str,
                                 output_dir: str,
                                 config_path: str,
                                 scene_name: str,
                                 cruise_speed: Optional[float],
                                 fixed_scene_maps: bool,
                                 seed: int,
                                 max_steps: int,
                                 max_checkpoints: int,
                                 checkpoint_episodes: Optional[List[int]],
                                 include_untrained: bool,
                                 random_ash_scenes: bool = False,
                                 random_centers_range: Tuple[int, int] = (1, 6),
                                 random_scene_seed: Optional[int] = None,
                                 dynamic_ash: bool = False,
                                 ash_advection_speed: Optional[float] = None,
                                 ash_diffusion_sigma: Optional[float] = None,
                                 ash_decay_rate: Optional[float] = None,
                                 ash_turbulence_drift: Optional[float] = None) -> List[Dict]:
    config = VolcanicAshConfig.load(config_path)
    if dynamic_ash:
        config.enable_dynamic_ash = True
    if ash_advection_speed is not None:
        config.ash_advection_speed = ash_advection_speed
    if ash_diffusion_sigma is not None:
        config.ash_diffusion_sigma = ash_diffusion_sigma
    if ash_decay_rate is not None:
        config.ash_decay_rate = ash_decay_rate
    if ash_turbulence_drift is not None:
        config.ash_turbulence_drift = ash_turbulence_drift
    if random_ash_scenes:
        config.use_random_ash_scenes = True
        config.random_scene_seed = seed if random_scene_seed is None else random_scene_seed
        config.random_scene_min_centers = random_centers_range[0]
        config.random_scene_max_centers = random_centers_range[1]
        config.scene_name = scene_name
        config.model_type = 'random_rotated_gmm'
        config.training_scene_names = []
        scene_configs = [VolcanicAshConfig.from_dict(config.to_dict())]
    else:
        scene_configs = get_training_scene_configs(parse_scene_names(scene_name))
    apply_aircraft_runtime_config(
        config,
        scene_configs,
        cruise_speed=cruise_speed,
        cruise_speed_mode='fixed',
        fixed_scene_maps=fixed_scene_maps,
        dynamic_ash=dynamic_ash
    )
    env = VolcanicAshEnv(config, scene_configs=scene_configs)
    env.max_steps = max_steps
    obs, _ = env.reset(seed=seed)
    state_dim = len(DDPGAgent.flatten_state(obs))
    action_dim = int(np.prod(env.action_space.shape))
    env.scene_cursor = -1
    if hasattr(env, 'random_scene_counter'):
        env.random_scene_counter = 0

    milestones = []
    if include_untrained:
        milestones.append(('ep0000_untrained', None))
    milestones.extend(find_milestone_checkpoints(
        model_dir,
        max_checkpoints,
        include_final=True,
        checkpoint_episodes=checkpoint_episodes
    ))

    outputs = []
    for label, model_path in milestones:
        try:
            if model_path is None:
                algorithm = 'td3'
            else:
                algorithm = infer_checkpoint_algorithm(model_path)
            agent = create_agent(algorithm, state_dim=state_dim, action_dim=action_dim, device='cpu')
            if model_path is not None:
                agent.load_model(model_path)
        except Exception as exc:
            outputs.append({'label': label, 'model_path': model_path, 'skipped': str(exc)})
            continue

        env.scene_cursor = -1
        if hasattr(env, 'random_scene_counter'):
            env.random_scene_counter = 0
        path_result = simulate_episode(
            env=env,
            agent=agent,
            seed=seed,
            max_steps=max_steps,
            scene_label=scene_name,
            milestone_label=label
        )
        exporter = ValidationAnimationExporter(env.config, env.concentration_map)
        milestone_dir = os.path.join(output_dir, 'animations', safe_name(scene_name), label)
        try:
            export_info = exporter.export(
                path_result,
                output_dir=milestone_dir,
                gif_path=os.path.join(milestone_dir, f'{label}.gif'),
                video_path=os.path.join(milestone_dir, f'{label}.mp4'),
                fps=12,
                save_frames=True,
                max_frames=140,
                hold_last_frames=12
            )
            outputs.append({
                'label': label,
                'model_path': model_path,
                'success': path_result['success'],
                'total_reward': path_result['total_reward'],
                'max_concentration': path_result['max_concentration'],
                'final_distance': path_result['validation_info']['distance_to_target'],
                **export_info
            })
        except Exception as exc:
            outputs.append({'label': label, 'model_path': model_path, 'skipped': str(exc)})

    return outputs


def main():
    parser = argparse.ArgumentParser(
        description='Generate training visual assets: metric plots and checkpoint flight animations.'
    )
    parser.add_argument('--model-dir', default='models/turn_controller_single_v3')
    parser.add_argument('--config', default='output/current_config.json')
    parser.add_argument('--scene', default='单中心_强风拉伸')
    parser.add_argument('--output-dir', default='output/demo_assets')
    parser.add_argument('--cruise-speed', type=float, default=9.0)
    parser.add_argument('--fixed-scene-maps', action='store_true')
    parser.add_argument('--random-ash-scenes', action='store_true',
                        help='Render checkpoint animations on generated random rotated-GMM ash scenes.')
    parser.add_argument('--random-centers-range', type=parse_int_pair, default=(1, 6),
                        help='Random Gaussian center count range as min,max.')
    parser.add_argument('--random-demo-scenes', type=int, default=1,
                        help='Number of deterministic random scenes to render.')
    parser.add_argument('--random-scene-seed', type=int, default=None,
                        help='Base seed for deterministic random demo scenes.')
    parser.add_argument('--dynamic-ash', action='store_true',
                        help='Render moving ash clouds in checkpoint animations.')
    parser.add_argument('--ash-advection-speed', type=float, default=None,
                        help='Ash cloud wind advection speed in pixels per environment step.')
    parser.add_argument('--ash-diffusion-sigma', type=float, default=None)
    parser.add_argument('--ash-decay-rate', type=float, default=None)
    parser.add_argument('--ash-turbulence-drift', type=float, default=None)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--max-steps', type=int, default=260)
    parser.add_argument('--max-checkpoints', type=int, default=4)
    parser.add_argument('--checkpoint-episodes', default=None,
                        help='Comma-separated checkpoint episodes to render, e.g. 1,5,10,25,50,100.')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Fallback learning rate used for histories that do not store learning_rates.')
    parser.add_argument('--include-untrained', action='store_true')
    parser.add_argument('--skip-animations', action='store_true')
    parser.add_argument('--stitch-progress-gif', action='store_true',
                        help='Concatenate milestone GIFs into one continuous training progress GIF.')
    parser.add_argument('--progress-gif-name', default='training_progress.gif',
                        help='Filename for the stitched training progress GIF.')
    parser.add_argument('--max-frames-per-progress-clip', type=int, default=90,
                        help='Maximum frames kept from each milestone clip in the stitched GIF.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    summary = {
        'model_dir': args.model_dir,
        'config': args.config,
        'scene': args.scene,
        'random_ash_scenes': args.random_ash_scenes,
        'dynamic_ash': args.dynamic_ash,
        'random_centers_range': list(args.random_centers_range),
        'random_demo_scenes': args.random_demo_scenes,
        'metric_plots': [],
        'animations': [],
        'stitched_progress_gif': None
    }

    history = load_history(args.model_dir)
    if history is not None:
        if 'learning_rates' not in history and args.learning_rate is not None:
            history['learning_rates'] = [args.learning_rate for _ in history.get('episodes', [])]
        summary['metric_plots'] = save_training_metric_plots(
            history,
            os.path.join(args.output_dir, 'metrics')
        )

    if not args.skip_animations:
        if args.random_ash_scenes:
            base_seed = args.seed if args.random_scene_seed is None else args.random_scene_seed
            scene_names = [
                f'随机旋转GMM_演示场景_{index + 1:02d}_seed{base_seed + index}'
                for index in range(max(1, args.random_demo_scenes))
            ]
        else:
            scene_names = parse_scene_names(args.scene) or [args.scene]
        summary['scene'] = scene_names if args.random_ash_scenes else args.scene
        for scene_index, scene_name in enumerate(scene_names):
            summary['animations'].extend(export_checkpoint_animations(
                model_dir=args.model_dir,
                output_dir=args.output_dir,
                config_path=args.config,
                scene_name=scene_name,
                cruise_speed=args.cruise_speed,
                fixed_scene_maps=args.fixed_scene_maps,
                seed=args.seed,
                max_steps=args.max_steps,
                max_checkpoints=args.max_checkpoints,
                checkpoint_episodes=parse_episode_list(args.checkpoint_episodes),
                include_untrained=args.include_untrained,
                random_ash_scenes=args.random_ash_scenes,
                random_centers_range=args.random_centers_range,
                random_scene_seed=(
                    (args.random_scene_seed if args.random_scene_seed is not None else args.seed)
                    + scene_index
                    if args.random_ash_scenes else None
                ),
                dynamic_ash=args.dynamic_ash,
                ash_advection_speed=args.ash_advection_speed,
                ash_diffusion_sigma=args.ash_diffusion_sigma,
                ash_decay_rate=args.ash_decay_rate,
                ash_turbulence_drift=args.ash_turbulence_drift
            ))

    if args.stitch_progress_gif and summary['animations']:
        summary['stitched_progress_gif'] = stitch_progress_gif(
            summary['animations'],
            output_dir=args.output_dir,
            filename=args.progress_gif_name,
            max_frames_per_clip=args.max_frames_per_progress_clip
        )

    summary_path = os.path.join(args.output_dir, 'asset_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f'Visual assets saved to: {args.output_dir}')
    print(f'Summary: {summary_path}')


if __name__ == '__main__':
    main()
