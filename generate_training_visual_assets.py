import argparse
import json
import os
import re
import shutil
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
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
        ('safety_factor_curve.png', 'Safety Factor', 'Risk Penalty Multiplier', history.get('safety_factors', []), False),
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
        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
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
    outputs.extend(save_safety_factor_metric_plots(history, output_dir, episodes))
    return outputs


def save_safety_factor_metric_plots(history: Dict,
                                    output_dir: str,
                                    episodes: np.ndarray) -> List[str]:
    safety_factors = np.asarray(history.get('safety_factors', []), dtype=np.float32)
    if safety_factors.size == 0:
        return []

    outputs = []
    ash_exposures = np.asarray(history.get('ash_exposures', []), dtype=np.float32)
    final_distances = np.asarray(history.get('final_distances', []), dtype=np.float32)
    success_flags = np.asarray(history.get('success_flags', []), dtype=bool)

    scatter_path = os.path.join(output_dir, 'safety_factor_vs_exposure.png')
    fig, ax = plt.subplots(figsize=(10, 5))
    count = min(len(safety_factors), len(ash_exposures), len(success_flags))
    if count > 0:
        colors = np.where(success_flags[:count], '#2e7d32', '#c62828')
        ax.scatter(
            safety_factors[:count],
            ash_exposures[:count],
            c=colors,
            s=24,
            alpha=0.7,
            edgecolors='none'
        )
    ax.set_title('Safety Factor vs Ash Exposure')
    ax.set_xlabel('Safety Factor')
    ax.set_ylabel('Ash Exposure')
    ax.grid(True, alpha=0.25)
    ax.legend(
        handles=[
            Patch(facecolor='#2e7d32', label='Success'),
            Patch(facecolor='#c62828', label='Not success')
        ],
        loc='best'
    )
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=160)
    plt.close(fig)
    outputs.append(scatter_path)

    bin_path = os.path.join(output_dir, 'safety_factor_bins.png')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = ['Low safety', 'Medium safety', 'High safety']
    bins = [0.0, 0.9, 1.3, np.inf]
    exposure_means = []
    success_means = []
    distance_means = []
    for lower, upper in zip(bins[:-1], bins[1:]):
        mask = (safety_factors >= lower) & (safety_factors < upper)
        exposure_mask = mask[:len(ash_exposures)]
        success_mask = mask[:len(success_flags)]
        distance_mask = mask[:len(final_distances)]
        exposure_means.append(float(np.mean(ash_exposures[exposure_mask])) if np.any(exposure_mask) else 0.0)
        success_means.append(float(np.mean(success_flags[success_mask]) * 100.0) if np.any(success_mask) else 0.0)
        distance_means.append(float(np.mean(final_distances[distance_mask])) if np.any(distance_mask) else 0.0)

    x = np.arange(len(labels))
    axes[0].bar(x, exposure_means, color=['#66bb6a', '#ffa726', '#ef5350'])
    axes[0].set_title('Average Ash Exposure by Safety Level')
    axes[0].set_xticks(x, labels, rotation=12)
    axes[0].set_ylabel('Ash Exposure')
    axes[0].grid(True, axis='y', alpha=0.25)

    width = 0.36
    axes[1].bar(x - width / 2, success_means, width=width, label='Success %', color='#42a5f5')
    axes[1].bar(x + width / 2, distance_means, width=width, label='Final distance', color='#ab47bc')
    axes[1].set_title('Outcome by Safety Level')
    axes[1].set_xticks(x, labels, rotation=12)
    axes[1].grid(True, axis='y', alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(bin_path, dpi=160)
    plt.close(fig)
    outputs.append(bin_path)

    episode_path = os.path.join(output_dir, 'safety_factor_timeline.png')
    fig, ax = plt.subplots(figsize=(10, 4))
    xs = episodes[:len(safety_factors)]
    ax.plot(xs, safety_factors, linewidth=1.5, color='#1565c0')
    ax.fill_between(xs, safety_factors, 1.0, color='#90caf9', alpha=0.3)
    ax.axhline(1.0, color='black', linewidth=1, alpha=0.5)
    ax.set_title('Episode Safety Factor Timeline')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Safety Factor')
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(episode_path, dpi=160)
    plt.close(fig)
    outputs.append(episode_path)
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


def parse_float_list(raw: str) -> List[float]:
    values = []
    for part in raw.split(','):
        text = part.strip()
        if not text:
            continue
        values.append(float(text))
    if not values:
        raise argparse.ArgumentTypeError('Expected at least one numeric value.')
    return values


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
        'start_coordinate': [float(positions[0][1]), float(positions[0][0])],
        'target_coordinate': [float(env.target_pos[1]), float(env.target_pos[0])],
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


def latest_model_path(model_dir: str) -> Optional[str]:
    final_path = os.path.join(model_dir, 'final_model.pth')
    if os.path.exists(final_path):
        return final_path
    checkpoints = find_milestone_checkpoints(
        model_dir,
        max_checkpoints=1,
        include_final=False,
        checkpoint_episodes=None
    )
    return checkpoints[-1][1] if checkpoints else None


def build_demo_scene_configs(config: VolcanicAshConfig,
                             scene_name: str,
                             random_ash_scenes: bool,
                             random_centers_range: Tuple[int, int],
                             random_scene_seed: Optional[int],
                             seed: int) -> List[VolcanicAshConfig]:
    if random_ash_scenes:
        config.use_random_ash_scenes = True
        config.random_scene_seed = seed if random_scene_seed is None else random_scene_seed
        config.random_scene_min_centers = random_centers_range[0]
        config.random_scene_max_centers = random_centers_range[1]
        config.scene_name = scene_name
        config.model_type = 'random_rotated_gmm'
        config.training_scene_names = []
        return [VolcanicAshConfig.from_dict(config.to_dict())]
    return get_training_scene_configs(parse_scene_names(scene_name))


def export_safety_factor_comparison(model_dir: str,
                                    output_dir: str,
                                    config_path: str,
                                    scene_name: str,
                                    cruise_speed: Optional[float],
                                    fixed_scene_maps: bool,
                                    seed: int,
                                    max_steps: int,
                                    random_ash_scenes: bool,
                                    random_centers_range: Tuple[int, int],
                                    random_scene_seed: Optional[int],
                                    dynamic_ash: bool,
                                    ash_advection_speed: Optional[float],
                                    ash_diffusion_sigma: Optional[float],
                                    ash_decay_rate: Optional[float],
                                    ash_turbulence_drift: Optional[float],
                                    safety_factor_values: Sequence[float],
                                    output_subdir: str = 'safety_factor_comparison') -> Optional[Dict]:
    model_path = latest_model_path(model_dir)
    if model_path is None:
        return None

    outputs_dir = os.path.join(output_dir, output_subdir)
    os.makedirs(outputs_dir, exist_ok=True)
    path_results = []
    base_map = None
    start_coordinate = None
    target_coordinate = None
    agent = None

    for safety_factor in safety_factor_values:
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
        config.safety_factor_mode = 'fixed'
        config.fixed_safety_factor = float(safety_factor)
        config.min_safety_factor = min(config.min_safety_factor, float(safety_factor))
        config.max_safety_factor = max(config.max_safety_factor, float(safety_factor))

        scene_configs = build_demo_scene_configs(
            config=config,
            scene_name=scene_name,
            random_ash_scenes=random_ash_scenes,
            random_centers_range=random_centers_range,
            random_scene_seed=random_scene_seed,
            seed=seed
        )
        apply_aircraft_runtime_config(
            config,
            scene_configs,
            cruise_speed=cruise_speed,
            cruise_speed_mode='fixed',
            fixed_scene_maps=fixed_scene_maps,
            dynamic_ash=dynamic_ash
        )
        for scene in scene_configs:
            scene.safety_factor_mode = 'fixed'
            scene.fixed_safety_factor = float(safety_factor)
            scene.min_safety_factor = config.min_safety_factor
            scene.max_safety_factor = config.max_safety_factor

        env = VolcanicAshEnv(config, scene_configs=scene_configs)
        env.max_steps = max_steps
        env.scene_cursor = -1
        if hasattr(env, 'random_scene_counter'):
            env.random_scene_counter = 0
        obs, _ = env.reset(seed=seed)
        if agent is None:
            state_dim = len(DDPGAgent.flatten_state(obs))
            action_dim = int(np.prod(env.action_space.shape))
            algorithm = infer_checkpoint_algorithm(model_path)
            agent = create_agent(algorithm, state_dim=state_dim, action_dim=action_dim, device='cpu')
            agent.load_model(model_path)
        if hasattr(env, 'random_scene_counter'):
            env.random_scene_counter = 0

        result = simulate_episode(
            env=env,
            agent=agent,
            seed=seed,
            max_steps=max_steps,
            scene_label=scene_name,
            milestone_label=f'safety_{safety_factor:.2f}'
        )
        result['safety_factor'] = float(safety_factor)
        path_results.append(result)
        if base_map is None:
            base_map = env.concentration_map.copy()
            start_coordinate = result.get('start_coordinate')
            target_coordinate = result.get('target_coordinate')

    if base_map is None or not path_results:
        return None

    display_map = np.array(base_map, copy=True)
    dynamic_maps = []
    for result in path_results:
        for map_array in result.get('concentration_maps', []):
            map_float = np.asarray(map_array, dtype=np.float32)
            if map_float.dtype == np.uint8 or np.max(map_float) > 1.0:
                map_float = map_float / 255.0
            dynamic_maps.append(np.clip(map_float, 0.0, 1.0))
    if dynamic_maps:
        display_map = np.maximum.reduce([display_map, *dynamic_maps])

    exporter = ValidationAnimationExporter(VolcanicAshConfig.load(config_path), display_map)
    map_rgb = exporter._build_colored_map(render_scale=2, concentration_map=display_map)
    scale_x = map_rgb.shape[1] / display_map.shape[1]
    scale_y = map_rgb.shape[0] / display_map.shape[0]
    colors = ['#2e7d32', '#f9a825', '#c62828', '#1565c0', '#6a1b9a']

    figure_path = os.path.join(outputs_dir, 'safety_factor_path_comparison.png')
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.imshow(map_rgb)
    for index, result in enumerate(path_results):
        coords = np.asarray(result.get('path_coordinates', []), dtype=np.float32)
        if coords.size == 0:
            continue
        xs = coords[:, 0] * scale_x
        ys = coords[:, 1] * scale_y
        label = (
            f"safety {result['safety_factor']:.2f} | "
            f"{'success' if result.get('success') else 'not success'} | "
            f"exposure {result['validation_info'].get('ash_exposure', 0.0):.1f}"
        )
        ax.plot(xs, ys, color=colors[index % len(colors)], linewidth=2.8, label=label)
    if start_coordinate:
        ax.scatter(start_coordinate[0] * scale_x, start_coordinate[1] * scale_y,
                   s=90, color='#00c853', edgecolor='white', linewidth=1.5, zorder=5)
        ax.text(start_coordinate[0] * scale_x + 8, start_coordinate[1] * scale_y,
                'START', color='black', fontsize=10, weight='bold')
    if target_coordinate:
        ax.scatter(target_coordinate[0] * scale_x, target_coordinate[1] * scale_y,
                   s=110, marker='x', color='#d32f2f', linewidth=3, zorder=5)
        ax.text(target_coordinate[0] * scale_x + 8, target_coordinate[1] * scale_y,
                'TARGET', color='black', fontsize=10, weight='bold')
    title_suffix = 'Ash Motion Envelope' if dynamic_maps else 'Initial Ash Map'
    ax.set_title(f'Same Scene, Different Safety Factors ({title_suffix})')
    ax.set_axis_off()
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.08), ncol=1, frameon=True)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=170, bbox_inches='tight')
    plt.close(fig)

    metric_path = os.path.join(outputs_dir, 'safety_factor_outcome_bars.png')
    labels = [f"{result['safety_factor']:.2f}" for result in path_results]
    exposures = [float(result['validation_info'].get('ash_exposure', 0.0)) for result in path_results]
    rewards = [float(result.get('total_reward', 0.0)) for result in path_results]
    final_distances = [float(result['validation_info'].get('distance_to_target', 0.0)) for result in path_results]
    path_lengths = []
    for result in path_results:
        coords = np.asarray(result.get('path_coordinates', []), dtype=np.float32)
        path_lengths.append(
            float(np.linalg.norm(np.diff(coords, axis=0), axis=1).sum())
            if len(coords) > 1 else 0.0
        )
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    axes[0].bar(x, exposures, color=colors[:len(labels)])
    axes[0].set_title('Ash Exposure')
    axes[1].bar(x, path_lengths, color=colors[:len(labels)])
    axes[1].set_title('Path Length')
    axes[2].bar(x, rewards, color=colors[:len(labels)])
    axes[2].set_title('Total Reward')
    axes[3].bar(x, final_distances, color=colors[:len(labels)])
    axes[3].set_title('Final Distance')
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.set_xlabel('Safety Factor')
        ax.grid(True, axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(metric_path, dpi=160)
    plt.close(fig)

    summary_path = os.path.join(outputs_dir, 'safety_factor_comparison.json')
    summary = {
        'model_path': model_path,
        'scene_name': scene_name,
        'seed': seed,
        'outputs': [figure_path, metric_path],
        'runs': [
            {
                'safety_factor': result['safety_factor'],
                'success': result['success'],
                'total_reward': result['total_reward'],
                'ash_exposure': result['validation_info'].get('ash_exposure', 0.0),
                'final_distance': result['validation_info'].get('distance_to_target', 0.0),
                'path_progress_ratio': result['validation_info'].get('path_progress_ratio', 0.0),
                'path_length': path_lengths[index]
            }
            for index, result in enumerate(path_results)
        ]
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    summary['outputs'].append(summary_path)
    return summary


def save_safety_factor_multi_scene_overview(comparisons: List[Dict],
                                            output_dir: str) -> Optional[str]:
    valid = [item for item in comparisons if item and item.get('runs')]
    if not valid:
        return None

    safety_values = sorted({
        float(run['safety_factor'])
        for item in valid
        for run in item.get('runs', [])
    })
    if not safety_values:
        return None

    scene_labels = [
        f"Scene {index + 1:02d} | seed {item.get('seed', '')}"
        for index, item in enumerate(valid)
    ]
    exposure_matrix = np.full((len(valid), len(safety_values)), np.nan, dtype=np.float32)
    distance_matrix = np.full_like(exposure_matrix, np.nan)
    length_matrix = np.full_like(exposure_matrix, np.nan)
    for scene_index, item in enumerate(valid):
        for run in item.get('runs', []):
            safety = float(run['safety_factor'])
            col = safety_values.index(safety)
            exposure_matrix[scene_index, col] = float(run.get('ash_exposure', np.nan))
            distance_matrix[scene_index, col] = float(run.get('final_distance', np.nan))
            length_matrix[scene_index, col] = float(run.get('path_length', np.nan))

    overview_path = os.path.join(output_dir, 'safety_factor_multi_scene_overview.png')
    fig, axes = plt.subplots(3, 1, figsize=(13, max(10, len(valid) * 1.1)))
    y = np.arange(len(valid))
    bar_height = 0.8 / max(1, len(safety_values))
    colors = ['#2e7d32', '#f9a825', '#c62828', '#1565c0', '#6a1b9a']
    for index, safety in enumerate(safety_values):
        offset = (index - (len(safety_values) - 1) / 2.0) * bar_height
        axes[0].barh(
            y + offset,
            exposure_matrix[:, index],
            height=bar_height,
            color=colors[index % len(colors)],
            label=f'safety {safety:.2f}'
        )
        axes[1].barh(
            y + offset,
            length_matrix[:, index],
            height=bar_height,
            color=colors[index % len(colors)],
            label=f'safety {safety:.2f}'
        )
        axes[2].barh(
            y + offset,
            distance_matrix[:, index],
            height=bar_height,
            color=colors[index % len(colors)],
            label=f'safety {safety:.2f}'
        )

    axes[0].set_title('Ash Exposure Across Safety Factors and Scenes')
    axes[0].set_xlabel('Ash Exposure')
    axes[1].set_title('Path Length Across Safety Factors and Scenes')
    axes[1].set_xlabel('Path Length')
    axes[2].set_title('Final Distance Across Safety Factors and Scenes')
    axes[2].set_xlabel('Final Distance')
    for ax in axes:
        ax.set_yticks(y, scene_labels)
        ax.grid(True, axis='x', alpha=0.25)
        ax.legend(loc='best')
    fig.tight_layout()
    fig.savefig(overview_path, dpi=160)
    plt.close(fig)
    return overview_path


def export_safety_factor_comparison_batch(model_dir: str,
                                          output_dir: str,
                                          config_path: str,
                                          base_scene_name: str,
                                          cruise_speed: Optional[float],
                                          fixed_scene_maps: bool,
                                          seed: int,
                                          max_steps: int,
                                          random_ash_scenes: bool,
                                          random_centers_range: Tuple[int, int],
                                          random_scene_seed: Optional[int],
                                          dynamic_ash: bool,
                                          ash_advection_speed: Optional[float],
                                          ash_diffusion_sigma: Optional[float],
                                          ash_decay_rate: Optional[float],
                                          ash_turbulence_drift: Optional[float],
                                          safety_factor_values: Sequence[float],
                                          scene_count: int,
                                          candidate_multiplier: int = 8,
                                          min_exposure_diff: float = 1.0,
                                          min_path_length_diff: float = 35.0,
                                          min_max_exposure: float = 2.0,
                                          require_all_success: bool = True,
                                          require_high_safety_lower_exposure: bool = True) -> Optional[Dict]:
    scene_count = max(1, int(scene_count))
    root_dir = os.path.join(output_dir, 'safety_factor_comparison')
    os.makedirs(root_dir, exist_ok=True)
    base_seed = seed if random_scene_seed is None else random_scene_seed
    comparisons = []
    rejected = []
    max_candidates = max(scene_count, scene_count * max(1, int(candidate_multiplier)))
    candidate_index = 0
    while len(comparisons) < scene_count and candidate_index < max_candidates:
        scene_seed = base_seed + candidate_index
        scene_name = (
            f'随机旋转GMM_安全系数对比_{len(comparisons) + 1:02d}_seed{scene_seed}'
            if random_ash_scenes
            else base_scene_name
        )
        subdir = os.path.join('safety_factor_comparison', f'scene_{len(comparisons) + 1:02d}_seed{scene_seed}')
        comparison = export_safety_factor_comparison(
            model_dir=model_dir,
            output_dir=output_dir,
            config_path=config_path,
            scene_name=scene_name,
            cruise_speed=cruise_speed,
            fixed_scene_maps=fixed_scene_maps,
            seed=seed + candidate_index,
            max_steps=max_steps,
            random_ash_scenes=random_ash_scenes,
            random_centers_range=random_centers_range,
            random_scene_seed=scene_seed if random_ash_scenes else random_scene_seed,
            dynamic_ash=dynamic_ash,
            ash_advection_speed=ash_advection_speed,
            ash_diffusion_sigma=ash_diffusion_sigma,
            ash_decay_rate=ash_decay_rate,
            ash_turbulence_drift=ash_turbulence_drift,
            safety_factor_values=safety_factor_values,
            output_subdir=subdir
        )
        candidate_index += 1
        if not comparison:
            continue

        exposures = [float(run.get('ash_exposure', 0.0)) for run in comparison.get('runs', [])]
        lengths = [float(run.get('path_length', 0.0)) for run in comparison.get('runs', [])]
        successes = [bool(run.get('success', False)) for run in comparison.get('runs', [])]
        runs_by_safety = {
            float(run.get('safety_factor', 0.0)): run
            for run in comparison.get('runs', [])
        }
        min_safety = min(runs_by_safety.keys()) if runs_by_safety else 0.0
        max_safety = max(runs_by_safety.keys()) if runs_by_safety else 0.0
        high_safety_has_lower_exposure = True
        if runs_by_safety and min_safety != max_safety:
            high_safety_has_lower_exposure = (
                float(runs_by_safety[max_safety].get('ash_exposure', 0.0))
                < float(runs_by_safety[min_safety].get('ash_exposure', 0.0))
            )
        exposure_diff = max(exposures) - min(exposures) if exposures else 0.0
        length_diff = max(lengths) - min(lengths) if lengths else 0.0
        max_exposure = max(exposures) if exposures else 0.0
        is_informative = (
            (not require_all_success or all(successes))
            and
            (not require_high_safety_lower_exposure or high_safety_has_lower_exposure)
            and
            max_exposure >= min_max_exposure
            and (
                exposure_diff >= min_exposure_diff
                or length_diff >= min_path_length_diff
            )
        )
        if is_informative or candidate_index >= max_candidates:
            comparisons.append(comparison)
        else:
            rejected.append({
                'scene_name': comparison.get('scene_name'),
                'seed': scene_seed,
                'max_exposure': max_exposure,
                'exposure_diff': exposure_diff,
                'path_length_diff': length_diff,
                'all_success': all(successes),
                'high_safety_has_lower_exposure': high_safety_has_lower_exposure
            })
            for output in comparison.get('outputs', []):
                parent = os.path.dirname(output)
                if os.path.isdir(parent):
                    shutil.rmtree(parent, ignore_errors=True)
                    break

    overview_path = save_safety_factor_multi_scene_overview(comparisons, root_dir)
    summary = {
        'scene_count': len(comparisons),
        'candidate_count': candidate_index,
        'rejected_count': len(rejected),
        'rejected': rejected,
        'overview_path': overview_path,
        'comparisons': comparisons
    }
    summary_path = os.path.join(root_dir, 'safety_factor_comparison_batch.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    summary['summary_path'] = summary_path
    return summary


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
                                 ash_turbulence_drift: Optional[float] = None,
                                 vary_scene_per_checkpoint: bool = False) -> List[Dict]:
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
            if vary_scene_per_checkpoint:
                env.random_scene_counter = checkpoint_episode(model_path) if model_path else 0
            else:
                env.random_scene_counter = 0
        rollout_seed = seed
        if vary_scene_per_checkpoint:
            rollout_seed = seed + (checkpoint_episode(model_path) if model_path else 0)
        path_result = simulate_episode(
            env=env,
            agent=agent,
            seed=rollout_seed,
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
    parser.add_argument('--vary-scene-per-checkpoint', action='store_true',
                        help='Use a different random preview scene for each checkpoint animation.')
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
    parser.add_argument('--safety-factor-comparison', action='store_true',
                        help='Render same-scene low/medium/high safety factor path comparison assets.')
    parser.add_argument('--safety-factor-comparison-scenes', type=int, default=1,
                        help='Number of random scenes rendered for safety factor comparison.')
    parser.add_argument('--safety-factor-candidate-multiplier', type=int, default=8,
                        help='How many random candidates to try per accepted safety comparison scene.')
    parser.add_argument('--min-safety-exposure-diff', type=float, default=1.0,
                        help='Minimum ash exposure spread required for an informative safety comparison scene.')
    parser.add_argument('--min-safety-path-length-diff', type=float, default=35.0,
                        help='Minimum path length spread required for an informative safety comparison scene.')
    parser.add_argument('--min-safety-max-exposure', type=float, default=2.0,
                        help='Minimum maximum ash exposure required for an informative safety comparison scene.')
    parser.add_argument('--allow-failed-safety-comparison', action='store_true',
                        help='Allow safety comparison scenes where some safety factors fail to reach the target.')
    parser.add_argument('--allow-high-safety-higher-exposure', action='store_true',
                        help='Allow scenes where the highest safety factor has higher exposure than the lowest safety factor.')
    parser.add_argument('--safety-factor-values', type=parse_float_list, default=[0.6, 1.0, 1.8],
                        help='Comma-separated safety factors for comparison, e.g. 0.6,1.0,1.8.')
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
        'vary_scene_per_checkpoint': args.vary_scene_per_checkpoint,
        'metric_plots': [],
        'animations': [],
        'safety_factor_comparison': None,
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
                ash_turbulence_drift=args.ash_turbulence_drift,
                vary_scene_per_checkpoint=args.vary_scene_per_checkpoint
            ))

    if args.stitch_progress_gif and summary['animations']:
        summary['stitched_progress_gif'] = stitch_progress_gif(
            summary['animations'],
            output_dir=args.output_dir,
            filename=args.progress_gif_name,
            max_frames_per_clip=args.max_frames_per_progress_clip
        )

    if args.safety_factor_comparison:
        comparison_scene = (
            summary['scene'][0]
            if isinstance(summary.get('scene'), list) and summary['scene']
            else args.scene
        )
        summary['safety_factor_comparison'] = export_safety_factor_comparison_batch(
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            config_path=args.config,
            base_scene_name=comparison_scene,
            cruise_speed=args.cruise_speed,
            fixed_scene_maps=args.fixed_scene_maps,
            seed=args.seed,
            max_steps=args.max_steps,
            random_ash_scenes=args.random_ash_scenes,
            random_centers_range=args.random_centers_range,
            random_scene_seed=args.random_scene_seed,
            dynamic_ash=args.dynamic_ash,
            ash_advection_speed=args.ash_advection_speed,
            ash_diffusion_sigma=args.ash_diffusion_sigma,
            ash_decay_rate=args.ash_decay_rate,
            ash_turbulence_drift=args.ash_turbulence_drift,
            safety_factor_values=args.safety_factor_values,
            scene_count=args.safety_factor_comparison_scenes,
            candidate_multiplier=args.safety_factor_candidate_multiplier,
            min_exposure_diff=args.min_safety_exposure_diff,
            min_path_length_diff=args.min_safety_path_length_diff,
            min_max_exposure=args.min_safety_max_exposure,
            require_all_success=not args.allow_failed_safety_comparison,
            require_high_safety_lower_exposure=not args.allow_high_safety_higher_exposure
        )

    summary_path = os.path.join(args.output_dir, 'asset_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f'Visual assets saved to: {args.output_dir}')
    print(f'Summary: {summary_path}')


if __name__ == '__main__':
    main()
