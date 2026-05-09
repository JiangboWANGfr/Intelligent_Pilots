from flask import Flask, send_from_directory, jsonify, request
from flask_cors import CORS
from datetime import datetime
import os
os.environ.setdefault('MPLBACKEND', 'Agg')
import json
import re
import base64
import cv2
import numpy as np
from urllib.parse import quote
from werkzeug.utils import secure_filename

# Get the absolute path of the server.py file location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(BASE_DIR, 'web')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

app = Flask(__name__, static_folder=WEB_DIR)
CORS(app)

def _parse_request_payload():
    if request.is_json:
        return request.get_json(silent=True) or {}

    payload = {}
    for key, value in request.form.items():
        if isinstance(value, str):
            stripped = value.strip()
            if stripped[:1] in {'{', '['}:
                try:
                    payload[key] = json.loads(stripped)
                    continue
                except json.JSONDecodeError:
                    pass
        payload[key] = value
    return payload

def _parse_bool(value, default=False):
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'on'}
    return default

def _parse_float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def _parse_int(value, default):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default

def _parse_size_pair(value, default):
    if value is None:
        return default
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = [part.strip() for part in value.split(',')]
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        height = max(1, _parse_int(value[0], default[0]))
        width = max(1, _parse_int(value[1], default[1]))
        return height, width
    return default

def _parse_pixel_pair(payload, pair_key, x_key, y_key):
    raw = payload.get(pair_key)
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = None
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        x = _parse_float(raw[0])
        y = _parse_float(raw[1])
        if x is not None and y is not None:
            return x, y

    x = _parse_float(payload.get(x_key))
    y = _parse_float(payload.get(y_key))
    if x is None or y is None:
        return None
    return x, y

def _parse_geo_pair(payload, pair_key, lat_key, lon_key, default):
    raw = payload.get(pair_key)
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            raw = None

    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        lat = _parse_float(raw[0], default[0])
        lon = _parse_float(raw[1], default[1])
        return lat, lon

    lat = _parse_float(payload.get(lat_key), default[0])
    lon = _parse_float(payload.get(lon_key), default[1])
    return lat, lon

def _load_validation_config(payload):
    from src.config.volcanic_ash_config import VolcanicAshConfig

    config_payload = payload.get('config', {})
    if isinstance(config_payload, str):
        try:
            config_payload = json.loads(config_payload)
        except json.JSONDecodeError:
            config_payload = {}

    if isinstance(config_payload, dict) and config_payload:
        config = VolcanicAshConfig.from_dict(config_payload)
        if 'image_size' in config_payload:
            config.image_size = _parse_size_pair(config_payload.get('image_size'), config.image_size)
        return config

    current_config_path = os.path.join(OUTPUT_DIR, 'current_config.json')
    if os.path.exists(current_config_path):
        return VolcanicAshConfig.load(current_config_path)

    return VolcanicAshConfig()

def _save_uploaded_image(uploaded_file, scene_name):
    scene_slug = secure_filename(scene_name) or 'image_validation_scene'
    filename = secure_filename(uploaded_file.filename or '')
    _, ext = os.path.splitext(filename)
    extension = ext.lower() or '.png'
    upload_dir = os.path.join(OUTPUT_DIR, 'web_uploads', scene_slug)
    os.makedirs(upload_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    file_path = os.path.join(upload_dir, f'{scene_slug}_{timestamp}{extension}')
    uploaded_file.save(file_path)
    return file_path

def _resolve_validation_output_paths(payload, scene_name, web_request):
    if not web_request:
        animation_output_dir = payload.get('animation_output_dir') or 'output/validation_animation'
        return {
            'output_json': payload.get('output_json_path', 'output/validated_path.json'),
            'output_plot': payload.get('output_plot_path', 'output/validated_path.png'),
            'animation_output_dir': animation_output_dir,
            'animation_gif_path': payload.get('animation_gif_path') or os.path.join(animation_output_dir, 'validated_path.gif'),
            'animation_video_path': '',
            'animation_manifest_dir': payload.get('animation_manifest_dir', 'output/validation_frames')
        }

    scene_slug = secure_filename(scene_name) or 'image_validation_scene'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    root_dir = os.path.join(OUTPUT_DIR, 'web_validation', f'{scene_slug}_{timestamp}')
    animation_output_dir = payload.get('animation_output_dir') or os.path.join(root_dir, 'animation')
    return {
        'output_json': payload.get('output_json_path') or os.path.join(root_dir, 'validated_path.json'),
        'output_plot': payload.get('output_plot_path') or os.path.join(root_dir, 'validated_path.png'),
        'animation_output_dir': animation_output_dir,
        'animation_gif_path': payload.get('animation_gif_path') or os.path.join(animation_output_dir, 'validated_path.gif'),
        'animation_video_path': '',
        'animation_manifest_dir': payload.get('animation_manifest_dir') or os.path.join(root_dir, 'validation_frames')
    }

def _build_output_url(file_path):
    if not file_path:
        return None

    absolute_path = file_path if os.path.isabs(file_path) else os.path.abspath(os.path.join(BASE_DIR, file_path))
    output_root = os.path.abspath(OUTPUT_DIR)
    if not absolute_path.startswith(output_root):
        return None

    relative_path = os.path.relpath(absolute_path, output_root).replace('\\', '/')
    return f"/api/output/{quote(relative_path, safe='/')}"

def _to_jsonable(value):
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value

def _attach_file_urls(manifest):
    if isinstance(manifest, dict):
        result = {}
        for key, value in manifest.items():
            if key in {'files', 'scene_files'} and isinstance(value, dict):
                result[key] = value
                url_key = 'file_urls' if key == 'files' else 'scene_file_urls'
                result[url_key] = {
                    file_key: _build_output_url(file_path)
                    for file_key, file_path in value.items()
                    if file_path
                }
            elif key == 'outputs' and isinstance(value, dict):
                result[key] = value
                result['output_urls'] = {
                    file_key: _build_output_url(file_path)
                    for file_key, file_path in value.items()
                    if file_path
                }
            else:
                result[key] = _attach_file_urls(value)
        return result
    if isinstance(manifest, list):
        return [_attach_file_urls(item) for item in manifest]
    return manifest

def build_agent_for_config(config, model_path=None, allow_missing_model=False):
    from src.rl_env.volcanic_ash_env import VolcanicAshEnv
    from src.rl_training.ddpg_agent import DDPGAgent, create_agent, infer_checkpoint_algorithm

    env = VolcanicAshEnv(config)
    state_dim = len(DDPGAgent.flatten_state(env.reset()[0]))
    action_dim = int(np.prod(env.action_space.shape))
    algorithm = 'td3'

    if model_path and os.path.isdir(model_path):
        model_path = os.path.join(model_path, 'final_model.pth')

    if model_path and os.path.exists(model_path):
        algorithm = infer_checkpoint_algorithm(model_path)
    agent = create_agent(algorithm, state_dim=state_dim, action_dim=action_dim)

    if model_path:
        if not os.path.exists(model_path):
            if allow_missing_model:
                return None
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        agent.load_model(model_path)

    return agent


def _checkpoint_episode_number(path: str) -> int:
    match = re.search(r'checkpoint_ep(\d+)\.pth$', os.path.basename(path))
    return int(match.group(1)) if match else -1


def _resolve_checkpoint_selection(model_path: str, max_checkpoints: int = 5):
    if not model_path:
        return []

    if os.path.isdir(model_path):
        model_dir = model_path
        final_path = os.path.join(model_dir, 'final_model.pth')
    else:
        model_dir = os.path.dirname(model_path) or '.'
        final_path = model_path if os.path.basename(model_path) == 'final_model.pth' else os.path.join(model_dir, 'final_model.pth')

    checkpoint_paths = sorted(
        [
            os.path.join(model_dir, name)
            for name in os.listdir(model_dir)
            if re.match(r'checkpoint_ep\d+\.pth$', name)
        ] if os.path.isdir(model_dir) else [],
        key=_checkpoint_episode_number
    )

    selected = []
    if checkpoint_paths:
        take = min(max(1, int(max_checkpoints)), len(checkpoint_paths))
        indices = np.linspace(0, len(checkpoint_paths) - 1, take, dtype=int)
        seen = set()
        for index in indices.tolist():
            path = checkpoint_paths[index]
            if path in seen:
                continue
            seen.add(path)
            selected.append({
                'label': f'ep{_checkpoint_episode_number(path):04d}',
                'episode': _checkpoint_episode_number(path),
                'path': path,
            })

    if os.path.exists(final_path):
        selected.append({
            'label': 'final',
            'episode': None,
            'path': final_path,
        })

    deduped = []
    seen_paths = set()
    for item in selected:
        if item['path'] in seen_paths:
            continue
        seen_paths.add(item['path'])
        deduped.append(item)
    return deduped


def _load_model_history(model_path: str):
    if not model_path:
        return None
    model_dir = model_path if os.path.isdir(model_path) else (os.path.dirname(model_path) or '.')
    history_path = os.path.join(model_dir, 'training_history.json')
    if not os.path.exists(history_path):
        return None
    with open(history_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _history_snapshot_for_episode(history, episode):
    if not history or episode is None:
        return None
    episodes = history.get('episodes') or []
    if not episodes:
        return None
    try:
        best_index = min(range(len(episodes)), key=lambda idx: abs(int(episodes[idx]) - int(episode)))
    except Exception:
        return None

    snapshot = {'episode': int(episodes[best_index])}
    for key in [
        'rewards',
        'losses',
        'actor_losses',
        'critic_losses',
        'success_rates',
        'ash_exposures',
        'cross_track_errors',
        'final_distances',
        'safety_factors',
    ]:
        values = history.get(key) or []
        if best_index < len(values):
            snapshot[key] = values[best_index]
    return snapshot


class _RandomActionAgent:
    def __init__(self, seed: int = 0, mode: str = 'wander'):
        self.rng = np.random.default_rng(seed)
        self.mode = mode
        self.step = 0

    def select_action(self, _state, evaluate=True):
        self.step += 1
        if self.mode == 'orbit_left':
            action = 0.85
        elif self.mode == 'orbit_right':
            action = -0.85
        elif self.mode == 'snake':
            action = 0.95 * np.sin(self.step / 5.0)
        elif self.mode == 'drift_left':
            action = 0.35 if self.step < 18 else 0.95
        elif self.mode == 'drift_right':
            action = -0.35 if self.step < 18 else -0.95
        elif self.mode == 'late_panic':
            action = 0.0 if self.step < 24 else (0.95 if (self.step // 8) % 2 == 0 else -0.95)
        elif self.mode == 'hesitate':
            action = 0.18 * np.sin(self.step / 2.0) + self.rng.normal(0.0, 0.08)
        else:
            action = self.rng.uniform(-1.0, 1.0)
        return np.array([float(np.clip(action, -1.0, 1.0))], dtype=np.float32)


def _run_random_policy_preview(planner,
                               config,
                               concentration_map,
                               start_pixel,
                               target_pixel,
                               start_geo,
                               target_geo,
                               scene_name,
                               max_steps: int,
                               reason: str,
                               mode: str = 'wander'):
    random_seed = int(abs(hash((scene_name, tuple(start_pixel), tuple(target_pixel), reason))) % (2**31 - 1))
    planner.set_agent(_RandomActionAgent(seed=random_seed, mode=mode))
    random_result = planner.plan_path(tuple(start_pixel), tuple(target_pixel), max_steps=max_steps)
    random_result['start_geo'] = list(start_geo)
    random_result['target_geo'] = list(target_geo)
    random_result['start_pixel'] = list(start_pixel)
    random_result['target_pixel'] = list(target_pixel)
    random_result['scene_name'] = scene_name
    random_result['planning_method'] = 'random_policy'
    random_result['success'] = False
    random_result['cloud_info'] = random_result.get('cloud_info') or planner.build_cloud_info()
    random_result['validation_info'] = {
        'used_fallback': False,
        'demo_mode': 'random_exploration',
        'fallback_reason': '',
        'agent_status': reason,
        'random_mode': mode,
        'scene_id': None,
        'case_id': None
    }
    return random_result


def _encode_dynamic_preview_frames(dynamic_maps,
                                   threshold: float,
                                   max_frames: int = 36,
                                   target_size=(256, 256)):
    if not dynamic_maps:
        return []
    take = min(max_frames, len(dynamic_maps))
    indices = np.linspace(0, len(dynamic_maps) - 1, take, dtype=int)
    frames = []
    for index in indices.tolist():
        raw_map = np.asarray(dynamic_maps[index])
        map_array = raw_map.astype(np.float32, copy=False)
        if np.issubdtype(raw_map.dtype, np.integer):
            map_array = map_array / 255.0
        map_array = np.clip(map_array, 0.0, 1.0)
        rgba = np.zeros((map_array.shape[0], map_array.shape[1], 4), dtype=np.uint8)
        safe = (map_array > 0.02) & (map_array < 0.20)
        low = (map_array >= 0.20) & (map_array < 0.50)
        medium = (map_array >= 0.50) & (map_array < 0.80)
        high = map_array >= 0.80
        rgba[safe] = [0, 255, 0, 78]
        rgba[low] = [255, 255, 0, 120]
        rgba[medium] = [255, 165, 0, 155]
        rgba[high] = [255, 0, 0, 190]
        rgba[:, :, 3] = np.maximum(rgba[:, :, 3], np.clip(map_array * 165, 0, 190).astype(np.uint8))
        if target_size:
            rgba = cv2.resize(rgba, target_size, interpolation=cv2.INTER_LINEAR)
        success, encoded = cv2.imencode('.png', cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
        if not success:
            continue
        frames.append(f"data:image/png;base64,{base64.b64encode(encoded.tobytes()).decode('ascii')}")
    return frames


def _cloud_center_pixel(concentration_map: np.ndarray):
    weights = np.clip(np.asarray(concentration_map, dtype=np.float32), 0.0, 1.0)
    if float(np.sum(weights)) <= 1e-6:
        h, w = weights.shape[:2]
        return np.array([h / 2.0, w / 2.0], dtype=np.float32)
    ys, xs = np.indices(weights.shape, dtype=np.float32)
    total = float(np.sum(weights))
    cy = float(np.sum(ys * weights) / total)
    cx = float(np.sum(xs * weights) / total)
    return np.array([cy, cx], dtype=np.float32)


def _cloud_hotspot_pixel(concentration_map: np.ndarray):
    weights = np.asarray(concentration_map, dtype=np.float32)
    if weights.size == 0:
        return np.array([0.0, 0.0], dtype=np.float32)
    flat_index = int(np.argmax(weights))
    y, x = np.unravel_index(flat_index, weights.shape)
    return np.array([float(y), float(x)], dtype=np.float32)


def _clip_pixel_path(points, image_size):
    h, w = image_size
    clipped = []
    for point in points:
        clipped.append((
            float(np.clip(point[0], 0, h - 1)),
            float(np.clip(point[1], 0, w - 1)),
        ))
    return clipped


def _build_demo_failure_pixel_path(concentration_map,
                                   config,
                                   start_pixel,
                                   target_pixel,
                                   mode: str,
                                   desired_points: int = 160):
    start = np.array(start_pixel, dtype=np.float32)
    target = np.array(target_pixel, dtype=np.float32)
    cloud = _cloud_center_pixel(concentration_map)
    hotspot = _cloud_hotspot_pixel(concentration_map)
    base = target - start
    dist = float(np.linalg.norm(base))
    if dist < 1e-6:
        dist = 1.0
        direction = np.array([0.0, 1.0], dtype=np.float32)
    else:
        direction = base / dist
    perp = np.array([-direction[1], direction[0]], dtype=np.float32)

    to_cloud = hotspot - start
    proj_len = float(np.dot(to_cloud, direction))
    proj_point = start + direction * np.clip(proj_len, dist * 0.22, dist * 0.78)
    danger_anchor = proj_point * 0.18 + cloud * 0.22 + hotspot * 0.60
    orbit_radius = max(dist * 0.08, min(config.image_size) * 0.05)
    amplitude = max(dist * 0.12, min(config.image_size) * 0.08)

    points = []
    for i in range(desired_points):
        t = i / max(desired_points - 1, 1)
        if mode == 'orbit_left':
            if t < 0.34:
                point = start + (danger_anchor - start) * (t / 0.34)
            else:
                angle = 2.8 * np.pi * ((t - 0.34) / 0.66) + np.pi * 0.3
                center = danger_anchor + perp * amplitude * 0.10
                point = center + orbit_radius * 0.82 * (np.cos(angle) * direction + np.sin(angle) * perp)
        elif mode == 'orbit_right':
            if t < 0.34:
                point = start + (danger_anchor - start) * (t / 0.34)
            else:
                angle = -2.8 * np.pi * ((t - 0.34) / 0.66) + np.pi * 0.2
                center = danger_anchor - perp * amplitude * 0.10
                point = center + orbit_radius * 0.82 * (np.cos(angle) * direction + np.sin(angle) * perp)
        elif mode == 'snake':
            point = start + (danger_anchor - start) * min(t * 1.08, 1.0)
            point = point + perp * (np.sin(t * 5.2 * np.pi) * amplitude * (0.9 - 0.20 * t))
        elif mode == 'drift_left':
            if t < 0.52:
                point = start + (danger_anchor - start) * (t / 0.52)
            else:
                drift_t = (t - 0.52) / 0.48
                point = danger_anchor + direction * drift_t * amplitude * 0.18
                point = point + perp * (0.12 + 0.82 * drift_t) * amplitude
        elif mode == 'drift_right':
            if t < 0.52:
                point = start + (danger_anchor - start) * (t / 0.52)
            else:
                drift_t = (t - 0.52) / 0.48
                point = danger_anchor + direction * drift_t * amplitude * 0.18
                point = point - perp * (0.12 + 0.82 * drift_t) * amplitude
        elif mode == 'late_panic':
            if t < 0.45:
                point = start + (danger_anchor - start) * (t / 0.45)
            else:
                panic_t = (t - 0.45) / 0.55
                point = danger_anchor + direction * (panic_t * amplitude * 0.4) + perp * np.sin(panic_t * 2.6 * np.pi) * amplitude * 1.35
        elif mode == 'hesitate':
            point = start + (danger_anchor - start) * min(t * 0.96, 0.96)
            point = point + perp * np.sin(t * 9.0 * np.pi) * amplitude * 0.42
            point = point - direction * np.sin(t * 4.0 * np.pi) * amplitude * 0.18
        else:  # wander
            point = start + (danger_anchor - start) * min(t * 1.02, 1.0)
            point = point + perp * np.sin(t * 7.0 * np.pi) * amplitude * 0.55
            point = point + direction * np.sin(t * 3.0 * np.pi) * amplitude * 0.18
        points.append((float(point[0]), float(point[1])))
    return _clip_pixel_path(points, config.image_size)


def _execute_case_planning(data,
                           *,
                           save_case: bool = True,
                           export_json: bool = None,
                           export_plot: bool = None,
                           export_animation: bool = None,
                           allow_fallback: bool = True,
                           random_on_missing_agent: bool = False,
                           random_demo_mode: str = None,
                           synthetic_failure_mode: str = None,
                           model_override: str = None,
                           case_name_override: str = None):
    from src.config.volcanic_ash_config import VolcanicAshConfig
    from src.path_planning.animation_exporter import ValidationAnimationExporter
    from src.path_planning.fallback_planner import FallbackPlanner
    from src.path_planning.planner import PathPlanner
    from src.web.case_manager import WebCaseManager

    manager = WebCaseManager(OUTPUT_DIR)
    scene_id = str(data.get('scene_id') or '')
    if not scene_id:
        raise ValueError('scene_id is required')
    scene = manager.load_scene(scene_id)
    scene_concentration_map = manager.load_scene_map(scene)
    scene_config = VolcanicAshConfig.from_dict(scene.get('config', {}))
    requested_scene_config = _apply_conversion_request_config(
        VolcanicAshConfig.from_dict(scene_config.to_dict()),
        data
    )
    config = _apply_planning_options(VolcanicAshConfig.from_dict(requested_scene_config.to_dict()), data)

    scene_name = str(case_name_override or data.get('case_name') or scene.get('scene_name') or 'planning_case')
    model_path = model_override or data.get('model_path', 'models/final_model.pth')
    if export_json is None:
        export_json = save_case
    if export_plot is None:
        export_plot = save_case
    if export_animation is None:
        export_animation = save_case

    needs_output_dir = save_case or export_json or export_plot or export_animation
    case_id = None
    case_dir = None
    copied_scene_files = {}
    if needs_output_dir:
        case_meta = manager.create_case_dir(scene_name)
        case_id = case_meta['case_id']
        case_dir = case_meta['case_dir']
        copied_scene_files = manager.copy_scene_files_to_case(scene, case_dir)

    start_pixel = _parse_pixel_pair(data, 'start_pixel', 'start_pixel_x', 'start_pixel_y')
    target_pixel = _parse_pixel_pair(data, 'target_pixel', 'target_pixel_x', 'target_pixel_y')
    default_start = (31.1443, 121.8083)
    default_target = (35.5494, 139.7798)
    start_geo = _parse_geo_pair(data, 'start_position', 'start_lat', 'start_lon', default_start)
    target_geo = _parse_geo_pair(data, 'target_position', 'target_lat', 'target_lon', default_target)

    if start_pixel is None or target_pixel is None:
        requested_size = _parse_size_pair(data.get('planning_image_size'), (1024, 1024))
        config = _expand_planning_config_for_route(config, start_geo, target_geo, requested_size)
        concentration_map = _embed_scene_map_in_planning_map(
            scene_concentration_map,
            requested_scene_config,
            config
        )
    else:
        concentration_map = scene_concentration_map

    prefer_fallback = _parse_bool(data.get('prefer_fallback_planner'), default=False)
    agent = None
    fallback_reason = ''
    if not prefer_fallback:
        try:
            agent = build_agent_for_config(
                config,
                model_path=model_path,
                allow_missing_model=True
            )
        except Exception as exc:
            fallback_reason = f'agent_load_error:{exc}'
    planner = PathPlanner(config, agent)
    planner.set_external_concentration_map(concentration_map, scene_name=scene_name)

    if start_pixel is None or target_pixel is None:
        start_pixel, target_pixel = planner.convert_geo_input(start_geo, target_geo)
    else:
        x0, y0 = start_pixel
        x1, y1 = target_pixel
        h, w = tuple(config.image_size)
        start_pixel = (int(np.clip(round(y0), 0, h - 1)), int(np.clip(round(x0), 0, w - 1)))
        target_pixel = (int(np.clip(round(y1), 0, h - 1)), int(np.clip(round(x1), 0, w - 1)))
        start_geo = planner.ash_model.pixel_to_geo(start_pixel[1], start_pixel[0])
        target_geo = planner.ash_model.pixel_to_geo(target_pixel[1], target_pixel[0])

    max_steps = _parse_int(data.get('max_steps'), 500)
    fallback_limit = _parse_float(data.get('fallback_concentration_limit'), config.concentration_threshold)
    effective_fallback_limit, effective_risk_radius, fallback_cost_safety_factor = _safety_adjusted_planning_limits(
        config,
        fallback_limit
    )
    setattr(config, 'fallback_cost_safety_factor', fallback_cost_safety_factor)
    used_fallback = False
    rl_result = None
    if agent is not None and not prefer_fallback:
        try:
            rl_result = planner.plan_path(tuple(start_pixel), tuple(target_pixel), max_steps=max_steps)
            rl_result['start_geo'] = list(start_geo)
            rl_result['target_geo'] = list(target_geo)
            rl_result['start_pixel'] = list(start_pixel)
            rl_result['target_pixel'] = list(target_pixel)
            rl_result['cloud_info'] = rl_result.get('cloud_info') or planner.build_cloud_info()
        except Exception as exc:
            fallback_reason = f'rl_error:{exc}'

    if prefer_fallback and allow_fallback:
        used_fallback = True
        fallback_reason = 'safety_aware_web_planner'
    elif rl_result is None and allow_fallback:
        used_fallback = True
        fallback_reason = fallback_reason or 'rl_agent_unavailable'
    elif allow_fallback and (not rl_result.get('success', False) or rl_result.get('max_concentration', 1.0) > effective_fallback_limit):
        used_fallback = True
        fallback_reason = 'rl_path_rejected'

    if used_fallback:
        fallback_points = FallbackPlanner(config).plan(
            concentration_map,
            tuple(start_pixel),
            tuple(target_pixel),
            max_concentration=effective_fallback_limit,
            risk_inflation_radius=effective_risk_radius,
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
        path_result['validation_info'] = {
            'used_fallback': True,
            'fallback_reason': fallback_reason,
            'fallback_limit': fallback_limit,
            'effective_fallback_limit': effective_fallback_limit,
            'effective_risk_inflation_radius': effective_risk_radius,
            'fallback_cost_safety_factor': fallback_cost_safety_factor,
            'fallback_summary': FallbackPlanner(config).summarize_path(concentration_map, fallback_points),
            'scene_id': scene_id,
            'case_id': case_id
        }
        if rl_result is not None:
            path_result['validation_info']['rl_attempt'] = {
                'success': rl_result.get('success', False),
                'max_concentration': rl_result.get('max_concentration', 0.0),
                'steps_taken': rl_result.get('steps_taken', 0)
            }
    elif synthetic_failure_mode:
        demo_points = _build_demo_failure_pixel_path(
            concentration_map,
            config,
            start_pixel,
            target_pixel,
            synthetic_failure_mode,
            desired_points=min(max_steps, 180)
        )
        path_result = planner.build_path_data_from_pixel_path(
            demo_points,
            planning_method='demo_failure',
            start_geo=start_geo,
            target_geo=target_geo,
            start_pixel=start_pixel,
            target_pixel=target_pixel
        )
        path_result['success'] = False
        path_result['validation_info'] = {
            'used_fallback': False,
            'demo_mode': 'synthetic_failure',
            'random_mode': synthetic_failure_mode,
            'fallback_reason': '',
            'agent_status': fallback_reason or 'low_success_checkpoint_demo',
            'fallback_limit': fallback_limit,
            'effective_fallback_limit': effective_fallback_limit,
            'effective_risk_inflation_radius': effective_risk_radius,
            'fallback_cost_safety_factor': fallback_cost_safety_factor,
            'scene_id': scene_id,
            'case_id': case_id
        }
    elif rl_result is None and random_on_missing_agent:
        path_result = _run_random_policy_preview(
            planner,
            config,
            concentration_map,
            start_pixel,
            target_pixel,
            start_geo,
            target_geo,
            scene_name,
            max_steps,
            fallback_reason or 'rl_agent_unavailable',
            mode=random_demo_mode or 'wander'
        )
        path_result['validation_info'].update({
            'fallback_limit': fallback_limit,
            'effective_fallback_limit': effective_fallback_limit,
            'effective_risk_inflation_radius': effective_risk_radius,
            'fallback_cost_safety_factor': fallback_cost_safety_factor,
            'scene_id': scene_id,
            'case_id': case_id
        })
    else:
        path_result = rl_result
        path_result['validation_info'] = path_result.get('validation_info') or {
            'used_fallback': False,
            'fallback_reason': '',
            'fallback_limit': fallback_limit,
            'effective_fallback_limit': effective_fallback_limit,
            'effective_risk_inflation_radius': effective_risk_radius,
            'fallback_cost_safety_factor': fallback_cost_safety_factor,
            'scene_id': scene_id,
            'case_id': case_id
        }

    path_result['scene_name'] = scene_name
    path_result['cloud_info'] = path_result.get('cloud_info') or planner.build_cloud_info()
    json_ready_result = dict(path_result)
    dynamic_maps = json_ready_result.pop('concentration_maps', None)
    if dynamic_maps is None and bool(getattr(config, 'enable_dynamic_ash', False)) and export_animation:
        dynamic_maps = _generate_dynamic_maps_for_waypoints(
            config,
            concentration_map,
            len(json_ready_result.get('waypoints', []))
        )
    dynamic_preview_frames = []
    if bool(getattr(config, 'enable_dynamic_ash', False)):
        preview_config = VolcanicAshConfig.from_dict(requested_scene_config.to_dict())
        for attr in [
            'enable_dynamic_ash',
            'ash_advection_speed',
            'ash_diffusion_sigma',
            'ash_decay_rate',
            'ash_turbulence_drift',
            'ash_local_deformation_strength',
            'ash_local_flow_scale',
            'ash_local_flow_smoothness',
            'ash_shear_strength',
            'ash_local_flow_update_interval',
            'ash_dynamic_update_interval',
            'ash_dynamic_renormalize',
            'ash_advection_speed_min',
            'ash_advection_speed_max',
            'ash_wind_direction_jitter',
            'ash_rotation_rate',
            'ash_rotation_rate_jitter',
        ]:
            if hasattr(config, attr):
                setattr(preview_config, attr, getattr(config, attr))
        preview_dynamic_maps = _generate_dynamic_maps_for_waypoints(
            preview_config,
            scene_concentration_map,
            len(json_ready_result.get('waypoints', []))
        )
        dynamic_preview_frames = _encode_dynamic_preview_frames(
            preview_dynamic_maps,
            float(preview_config.concentration_threshold),
            max_frames=32,
            target_size=(320, 320)
        ) if preview_dynamic_maps is not None else []
        if dynamic_preview_frames:
            json_ready_result['dynamic_preview_frames'] = dynamic_preview_frames

    output_json = None
    output_plot = None
    animation_export = {}
    if export_json and case_dir:
        output_json = os.path.join(case_dir, 'planned_path.json')
        planner.export_path_json(json_ready_result, output_json)
    if export_plot and case_dir:
        output_plot = os.path.join(case_dir, 'path_plot.png')
        planner.visualize_path(json_ready_result, save_path=output_plot)
    if export_animation and case_dir:
        animation_dir = os.path.join(case_dir, 'animation')
        animation_gif = os.path.join(animation_dir, 'flight_animation.gif')
        animation_result = dict(path_result)
        if dynamic_maps is not None:
            animation_result['concentration_maps'] = dynamic_maps
        animation_export = ValidationAnimationExporter(config, concentration_map).export(
            animation_result,
            output_dir=animation_dir,
            gif_path=animation_gif,
            video_path=None,
            fps=_parse_int(data.get('animation_fps'), 12),
            max_frames=_parse_int(data.get('animation_max_frames'), 180),
            save_frames=_parse_bool(data.get('animation_save_frames'), False)
        ) or {}
        if isinstance(animation_export, dict):
            animation_export.pop('video_path', None)
            animation_export.pop('video_codec', None)
    if animation_export:
        json_ready_result.setdefault('validation_info', {})['animation_export'] = animation_export

    case_payload = None
    if save_case and case_dir and case_id:
        outputs = {
            'planned_path': output_json,
            'path_plot': output_plot,
            'animation_gif': animation_export.get('gif_path'),
            'case_manifest': os.path.join(case_dir, 'case.json')
        }
        manifest = {
            'case_id': case_id,
            'case_name': scene_name,
            'created_at': datetime.now().isoformat(timespec='seconds'),
            'scene_id': scene_id,
            'scene_name': scene.get('scene_name'),
            'scene_files': copied_scene_files,
            'map': {
                'image_size': list(config.image_size),
                'geo_center_lat': config.geo_center_lat,
                'geo_center_lon': config.geo_center_lon,
                'geo_span_lat': config.geo_span_lat,
                'geo_span_lon': config.geo_span_lon
            },
            'flight': {
                'start_pixel': [int(start_pixel[1]), int(start_pixel[0])],
                'target_pixel': [int(target_pixel[1]), int(target_pixel[0])],
                'start_geo': list(start_geo),
                'target_geo': list(target_geo)
            },
            'planning': {
                'model_path': model_path,
                'safety_factor': float(config.fixed_safety_factor),
                'dynamic_ash': bool(config.enable_dynamic_ash),
                'max_steps': max_steps,
                'fallback_concentration_limit': fallback_limit,
                'effective_fallback_concentration_limit': effective_fallback_limit,
                'effective_risk_inflation_radius': effective_risk_radius,
                'fallback_cost_safety_factor': fallback_cost_safety_factor,
                'planning_method': json_ready_result.get('planning_method')
            },
            'outputs': outputs,
            'path_data': json_ready_result
        }
        manager.save_case(case_id, manifest)
        case_payload = _attach_file_urls(manifest)

    return _to_jsonable({
        'success': True,
        'preview_only': not save_case,
        'case': case_payload,
        'path_data': json_ready_result,
        'dynamic_preview_frames': dynamic_preview_frames or None,
        'animation_export': animation_export or None,
        'output_file_url': _build_output_url(output_json) if output_json else None,
        'plot_file_url': _build_output_url(output_plot) if output_plot else None,
        'animation_gif_url': _build_output_url(animation_export.get('gif_path')) if animation_export.get('gif_path') else None,
        'message': '规划完成并已保存为案例' if save_case else '规划完成，已返回即时预览结果'
    })

def _apply_conversion_request_config(config, payload):
    config.image_size = _parse_size_pair(
        payload.get('image_size'),
        tuple(getattr(config, 'image_size', (768, 768)))
    )
    threshold = _parse_float(payload.get('concentration_threshold'))
    if threshold is not None:
        config.concentration_threshold = threshold
    config.geo_center_lat = _parse_float(payload.get('geo_center_lat'), config.geo_center_lat)
    config.geo_center_lon = _parse_float(payload.get('geo_center_lon'), config.geo_center_lon)
    config.geo_span_lat = _parse_float(payload.get('geo_span_lat'), config.geo_span_lat)
    config.geo_span_lon = _parse_float(payload.get('geo_span_lon'), config.geo_span_lon)
    return config

def _apply_planning_options(config, payload):
    config = _apply_conversion_request_config(config, payload)
    safety_factor = _parse_float(payload.get('safety_factor'), None)
    if safety_factor is not None:
        config.safety_factor_mode = 'fixed'
        config.fixed_safety_factor = safety_factor
        config.min_safety_factor = min(float(config.min_safety_factor), safety_factor)
        config.max_safety_factor = max(float(config.max_safety_factor), safety_factor)
    config.enable_dynamic_ash = _parse_bool(
        payload.get('dynamic_ash'),
        default=bool(getattr(config, 'enable_dynamic_ash', False))
    )
    for key in [
        'ash_advection_speed',
        'ash_diffusion_sigma',
        'ash_decay_rate',
        'ash_turbulence_drift',
        'ash_local_deformation_strength',
        'ash_local_flow_scale',
        'ash_local_flow_smoothness',
        'ash_shear_strength'
    ]:
        parsed = _parse_float(payload.get(key), None)
        if parsed is not None:
            setattr(config, key, parsed)
    interval = _parse_int(payload.get('ash_local_flow_update_interval'), None)
    if interval is not None:
        config.ash_local_flow_update_interval = max(1, interval)
    return config

def _geo_bounds_from_config(config):
    half_lat = float(config.geo_span_lat) / 2.0
    half_lon = float(config.geo_span_lon) / 2.0
    return {
        'south': float(config.geo_center_lat) - half_lat,
        'north': float(config.geo_center_lat) + half_lat,
        'west': float(config.geo_center_lon) - half_lon,
        'east': float(config.geo_center_lon) + half_lon
    }

def _geo_to_pixel_in_config(config, lat, lon):
    img_h, img_w = tuple(config.image_size)
    px = int(round((float(lon) - config.geo_center_lon) / config.geo_span_lon * img_w + img_w / 2))
    py = int(round((0.5 - (float(lat) - config.geo_center_lat) / config.geo_span_lat) * img_h))
    return py, px

def _expand_planning_config_for_route(scene_config, start_geo, target_geo, planning_image_size=(1024, 1024)):
    scene_bounds = _geo_bounds_from_config(scene_config)
    margin_lat = max(float(scene_config.geo_span_lat) * 0.45, 0.6)
    margin_lon = max(float(scene_config.geo_span_lon) * 0.45, 0.6)
    lats = [scene_bounds['south'], scene_bounds['north'], float(start_geo[0]), float(target_geo[0])]
    lons = [scene_bounds['west'], scene_bounds['east'], float(start_geo[1]), float(target_geo[1])]
    south = max(-89.0, min(lats) - margin_lat)
    north = min(89.0, max(lats) + margin_lat)
    west = max(-179.0, min(lons) - margin_lon)
    east = min(179.0, max(lons) + margin_lon)

    planning_config = scene_config.from_dict(scene_config.to_dict())
    planning_config.geo_center_lat = (south + north) / 2.0
    planning_config.geo_center_lon = (west + east) / 2.0
    planning_config.geo_span_lat = max(north - south, float(scene_config.geo_span_lat))
    planning_config.geo_span_lon = max(east - west, float(scene_config.geo_span_lon))
    planning_config.image_size = tuple(planning_image_size)
    return planning_config

def _embed_scene_map_in_planning_map(scene_map, scene_config, planning_config):
    planning_h, planning_w = tuple(planning_config.image_size)
    planning_map = np.zeros((planning_h, planning_w), dtype=np.float32)
    scene_bounds = _geo_bounds_from_config(scene_config)

    y_north, x_west = _geo_to_pixel_in_config(planning_config, scene_bounds['north'], scene_bounds['west'])
    y_south, x_east = _geo_to_pixel_in_config(planning_config, scene_bounds['south'], scene_bounds['east'])
    x0, x1 = sorted((x_west, x_east))
    y0, y1 = sorted((y_north, y_south))
    x0_clip, x1_clip = max(0, x0), min(planning_w, x1)
    y0_clip, y1_clip = max(0, y0), min(planning_h, y1)
    if x1_clip <= x0_clip or y1_clip <= y0_clip:
        return planning_map

    resized = cv2.resize(
        np.asarray(scene_map, dtype=np.float32),
        (max(1, x1 - x0), max(1, y1 - y0)),
        interpolation=cv2.INTER_LINEAR
    )
    sx0 = x0_clip - x0
    sx1 = sx0 + (x1_clip - x0_clip)
    sy0 = y0_clip - y0
    sy1 = sy0 + (y1_clip - y0_clip)
    planning_map[y0_clip:y1_clip, x0_clip:x1_clip] = np.maximum(
        planning_map[y0_clip:y1_clip, x0_clip:x1_clip],
        resized[sy0:sy1, sx0:sx1]
    )
    return planning_map

def _safety_adjusted_planning_limits(config, base_limit):
    safety_factor = max(float(getattr(config, 'fixed_safety_factor', 1.0)), 0.05)
    base = float(base_limit)
    if safety_factor <= 0.75:
        effective_limit = min(0.82, base + 0.28)
        inflation_radius = max(1.0, float(getattr(config, 'path_risk_inflation_radius', 8.0)) * 0.45)
        fallback_cost_safety_factor = 0.08
    elif safety_factor >= 1.5:
        effective_limit = max(0.08, base * 0.45)
        inflation_radius = max(10.0, float(getattr(config, 'path_risk_inflation_radius', 8.0)) * 2.2)
        fallback_cost_safety_factor = 2.4
    else:
        effective_limit = base
        inflation_radius = float(getattr(config, 'path_risk_inflation_radius', 8.0))
        fallback_cost_safety_factor = 1.0
    return effective_limit, inflation_radius, fallback_cost_safety_factor

def _parse_plume_scale(value, default=1.0):
    scale = _parse_float(value, default)
    if scale is None:
        return float(default)
    return float(np.clip(scale, 0.5, 6.0))

def _generate_dynamic_maps_for_waypoints(config, concentration_map, waypoint_count):
    if not bool(getattr(config, 'enable_dynamic_ash', False)) or waypoint_count <= 0:
        return None
    from src.rl_env.volcanic_ash_env import VolcanicAshEnv

    env = VolcanicAshEnv(config)
    env.set_external_concentration_map(
        concentration_map,
        config=config,
        scene_name=config.scene_name or 'dynamic_case'
    )
    env._initialize_dynamic_ash_fields()
    maps = []
    for _ in range(int(waypoint_count)):
        maps.append((env.concentration_map * 255.0).astype(np.uint8))
        env.step_count += 1
        env._advance_dynamic_ash()
    return maps

@app.route('/')
def serve_index():
    return send_from_directory(WEB_DIR, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(WEB_DIR, path)

@app.route('/api/presets', methods=['GET'])
def get_presets():
    from src.config.volcanic_ash_config import get_preset_configs
    
    presets = get_preset_configs()
    result = {}
    for name, config in presets.items():
        result[name] = config.to_dict()
    
    return jsonify({'success': True, 'presets': result})

@app.route('/api/generate', methods=['POST'])
def generate_images():
    data = request.json
    
    from src.config.volcanic_ash_config import VolcanicAshConfig
    from src.generation.image_generator import StaticImageGenerator, DynamicSimulation
    from src.model.gmm_model import GMMVolcanicAshModel
    
    try:
        config = VolcanicAshConfig.from_dict(data.get('config', {}))
        
        generator = StaticImageGenerator(config)
        num_images = data.get('num_images', 5)
        static_results = generator.generate_static_images(
            output_dir='output/static',
            num_images=num_images
        )
        
        if data.get('generate_dynamic', False):
            simulator = DynamicSimulation(GMMVolcanicAshModel(config))
            dynamic_results = simulator.generate_dynamic_sequence(
                num_frames=data.get('dynamic_frames', 20),
                output_dir='output/dynamic'
            )
            
            static_results['dynamic'] = dynamic_results
        
        config.save('output/current_config.json')
        
        return jsonify({
            'success': True,
            'results': static_results,
            'message': f'成功生成 {len(static_results["generated_images"])} 张静态图像'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/train', methods=['POST'])
def train_model():
    data = request.json
    
    from src.config.volcanic_ash_config import VolcanicAshConfig, get_training_scene_configs
    from src.rl_training.trainer import Trainer
    
    try:
        config = VolcanicAshConfig.from_dict(data.get('config', {}))
        scene_names = data.get('training_scene_names') or config.training_scene_names
        scene_configs = get_training_scene_configs(scene_names) if scene_names else [VolcanicAshConfig.from_dict(config.to_dict())]
        config.training_scene_names = [scene.scene_name for scene in scene_configs]
        
        trainer = Trainer(
            config=config,
            num_episodes=data.get('episodes', 300),
            max_steps_per_episode=data.get('max_steps', 300),
            learning_rate=data.get('learning_rate', 1e-4),
            buffer_size=data.get('buffer_size', 300000),
            batch_size=data.get('batch_size', 128),
            noise_decay=data.get('noise_decay', 0.999),
            algorithm=data.get('algorithm', 'td3'),
            policy_noise=data.get('policy_noise', 0.2),
            noise_clip=data.get('noise_clip', 0.5),
            policy_delay=data.get('policy_delay', 2),
            device=data.get('device', 'auto'),
            save_dir='models',
            scene_configs=scene_configs
        )
        
        agent, history = trainer.train(
            update_every=data.get('update_every', 10),
            log_interval=50
        )
        
        model_info = {
            'model_path': 'models/final_model.pth',
            'training_curves': 'models/training_curves.png',
            'total_episodes': trainer.num_episodes,
            'algorithm': trainer.algorithm,
            'final_reward': history['rewards'][-1] if history['rewards'] else 0,
            'training_scene_names': config.training_scene_names
        }
        
        return jsonify({
            'success': True,
            'model_info': model_info,
            'message': '模型训练完成！'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/plan', methods=['POST'])
def plan_path():
    data = request.json
    
    from src.config.volcanic_ash_config import VolcanicAshConfig
    from src.path_planning.planner import PathPlanner
    
    try:
        config = VolcanicAshConfig.from_dict(data.get('config', {}))
        
        agent = build_agent_for_config(
            config,
            model_path=data.get('model_path', 'models/final_model.pth'),
            allow_missing_model=True
        )
        
        planner = PathPlanner(config, agent)
         
        start_geo = tuple(data.get('start_position', [34.5, 119.5]))
        target_geo = tuple(data.get('target_position', [35.5, 120.5]))
         
        path_result = planner.plan_path_geo_with_fallback(
            start_geo,
            target_geo,
            max_steps=data.get('max_steps', 500),
            max_concentration=data.get('fallback_concentration_limit')
        )
        
        output_file = 'output/planned_path.json'
        planner.export_path_json(path_result, output_file)
        
        return jsonify({
            'success': True,
            'path_data': path_result,
            'output_file': output_file,
            'message': '路径规划完成！'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/multi-plan', methods=['POST'])
def multi_constraint_plan():
    data = request.json
    
    from src.config.volcanic_ash_config import VolcanicAshConfig
    from src.path_planning.multi_constraint import MultiConstraintPlanner
    
    try:
        config = VolcanicAshConfig.from_dict(data.get('config', {}))
        
        agent = build_agent_for_config(config, model_path=data.get('model_path', 'models/final_model.pth'))
        
        multi_planner = MultiConstraintPlanner(config, agent)
        
        solutions = multi_planner.generate_multiple_solutions(
            start_geo=tuple(data.get('start_position', [34.8, 119.8])),
            target_geo=tuple(data.get('target_position', [35.2, 120.2])),
            risk_tolerance_levels=data.get('risk_levels', ['low', 'medium', 'high']),
            fuel_constraints=data.get('fuel_limits', [80.0, 120.0]),
            max_steps=data.get('max_steps', 350)
        )
        
        output_file = 'output/multi_constraint_solutions.json'
        multi_planner.export_solutions_json(solutions, output_file)
        
        report = multi_planner.generate_comparison_report(solutions)
        
        return jsonify({
            'success': True,
            'solutions_count': len(solutions.get('solutions', [])),
            'solutions': solutions,
            'report': report,
            'output_file': output_file
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/validate-image', methods=['POST'])
def validate_image():
    data = _parse_request_payload()
     
    from src.path_planning.validation_pipeline import ValidationPipeline
     
    try:
        config = _apply_conversion_request_config(_load_validation_config(data), data)
        pipeline = ValidationPipeline(
            config=config,
            model_path=data.get('model_path', 'models/final_model.pth')
        )
         
        scene_name = str(data.get('scene_name') or 'image_validation_scene')
        uploaded_file = request.files.get('image_file')
        image_path = data.get('image_path')
        if uploaded_file and uploaded_file.filename:
            image_path = _save_uploaded_image(uploaded_file, scene_name)
        if not image_path:
            raise ValueError('image_path or image_file is required')

        web_request = _parse_bool(data.get('web_request'), default=uploaded_file is not None)
        output_paths = _resolve_validation_output_paths(data, scene_name, web_request)
        conversion_output_dir = data.get('conversion_output_dir') or os.path.join(
            os.path.dirname(output_paths['output_json']),
            'converted_map'
        )
        default_start = (
            config.geo_center_lat - config.geo_span_lat * 0.35,
            config.geo_center_lon - config.geo_span_lon * 0.35
        )
        default_target = (
            config.geo_center_lat + config.geo_span_lat * 0.35,
            config.geo_center_lon + config.geo_span_lon * 0.35
        )
        start_pixel = _parse_pixel_pair(data, 'start_pixel', 'start_pixel_x', 'start_pixel_y')
        target_pixel = _parse_pixel_pair(data, 'target_pixel', 'target_pixel_x', 'target_pixel_y')
         
        result = pipeline.validate_image(
            image_source=image_path,
            start_geo=None if start_pixel is not None else _parse_geo_pair(
                data, 'start_position', 'start_lat', 'start_lon', default_start
            ),
            target_geo=None if target_pixel is not None else _parse_geo_pair(
                data, 'target_position', 'target_lat', 'target_lon', default_target
            ),
            start_pixel=start_pixel,
            target_pixel=target_pixel,
            output_json_path=output_paths['output_json'],
            output_plot_path=output_paths['output_plot'],
            scene_name=scene_name,
            fallback_concentration_limit=_parse_float(data.get('fallback_concentration_limit')),
            animation_output_dir=output_paths['animation_output_dir'],
            animation_gif_path=output_paths['animation_gif_path'],
            animation_video_path=output_paths['animation_video_path'],
            animation_fps=_parse_int(data.get('animation_fps', 12), 12),
            animation_save_frames=_parse_bool(data.get('animation_save_frames', False), False),
            animation_max_frames=_parse_int(data.get('animation_max_frames', 180), 180),
            conversion_mode=data.get('conversion_mode', 'auto'),
            invert=data.get('invert', 'auto'),
            blur_kernel=_parse_int(data.get('blur_kernel', 5), 5),
            plume_scale=_parse_plume_scale(data.get('plume_scale'), 1.5),
            conversion_output_dir=conversion_output_dir
        )
         
        manifest = pipeline.build_animation_export_manifest(
            result,
            output_dir=output_paths['animation_manifest_dir']
        )
        animation_export = result.get('validation_info', {}).get('animation_export') or {}
        animation_gif_path = animation_export.get('gif_path') or output_paths['animation_gif_path']
        if isinstance(animation_export, dict):
            animation_export.pop('video_path', None)
            animation_export.pop('video_codec', None)
 
        response_payload = _to_jsonable({
            'success': True,
            'path_data': result,
            'output_file': output_paths['output_json'],
            'output_file_url': _build_output_url(output_paths['output_json']),
            'plot_file': output_paths['output_plot'],
            'plot_file_url': _build_output_url(output_paths['output_plot']),
            'animation_manifest': manifest,
            'animation_export': animation_export,
            'animation_gif_path': animation_gif_path,
            'animation_gif_url': _build_output_url(animation_gif_path),
            'converted_map': result.get('validation_info', {}).get('conversion_outputs'),
            'converted_map_urls': {
                key: _build_output_url(value)
                for key, value in (result.get('validation_info', {}).get('conversion_outputs') or {}).items()
            },
            'source_image_path': image_path,
            'source_image_url': _build_output_url(image_path),
            'message': '图像验证路径生成完成！'
        })

        return jsonify(response_payload)
     
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/convert-image', methods=['POST'])
def convert_image():
    data = _parse_request_payload()

    from src.generation.image_converter import AshImageConverter

    try:
        config = _apply_conversion_request_config(_load_validation_config(data), data)
        scene_name = str(data.get('scene_name') or 'image_validation_scene')
        uploaded_file = request.files.get('image_file')
        image_path = data.get('image_path')
        if uploaded_file and uploaded_file.filename:
            image_path = _save_uploaded_image(uploaded_file, scene_name)
        if not image_path:
            raise ValueError('image_path or image_file is required')

        converter = AshImageConverter(config)
        converted = converter.convert_to_scene(
            image_path,
            scene_name=scene_name,
            mode=data.get('conversion_mode', 'auto'),
            invert=data.get('invert', 'auto'),
            blur_kernel=_parse_int(data.get('blur_kernel', 5), 5),
            plume_scale=_parse_plume_scale(data.get('plume_scale'), 1.5)
        )
        scene_slug = secure_filename(scene_name) or 'image_validation_scene'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        output_dir = os.path.join(OUTPUT_DIR, 'web_converted_maps', f'{scene_slug}_{timestamp}')
        outputs = converter.save_standard_outputs(
            converted['concentration_map'],
            output_dir,
            prefix='real_ash'
        )
        config_path = os.path.join(output_dir, 'real_ash_config.json')
        converted['config'].save(config_path)

        return jsonify(_to_jsonable({
            'success': True,
            'scene_name': scene_name,
            'summary': converted['summary'],
            'config': converted['config'].to_dict(),
            'config_path': config_path,
            'config_url': _build_output_url(config_path),
            'outputs': outputs,
            'output_urls': {key: _build_output_url(value) for key, value in outputs.items()},
            'source_image_path': image_path,
            'source_image_url': _build_output_url(image_path),
            'message': '现实图已转换为标准浓度图'
        }))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/scenes/import-image', methods=['POST'])
def import_scene_image():
    data = _parse_request_payload()

    from src.web.case_manager import WebCaseManager

    try:
        manager = WebCaseManager(OUTPUT_DIR)
        config = _apply_conversion_request_config(_load_validation_config(data), data)
        scene_name = str(data.get('scene_name') or 'image_scene')
        uploaded_file = request.files.get('image_file')
        image_path = data.get('image_path')
        if uploaded_file and uploaded_file.filename:
            image_path = _save_uploaded_image(uploaded_file, scene_name)
        if not image_path:
            raise ValueError('image_path or image_file is required')

        manifest = manager.save_imported_scene(
            image_path=image_path,
            config=config,
            scene_name=scene_name,
            conversion_mode=data.get('conversion_mode', 'auto'),
            invert=data.get('invert', 'auto'),
            blur_kernel=_parse_int(data.get('blur_kernel', 5), 5),
            plume_scale=_parse_plume_scale(data.get('plume_scale'), 1.5)
        )
        return jsonify(_to_jsonable({
            'success': True,
            'scene': _attach_file_urls(manifest),
            'message': '场景已导入并保存'
        }))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/scenes/generated', methods=['POST'])
def generate_scene():
    data = request.get_json(silent=True) or {}

    from src.config.volcanic_ash_config import VolcanicAshConfig
    from src.model.gmm_model import GMMVolcanicAshModel
    from src.model.random_ash_scene_generator import RandomAshSceneGenerator
    from src.web.case_manager import WebCaseManager

    try:
        manager = WebCaseManager(OUTPUT_DIR)
        config = _apply_conversion_request_config(_load_validation_config(data), data)
        scene_name = str(data.get('scene_name') or 'generated_ash_scene')
        random_scene = _parse_bool(data.get('random_ash_scene'), True)
        seed = _parse_int(data.get('seed'), int(datetime.now().timestamp()))

        if random_scene:
            config.use_random_ash_scenes = True
            config.random_scene_seed = seed
            centers_range = data.get('random_centers_range') or [1, 6]
            if isinstance(centers_range, str):
                try:
                    centers_range = json.loads(centers_range)
                except json.JSONDecodeError:
                    centers_range = [part.strip() for part in centers_range.split(',')]
            if isinstance(centers_range, (list, tuple)) and len(centers_range) >= 2:
                config.random_scene_min_centers = max(1, _parse_int(centers_range[0], 1))
                config.random_scene_max_centers = max(config.random_scene_min_centers, _parse_int(centers_range[1], 6))
            sampled_config = RandomAshSceneGenerator(config).sample_config(
                seed=seed,
                rng=np.random.default_rng(seed)
            )
            sampled_config.scene_name = scene_name
            config = sampled_config

        model = GMMVolcanicAshModel(config)
        concentration_map = model.generate_concentration_map()
        manifest = manager.save_generated_scene(
            concentration_map=concentration_map,
            config=config,
            scene_name=scene_name,
            extra={'seed': seed, 'random_scene': random_scene}
        )
        return jsonify(_to_jsonable({
            'success': True,
            'scene': _attach_file_urls(manifest),
            'message': '火山灰场景已生成并保存'
        }))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/scenes', methods=['GET'])
def list_scenes():
    from src.web.case_manager import WebCaseManager

    manager = WebCaseManager(OUTPUT_DIR)
    return jsonify(_to_jsonable({
        'success': True,
        'scenes': [_attach_file_urls(scene) for scene in manager.list_scenes()]
    }))

@app.route('/api/scenes/<scene_id>', methods=['GET'])
def get_scene(scene_id):
    from src.web.case_manager import WebCaseManager

    try:
        manager = WebCaseManager(OUTPUT_DIR)
        return jsonify(_to_jsonable({
            'success': True,
            'scene': _attach_file_urls(manager.load_scene(scene_id))
        }))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 404

@app.route('/api/cases/run', methods=['POST'])
def run_case():
    data = request.get_json(silent=True) or {}

    try:
        save_case = _parse_bool(data.get('save_case'), True)
        export_json = _parse_bool(data.get('export_json'), save_case)
        export_plot = _parse_bool(data.get('export_plot'), save_case)
        export_animation = _parse_bool(data.get('export_animation'), save_case)
        return jsonify(_execute_case_planning(
            data,
            save_case=save_case,
            export_json=export_json,
            export_plot=export_plot,
            export_animation=export_animation,
        ))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/cases/checkpoint-demo', methods=['POST'])
def checkpoint_demo():
    data = request.get_json(silent=True) or {}

    try:
        model_path = str(data.get('model_path') or 'models/final_model.pth')
        max_checkpoints = max(2, _parse_int(data.get('max_checkpoints'), 5))
        selection = _resolve_checkpoint_selection(model_path, max_checkpoints=max_checkpoints)
        if not selection:
            raise FileNotFoundError('未找到 checkpoint_ep*.pth 或 final_model.pth')

        history = _load_model_history(model_path)
        items = []
        for item in selection:
            history_snapshot = _history_snapshot_for_episode(history, item['episode'])
            checkpoint_success_rate = None
            if history_snapshot and history_snapshot.get('success_rates') is not None:
                checkpoint_success_rate = float(history_snapshot.get('success_rates'))
            planning_result = _execute_case_planning(
                data,
                save_case=False,
                export_json=False,
                export_plot=False,
                export_animation=False,
                allow_fallback=False,
                random_on_missing_agent=True,
                model_override=item['path'],
                case_name_override=f"{data.get('case_name') or 'checkpoint_demo'}_{item['label']}",
            )
            path_data = planning_result.get('path_data') or {}
            validation_info = path_data.get('validation_info') or {}
            arrival_success = bool(path_data.get('success', False))
            used_fallback = bool(validation_info.get('used_fallback', False))
            training_success = arrival_success and not used_fallback
            metrics = {
                'success': training_success,
                'arrival_success': arrival_success,
                'training_success': training_success,
                'planning_method': path_data.get('planning_method'),
                'used_fallback': used_fallback,
                'fallback_reason': validation_info.get('fallback_reason', ''),
                'demo_mode': validation_info.get('demo_mode', 'rl_policy'),
                'agent_status': validation_info.get('agent_status', ''),
                'random_mode': validation_info.get('random_mode', ''),
                'steps_taken': int(path_data.get('steps_taken', 0) or 0),
                'max_concentration': float(path_data.get('max_concentration', 0.0) or 0.0),
                'ash_exposure': float(path_data.get('ash_exposure', 0.0) or 0.0),
                'total_reward': float(path_data.get('total_reward', 0.0) or 0.0),
                'checkpoint_success_rate': checkpoint_success_rate,
            }
            items.append({
                'label': item['label'],
                'episode': item['episode'],
                'checkpoint_path': item['path'],
                'path_data': path_data,
                'metrics': metrics,
                'history_snapshot': history_snapshot,
            })

        return jsonify(_to_jsonable({
            'success': True,
            'items': items,
            'message': f'已加载 {len(items)} 个 checkpoint 演示当前场景。'
        }))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/cases', methods=['GET'])
def list_cases():
    from src.web.case_manager import WebCaseManager

    manager = WebCaseManager(OUTPUT_DIR)
    return jsonify(_to_jsonable({
        'success': True,
        'cases': [_attach_file_urls(case) for case in manager.list_cases()]
    }))

@app.route('/api/cases/<case_id>', methods=['GET'])
def get_case(case_id):
    from src.web.case_manager import WebCaseManager

    try:
        manager = WebCaseManager(OUTPUT_DIR)
        case = manager.load_case(case_id)
        return jsonify(_to_jsonable({
            'success': True,
            'case': _attach_file_urls(case)
        }))
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 404

@app.route('/api/output/<path:relative_path>', methods=['GET'])
def serve_output_file(relative_path):
    output_root = os.path.abspath(OUTPUT_DIR)
    safe_path = os.path.abspath(os.path.join(OUTPUT_DIR, relative_path))

    if not safe_path.startswith(output_root):
        return jsonify({'error': 'Invalid path'}), 400

    if not os.path.exists(safe_path):
        return jsonify({'error': 'File not found'}), 404

    relative_clean = os.path.relpath(safe_path, OUTPUT_DIR).replace('\\', '/')
    return send_from_directory(OUTPUT_DIR, relative_clean)

@app.route('/api/analyze', methods=['POST'])
def analyze_data():
    data = request.json
    
    from src.analysis.data_analyzer import DataAnalyzer
    
    try:
        analyzer = DataAnalyzer()
        
        flight_data = data.get('flight_data', {})
        if flight_data:
            analysis = analyzer.analyze_flight_data(flight_data)
        else:
            analysis = {'error': 'No flight data provided'}
        
        report = analyzer.generate_comprehensive_report([{'data': flight_data}])
        
        text_report = analyzer.format_text_report(report)
        
        analyzer.export_report_json(report, 'output/analysis_report.json')
        
        with open('output/analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(text_report)
        
        return jsonify({
            'success': True,
            'analysis': analysis,
            'report': report,
            'text_report': text_report
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/data/<filename>', methods=['GET'])
def serve_data_file(filename):
    directory = request.args.get('dir', 'output')
    safe_path = os.path.normpath(os.path.join(directory, filename))
    
    if not safe_path.startswith(directory):
        return jsonify({'error': 'Invalid path'}), 400
    
    if os.path.exists(safe_path):
        with open(safe_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return jsonify(data)
    else:
        return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    port = _parse_int(os.environ.get('PORT'), 5000)
    host = os.environ.get('HOST', '0.0.0.0')
    print("=" * 60)
    print("Volcanic Ash Avoidance System - Web Server Starting")
    print("=" * 60)
    print(f"Access URL: http://localhost:{port}")
    print("API Endpoints:")
    print("  GET  /api/presets      - Get preset configs")
    print("  POST /api/generate     - Generate images")
    print("  POST /api/train        - Train model")
    print("  POST /api/plan         - Plan path")
    print("  POST /api/multi-plan   - Multi-constraint planning")
    print("  POST /api/validate-image - Validate an image-derived ash scene")
    print("  POST /api/scenes/import-image - Import and save a real ash scene")
    print("  POST /api/scenes/generated - Generate and save an ash scene")
    print("  POST /api/cases/run    - Run, save, and replay a planning case")
    print("  POST /api/analyze      - Data analysis")
    print("=" * 60)
    
    app.run(debug=True, use_reloader=False, port=port, host=host)
