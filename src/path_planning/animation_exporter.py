import os
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.config.volcanic_ash_config import VolcanicAshConfig


class ValidationAnimationExporter:
    def __init__(self, config: VolcanicAshConfig, concentration_map: np.ndarray):
        self.config = VolcanicAshConfig.from_dict(config.to_dict())
        self.concentration_map = np.asarray(concentration_map, dtype=np.float32)
        if self.concentration_map.ndim != 2:
            raise ValueError('concentration_map must be a 2D array')
        self.concentration_map = np.clip(self.concentration_map, 0.0, 1.0)
        self.background_rgb = (245, 247, 250)
        self.level_colors_rgb = {
            'trace': (0, 255, 255),
            'aircraft': (0, 255, 255),
            'start': (0, 200, 83),
            'target': (229, 57, 53),
            'safe': (0, 255, 0),
            'low': (255, 255, 0),
            'medium': (255, 165, 0),
            'high': (255, 0, 0)
        }
        self.font_path = self._find_font_path()

    def export(self,
               path_result: Dict,
               output_dir: str = 'output/validation_animation',
               gif_path: Optional[str] = None,
               video_path: Optional[str] = None,
               fps: int = 12,
               save_frames: bool = False,
               max_frames: int = 180,
               hold_last_frames: int = 10) -> Dict:
        waypoints = path_result.get('waypoints', [])
        if not waypoints:
            raise ValueError('path_result contains no waypoints to animate')

        os.makedirs(output_dir, exist_ok=True)
        scene_name = str(path_result.get('scene_name', self.config.scene_name or self.config.model_type or 'validation_animation'))
        base_name = self._sanitize_filename(scene_name)
        if gif_path is None:
            gif_path = os.path.join(output_dir, f'{base_name}_avoidance.gif')
        if video_path is None:
            video_path = os.path.join(output_dir, f'{base_name}_avoidance.mp4')

        render_scale = self._compute_render_scale()
        map_rgb = self._build_colored_map(render_scale)
        panel_width = max(300, int(map_rgb.shape[1] * 0.34))
        frame_size = (map_rgb.shape[1] + panel_width, map_rgb.shape[0])
        frame_indices = self._build_frame_indices(len(waypoints), max_frames=max_frames)
        if hold_last_frames > 0:
            frame_indices.extend([len(waypoints) - 1] * hold_last_frames)

        frames_dir = None
        if save_frames:
            frames_dir = os.path.join(output_dir, f'{base_name}_frames')
            os.makedirs(frames_dir, exist_ok=True)

        gif_frames: List[Image.Image] = []
        video_writer = None
        actual_video_path = None
        video_codec = None
        if video_path:
            video_writer, actual_video_path, video_codec = self._create_video_writer(video_path, frame_size, fps)

        for frame_number, waypoint_index in enumerate(frame_indices):
            frame_rgb = self._render_frame(
                map_rgb=map_rgb,
                panel_width=panel_width,
                path_result=path_result,
                waypoint_index=waypoint_index,
                total_waypoints=len(waypoints)
            )
            if save_frames and frames_dir:
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(frames_dir, f'frame_{frame_number:04d}.png'), frame_bgr)
            if gif_path:
                gif_frames.append(Image.fromarray(frame_rgb))
            if video_writer is not None:
                video_writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

        if video_writer is not None:
            video_writer.release()

        if gif_frames:
            os.makedirs(os.path.dirname(gif_path) or '.', exist_ok=True)
            gif_frames[0].save(
                gif_path,
                save_all=True,
                append_images=gif_frames[1:],
                duration=max(1, int(round(1000 / max(fps, 1)))),
                loop=0,
                optimize=False
            )

        return {
            'scene_name': scene_name,
            'planning_method': path_result.get('planning_method', 'rl'),
            'fps': int(fps),
            'frame_count': len(frame_indices),
            'sampled_waypoint_count': len({index for index in frame_indices}),
            'resolution': [int(frame_size[0]), int(frame_size[1])],
            'gif_path': gif_path if gif_frames else None,
            'video_path': actual_video_path,
            'video_codec': video_codec,
            'frames_dir': frames_dir,
            'output_dir': output_dir
        }

    def _compute_render_scale(self) -> int:
        height, width = self.concentration_map.shape
        longest_side = max(height, width)
        return max(2, min(6, int(np.ceil(720 / max(longest_side, 1)))))

    def _build_colored_map(self, render_scale: int) -> np.ndarray:
        map_array = self.concentration_map
        rgb = np.full((map_array.shape[0], map_array.shape[1], 3), self.background_rgb, dtype=np.uint8)
        threshold = float(getattr(self.config, 'concentration_threshold', 0.3))
        safe_mask = (map_array >= threshold * 0.15) & (map_array < threshold * 0.5)
        low_mask = (map_array >= threshold * 0.5) & (map_array < threshold)
        medium_mask = (map_array >= threshold) & (map_array < threshold * 1.5)
        high_mask = map_array >= threshold * 1.5
        rgb[safe_mask] = self.level_colors_rgb['safe']
        rgb[low_mask] = self.level_colors_rgb['low']
        rgb[medium_mask] = self.level_colors_rgb['medium']
        rgb[high_mask] = self.level_colors_rgb['high']
        scaled = cv2.resize(
            rgb,
            (rgb.shape[1] * render_scale, rgb.shape[0] * render_scale),
            interpolation=cv2.INTER_NEAREST
        )
        return scaled

    def _render_frame(self,
                      map_rgb: np.ndarray,
                      panel_width: int,
                      path_result: Dict,
                      waypoint_index: int,
                      total_waypoints: int) -> np.ndarray:
        map_canvas = np.array(map_rgb, copy=True)
        canvas = np.full((map_canvas.shape[0], map_canvas.shape[1] + panel_width, 3), 22, dtype=np.uint8)
        canvas[:, :map_canvas.shape[1], :] = map_canvas

        path_coordinates = path_result.get('path_coordinates', [])
        scale_x = map_canvas.shape[1] / max(self.concentration_map.shape[1], 1)
        scale_y = map_canvas.shape[0] / max(self.concentration_map.shape[0], 1)
        visible_points = [
            self._canvas_point(point, scale_x, scale_y)
            for point in path_coordinates[:waypoint_index + 1]
        ]
        full_points = [self._canvas_point(point, scale_x, scale_y) for point in path_coordinates]

        if full_points:
            self._draw_marker(canvas, full_points[0], self.level_colors_rgb['start'], 'START')
            self._draw_target_marker(canvas, full_points[-1], self.level_colors_rgb['target'])

        if len(visible_points) >= 2:
            polyline = np.array(visible_points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [polyline], False, self.level_colors_rgb['trace'], thickness=4, lineType=cv2.LINE_AA)

        if visible_points:
            self._draw_aircraft(canvas, visible_points, waypoint_index)

        current_waypoint = path_result['waypoints'][min(waypoint_index, total_waypoints - 1)]
        progress = 100.0 * min(waypoint_index + 1, total_waypoints) / max(total_waypoints, 1)
        self._draw_map_frame(canvas, map_canvas.shape[1], map_canvas.shape[0])
        self._draw_progress_bar(canvas, map_canvas.shape[1], map_canvas.shape[0], progress)
        self._draw_side_panel(
            canvas=canvas,
            map_width=map_canvas.shape[1],
            panel_width=panel_width,
            path_result=path_result,
            current_waypoint=current_waypoint,
            waypoint_index=waypoint_index,
            total_waypoints=total_waypoints,
            progress=progress
        )
        return canvas

    def _draw_map_frame(self, canvas: np.ndarray, map_width: int, map_height: int):
        cv2.rectangle(canvas, (0, 0), (map_width - 1, map_height - 1), (50, 50, 50), thickness=2)

    def _draw_progress_bar(self, canvas: np.ndarray, map_width: int, map_height: int, progress: float):
        bar_left = 18
        bar_right = map_width - 18
        bar_top = map_height - 26
        bar_bottom = map_height - 10
        cv2.rectangle(canvas, (bar_left, bar_top), (bar_right, bar_bottom), (35, 35, 35), thickness=-1)
        filled_right = int(round(bar_left + (bar_right - bar_left) * np.clip(progress / 100.0, 0.0, 1.0)))
        cv2.rectangle(canvas, (bar_left, bar_top), (filled_right, bar_bottom), self.level_colors_rgb['trace'], thickness=-1)
        cv2.rectangle(canvas, (bar_left, bar_top), (bar_right, bar_bottom), (220, 220, 220), thickness=1)

    def _draw_side_panel(self,
                         canvas: np.ndarray,
                         map_width: int,
                         panel_width: int,
                         path_result: Dict,
                         current_waypoint: Dict,
                         waypoint_index: int,
                         total_waypoints: int,
                         progress: float):
        panel_left = map_width
        cv2.rectangle(canvas, (panel_left, 0), (panel_left + panel_width, canvas.shape[0]), (20, 20, 24), thickness=-1)

        info = path_result.get('validation_info', {})
        text_color = (240, 240, 240)
        sub_color = (184, 184, 184)
        accent_color = self.level_colors_rgb['trace']
        x = panel_left + 18
        y = 34
        self._draw_text(canvas, 'Validation Animation', (x, y - 20), font_size=25, color=accent_color)
        y += 30
        scene_name = str(path_result.get('scene_name', self.config.scene_name or self.config.model_type))
        method = str(path_result.get('planning_method', 'rl')).upper()
        fallback_reason = str(info.get('fallback_reason', '')) or 'none'

        lines = [
            ('Scene', scene_name),
            ('Method', method),
            ('Step', f'{min(waypoint_index + 1, total_waypoints)}/{total_waypoints}'),
            ('Progress', f'{progress:.1f}%'),
            ('Current Conc', f"{float(current_waypoint.get('concentration', 0.0)):.4f}"),
            ('Max Conc', f"{float(path_result.get('max_concentration', 0.0)):.4f}"),
            ('Total Fuel', f"{float(path_result.get('total_fuel', 0.0)):.2f}"),
            ('Success', 'YES' if path_result.get('success') else 'NO'),
            ('Fallback', 'YES' if info.get('used_fallback') else 'NO'),
            ('Fallback Reason', fallback_reason)
        ]

        for label, value in lines:
            self._draw_text(canvas, label, (x, y - 14), font_size=15, color=sub_color)
            y += 18
            display_value = str(value)
            wrapped = self._wrap_text(display_value, width=25)
            for part in wrapped:
                self._draw_text(canvas, part, (x, y - 16), font_size=17, color=text_color)
                y += 20
            y += 10

        self._draw_text(canvas, 'Legend', (x, y - 18), font_size=19, color=accent_color)
        y += 24
        threshold = float(getattr(self.config, 'concentration_threshold', 0.3))
        legend_items = [
            (f'Safe {threshold * 0.15:.2f}-{threshold * 0.5:.2f}', self.level_colors_rgb['safe']),
            (f'Low {threshold * 0.5:.2f}-{threshold:.2f}', self.level_colors_rgb['low']),
            (f'Medium {threshold:.2f}-{threshold * 1.5:.2f}', self.level_colors_rgb['medium']),
            (f'High >= {threshold * 1.5:.2f}', self.level_colors_rgb['high']),
            ('Flight Trace', self.level_colors_rgb['trace'])
        ]
        for text, color in legend_items:
            cv2.rectangle(canvas, (x, y - 12), (x + 18, y + 6), color, thickness=-1)
            cv2.rectangle(canvas, (x, y - 12), (x + 18, y + 6), (230, 230, 230), thickness=1)
            self._draw_text(canvas, text, (x + 28, y - 13), font_size=15, color=text_color)
            y += 26

    def _draw_marker(self, canvas: np.ndarray, point: Tuple[int, int], color_rgb: Tuple[int, int, int], label: str):
        cv2.circle(canvas, point, 8, color_rgb, thickness=-1)
        cv2.circle(canvas, point, 11, (255, 255, 255), thickness=2)
        self._draw_label_near_point(canvas, label, point, color=(25, 25, 25))

    def _draw_target_marker(self, canvas: np.ndarray, point: Tuple[int, int], color_rgb: Tuple[int, int, int]):
        cv2.circle(canvas, point, 13, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, point, 9, color_rgb, thickness=3, lineType=cv2.LINE_AA)
        cv2.drawMarker(canvas, point, color_rgb, markerType=cv2.MARKER_TILTED_CROSS,
                       markerSize=24, thickness=4, line_type=cv2.LINE_AA)
        self._draw_label_near_point(canvas, 'TARGET', point, color=(25, 25, 25))

    def _draw_aircraft(self, canvas: np.ndarray, visible_points: Sequence[Tuple[int, int]], waypoint_index: int):
        current = np.array(visible_points[-1], dtype=np.float32)
        if len(visible_points) >= 2:
            previous = np.array(visible_points[-2], dtype=np.float32)
        else:
            previous = current + np.array([-1.0, 0.0], dtype=np.float32)
        direction = current - previous
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            direction = np.array([1.0, 0.0], dtype=np.float32)
            norm = 1.0
        direction = direction / norm
        normal = np.array([-direction[1], direction[0]], dtype=np.float32)
        tip = current + direction * 14.0
        rear_center = current - direction * 10.0
        left = rear_center + normal * 8.0
        right = rear_center - normal * 8.0
        shape = np.array([tip, left, right], dtype=np.int32)
        cv2.fillConvexPoly(canvas, shape, self.level_colors_rgb['aircraft'], lineType=cv2.LINE_AA)
        cv2.polylines(canvas, [shape.reshape((-1, 1, 2))], True, (255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
        cv2.circle(canvas, tuple(np.round(current).astype(int)), 3, (255, 255, 255), thickness=-1)

    def _canvas_point(self, point: Sequence[float], scale_x: float, scale_y: float) -> Tuple[int, int]:
        px = float(point[0])
        py = float(point[1])
        return int(round((px + 0.5) * scale_x - 0.5)), int(round((py + 0.5) * scale_y - 0.5))

    def _build_frame_indices(self, total_waypoints: int, max_frames: int) -> List[int]:
        if total_waypoints <= 0:
            return [0]
        stride = max(1, int(np.ceil(total_waypoints / max(max_frames, 1))))
        indices = list(range(0, total_waypoints, stride))
        if indices[-1] != total_waypoints - 1:
            indices.append(total_waypoints - 1)
        return indices

    def _create_video_writer(self,
                             video_path: str,
                             frame_size: Tuple[int, int],
                             fps: int) -> Tuple[Optional[cv2.VideoWriter], Optional[str], Optional[str]]:
        os.makedirs(os.path.dirname(video_path) or '.', exist_ok=True)
        root, ext = os.path.splitext(video_path)
        ext = ext.lower()
        candidates: List[Tuple[str, str]] = []
        if ext == '.avi':
            candidates.extend([('MJPG', video_path), ('XVID', video_path)])
        else:
            candidates.extend([('mp4v', video_path), ('avc1', video_path), ('MJPG', root + '.avi')])
        for codec, candidate_path in candidates:
            writer = cv2.VideoWriter(
                candidate_path,
                cv2.VideoWriter_fourcc(*codec),
                float(max(fps, 1)),
                (int(frame_size[0]), int(frame_size[1]))
            )
            if writer.isOpened():
                return writer, candidate_path, codec
            writer.release()
        return None, None, None

    def _sanitize_filename(self, value: str) -> str:
        sanitized = ''.join(char if char.isalnum() or char in ('-', '_') else '_' for char in value.strip())
        sanitized = sanitized.strip('_')
        return sanitized or 'validation_animation'

    def _wrap_text(self, value: str, width: int) -> List[str]:
        if len(value) <= width:
            return [value]
        wrapped = []
        start = 0
        while start < len(value):
            wrapped.append(value[start:start + width])
            start += width
        return wrapped

    def _find_font_path(self) -> Optional[str]:
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

    def _get_font(self, size: int):
        if self.font_path:
            try:
                return ImageFont.truetype(self.font_path, size=size)
            except OSError:
                pass
        return ImageFont.load_default()

    def _draw_text(self,
                   canvas: np.ndarray,
                   text: str,
                   position: Tuple[int, int],
                   font_size: int,
                   color: Tuple[int, int, int]):
        image = Image.fromarray(canvas)
        draw = ImageDraw.Draw(image)
        draw.text(position, text, font=self._get_font(font_size), fill=tuple(color))
        canvas[:] = np.asarray(image)

    def _draw_label_near_point(self,
                               canvas: np.ndarray,
                               label: str,
                               point: Tuple[int, int],
                               color: Tuple[int, int, int]):
        x = min(max(point[0] + 12, 4), canvas.shape[1] - 90)
        y = min(max(point[1] - 24, 4), canvas.shape[0] - 24)
        self._draw_text(canvas, label, (x, y), font_size=16, color=color)
