import heapq
from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import maximum_filter

from src.config.volcanic_ash_config import VolcanicAshConfig


class FallbackPlanner:
    def __init__(self, config: VolcanicAshConfig, grid_stride: Optional[int] = None):
        self.config = config
        self.grid_stride = grid_stride

    def plan(self,
             concentration_map: np.ndarray,
             start_pos: Tuple[float, float],
             target_pos: Tuple[float, float],
             max_concentration: Optional[float] = None,
             risk_inflation_radius: Optional[float] = None,
             boundary_margin: Optional[float] = None,
             desired_points: int = 160) -> List[Tuple[float, float]]:
        map_array = np.asarray(concentration_map, dtype=np.float32)
        if map_array.ndim != 2:
            raise ValueError('concentration_map must be 2D')

        threshold = float(max_concentration if max_concentration is not None
                          else self.config.concentration_threshold)
        stride = self.grid_stride or max(4, min(map_array.shape) // 96)
        coarse_map = self._build_coarse_map(map_array, stride)
        planning_map = self._inflate_risk_map(map_array, risk_inflation_radius)
        clearance_map = self._build_coarse_map(planning_map, stride)
        boundary_margin_cells = float(boundary_margin or 0.0) / max(stride, 1)
        start_node = self._to_grid(start_pos, stride, coarse_map.shape)
        target_node = self._to_grid(target_pos, stride, coarse_map.shape)

        coarse_path = self._astar(
            coarse_map,
            clearance_map,
            start_node,
            target_node,
            threshold,
            boundary_margin_cells
        )
        if not coarse_path:
            return self._smooth_path(
                self._interpolate_path([start_pos, target_pos], desired_points, map_array.shape),
                map_array.shape
            )

        fine_path: List[Tuple[float, float]] = [tuple(map(float, start_pos))]
        for node in coarse_path[1:-1]:
            fine_path.append(self._to_pixel(node, stride, map_array.shape))
        fine_path.append(tuple(map(float, target_pos)))

        interpolated = self._interpolate_path(fine_path, desired_points, map_array.shape)
        return self._smooth_path(interpolated, map_array.shape)

    def _inflate_risk_map(self,
                          concentration_map: np.ndarray,
                          risk_inflation_radius: Optional[float]) -> np.ndarray:
        radius = int(round(float(risk_inflation_radius or 0.0)))
        if radius <= 0:
            return concentration_map

        kernel_size = max(1, 2 * radius + 1)
        return maximum_filter(concentration_map, size=kernel_size, mode='nearest')

    def summarize_path(self,
                       concentration_map: np.ndarray,
                       path_points: List[Tuple[float, float]]) -> dict:
        map_array = np.asarray(concentration_map, dtype=np.float32)
        if not path_points:
            return {'max_concentration': 0.0, 'mean_concentration': 0.0, 'path_length': 0.0}

        sampled = []
        path_length = 0.0
        previous = None
        for point in path_points:
            y = int(np.clip(round(point[0]), 0, map_array.shape[0] - 1))
            x = int(np.clip(round(point[1]), 0, map_array.shape[1] - 1))
            sampled.append(float(map_array[y, x]))
            if previous is not None:
                path_length += float(np.linalg.norm(np.array(point) - np.array(previous)))
            previous = point

        return {
            'max_concentration': float(np.max(sampled)),
            'mean_concentration': float(np.mean(sampled)),
            'path_length': float(path_length)
        }

    def _build_coarse_map(self, concentration_map: np.ndarray, stride: int) -> np.ndarray:
        height, width = concentration_map.shape
        coarse_height = int(np.ceil(height / stride))
        coarse_width = int(np.ceil(width / stride))
        coarse_map = np.zeros((coarse_height, coarse_width), dtype=np.float32)

        for gy in range(coarse_height):
            for gx in range(coarse_width):
                y0 = gy * stride
                y1 = min(height, (gy + 1) * stride)
                x0 = gx * stride
                x1 = min(width, (gx + 1) * stride)
                coarse_map[gy, gx] = float(np.max(concentration_map[y0:y1, x0:x1]))

        return coarse_map

    def _astar(self,
               coarse_map: np.ndarray,
               clearance_map: np.ndarray,
               start_node: Tuple[int, int],
               target_node: Tuple[int, int],
               threshold: float,
               boundary_margin_cells: float = 0.0) -> List[Tuple[int, int]]:
        neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

        frontier = [(0.0, start_node)]
        came_from = {start_node: None}
        cost_so_far = {start_node: 0.0}

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == target_node:
                break

            for dy, dx in neighbors:
                next_node = (current[0] + dy, current[1] + dx)
                if not (0 <= next_node[0] < coarse_map.shape[0] and 0 <= next_node[1] < coarse_map.shape[1]):
                    continue

                risk = float(coarse_map[next_node])
                clearance_risk = max(risk, float(clearance_map[next_node]))
                movement_cost = float(np.hypot(dy, dx))
                safe_threshold = max(threshold, 1e-6)
                risk_ratio = risk / safe_threshold
                clearance_ratio = clearance_risk / safe_threshold
                excess_ratio = max(0.0, risk - threshold) / safe_threshold
                clearance_excess_ratio = max(0.0, clearance_risk - threshold) / safe_threshold
                safety_factor = max(float(getattr(
                    self.config,
                    'fallback_cost_safety_factor',
                    getattr(self.config, 'fixed_safety_factor', 1.0)
                )), 0.02)
                risk_penalty = (
                    1.0
                    + safety_factor * (
                        6.0 * risk
                        + 3.0 * risk_ratio ** 2
                        + 2.0 * clearance_ratio ** 2
                    )
                )
                if clearance_risk > threshold:
                    risk_penalty += safety_factor * (
                        20.0 * clearance_excess_ratio
                        + 60.0 * clearance_excess_ratio ** 2
                    )
                if risk > threshold:
                    risk_penalty += safety_factor * (
                        120.0 * excess_ratio + 600.0 * excess_ratio ** 2
                    )
                if risk > threshold * 1.5:
                    risk_penalty += safety_factor * 2000.0
                if risk > threshold * 2.0:
                    risk_penalty += safety_factor * 8000.0
                if risk > 0.95:
                    risk_penalty += safety_factor * 20000.0
                if boundary_margin_cells > 0.0:
                    edge_distance = min(
                        next_node[0],
                        next_node[1],
                        coarse_map.shape[0] - 1 - next_node[0],
                        coarse_map.shape[1] - 1 - next_node[1]
                    )
                    if edge_distance < boundary_margin_cells:
                        edge_ratio = (boundary_margin_cells - edge_distance) / boundary_margin_cells
                        risk_penalty += 20.0 * edge_ratio + 80.0 * edge_ratio ** 2

                new_cost = cost_so_far[current] + movement_cost * risk_penalty
                if next_node not in cost_so_far or new_cost < cost_so_far[next_node]:
                    cost_so_far[next_node] = new_cost
                    priority = new_cost + self._heuristic(next_node, target_node)
                    heapq.heappush(frontier, (priority, next_node))
                    came_from[next_node] = current

        if target_node not in came_from:
            return []

        path = []
        current = target_node
        while current is not None:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path

    def _heuristic(self, current: Tuple[int, int], target: Tuple[int, int]) -> float:
        return float(np.hypot(target[0] - current[0], target[1] - current[1]))

    def _to_grid(self,
                 position: Tuple[float, float],
                 stride: int,
                 grid_shape: Tuple[int, int]) -> Tuple[int, int]:
        y = int(np.clip(round(position[0] / stride), 0, grid_shape[0] - 1))
        x = int(np.clip(round(position[1] / stride), 0, grid_shape[1] - 1))
        return y, x

    def _to_pixel(self,
                  node: Tuple[int, int],
                  stride: int,
                  map_shape: Tuple[int, int]) -> Tuple[float, float]:
        y = float(np.clip(node[0] * stride + stride / 2, 0, map_shape[0] - 1))
        x = float(np.clip(node[1] * stride + stride / 2, 0, map_shape[1] - 1))
        return y, x

    def _interpolate_path(self,
                          path_points: List[Tuple[float, float]],
                          desired_points: int,
                          map_shape: Tuple[int, int]) -> List[Tuple[float, float]]:
        if len(path_points) <= 1:
            return [self._clip_point(path_points[0], map_shape)] if path_points else []

        points = np.asarray(path_points, dtype=np.float32)
        segment_lengths = np.linalg.norm(points[1:] - points[:-1], axis=1)
        cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
        total_length = float(cumulative[-1])
        if total_length <= 1e-6:
            return [self._clip_point(tuple(point), map_shape) for point in points]

        samples = np.linspace(0.0, total_length, max(desired_points, len(path_points)))
        interpolated: List[Tuple[float, float]] = []
        for sample in samples:
            index = int(np.searchsorted(cumulative, sample, side='right') - 1)
            index = min(index, len(points) - 2)
            local_length = cumulative[index + 1] - cumulative[index]
            ratio = 0.0 if local_length <= 1e-6 else (sample - cumulative[index]) / local_length
            point = points[index] * (1.0 - ratio) + points[index + 1] * ratio
            interpolated.append(self._clip_point(tuple(point.tolist()), map_shape))

        interpolated[0] = self._clip_point(tuple(points[0].tolist()), map_shape)
        interpolated[-1] = self._clip_point(tuple(points[-1].tolist()), map_shape)
        return interpolated

    def _smooth_path(self,
                     path_points: List[Tuple[float, float]],
                     map_shape: Tuple[int, int],
                     window_size: int = 5) -> List[Tuple[float, float]]:
        if len(path_points) <= 2:
            return [self._clip_point(point, map_shape) for point in path_points]

        points = np.asarray(path_points, dtype=np.float32)
        smoothed = points.copy()
        half_window = window_size // 2
        for index in range(1, len(points) - 1):
            start = max(0, index - half_window)
            end = min(len(points), index + half_window + 1)
            smoothed[index] = np.mean(points[start:end], axis=0)

        smoothed[0] = points[0]
        smoothed[-1] = points[-1]
        return [self._clip_point(tuple(point.tolist()), map_shape) for point in smoothed]

    def _clip_point(self,
                    point: Tuple[float, float],
                    map_shape: Tuple[int, int]) -> Tuple[float, float]:
        y = float(np.clip(point[0], 0, map_shape[0] - 1))
        x = float(np.clip(point[1], 0, map_shape[1] - 1))
        return y, x
