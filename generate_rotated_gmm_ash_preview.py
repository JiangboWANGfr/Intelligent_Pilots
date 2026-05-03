import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config.volcanic_ash_config import VolcanicAshConfig
from src.model.gmm_model import GMMVolcanicAshModel


def parse_pair(raw: str, value_type=float) -> Tuple:
    parts = [part.strip() for part in raw.split(',')]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError('Expected two comma-separated values.')
    return value_type(parts[0]), value_type(parts[1])


def sample_rotated_centers(args, rng: np.random.Generator) -> List[Dict]:
    min_centers, max_centers = args.center_count_range
    num_centers = int(rng.integers(min_centers, max_centers + 1))
    height, width = args.image_size
    margin = float(args.position_margin)
    std_min, std_max = args.std_range
    anisotropy_min, anisotropy_max = args.anisotropy_range

    raw_weights = rng.uniform(args.weight_range[0], args.weight_range[1], size=num_centers)
    normalized_weights = raw_weights / max(float(np.sum(raw_weights)), 1e-6)

    centers = []
    for index in range(num_centers):
        base_std = float(rng.uniform(std_min, std_max))
        anisotropy = float(rng.uniform(anisotropy_min, anisotropy_max))
        if rng.random() < 0.5:
            std_x = base_std * anisotropy
            std_y = base_std
        else:
            std_x = base_std
            std_y = base_std * anisotropy

        centers.append({
            'x': float(rng.uniform(margin, width - margin)),
            'y': float(rng.uniform(margin, height - margin)),
            'weight': float(normalized_weights[index]),
            'std_x': float(std_x),
            'std_y': float(std_y),
            'theta': float(rng.uniform(0.0, np.pi))
        })

    return centers


def build_preview_image(concentration_map: np.ndarray,
                        config: VolcanicAshConfig,
                        centers: List[Dict]) -> np.ndarray:
    height, width = concentration_map.shape
    preview = np.zeros((height, width, 3), dtype=np.uint8)
    preview[:] = (16, 23, 25)

    low_mask = concentration_map >= config.concentration_threshold * 0.45
    medium_mask = concentration_map >= config.concentration_threshold
    high_mask = concentration_map >= config.concentration_threshold * 1.45
    extreme_mask = concentration_map >= config.concentration_threshold * 2.0

    preview[low_mask] = (191, 112, 25)
    preview[medium_mask] = (228, 69, 31)
    preview[high_mask] = (207, 16, 79)
    preview[extreme_mask] = (147, 0, 92)

    for step in range(0, width, 16):
        cv2.line(preview, (step, 0), (step, height - 1), (35, 45, 47), 1)
    for step in range(0, height, 16):
        cv2.line(preview, (0, step), (width - 1, step), (35, 45, 47), 1)
    for step in range(0, width, 64):
        cv2.line(preview, (step, 0), (step, height - 1), (45, 58, 60), 1)
    for step in range(0, height, 64):
        cv2.line(preview, (0, step), (width - 1, step), (45, 58, 60), 1)

    for index, center in enumerate(centers, start=1):
        x = int(round(center['x']))
        y = int(round(center['y']))
        cv2.circle(preview, (x, y), 8, (255, 185, 50), -1, lineType=cv2.LINE_AA)
        cv2.circle(preview, (x, y), 9, (90, 45, 20), 2, lineType=cv2.LINE_AA)
        cv2.line(preview, (x - 8, y), (x + 8, y), (90, 45, 20), 2, lineType=cv2.LINE_AA)
        cv2.line(preview, (x, y - 8), (x, y + 8), (90, 45, 20), 2, lineType=cv2.LINE_AA)
        cv2.putText(
            preview,
            f'C{index}',
            (x + 11, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (245, 245, 245),
            1,
            cv2.LINE_AA
        )

    return preview


def summarize_area(concentration_map: np.ndarray, config: VolcanicAshConfig) -> Dict[str, float]:
    return {
        'low_risk_area_percent': float(
            np.mean(concentration_map >= config.concentration_threshold * 0.45) * 100.0
        ),
        'medium_risk_area_percent': float(
            np.mean(concentration_map >= config.concentration_threshold) * 100.0
        ),
        'high_risk_area_percent': float(
            np.mean(concentration_map >= config.concentration_threshold * 1.45) * 100.0
        )
    }


def generate_valid_case(case_index: int,
                        args,
                        rng: np.random.Generator) -> Tuple[List[Dict], VolcanicAshConfig, np.ndarray]:
    last_case = None
    for _ in range(args.max_attempts):
        centers = sample_rotated_centers(args, rng)
        config = VolcanicAshConfig(
            model_type='rotated_random_gmm',
            num_centers=len(centers),
            cloud_size=args.cloud_size,
            concentration_threshold=args.threshold,
            mass_ratio=1.0,
            image_size=args.image_size,
            centers=centers,
            enable_irregular=False,
            random_seed=args.seed + case_index
        )
        model = GMMVolcanicAshModel(config)
        concentration_map = model.generate_concentration_map(irregular=False)
        area = summarize_area(concentration_map, config)
        last_case = (centers, config, concentration_map)

        if (
            args.min_medium_area <= area['medium_risk_area_percent'] <= args.max_medium_area
            and area['low_risk_area_percent'] <= args.max_low_area
            and area['high_risk_area_percent'] <= args.max_high_area
        ):
            return centers, config, concentration_map

    return last_case


def save_case(case_index: int,
              args,
              rng: np.random.Generator,
              output_dir: str) -> Dict:
    centers, config, concentration_map = generate_valid_case(case_index, args, rng)
    model = GMMVolcanicAshModel(config)
    area = summarize_area(concentration_map, config)

    prefix = f'rotated_gmm_{case_index:03d}'
    concentration_path = os.path.join(output_dir, f'{prefix}_concentration.png')
    danger_path = os.path.join(output_dir, f'{prefix}_danger.png')
    preview_path = os.path.join(output_dir, f'{prefix}_preview.png')
    config_path = os.path.join(output_dir, f'{prefix}_config.json')

    grayscale = model.generate_grayscale_image(concentration_map)
    danger_rgb = model.generate_danger_zone_image(concentration_map)
    preview_rgb = build_preview_image(concentration_map, config, centers)

    cv2.imwrite(concentration_path, grayscale)
    cv2.imwrite(danger_path, cv2.cvtColor(danger_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(preview_path, cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR))
    config.save(config_path)

    return {
        'case': case_index,
        'num_centers': len(centers),
        'concentration_path': concentration_path,
        'danger_path': danger_path,
        'preview_path': preview_path,
        'config_path': config_path,
        'max_concentration': float(np.max(concentration_map)),
        'mean_concentration': float(np.mean(concentration_map)),
        **area,
        'centers': centers
    }


def main():
    parser = argparse.ArgumentParser(
        description='Generate random rotated-GMM volcanic ash preview images without training.'
    )
    parser.add_argument('--output-dir', default='output/rotated_gmm_ash_preview')
    parser.add_argument('--samples', type=int, default=6)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--image-size', type=lambda value: parse_pair(value, int),
                        default=(512, 512),
                        help='Image size as height,width.')
    parser.add_argument('--center-count-range', type=lambda value: parse_pair(value, int),
                        default=(2, 4),
                        help='Random number of Gaussian centers as min,max.')
    parser.add_argument('--position-margin', type=float, default=90.0)
    parser.add_argument('--weight-range', type=lambda value: parse_pair(value, float),
                        default=(0.3, 1.0))
    parser.add_argument('--std-range', type=lambda value: parse_pair(value, float),
                        default=(22.0, 55.0))
    parser.add_argument('--anisotropy-range', type=lambda value: parse_pair(value, float),
                        default=(1.5, 3.2),
                        help='Axis ratio range. Larger values make longer ellipses.')
    parser.add_argument('--cloud-size', type=float, default=75.0)
    parser.add_argument('--threshold', type=float, default=0.3)
    parser.add_argument('--min-medium-area', type=float, default=4.0,
                        help='Minimum percent of pixels above threshold.')
    parser.add_argument('--max-medium-area', type=float, default=22.0,
                        help='Maximum percent of pixels above threshold.')
    parser.add_argument('--max-low-area', type=float, default=45.0,
                        help='Maximum percent of pixels in the visible low-risk cloud.')
    parser.add_argument('--max-high-area', type=float, default=9.0,
                        help='Maximum percent of pixels in the high-risk core.')
    parser.add_argument('--max-attempts', type=int, default=200,
                        help='Rejection-sampling attempts per preview case.')
    args = parser.parse_args()

    if args.center_count_range[0] < 1 or args.center_count_range[0] > args.center_count_range[1]:
        raise ValueError('--center-count-range must be valid and start at >= 1')

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    results = [
        save_case(index + 1, args, rng, args.output_dir)
        for index in range(args.samples)
    ]

    summary_path = os.path.join(args.output_dir, 'summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            'seed': args.seed,
            'samples': args.samples,
            'results': results
        }, f, ensure_ascii=False, indent=2)

    print(f'Generated {len(results)} rotated-GMM ash preview cases.')
    print(f'Output directory: {args.output_dir}')
    print(f'Summary: {summary_path}')


if __name__ == '__main__':
    main()
