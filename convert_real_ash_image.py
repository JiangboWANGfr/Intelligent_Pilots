import argparse
import json
import os

from src.config.volcanic_ash_config import VolcanicAshConfig
from src.generation.image_converter import AshImageConverter


def parse_size(raw: str):
    parts = [part.strip() for part in raw.split(',')]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError('Expected height,width, for example 768,768.')
    height, width = int(parts[0]), int(parts[1])
    if height <= 0 or width <= 0:
        raise argparse.ArgumentTypeError('Image size values must be positive.')
    return height, width


def main():
    parser = argparse.ArgumentParser(
        description='Convert a real volcanic ash image into the standard concentration-map format.'
    )
    parser.add_argument('image', help='Input image path.')
    parser.add_argument('--config', default='output/current_config.json',
                        help='Base volcanic ash config.')
    parser.add_argument('--output-dir', default='output/real_ash_converted',
                        help='Directory for converted map outputs.')
    parser.add_argument('--scene-name', default='real_ash_image_scene')
    parser.add_argument('--image-size', type=parse_size, default=None,
                        help='Output map size as height,width, e.g. 768,768.')
    parser.add_argument('--mode', choices=['auto', 'color', 'grayscale'], default='auto',
                        help='Conversion mode. Use color for weather-map style images.')
    parser.add_argument('--invert', default='auto',
                        help='Grayscale inversion: auto, true, or false.')
    parser.add_argument('--blur-kernel', type=int, default=5)
    args = parser.parse_args()

    if os.path.exists(args.config):
        config = VolcanicAshConfig.load(args.config)
    else:
        config = VolcanicAshConfig()
    if args.image_size is not None:
        config.image_size = args.image_size

    converter = AshImageConverter(config)
    converted = converter.convert_to_scene(
        args.image,
        scene_name=args.scene_name,
        mode=args.mode,
        invert=args.invert,
        blur_kernel=args.blur_kernel
    )
    os.makedirs(args.output_dir, exist_ok=True)
    outputs = converter.save_standard_outputs(
        converted['concentration_map'],
        args.output_dir,
        prefix='real_ash'
    )
    config_path = os.path.join(args.output_dir, 'real_ash_config.json')
    summary_path = os.path.join(args.output_dir, 'real_ash_summary.json')
    converted['config'].save(config_path)
    summary = {
        'input_image': args.image,
        'scene_name': args.scene_name,
        'conversion_mode': args.mode,
        'image_size': list(converted['concentration_map'].shape),
        'outputs': outputs,
        'config_path': config_path,
        'summary': converted['summary']
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print('Converted real ash image.')
    print(f'  Preview: {outputs["preview_path"]}')
    print(f'  Concentration PNG: {outputs["grayscale_path"]}')
    print(f'  Concentration NPY: {outputs["npy_path"]}')
    print(f'  Config: {config_path}')
    print(f'  Summary: {summary_path}')


if __name__ == '__main__':
    main()
