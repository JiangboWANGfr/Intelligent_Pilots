import argparse
import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

os.makedirs('output/mpl_cache', exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', 'output/mpl_cache')

from src.config.volcanic_ash_config import VolcanicAshConfig, get_preset_configs, get_training_scene_configs
from src.rl_training.trainer import Trainer


CURRICULUM_STAGES: List[Dict] = [
    {
        'name': 'stage1_single_regular',
        'episodes': 3000,
        'scenes': ['单中心模型_规则']
    },
    {
        'name': 'stage2_single_irregular',
        'episodes': 3000,
        'scenes': ['单中心模型_规则', '单中心_轻度扰动']
    },
    {
        'name': 'stage3_double_triple',
        'episodes': 5000,
        'scenes': [
            '单中心模型_规则',
            '单中心_轻度扰动',
            '双中心_复杂扩散',
            '三中心_多细丝'
        ]
    },
    {
        'name': 'stage4_all_scenes',
        'episodes': 10000,
        'scenes': [
            '单中心模型_规则',
            '单中心_轻度扰动',
            '单中心_强风拉伸',
            '双中心_复杂扩散',
            '三中心_多细丝',
            '环形_高分形',
            '偏离位置_左上',
            '偏离位置_右下',
            '极度不规则'
        ]
    }
]


def load_base_config(config_path: str) -> VolcanicAshConfig:
    if os.path.exists(config_path):
        return VolcanicAshConfig.load(config_path)
    return get_preset_configs()['单中心模型_规则']


def select_stages(start_stage: int, stop_stage: Optional[int]) -> List[Dict]:
    start_index = max(start_stage - 1, 0)
    stop_index = stop_stage if stop_stage is not None else len(CURRICULUM_STAGES)
    return CURRICULUM_STAGES[start_index:stop_index]


def train_stage(stage: Dict,
                config_path: str,
                save_root: str,
                max_steps: int,
                learning_rate: float,
                batch_size: int,
                buffer_size: int,
                noise_decay: float,
                min_noise: float,
                algorithm: str,
                policy_noise: float,
                noise_clip: float,
                policy_delay: int,
                update_every: int,
                load_model: Optional[str] = None,
                episodes_override: Optional[int] = None) -> str:
    base_config = load_base_config(config_path)
    scene_configs = get_training_scene_configs(stage['scenes'])
    stage_config = VolcanicAshConfig.from_dict(scene_configs[0].to_dict())
    stage_config.geo_center_lat = base_config.geo_center_lat
    stage_config.geo_center_lon = base_config.geo_center_lon
    stage_config.geo_span_lat = base_config.geo_span_lat
    stage_config.geo_span_lon = base_config.geo_span_lon
    stage_config.training_scene_names = [scene.scene_name for scene in scene_configs]

    episodes = episodes_override if episodes_override is not None else stage['episodes']
    save_dir = os.path.join(save_root, stage['name'])

    print('=' * 80)
    print(f"Training stage: {stage['name']}")
    print(f"Algorithm: {algorithm.upper()}")
    print(f"Episodes: {episodes}")
    print(f"Scenes: {', '.join(stage['scenes'])}")
    print(f"Save dir: {save_dir}")
    print('=' * 80)

    trainer = Trainer(
        config=stage_config,
        num_episodes=episodes,
        max_steps_per_episode=max_steps,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        noise_decay=noise_decay,
        min_noise=min_noise,
        algorithm=algorithm,
        policy_noise=policy_noise,
        noise_clip=noise_clip,
        policy_delay=policy_delay,
        save_dir=save_dir,
        scene_configs=scene_configs
    )

    if load_model:
        trainer.agent.load_model(load_model)
        print(f'Loaded previous model: {load_model}')

    trainer.train(update_every=update_every, log_interval=50)
    return os.path.join(save_dir, 'final_model.pth')


def main():
    parser = argparse.ArgumentParser(description='Run staged curriculum training for volcanic ash avoidance.')
    parser.add_argument('--config', default='output/current_config.json',
                        help='Base config path used for geographic metadata.')
    parser.add_argument('--save-root', default='models/curriculum',
                        help='Directory that will contain stage subdirectories.')
    parser.add_argument('--start-stage', type=int, default=1,
                        help='1-based stage index to start from.')
    parser.add_argument('--stop-stage', type=int, default=None,
                        help='1-based stage index to stop at, inclusive.')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Override episodes for every selected stage.')
    parser.add_argument('--max-steps', type=int, default=400)
    parser.add_argument('--algorithm', choices=['td3', 'ddpg'], default='td3')
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--buffer-size', type=int, default=300000)
    parser.add_argument('--noise-decay', type=float, default=0.999)
    parser.add_argument('--min-noise', type=float, default=0.05)
    parser.add_argument('--policy-noise', type=float, default=0.2)
    parser.add_argument('--noise-clip', type=float, default=0.5)
    parser.add_argument('--policy-delay', type=int, default=2)
    parser.add_argument('--update-every', type=int, default=10)
    parser.add_argument('--load-model', default=None,
                        help='Optional checkpoint to load before the first selected stage.')
    args = parser.parse_args()

    stop_stage = args.stop_stage
    if stop_stage is not None:
        stop_stage = max(stop_stage, args.start_stage)

    stages = select_stages(args.start_stage, stop_stage)
    if not stages:
        raise ValueError('No curriculum stages selected.')

    os.makedirs(args.save_root, exist_ok=True)
    previous_model = args.load_model
    for stage in stages:
        previous_model = train_stage(
            stage=stage,
            config_path=args.config,
            save_root=args.save_root,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            noise_decay=args.noise_decay,
            min_noise=args.min_noise,
            algorithm=args.algorithm,
            policy_noise=args.policy_noise,
            noise_clip=args.noise_clip,
            policy_delay=args.policy_delay,
            update_every=args.update_every,
            load_model=previous_model,
            episodes_override=args.episodes
        )

    print(f'Final curriculum model: {previous_model}')


if __name__ == '__main__':
    main()
