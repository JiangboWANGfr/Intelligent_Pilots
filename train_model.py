import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.config.volcanic_ash_config import VolcanicAshConfig, get_training_scene_configs, get_preset_configs
from src.rl_training.trainer import Trainer

print('='*70)
print('STARTING REINFORCEMENT LEARNING TRAINING')
print('='*70)
print()
print('Configuration:')
print('  - Algorithm: DDPG (Deep Deterministic Policy Gradient)')
print('  - Episodes: 300')
print('  - Max steps per episode: 300')
print('  - Learning rate: 0.0001')
print('  - Environment: VolcanicAshEnv (Gymnasium)')
print()

# Load configuration from GUI-saved file
config_path = 'output/current_config.json'
if os.path.exists(config_path):
    config = VolcanicAshConfig.load(config_path)
    print(f'✓ Loaded configuration from: {config_path}')
else:
    print(f'⚠ Configuration file not found: {config_path}')
    print('  Please run GUI first (python config_gui.py) and save configuration')
    print('  Using default double center model as fallback...')
    config = get_preset_configs()['双中心_复杂扩散']

if config.training_scene_names:
    scene_configs = get_training_scene_configs(config.training_scene_names)
else:
    scene_configs = get_training_scene_configs()

config.training_scene_names = [scene.scene_name for scene in scene_configs]

print(f'Volcanic Ash Model: {config.model_type}')
print(f'  Centers: {len(config.centers)}')
print(f'  Cloud size: {config.cloud_size}')
print(f'  Threshold: {config.concentration_threshold}')
print(f'  Geo position: ({config.geo_center_lat}, {config.geo_center_lon})')
print(f'  Training scenes: {len(scene_configs)}')
print()

trainer = Trainer(
    config=config,
    num_episodes=300,
    max_steps_per_episode=300,
    learning_rate=1e-4,
    save_dir='models',
    scene_configs=scene_configs
)

print('Starting training...')
print('-'*70)

agent, history = trainer.train(log_interval=50)

print()
print('='*70)
print('TRAINING COMPLETED SUCCESSFULLY!')
print('='*70)
final_reward = history['rewards'][-1] if history['rewards'] else 0
print(f'Model saved to: models/final_model.pth')
print(f'Training curves: models/training_curves.png')
print(f'Final reward: {final_reward:.2f}')
