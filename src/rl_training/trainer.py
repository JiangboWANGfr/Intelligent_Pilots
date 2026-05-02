import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, Tuple, List, Optional
import json
import os
from datetime import datetime
from collections import deque

from src.rl_env.volcanic_ash_env import VolcanicAshEnv
from src.rl_training.ddpg_agent import DDPGAgent, create_agent
from src.config.volcanic_ash_config import VolcanicAshConfig


class Trainer:
    def __init__(self, config: VolcanicAshConfig,
                 num_episodes: int = 1000,
                 max_steps_per_episode: int = 500,
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 buffer_size: int = 100000,
                 batch_size: int = 64,
                 noise_decay: float = 0.995,
                 min_noise: float = 0.05,
                 save_dir: str = 'models',
                 algorithm: str = 'td3',
                 policy_noise: float = 0.2,
                 noise_clip: float = 0.5,
                 policy_delay: int = 2,
                 device: str = 'auto',
                 scene_configs: Optional[List[VolcanicAshConfig]] = None):
        
        self.config = config
        self.scene_configs = scene_configs or [config]
        self.num_episodes = num_episodes
        self.max_steps = max_steps_per_episode
        self.save_dir = save_dir
        self.noise_decay = noise_decay
        self.min_noise = min_noise
        self.algorithm = algorithm.lower()
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.env = VolcanicAshEnv(config, render_mode=None, scene_configs=self.scene_configs)
        self.env.max_steps = max_steps_per_episode
        
        state_dim = self._calculate_state_dim()
        action_dim = int(np.prod(self.env.action_space.shape))
        
        self.agent = create_agent(
            self.algorithm,
            state_dim=state_dim,
            action_dim=action_dim,
            learning_rate=learning_rate,
            gamma=gamma,
            tau=tau,
            buffer_size=buffer_size,
            batch_size=batch_size,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            policy_delay=policy_delay,
            device=device
        )
        
        self.noise_scale = 0.3
        
        self.training_history = {
            'episodes': [],
            'rewards': [],
            'losses': [],
            'actor_losses': [],
            'critic_losses': [],
            'steps': [],
            'success_rates': [],
            'success_flags': [],
            'timeout_flags': [],
            'danger_violations': [],
            'max_concentrations': [],
            'avg_concentrations': [],
            'fuel_consumptions': [],
            'ash_exposures': [],
            'path_progress_ratios': [],
            'cross_track_errors': [],
            'final_distances': [],
            'termination_reasons': [],
            'scene_names': [],
            'algorithm': self.algorithm,
            'device': str(self.agent.device),
            'training_scene_names': [scene.scene_name or scene.model_type for scene in self.scene_configs]
        }
    
    def _calculate_state_dim(self) -> int:
        obs = self.env.reset()[0]
        flat_state = DDPGAgent.flatten_state(obs)
        return len(flat_state)
    
    def train(self, update_every: int = 10, log_interval: int = 10):
        success_count = 0
        recent_rewards = deque(maxlen=100)
        
        pbar = tqdm(range(self.num_episodes), desc="Training")
        
        for episode in pbar:
            state, reset_info = self.env.reset()
            current_scene_name = reset_info.get('scene_name', self.env.scene_name)
            episode_reward = 0
            episode_loss_sum = 0
            episode_actor_loss_sum = 0
            episode_critic_loss_sum = 0
            loss_count = 0
            actor_loss_count = 0
            concentrations = [float(reset_info.get('current_concentration', 0.0))]
            cross_track_errors = [float(reset_info.get('cross_track_error', 0.0))]
            danger_violation = bool(reset_info.get('is_in_danger_zone', False))
            success = False
            timeout = False
            termination_reason = 'max_steps'
            final_info = reset_info
            
            for step in range(self.max_steps):
                action = self.agent.select_action(state, self.noise_scale)
                
                next_state, reward, terminated, truncated, info = self.env.step(action)
                final_info = info
                current_concentration = float(info.get('current_concentration', 0.0))
                concentrations.append(current_concentration)
                cross_track_errors.append(float(info.get('cross_track_error', 0.0)))
                danger_violation = danger_violation or bool(info.get('is_in_danger_zone', False))
                
                done = terminated or truncated
                
                flat_state = DDPGAgent.flatten_state(state)
                flat_next_state = DDPGAgent.flatten_state(next_state)
                
                self.agent.replay_buffer.push(flat_state, action, reward,
                                              flat_next_state, done)
                
                loss = (None, None)
                if step % update_every == 0:
                    loss = self.agent.train_step()
                if loss is not None and len(loss) == 2:
                    actor_loss, critic_loss = loss
                    if critic_loss is not None:
                        episode_loss_sum += critic_loss
                        episode_critic_loss_sum += critic_loss
                        loss_count += 1
                    if actor_loss is not None:
                        episode_loss_sum += actor_loss
                        episode_actor_loss_sum += actor_loss
                        actor_loss_count += 1
                
                episode_reward += reward
                state = next_state

                if done:
                    if terminated and info['distance_to_target'] < self.env.success_threshold:
                        success_count += 1
                        success = True
                        termination_reason = 'success'
                    elif truncated:
                        timeout = True
                        termination_reason = 'timeout'
                    elif current_concentration > 0.9:
                        termination_reason = 'extreme_concentration'
                    else:
                        termination_reason = 'terminated'
                    break
            
            recent_rewards.append(episode_reward)
            avg_reward = np.mean(recent_rewards)
            success_rate = success_count / (episode + 1) * 100
            
            self.noise_scale = max(self.min_noise,
                                   self.noise_scale * self.noise_decay)
            
            self.training_history['episodes'].append(episode)
            self.training_history['rewards'].append(episode_reward)
            self.training_history['losses'].append(episode_loss_sum / max(loss_count, 1))
            self.training_history['actor_losses'].append(episode_actor_loss_sum / max(actor_loss_count, 1))
            self.training_history['critic_losses'].append(episode_critic_loss_sum / max(loss_count, 1))
            self.training_history['steps'].append(step + 1)
            self.training_history['success_rates'].append(success_rate)
            self.training_history['success_flags'].append(success)
            self.training_history['timeout_flags'].append(timeout)
            self.training_history['danger_violations'].append(danger_violation)
            self.training_history['max_concentrations'].append(max(concentrations))
            self.training_history['avg_concentrations'].append(float(np.mean(concentrations)))
            self.training_history['fuel_consumptions'].append(float(final_info.get('fuel_consumed', 0.0)))
            self.training_history['ash_exposures'].append(float(final_info.get('ash_exposure', 0.0)))
            self.training_history['path_progress_ratios'].append(float(final_info.get('path_progress_ratio', 0.0)))
            self.training_history['cross_track_errors'].append(float(np.mean(cross_track_errors)))
            self.training_history['final_distances'].append(float(final_info.get('distance_to_target', 0.0)))
            self.training_history['termination_reasons'].append(termination_reason)
            self.training_history['scene_names'].append(current_scene_name)
            
            if episode % log_interval == 0:
                pbar.set_postfix({
                    'Reward': f'{avg_reward:.1f}',
                    'Success': f'{success_rate:.1f}%',
                    'Scene': current_scene_name
                })
            
            if (episode + 1) % 100 == 0:
                self.save_checkpoint(episode + 1)
        
        final_model_path = os.path.join(self.save_dir, 'final_model.pth')
        self.agent.save_model(final_model_path)
        
        self.save_training_history()
        self.plot_training_curves()
        
        print(f"\nTraining completed!")
        print(f"Final model saved to: {final_model_path}")
        print(f"Success rate: {success_count/self.num_episodes*100:.1f}%")
        
        return self.agent, self.training_history
    
    def save_checkpoint(self, episode: int):
        checkpoint_path = os.path.join(self.save_dir,
                                       f'checkpoint_ep{episode}.pth')
        self.agent.save_model(checkpoint_path)
    
    def save_training_history(self):
        history_path = os.path.join(self.save_dir, 'training_history.json')

        def make_serializable(value):
            if isinstance(value, list):
                return [make_serializable(item) for item in value]
            if isinstance(value, (np.integer,)):
                return int(value)
            if isinstance(value, (np.floating,)):
                return float(value)
            if isinstance(value, (np.bool_,)):
                return bool(value)
            return value

        serializable_history = {
            key: make_serializable(value)
            for key, value in self.training_history.items()
        }
        
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
    
    def plot_training_curves(self):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        axes[0, 0].plot(self.training_history['episodes'],
                        self.training_history['rewards'],
                        alpha=0.6, color='blue', label='Episode Reward')
        
        if len(self.training_history['rewards']) > 10:
            window = min(50, len(self.training_history['rewards']) // 4)
            moving_avg = np.convolve(self.training_history['rewards'],
                                    np.ones(window)/window, mode='valid')
            axes[0, 0].plot(range(window-1, len(self.training_history['rewards'])),
                           moving_avg, color='red', linewidth=2,
                           label=f'Moving Avg (window={window})')
        
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].set_title('Training Rewards')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(self.training_history['episodes'],
                        self.training_history['losses'],
                        alpha=0.6, color='orange')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_title('Training Loss')
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].plot(self.training_history['episodes'],
                        self.training_history['steps'],
                        alpha=0.6, color='green')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].set_title('Steps per Episode')
        axes[1, 0].grid(True, alpha=0.3)
        
        if len(self.training_history['rewards']) > 10:
            window = min(50, len(self.training_history['rewards']) // 4)
            success_rate = [np.mean(np.array(self.training_history['rewards']
                                           [max(0,i-window):i+1]) > 0) * 100
                          for i in range(len(self.training_history['rewards']))]
            axes[1, 1].plot(self.training_history['episodes'], success_rate,
                           color='purple', linewidth=2)
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Success Rate (%)')
        axes[1, 1].set_title('Approximate Success Rate')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 105)
        
        plt.tight_layout()
        plot_path = os.path.join(self.save_dir, 'training_curves.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Training curves saved to: {plot_path}")
