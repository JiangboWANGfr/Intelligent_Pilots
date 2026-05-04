import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, Tuple, List, Optional
import json
import os
from datetime import datetime
from collections import deque
from copy import deepcopy

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
                 initial_noise: float = 0.3,
                 expert_warmup_episodes: int = 0,
                 behavior_clone_steps: int = 0,
                 bc_regularization_steps: int = 0,
                 expert_gain: float = 1.0,
                 imitation_only: bool = False,
                 checkpoint_interval: int = 100,
                 live_preview: bool = False,
                 preview_interval: int = 50,
                 preview_steps: int = 220,
                 preview_seed: int = 2026,
                 scene_configs: Optional[List[VolcanicAshConfig]] = None):
        
        self.config = config
        self.scene_configs = scene_configs or [config]
        self.num_episodes = num_episodes
        self.max_steps = max_steps_per_episode
        self.save_dir = save_dir
        self.learning_rate = learning_rate
        self.noise_decay = noise_decay
        self.min_noise = min_noise
        self.algorithm = algorithm.lower()
        self.initial_noise = initial_noise
        self.expert_warmup_episodes = expert_warmup_episodes
        self.behavior_clone_steps = behavior_clone_steps
        self.bc_regularization_steps = bc_regularization_steps
        self.expert_gain = expert_gain
        self.imitation_only = imitation_only
        self.checkpoint_interval = max(0, int(checkpoint_interval))
        self.live_preview = bool(live_preview)
        self.preview_interval = max(1, int(preview_interval))
        self.preview_steps = max(1, int(preview_steps))
        self.preview_seed = int(preview_seed)
        self.preview_env = None
        self.preview_window_name = 'Training Preview'
        self.preview_available = self.live_preview
        self.expert_states = None
        self.expert_actions = None
        
        os.makedirs(save_dir, exist_ok=True)
        
        self.env = VolcanicAshEnv(config, render_mode=None, scene_configs=self.scene_configs)
        self.env.max_steps = max_steps_per_episode
        if self.live_preview:
            self.preview_env = VolcanicAshEnv(
                VolcanicAshConfig.from_dict(config.to_dict()),
                render_mode=None,
                scene_configs=[
                    VolcanicAshConfig.from_dict(scene.to_dict())
                    for scene in self.scene_configs
                ]
            )
            self.preview_env.max_steps = self.preview_steps
        
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
        
        self.noise_scale = initial_noise
        
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
            'learning_rate': self.learning_rate,
            'initial_noise': self.initial_noise,
            'expert_warmup_episodes': self.expert_warmup_episodes,
            'behavior_clone_steps': self.behavior_clone_steps,
            'bc_regularization_steps': self.bc_regularization_steps,
            'expert_gain': self.expert_gain,
            'imitation_only': self.imitation_only,
            'checkpoint_interval': self.checkpoint_interval,
            'live_preview': self.live_preview,
            'preview_interval': self.preview_interval,
            'preview_steps': self.preview_steps,
            'preview_seed': self.preview_seed,
            'training_scene_names': [scene.scene_name or scene.model_type for scene in self.scene_configs]
        }
    
    def _calculate_state_dim(self) -> int:
        obs = self.env.reset()[0]
        flat_state = DDPGAgent.flatten_state(obs)
        return len(flat_state)

    def _collect_expert_transitions(self) -> Tuple[np.ndarray, np.ndarray]:
        expert_states = []
        expert_actions = []

        for _ in range(self.expert_warmup_episodes):
            state, _ = self.env.reset()
            for _step in range(self.max_steps):
                flat_state = DDPGAgent.flatten_state(state)
                action = np.array([
                    self.env.get_reference_turn_command(gain=self.expert_gain)
                ], dtype=np.float32)

                next_state, reward, terminated, truncated, _info = self.env.step(action)
                flat_next_state = DDPGAgent.flatten_state(next_state)
                done = terminated or truncated

                self.agent.replay_buffer.push(
                    flat_state,
                    action,
                    reward,
                    flat_next_state,
                    done
                )
                expert_states.append(flat_state)
                expert_actions.append(action)
                state = next_state

                if done:
                    break

        if not expert_states:
            return np.empty((0, self.agent.state_dim), dtype=np.float32), \
                np.empty((0, self.agent.action_dim), dtype=np.float32)

        return (
            np.asarray(expert_states, dtype=np.float32),
            np.asarray(expert_actions, dtype=np.float32)
        )

    def _behavior_clone_actor(self,
                              expert_states: np.ndarray,
                              expert_actions: np.ndarray,
                              steps: Optional[int] = None):
        clone_steps = self.behavior_clone_steps if steps is None else steps
        if clone_steps <= 0 or len(expert_states) == 0:
            return

        batch_size = min(self.agent.batch_size, len(expert_states))
        states = torch.as_tensor(expert_states, dtype=torch.float32, device=self.agent.device)
        actions = torch.as_tensor(expert_actions, dtype=torch.float32, device=self.agent.device)

        self.agent.actor.train()
        for _ in range(clone_steps):
            indices = torch.randint(0, len(states), (batch_size,), device=self.agent.device)
            predicted_actions = self.agent.actor(states[indices])
            bc_loss = torch.nn.functional.mse_loss(predicted_actions, actions[indices])

            self.agent.actor_optimizer.zero_grad()
            bc_loss.backward()
            self.agent.actor_optimizer.step()

        DDPGAgent.hard_update(self.agent.actor_target, self.agent.actor)

    def _build_preview_frame(self,
                             episode: int,
                             episode_reward: float,
                             success_rate: float,
                             path_points: List[np.ndarray],
                             info: Dict,
                             success: bool) -> np.ndarray:
        import cv2

        env = self.preview_env
        concentration_map = np.asarray(env.concentration_map, dtype=np.float32)
        threshold = float(env.safety_threshold)
        frame = np.full((env.height, env.width, 3), (245, 247, 250), dtype=np.uint8)
        trace_mask = (concentration_map >= threshold * 0.15) & (concentration_map < threshold * 0.5)
        low_mask = (concentration_map >= threshold * 0.5) & (concentration_map < threshold)
        medium_mask = (concentration_map >= threshold) & (concentration_map < threshold * 1.5)
        high_mask = concentration_map >= threshold * 1.5
        frame[trace_mask] = (0, 255, 0)
        frame[low_mask] = (255, 255, 0)
        frame[medium_mask] = (255, 165, 0)
        frame[high_mask] = (255, 0, 0)

        if env.reference_path is not None and len(env.reference_path) > 1:
            ref_points = np.round(env.reference_path[:, [1, 0]]).astype(np.int32)
            cv2.polylines(frame, [ref_points], False, (255, 255, 255), 2, cv2.LINE_AA)

        if len(path_points) > 1:
            trajectory = np.round(np.asarray(path_points)[:, [1, 0]]).astype(np.int32)
            cv2.polylines(frame, [trajectory], False, (0, 255, 255), 3, cv2.LINE_AA)

        start_xy = tuple(np.round(path_points[0][[1, 0]]).astype(int).tolist())
        aircraft_xy = tuple(np.round(env.aircraft_pos[[1, 0]]).astype(int).tolist())
        target_xy = tuple(np.round(env.target_pos[[1, 0]]).astype(int).tolist())
        cv2.circle(frame, start_xy, 6, (0, 160, 80), -1, cv2.LINE_AA)
        cv2.circle(frame, target_xy, 10, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.drawMarker(frame, target_xy, (0, 0, 255), cv2.MARKER_TILTED_CROSS, 22, 3, cv2.LINE_AA)
        cv2.drawMarker(frame, aircraft_xy, (255, 255, 255), cv2.MARKER_TRIANGLE_UP, 22, 3, cv2.LINE_AA)
        cv2.drawMarker(frame, aircraft_xy, (0, 255, 255), cv2.MARKER_TRIANGLE_UP, 18, 2, cv2.LINE_AA)

        panel_width = 360
        scale = 2
        frame = cv2.resize(frame, (env.width * scale, env.height * scale), interpolation=cv2.INTER_NEAREST)
        canvas = np.full((frame.shape[0], frame.shape[1] + panel_width, 3), 24, dtype=np.uint8)
        canvas[:, :frame.shape[1]] = frame

        x = frame.shape[1] + 18
        y = 36
        lines = [
            f'Episode: {episode + 1}/{self.num_episodes}',
            f'Preview: {"SUCCESS" if success else "NOT SUCCESS"}',
            f'Train reward: {episode_reward:.1f}',
            f'Train success: {success_rate:.1f}%',
            f'Final distance: {float(info.get("distance_to_target", 0.0)):.1f}',
            f'Ash exposure: {float(info.get("ash_exposure", 0.0)):.2f}',
            f'Path progress: {float(info.get("path_progress_ratio", 0.0)):.2f}',
            f'Max conc: {float(info.get("max_concentration_exposure", 0.0)):.3f}',
            f'Scene: {env.scene_name[:22]}'
        ]
        cv2.putText(canvas, 'Live Training Preview', (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.72, (0, 255, 255), 2, cv2.LINE_AA)
        y += 42
        for line in lines:
            cv2.putText(canvas, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (235, 235, 235), 1, cv2.LINE_AA)
            y += 30

        legend = [
            ('Safe trace', (0, 255, 0)),
            ('Low', (255, 255, 0)),
            ('Medium', (255, 165, 0)),
            ('High', (255, 0, 0)),
            ('Flight', (0, 255, 255))
        ]
        y += 10
        for label, color in legend:
            cv2.rectangle(canvas, (x, y - 14), (x + 22, y + 4), color, -1)
            cv2.putText(canvas, label, (x + 32, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (235, 235, 235), 1, cv2.LINE_AA)
            y += 26

        return canvas

    def _update_live_preview(self,
                             episode: int,
                             episode_reward: float,
                             success_rate: float):
        if not self.preview_available or self.preview_env is None:
            return

        try:
            import cv2
            if hasattr(self.preview_env, 'random_scene_counter'):
                self.preview_env.random_scene_counter = 0
            state, info = self.preview_env.reset(seed=self.preview_seed)
            path_points = [self.preview_env.aircraft_pos.copy()]
            success = False
            final_info = info

            for _ in range(self.preview_steps):
                action = self.agent.select_action(state, evaluate=True)
                state, _reward, terminated, truncated, final_info = self.preview_env.step(action)
                path_points.append(self.preview_env.aircraft_pos.copy())
                if terminated or truncated:
                    success = bool(
                        terminated and final_info.get('distance_to_target', float('inf')) <
                        self.preview_env.success_threshold
                    )
                    break

            frame = self._build_preview_frame(
                episode=episode,
                episode_reward=episode_reward,
                success_rate=success_rate,
                path_points=path_points,
                info=final_info,
                success=success
            )
            cv2.imshow(self.preview_window_name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                self.preview_available = False
                cv2.destroyWindow(self.preview_window_name)
        except Exception as exc:
            self.preview_available = False
            print(f'Live preview disabled: {exc}')
    
    def train(self, update_every: int = 10, log_interval: int = 10):
        if self.expert_warmup_episodes > 0:
            self.expert_states, self.expert_actions = self._collect_expert_transitions()
            self._behavior_clone_actor(self.expert_states, self.expert_actions)
            print(
                f"Expert warmup collected {len(self.expert_states)} transitions; "
                f"replay buffer size: {len(self.agent.replay_buffer)}"
            )

        if self.imitation_only or self.num_episodes <= 0:
            final_model_path = os.path.join(self.save_dir, 'final_model.pth')
            self.agent.save_model(final_model_path)
            self.save_training_history()
            self.plot_training_curves()
            print("\nImitation training completed!")
            print(f"Final model saved to: {final_model_path}")
            return self.agent, self.training_history

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
            self.training_history.setdefault('learning_rates', []).append(self.learning_rate)
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

            if self.live_preview and episode % self.preview_interval == 0:
                self._update_live_preview(
                    episode=episode,
                    episode_reward=episode_reward,
                    success_rate=success_rate
                )

            if self.bc_regularization_steps > 0 and self.expert_states is not None:
                self._behavior_clone_actor(
                    self.expert_states,
                    self.expert_actions,
                    steps=self.bc_regularization_steps
                )
            
            if self.checkpoint_interval > 0 and (episode + 1) % self.checkpoint_interval == 0:
                self.save_checkpoint(episode + 1)
        
        final_model_path = os.path.join(self.save_dir, 'final_model.pth')
        self.agent.save_model(final_model_path)
        
        self.save_training_history()
        self.plot_training_curves()
        
        print(f"\nTraining completed!")
        print(f"Final model saved to: {final_model_path}")
        print(f"Success rate: {success_count/self.num_episodes*100:.1f}%")
        if self.preview_available:
            try:
                import cv2
                cv2.destroyWindow(self.preview_window_name)
            except Exception:
                pass
        
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
