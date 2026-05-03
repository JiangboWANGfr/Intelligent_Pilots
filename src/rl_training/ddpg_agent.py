import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime


def resolve_device(device: str = 'auto') -> torch.device:
    requested = (device or 'auto').lower()
    if requested == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            return torch.device('mps')
        return torch.device('cpu')
    if requested == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available in this PyTorch environment.')
    if requested == 'mps' and not (getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()):
        raise RuntimeError('MPS was requested but is not available in this PyTorch environment.')
    return torch.device(requested)


class ActorNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_dim),
            nn.Tanh()
        )
        
    def forward(self, state):
        return self.net(state)


class CriticNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 100000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


class DDPGAgent:
    def __init__(self, state_dim: int, action_dim: int,
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 buffer_size: int = 100000,
                 batch_size: int = 64,
                 hidden_size: int = 256,
                 device: str = 'auto'):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = resolve_device(device)
        
        self.actor = ActorNetwork(state_dim, action_dim, hidden_size).to(self.device)
        self.actor_target = ActorNetwork(state_dim, action_dim, hidden_size).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        
        self.critic = CriticNetwork(state_dim, action_dim, hidden_size).to(self.device)
        self.critic_target = CriticNetwork(state_dim, action_dim, hidden_size).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        self.hard_update(self.actor, self.actor_target)
        self.hard_update(self.critic, self.critic_target)
        
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        self.training_stats = {
            'episode_rewards': [],
            'episode_losses': [],
            'actor_losses': [],
            'critic_losses': []
        }
    
    @staticmethod
    def hard_update(target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
    
    def select_action(self, state, noise_scale: float = 0.1, evaluate: bool = False):
        if isinstance(state, dict):
            state = self.flatten_state(state)
        
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        self.actor.train()
        
        if not evaluate:
            noise = np.random.normal(0, noise_scale, size=self.action_dim)
            action = action + noise
            action = np.clip(action, -1, 1)
        
        return action
    
    @staticmethod
    def flatten_state(state_dict: Dict) -> np.ndarray:
        state_list = []
        state_list.extend(state_dict['aircraft_pos'])
        state_list.extend(state_dict['goal_vector'])
        state_list.extend(state_dict['heading_vec'])
        speed_value = state_dict.get('cruise_speed', state_dict.get('speed'))
        state_list.extend(speed_value)
        state_list.extend(state_dict['distance_to_target'])
        state_list.extend(state_dict['current_concentration'])
        state_list.extend(state_dict['forward_concentration'])
        state_list.extend(state_dict['lookahead_vector'])
        state_list.extend(state_dict['lookahead_heading_error'])
        state_list.extend(state_dict['reference_turn_cmd'])
        state_list.extend(state_dict['cross_track_error'])
        state_list.extend(state_dict['path_progress_ratio'])
        return np.array(state_list, dtype=np.float32)
    
    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None, None
        
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)
        
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        current_q = self.critic(states, actions)
        
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.soft_update(self.actor_target, self.actor)
        self.soft_update(self.critic_target, self.critic)
        
        return actor_loss.item(), critic_loss.item()
    
    def save_model(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        checkpoint = {
            'algorithm': 'ddpg',
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'training_stats': self.training_stats,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'device': str(self.device)
        }
        
        torch.save(checkpoint, filepath)
    
    def load_model(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)

        if checkpoint.get('algorithm') == 'td3' or 'critic1_state_dict' in checkpoint:
            raise ValueError('This checkpoint is TD3. Use TD3Agent to load it.')
        checkpoint_state_dim = checkpoint.get('state_dim')
        checkpoint_action_dim = checkpoint.get('action_dim')
        if checkpoint_state_dim is not None and checkpoint_state_dim != self.state_dim:
            raise ValueError(
                f'Checkpoint state_dim={checkpoint_state_dim} does not match '
                f'environment state_dim={self.state_dim}. Retrain with the current observation space.'
            )
        if checkpoint_action_dim is not None and checkpoint_action_dim != self.action_dim:
            raise ValueError(
                f'Checkpoint action_dim={checkpoint_action_dim} does not match '
                f'environment action_dim={self.action_dim}. Retrain with the fixed-speed turn controller.'
            )
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        if 'training_stats' in checkpoint:
            self.training_stats = checkpoint['training_stats']


class TD3Agent:
    def __init__(self, state_dim: int, action_dim: int,
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 buffer_size: int = 100000,
                 batch_size: int = 64,
                 hidden_size: int = 256,
                 policy_noise: float = 0.2,
                 noise_clip: float = 0.5,
                 policy_delay: int = 2,
                 device: str = 'auto'):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.total_it = 0
        self.device = resolve_device(device)

        self.actor = ActorNetwork(state_dim, action_dim, hidden_size).to(self.device)
        self.actor_target = ActorNetwork(state_dim, action_dim, hidden_size).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)

        self.critic1 = CriticNetwork(state_dim, action_dim, hidden_size).to(self.device)
        self.critic2 = CriticNetwork(state_dim, action_dim, hidden_size).to(self.device)
        self.critic1_target = CriticNetwork(state_dim, action_dim, hidden_size).to(self.device)
        self.critic2_target = CriticNetwork(state_dim, action_dim, hidden_size).to(self.device)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=learning_rate
        )

        DDPGAgent.hard_update(self.actor, self.actor_target)
        DDPGAgent.hard_update(self.critic1, self.critic1_target)
        DDPGAgent.hard_update(self.critic2, self.critic2_target)

        self.replay_buffer = ReplayBuffer(buffer_size)

        self.training_stats = {
            'episode_rewards': [],
            'episode_losses': [],
            'actor_losses': [],
            'critic_losses': []
        }

    @staticmethod
    def flatten_state(state_dict: Dict) -> np.ndarray:
        return DDPGAgent.flatten_state(state_dict)

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def select_action(self, state, noise_scale: float = 0.1, evaluate: bool = False):
        if isinstance(state, dict):
            state = self.flatten_state(state)

        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        self.actor.train()

        if not evaluate:
            noise = np.random.normal(0, noise_scale, size=self.action_dim)
            action = np.clip(action + noise, -1, 1)

        return action

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None, None

        self.total_it += 1

        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)

        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        with torch.no_grad():
            noise = torch.randn_like(actions) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-1, 1)

            target_q1 = self.critic1_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss_value = None
        if self.total_it % self.policy_delay == 0:
            predicted_actions = self.actor(states)
            actor_loss = -self.critic1(states, predicted_actions).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.actor_target, self.actor)
            self.soft_update(self.critic1_target, self.critic1)
            self.soft_update(self.critic2_target, self.critic2)
            actor_loss_value = actor_loss.item()

        return actor_loss_value, critic_loss.item()

    def save_model(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        checkpoint = {
            'algorithm': 'td3',
            'actor_state_dict': self.actor.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic1_state_dict': self.critic1.state_dict(),
            'critic2_state_dict': self.critic2.state_dict(),
            'critic1_target_state_dict': self.critic1_target.state_dict(),
            'critic2_target_state_dict': self.critic2_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'training_stats': self.training_stats,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'policy_noise': self.policy_noise,
            'noise_clip': self.noise_clip,
            'policy_delay': self.policy_delay,
            'total_it': self.total_it,
            'device': str(self.device)
        }

        torch.save(checkpoint, filepath)

    def load_model(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)

        if checkpoint.get('algorithm') not in (None, 'td3'):
            raise ValueError(f"Checkpoint algorithm is {checkpoint.get('algorithm')}, not td3")
        if 'critic1_state_dict' not in checkpoint:
            raise ValueError('This checkpoint does not contain TD3 critics. Train a new TD3 model first.')
        checkpoint_state_dim = checkpoint.get('state_dim')
        checkpoint_action_dim = checkpoint.get('action_dim')
        if checkpoint_state_dim is not None and checkpoint_state_dim != self.state_dim:
            raise ValueError(
                f'Checkpoint state_dim={checkpoint_state_dim} does not match '
                f'environment state_dim={self.state_dim}. Retrain with the current observation space.'
            )
        if checkpoint_action_dim is not None and checkpoint_action_dim != self.action_dim:
            raise ValueError(
                f'Checkpoint action_dim={checkpoint_action_dim} does not match '
                f'environment action_dim={self.action_dim}. Retrain with the fixed-speed turn controller.'
            )

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic1.load_state_dict(checkpoint['critic1_state_dict'])
        self.critic2.load_state_dict(checkpoint['critic2_state_dict'])
        self.critic1_target.load_state_dict(checkpoint['critic1_target_state_dict'])
        self.critic2_target.load_state_dict(checkpoint['critic2_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.policy_noise = checkpoint.get('policy_noise', self.policy_noise)
        self.noise_clip = checkpoint.get('noise_clip', self.noise_clip)
        self.policy_delay = checkpoint.get('policy_delay', self.policy_delay)
        self.total_it = checkpoint.get('total_it', 0)

        if 'training_stats' in checkpoint:
            self.training_stats = checkpoint['training_stats']


def infer_checkpoint_algorithm(filepath: str) -> str:
    checkpoint = torch.load(filepath, map_location='cpu')
    if checkpoint.get('algorithm') == 'td3' or 'critic1_state_dict' in checkpoint:
        return 'td3'
    return 'ddpg'


def create_agent(algorithm: str, state_dim: int, action_dim: int, **kwargs):
    algorithm = algorithm.lower()
    if algorithm == 'td3':
        return TD3Agent(state_dim=state_dim, action_dim=action_dim, **kwargs)
    if algorithm == 'ddpg':
        ddpg_kwargs = {
            key: value for key, value in kwargs.items()
            if key not in {'policy_noise', 'noise_clip', 'policy_delay'}
        }
        return DDPGAgent(state_dim=state_dim, action_dim=action_dim, **ddpg_kwargs)
    raise ValueError(f'Unsupported algorithm: {algorithm}')
