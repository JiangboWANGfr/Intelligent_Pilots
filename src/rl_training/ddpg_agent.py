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
                 hidden_size: int = 256):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        
        self.actor = ActorNetwork(state_dim, action_dim, hidden_size)
        self.actor_target = ActorNetwork(state_dim, action_dim, hidden_size)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        
        self.critic = CriticNetwork(state_dim, action_dim, hidden_size)
        self.critic_target = CriticNetwork(state_dim, action_dim, hidden_size)
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
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
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
        state_list.extend(state_dict['target_pos'])
        state_list.extend(state_dict['goal_vector'])
        state_list.extend(state_dict['local_concentration'])
        state_list.extend(state_dict['velocity'])
        state_list.extend(state_dict['distance_to_target'])
        return np.array(state_list, dtype=np.float32)
    
    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return None, None
        
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)
        
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
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
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'training_stats': self.training_stats,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim
        }
        
        torch.save(checkpoint, filepath)
    
    def load_model(self, filepath: str):
        checkpoint = torch.load(filepath, map_location='cpu')
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        if 'training_stats' in checkpoint:
            self.training_stats = checkpoint['training_stats']
