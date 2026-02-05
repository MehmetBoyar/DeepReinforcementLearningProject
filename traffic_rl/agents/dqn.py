import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from collections import deque
from traffic_rl.agents.base import BaseAgent
from traffic_rl.config import AgentConfig

class DQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, dueling=False):
        super(DQNetwork, self).__init__()
        self.dueling = dueling

        # Shared Feature Extractor
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        if self.dueling:
            # Value Stream
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            # Advantage Stream
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
        else:
            # Standard Output
            self.output_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        features = self.feature_layer(x)
        
        if self.dueling:
            V = self.value_stream(features)
            A = self.advantage_stream(features)
            return V + (A - A.mean(dim=1, keepdim=True))
        else:
            return self.output_layer(features)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), np.array(action), np.array(reward), 
                np.array(next_state), np.array(done))

    def __len__(self):
        return len(self.buffer)

class DQNAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, config: AgentConfig, device="cpu"):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # DEVICE DETECTION 
        if device == "cuda" and not torch.cuda.is_available():
            print("⚠️ Warning: Config requested CUDA but Torch not compiled with CUDA. Switching to CPU.")
            self.device = torch.device("cpu")
        elif device == "cpu" and torch.cuda.is_available():
            print("ℹ️ Info: CPU requested but CUDA available. Using CUDA for speed.")
            self.device = torch.device("cuda")
        else:
            self.device = torch.device(device)
        
        self.is_double = getattr(config, 'double_dqn', False)
        self.is_dueling = getattr(config, 'dueling_dqn', False)

        self.epsilon = config.epsilon_start
        self.epsilon_min = config.epsilon_min
        self.epsilon_decay = config.epsilon_decay
        
        self.policy_net = DQNetwork(state_dim, action_dim, config.hidden_dim, dueling=self.is_dueling).to(self.device)
        self.target_net = DQNetwork(state_dim, action_dim, config.hidden_dim, dueling=self.is_dueling).to(self.device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.lr)
        self.memory = ReplayBuffer(config.buffer_size)
        
        self.update_step_count = 0
        self.target_update_freq = 100 

    def act(self, state, training=True):
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return q_values.argmax().item()

    def update(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
        
        if len(self.memory) < self.config.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.config.batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        curr_q = self.policy_net(states).gather(1, actions)
        
        with torch.no_grad():
            if self.is_double:
                best_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
                next_q_values = self.target_net(next_states).gather(1, best_actions)
                expected_q = rewards + (self.config.gamma * next_q_values * (1 - dones))
            else:
                max_next_q = self.target_net(next_states).max(1)[0].unsqueeze(1)
                expected_q = rewards + (self.config.gamma * max_next_q * (1 - dones))

        loss = nn.MSELoss()(curr_q, expected_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.update_step_count += 1
        if self.update_step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        return loss.item()

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model': self.policy_net.state_dict(),
            'optim': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'config': {
                'hidden_dim': self.config.hidden_dim,
                'double_dqn': self.is_double,
                'dueling_dqn': self.is_dueling
            }
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        state_dict = ckpt['model']
        
        # Legacy Support Logic
        if any(k.startswith("net.") for k in state_dict.keys()):
            print(f"⚠️  Converting legacy model architecture for {os.path.basename(path)}")
            new_state_dict = {}
            for key, value in state_dict.items():
                if "net.0" in key: new_key = key.replace("net.0", "feature_layer.0")
                elif "net.2" in key: new_key = key.replace("net.2", "feature_layer.2")
                elif "net.4" in key: new_key = key.replace("net.4", "output_layer")
                else: new_key = key
                new_state_dict[new_key] = value
            state_dict = new_state_dict

        try:
            self.policy_net.load_state_dict(state_dict)
            self.target_net.load_state_dict(state_dict)
            self.optimizer.load_state_dict(ckpt['optim'])
            self.epsilon = ckpt.get('epsilon', self.epsilon_min)
        except RuntimeError as e:
            print(f"❌ Critical Error loading model: {e}")