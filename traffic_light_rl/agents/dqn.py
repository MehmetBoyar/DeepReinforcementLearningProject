"""
Deep Q-Network (DQN) Agent for Traffic Light Control.

Uses neural network function approximation for Q-values.
"""

import numpy as np
from typing import Optional, Dict, Tuple
import os

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. DQN agent will not work.")

from .replay_buffer import ReplayBuffer


class DQNetwork(nn.Module):
    """
    Deep Q-Network architecture.
    
    A feed-forward neural network that maps states to Q-values for each action.
    """
    
    def __init__(self, state_dim: int = 14, action_dim: int = 4, 
                 hidden_dims: Tuple[int, ...] = (128, 128, 64)):
        """
        Initialize the Q-network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            hidden_dims: Tuple of hidden layer dimensions
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build network layers
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        return self.network(x)


class DuelingDQNetwork(nn.Module):
    """
    Dueling DQN architecture.
    
    Separates value and advantage streams for better learning.
    Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
    """
    
    def __init__(self, state_dim: int = 14, action_dim: int = 4,
                 hidden_dim: int = 128):
        """
        Initialize dueling network.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            hidden_dim: Hidden layer dimension
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared feature layers
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Value stream
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dueling architecture"""
        features = self.feature(x)
        value = self.value(features)
        advantage = self.advantage(features)
        
        # Combine value and advantage
        # Q = V + (A - mean(A))
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_values


class DQNAgent:
    """
    Deep Q-Network agent for traffic light control.
    
    Features:
    - Experience replay for stable training
    - Target network for stable Q-value targets
    - Epsilon-greedy exploration
    """
    
    def __init__(self,
                 state_dim: int = 14,
                 action_dim: int = 4,
                 hidden_dims: Tuple[int, ...] = (128, 128, 64),
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 buffer_capacity: int = 100000,
                 batch_size: int = 64,
                 target_update_freq: int = 100,
                 use_dueling: bool = False,
                 device: Optional[str] = None):
        """
        Initialize DQN agent.
        
        Args:
            state_dim: State dimension
            action_dim: Number of actions
            hidden_dims: Hidden layer dimensions
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Epsilon decay rate
            buffer_capacity: Replay buffer capacity
            batch_size: Training batch size
            target_update_freq: Steps between target network updates
            use_dueling: Whether to use dueling architecture
            device: Device to use ('cuda' or 'cpu')
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DQN agent")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Create networks
        if use_dueling:
            self.q_network = DuelingDQNetwork(state_dim, action_dim).to(self.device)
            self.target_network = DuelingDQNetwork(state_dim, action_dim).to(self.device)
        else:
            self.q_network = DQNetwork(state_dim, action_dim, hidden_dims).to(self.device)
            self.target_network = DQNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        
        # Training statistics
        self.train_step = 0
        self.episodes = 0
        self.losses = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode
            
        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        
        return int(q_values.argmax().item())
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store a transition in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train(self) -> Optional[float]:
        """
        Train the agent on a batch from replay buffer.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        if not self.replay_buffer.is_ready(self.batch_size):
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Compute loss
        loss = F.mse_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        
        # Update target network
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.update_target_network()
        
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        return loss_value
    
    def update_target_network(self):
        """Copy weights from Q-network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """Decay epsilon after each episode"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episodes += 1
    
    def get_statistics(self) -> Dict:
        """Get training statistics"""
        avg_loss = np.mean(self.losses[-100:]) if self.losses else 0.0
        return {
            'train_steps': self.train_step,
            'episodes': self.episodes,
            'epsilon': self.epsilon,
            'buffer_size': len(self.replay_buffer),
            'avg_loss': avg_loss
        }
    
    def save(self, filepath: str):
        """Save agent to file"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step,
            'episodes': self.episodes
        }, filepath)
    
    def load(self, filepath: str):
        """Load agent from file"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.train_step = checkpoint['train_step']
        self.episodes = checkpoint['episodes']
    
    def get_name(self) -> str:
        """Get agent name"""
        return "DQN"
    
    def get_config(self) -> Dict:
        """Get agent configuration"""
        return {
            'type': 'DQN',
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'gamma': self.gamma,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'batch_size': self.batch_size,
            'target_update_freq': self.target_update_freq,
            'device': str(self.device)
        }


class DoubleDQNAgent(DQNAgent):
    """
    Double DQN agent.
    
    Uses the online network to select actions and the target network
    to evaluate them, reducing overestimation bias.
    """
    
    def train(self) -> Optional[float]:
        """Train with double DQN update rule"""
        if not self.replay_buffer.is_ready(self.batch_size):
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute current Q-values
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: use online network to select action, target network to evaluate
        with torch.no_grad():
            # Select best actions using online network
            next_actions = self.q_network(next_states).argmax(1)
            # Evaluate using target network
            next_q = self.target_network(next_states).gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)
        
        # Compute loss
        loss = F.mse_loss(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        
        # Update target network
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.update_target_network()
        
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        return loss_value
    
    def get_name(self) -> str:
        return "Double DQN"


if __name__ == "__main__":
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Skipping DQN tests.")
    else:
        # Test DQN agent
        agent = DQNAgent()
        
        print("Testing DQN Agent...")
        print(f"Device: {agent.device}")
        print(f"Config: {agent.get_config()}")
        
        # Simulate some training
        for i in range(200):
            state = np.random.rand(14).astype(np.float32) * 20
            action = agent.select_action(state)
            reward = -np.random.rand() * 10
            next_state = np.random.rand(14).astype(np.float32) * 20
            done = (i + 1) % 100 == 0
            
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train()
            
            if done:
                agent.decay_epsilon()
        
        print(f"\nStatistics: {agent.get_statistics()}")
        
        # Test action selection
        test_state = np.random.rand(14).astype(np.float32)
        action = agent.select_action(test_state, training=False)
        print(f"\nTest action: {action}")
