"""
Experience Replay Buffer for DQN.

Stores transitions and provides random sampling for training.
"""

import numpy as np
from collections import deque
import random
from typing import Tuple


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.
    
    Stores (state, action, reward, next_state, done) tuples and
    provides random batch sampling for training.
    """
    
    def __init__(self, capacity: int = 100000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                np.ndarray, np.ndarray]:
        """
        Sample a random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as numpy arrays
        """
        batch = random.sample(self.buffer, batch_size)
        
        states = np.array([t[0] for t in batch], dtype=np.float32)
        actions = np.array([t[1] for t in batch], dtype=np.int64)
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = np.array([t[3] for t in batch], dtype=np.float32)
        dones = np.array([t[4] for t in batch], dtype=np.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return current buffer size"""
        return len(self.buffer)
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples for a batch"""
        return len(self.buffer) >= batch_size
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()


if __name__ == "__main__":
    # Test replay buffer
    buffer = ReplayBuffer(capacity=1000)
    
    print("Testing ReplayBuffer...")
    
    # Add some transitions
    for i in range(100):
        state = np.random.randn(14).astype(np.float32)
        action = np.random.randint(4)
        reward = np.random.randn()
        next_state = np.random.randn(14).astype(np.float32)
        done = i == 99
        
        buffer.push(state, action, reward, next_state, done)
    
    print(f"Buffer size: {len(buffer)}")
    print(f"Is ready for batch of 32: {buffer.is_ready(32)}")
    
    # Sample a batch
    states, actions, rewards, next_states, dones = buffer.sample(32)
    
    print(f"\nSampled batch shapes:")
    print(f"  States: {states.shape}")
    print(f"  Actions: {actions.shape}")
    print(f"  Rewards: {rewards.shape}")
    print(f"  Next states: {next_states.shape}")
    print(f"  Dones: {dones.shape}")
