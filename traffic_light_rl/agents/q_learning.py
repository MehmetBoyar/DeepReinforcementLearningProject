"""
Tabular Q-Learning Agent for Traffic Light Control.

Uses state discretization to handle continuous state space.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List
import pickle


def discretize_state(state: np.ndarray, 
                     queue_bins: Optional[List[float]] = None,
                     duration_bins: Optional[List[float]] = None) -> Tuple:
    """
    Discretize continuous state for tabular Q-learning.
    
    Args:
        state: 14-dimensional continuous state vector
        queue_bins: Bin edges for queue values
        duration_bins: Bin edges for phase duration
        
    Returns:
        Tuple of discrete state indices
    """
    if queue_bins is None:
        # Default bins: 0, 1-5, 6-10, 11-20, 21+
        queue_bins = [0, 1, 6, 11, 21, np.inf]
    
    if duration_bins is None:
        # Default bins: 0-10, 11-30, 31-60, 61+
        duration_bins = [0, 11, 31, 61, np.inf]
    
    discrete = []
    
    # Discretize queue values (indices 0-11)
    for i in range(12):
        bin_idx = np.digitize(state[i], queue_bins) - 1
        bin_idx = max(0, min(bin_idx, len(queue_bins) - 2))
        discrete.append(bin_idx)
    
    # Phase is already discrete (index 12)
    discrete.append(int(state[12]))
    
    # Discretize phase duration (index 13)
    duration_bin = np.digitize(state[13], duration_bins) - 1
    duration_bin = max(0, min(duration_bin, len(duration_bins) - 2))
    discrete.append(duration_bin)
    
    return tuple(discrete)


class QLearningAgent:
    """
    Tabular Q-Learning agent for traffic light control.
    
    Uses a dictionary-based Q-table for sparse state representation.
    """
    
    def __init__(self,
                 n_actions: int = 4,
                 alpha: float = 0.1,
                 gamma: float = 0.95,
                 epsilon: float = 1.0,
                 epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995,
                 queue_bins: Optional[List[float]] = None,
                 duration_bins: Optional[List[float]] = None):
        """
        Initialize Q-Learning agent.
        
        Args:
            n_actions: Number of actions
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_min: Minimum exploration rate
            epsilon_decay: Epsilon decay rate per episode
            queue_bins: Discretization bins for queues
            duration_bins: Discretization bins for duration
        """
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Discretization parameters
        self.queue_bins = queue_bins or [0, 1, 6, 11, 21, np.inf]
        self.duration_bins = duration_bins or [0, 11, 31, 61, np.inf]
        
        # Q-table as dictionary (sparse representation)
        self.Q: Dict[Tuple, np.ndarray] = {}
        
        # Statistics
        self.training_steps = 0
        self.episodes = 0
    
    def _discretize(self, state: np.ndarray) -> Tuple:
        """Discretize a state"""
        return discretize_state(state, self.queue_bins, self.duration_bins)
    
    def _get_q_values(self, state: Tuple) -> np.ndarray:
        """Get Q-values for a discrete state, initializing if necessary"""
        if state not in self.Q:
            self.Q[state] = np.zeros(self.n_actions)
        return self.Q[state]
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current continuous state
            training: Whether in training mode (affects exploration)
            
        Returns:
            Selected action
        """
        discrete_state = self._discretize(state)
        
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        # If state never seen during training, use heuristic fallback
        # instead of defaulting to action 0 (KEEP) which causes queue buildup
        if discrete_state not in self.Q:
            # Fallback heuristic: switch phase if queues are getting large
            max_queue = max(state[:12])
            if max_queue > 10:
                return 1  # NEXT - switch to help clear queues
            return 0  # KEEP if queues are manageable
        
        q_values = self.Q[discrete_state]
        return int(np.argmax(q_values))
    
    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool):
        """
        Update Q-value using Q-learning update rule.
        
        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        discrete_state = self._discretize(state)
        discrete_next_state = self._discretize(next_state)
        
        # Get current Q-value
        current_q = self._get_q_values(discrete_state)[action]
        
        # Calculate target
        if done:
            target = reward
        else:
            next_q_values = self._get_q_values(discrete_next_state)
            target = reward + self.gamma * np.max(next_q_values)
        
        # Update Q-value
        self.Q[discrete_state][action] += self.alpha * (target - current_q)
        
        self.training_steps += 1
    
    def decay_epsilon(self):
        """Decay epsilon after each episode"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.episodes += 1
    
    def get_q_table_size(self) -> int:
        """Get number of states in Q-table"""
        return len(self.Q)
    
    def get_statistics(self) -> Dict:
        """Get training statistics"""
        return {
            'training_steps': self.training_steps,
            'episodes': self.episodes,
            'epsilon': self.epsilon,
            'q_table_size': self.get_q_table_size()
        }
    
    def save(self, filepath: str):
        """Save agent to file"""
        data = {
            'Q': dict(self.Q),
            'n_actions': self.n_actions,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'queue_bins': self.queue_bins,
            'duration_bins': self.duration_bins,
            'training_steps': self.training_steps,
            'episodes': self.episodes
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str):
        """Load agent from file"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.Q = {k: v for k, v in data['Q'].items()}
        self.n_actions = data['n_actions']
        self.alpha = data['alpha']
        self.gamma = data['gamma']
        self.epsilon = data['epsilon']
        self.epsilon_min = data['epsilon_min']
        self.epsilon_decay = data['epsilon_decay']
        self.queue_bins = data['queue_bins']
        self.duration_bins = data['duration_bins']
        self.training_steps = data['training_steps']
        self.episodes = data['episodes']
    
    def get_name(self) -> str:
        """Get agent name"""
        return "Q-Learning"
    
    def get_config(self) -> Dict:
        """Get agent configuration"""
        return {
            'type': 'Q-Learning',
            'n_actions': self.n_actions,
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon_min': self.epsilon_min,
            'epsilon_decay': self.epsilon_decay,
            'queue_bins': self.queue_bins,
            'duration_bins': self.duration_bins
        }


if __name__ == "__main__":
    # Test Q-Learning agent
    agent = QLearningAgent()
    
    print("Testing Q-Learning Agent...")
    print(f"Config: {agent.get_config()}")
    
    # Simulate some training steps
    for i in range(100):
        state = np.random.rand(14) * 20  # Random state
        state[12] = np.random.randint(4)  # Random phase
        state[13] = np.random.rand() * 60  # Random duration
        
        action = agent.select_action(state)
        reward = -np.random.rand() * 10
        next_state = np.random.rand(14) * 20
        next_state[12] = np.random.randint(4)
        next_state[13] = np.random.rand() * 60
        done = i == 99
        
        agent.update(state, action, reward, next_state, done)
    
    agent.decay_epsilon()
    
    print(f"\nStatistics: {agent.get_statistics()}")
    
    # Test discretization
    test_state = np.array([10, 2, 3, 8, 1, 2, 15, 3, 2, 12, 2, 3, 0, 25], dtype=np.float32)
    discrete = discretize_state(test_state)
    print(f"\nTest state discretization:")
    print(f"  Continuous: {test_state}")
    print(f"  Discrete: {discrete}")
