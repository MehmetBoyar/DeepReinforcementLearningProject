import numpy as np
import pickle
import os
from traffic_rl.agents.base import BaseAgent
from traffic_rl.config import AgentConfig

class QLearningAgent(BaseAgent):
    def __init__(self, action_dim, config: AgentConfig):
        self.action_dim = action_dim
        self.config = config
        
        # Exploration parameters
        self.epsilon = config.epsilon_start
        self.epsilon_min = config.epsilon_min
        self.epsilon_decay = config.epsilon_decay
        self.alpha = config.alpha
        self.gamma = config.gamma
        
        # The Q-Table: Dictionary mapping state_key -> np.array([q_val_0, q_val_1, ...])
        self.Q = {} 
        
        # --- PARITY WITH OLD CODE ---
        # Specific bins from your original config.yaml to ensure fair comparison
        # Queue Bins: [0, 1, 6, 11, 21, inf]
        self.queue_bins = [0, 1, 6, 11, 21]
        # Duration Bins: [0, 11, 31, 61, inf]
        self.dur_bins = [0, 11, 31, 61]

    def _get_state_key(self, state):
        """
        Discretizes the continuous state into a hashable tuple key.
        State indices: [0-11]=Queues, [12]=Phase, [13]=Duration
        """
        # Discretize queues
        queues = [np.digitize(q, self.queue_bins) for q in state[:12]]
        
        # Phase is already discrete
        phase = int(state[12])
        
        # Discretize duration
        duration = np.digitize(state[13], self.dur_bins)
        
        return tuple(queues + [phase, duration])

    def act(self, state, training=True):
        state_key = self._get_state_key(state)
        
        # Epsilon-Greedy Exploration
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        # Exploitation
        if state_key not in self.Q:
            # Fallback heuristic for unseen states: 
            # If total queue is huge, try switching (Action 1), else Keep (Action 0)
            # This is a common trick to prevent getting stuck in unseen states
            total_q = np.sum(state[:12])
            return 1 if total_q > 30 else 0
            
        return int(np.argmax(self.Q[state_key]))

    def update(self, state, action, reward, next_state, done):
        """
        Q(s,a) = Q(s,a) + alpha * [r + gamma * max(Q(s',a')) - Q(s,a)]
        """
        state_key = self._get_state_key(state)
        next_key = self._get_state_key(next_state)
        
        # Initialize keys if seen for the first time
        if state_key not in self.Q:
            self.Q[state_key] = np.zeros(self.action_dim)
        if next_key not in self.Q:
            self.Q[next_key] = np.zeros(self.action_dim)
            
        current_q = self.Q[state_key][action]
        
        # Target calculation
        if done:
            target = reward
        else:
            max_next_q = np.max(self.Q[next_key])
            target = reward + self.gamma * max_next_q
        
        # Update Rule
        self.Q[state_key][action] += self.alpha * (target - current_q)
        
        # Decay Epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Return TD-error (loss proxy)
        return abs(target - current_q)

    def save(self, path):
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'Q': self.Q, 
                'epsilon': self.epsilon,
                # Save config params to ensure compatibility on load
                'bins': (self.queue_bins, self.dur_bins) 
            }, f)

    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.Q = data['Q']
            self.epsilon = data.get('epsilon', self.epsilon_min)
            # Optional: warn if bins don't match, but usually we just trust the saved model