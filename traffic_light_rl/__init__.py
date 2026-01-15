"""
Traffic Light Control with Deep Reinforcement Learning

A complete implementation of traffic signal control using:
- Tabular Q-Learning
- Deep Q-Network (DQN)

Compared against baseline controllers:
- Fixed-time control
- Adaptive control

Usage:
    # Train agents
    python -m traffic_light_rl.train --episodes 1000
    
    # Evaluate agents
    python -m traffic_light_rl.evaluate --episodes 50
"""

from .env import TrafficLightEnv, PhaseManager
from .agents import QLearningAgent, DQNAgent
from .baselines import FixedTimeController, AdaptiveController

__version__ = '1.0.0'
__author__ = 'TUM Deep RL Project'

__all__ = [
    'TrafficLightEnv',
    'PhaseManager',
    'QLearningAgent',
    'DQNAgent',
    'FixedTimeController',
    'AdaptiveController'
]
