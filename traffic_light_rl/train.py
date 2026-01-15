"""
Training script for traffic light RL agents.

Trains Q-Learning and DQN agents and compares with baselines.
"""

import os
import sys
import argparse
import numpy as np
from typing import Dict, Any, Optional
import yaml
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from traffic_light_rl.env import TrafficLightEnv
from traffic_light_rl.agents import QLearningAgent, DQNAgent
from traffic_light_rl.baselines import FixedTimeController, AdaptiveController
from traffic_light_rl.utils.metrics import evaluate_agent, compare_agents, print_comparison_table


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_q_learning(env, config: Dict[str, Any], verbose: bool = True) -> QLearningAgent:
    """
    Train Q-Learning agent.
    
    Args:
        env: Traffic light environment
        config: Configuration dictionary
        verbose: Whether to print progress
        
    Returns:
        Trained Q-Learning agent
    """
    q_config = config.get('q_learning', {})
    train_config = config.get('training', {})
    
    # Create agent
    agent = QLearningAgent(
        n_actions=4,
        alpha=q_config.get('alpha', 0.1),
        gamma=q_config.get('gamma', 0.95),
        epsilon=q_config.get('epsilon', 1.0),
        epsilon_min=q_config.get('epsilon_min', 0.01),
        epsilon_decay=q_config.get('epsilon_decay', 0.995)
    )
    
    n_episodes = train_config.get('n_episodes', 1000)
    eval_frequency = train_config.get('eval_frequency', 100)
    
    episode_rewards = []
    best_reward = float('-inf')
    
    if verbose:
        print("\n" + "="*60)
        print("Training Q-Learning Agent")
        print("="*60)
    
    start_time = time.time()
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
        
        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        
        if total_reward > best_reward:
            best_reward = total_reward
        
        # Progress update
        if verbose and (episode + 1) % eval_frequency == 0:
            avg_reward = np.mean(episode_rewards[-eval_frequency:])
            stats = agent.get_statistics()
            print(f"Episode {episode + 1}/{n_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Epsilon: {stats['epsilon']:.4f} | "
                  f"Q-table size: {stats['q_table_size']}")
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"\nTraining completed in {elapsed:.1f} seconds")
        print(f"Best episode reward: {best_reward:.2f}")
        print(f"Final Q-table size: {agent.get_q_table_size()}")
    
    return agent, episode_rewards


def train_dqn(env, config: Dict[str, Any], verbose: bool = True):
    """
    Train DQN agent.
    
    Args:
        env: Traffic light environment
        config: Configuration dictionary
        verbose: Whether to print progress
        
    Returns:
        Trained DQN agent and episode rewards
    """
    try:
        import torch
    except ImportError:
        print("PyTorch not available. Skipping DQN training.")
        return None, []
    
    dqn_config = config.get('dqn', {})
    train_config = config.get('training', {})
    
    # Create agent
    agent = DQNAgent(
        state_dim=14,
        action_dim=4,
        hidden_dims=tuple(dqn_config.get('hidden_dims', [128, 128, 64])),
        learning_rate=dqn_config.get('learning_rate', 0.001),
        gamma=dqn_config.get('gamma', 0.99),
        epsilon=dqn_config.get('epsilon', 1.0),
        epsilon_min=dqn_config.get('epsilon_min', 0.01),
        epsilon_decay=dqn_config.get('epsilon_decay', 0.995),
        buffer_capacity=dqn_config.get('buffer_capacity', 100000),
        batch_size=dqn_config.get('batch_size', 64),
        target_update_freq=dqn_config.get('target_update_freq', 100)
    )
    
    n_episodes = train_config.get('n_episodes', 1000)
    eval_frequency = train_config.get('eval_frequency', 100)
    
    episode_rewards = []
    best_reward = float('-inf')
    
    if verbose:
        print("\n" + "="*60)
        print("Training DQN Agent")
        print("="*60)
        print(f"Device: {agent.device}")
    
    start_time = time.time()
    
    for episode in range(n_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        
        while not done:
            action = agent.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            
            agent.store_transition(state, action, reward, next_state, done)
            agent.train()
            
            state = next_state
            total_reward += reward
            step += 1
        
        agent.decay_epsilon()
        episode_rewards.append(total_reward)
        
        if total_reward > best_reward:
            best_reward = total_reward
        
        # Progress update
        if verbose and (episode + 1) % eval_frequency == 0:
            avg_reward = np.mean(episode_rewards[-eval_frequency:])
            stats = agent.get_statistics()
            print(f"Episode {episode + 1}/{n_episodes} | "
                  f"Avg Reward: {avg_reward:.2f} | "
                  f"Epsilon: {stats['epsilon']:.4f} | "
                  f"Avg Loss: {stats['avg_loss']:.4f}")
    
    elapsed = time.time() - start_time
    
    if verbose:
        print(f"\nTraining completed in {elapsed:.1f} seconds")
        print(f"Best episode reward: {best_reward:.2f}")
    
    return agent, episode_rewards


def main():
    parser = argparse.ArgumentParser(description='Train traffic light RL agents')
    parser.add_argument('--config', type=str, default='traffic_light_rl/config/config.yaml',
                        help='Path to config file')
    parser.add_argument('--agent', type=str, choices=['qlearning', 'dqn', 'both'], 
                        default='both', help='Agent to train')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Number of training episodes (overrides config)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no-eval', action='store_true', help='Skip evaluation')
    parser.add_argument('--save', action='store_true', help='Save trained models')
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Load config
    if os.path.exists(args.config):
        config = load_config(args.config)
    else:
        print(f"Config file not found: {args.config}")
        print("Using default configuration")
        config = {}
    
    # Override episodes if specified
    if args.episodes:
        if 'training' not in config:
            config['training'] = {}
        config['training']['n_episodes'] = args.episodes
    
    # Create environment
    env_config = config.get('environment', {})
    env = TrafficLightEnv(
        max_queue=env_config.get('max_queue', 50),
        max_duration=env_config.get('max_duration', 120),
        max_steps=env_config.get('max_steps', 1000),
        seed=args.seed
    )
    
    print("\n" + "="*60)
    print("TRAFFIC LIGHT RL TRAINING")
    print("="*60)
    print(f"Environment: {env.observation_space.shape[0]}D state, {env.action_space.n} actions")
    print(f"Max steps per episode: {env.max_steps}")
    
    # Train agents
    trained_agents = {}
    rewards_history = {}
    
    if args.agent in ['qlearning', 'both']:
        q_agent, q_rewards = train_q_learning(env, config)
        trained_agents['Q-Learning'] = q_agent
        rewards_history['Q-Learning'] = q_rewards
    
    if args.agent in ['dqn', 'both']:
        dqn_agent, dqn_rewards = train_dqn(env, config)
        if dqn_agent is not None:
            trained_agents['DQN'] = dqn_agent
            rewards_history['DQN'] = dqn_rewards
    
    # Save models
    if args.save:
        os.makedirs('models', exist_ok=True)
        for name, agent in trained_agents.items():
            filepath = f"models/{name.lower().replace(' ', '_')}.pkl"
            agent.save(filepath)
            print(f"Saved {name} to {filepath}")
    
    # Evaluation
    if not args.no_eval:
        print("\n" + "="*60)
        print("EVALUATION")
        print("="*60)
        
        # Add baselines
        baseline_config = config.get('baselines', {})
        ft_config = baseline_config.get('fixed_time', {})
        adaptive_config = baseline_config.get('adaptive', {})
        
        all_agents = {
            'Fixed-Time': FixedTimeController(
                phase_durations=ft_config.get('phase_durations', {0: 30, 1: 15, 2: 30, 3: 15})
            ),
            'Adaptive': AdaptiveController(
                min_phase_duration=adaptive_config.get('min_phase_duration', 10),
                max_phase_duration=adaptive_config.get('max_phase_duration', 60)
            ),
            **trained_agents
        }
        
        eval_config = config.get('evaluation', {})
        n_eval_episodes = eval_config.get('n_episodes', 50)
        
        results = compare_agents(env, all_agents, n_episodes=n_eval_episodes, seed=args.seed)
        print_comparison_table(results)
    
    # Plot learning curves
    try:
        from traffic_light_rl.utils.visualization import plot_learning_curve
        
        for name, rewards in rewards_history.items():
            plot_learning_curve(rewards, title=f"{name} Learning Curve")
    except Exception as e:
        print(f"Could not plot learning curves: {e}")
    
    print("\nTraining complete!")
    

if __name__ == "__main__":
    main()
