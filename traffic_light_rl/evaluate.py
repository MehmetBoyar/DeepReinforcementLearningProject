"""
Evaluation script for traffic light RL agents.

Evaluates trained agents and compares with baselines.
"""

import os
import sys
import argparse
import numpy as np
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from traffic_light_rl.env import TrafficLightEnv
from traffic_light_rl.agents import QLearningAgent, DQNAgent
from traffic_light_rl.baselines import FixedTimeController, AdaptiveController
from traffic_light_rl.utils.metrics import evaluate_agent, compare_agents, print_comparison_table


def run_demo(env, agent, n_steps: int = 100, agent_name: str = "Agent"):
    """
    Run a demo of an agent controlling the traffic light.
    
    Args:
        env: Traffic light environment
        agent: Agent or controller
        n_steps: Number of steps to run
        agent_name: Name of the agent for display
    """
    print(f"\n{'='*60}")
    print(f"DEMO: {agent_name}")
    print("="*60)
    
    state = env.reset()
    total_reward = 0
    
    for step in range(n_steps):
        # Get action
        if hasattr(agent, 'select_action'):
            action = agent.select_action(state, training=False)
        else:
            action = agent.get_action(state)
        
        # Take step
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        # Render
        if step % 10 == 0 or done:
            env.render()
            print(f"Action: {env.ACTION_NAMES[action]} | Reward: {reward:.2f}")
        
        if done:
            break
        
        state = next_state
    
    print(f"\n{'-'*60}")
    print(f"Demo completed: Total reward = {total_reward:.2f}")
    print(f"Vehicles passed: {info['total_vehicles_passed']}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate traffic light RL agents')
    parser.add_argument('--agent', type=str, choices=['qlearning', 'dqn', 'fixed', 'adaptive', 'all'],
                        default='all', help='Agent to evaluate')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to saved model')
    parser.add_argument('--episodes', type=int, default=50,
                        help='Number of evaluation episodes')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--demo', action='store_true', help='Run interactive demo')
    parser.add_argument('--demo-steps', type=int, default=100,
                        help='Number of steps for demo')
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create environment
    env = TrafficLightEnv(seed=args.seed)
    
    print("\n" + "="*60)
    print("TRAFFIC LIGHT RL EVALUATION")
    print("="*60)
    
    # Create agents
    agents = {}
    
    if args.agent in ['fixed', 'all']:
        agents['Fixed-Time'] = FixedTimeController()
    
    if args.agent in ['adaptive', 'all']:
        agents['Adaptive'] = AdaptiveController()
    
    if args.agent in ['qlearning', 'all']:
        q_agent = QLearningAgent()
        if args.model_path and 'qlearning' in args.model_path.lower():
            try:
                q_agent.load(args.model_path)
                print(f"Loaded Q-Learning model from {args.model_path}")
            except Exception as e:
                print(f"Could not load Q-Learning model: {e}")
        elif os.path.exists('models/q-learning.pkl'):
            try:
                q_agent.load('models/q-learning.pkl')
                print("Loaded Q-Learning model from models/q-learning.pkl")
            except:
                pass
        agents['Q-Learning'] = q_agent
    
    if args.agent in ['dqn', 'all']:
        try:
            dqn_agent = DQNAgent()
            if args.model_path and 'dqn' in args.model_path.lower():
                try:
                    dqn_agent.load(args.model_path)
                    print(f"Loaded DQN model from {args.model_path}")
                except Exception as e:
                    print(f"Could not load DQN model: {e}")
            elif os.path.exists('models/dqn.pkl'):
                try:
                    dqn_agent.load('models/dqn.pkl')
                    print("Loaded DQN model from models/dqn.pkl")
                except:
                    pass
            agents['DQN'] = dqn_agent
        except ImportError:
            print("PyTorch not available. Skipping DQN.")
    
    # Run demo if requested
    if args.demo:
        for name, agent in agents.items():
            run_demo(env, agent, n_steps=args.demo_steps, agent_name=name)
    
    # Run evaluation
    print("\n" + "="*60)
    print(f"EVALUATION ({args.episodes} episodes each)")
    print("="*60)
    
    results = compare_agents(env, agents, n_episodes=args.episodes, seed=args.seed)
    print_comparison_table(results)
    
    # Try to plot results
    try:
        from traffic_light_rl.utils.visualization import plot_multi_comparison
        plot_multi_comparison(results)
    except Exception as e:
        print(f"Could not plot results: {e}")
    
    # Detailed results
    print("\n" + "="*60)
    print("DETAILED RESULTS")
    print("="*60)
    
    for name, res in results.items():
        print(f"\n{name}:")
        print(f"  Mean reward: {res['mean_reward']:.2f} ± {res['std_reward']:.2f}")
        print(f"  Mean throughput: {res['mean_throughput']:.0f} ± {res['std_throughput']:.0f}")
        print(f"  Mean wait time: {res['mean_wait_time']:.0f} ± {res['std_wait_time']:.0f}")
        print(f"  Mean queue length: {res['mean_queue_length']:.2f}")
        print(f"  Mean max queue: {res['mean_max_queue']:.2f}")
        print(f"  Mean phase switches: {res['mean_switches']:.1f}")
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
