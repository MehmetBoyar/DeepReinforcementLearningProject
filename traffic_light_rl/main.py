"""
=============================================================================
TRAFFIC LIGHT CONTROL WITH DEEP REINFORCEMENT LEARNING
=============================================================================

Main training and evaluation script for Spyder IDE.

HOW TO RUN:
1. Open this file in Spyder
2. Set working directory to the Project folder
3. Press F5 to run

CONFIGURATION:
- Modify the settings in the "CONFIGURATION" section below
- Set TRAIN_DQN = True if you have PyTorch installed
"""

# Fix OpenMP library conflict on Windows (must be before other imports)
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import numpy as np
import time

# =============================================================================
# CONFIGURATION - Modify these settings
# =============================================================================

# Training settings
N_EPISODES = 1000         # Training episodes (more for better Q-table coverage)
EVAL_EPISODES = 20        # Evaluation episodes
SEED = 42                 # Random seed for reproducibility

# What to train
TRAIN_QLEARNING = True    # Train Q-Learning agent
TRAIN_DQN = True          # Train DQN agent (requires PyTorch)

# Display settings
SHOW_PLOTS = True         # Show matplotlib plots
VERBOSE = True            # Print progress during training
DEMO_STEPS = 20           # Steps to show in final demo

# =============================================================================
# Setup - Don't modify below unless you know what you're doing
# =============================================================================

# Set working directory to project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
os.chdir(project_dir)
sys.path.insert(0, project_dir)

print("="*70)
print("TRAFFIC LIGHT CONTROL WITH DEEP REINFORCEMENT LEARNING")
print("="*70)
print(f"Working directory: {os.getcwd()}")

# Set random seed
np.random.seed(SEED)

# =============================================================================
# Import modules
# =============================================================================

from traffic_light_rl.env import TrafficLightEnv
from traffic_light_rl.agents import QLearningAgent
from traffic_light_rl.baselines import FixedTimeController, AdaptiveController
from traffic_light_rl.utils.metrics import compare_agents, print_comparison_table

# Check if PyTorch is available for DQN
PYTORCH_AVAILABLE = False
if TRAIN_DQN:
    try:
        import torch
        from traffic_light_rl.agents import DQNAgent
        PYTORCH_AVAILABLE = True
        print(f"PyTorch version: {torch.__version__}")
        print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        torch.manual_seed(SEED)
    except ImportError:
        print("WARNING: PyTorch not installed. DQN training disabled.")
        print("Install with: conda install pytorch cpuonly -c pytorch")
        TRAIN_DQN = False

# =============================================================================
# Create Environment
# =============================================================================

print("\n" + "-"*70)
print("ENVIRONMENT SETUP")
print("-"*70)

env = TrafficLightEnv(seed=SEED)

print(f"State space: {env.observation_space.shape[0]} dimensions")
print(f"  - 12 queue lengths (vehicles waiting per movement)")
print(f"  - 1 current phase (0-3)")
print(f"  - 1 phase duration (seconds in current phase)")
print(f"Action space: {env.action_space.n} actions")
print(f"  - 0: KEEP (stay in current phase)")
print(f"  - 1: NEXT (move to next phase)")
print(f"  - 2: SKIP_TO_NS (jump to N-S through)")
print(f"  - 3: SKIP_TO_EW (jump to E-W through)")
print(f"Max steps per episode: {env.max_steps}")

# Quick environment test
print("\nTesting environment...")
state = env.reset()
for _ in range(5):
    action = env.action_space.sample()
    state, reward, done, info = env.step(action)
print("Environment test passed!")

# =============================================================================
# Storage for results
# =============================================================================

trained_agents = {}
training_rewards = {}
training_metrics = {}  # Store additional metrics: queue lengths, throughput, wait time

# =============================================================================
# Train Q-Learning
# =============================================================================

if TRAIN_QLEARNING:
    print("\n" + "="*70)
    print("TRAINING: Q-LEARNING")
    print("="*70)
    
    q_agent = QLearningAgent(
        n_actions=4,
        alpha=0.2,           # Higher learning rate for faster learning
        gamma=0.95,          # Discount factor
        epsilon=1.0,         # Initial exploration rate
        epsilon_min=0.05,    # Slightly higher min for more exploration
        epsilon_decay=0.997, # Slower decay for more exploration
        queue_bins=[0, 5, 15, np.inf],      # Coarser bins (3 levels) for better coverage
        duration_bins=[0, 20, 60, np.inf]   # Coarser bins (3 levels)
    )
    
    q_rewards = []
    q_avg_queues = []      # Average queue per episode
    q_throughputs = []     # Vehicles passed per episode
    q_wait_times = []      # Total wait time per episode
    # Reward components tracking
    q_queue_penalties = []
    q_throughput_rewards = []
    q_switch_penalties = []
    q_fairness_penalties = []
    # Action distribution tracking
    q_action_counts = []   # Action distribution per episode
    start_time = time.time()
    
    for episode in range(N_EPISODES):
        state = env.reset()
        total_reward = 0
        episode_queues = []
        episode_actions = {0: 0, 1: 0, 2: 0, 3: 0}
        # Reward components for this episode
        ep_queue_pen = 0
        ep_throughput_rew = 0
        ep_switch_pen = 0
        ep_fairness_pen = 0
        done = False
        
        while not done:
            action = q_agent.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            q_agent.update(state, action, reward, next_state, done)
            episode_queues.append(info['total_queue'])
            episode_actions[action] += 1
            # Accumulate reward components
            rc = info['reward_components']
            ep_queue_pen += rc['queue_penalty']
            ep_throughput_rew += rc['throughput_reward']
            ep_switch_pen += rc['switch_penalty']
            ep_fairness_pen += rc['fairness_penalty']
            state = next_state
            total_reward += reward
        
        q_agent.decay_epsilon()
        q_rewards.append(total_reward)
        q_avg_queues.append(np.mean(episode_queues))
        q_throughputs.append(info['total_vehicles_passed'])
        q_wait_times.append(info['total_wait_time'])
        q_queue_penalties.append(ep_queue_pen)
        q_throughput_rewards.append(ep_throughput_rew)
        q_switch_penalties.append(ep_switch_pen)
        q_fairness_penalties.append(ep_fairness_pen)
        q_action_counts.append(episode_actions.copy())
        
        if VERBOSE and (episode + 1) % 50 == 0:
            avg = np.mean(q_rewards[-50:])
            avg_queue = np.mean(q_avg_queues[-50:])
            print(f"Episode {episode+1:4d}/{N_EPISODES} | "
                  f"Avg Reward: {avg:8.2f} | "
                  f"Avg Queue: {avg_queue:5.1f} | "
                  f"Epsilon: {q_agent.epsilon:.4f} | "
                  f"Q-table: {q_agent.get_q_table_size()} states")
    
    elapsed = time.time() - start_time
    print(f"\nQ-Learning training completed in {elapsed:.1f}s")
    print(f"Final Q-table size: {q_agent.get_q_table_size()} states")
    
    trained_agents['Q-Learning'] = q_agent
    training_rewards['Q-Learning'] = q_rewards
    training_metrics['Q-Learning'] = {
        'avg_queues': q_avg_queues,
        'throughputs': q_throughputs,
        'wait_times': q_wait_times,
        'queue_penalties': q_queue_penalties,
        'throughput_rewards': q_throughput_rewards,
        'switch_penalties': q_switch_penalties,
        'fairness_penalties': q_fairness_penalties,
        'action_counts': q_action_counts
    }

# =============================================================================
# Train DQN
# =============================================================================

if TRAIN_DQN and PYTORCH_AVAILABLE:
    print("\n" + "="*70)
    print("TRAINING: DEEP Q-NETWORK (DQN)")
    print("="*70)
    
    # Simplified DQN configuration for faster training
    dqn_agent = DQNAgent(
        state_dim=14,
        action_dim=4,
        hidden_dims=(64, 64),         # Smaller network (was 128, 128, 64)
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        buffer_capacity=10000,        # Smaller buffer (was 100000)
        batch_size=32,                # Smaller batch (was 64)
        target_update_freq=200        # Less frequent updates (was 100)
    )
    
    TRAIN_FREQ = 4  # Train every N steps (speeds up significantly)
    
    dqn_rewards = []
    dqn_losses = []
    dqn_avg_queues = []    # Average queue per episode
    dqn_throughputs = []   # Vehicles passed per episode
    dqn_wait_times = []    # Total wait time per episode
    # Reward components tracking
    dqn_queue_penalties = []
    dqn_throughput_rewards = []
    dqn_switch_penalties = []
    dqn_fairness_penalties = []
    # Action distribution tracking
    dqn_action_counts = []
    total_steps = 0        # Track total steps for training frequency
    start_time = time.time()
    
    for episode in range(N_EPISODES):
        state = env.reset()
        total_reward = 0
        episode_losses = []
        episode_queues = []
        episode_actions = {0: 0, 1: 0, 2: 0, 3: 0}
        # Reward components for this episode
        ep_queue_pen = 0
        ep_throughput_rew = 0
        ep_switch_pen = 0
        ep_fairness_pen = 0
        done = False
        
        while not done:
            action = dqn_agent.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)
            
            dqn_agent.store_transition(state, action, reward, next_state, done)
            total_steps += 1
            
            # Only train every TRAIN_FREQ steps (much faster!)
            if total_steps % TRAIN_FREQ == 0:
                loss = dqn_agent.train()
                if loss is not None:
                    episode_losses.append(loss)
            
            episode_queues.append(info['total_queue'])
            episode_actions[action] += 1
            # Accumulate reward components
            rc = info['reward_components']
            ep_queue_pen += rc['queue_penalty']
            ep_throughput_rew += rc['throughput_reward']
            ep_switch_pen += rc['switch_penalty']
            ep_fairness_pen += rc['fairness_penalty']
            
            state = next_state
            total_reward += reward
        
        dqn_agent.decay_epsilon()
        dqn_rewards.append(total_reward)
        dqn_avg_queues.append(np.mean(episode_queues))
        dqn_throughputs.append(info['total_vehicles_passed'])
        dqn_wait_times.append(info['total_wait_time'])
        dqn_queue_penalties.append(ep_queue_pen)
        dqn_throughput_rewards.append(ep_throughput_rew)
        dqn_switch_penalties.append(ep_switch_pen)
        dqn_fairness_penalties.append(ep_fairness_pen)
        dqn_action_counts.append(episode_actions.copy())
        if episode_losses:
            dqn_losses.append(np.mean(episode_losses))
        
        if VERBOSE and (episode + 1) % 50 == 0:
            avg = np.mean(dqn_rewards[-50:])
            avg_queue = np.mean(dqn_avg_queues[-50:])
            avg_loss = np.mean(dqn_losses[-50:]) if dqn_losses else 0
            print(f"Episode {episode+1:4d}/{N_EPISODES} | "
                  f"Avg Reward: {avg:8.2f} | "
                  f"Avg Queue: {avg_queue:5.1f} | "
                  f"Loss: {avg_loss:.4f}")
    
    elapsed = time.time() - start_time
    print(f"\nDQN training completed in {elapsed:.1f}s")
    
    trained_agents['DQN'] = dqn_agent
    training_rewards['DQN'] = dqn_rewards
    training_metrics['DQN'] = {
        'avg_queues': dqn_avg_queues,
        'throughputs': dqn_throughputs,
        'wait_times': dqn_wait_times,
        'losses': dqn_losses,
        'queue_penalties': dqn_queue_penalties,
        'throughput_rewards': dqn_throughput_rewards,
        'switch_penalties': dqn_switch_penalties,
        'fairness_penalties': dqn_fairness_penalties,
        'action_counts': dqn_action_counts
    }

# =============================================================================
# Evaluation
# =============================================================================

print("\n" + "="*70)
print("EVALUATION")
print("="*70)

# Create all agents for comparison
all_agents = {
    'Fixed-Time': FixedTimeController(
        phase_durations={0: 30, 1: 15, 2: 30, 3: 15}
    ),
    'Adaptive': AdaptiveController(
        min_phase_duration=10,
        max_phase_duration=60
    ),
    **trained_agents
}

# Run evaluation
print(f"\nEvaluating {len(all_agents)} controllers over {EVAL_EPISODES} episodes each...")
results = compare_agents(env, all_agents, n_episodes=EVAL_EPISODES, seed=SEED)

# Print results table
print_comparison_table(results)

# =============================================================================
# Visualization
# =============================================================================

if SHOW_PLOTS:
    try:
        import matplotlib.pyplot as plt
        
        # Create comprehensive visualization with multiple subplots
        n_trained = len(training_rewards)
        colors = {'Q-Learning': 'green', 'DQN': 'blue'}
        window = 50
        
        # Figure 1: Training Progress (2 rows x 2 cols per agent + comparison)
        fig1, axes1 = plt.subplots(2, 2 + n_trained, figsize=(5*(2+n_trained), 8))
        fig1.suptitle('Training Progress & Agent Comparison', fontsize=14, fontweight='bold')
        
        # Plot learning curves (rewards) - Row 1
        for idx, (name, rewards) in enumerate(training_rewards.items()):
            ax = axes1[0, idx]
            ax.plot(rewards, alpha=0.3, color=colors.get(name, 'blue'))
            
            if len(rewards) >= window:
                smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(rewards)), smoothed,
                       color=colors.get(name, 'blue'), linewidth=2)
            
            ax.set_title(f'{name}: Episode Reward')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Total Reward')
            ax.grid(True, alpha=0.3)
        
        # Plot average queue length during training - Row 1
        for idx, (name, metrics) in enumerate(training_metrics.items()):
            ax = axes1[1, idx]
            queues = metrics['avg_queues']
            ax.plot(queues, alpha=0.3, color=colors.get(name, 'blue'))
            
            if len(queues) >= window:
                smoothed = np.convolve(queues, np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(queues)), smoothed,
                       color=colors.get(name, 'blue'), linewidth=2)
            
            ax.set_title(f'{name}: Avg Queue Length')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Vehicles Waiting')
            ax.grid(True, alpha=0.3)
        
        # Comparison bar charts - Column 3 & 4
        agent_names = list(results.keys())
        bar_colors = ['gray', 'orange'] + [colors.get(n, 'purple') for n in agent_names[2:]]
        
        # Reward comparison (Row 1, Col 3)
        ax = axes1[0, n_trained]
        rewards_vals = [results[a]['mean_reward'] for a in agent_names]
        errors = [results[a]['std_reward'] for a in agent_names]
        bars = ax.bar(agent_names, rewards_vals, yerr=errors, capsize=5,
                     color=bar_colors[:len(agent_names)], alpha=0.8)
        ax.set_title('Mean Reward')
        ax.set_ylabel('Reward')
        ax.tick_params(axis='x', rotation=15)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, rewards_vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val:.0f}', ha='center', va='bottom', fontsize=8)
        
        # Throughput comparison (Row 1, Col 4)
        ax = axes1[0, n_trained + 1]
        throughputs = [results[a]['mean_throughput'] for a in agent_names]
        bars = ax.bar(agent_names, throughputs, color=bar_colors[:len(agent_names)], alpha=0.8)
        ax.set_title('Vehicles Passed')
        ax.set_ylabel('Count')
        ax.tick_params(axis='x', rotation=15)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, throughputs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val:.0f}', ha='center', va='bottom', fontsize=8)
        
        # Average Queue comparison (Row 2, Col 3)
        ax = axes1[1, n_trained]
        avg_queues = [results[a]['mean_queue_length'] for a in agent_names]
        bars = ax.bar(agent_names, avg_queues, color=bar_colors[:len(agent_names)], alpha=0.8)
        ax.set_title('Avg Queue Length')
        ax.set_ylabel('Vehicles')
        ax.tick_params(axis='x', rotation=15)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, avg_queues):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val:.1f}', ha='center', va='bottom', fontsize=8)
        
        # Wait Time comparison (Row 2, Col 4)
        ax = axes1[1, n_trained + 1]
        wait_times = [results[a]['mean_wait_time'] for a in agent_names]
        bars = ax.bar(agent_names, wait_times, color=bar_colors[:len(agent_names)], alpha=0.8)
        ax.set_title('Total Wait Time')
        ax.set_ylabel('Time Steps')
        ax.tick_params(axis='x', rotation=15)
        ax.grid(True, alpha=0.3, axis='y')
        for bar, val in zip(bars, wait_times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{val:.0f}', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig('training_results.png', dpi=150, bbox_inches='tight')
        print("\nPlot saved to 'training_results.png'")
        plt.show()
        
        # Figure 2: Training Metrics Over Time (throughput, wait time)
        if training_metrics:
            fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
            fig2.suptitle('Training Metrics Over Time', fontsize=14, fontweight='bold')
            
            # Throughput over training
            ax = axes2[0]
            for name, metrics in training_metrics.items():
                throughputs = metrics['throughputs']
                if len(throughputs) >= window:
                    smoothed = np.convolve(throughputs, np.ones(window)/window, mode='valid')
                    ax.plot(range(window-1, len(throughputs)), smoothed,
                           color=colors.get(name, 'purple'), linewidth=2, label=name)
            ax.set_title('Throughput (Vehicles Passed per Episode)')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Vehicles')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Wait time over training
            ax = axes2[1]
            for name, metrics in training_metrics.items():
                wait_times = metrics['wait_times']
                if len(wait_times) >= window:
                    smoothed = np.convolve(wait_times, np.ones(window)/window, mode='valid')
                    ax.plot(range(window-1, len(wait_times)), smoothed,
                           color=colors.get(name, 'purple'), linewidth=2, label=name)
            ax.set_title('Total Wait Time per Episode')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Time Steps')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('training_metrics.png', dpi=150, bbox_inches='tight')
            print("Plot saved to 'training_metrics.png'")
            plt.show()
        
        # Figure 3: Reward Components Analysis
        if training_metrics and any('queue_penalties' in m for m in training_metrics.values()):
            fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
            fig3.suptitle('Reward Function Components Analysis', fontsize=14, fontweight='bold')
            
            component_names = ['queue_penalties', 'throughput_rewards', 'switch_penalties', 'fairness_penalties']
            component_titles = ['Queue Penalty (negative)', 'Throughput Reward (positive)', 
                              'Switch Penalty (negative)', 'Fairness Penalty (negative)']
            component_colors = ['red', 'green', 'orange', 'purple']
            
            for idx, (comp_name, comp_title, comp_color) in enumerate(zip(component_names, component_titles, component_colors)):
                ax = axes3[idx // 2, idx % 2]
                for name, metrics in training_metrics.items():
                    if comp_name in metrics:
                        data = metrics[comp_name]
                        if len(data) >= window:
                            smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
                            ax.plot(range(window-1, len(data)), smoothed,
                                   color=colors.get(name, 'blue'), linewidth=2, label=name)
                ax.set_title(comp_title)
                ax.set_xlabel('Episode')
                ax.set_ylabel('Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('reward_components.png', dpi=150, bbox_inches='tight')
            print("Plot saved to 'reward_components.png'")
            plt.show()
        
        # Figure 4: Action Distribution Comparison
        if training_metrics and any('action_counts' in m for m in training_metrics.values()):
            fig4, axes4 = plt.subplots(1, len(training_metrics) + 1, figsize=(5 * (len(training_metrics) + 1), 5))
            fig4.suptitle('Action Distribution Analysis', fontsize=14, fontweight='bold')
            
            action_names = ['KEEP', 'NEXT', 'SKIP_NS', 'SKIP_EW']
            action_colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6']
            
            # Plot action distribution over training for each agent
            for idx, (name, metrics) in enumerate(training_metrics.items()):
                ax = axes4[idx] if len(training_metrics) > 1 else axes4[0]
                if 'action_counts' in metrics:
                    action_counts = metrics['action_counts']
                    # Calculate cumulative action percentages over windows
                    n_episodes = len(action_counts)
                    action_pcts = {i: [] for i in range(4)}
                    
                    for ep in range(window, n_episodes):
                        window_counts = {i: 0 for i in range(4)}
                        for ac in action_counts[ep-window:ep]:
                            for a, c in ac.items():
                                window_counts[a] += c
                        total = sum(window_counts.values())
                        if total > 0:
                            for a in range(4):
                                action_pcts[a].append(100 * window_counts[a] / total)
                    
                    x = range(window, n_episodes)
                    ax.stackplot(x, [action_pcts[i] for i in range(4)], 
                                labels=action_names, colors=action_colors, alpha=0.8)
                    ax.set_title(f'{name}: Action Distribution Over Training')
                    ax.set_xlabel('Episode')
                    ax.set_ylabel('Percentage (%)')
                    ax.set_ylim(0, 100)
                    ax.legend(loc='upper right')
                    ax.grid(True, alpha=0.3)
            
            # Final action distribution comparison (pie charts or bar)
            ax = axes4[-1] if len(training_metrics) > 1 else axes4[1]
            x_pos = np.arange(4)
            width = 0.35
            
            bars_list = []
            for idx, (name, metrics) in enumerate(training_metrics.items()):
                if 'action_counts' in metrics:
                    # Get final 100 episodes action distribution
                    final_counts = {i: 0 for i in range(4)}
                    for ac in metrics['action_counts'][-100:]:
                        for a, c in ac.items():
                            final_counts[a] += c
                    total = sum(final_counts.values())
                    pcts = [100 * final_counts[i] / total if total > 0 else 0 for i in range(4)]
                    offset = width * (idx - 0.5)
                    bars = ax.bar(x_pos + offset, pcts, width, label=name, 
                                 color=colors.get(name, 'gray'), alpha=0.8)
                    bars_list.append(bars)
            
            ax.set_title('Final Action Distribution (Last 100 Episodes)')
            ax.set_xlabel('Action')
            ax.set_ylabel('Percentage (%)')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(action_names)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig('action_distribution.png', dpi=150, bbox_inches='tight')
            print("Plot saved to 'action_distribution.png'")
            plt.show()
        
    except ImportError:
        print("\nMatplotlib not available - skipping plots")
    except Exception as e:
        print(f"\nCould not create plots: {e}")

# =============================================================================
# Demo
# =============================================================================

print("\n" + "="*70)
print("DEMO: Best Trained Agent")
print("="*70)

# Pick the best trained agent
if trained_agents:
    best_name = max(trained_agents.keys(), 
                    key=lambda x: results[x]['mean_reward'])
    best_agent = trained_agents[best_name]
    print(f"Showing {best_name} agent in action:")
    
    state = env.reset()
    total_reward = 0
    
    for step in range(DEMO_STEPS):
        action = best_agent.select_action(state, training=False)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        if step % 5 == 0:
            env.render()
            print(f"Step {step}: Action = {env.ACTION_NAMES[action]}")
        
        if done:
            break
        state = next_state
    
    print(f"\nDemo finished! Reward: {total_reward:.2f}, "
          f"Vehicles passed: {info['total_vehicles_passed']}")

# =============================================================================
# Summary
# =============================================================================

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print("\nFinal Rankings (by mean reward):")
ranked = sorted(results.items(), key=lambda x: x[1]['mean_reward'], reverse=True)
for i, (name, res) in enumerate(ranked, 1):
    print(f"  {i}. {name:15s}: {res['mean_reward']:8.2f} ± {res['std_reward']:.2f}")

print("\n" + "="*70)
print("TRAINING COMPLETE!")
print("="*70)
