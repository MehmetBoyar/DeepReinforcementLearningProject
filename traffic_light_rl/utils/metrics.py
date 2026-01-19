"""
Evaluation metrics for traffic light control.

Provides functions to evaluate and compare different controllers.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    
    # Episode-level metrics
    total_reward: float = 0.0
    total_vehicles_passed: int = 0
    total_wait_time: float = 0.0
    total_steps: int = 0
    phase_switches: int = 0
    
    # Running statistics
    rewards: List[float] = field(default_factory=list)
    queue_lengths: List[float] = field(default_factory=list)
    throughputs: List[int] = field(default_factory=list)
    
    def update(self, reward: float, info: Dict[str, Any]):
        """Update metrics with step information"""
        self.total_reward += reward
        self.rewards.append(reward)
        
        if 'total_queue' in info:
            self.queue_lengths.append(info['total_queue'])
        if 'vehicles_passed' in info:
            self.throughputs.append(info['vehicles_passed'])
        if 'total_vehicles_passed' in info:
            self.total_vehicles_passed = info['total_vehicles_passed']
        if 'total_wait_time' in info:
            self.total_wait_time = info['total_wait_time']
        if 'phase_switches' in info:
            self.phase_switches = info['phase_switches']
        
        self.total_steps += 1

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics"""
        passed_count = max(1, self.total_vehicles_passed)
        avg_wait_per_vehicle = self.total_wait_time / passed_count
        
        return {
            'total_reward': self.total_reward,
            'avg_reward': np.mean(self.rewards) if self.rewards else 0.0,
            'total_vehicles_passed': self.total_vehicles_passed,
            'total_wait_time': self.total_wait_time,
            'avg_wait_per_vehicle': avg_wait_per_vehicle,  
            'avg_queue_length': np.mean(self.queue_lengths) if self.queue_lengths else 0.0,
            'max_queue_length': max(self.queue_lengths) if self.queue_lengths else 0.0,
            'avg_throughput': np.mean(self.throughputs) if self.throughputs else 0.0,
            'total_steps': self.total_steps,
            'phase_switches': self.phase_switches,
            'switches_per_step': self.phase_switches / max(1, self.total_steps)
        }  
    
    def reset(self):
        """Reset all metrics"""
        self.total_reward = 0.0
        self.total_vehicles_passed = 0
        self.total_wait_time = 0.0
        self.total_steps = 0
        self.phase_switches = 0
        self.rewards.clear()
        self.queue_lengths.clear()
        self.throughputs.clear()


def evaluate_agent(env, agent, n_episodes: int = 10, 
                   render: bool = False, seed: Optional[int] = None) -> Dict[str, Any]:
    """
    Evaluate an agent over multiple episodes.
    
    Args:
        env: Traffic light environment
        agent: Agent or controller to evaluate
        n_episodes: Number of episodes to run
        render: Whether to render the environment
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary of evaluation results
    """
    if seed is not None:
        np.random.seed(seed)
    
    all_metrics = []
    episode_rewards = []
    episode_throughputs = []
    episode_wait_times = []
    
    for episode in range(n_episodes):
        metrics = EvaluationMetrics()
        state = env.reset()
        done = False
        
        while not done:
            # Get action from agent/controller
            if hasattr(agent, 'select_action'):
                action = agent.select_action(state, training=False)
            else:
                action = agent.get_action(state)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            metrics.update(reward, info)
            
            if render:
                env.render()
            
            state = next_state
        
        summary = metrics.get_summary()
        all_metrics.append(summary)
        episode_rewards.append(summary['total_reward'])
        episode_throughputs.append(summary['total_vehicles_passed'])
        episode_wait_times.append(summary['total_wait_time'])
    
    # Aggregate results
    results = {
        'n_episodes': n_episodes,
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'mean_throughput': np.mean(episode_throughputs),
        'std_throughput': np.std(episode_throughputs),
        'mean_wait_time': np.mean(episode_wait_times),
        'std_wait_time': np.std(episode_wait_times),
        'mean_wait_per_vehicle': np.mean([m['avg_wait_per_vehicle'] for m in all_metrics]),
        'mean_queue_length': np.mean([m['avg_queue_length'] for m in all_metrics]),
        'mean_max_queue': np.mean([m['max_queue_length'] for m in all_metrics]),
        'mean_switches': np.mean([m['phase_switches'] for m in all_metrics]),
        'episode_metrics': all_metrics
    }
    
    return results


def compare_agents(env, agents: Dict[str, Any], n_episodes: int = 10,
                   seed: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
    """
    Compare multiple agents/controllers.
    
    Args:
        env: Traffic light environment
        agents: Dictionary mapping agent name to agent object
        n_episodes: Number of episodes per agent
        seed: Random seed
        
    Returns:
        Dictionary mapping agent name to evaluation results
    """
    results = {}
    
    for name, agent in agents.items():
        print(f"\nEvaluating {name}...")
        
        # Reset agent if possible
        if hasattr(agent, 'reset'):
            agent.reset()
        
        results[name] = evaluate_agent(env, agent, n_episodes, seed=seed)
        
        print(f"  Mean reward: {results[name]['mean_reward']:.2f} ± {results[name]['std_reward']:.2f}")
        print(f"  Mean throughput: {results[name]['mean_throughput']:.0f} ± {results[name]['std_throughput']:.0f}")
        print(f"  Mean wait time: {results[name]['mean_wait_time']:.0f}")
    
    return results


def print_comparison_table(results: Dict[str, Dict[str, Any]]):
    """Print a comparison table of agent results"""
    
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS")
    print("=" * 80)
    
    # Header
    header = f"{'Agent':<20} {'Reward':>12} {'Throughput':>12} {'Wait Time':>12} {'Avg Queue':>12}"
    print(header)
    print("-" * 80)
    
    # Rows
    for name, res in results.items():
        row = f"{name:<20} {res['mean_reward']:>12.2f} {res['mean_throughput']:>12.0f} {res['mean_wait_time']:>12.0f} {res['mean_queue_length']:>12.2f}"
        print(row)
    
    print("=" * 80)


if __name__ == "__main__":
    print("Metrics module loaded successfully.")
    
    # Test EvaluationMetrics
    metrics = EvaluationMetrics()
    
    for i in range(100):
        reward = -np.random.rand() * 5
        info = {
            'total_queue': np.random.rand() * 20,
            'vehicles_passed': np.random.randint(0, 5),
            'total_vehicles_passed': i * 2,
            'total_wait_time': i * 10,
            'phase_switches': i // 20
        }
        metrics.update(reward, info)
    
    summary = metrics.get_summary()
    print("\nTest metrics summary:")
    for key, value in summary.items():
        print(f"  {key}: {value:.2f}")
