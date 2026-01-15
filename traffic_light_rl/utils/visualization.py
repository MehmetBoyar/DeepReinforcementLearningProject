"""
Visualization utilities for traffic light control.

Provides plotting functions for learning curves, comparisons, and intersection state.
"""

import numpy as np
from typing import Dict, List, Any, Optional
import os

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available. Visualization functions will not work.")


def plot_learning_curve(rewards: List[float], 
                        window: int = 100,
                        title: str = "Learning Curve",
                        save_path: Optional[str] = None):
    """
    Plot learning curve with smoothing.
    
    Args:
        rewards: List of episode rewards
        window: Smoothing window size
        title: Plot title
        save_path: Path to save figure (optional)
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Raw rewards
    ax.plot(rewards, alpha=0.3, color='blue', label='Raw')
    
    # Smoothed rewards
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(rewards)), smoothed, color='blue', 
                linewidth=2, label=f'Smoothed (window={window})')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()


def plot_comparison(results: Dict[str, Dict[str, Any]],
                    metric: str = 'mean_reward',
                    title: str = "Agent Comparison",
                    save_path: Optional[str] = None):
    """
    Plot comparison bar chart of agent results.
    
    Args:
        results: Dictionary of agent results from compare_agents()
        metric: Metric to compare
        title: Plot title
        save_path: Path to save figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    agents = list(results.keys())
    values = [results[a][metric] for a in agents]
    
    # Error bars if available
    std_key = metric.replace('mean_', 'std_')
    if std_key in results[agents[0]]:
        errors = [results[a][std_key] for a in agents]
    else:
        errors = None
    
    bars = ax.bar(agents, values, yerr=errors, capsize=5, color='steelblue', alpha=0.8)
    
    ax.set_xlabel('Agent')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_multi_comparison(results: Dict[str, Dict[str, Any]],
                          metrics: List[str] = None,
                          save_path: Optional[str] = None):
    """
    Plot multiple metrics comparison.
    
    Args:
        results: Dictionary of agent results
        metrics: List of metrics to compare
        save_path: Path to save figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available")
        return
    
    if metrics is None:
        metrics = ['mean_reward', 'mean_throughput', 'mean_queue_length', 'mean_switches']
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))
    
    if n_metrics == 1:
        axes = [axes]
    
    agents = list(results.keys())
    colors = plt.cm.Set2(np.linspace(0, 1, len(agents)))
    
    for ax, metric in zip(axes, metrics):
        values = [results[a].get(metric, 0) for a in agents]
        bars = ax.bar(agents, values, color=colors, alpha=0.8)
        
        ax.set_title(metric.replace('_', ' ').title())
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_episode_metrics(metrics: Dict[str, List[float]], 
                         save_path: Optional[str] = None):
    """
    Plot metrics over time within an episode.
    
    Args:
        metrics: Dictionary of metric name -> list of values
        save_path: Path to save figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available")
        return
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 3 * n_metrics), sharex=True)
    
    if n_metrics == 1:
        axes = [axes]
    
    for ax, (name, values) in zip(axes, metrics.items()):
        ax.plot(values, linewidth=1)
        ax.set_ylabel(name.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time Step')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def render_intersection(queues: Dict[str, float], phase: int,
                        save_path: Optional[str] = None):
    """
    Render intersection state as a visual diagram.
    
    Args:
        queues: Dictionary of movement -> queue length
        phase: Current phase (0-3)
        save_path: Path to save figure
    """
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available")
        return
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw roads
    road_color = '#404040'
    ax.add_patch(patches.Rectangle((-1, -3), 2, 6, facecolor=road_color))
    ax.add_patch(patches.Rectangle((-3, -1), 6, 2, facecolor=road_color))
    
    # Draw intersection
    ax.add_patch(patches.Rectangle((-1, -1), 2, 2, facecolor='#606060'))
    
    # Define positions for queue labels
    positions = {
        'N': (0, 2.5),
        'S': (0, -2.5),
        'E': (2.5, 0),
        'W': (-2.5, 0)
    }
    
    # Phase colors
    phase_colors = {
        0: {'N': 'green', 'S': 'green', 'E': 'red', 'W': 'red'},
        1: {'N': 'yellow', 'S': 'yellow', 'E': 'red', 'W': 'red'},
        2: {'N': 'red', 'S': 'red', 'E': 'green', 'W': 'green'},
        3: {'N': 'red', 'S': 'red', 'E': 'yellow', 'W': 'yellow'},
    }
    
    # Draw traffic lights
    light_pos = {
        'N': (0, 1.3),
        'S': (0, -1.3),
        'E': (1.3, 0),
        'W': (-1.3, 0)
    }
    
    for direction, pos in light_pos.items():
        color = phase_colors[phase][direction]
        ax.plot(pos[0], pos[1], 'o', markersize=15, color=color)
    
    # Draw queue information
    queue_text = {
        'N': f"N: {queues.get('N_to_S', 0):.0f}↓ {queues.get('N_to_E', 0):.0f}→ {queues.get('N_to_W', 0):.0f}←",
        'S': f"S: {queues.get('S_to_N', 0):.0f}↑ {queues.get('S_to_W', 0):.0f}← {queues.get('S_to_E', 0):.0f}→",
        'E': f"E: {queues.get('E_to_W', 0):.0f}← {queues.get('E_to_N', 0):.0f}↑ {queues.get('E_to_S', 0):.0f}↓",
        'W': f"W: {queues.get('W_to_E', 0):.0f}→ {queues.get('W_to_N', 0):.0f}↑ {queues.get('W_to_S', 0):.0f}↓"
    }
    
    for direction, pos in positions.items():
        ax.text(pos[0], pos[1], queue_text[direction], ha='center', va='center',
                fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Phase names
    phase_names = {
        0: "N-S Through",
        1: "N-S Protected Left",
        2: "E-W Through",
        3: "E-W Protected Left"
    }
    
    ax.set_title(f"Phase {phase}: {phase_names[phase]}", fontsize=14)
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


if __name__ == "__main__":
    if MATPLOTLIB_AVAILABLE:
        # Test learning curve
        rewards = np.random.randn(500).cumsum() - np.arange(500) * 0.1
        plot_learning_curve(rewards, title="Test Learning Curve")
        
        # Test intersection render
        test_queues = {
            'N_to_S': 10, 'N_to_E': 3, 'N_to_W': 2,
            'S_to_N': 8, 'S_to_W': 2, 'S_to_E': 1,
            'E_to_W': 5, 'E_to_S': 1, 'E_to_N': 2,
            'W_to_E': 6, 'W_to_N': 2, 'W_to_S': 1
        }
        render_intersection(test_queues, phase=0)
    else:
        print("matplotlib not available - visualization tests skipped")
