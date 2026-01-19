"""
compare_literature.py

Generates the "Apples-to-Apples" comparison table for your paper.
Calculates % improvement over Fixed-Time baseline using literature-standard metrics.
"""

import os
import sys
import numpy as np
import yaml
import copy

# Add current directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)
os.chdir(project_dir)

from traffic_light_rl.env import TrafficLightEnv
from traffic_light_rl.agents import QLearningAgent, DQNAgent
from traffic_light_rl.baselines import FixedTimeController, AdaptiveController
from traffic_light_rl.utils.metrics import evaluate_agent

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_comparison():
    config_path = 'traffic_light_rl/config/config.yaml'
    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        return

    config = load_config(config_path)
    base_rates = config['environment']['arrival_rates']
    
    # 1. Load Agents
    print("Loading Agents...")
    agents = {}
    
    # Baselines
    agents['Fixed-Time'] = FixedTimeController(
        phase_durations=config['baselines']['fixed_time']['phase_durations']
    )
    agents['Adaptive'] = AdaptiveController(
        min_phase_duration=config['baselines']['adaptive']['min_phase_duration'],
        max_phase_duration=config['baselines']['adaptive']['max_phase_duration']
    )
    
    # RL Agents (Load your trained models)
    try:
        # Load Q-Learning
        if os.path.exists('models/q_learning.pkl'):
            q_agent = QLearningAgent()
            q_agent.load('models/q_learning.pkl')
            # Turn off exploration
            q_agent.epsilon = 0.0
            agents['Tabular Q'] = q_agent
            print("  - Loaded Tabular Q-Learning")
        
        # Load DQN
        if os.path.exists('models/dqn.pkl'):
            # NOTE: Architecture must match training (main.py uses 64,64)
            dqn_agent = DQNAgent(state_dim=14, action_dim=4, hidden_dims=(64, 64)) 
            dqn_agent.load('models/dqn.pkl')
            # Turn off exploration
            dqn_agent.epsilon = 0.0 
            agents['DQN'] = dqn_agent
            print("  - Loaded DQN")
    except Exception as e:
        print(f"Warning: Issue loading models. {e}")

    # 2. Run Scenarios
    scenarios = config.get('scenarios', {
        'low_traffic': {'arrival_factor': 0.5},
        'high_traffic': {'arrival_factor': 1.5}
    })

    results_table = {}

    for scen_name, scen_cfg in scenarios.items():
        factor = scen_cfg['arrival_factor']
        print(f"\n--- Running Scenario: {scen_name.upper()} (Factor: {factor}x) ---")
        
        # Modify arrival rates for this scenario
        current_rates = {k: v * factor for k, v in base_rates.items()}
        
        # Create env with specific rates for this scenario
        env = TrafficLightEnv(seed=42, arrival_rates=current_rates, max_queue=200)
        
        results_table[scen_name] = {}
        
        for name, agent in agents.items():
            print(f"  Evaluating {name}...", end='\r')
            
            # Evaluate (30 episodes for speed, 50 for final paper)
            metrics = evaluate_agent(env, agent, n_episodes=30, seed=42)
            
            # Use the new metric: Average Wait Time per Vehicle
            avg_wait = metrics['mean_wait_per_vehicle']
            
            # Store result
            results_table[scen_name][name] = avg_wait
            print(f"  Evaluating {name}: {avg_wait:.2f}s avg wait")

# 3. Print The "Paper-Ready" Table
    print("\n\n" + "="*115)
    print(f"{'LITERATURE COMPARISON TABLE':^115}")
    print("="*115)
    print(f"{'Method':<20} | {'Low (1.0x)':<18} | {'High (4.0x)':<18} | {'Sat (6.0x)':<18} | {'Imp (Sat)':<15}")
    print("-" * 115)
    
    # Get Fixed-Time baselines
    fixed_sat = results_table.get('saturation_traffic', {}).get('Fixed-Time', 0.0)
    
    for name in ['Fixed-Time', 'Adaptive', 'Tabular Q', 'DQN']:
        if name not in agents:
            continue
            
        # Get results safely (default to 0.0 if missing)
        low_res = results_table.get('low_traffic', {}).get(name, 0.0)
        high_res = results_table.get('high_traffic', {}).get(name, 0.0)
        sat_res = results_table.get('saturation_traffic', {}).get(name, 0.0)
        
        # Calculate Improvement % relative to Fixed-Time Saturation
        if name == 'Fixed-Time' or fixed_sat == 0:
            imp_str = "-"
        else:
            # Formula: (Baseline - Method) / Baseline
            improvement = ((fixed_sat - sat_res) / fixed_sat) * 100
            imp_str = f"{improvement:+.1f}%"
            
        print(f"{name:<20} | {low_res:>15.2f} s | {high_res:>15.2f} s | {sat_res:>15.2f} s | {imp_str:>15}")
    
    print("="*115)
    print(f"Baseline (Fixed-Time) Saturation Wait: {fixed_sat:.2f}s")
    print("NOTE: 'Imp (Sat)' shows % improvement over Fixed-Time in the Saturation (6.0x) scenario.")

if __name__ == "__main__":
    run_comparison()