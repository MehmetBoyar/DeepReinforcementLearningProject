import os
import glob
import re
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from dataclasses import asdict

# Project Imports
from traffic_rl.config import EnvConfig, AgentConfig
from traffic_rl.env.traffic_env import TrafficLightEnv
from traffic_rl.agents.dqn import DQNAgent

SCENARIOS = {
    "Low (1.0x)": 1.0, 
    "Normal (1.5x)": 1.5,
    "High (4.0x)": 4.0, 
    "Saturated (6.0x)": 6.0
}

def parse_folder_name(folder_name):
    try:
        eps = int(re.search(r'Eps(\d+)', folder_name).group(1))
        mult = float(re.search(r'Mult([\d.]+)x', folder_name).group(1))
        exp_id = re.search(r'(Exp\d+)', folder_name).group(1)
        return exp_id, eps, mult
    except:
        return "Unknown", 0, 0.0

def evaluate_dqn(base_env_config, agent_config, model_path, episodes=3):
    """
    Evaluates a single DQN model across all scenarios.
    Uses the SPECIFIC agent_config for architecture, but base_env_config for scenarios.
    """
    scores = {}
    
    try:
        # Init agent with its specific config (Hidden Dim, Dueling, etc.)
        agent = DQNAgent(14, 4, agent_config, device="cpu")
        agent.load(model_path)
    except Exception as e:
        print(f"Error loading agent from {model_path}: {e}")
        return None

    for scen_name, multiplier in SCENARIOS.items():
        env_conf = EnvConfig(
            arrival_rates=base_env_config.arrival_rates, 
            traffic_multiplier=multiplier,
            max_steps=1000 
        )
        env = TrafficLightEnv(env_conf)
        
        avg_waits = []
        for _ in range(episodes):
            state, _ = env.reset()
            done = False
            total_queue = 0
            total_throughput = 0
            
            while not done:
                action = agent.act(state, training=False)
                next_state, _, done, truncated, info = env.step(action)
                done = done or truncated
                state = next_state
                
                total_queue += info['total_queue']
                total_throughput += info['throughput']
            
            if total_throughput > 0:
                wait = total_queue / total_throughput
            else:
                wait = total_queue 
            avg_waits.append(wait)
        scores[scen_name] = np.mean(avg_waits)
        
    return scores

def scan_and_evaluate(folder_path, label, base_config):
    """
    Scans a folder, auto-detects config.json per experiment, and evaluates.
    """
    print(f"Scanning {label}: {folder_path}")
    results = []
    
    if not os.path.exists(folder_path):
        print(f"  Warning: Path not found.")
        return []

    subdirs = [f.path for f in os.scandir(folder_path) if f.is_dir()]
    # Natural Sort
    subdirs.sort(key=lambda x: int(re.search(r'Exp(\d+)', x).group(1)) if re.search(r'Exp(\d+)', x) else 999)

    for folder in subdirs:
        fname = os.path.basename(folder)
        
        # 1. Look for Model
        dqn_fs = glob.glob(os.path.join(folder, "dqn_*", "model.pt"))
        # Also check for best_model.pt
        best_fs = glob.glob(os.path.join(folder, "dqn_*", "best_model.pt"))
        model_file = best_fs[0] if best_fs else (dqn_fs[0] if dqn_fs else None)
        
        if model_file:
            exp_id, eps, mult = parse_folder_name(fname)
            print(f"  Evaluating {exp_id}...")
            
            # 2. Auto-Detect Configuration
            # We default to base_config.agent, but try to overwrite if config.json exists
            specific_agent_conf = base_config.agent
            config_json_path = os.path.join(folder, "config.json")
            
            if os.path.exists(config_json_path):
                try:
                    with open(config_json_path, 'r') as f:
                        saved_data = json.load(f)
                    # Reconstruct AgentConfig from saved JSON
                    # We use .get() to handle backward compatibility if new flags were added later
                    a_data = saved_data.get('agent', {})
                    specific_agent_conf = AgentConfig(
                        name=a_data.get('name', 'dqn'),
                        gamma=a_data.get('gamma', 0.99),
                        lr=a_data.get('lr', 0.001),
                        batch_size=a_data.get('batch_size', 64),
                        hidden_dim=a_data.get('hidden_dim', 64), # Loaded from file!
                        buffer_size=a_data.get('buffer_size', 50000),
                        epsilon_start=a_data.get('epsilon_start', 1.0),
                        epsilon_min=a_data.get('epsilon_min', 0.01),
                        epsilon_decay=a_data.get('epsilon_decay', 0.99),
                        alpha=a_data.get('alpha', 0.1),
                        double_dqn=a_data.get('double_dqn', False), # Loaded from file!
                        dueling_dqn=a_data.get('dueling_dqn', False) # Loaded from file!
                    )
                except Exception as e:
                    print(f"    Warning: Could not load config.json ({e}). Using default.")

            # 3. Evaluate
            scores = evaluate_dqn(base_config.env, specific_agent_conf, model_file)
            
            if scores:
                for scen, val in scores.items():
                    results.append({
                        "Exp": exp_id, 
                        "Train_Cfg": f"{eps}ep/{mult}x",
                        "Version": label, 
                        "Scenario": scen, 
                        "Wait_Time": val
                    })

    return results

def run_comparison_suite(folder_a, folder_b, base_config):
    """
    Main entry point for comparing two experiment batches.
    """
    print(f"\n" + "="*60)
    print("STARTING COMPARISON SUITE")
    print(f"Base (Old): {folder_a}")
    print(f"Test (New): {folder_b}")
    print("="*60)
    
    # No longer need manual dims passed in!
    data_a = scan_and_evaluate(folder_a, "Baseline (Old)", base_config)
    data_b = scan_and_evaluate(folder_b, "Optimized (New)", base_config)
    
    all_data = data_a + data_b
    if not all_data:
        print("No data found.")
        return

    df = pd.DataFrame(all_data)
    
    # Generate Dashboard Plot
    print("\nGenerating Dashboard Plot...")
    try:
        sns.set_style("whitegrid")
        g = sns.catplot(
            data=df, kind="bar", x="Exp", y="Wait_Time", hue="Version",
            col="Scenario", col_wrap=2, height=4, aspect=1.5, sharey=False,
            palette={"Baseline (Old)": "gray", "Optimized (New)": "green"}
        )
        g.set_axis_labels("Experiment ID", "Avg Wait Time (s)")
        g.fig.suptitle("Old vs New Configuration Performance", y=1.02)
        plt.savefig("comparison_dashboard.png", bbox_inches='tight')
    except Exception as e:
        print(f"Plotting Error: {e}")
    
    # Save CSV
    df.to_csv("full_comparison_results.csv", index=False)