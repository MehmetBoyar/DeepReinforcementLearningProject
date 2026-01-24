import os
import glob
import re
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate

# Project Imports
from traffic_rl.config import EnvConfig
from traffic_rl.env.traffic_env import TrafficLightEnv
from traffic_rl.agents.dqn import DQNAgent
from traffic_rl.agents.q_learning import QLearningAgent
from traffic_rl.agents.baselines import FixedTimeAgent, AdaptiveAgent

# Standard Test Scenarios
SCENARIOS = {
    "Low (1.0x)": 1.0,
    "Normal (1.5x)": 1.5,
    "High (4.0x)": 4.0,
    "Saturated (6.0x)": 6.0
}

def parse_folder_name(folder_name):
    """Extracts metadata (ExpID, Eps, Mult) from folder string."""
    try:
        eps = int(re.search(r'Eps(\d+)', folder_name).group(1))
        mult = float(re.search(r'Mult([\d.]+)x', folder_name).group(1))
        exp_id = re.search(r'(Exp\d+)', folder_name).group(1)
        return exp_id, eps, mult
    except:
        return folder_name, 0, 0.0

def evaluate_agent_scenarios(base_config, agent, episodes=3):
    """Runs agent through all defined scenarios."""
    scores = {}
    for scen_name, multiplier in SCENARIOS.items():
        # Create temp config for this specific scenario
        env_conf = EnvConfig(
            arrival_rates=base_config.env.arrival_rates, 
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
            
            # Metric: Average Wait Time (Queue / Throughput)
            if total_throughput > 0:
                wait = total_queue / total_throughput
            else:
                wait = total_queue # Fallback for gridlock
            avg_waits.append(wait)
        scores[scen_name] = np.mean(avg_waits)
    return scores

def generate_plots(df, output_dir):
    """Generates Performance Heatmap."""
    try:
        sns.set_style("whitegrid")
        
        # HEATMAP: Improvement over FixedTime
        heatmap_data = df.copy()
        
        # Create Label
        heatmap_data['Label'] = heatmap_data.apply(
            lambda x: f"{x['Exp']} {x['Model']}", axis=1
        )
        
        scen_cols = list(SCENARIOS.keys())
        imp_matrix = []
        labels = []
        
        # Get baseline row
        baseline_rows = df[df['Model'] == 'FixedTime']
        if baseline_rows.empty:
            print("Warning: FixedTime baseline not found. Skipping Heatmap.")
            return

        baseline_fixed = baseline_rows.iloc[0]

        for idx, row in heatmap_data.iterrows():
            imp_row = []
            for scen in scen_cols:
                base_val = baseline_fixed[scen]
                val = row[scen]
                if base_val > 0:
                    # Calculate % Improvement
                    imp = ((base_val - val) / base_val) * 100
                    imp = max(imp, -100) # Cap negative values for visual clarity
                else:
                    imp = 0
                imp_row.append(imp)
            imp_matrix.append(imp_row)
            labels.append(row['Label'])

        imp_df = pd.DataFrame(imp_matrix, columns=scen_cols, index=labels)
        
        plt.figure(figsize=(10, 12))
        sns.heatmap(imp_df, annot=True, fmt=".0f", cmap="RdYlGn", center=0)
        plt.title("Improvement over FixedTime (%)")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "heatmap_improvement.png"))
        plt.close()
        print("  -> Generated heatmap_improvement.png")
            
    except Exception as e:
        print(f"Plotting Error: {e}")

def run_analysis_suite(target_folder, base_config):
    """
    Main entry point for analysis.
    """
    print(f"\n" + "="*60)
    print(f"STARTING ANALYSIS: {target_folder}")
    print("="*60)
    
    if not os.path.exists(target_folder):
        print(f"Error: Folder {target_folder} not found.")
        return

    # Create output directory for results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join("results", f"analysis_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)
    
    final_results = []

    # 1. EVALUATE BASELINES
    print("Evaluating Baselines...")
    for name, agent in [("FixedTime", FixedTimeAgent()), ("Adaptive", AdaptiveAgent())]:
        scores = evaluate_agent_scenarios(base_config, agent)
        row = {"Exp": "Base", "Model": name, "Train_Cfg": "-"}
        row.update(scores)
        final_results.append(row)

    # 2. EVALUATE TRAINED MODELS
    subdirs = [f.path for f in os.scandir(target_folder) if f.is_dir()]
    # Natural sort
    subdirs.sort(key=lambda x: int(re.search(r'Exp(\d+)', x).group(1)) if re.search(r'Exp(\d+)', x) else 999)

    for folder in subdirs:
        folder_name = os.path.basename(folder)
        exp_id, train_eps, train_mult = parse_folder_name(folder_name)
        train_str = f"{train_eps}ep / {train_mult}x"
        print(f"Processing {exp_id}...")

        # Find Models
        dqn_files = glob.glob(os.path.join(folder, "dqn_*", "model.pt"))
        # Look for .pt OR .pkl for Q-Learning
        q_files = glob.glob(os.path.join(folder, "q_learning_*", "model.pt")) + \
                  glob.glob(os.path.join(folder, "q_learning_*", "model.pkl"))

        # Evaluate DQN
        if dqn_files:
            try:
                agent = DQNAgent(14, 4, base_config.agent, device="cpu")
                agent.load(dqn_files[0])
                scores = evaluate_agent_scenarios(base_config, agent)
                row = {"Exp": exp_id, "Model": "DQN", "Train_Cfg": train_str}
                row.update(scores)
                final_results.append(row)
            except Exception as e: print(f"  Err DQN: {e}")

        # Evaluate Q-Learn
        if q_files:
            try:
                agent = QLearningAgent(4, base_config.agent)
                agent.load(q_files[0])
                scores = evaluate_agent_scenarios(base_config, agent)
                row = {"Exp": exp_id, "Model": "Q-Learn", "Train_Cfg": train_str}
                row.update(scores)
                final_results.append(row)
            except Exception as e: print(f"  Err Q: {e}")

    # 3. REPORTING
    if not final_results:
        print("No results found.")
        return

    df = pd.DataFrame(final_results)
    
    # Calc Overall Score
    scen_cols = list(SCENARIOS.keys())
    df['Overall_Avg_Wait'] = df[scen_cols].mean(axis=1)
    df = df.sort_values('Overall_Avg_Wait')

    # Print Table (using lambda to fix pandas version issue)
    print("\n" + "="*80)
    print("LEADERBOARD (Avg Wait Time [s])")
    print("="*80)
    print(tabulate(df, headers='keys', tablefmt='simple', showindex=False, floatfmt=".2f"))

    # Save CSV
    csv_path = os.path.join(results_dir, "analysis_data.csv")
    df.to_csv(csv_path, index=False)
    
    # Generate Plots
    generate_plots(df, results_dir)
    
    print(f"\nResults saved to: {results_dir}")