import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_behavior(env, agent, episodes=3):
    """
    Runs episodes and returns detailed behavioral stats.
    """
    action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    action_labels = {0: "KEEP", 1: "NEXT", 2: "SKIP_NS", 3: "SKIP_EW"}
    
    # Track queue max per lane to check fairness
    # Lanes: N(0-2), S(3-5), E(6-8), W(9-11)
    lane_max_queues = {i: [] for i in range(12)}
    
    for _ in range(episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = agent.act(state, training=False)
            state, _, done, truncated, _ = env.step(action)
            done = done or truncated
            
            # Log Action
            action_counts[action] += 1
            
            # Log Queues
            queues = state[:12]
            for i in range(12):
                lane_max_queues[i].append(queues[i])

    # 1. Action Distribution Data
    total_actions = sum(action_counts.values())
    action_data = []
    for act, count in action_counts.items():
        pct = (count / total_actions) * 100 if total_actions > 0 else 0
        action_data.append({"Action": action_labels[act], "Count": count, "Percentage": pct})
    
    df_actions = pd.DataFrame(action_data)

    # 2. Fairness Data (Max Queue observed per lane avg over time)
    fairness_data = []
    directions = ['N', 'N', 'N', 'S', 'S', 'S', 'E', 'E', 'E', 'W', 'W', 'W']
    types = ['S', 'L', 'R'] * 4
    
    for i in range(12):
        avg_q = np.mean(lane_max_queues[i])
        fairness_data.append({
            "Lane Index": i,
            "Label": f"{directions[i]}-{types[i]}",
            "Avg Queue": avg_q
        })
    
    df_fairness = pd.DataFrame(fairness_data)
    
    return df_actions, df_fairness

def plot_behavior(df_actions, df_fairness):
    """Generates figures for the GUI"""
    # Plot 1: Actions
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.barplot(data=df_actions, x="Action", y="Count", ax=ax1, palette="viridis")
    ax1.set_title("Action Distribution")
    
    # Plot 2: Fairness
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.barplot(data=df_fairness, x="Label", y="Avg Queue", ax=ax2, palette="magma")
    ax2.set_title("Average Queue Length per Lane (Fairness Check)")
    ax2.tick_params(axis='x', rotation=45)
    
    return fig1, fig2