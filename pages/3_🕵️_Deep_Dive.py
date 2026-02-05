import streamlit as st
import os
import glob
import pandas as pd
import numpy as np
import json
import re # Needed for regex parsing
from traffic_rl.config import AppConfig, EnvConfig
from traffic_rl.env.traffic_env import TrafficLightEnv
from traffic_rl.agents.dqn import DQNAgent
from traffic_rl.agents.q_learning import QLearningAgent
from traffic_rl.tools.visualizer import generate_gif
from traffic_rl.tools.behavior import analyze_behavior, plot_behavior
from traffic_rl.gui_utils import render_sidebar, get_experiment_folders

st.set_page_config(page_title="Agent Deep Dive", layout="wide", page_icon="🕵️")
render_sidebar()

st.header("🕵️ Agent Deep Dive")

folders = get_experiment_folders()
if not folders:
    st.warning("No experiments found. Run a training batch first!")
    st.stop()

# --- CONSTANTS ---
SCENARIOS = {
    "Low (1.0x)": 1.0,
    "Normal (1.5x)": 1.5,
    "High (4.0x)": 4.0,
    "Saturated (6.0x)": 6.0
}

# --- HELPER 1: Reward Scanner (For Tab 3) ---
def scan_rewards(root_folder):
    data = []
    # Find all metrics.csv files recursively
    files = glob.glob(os.path.join(root_folder, "**", "metrics.csv"), recursive=True)
    
    for f in files:
        try:
            df = pd.read_csv(f)
            if df.empty or len(df) < 10: continue
            
            folder_path = os.path.dirname(f)
            folder_name = os.path.basename(folder_path)
            
            # Extract Traffic Multiplier using Regex
            traffic_load = "Unknown"
            if "Mult" in folder_name:
                match = re.search(r"Mult([\d\.]+)x", folder_name)
                if match: traffic_load = f"{match.group(1)}x"

            last_50_avg = df['reward'].tail(50).mean()
            best_episode = df['reward'].max()
            total_eps = len(df)
            
            data.append({
                "Run Name": folder_name,
                "Traffic": traffic_load,
                "Converged Reward (Last 50)": last_50_avg,
                "Peak Reward": best_episode,
                "Episodes": total_eps,
                "Full_Path": folder_path 
            })
        except:
            pass
    return pd.DataFrame(data)

# --- HELPER 2: Evaluate Specific Agent (For Tab 3) ---
def evaluate_winner_scenarios(agent_dir):
    """Loads the agent from the directory and runs the 4 scenarios."""
    
    # 1. Find Model File
    pt_files = glob.glob(os.path.join(agent_dir, "*.pt"))
    pkl_files = glob.glob(os.path.join(agent_dir, "*.pkl"))
    
    # Prefer best_model, then model.pt
    model_path = None
    is_dqn = True
    
    if pt_files:
        model_path = next((f for f in pt_files if "best_model" in f), pt_files[0])
    elif pkl_files:
        model_path = next((f for f in pkl_files if "best_model" in f), pkl_files[0])
        is_dqn = False
        
    if not model_path:
        return {"Error": "No model file found"}

    # 2. Load Config from JSON if available
    conf = AppConfig.load("configs/default.yaml")
    config_json = os.path.join(agent_dir, "config.json")
    if os.path.exists(config_json):
        try:
            with open(config_json) as f: saved = json.load(f)
            a_data = saved.get('agent', {})
            if is_dqn:
                conf.agent.hidden_dim = a_data.get('hidden_dim', 128)
                conf.agent.double_dqn = a_data.get('double_dqn', False)
                conf.agent.dueling_dqn = a_data.get('dueling_dqn', False)
        except: pass

    # 3. Init Agent
    if is_dqn:
        agent = DQNAgent(14, 4, conf.agent, device="cpu")
    else:
        agent = QLearningAgent(4, conf.agent)
        
    try:
        agent.load(model_path)
    except:
        return {"Error": "Failed to load weights"}

    # 4. Run Scenarios
    results = {}
    for name, mult in SCENARIOS.items():
        # Temp Env Config
        env_conf = EnvConfig(
            arrival_rates=conf.env.arrival_rates,
            traffic_multiplier=mult,
            max_steps=1000
        )
        env = TrafficLightEnv(env_conf)
        
        # Eval Loop (3 episodes average)
        waits = []
        for _ in range(3):
            state, _ = env.reset()
            done = False
            total_q = 0
            total_thru = 0
            while not done:
                action = agent.act(state, training=False)
                state, _, done, truncated, info = env.step(action)
                done = done or truncated
                total_q += info['total_queue']
                total_thru += info['throughput']
            
            if total_thru > 0: waits.append(total_q / total_thru)
            else: waits.append(total_q) # Gridlock fallback
            
        results[name] = np.mean(waits)
        
    return results

# ==========================================
# 1. GLOBAL SELECTION UI (Required for Tabs 1 & 2)
# ==========================================
folder = st.selectbox("Select Experiment Folder", folders)

# Sub-Experiment Selector
target_dir = folder
subfolders = [f.name for f in os.scandir(folder) if f.is_dir() and "Exp" in f.name]

c_sel1, c_sel2 = st.columns(2)
with c_sel1:
    if subfolders:
        # Natural sort (Exp1, Exp2... Exp10)
        subfolders.sort(key=lambda x: int(x.split('Exp')[1].split('_')[0]) if 'Exp' in x else 999)
        selected_sub = st.selectbox("1. Select Sub-Experiment", subfolders)
        target_dir = os.path.join(folder, selected_sub)
    else:
        st.caption("Using root folder (Single Experiment)")

# Model File Selector
found_files = glob.glob(os.path.join(target_dir, "**", "*.pt"), recursive=True) + \
              glob.glob(os.path.join(target_dir, "**", "*.pkl"), recursive=True)

selected_model_path = None
with c_sel2:
    if found_files:
        file_options = {
            f"{os.path.basename(os.path.dirname(p))} / {os.path.basename(p)}": p 
            for p in found_files
        }
        # Auto-select best_model
        default_idx = 0
        keys_list = list(file_options.keys())
        for i, k in enumerate(keys_list):
            if "best_model" in k:
                default_idx = i
                break
        
        selected_label = st.selectbox("2. Select Agent Model", keys_list, index=default_idx)
        selected_model_path = file_options[selected_label]
    else:
        st.warning(f"No model files found in {os.path.basename(target_dir)}")

# --- HELPER: Load Selected Agent (Used by Tab 1 & 2) ---
def load_selected_agent():
    if not selected_model_path: return None, None
    
    is_dqn = selected_model_path.endswith(".pt")
    
    conf = AppConfig.load("configs/default.yaml")
    
    # Try to recover specific architecture from config.json
    agent_dir = os.path.dirname(selected_model_path)
    config_json = os.path.join(agent_dir, "config.json")
    
    if os.path.exists(config_json):
        try:
            with open(config_json) as f: saved = json.load(f)
            agent_data = saved.get('agent', {})
            if is_dqn:
                conf.agent.hidden_dim = agent_data.get('hidden_dim', 128)
                conf.agent.double_dqn = agent_data.get('double_dqn', False)
                conf.agent.dueling_dqn = agent_data.get('dueling_dqn', False)
        except: pass
    
    # Setup Environment (1.3x Traffic for stress test visualization)
    conf.env.traffic_multiplier = 1.3
    env = TrafficLightEnv(conf.env)
    
    if is_dqn:
        agent = DQNAgent(14, 4, conf.agent, device="cpu")
    else:
        agent = QLearningAgent(4, conf.agent)
        
    try:
        agent.load(selected_model_path)
    except Exception as e:
        st.error(f"Error loading weights: {e}")
        return None, None
        
    return env, agent

# ==========================================
# TABS
# ==========================================
tab_vis, tab_beh, tab_rew = st.tabs(["🎥 Visual Replay", "🧠 Behavioral Analysis", "🏆 Reward Leaderboard"])

# --- TAB 1: VISUAL REPLAY ---
with tab_vis:
    if selected_model_path:
        is_dqn = selected_model_path.endswith(".pt")
        agent_type = "DQN" if is_dqn else "Q-Learning"
        
        st.info(f"Visualizing: **{agent_type}** from `{selected_label}`")
        
        col_v1, col_v2 = st.columns([1, 2])
        with col_v1:
            st.write("Generate a visual replay (GIF) of this specific agent controlling traffic.")
            st.caption("Traffic Load: 1.3x (Moderate Stress Test)")
            
            if st.button("Generate Replay GIF"):
                with st.spinner("Simulating episode..."):
                    env, agent = load_selected_agent()
                    if env and agent:
                        gif_path = os.path.join(os.path.dirname(selected_model_path), "replay_gui.gif")
                        generate_gif(env, agent, gif_path)
                        st.session_state['last_gif'] = gif_path
        
        with col_v2:
            display_gif = st.session_state.get('last_gif', None)
            
            # Fallback to existing file
            if not display_gif and os.path.exists(os.path.join(os.path.dirname(selected_model_path), "replay_gui.gif")):
                 display_gif = os.path.join(os.path.dirname(selected_model_path), "replay_gui.gif")

            if display_gif and os.path.exists(display_gif):
                st.image(display_gif, caption=f"{agent_type} Replay", width=600)

# --- TAB 2: BEHAVIORAL ANALYSIS ---
with tab_beh:
    if selected_model_path:
        st.write(f"Analyzing behavioral patterns for `{selected_label}`.")
        
        if st.button("Run Behavior Check"):
            with st.spinner("Analyzing decisions (5 Episodes)..."):
                env, agent = load_selected_agent()
                if env and agent:
                    df_act, df_fair = analyze_behavior(env, agent, episodes=5)
                    fig1, fig2 = plot_behavior(df_act, df_fair)
                    
                    c_beh1, c_beh2 = st.columns([1, 2])
                    
                    with c_beh1:
                        st.markdown("##### Decision Distribution")
                        st.dataframe(df_act, hide_index=True)
                        
                    with c_beh2:
                        st.pyplot(fig1)
                    
                    st.divider()
                    st.markdown("##### Fairness Check (Max Queue Lengths)")
                    st.caption("High bars indicate specific lanes are being neglected (Lane Starvation). Ideally, bars should be roughly equal.")
                    st.pyplot(fig2)

# --- TAB 3: REWARD LEADERBOARD (NEW) ---
with tab_rew:
    st.markdown("### 🏆 Top Performing Agents")
    st.markdown("Find the mathematically best agent in the batch and **verify** its real-world wait times.")
    
    col_scan, col_res = st.columns([1, 4])
    
    with col_scan:
        if st.button("🔄 Scan Results", type="primary"):
            with st.spinner(f"Scanning {folder}..."):
                df_rew = scan_rewards(folder)
                st.session_state['reward_df'] = df_rew

    if 'reward_df' in st.session_state:
        df = st.session_state['reward_df']
        
        if not df.empty:
            # 1. Filter
            traffic_opts = ["All"] + sorted(list(df['Traffic'].unique()))
            sel_traffic = st.selectbox("Filter by Training Traffic:", traffic_opts, index=0)
            
            if sel_traffic != "All":
                show_df = df[df['Traffic'] == sel_traffic].copy()
            else:
                show_df = df.copy()
            
            # 2. Sort
            show_df = show_df.sort_values(by="Converged Reward (Last 50)", ascending=False).reset_index(drop=True)
            
            # 3. WINNER SECTION
            if not show_df.empty:
                best_row = show_df.iloc[0]
                best_name = best_row['Run Name']
                best_path = best_row['Full_Path']
                
                st.divider()
                st.subheader(f"🥇 Winner ({sel_traffic})")
                st.info(f"**Agent:** {best_name}  \n**Reward:** {best_row['Converged Reward (Last 50)']:.2f}")
                
                # 4. EVALUATION BUTTON
                if st.button(f"📊 Calculate Wait Times for '{best_name}'"):
                    with st.spinner("Running simulations on 4 scenarios (approx 10s)..."):
                        eval_results = evaluate_winner_scenarios(best_path)
                        
                        if "Error" in eval_results:
                            st.error(eval_results["Error"])
                        else:
                            # Display Metrics nicely
                            st.markdown("#### ⏱️ Average Wait Times (Lower is Better)")
                            c1, c2, c3, c4 = st.columns(4)
                            c1.metric("Low (1.0x)", f"{eval_results['Low (1.0x)']:.2f} s")
                            c2.metric("Normal (1.5x)", f"{eval_results['Normal (1.5x)']:.2f} s")
                            c3.metric("High (4.0x)", f"{eval_results['High (4.0x)']:.2f} s")
                            c4.metric("Saturated (6.0x)", f"{eval_results['Saturated (6.0x)']:.2f} s")
                            
                            # Comparison context
                            if eval_results['Normal (1.5x)'] < 15.0:
                                st.success("✅ This agent is beating the Adaptive Baseline (<30s)!")
                            elif eval_results['Normal (1.5x)'] < 35.0:
                                st.warning("⚠️ Average performance (Beating Fixed-Time).")
                            else:
                                st.error("❌ Poor performance (Worse than Fixed-Time).")

            st.divider()
            st.markdown("#### Full Leaderboard")
            st.dataframe(
                show_df.style.background_gradient(subset=["Converged Reward (Last 50)"], cmap="Greens"),
                column_config={"Full_Path": None},
                use_container_width=True
            )
        else:
            st.warning("No metrics.csv data found in this folder.")