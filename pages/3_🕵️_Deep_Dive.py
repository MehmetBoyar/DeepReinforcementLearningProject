import streamlit as st
import os
import glob
from traffic_rl.config import AppConfig
from traffic_rl.env.traffic_env import TrafficLightEnv
from traffic_rl.agents.dqn import DQNAgent
from traffic_rl.agents.q_learning import QLearningAgent
from traffic_rl.tools.visualizer import generate_gif
from traffic_rl.tools.behavior import analyze_behavior, plot_behavior
from traffic_rl.gui_utils import render_sidebar, get_experiment_folders

st.set_page_config(page_title="Agent Deep Dive", layout="wide", page_icon="🕵️")
render_sidebar()

st.header("🕵️ Agent Deep Dive")
st.markdown("Visualize replay and analyze behavioral patterns (fairness, decision distribution).")

folders = get_experiment_folders()
if not folders:
    st.warning("No experiments found.")
    st.stop()

# 1. Select Main Batch Folder
folder = st.selectbox("Select Experiment Folder", folders)

# 2. Select Sub-Experiment (The specific run)
subfolders = [f.name for f in os.scandir(folder) if f.is_dir() and "Exp" in f.name]
target_dir = folder

c_sel1, c_sel2 = st.columns(2)
with c_sel1:
    if subfolders:
        # Natural sort (Exp1, Exp2, Exp10...)
        subfolders.sort(key=lambda x: int(x.split('Exp')[1].split('_')[0]) if 'Exp' in x else 999)
        selected_sub = st.selectbox("1. Select Sub-Experiment", subfolders)
        target_dir = os.path.join(folder, selected_sub)
    else:
        st.caption("Using root folder (Single Experiment)")

# 3. Select Model File (Recursive Search)
# FIX: Look recursively into subfolders (e.g. dqn_timestamp/model.pt)
found_files = glob.glob(os.path.join(target_dir, "**", "*.pt"), recursive=True) + \
              glob.glob(os.path.join(target_dir, "**", "*.pkl"), recursive=True)

selected_model_path = None
with c_sel2:
    if found_files:
        # Create friendly names like "dqn_2026... / best_model.pt"
        file_options = {
            f"{os.path.basename(os.path.dirname(p))} / {os.path.basename(p)}": p 
            for p in found_files
        }
        
        # Auto-select 'best_model.pt' if available
        default_idx = 0
        keys_list = list(file_options.keys())
        for i, k in enumerate(keys_list):
            if "best_model.pt" in k:
                default_idx = i
                break
        
        selected_label = st.selectbox("2. Select Agent Model", keys_list, index=default_idx)
        selected_model_path = file_options[selected_label]
    else:
        st.warning(f"No .pt or .pkl model files found inside `{os.path.basename(target_dir)}`")

# 4. Load & Visualize
if selected_model_path:
    is_dqn = selected_model_path.endswith(".pt")
    agent_type = "DQN" if is_dqn else "Q-Learning"
    
    st.info(f"Loaded **{agent_type}** from `{selected_label}`")

    def load_selected_agent():
        conf = AppConfig.load("configs/default.yaml")
        
        # Try to recover specific architecture from config.json in the agent's specific folder
        agent_dir = os.path.dirname(selected_model_path)
        config_json = os.path.join(agent_dir, "config.json")
        
        if os.path.exists(config_json):
            import json
            try:
                with open(config_json) as f: saved = json.load(f)
                agent_data = saved.get('agent', {})
                if is_dqn:
                    conf.agent.hidden_dim = agent_data.get('hidden_dim', 128)
                    conf.agent.double_dqn = agent_data.get('double_dqn', False)
                    conf.agent.dueling_dqn = agent_data.get('dueling_dqn', False)
            except:
                pass # Fallback to default if JSON invalid
        
        # Setup Environment (1.3x Traffic for stress test)
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

    # --- TABS FOR ANALYSIS ---
    tab_vis, tab_beh = st.tabs(["🎥 Visual Replay", "🧠 Behavioral Analysis"])
    
    # TAB 1: GIF REPLAY
    with tab_vis:
        col_v1, col_v2 = st.columns([1, 2])
        with col_v1:
            st.write("Generate a visual replay of the agent controlling traffic (1.3x Load).")
            if st.button("Generate Replay GIF"):
                with st.spinner("Simulating episode..."):
                    env, agent = load_selected_agent()
                    if env and agent:
                        gif_path = os.path.join(os.path.dirname(selected_model_path), "replay_gui.gif")
                        generate_gif(env, agent, gif_path)
                        st.session_state['last_gif'] = gif_path
        
        with col_v2:
            # Check for GIF in session state or file system
            display_gif = st.session_state.get('last_gif', None)
            
            # If no new gif generated, look for existing one in folder
            if not display_gif and os.path.exists(os.path.join(os.path.dirname(selected_model_path), "replay_gui.gif")):
                 display_gif = os.path.join(os.path.dirname(selected_model_path), "replay_gui.gif")

            if display_gif and os.path.exists(display_gif):
                st.image(display_gif, caption=f"{agent_type} Replay", width=600)

    # TAB 2: BEHAVIORAL ANALYSIS
    with tab_beh:
        st.write("Analyze the agent's decisions and fairness over 5 episodes.")
        
        if st.button("Run Behavior Check"):
            with st.spinner("Analyzing decisions..."):
                env, agent = load_selected_agent()
                if env and agent:
                    df_act, df_fair = analyze_behavior(env, agent, episodes=5)
                    
                    fig1, fig2 = plot_behavior(df_act, df_fair)
                    
                    c_beh1, c_beh2 = st.columns([1, 2])
                    
                    with c_beh1:
                        st.markdown("##### Decision Distribution")
                        st.dataframe(df_act, width=None) # Auto width
                        
                    with c_beh2:
                        st.pyplot(fig1)
                    
                    st.markdown("---")
                    st.markdown("##### Fairness Check (Max Queue Lengths)")
                    st.caption("High bars indicate specific lanes are being neglected (Gridlock risk). Ideally, bars should be roughly equal.")
                    st.pyplot(fig2)