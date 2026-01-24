import streamlit as st
import time
from traffic_rl.config import AppConfig
from traffic_rl.tools.batch_runner import run_batch_experiments
from traffic_rl.gui_utils import render_sidebar, load_config, save_config
try:
    from csv_to_tensorboard import sync_csv_to_tensorboard
except ImportError:
    sync_csv_to_tensorboard = None

st.set_page_config(page_title="Train Agents", layout="wide", page_icon="🏋️")
render_sidebar()

st.header("🏋️ Train Agents")

config_data = load_config("configs/default.yaml")
if not config_data: st.stop()

col_batch, col_conf = st.columns([1, 2])

with col_batch:
    st.subheader("1. Experiment Strategy")
    out_name = st.text_input("Experiment Name", value=f"experiments_gui_{int(time.time())}")
    
    st.write("**Traffic Scenarios:**")
    run_low = st.checkbox("Low (1.0x)", value=True)
    run_med = st.checkbox("Medium (1.5x)", value=False)
    run_high = st.checkbox("High (2.0x)", value=False)
    run_ext = st.checkbox("Extreme (4.0x)", value=False)
    
    multipliers = []
    if run_low: multipliers.append(1.0)
    if run_med: multipliers.append(1.5)
    if run_high: multipliers.append(2.0)
    if run_ext: multipliers.append(4.0)
    
    episodes_run = st.number_input("Episodes per Run", value=500, step=100)
    seeds = st.number_input("Seeds per Config", min_value=1, value=1)
    
    st.write("**Agents to Train:**")
    do_dqn = st.checkbox("DQN", value=True)
    do_q = st.checkbox("Q-Learning", value=False)

with col_conf:
    st.subheader("2. Hyperparameters")
    
    with st.expander("🤖 Network & Learning (DQN)", expanded=True):
        c1, c2 = st.columns(2)
        agent_conf = config_data.get('agent', {})
        
        new_lr = c1.number_input("Learning Rate", value=float(agent_conf.get('lr', 0.001)), format="%.6f")
        new_gamma = c2.number_input("Gamma", value=float(agent_conf.get('gamma', 0.99)))
        
        c3, c4 = st.columns(2)
        h_dims = [64, 128, 256, 512]
        curr_dim = int(agent_conf.get('hidden_dim', 128))
        idx_dim = h_dims.index(curr_dim) if curr_dim in h_dims else 1
        
        new_dim = c3.selectbox("Hidden Dim", h_dims, index=idx_dim)
        
        b_sizes = [32, 64, 128]
        curr_batch = int(agent_conf.get('batch_size', 64))
        idx_batch = b_sizes.index(curr_batch) if curr_batch in b_sizes else 1
        
        new_batch = c4.selectbox("Batch Size", b_sizes, index=idx_batch)
        
        c5, c6 = st.columns(2)
        new_buffer = c5.number_input("Replay Buffer", value=int(agent_conf.get('buffer_size', 50000)), step=10000)
        
        st.write("**DQN Architecture:**")
        ac1, ac2 = st.columns(2)
        use_double = ac1.checkbox("Double DQN", value=agent_conf.get('double_dqn', False))
        use_dueling = ac2.checkbox("Dueling DQN", value=agent_conf.get('dueling_dqn', False))

    with st.expander("🎲 Exploration (Epsilon)", expanded=False):
        e1, e2, e3 = st.columns(3)
        new_eps_start = e1.number_input("Start", value=float(agent_conf.get('epsilon_start', 1.0)), step=0.1)
        new_eps_min = e2.number_input("Min", value=float(agent_conf.get('epsilon_min', 0.01)), format="%.3f")
        new_eps_decay = e3.number_input("Decay", value=float(agent_conf.get('epsilon_decay', 0.995)), format="%.4f")

st.markdown("---")

agents_count = sum([do_dqn, do_q])
total_runs = len(multipliers) * seeds * agents_count

st.info(f"Total Runs Queued: {total_runs}")

if st.button(f"🚀 Start Batch Experiment", type="primary"):
    if total_runs == 0:
        st.error("Please select at least one Scenario and one Agent.")
    else:
        config_data['agent']['lr'] = new_lr
        config_data['agent']['gamma'] = new_gamma
        config_data['agent']['hidden_dim'] = new_dim
        config_data['agent']['batch_size'] = new_batch
        config_data['agent']['buffer_size'] = new_buffer
        config_data['agent']['double_dqn'] = use_double
        config_data['agent']['dueling_dqn'] = use_dueling
        config_data['agent']['epsilon_start'] = new_eps_start
        config_data['agent']['epsilon_min'] = new_eps_min
        config_data['agent']['epsilon_decay'] = new_eps_decay
        
        temp_path = save_config(config_data)
        final_conf = AppConfig.load(temp_path)
        
        with st.spinner(f"Running {total_runs} experiments..."):
            try:
                run_batch_experiments(final_conf, out_name, multipliers=multipliers, episodes=[episodes_run], seeds=seeds, run_dqn=do_dqn, run_q_learning=do_q)
                st.success("Batch Run Complete!")
                if sync_csv_to_tensorboard:
                    sync_csv_to_tensorboard(out_name, force=True)
            except Exception as e:
                st.error(f"Execution failed: {e}")