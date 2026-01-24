import streamlit as st
import os
import yaml
import subprocess
import signal

# Try importing the sync tool
try:
    from csv_to_tensorboard import sync_csv_to_tensorboard
except ImportError:
    sync_csv_to_tensorboard = None

def get_experiment_folders():
    if not os.path.exists("."): return []
    # Filter for folders that look like experiment outputs
    folders = [d for d in os.listdir(".") if os.path.isdir(d) and ("experiments" in d or "Exp" in d or "dqn" in d)]
    return sorted(folders, reverse=True)

def load_config(path="configs/default.yaml"):
    if not os.path.exists(path):
        st.error(f"Config file not found: {path}")
        return None
    with open(path, "r") as f: return yaml.safe_load(f)

def save_config(data, path="configs/temp_gui_config.yaml"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f: yaml.dump(data, f)
    return path

def render_sidebar():
    """Renders the common sidebar for all pages."""
    st.sidebar.title("🚦 Traffic RL")
    st.sidebar.markdown("---")
    
    st.sidebar.subheader("📈 Live Monitoring")

    # 1. Sync Button
    if st.sidebar.button("🔄 Sync CSV to TensorBoard"):
        if sync_csv_to_tensorboard:
            with st.spinner("Translating CSV logs..."):
                sync_csv_to_tensorboard(".", force=True)
            st.sidebar.success("Sync Complete!")
        else:
            st.sidebar.error("csv_to_tensorboard.py missing!")

    # 2. TensorBoard Launcher
    if 'tb_pid' not in st.session_state: 
        st.session_state['tb_pid'] = None

    col_tb1, col_tb2 = st.sidebar.columns(2)

    with col_tb1:
        if st.button("🚀 Launch TB"):
            if st.session_state['tb_pid'] is None:
                try:
                    proc = subprocess.Popen(
                        ["tensorboard", "--logdir", ".", "--port", "6006"], 
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE
                    )
                    st.session_state['tb_pid'] = proc.pid
                    st.sidebar.success("Started!")
                except Exception as e: 
                    st.sidebar.error(f"Error: {e}")
            else: 
                st.sidebar.info("Running...")

    with col_tb2:
        if st.button("💀 Kill TB"):
            if st.session_state['tb_pid']:
                try:
                    os.kill(st.session_state['tb_pid'], signal.SIGTERM)
                    st.session_state['tb_pid'] = None
                    st.sidebar.warning("Stopped.")
                except:
                    st.session_state['tb_pid'] = None

    if st.session_state['tb_pid']:
        st.sidebar.markdown("[👉 Open TensorBoard](http://localhost:6006)")
    else:
        st.sidebar.caption("TensorBoard is stopped.")
    
    st.sidebar.markdown("---")