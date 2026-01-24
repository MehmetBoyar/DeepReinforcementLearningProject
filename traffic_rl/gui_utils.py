import streamlit as st
import os
import yaml
import subprocess
import signal
import sys
import time

# Try importing the sync tool
try:
    from csv_to_tensorboard import sync_csv_to_tensorboard
except ImportError:
    sync_csv_to_tensorboard = None

def get_experiment_folders():
    if not os.path.exists("."): return []
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

# --- HELPER: Aggressively Kill TensorBoard ---
def kill_existing_tensorboard():
    """Kills any process named tensorboard.exe to free up Port 6006"""
    if os.name == 'nt':
        try:
            # Force kill any tensorboard.exe
            subprocess.run(["taskkill", "/F", "/IM", "tensorboard.exe"], 
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

def render_sidebar():
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
    col_tb1, col_tb2 = st.sidebar.columns(2)

    # We use a file to track if it's running because SessionState can be flaky with subprocesses
    log_file = "tb_launch_log.txt"

    with col_tb1:
        if st.button("🚀 Launch TB"):
            try:
                # A. CLEANUP: Kill any zombies first (The "Centralized" Fix)
                kill_existing_tensorboard()
                time.sleep(0.5)

                # B. PREPARE LOGS: Open a file to catch errors
                with open(log_file, "w") as out:
                    # C. LAUNCH: Use sys.executable to ensure we use the venv python
                    cmd = [
                        sys.executable, "-m", "tensorboard.main", 
                        "--logdir", ".", 
                        "--port", "6006", 
                        "--host", "127.0.0.1" 
                    ]
                    
                    # Start detached process
                    subprocess.Popen(cmd, stdout=out, stderr=out, cwd=os.getcwd())
                
                # Wait a moment for it to spin up or fail
                time.sleep(2)
                st.session_state['tb_active'] = True
                st.rerun()
                
            except Exception as e:
                st.sidebar.error(f"Launch Error: {e}")

    with col_tb2:
        if st.button("💀 Kill TB"):
            kill_existing_tensorboard()
            st.session_state['tb_active'] = False
            st.rerun()

    # 3. Status Check & Link
    # Check the log file to see if it actually started
    is_running = st.session_state.get('tb_active', False)
    
    if is_running:
        # Quick check of the log file for errors
        if os.path.exists(log_file):
            with open(log_file, "r") as f:
                content = f.read()
                if "Address already in use" in content:
                    st.sidebar.error("❌ Port 6006 Busy. Click 'Kill TB' and try again.")
                elif "error" in content.lower() and "deprecated" not in content.lower():
                    # Ignore deprecation warnings, show real errors
                    st.sidebar.warning("⚠️ TB might have failed. Check logs below.")
                    with st.sidebar.expander("Log Output"):
                        st.text(content)
        
        st.sidebar.success("✅ Running")
        st.sidebar.markdown("[👉 **Open Dashboard**](http://127.0.0.1:6006)")
    else:
        st.sidebar.caption("TensorBoard is stopped.")
    
    st.sidebar.markdown("---")