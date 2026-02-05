import streamlit as st
from traffic_rl.gui_utils import render_sidebar

# Page Config
st.set_page_config(
    page_title="Traffic RL", 
    layout="wide", 
    page_icon="🚦",
    initial_sidebar_state="expanded"
)

# Render the common sidebar
render_sidebar()

# --- MAIN HOME PAGE CONTENT ---
st.title("🚦 Traffic Control Reinforcement Learning")

st.markdown("""
### 🤖 Research Dashboard

Welcome! This dashboard allows you to train, analyze, and visualize Reinforcement Learning agents for traffic signal control.

---

### 📚 How to use this App:

#### 1. 🏋️ Train
Go here to run new experiments. You can run **batch experiments** (e.g., Low vs High traffic) to generate data.
* **Tip:** Start with "Low (1.0x)" traffic and a "DQN" agent for a quick test.

#### 2. 📊 Analysis
Once you have trained agents, go here to see the **Leaderboard**.
* View average wait times.
* See improvement heatmaps vs Baselines.

#### 3. 🕵️ Deep Dive
Want to see *why* an agent is acting a certain way?
* **Visual Replay:** Watch a GIF of the agent controlling the intersection.
* **Behavior Check:** See if the agent is ignoring specific lanes (Fairness).

#### 4. ⚖️ Compare
Head-to-head comparison of two different experiment runs (e.g., "Old Model" vs "New Model").

#### 5. 🧪 Optimize
Use **Optuna** to automatically find the best Hyperparameters (Learning Rate, Gamma, etc.) for your specific traffic scenario.

---
""")

st.info("💡 **Tip:** Ensure `configs/default.yaml` exists before starting training.")