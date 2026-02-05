# 🚦 Traffic Signal Control via Deep Reinforcement Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-ee4c2c)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b)
![License](https://img.shields.io/badge/License-MIT-green)

A modular research framework for optimizing traffic signal phases using **Deep Q-Networks (DQN)** and **Tabular Q-Learning**. This project provides a robust environment to train agents, visualize traffic flow, and compare performance against **Fixed-Time** and **Adaptive** (rule-based) baselines.

The project features a **Streamlit Dashboard** for GUI-based management, a **CLI** for batch processing, and a **Jupyter Notebook** for deep-dive analysis.

---

## 🌟 Key Features

### 🧠 Intelligent Agents
- **DQN:** Deep Q-Network with support for **Double DQN** and **Dueling Architecture**.
- **Q-Learning:** Classic tabular reinforcement learning with state discretization.
- **Baselines:** Includes **Fixed-Time** sequencers and **Adaptive** controllers (logic similar to SCATS/SCOOT).

### 🖥️ Interactive Dashboard
- **Train:** Queue batch experiments with varying traffic loads (Low, Med, High, Saturated).
- **Analyze:** Leaderboards, improvement heatmaps, and wait-time statistics.
- **Deep Dive:** Watch agents work via **GIF replay** and analyze lane fairness.
- **Optimize:** Hyperparameter tuning using **Optuna**.

### 📉 Advanced Metrics
- Logs Reward, Loss, Average Queue Length, and Throughput.
- **TensorBoard** integration (auto-converts CSV logs).

### 🔬 Custom Environment
- Gymnasium-based 4-phase intersection (N/S/E/W) with 12 lanes.
- Simulates "Rush Hour" dynamics using time-varying Poisson arrival rates.

---

## 📂 Project Structure

    /
    ├── Start_Dashboard.bat       # ⚡ Windows One-Click Launcher
    ├── Traffic_RL.py             # 🏠 Main Dashboard Entry Point
    ├── main.py                   # ⌨️ CLI Entry Point (Train/Batch/Optimize)
    ├── project.ipynb             # 📓 Jupyter Notebook for Manual Analysis
    ├── requirements.txt          # Dependencies
    ├── csv_to_tensorboard.py     # Tool to sync CSV logs to TB
    ├── configs/                  # YAML Configuration files
    ├── traffic_rl/               # 📦 Source Code
    │   ├── agents/               # DQN, Q-Learning, Baselines
    │   ├── env/                  # TrafficLightEnv (Gymnasium)
    │   ├── tools/                # Visualization, Analysis, Optuna
    │   └── core.py               # Training loops
    └── experiments/              # 💾 Output folder for Models & Logs

---

## ⚙️ Installation

### Option A: ⚡ Windows Quick Start (Recommended)
Simply double-click **`Start_Dashboard.bat`**. 
This script will automatically:
1. Check for Python.
2. Create the virtual environment.
3. Install all dependencies.
4. Launch the Dashboard in your browser.

### Option B: Manual Installation

1. **Clone the repository**

        git clone https://github.com/yourusername/traffic-rl.git
        cd traffic-rl

2. **Create a Virtual Environment**

        # Windows
        python -m venv venv
        venv\Scripts\activate

        # Mac/Linux
        python3 -m venv venv
        source venv/bin/activate

3. **Install Dependencies**

        pip install -r requirements.txt

---

## 🚀 Usage

### 1. The Dashboard
To launch the GUI manually:

    streamlit run Traffic_RL.py

- Navigate using the sidebar.
- **Train:** Select agents and traffic multipliers (e.g., 1.0x, 2.0x).
- **Compare:** Pit an "Old" experiment batch against a "New" one.
- **TensorBoard:** Launch TB directly from the sidebar to view live loss curves.

### 2. Command Line Interface (CLI)
Use `main.py` for headless training or scripting.

    # Train a single agent
    python main.py train --config configs/default.yaml

    # Run the full scientific batch (Low/Med/High traffic x Multiple Seeds)
    python main.py batch --out experiments_batch

    # Optimize Hyperparameters (Optuna)
    python main.py optimize --trials 50

    # Compare two result folders
    python main.py compare --old experiments/baseline --new experiments/dqn_v2

### 3. Jupyter Notebook (`project.ipynb`)
Use the notebook for granular analysis, specific model loading, and generating publication-ready plots.

1. Start Jupyter:
   
        jupyter notebook

2. Open `project.ipynb`.
3. **Features:**
   - **Universal Loader:** Automatically detects if a folder contains a DQN (`.pt`) or Q-Learning (`.pkl`) agent.
   - **Smoothing:** Plots training rewards with a rolling average window to visualize trends clearly.
   - **Visual Replay:** Generates and displays a GIF of the agent controlling the intersection inline.

---

## 🛠️ Configuration (`configs/default.yaml`)

You can control the environment and agent hyperparameters via YAML:

    environment:
      traffic_multiplier: 1.0  # 1.0 = Normal, 2.0 = Heavy
      max_steps: 1000          # Duration of one episode

    agent:
      name: "dqn"              # "dqn" or "q_learning"
      lr: 0.001                # Learning Rate
      batch_size: 64
      hidden_dim: 128
      double_dqn: true         # Enable Double DQN
      dueling_dqn: false       # Enable Dueling Architecture
      epsilon_decay: 0.995

---

## 📊 Monitoring & Results

### CSV to TensorBoard
The training loop logs metrics to `metrics.csv`. You can convert these to TensorBoard events:

    # Manual sync
    python csv_to_tensorboard.py --dir experiments_batch --force

Then run:

    tensorboard --logdir experiments_batch

### Output Files
Every experiment run generates:
- `metrics.csv`: Raw training data.
- `model.pt` / `model.pkl`: Final model weights.
- `config.json`: The specific config used for that run.
- `training_plot.png`: Static summary plot.

---

## 🤝 Contributing

Contributions are welcome! Please submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

