# 🚦 Adaptive Traffic Signal Control (RL)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-ee4c2c)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b)
![License](https://img.shields.io/badge/License-MIT-green)

A comprehensive Research Framework for optimizing traffic signal phases using **Deep Reinforcement Learning (DQN)** and **Q-Learning**. 

This project features a fully interactive **Streamlit Dashboard** for training agents, analyzing performance metrics, visualizing real-time behavior, and comparing experimental results.

---

## 🌟 Features

### 🖥️ Interactive Dashboard
A multi-page GUI to manage the entire lifecycle of the research project.
- **🏋️ Train:** Launch batch experiments with varying traffic loads (Low, Med, High, Extreme).
- **📊 Analysis:** View leaderboards, improvement heatmaps, and wait-time statistics.
- **🕵️ Deep Dive:** Replay episodes via **GIF visualization** and analyze fairness (queue distribution).
- **⚖️ Compare:** Head-to-head comparison of two model versions (e.g., Baseline vs. Optimized).
- **🧪 Optimize:** Automated hyperparameter tuning using **Optuna**.

### 🧠 Intelligent Agents
- **Deep Q-Network (DQN):** Supports Double DQN, Dueling DQN, and Experience Replay.
- **Q-Learning:** Tabular RL approach for comparison.
- **Baselines:** Fixed-Time and Rule-Based Adaptive controllers.

### 📈 Monitoring
- **TensorBoard Integration:** Real-time tracking of Reward, Loss, and Average Queue Length.
- **Custom Metrics:** Tracks throughput, wait times, and lane fairness.

---

## 📸 Screenshots

| Training Dashboard | Visual Replay |
|:---:|:---:|
| *(Add a screenshot of your Train page here)* | *(Add a GIF of your traffic simulation here)* |

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/traffic-rl.git
   cd traffic-rl

2. **Create a Virtual Environment (Recommended)**
    # Windows
    python -m venv venv
    venv\Scripts\activate

    # Mac/Linux
    python3 -m venv venv
    source venv/bin/activate
pip install -r requirements.txt

## 🚀 How to Run
**Option 1: The Dashboard (Recommended)**

This launches the GUI in your default web browser.

streamlit run Traffic_RL.py

**Option 2: Command Line (CLI)**

You can also run experiments directly from the terminal.

# Train a single agent
python main.py train --config configs/default.yaml

# Run the full scientific batch (Low/Med/High traffic)
python main.py batch --out experiments_batch

## 📂 Project Structure
```text
/
├── Traffic_RL.py           # 🏠 Main Dashboard Entry Point
├── pages/                  # 📄 Streamlit Pages
│   ├── 1_🏋️_Train.py
│   ├── 2_📊_Analysis.py
│   ├── 3_🕵️_Deep_Dive.py
│   ├── 4_⚖️_Compare.py
│   └── 5_🧪_Optimize.py
├── traffic_rl/             # 📦 Core Package
│   ├── agents/             # RL Agent Logic (DQN, Q-Learning)
│   ├── env/                # Gymnasium Environment (Traffic Logic)
│   └── tools/              # Analysis & Visualization Tools
├── configs/                # ⚙️ Configuration Files
├── experiments/            # 💾 Saved Models & Logs
└── requirements.txt        # Dependencies


## 🛠️ Configuration

environment:
  traffic_multiplier: 1.0  # 1.0 = Normal, 2.0 = High Traffic
  max_steps: 1000          # Duration of one episode

agent:
  name: "dqn"
  lr: 0.001
  batch_size: 64
  hidden_dim: 128
  double_dqn: true         # Enable Double DQN stability

## 📈 TensorBoard

To view live training metrics (Loss, Reward, Queue Lengths):

    Click "🚀 Launch TB" in the Dashboard Sidebar.

    Or run manually:
    code Bash

    tensorboard --logdir . --port 6006

    Open http://localhost:6006

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details. Code
---