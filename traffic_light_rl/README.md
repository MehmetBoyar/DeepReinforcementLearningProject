# Traffic Light Control with Deep Reinforcement Learning

A comprehensive implementation of traffic signal control at a 4-way intersection using Reinforcement Learning techniques.

## Overview

This project implements and compares different approaches to traffic light control:

- **Q-Learning**: Tabular reinforcement learning with state discretization
- **DQN**: Deep Q-Network with neural network function approximation
- **Fixed-Time**: Traditional fixed-duration phase cycling (baseline)
- **Adaptive**: Queue-based adaptive control (baseline)

## Project Structure

```
traffic_light_rl/
├── env/                    # Environment implementation
│   ├── traffic_env.py      # Main Gym environment
│   ├── phases.py           # 4-phase system definitions
│   └── traffic_generator.py # Vehicle arrival simulation
├── agents/                 # RL agents
│   ├── q_learning.py       # Tabular Q-Learning
│   ├── dqn.py              # Deep Q-Network
│   └── replay_buffer.py    # Experience replay
├── baselines/              # Baseline controllers
│   ├── fixed_time.py       # Fixed-time control
│   └── adaptive.py         # Adaptive control
├── utils/                  # Utilities
│   ├── visualization.py    # Plotting functions
│   └── metrics.py          # Evaluation metrics
├── config/
│   └── config.yaml         # Hyperparameters
├── main.py                 # Main script for Spyder
├── train.py                # Command-line training
├── evaluate.py             # Command-line evaluation
└── requirements.txt        # Dependencies
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install specific packages
pip install numpy gym torch matplotlib pyyaml
```

## Quick Start

### For Spyder IDE (Recommended)

1. Open `main.py` in Spyder
2. Set working directory to the Project folder
3. Press **F5** to run

Configure training at the top of `main.py`:
```python
N_EPISODES = 500          # Training episodes
TRAIN_QLEARNING = True    # Train Q-Learning
TRAIN_DQN = True          # Train DQN (requires PyTorch)
```

### For Command Line

```bash
# Train both Q-Learning and DQN
python -m traffic_light_rl.train --episodes 1000

# Evaluate all agents
python -m traffic_light_rl.evaluate --demo
```

## Environment Details

### State Space (14 dimensions)

| Index | Variable | Description | Range |
|-------|----------|-------------|-------|
| 0-2 | queue_N_to_* | North approach queues | [0, 50] |
| 3-5 | queue_S_to_* | South approach queues | [0, 50] |
| 6-8 | queue_E_to_* | East approach queues | [0, 50] |
| 9-11 | queue_W_to_* | West approach queues | [0, 50] |
| 12 | current_phase | Current signal phase | [0, 3] |
| 13 | phase_duration | Time in current phase | [0, 120] |

### Action Space (4 actions)

| Action | Name | Description |
|--------|------|-------------|
| 0 | KEEP | Stay in current phase |
| 1 | NEXT | Move to next phase (cyclic) |
| 2 | SKIP_TO_NS | Jump to Phase 0 (N-S through) |
| 3 | SKIP_TO_EW | Jump to Phase 2 (E-W through) |

### Phase Definitions

| Phase | Name | Allowed Movements |
|-------|------|-------------------|
| 0 | N-S Through | N→S, S→N, N→W, S→E |
| 1 | N-S Protected Left | N→W, S→E, N→E, S→W |
| 2 | E-W Through | E→W, W→E, E→N, W→S |
| 3 | E-W Protected Left | E→N, W→S, E→S, W→N |

### Reward Function

```
R = -0.1 × total_queue        # Queue penalty
  + 0.5 × vehicles_passed     # Throughput reward
  - 1.0 × phase_switch        # Switch penalty
  - 0.05 × max_queue²         # Fairness penalty
```

## MDP Formulation

### Transition Dynamics

The environment follows Markov Decision Process dynamics:

1. **Stochastic Arrivals**: New vehicles arrive following Poisson distribution
   ```
   arrivals[movement] ~ Poisson(λ[movement] × dt)
   ```

2. **Deterministic Phase Transitions**: Phase changes are deterministic based on action

3. **Capacity-Based Departures**: Vehicles depart based on movement capacity
   ```
   departures = min(queue, capacity[movement_type])
   ```

### Q-Learning Update Rule

```
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
```

### DQN Loss Function

```
L = E[(r + γ max_a' Q_target(s',a') - Q(s,a))²]
```

## Configuration

Edit `config/config.yaml` to customize:

- Environment parameters (arrival rates, capacities)
- Training hyperparameters (learning rate, epsilon decay)
- Evaluation settings
- Baseline configurations

## Results

Expected performance ranking (after training):
1. **DQN** - Best overall performance with proper hyperparameter tuning
2. **Q-Learning** - Good performance with appropriate discretization
3. **Adaptive** - Reasonable performance without learning
4. **Fixed-Time** - Baseline performance

## Requirements

- Python 3.8+
- NumPy
- Gym
- PyTorch (for DQN)
- Matplotlib (for visualization)
- PyYAML (for configuration)

## License

This project is developed for educational purposes as part of the TUM Deep Reinforcement Learning course.
