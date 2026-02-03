import time
import os
import gc
import torch
import traceback
import random
import numpy as np
from traffic_rl.env.traffic_env import TrafficLightEnv
from traffic_rl.agents.dqn import DQNAgent
from traffic_rl.agents.q_learning import QLearningAgent
from traffic_rl.logger import ExperimentLogger
from traffic_rl.core import train

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def run_batch_experiments(base_config, output_folder, multipliers=[1.0], episodes=[1000], seeds=1, 
                          run_dqn=True, run_q_learning=True):
    """
    Runs a dynamic batch of experiments with selectable agents.
    """
    
    # Generate the list of experiments
    experiments = []
    exp_id = 1
    
    for mult in multipliers:
        for eps in episodes:
            for s_idx in range(seeds):
                experiments.append({
                    "id": exp_id,
                    "mult": mult,
                    "eps": eps,
                    "seed_offset": s_idx
                })
                exp_id += 1
    
    print(f"\n" + "="*60)
    print(f"STARTING DYNAMIC BATCH RUN")
    print(f"Output: {output_folder}")
    print(f"Total Configs: {len(experiments)}")
    print(f"Agents: DQN={run_dqn}, Q-Learn={run_q_learning}")
    print("="*60)
    
    start_time = time.time()
    
    for exp in experiments:
        exp_name = f"Exp{exp['id']}_Mult{exp['mult']}x_Eps{exp['eps']}_Seed{exp['seed_offset']}"
        print(f"\nRunning {exp_name} ({exp['id']}/{len(experiments)})")

        # Setup Config
        current_config = base_config
        current_config.train.n_episodes = exp['eps']
        current_config.env.traffic_multiplier = exp['mult']
        run_seed = current_config.train.seed + exp['seed_offset']
        set_seed(run_seed)

        # DQN AGENT
        if run_dqn:
            try:
                current_config.agent.name = "dqn"
                env = TrafficLightEnv(current_config.env)
                agent = DQNAgent(
                    env.observation_space.shape[0], 
                    env.action_space.n, 
                    current_config.agent, 
                    device=current_config.train.device
                )
                logger = ExperimentLogger(current_config, base_dir=os.path.join(output_folder, exp_name))
                train(env, agent, current_config, logger)
                
                del env, agent, logger
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                print(f"  !!! DQN Failed: {e}")
                traceback.print_exc()

        # Q-LEARNING AGENT
        if run_q_learning:
            try:
                current_config.agent.name = "q_learning"
                env = TrafficLightEnv(current_config.env)
                agent = QLearningAgent(env.action_space.n, current_config.agent)
                logger = ExperimentLogger(current_config, base_dir=os.path.join(output_folder, exp_name))
                train(env, agent, current_config, logger)
                
                del env, agent, logger
                gc.collect()
            except Exception as e:
                print(f"  !!! Q-Learning Failed: {e}")

    duration = (time.time() - start_time) / 3600
    print(f"\nBATCH COMPLETE. Duration: {duration:.2f} hours.")