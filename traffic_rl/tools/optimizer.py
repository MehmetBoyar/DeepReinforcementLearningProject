import optuna
import torch
import numpy as np
import os
import yaml
from tqdm import tqdm

from traffic_rl.config import AppConfig, EnvConfig, AgentConfig, TrainConfig
from traffic_rl.env.traffic_env import TrafficLightEnv
from traffic_rl.agents.dqn import DQNAgent

# Fixed seed for Environment to ensure fair comparison
ENV_SEED = 42

def run_hyperparameter_optimization(base_config_path, n_trials, search_space):
    """
    Runs Optuna optimization based on a dynamic search space.
    """
    print(f"\n" + "="*60)
    print(f"STARTING DYNAMIC OPTIMIZATION")
    print(f"Trials: {n_trials}")
    print("="*60)

    # Load base config
    if os.path.exists(base_config_path):
        base_app_config = AppConfig.load(base_config_path)
    else:
        raise FileNotFoundError(f"Config not found at {base_config_path}")

    def objective(trial):
        # Start with defaults from config
        params = {
            'lr': base_app_config.agent.lr,
            'gamma': base_app_config.agent.gamma,
            'batch_size': base_app_config.agent.batch_size,
            'hidden_dim': base_app_config.agent.hidden_dim,
            'epsilon_decay': base_app_config.agent.epsilon_decay,
            'target_update': 100, # Default   
            'buffer_size': getattr(base_app_config.agent, 'buffer_size', 50000) 
        }

        # Override if enabled in search_space
        if search_space.get('lr', {}).get('enabled'):
            s = search_space['lr']
            params['lr'] = trial.suggest_float("lr", s['min'], s['max'], log=True)

        if search_space.get('gamma', {}).get('enabled'):
            s = search_space['gamma']
            params['gamma'] = trial.suggest_categorical("gamma", s['choices'])

        if search_space.get('batch_size', {}).get('enabled'):
            s = search_space['batch_size']
            params['batch_size'] = trial.suggest_categorical("batch_size", s['choices'])

        if search_space.get('hidden_dim', {}).get('enabled'):
            s = search_space['hidden_dim']
            params['hidden_dim'] = trial.suggest_categorical("hidden_dim", s['choices'])

        if search_space.get('epsilon_decay', {}).get('enabled'):
            s = search_space['epsilon_decay']
            params['epsilon_decay'] = trial.suggest_float("epsilon_decay", s['min'], s['max'])

        if search_space.get('target_update', {}).get('enabled'):
            s = search_space['target_update']
            params['target_update'] = trial.suggest_categorical("target_update", s['choices'])

        if search_space.get('buffer_size', {}).get('enabled'):
            s = search_space['buffer_size']
            params['buffer_size'] = trial.suggest_categorical("buffer_size", s['choices'])

        #  Setup Config (Stress Test: 1.3x Traffic) 
        env_conf = EnvConfig(
            arrival_rates=base_app_config.env.arrival_rates,
            traffic_multiplier=1.3, 
            max_steps=1000
        )
        
        # Short training run (150 eps) to fail fast
        train_conf = TrainConfig(n_episodes=150, eval_freq=999, seed=ENV_SEED, device="cuda")
        
        agent_conf = AgentConfig(
            name="dqn", 
            lr=params['lr'], 
            batch_size=params['batch_size'], 
            gamma=params['gamma'], 
            hidden_dim=params['hidden_dim'],
            buffer_size=params['buffer_size'], 
            epsilon_start=1.0, 
            epsilon_min=0.01, 
            epsilon_decay=params['epsilon_decay'],
            alpha=0.1,
            # Pass through architectural flags
            double_dqn=base_app_config.agent.double_dqn,
            dueling_dqn=base_app_config.agent.dueling_dqn
        )
        
        #  Run Training 
        env = TrafficLightEnv(config=env_conf)
        
        # CPU Fallback
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if train_conf.device == "cuda" and device == "cpu":
            train_conf.device = "cpu"

        agent = DQNAgent(14, 4, agent_conf, device=device)
        # set target update
        agent.target_update_freq = params['target_update']
        
        rewards = []
        pbar = tqdm(range(train_conf.n_episodes), desc=f"Trial {trial.number}", leave=False)
        
        for i in pbar:
            state, _ = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action = agent.act(state, training=True)
                next_state, reward, done, truncated, _ = env.step(action)
                done = done or truncated
                agent.update(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                
            agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
            rewards.append(total_reward)
            
            pbar.set_postfix({'rew': f"{total_reward:.0f}"})
            
            # Pruning (Kill bad trials early at ep 50)
            if i == 50:
                avg_50 = np.mean(rewards[-10:])
                trial.report(avg_50, 50)
                if trial.should_prune():
                    pbar.close()
                    raise optuna.TrialPruned()

        pbar.close()
        # Maximize Reward of last 20 episodes
        return np.mean(rewards[-20:])


    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    
    study.optimize(objective, n_trials=n_trials)
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print("Best Params found:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
        
    # Save best params
    output_path = "configs/best_params_found.yaml"
    with open(output_path, "w") as f:
        yaml.dump(study.best_params, f)
    print(f"\nSaved best parameters to: {output_path}")
    
    print("\n" + "="*60)
    print("OPTIMIZATION COMPLETE")
    print("="*60)
    print("Best Params found:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
        
    # Save best params
    output_path = "configs/best_params_found.yaml"
    with open(output_path, "w") as f:
        yaml.dump(study.best_params, f)
    print(f"\nSaved best parameters to: {output_path}")