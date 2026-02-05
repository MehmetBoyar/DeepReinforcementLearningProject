import argparse
import os
import torch
from traffic_rl.config import AppConfig
from traffic_rl.env.traffic_env import TrafficLightEnv
from traffic_rl.agents.dqn import DQNAgent
from traffic_rl.agents.q_learning import QLearningAgent
from traffic_rl.logger import ExperimentLogger
from traffic_rl.core import train

# Import Tools
from traffic_rl.tools.optimizer import run_hyperparameter_optimization
from traffic_rl.tools.batch_runner import run_batch_experiments
from traffic_rl.tools.analyzer import run_analysis_suite
from traffic_rl.tools.comparator import run_comparison_suite

def run_single_train(args):
    """
    Logic for training a SINGLE agent based on a config file.
    Useful for testing or specific runs outside the batch loop.
    """
    print(f"\n--- Starting Single Training Run ---")
    print(f"Config: {args.config}")
    
    config = AppConfig.load(args.config)
    
    # Override seed if provided via CLI
    if args.seed is not None:
        config.train.seed = args.seed
        
    # Initialize Environment
    env = TrafficLightEnv(config.env)
    
    # Initialize Agent based on Config Name
    if config.agent.name == "dqn":
        print(f"Initializing DQN (LR={config.agent.lr}, Dim={config.agent.hidden_dim})...")
        agent = DQNAgent(
            env.observation_space.shape[0], 
            env.action_space.n, 
            config.agent, 
            device=config.train.device
        )
    elif config.agent.name == "q_learning":
        print(f"Initializing Q-Learning (Alpha={config.agent.alpha})...")
        agent = QLearningAgent(env.action_space.n, config.agent)
    else:
        raise ValueError(f"Unknown agent name in config: {config.agent.name}")

    # Setup Logger
    exp_name = f"{config.agent.name}_single_run"
    logger = ExperimentLogger(config, base_dir=os.path.join("experiments_single", exp_name))
    
    # Run Training
    train(env, agent, config, logger)
    print(f"Training Complete. Results saved to: {logger.exp_dir}")

def main():
    parser = argparse.ArgumentParser(description="Traffic RL Research Framework")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # TRAIN Command 
    p_train = subparsers.add_parser("train", help="Train a single agent")
    p_train.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    p_train.add_argument("--seed", type=int, default=None, help="Override random seed")

    # 2. BATCH Command 
    p_batch = subparsers.add_parser("batch", help="Run the 9-experiment scientific suite")
    p_batch.add_argument("--config", default="configs/default.yaml", help="Base config to use")
    p_batch.add_argument("--out", default="experiments_batch", help="Output folder name")

    # OPTIMIZE Command 
    p_opt = subparsers.add_parser("optimize", help="Run Hyperparameter Tuning")
    p_opt.add_argument("--trials", type=int, default=25, help="Number of trials")
    p_opt.add_argument("--config", default="configs/default.yaml", help="Base config")

    # ANALYZE Command 
    p_an = subparsers.add_parser("analyze", help="Generate tables and plots for a result folder")
    p_an.add_argument("--dir", required=True, help="Folder containing experiment results")
    p_an.add_argument("--config", default="configs/default.yaml", help="Config for env setup")

    # COMPARE Command 
    p_comp = subparsers.add_parser("compare", help="Compare two experiment folders side-by-side")
    p_comp.add_argument("--old", required=True, help="Baseline/Old Folder")
    p_comp.add_argument("--new", required=True, help="Optimized/New Folder")
    p_comp.add_argument("--dim_old", type=int, default=128, help="Hidden Dim of Old models")
    p_comp.add_argument("--dim_new", type=int, default=128, help="Hidden Dim of New models")
    p_comp.add_argument("--config", default="configs/default.yaml")

    args = parser.parse_args()
    
    # Dispatch Logic
    if args.command == "train":
        if os.path.exists(args.config):
            run_single_train(args)
        else:
            print(f"Error: Config file '{args.config}' not found.")

    elif args.command == "batch":
        if os.path.exists(args.config):
            config = AppConfig.load(args.config)
            run_batch_experiments(config, args.out)
        else:
            print(f"Error: Config file '{args.config}' not found.")

    elif args.command == "optimize":
        run_hyperparameter_optimization(args.config, args.trials)

    elif args.command == "analyze":
        if os.path.exists(args.config):
            config = AppConfig.load(args.config)
            run_analysis_suite(args.dir, config)
        else:
            print(f"Error: Config file '{args.config}' not found.")

    elif args.command == "compare":
        if os.path.exists(args.config):
            config = AppConfig.load(args.config)
            run_comparison_suite(args.old, args.new, config, args.dim_old, args.dim_new)
        else:
            print(f"Error: Config file '{args.config}' not found.")

if __name__ == "__main__":
    main()