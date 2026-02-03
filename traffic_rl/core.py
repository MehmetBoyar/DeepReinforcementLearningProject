import numpy as np
import os
from tqdm import tqdm

def run_episode(env, agent, training=True):
    """
    Runs a single episode of simulation.
    Returns: (total_reward, metrics_dict)
    """
    state, _ = env.reset()
    total_reward = 0
    total_queue = 0
    total_throughput = 0
    step_count = 0
    
    done = False
    while not done:
        action = agent.act(state, training=training)
        next_state, reward, done, truncated, info = env.step(action)
        done = done or truncated
        
        if training and hasattr(agent, 'update'):
            agent.update(state, action, reward, next_state, done)
            
        state = next_state
        total_reward += reward
        
        # Accumulate metrics
        total_queue += info.get('total_queue', 0)
        total_throughput += info.get('throughput', 0)
        step_count += 1
        
    metrics = {
        "avg_queue": total_queue / step_count if step_count > 0 else 0,
        "throughput": total_throughput
    }
    return total_reward, metrics

def train(env, agent, config, logger=None):
    """
    Main training loop with progress bar, logging, and best-model saving.
    """
    rewards_history = []
    queue_history = []
    
    # Track performance to save checkpoints
    best_avg_reward = -float('inf')
    
    # Progress Bar
    pbar = tqdm(range(config.train.n_episodes), desc=f"Training {config.agent.name}")
    
    for episode in pbar:
        reward, metrics = run_episode(env, agent, training=True)
        rewards_history.append(reward)
        queue_history.append(metrics['avg_queue'])
        
        eps = getattr(agent, 'epsilon', 0.0)
        
        if logger:
            # Pass loss if available 
            # For strict loss logging, run_episode needs to return avg_loss. 
            # For now, we log None for loss to keep it simple.
            logger.log_step(episode, reward, metrics['avg_queue'], eps, loss=None)
            
        # 2. Update Progress Bar
        pbar.set_postfix({
            'rew': f"{reward:.0f}", 
            'eps': f"{eps:.2f}",
            'q': f"{metrics['avg_queue']:.1f}"
        })

        # We use a moving average of last 10 episodes to determine stability
        if len(rewards_history) >= 10:
            avg_recent = np.mean(rewards_history[-10:])
            
            # Only save if we beat the record AND we are past the chaos of early exploration
            if avg_recent > best_avg_reward and episode > 50:
                best_avg_reward = avg_recent
                if logger:
                    # Save as "best_model.pt"
                    save_name = "best_model.pt" if config.agent.name == "dqn" else "best_model.pkl"
                    agent.save(logger.get_save_path(save_name))

    # End of Training
    if logger:
        # Save plots
        logger.save_plot(rewards_history, queue_history)
        
        # Save Final Model (Standard name)
        final_name = "model.pt" if config.agent.name == "dqn" else "model.pkl"
        agent.save(logger.get_save_path(final_name))
        
        # Close TensorBoard writer
        if hasattr(logger, 'close'):
            logger.close()
        
    return rewards_history