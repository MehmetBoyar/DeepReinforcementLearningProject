import os
import json
import csv
import time
from dataclasses import asdict
import matplotlib.pyplot as plt

class ExperimentLogger:
    def __init__(self, config, base_dir="experiments"):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.exp_dir = os.path.join(base_dir, f"{config.agent.name}_{timestamp}")
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # Save Config
        with open(os.path.join(self.exp_dir, "config.json"), "w") as f:
            json.dump(asdict(config), f, indent=4)
            
        # Init CSV
        self.csv_path = os.path.join(self.exp_dir, "metrics.csv")
        self.headers = ["episode", "reward", "avg_queue", "epsilon", "loss"]
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)

    def log_step(self, episode, reward, avg_queue, epsilon, loss=None):
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode, reward, avg_queue, epsilon, loss])
            
    def save_plot(self, rewards, queues):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(rewards)
        plt.title("Rewards")
        plt.subplot(1, 2, 2)
        plt.plot(queues)
        plt.title("Avg Queue Length")
        plt.savefig(os.path.join(self.exp_dir, "training_plot.png"))
        plt.close()
        
    def get_save_path(self, filename):
        return os.path.join(self.exp_dir, filename)