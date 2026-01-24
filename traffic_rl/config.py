import yaml
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class EnvConfig:
    arrival_rates: Dict[str, float]
    max_steps: int
    traffic_multiplier: float

@dataclass
class AgentConfig:
    name: str
    gamma: float
    lr: float
    batch_size: int
    hidden_dim: int
    buffer_size: int
    epsilon_start: float
    epsilon_min: float
    epsilon_decay: float
    alpha: float
    double_dqn: bool = False
    dueling_dqn: bool = False 

@dataclass
class TrainConfig:
    n_episodes: int
    eval_freq: int
    seed: int
    device: str

@dataclass
class AppConfig:
    env: EnvConfig
    agent: AgentConfig
    train: TrainConfig

    @classmethod
    def load(cls, path: str) -> "AppConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        
        # Safe loading for the new flag
        agent_data = data['agent']
        if 'double_dqn' not in agent_data:
            agent_data['double_dqn'] = False

        return cls(
            env=EnvConfig(**data['environment']),
            agent=AgentConfig(**agent_data),
            train=TrainConfig(**data['train'])
        )