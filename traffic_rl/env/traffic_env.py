import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List

from traffic_rl.config import EnvConfig

class TrafficLightEnv(gym.Env):
    """
    Traffic Light Control Environment.
    Matches the physics and logic of the original 'Old Code' implementation.
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    # Array Index -> Movement Name
    MOVEMENT_INDICES = [
        'N_to_S', 'N_to_E', 'N_to_W',  # 0, 1, 2
        'S_to_N', 'S_to_W', 'S_to_E',  # 3, 4, 5
        'E_to_W', 'E_to_S', 'E_to_N',  # 6, 7, 8
        'W_to_E', 'W_to_N', 'W_to_S'   # 9, 10, 11
    ]

    def __init__(self, config: EnvConfig):
        super().__init__()
        self.config = config
        
        # Setup Base Arrival Rates (Before Rush Hour Logic)
        self.base_arrival_rates = []
        for key in self.MOVEMENT_INDICES:
            base_rate = config.arrival_rates.get(key, 0.0)
            # Apply the global multiplier 
            self.base_arrival_rates.append(base_rate * config.traffic_multiplier)
        self.base_arrival_rates = np.array(self.base_arrival_rates, dtype=np.float32)

        # Define Capacities (Cars per step) 
        # Indices: 0=Straight, 1=Left, 2=Right
        # Pattern repeats: [Straight, Left, Right] for N, S, E, W
        self.lane_capacities = np.array([
            2.0, 1.5, 1.0,  # N
            2.0, 1.5, 1.0,  # S
            2.0, 1.5, 1.0,  # E
            2.0, 1.5, 1.0   # W
        ], dtype=np.float32)

        # Define Spaces
        self.observation_space = spaces.Box(
            low=0, 
            high=np.inf, 
            shape=(14,), 
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)
        
        # Internal State
        self.queues = np.zeros(12, dtype=np.float32)
        self.current_phase = 0
        self.phase_duration = 0
        self.step_count = 0
        
        self.MAX_QUEUE = 50 
        self.MAX_STEPS = config.max_steps

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        self.queues = np.zeros(12, dtype=np.float32)
        self.current_phase = 0
        self.phase_duration = 0
        self.step_count = 0
        
        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.step_count += 1
        
        # Apply Action
        prev_phase = self.current_phase
        self._apply_action(action)
        did_switch = (self.current_phase != prev_phase)
        
        # Simulate Traffic
        vehicles_passed = self._simulate_dynamics()
        
        # Calculate Reward
        reward, r_components = self._calculate_reward(vehicles_passed, did_switch)
        
        #  Check Termination
        truncated = self.step_count >= self.MAX_STEPS
        done = False
        
        #  Observation
        obs = self._get_obs()
        
        info = {
            "total_queue": np.sum(self.queues),
            "max_queue": np.max(self.queues) if len(self.queues) > 0 else 0,
            "throughput": vehicles_passed,
            "phase": self.current_phase,
            "reward_components": r_components
        }
        
        return obs, reward, done, truncated, info

    def render(self):
        if self.render_mode == 'human':
            print(f"Step: {self.step_count} | Phase: {self.current_phase}")
            print(f"Queues: {self.queues.astype(int)}")

    # Helpers
    def _get_obs(self) -> np.ndarray:
        return np.concatenate([
            self.queues, 
            [float(self.current_phase)], 
            [float(self.phase_duration)]
        ], dtype=np.float32)

    def _apply_action(self, action: int):
        if action == 0:  # KEEP
            self.phase_duration += 1
        else:
            prev_phase = self.current_phase
            if action == 1: # NEXT
                self.current_phase = (self.current_phase + 1) % 4
            elif action == 2: # SKIP TO NS
                self.current_phase = 0
            elif action == 3: # SKIP TO EW
                self.current_phase = 2
            
            if self.current_phase != prev_phase:
                self.phase_duration = 0
            else:
                self.phase_duration += 1

    def _simulate_dynamics(self) -> float:
        """
        Simulates vehicle arrivals and departures using exact logic from old code.
        """
        # Time-Varying Traffic 
        cycle_pos = self.step_count % 1000
        time_factor = 1.0
        
        if (200 <= cycle_pos < 400) or (600 <= cycle_pos < 800):
            time_factor = 1.5 # Peak
        elif (400 <= cycle_pos < 600):
            time_factor = 0.7 # Off-peak
            
        current_rates = self.base_arrival_rates * time_factor

        # Poisson Arrivals (Matches old TrafficGenerator)
        # Using Poisson allows >1 car per step if rate is high
        new_arrivals = self.np_random.poisson(current_rates).astype(np.float32)
        
        # Cap queues
        for i in range(12):
            if self.queues[i] < self.MAX_QUEUE:
                self.queues[i] += new_arrivals[i]

        # Capacity-Based Departures
        total_passed = 0.0
        allowed_indices = self._get_allowed_indices(self.current_phase)
        
        for idx in allowed_indices:
            if self.queues[idx] > 0:
                # Get capacity for this specific lane type (Straight/Left/Right)
                capacity = self.lane_capacities[idx]
                
                # Determine actual flow (can't be more than queue or capacity)
                flow = min(self.queues[idx], capacity)
                
                self.queues[idx] -= flow
                total_passed += flow
                
        return total_passed

    def _calculate_reward(self, vehicles_passed: float, did_switch: bool) -> Tuple[float, Dict]:
        total_queue = np.sum(self.queues)
        max_queue = np.max(self.queues) if len(self.queues) > 0 else 0
        
        # Original weights
        W_QUEUE = -0.1
        W_THROUGHPUT = 0.5
        W_SWITCH = -1.0
        W_FAIRNESS = -0.05
        
        r_queue = W_QUEUE * total_queue
        r_throughput = W_THROUGHPUT * vehicles_passed
        r_switch = W_SWITCH if did_switch else 0.0
        r_fairness = W_FAIRNESS * (max_queue ** 2)
        
        total_reward = r_queue + r_throughput + r_switch + r_fairness
        
        return total_reward, {
            "queue_penalty": r_queue,
            "throughput_reward": r_throughput,
            "switch_penalty": r_switch,
            "fairness_penalty": r_fairness
        }

    def _get_allowed_indices(self, phase: int) -> List[int]:
        """
        Returns indices of moving lanes.
        Indices:
        N: 0(S), 1(L), 2(R) | S: 3(S), 4(L), 5(R)
        E: 6(S), 7(L), 8(R) | W: 9(S), 10(L), 11(R)
        """
        if phase == 0: # NS Through
            # N->S(0), S->N(3)
            # N->W(2), S->E(5) (Right turns allowed)
            return [0, 3, 2, 5]
            
        elif phase == 1: # NS Left
            # N->E(1), S->W(4)
            # N->W(2), S->E(5) (Right turns ALSO allowed here in old code)
            return [1, 4, 2, 5]
            
        elif phase == 2: # EW Through
            # E->W(6), W->E(9)
            # E->N(8), W->S(11) (Right turns allowed)
            return [6, 9, 8, 11]
            
        elif phase == 3: # EW Left
            # E->S(7), W->N(10)
            # E->N(8), W->S(11) (Right turns ALSO allowed here)
            return [7, 10, 8, 11]
            
        return []