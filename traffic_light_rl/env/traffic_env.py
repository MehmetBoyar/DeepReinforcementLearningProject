"""
Traffic Light Control Environment for Reinforcement Learning.

A Gym-compatible environment for training RL agents to control
traffic signals at a 4-way intersection.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any

# Support both gymnasium (new) and gym (old)
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    try:
        import gym
        from gym import spaces
    except ImportError:
        raise ImportError(
            "Neither 'gymnasium' nor 'gym' is installed. "
            "Please install with: pip install gymnasium"
        )

from .phases import PhaseManager
from .traffic_generator import TrafficGenerator, CapacityManager


class TrafficLightEnv(gym.Env):
    """
    Traffic Light Control Environment with 4-phase system.
    
    State Space (14 dimensions):
        - 12 queue lengths (vehicles waiting for each origin-destination pair)
        - 1 current phase (0-3)
        - 1 phase duration (time steps in current phase)
    
    Action Space (4 actions):
        - 0: KEEP - stay in current phase
        - 1: NEXT - move to next phase (cyclic)
        - 2: SKIP_TO_NS - jump to phase 0 (N-S through)
        - 3: SKIP_TO_EW - jump to phase 2 (E-W through)
    
    Reward:
        - Negative penalty for total queue length
        - Positive reward for vehicles passing through
        - Penalty for phase switches (to avoid excessive switching)
        - Fairness penalty for very long queues
    """
    
    metadata = {'render.modes': ['human', 'ansi']}
    
    # Movement indices in state vector
    MOVEMENT_INDICES = {
        'N_to_S': 0, 'N_to_E': 1, 'N_to_W': 2,
        'S_to_N': 3, 'S_to_W': 4, 'S_to_E': 5,
        'E_to_W': 6, 'E_to_S': 7, 'E_to_N': 8,
        'W_to_E': 9, 'W_to_N': 10, 'W_to_S': 11
    }
    
    # Action names
    ACTION_NAMES = {
        0: "KEEP",
        1: "NEXT_PHASE",
        2: "SKIP_TO_NS",
        3: "SKIP_TO_EW"
    }
    
    def __init__(self,
                 max_queue: int = 50,
                 max_duration: int = 120,
                 max_steps: int = 1000,
                 arrival_rates: Optional[Dict[str, float]] = None,
                 reward_weights: Optional[Dict[str, float]] = None,
                 seed: Optional[int] = None):
        """
        Initialize the traffic light environment.
        
        Args:
            max_queue: Maximum queue length per movement
            max_duration: Maximum phase duration
            max_steps: Maximum simulation steps per episode
            arrival_rates: Custom vehicle arrival rates
            reward_weights: Custom reward component weights
            seed: Random seed
        """
        super().__init__()
        
        self.max_queue = max_queue
        self.max_duration = max_duration
        self.max_steps = max_steps
        
        # Initialize components
        self.phase_manager = PhaseManager()
        self.traffic_generator = TrafficGenerator(arrival_rates, seed=seed)
        self.capacity_manager = CapacityManager()
        
        # Reward weights
        self.reward_weights = reward_weights or {
            'queue_penalty': -0.1,
            'throughput_reward': 0.5,
            'switch_penalty': -1.0,
            'fairness_penalty': -0.05
        }
        
        # Define observation space: 14 dimensions
        # 12 queues (0-max_queue) + phase (0-3) + duration (0-max_duration)
        low = np.zeros(14, dtype=np.float32)
        high = np.array([max_queue] * 12 + [3, max_duration], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        # Define action space: 4 actions
        self.action_space = spaces.Discrete(4)
        
        # Initialize state
        self.reset()
    
    def reset(self) -> np.ndarray:
        """
        Reset the environment to initial state.
        
        Returns:
            Initial observation
        """
        # Initialize all queues to 0
        self.queues = {m: 0.0 for m in self.MOVEMENT_INDICES}
        
        # Start with phase 0 (N-S through traffic)
        self.current_phase = 0
        self.phase_duration = 0
        
        # Reset step counter
        self.time_step = 0
        
        # Statistics tracking
        self.total_vehicles_passed = 0
        self.total_wait_time = 0
        self.phase_switches = 0
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """
        Get the current state as a numpy array.
        
        Returns:
            14-dimensional state vector
        """
        state = np.zeros(14, dtype=np.float32)
        
        # Fill queue values (indices 0-11)
        for movement, idx in self.MOVEMENT_INDICES.items():
            state[idx] = min(self.queues[movement], self.max_queue)
        
        # Fill phase information (indices 12-13)
        state[12] = float(self.current_phase)
        state[13] = min(float(self.phase_duration), self.max_duration)
        
        return state
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one time step in the environment.
        
        Args:
            action: Action to take (0-3)
            
        Returns:
            observation: New state
            reward: Reward for this step
            done: Whether episode is finished
            info: Additional information
        """
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        # 1. Generate new vehicle arrivals
        arrivals = self.traffic_generator.generate_arrivals(dt=1.0, time_step=self.time_step)
        for movement, count in arrivals.items():
            self.queues[movement] += count
        
        # 2. Process departures based on current phase
        departures = self._process_departures()
        
        # 3. Calculate reward and get components
        reward, reward_components = self._calculate_reward(action, departures)
        
        # 4. Update phase based on action
        old_phase = self.current_phase
        self.current_phase = self.phase_manager.get_next_phase(self.current_phase, action)
        
        if self.current_phase != old_phase:
            self.phase_duration = 0
            self.phase_switches += 1
        else:
            self.phase_duration += 1
        
        # 5. Update time step
        self.time_step += 1
        
        # 6. Update statistics
        total_passed = sum(departures.values())
        self.total_vehicles_passed += total_passed
        self.total_wait_time += sum(self.queues.values())
        
        # 7. Check if done
        done = self.time_step >= self.max_steps
        
        # 8. Calculate directional queues (N-S vs E-W)
        ns_queue = sum(self.queues[m] for m in ['N_to_S', 'N_to_E', 'N_to_W', 
                                                  'S_to_N', 'S_to_W', 'S_to_E'])
        ew_queue = sum(self.queues[m] for m in ['E_to_W', 'E_to_S', 'E_to_N',
                                                  'W_to_E', 'W_to_N', 'W_to_S'])
        
        # 9. Prepare info
        info = {
            'time_step': self.time_step,
            'phase': self.current_phase,
            'phase_name': self.phase_manager.get_phase_name(self.current_phase),
            'phase_duration': self.phase_duration,
            'total_queue': sum(self.queues.values()),
            'ns_queue': ns_queue,
            'ew_queue': ew_queue,
            'vehicles_passed': total_passed,
            'total_vehicles_passed': self.total_vehicles_passed,
            'total_wait_time': self.total_wait_time,
            'phase_switches': self.phase_switches,
            'arrivals': sum(arrivals.values()),
            'action_name': self.ACTION_NAMES[action],
            'reward_components': reward_components
        }
        
        return self._get_state(), reward, done, info
    
    def _process_departures(self) -> Dict[str, float]:
        """
        Process vehicle departures based on current phase.
        
        Returns:
            Dictionary of departures per movement
        """
        departures = {}
        
        for movement in self.MOVEMENT_INDICES:
            # Check if movement is allowed in current phase
            is_allowed = self.phase_manager.is_movement_allowed(
                self.current_phase, movement
            )
            
            # Get movement type for capacity
            movement_type = self.phase_manager.get_movement_type(movement)
            
            # Calculate departures
            departed = self.capacity_manager.process_departures(
                self.queues[movement],
                movement_type,
                is_allowed
            )
            
            departures[movement] = departed
            
            # Update queue
            self.queues[movement] = max(0, self.queues[movement] - departed)
        
        return departures
    
    def _calculate_reward(self, action: int, departures: Dict[str, float]) -> Tuple[float, Dict[str, float]]:
        """
        Calculate reward for current step.
        
        Components:
            1. Queue penalty: Penalize total waiting vehicles
            2. Throughput reward: Reward vehicles passing through
            3. Switch penalty: Penalize phase changes
            4. Fairness penalty: Penalize very long queues
            
        Args:
            action: Action taken
            departures: Vehicles that departed
            
        Returns:
            Tuple of (total reward, reward components dict)
        """
        # Component 1: Queue penalty
        total_queue = sum(self.queues.values())
        queue_penalty = self.reward_weights['queue_penalty'] * total_queue
        
        # Component 2: Throughput reward
        vehicles_passed = sum(departures.values())
        throughput_reward = self.reward_weights['throughput_reward'] * vehicles_passed
        
        # Component 3: Switch penalty
        switch_penalty = self.reward_weights['switch_penalty'] if action != 0 else 0.0
        
        # Component 4: Fairness penalty (quadratic on max queue)
        max_queue = max(self.queues.values()) if self.queues else 0
        fairness_penalty = self.reward_weights['fairness_penalty'] * (max_queue ** 2)
        
        # Total reward
        reward = queue_penalty + throughput_reward + switch_penalty + fairness_penalty
        
        # Return components for visualization
        components = {
            'queue_penalty': queue_penalty,
            'throughput_reward': throughput_reward,
            'switch_penalty': switch_penalty,
            'fairness_penalty': fairness_penalty
        }
        
        return reward, components
    
    def render(self, mode: str = 'human') -> Optional[str]:
        """
        Render the environment state.
        
        Args:
            mode: Render mode ('human' or 'ansi')
            
        Returns:
            String representation if mode is 'ansi'
        """
        phase_name = self.phase_manager.get_phase_name(self.current_phase)
        allowed = self.phase_manager.get_allowed_movements(self.current_phase)
        
        lines = []
        lines.append(f"\n{'='*70}")
        lines.append(f"Time: {self.time_step} | Phase {self.current_phase}: {phase_name} | Duration: {self.phase_duration}")
        lines.append(f"{'='*70}")
        lines.append(f"Allowed movements: {allowed}")
        lines.append(f"\nQueues:")
        lines.append(f"  From NORTH: →S={self.queues['N_to_S']:.0f}, →E={self.queues['N_to_E']:.0f}, →W={self.queues['N_to_W']:.0f}")
        lines.append(f"  From SOUTH: →N={self.queues['S_to_N']:.0f}, →W={self.queues['S_to_W']:.0f}, →E={self.queues['S_to_E']:.0f}")
        lines.append(f"  From EAST:  →W={self.queues['E_to_W']:.0f}, →S={self.queues['E_to_S']:.0f}, →N={self.queues['E_to_N']:.0f}")
        lines.append(f"  From WEST:  →E={self.queues['W_to_E']:.0f}, →N={self.queues['W_to_N']:.0f}, →S={self.queues['W_to_S']:.0f}")
        lines.append(f"\nTotal queue: {sum(self.queues.values()):.0f}")
        lines.append(f"Total vehicles passed: {self.total_vehicles_passed}")
        
        output = '\n'.join(lines)
        
        if mode == 'human':
            print(output)
        
        return output if mode == 'ansi' else None
    
    def get_queue_by_direction(self, direction: str) -> float:
        """Get total queue for vehicles coming from a direction"""
        direction_queues = {
            'north': ['N_to_S', 'N_to_E', 'N_to_W'],
            'south': ['S_to_N', 'S_to_W', 'S_to_E'],
            'east': ['E_to_W', 'E_to_S', 'E_to_N'],
            'west': ['W_to_E', 'W_to_N', 'W_to_S']
        }
        movements = direction_queues.get(direction.lower(), [])
        return sum(self.queues.get(m, 0) for m in movements)
    
    def close(self):
        """Clean up resources"""
        pass


# Register the environment (optional)
# try:
#     gym.register(
#         id='TrafficLight-v0',
#         entry_point='traffic_light_rl.env:TrafficLightEnv',
#     )
# except:
#     pass  # Already registered


if __name__ == "__main__":
    # Test the environment
    env = TrafficLightEnv(seed=42)
    
    print("Testing TrafficLightEnv...")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Run a few steps with random actions
    obs = env.reset()
    print(f"\nInitial state shape: {obs.shape}")
    
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if i < 3 or i >= 8:
            env.render()
            print(f"Action: {env.ACTION_NAMES[action]}, Reward: {reward:.2f}")
    
    print(f"\n{'='*70}")
    print(f"Total reward after 10 steps: {total_reward:.2f}")
    print(f"Total vehicles passed: {info['total_vehicles_passed']}")
    print(f"Phase switches: {info['phase_switches']}")
