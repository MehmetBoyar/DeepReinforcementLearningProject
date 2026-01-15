"""
Adaptive Traffic Light Controller.

A baseline that adapts phase timing based on queue lengths.
"""

import numpy as np
from typing import Optional, Dict


class AdaptiveController:
    """
    Simple adaptive traffic light controller.
    
    Makes decisions based on queue lengths:
    - Stays in current phase if queues in allowed directions are high
    - Switches when allowed direction queues are low and waiting directions are high
    - Respects minimum and maximum phase durations
    """
    
    # Movement indices in state vector
    MOVEMENT_INDICES = {
        'N_to_S': 0, 'N_to_E': 1, 'N_to_W': 2,
        'S_to_N': 3, 'S_to_W': 4, 'S_to_E': 5,
        'E_to_W': 6, 'E_to_S': 7, 'E_to_N': 8,
        'W_to_E': 9, 'W_to_N': 10, 'W_to_S': 11
    }
    
    # Movements allowed in each phase
    PHASE_MOVEMENTS = {
        0: ['N_to_S', 'S_to_N', 'N_to_W', 'S_to_E'],  # N-S through
        1: ['N_to_W', 'S_to_E', 'N_to_E', 'S_to_W'],  # N-S protected left
        2: ['E_to_W', 'W_to_E', 'E_to_N', 'W_to_S'],  # E-W through
        3: ['E_to_N', 'W_to_S', 'E_to_S', 'W_to_N'],  # E-W protected left
    }
    
    def __init__(self,
                 min_phase_duration: int = 10,
                 max_phase_duration: int = 60,
                 queue_threshold: float = 5.0,
                 switch_threshold: float = 0.3):
        """
        Initialize adaptive controller.
        
        Args:
            min_phase_duration: Minimum time to stay in a phase
            max_phase_duration: Maximum time before forced switch
            queue_threshold: Queue length threshold for switching decisions
            switch_threshold: Ratio threshold for switching
        """
        self.min_phase_duration = min_phase_duration
        self.max_phase_duration = max_phase_duration
        self.queue_threshold = queue_threshold
        self.switch_threshold = switch_threshold
    
    def reset(self):
        """Reset controller state"""
        pass  # Stateless controller
    
    def get_action(self, state: np.ndarray) -> int:
        """
        Get action based on current queue state.
        
        Decision logic:
        1. If below minimum duration: KEEP
        2. If above maximum duration: NEXT
        3. If current phase queues are empty and waiting queues are high: NEXT
        4. Otherwise: KEEP
        
        Args:
            state: Current environment state
            
        Returns:
            Action: 0 (KEEP), 1 (NEXT), 2 (SKIP_NS), or 3 (SKIP_EW)
        """
        current_phase = int(state[12])
        phase_duration = int(state[13])
        
        # Extract queue values
        queues = {}
        for movement, idx in self.MOVEMENT_INDICES.items():
            queues[movement] = state[idx]
        
        # Rule 1: Respect minimum phase duration
        if phase_duration < self.min_phase_duration:
            return 0  # KEEP
        
        # Rule 2: Respect maximum phase duration
        if phase_duration >= self.max_phase_duration:
            return 1  # NEXT
        
        # Calculate queue metrics
        current_queue = self._get_phase_queue(queues, current_phase)
        next_phase = (current_phase + 1) % 4
        next_queue = self._get_phase_queue(queues, next_phase)
        
        # Calculate queues for main phases (through traffic)
        ns_queue = self._get_ns_queue(queues)
        ew_queue = self._get_ew_queue(queues)
        
        # Rule 3: Switch if current queue is low and waiting queue is high
        total_queue = current_queue + next_queue
        if total_queue > 0:
            waiting_ratio = next_queue / total_queue
            if current_queue < self.queue_threshold and waiting_ratio > self.switch_threshold:
                return 1  # NEXT
        
        # Rule 4: For protected left phases, switch quickly if queues are empty
        if current_phase in [1, 3]:  # Protected left phases
            if current_queue < 2:
                return 1  # NEXT
        
        # Rule 5: Consider skipping to main phase if one direction is much busier
        if current_phase in [1, 3]:  # If in protected left
            if ns_queue > 2 * ew_queue and current_phase == 3:
                return 2  # SKIP_TO_NS
            elif ew_queue > 2 * ns_queue and current_phase == 1:
                return 3  # SKIP_TO_EW
        
        return 0  # KEEP
    
    def _get_phase_queue(self, queues: Dict[str, float], phase: int) -> float:
        """Get total queue for movements allowed in a phase"""
        movements = self.PHASE_MOVEMENTS.get(phase, [])
        return sum(queues.get(m, 0) for m in movements)
    
    def _get_ns_queue(self, queues: Dict[str, float]) -> float:
        """Get total queue for N-S direction"""
        ns_movements = ['N_to_S', 'N_to_E', 'N_to_W', 'S_to_N', 'S_to_W', 'S_to_E']
        return sum(queues.get(m, 0) for m in ns_movements)
    
    def _get_ew_queue(self, queues: Dict[str, float]) -> float:
        """Get total queue for E-W direction"""
        ew_movements = ['E_to_W', 'E_to_S', 'E_to_N', 'W_to_E', 'W_to_N', 'W_to_S']
        return sum(queues.get(m, 0) for m in ew_movements)
    
    def get_name(self) -> str:
        """Get controller name"""
        return "Adaptive"
    
    def get_config(self) -> dict:
        """Get controller configuration"""
        return {
            'type': 'Adaptive',
            'min_phase_duration': self.min_phase_duration,
            'max_phase_duration': self.max_phase_duration,
            'queue_threshold': self.queue_threshold,
            'switch_threshold': self.switch_threshold
        }


if __name__ == "__main__":
    # Test adaptive controller
    controller = AdaptiveController()
    
    print("Adaptive Controller Test")
    print(f"Config: {controller.get_config()}")
    
    # Test with various states
    test_cases = [
        # [N→S, N→E, N→W, S→N, S→W, S→E, E→W, E→S, E→N, W→E, W→N, W→S, phase, duration]
        np.array([10, 2, 2, 10, 2, 2, 5, 1, 1, 5, 1, 1, 0, 5], dtype=np.float32),   # Phase 0, low duration
        np.array([1, 0, 0, 1, 0, 0, 15, 3, 3, 15, 3, 3, 0, 25], dtype=np.float32),  # Phase 0, low NS queue
        np.array([10, 2, 2, 10, 2, 2, 5, 1, 1, 5, 1, 1, 0, 55], dtype=np.float32),  # Phase 0, high duration
        np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 15], dtype=np.float32),    # Phase 1, empty queues
    ]
    
    action_names = {0: "KEEP", 1: "NEXT", 2: "SKIP_NS", 3: "SKIP_EW"}
    
    for i, state in enumerate(test_cases):
        action = controller.get_action(state)
        phase = int(state[12])
        duration = int(state[13])
        print(f"\nTest {i+1}: Phase {phase}, Duration {duration}")
        print(f"  Action: {action_names[action]}")
