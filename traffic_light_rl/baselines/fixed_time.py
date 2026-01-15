"""
Fixed-Time Traffic Light Controller.

A simple baseline that switches phases at fixed intervals.
"""

import numpy as np
from typing import Optional


class FixedTimeController:
    """
    Fixed-time traffic light controller.
    
    Cycles through phases at fixed intervals, regardless of traffic conditions.
    This is the simplest baseline for comparison with RL agents.
    """
    
    def __init__(self, 
                 phase_durations: Optional[dict] = None,
                 default_duration: int = 30):
        """
        Initialize fixed-time controller.
        
        Args:
            phase_durations: Dictionary mapping phase -> duration in seconds
            default_duration: Default phase duration if not specified
        """
        self.default_duration = default_duration
        self.phase_durations = phase_durations or {
            0: 30,  # N-S through
            1: 15,  # N-S protected left
            2: 30,  # E-W through
            3: 15   # E-W protected left
        }
        
        self.current_phase = 0
        self.time_in_phase = 0
        self.total_steps = 0
    
    def reset(self):
        """Reset controller state"""
        self.current_phase = 0
        self.time_in_phase = 0
        self.total_steps = 0
    
    def get_action(self, state: np.ndarray) -> int:
        """
        Get action based on fixed timing.
        
        Args:
            state: Current environment state (ignored for fixed-time)
            
        Returns:
            Action: 0 (KEEP) or 1 (NEXT)
        """
        # Get current phase from state
        current_phase = int(state[12])
        phase_duration = int(state[13])
        
        # Check if it's time to switch
        target_duration = self.phase_durations.get(
            current_phase, self.default_duration
        )
        
        if phase_duration >= target_duration:
            return 1  # NEXT_PHASE
        else:
            return 0  # KEEP
    
    def set_phase_duration(self, phase: int, duration: int):
        """Set duration for a specific phase"""
        self.phase_durations[phase] = duration
    
    def get_name(self) -> str:
        """Get controller name"""
        return "Fixed-Time"
    
    def get_config(self) -> dict:
        """Get controller configuration"""
        return {
            'type': 'FixedTime',
            'phase_durations': self.phase_durations,
            'default_duration': self.default_duration
        }


if __name__ == "__main__":
    # Test fixed-time controller
    controller = FixedTimeController()
    
    print("Fixed-Time Controller Test")
    print(f"Phase durations: {controller.phase_durations}")
    
    # Simulate some states
    for phase in range(4):
        for duration in [0, 10, 20, 29, 30, 35]:
            state = np.zeros(14)
            state[12] = phase
            state[13] = duration
            
            action = controller.get_action(state)
            action_name = "KEEP" if action == 0 else "NEXT"
            print(f"Phase {phase}, Duration {duration}: {action_name}")
