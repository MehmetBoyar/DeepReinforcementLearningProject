"""
Phase definitions for 4-phase traffic light system.

Phase 0: North-South Through Traffic
Phase 1: North-South Protected Left Turns
Phase 2: East-West Through Traffic
Phase 3: East-West Protected Left Turns
"""

from dataclasses import dataclass
from typing import Dict, List

@dataclass
class MovementStatus:
    """Status of a movement in a phase"""
    ALLOWED = "allowed"
    BLOCKED = "blocked"  # Conflict with another movement
    RED = "red"  # Red light


class PhaseManager:
    """
    Manages the 4-phase traffic light system with movement permissions.
    """
    
    # Movement types (from driver's perspective)
    MOVEMENT_TYPES = {
        'N_to_S': 'straight', 'N_to_E': 'left', 'N_to_W': 'right',
        'S_to_N': 'straight', 'S_to_W': 'left', 'S_to_E': 'right',
        'E_to_W': 'straight', 'E_to_S': 'left', 'E_to_N': 'right',
        'W_to_E': 'straight', 'W_to_N': 'left', 'W_to_S': 'right'
    }
    
    def __init__(self):
        """Initialize the 4-phase system"""
        self.num_phases = 4
        self.phase_names = {
            0: "N-S Through Traffic",
            1: "N-S Protected Left",
            2: "E-W Through Traffic",
            3: "E-W Protected Left"
        }
        
        # Define allowed movements for each phase
        self.phase_rules = self._define_phase_rules()
        
    def _define_phase_rules(self) -> Dict[int, Dict[str, str]]:
        rules = {}
        
        # Phase 0: North-South Through Traffic
        rules[0] = {
            'N_to_S': MovementStatus.ALLOWED,
            'S_to_N': MovementStatus.ALLOWED,
            'N_to_W': MovementStatus.ALLOWED,
            'S_to_E': MovementStatus.ALLOWED,
            'N_to_E': MovementStatus.BLOCKED,
            'S_to_W': MovementStatus.BLOCKED,
            'E_to_W': MovementStatus.RED, 'E_to_S': MovementStatus.RED, 'E_to_N': MovementStatus.RED,
            'W_to_E': MovementStatus.RED, 'W_to_N': MovementStatus.RED, 'W_to_S': MovementStatus.RED,
        }
        
        # Phase 1: North-South Protected Left Turns
        rules[1] = {
            'N_to_E': MovementStatus.ALLOWED,
            'S_to_W': MovementStatus.ALLOWED,
            'N_to_W': MovementStatus.ALLOWED,
            'S_to_E': MovementStatus.ALLOWED,
            'N_to_S': MovementStatus.RED, 'S_to_N': MovementStatus.RED,
            'E_to_W': MovementStatus.RED, 'E_to_S': MovementStatus.RED, 'E_to_N': MovementStatus.RED,
            'W_to_E': MovementStatus.RED, 'W_to_N': MovementStatus.RED, 'W_to_S': MovementStatus.RED,
        }
        
        # Phase 2: East-West Through Traffic
        rules[2] = {
            'E_to_W': MovementStatus.ALLOWED,
            'W_to_E': MovementStatus.ALLOWED,
            'E_to_N': MovementStatus.ALLOWED,
            'W_to_S': MovementStatus.ALLOWED,
            'E_to_S': MovementStatus.BLOCKED,
            'W_to_N': MovementStatus.BLOCKED,
            'N_to_S': MovementStatus.RED, 'N_to_E': MovementStatus.RED, 'N_to_W': MovementStatus.RED,
            'S_to_N': MovementStatus.RED, 'S_to_W': MovementStatus.RED, 'S_to_E': MovementStatus.RED,
        }
        
        # Phase 3: East-West Protected Left Turns
        rules[3] = {
            'E_to_S': MovementStatus.ALLOWED,
            'W_to_N': MovementStatus.ALLOWED,
            'E_to_N': MovementStatus.ALLOWED,
            'W_to_S': MovementStatus.ALLOWED,
            'E_to_W': MovementStatus.RED, 'W_to_E': MovementStatus.RED,
            'N_to_S': MovementStatus.RED, 'N_to_E': MovementStatus.RED, 'N_to_W': MovementStatus.RED,
            'S_to_N': MovementStatus.RED, 'S_to_W': MovementStatus.RED, 'S_to_E': MovementStatus.RED,
        }
        
        return rules
    
    def get_allowed_movements(self, phase: int) -> List[str]:
        return [m for m, status in self.phase_rules[phase].items() 
                if status == MovementStatus.ALLOWED]
    
    def get_blocked_movements(self, phase: int) -> List[str]:
        return [m for m, status in self.phase_rules[phase].items() 
                if status == MovementStatus.BLOCKED]
    
    def get_red_movements(self, phase: int) -> List[str]:
        return [m for m, status in self.phase_rules[phase].items() 
                if status == MovementStatus.RED]
    
    def is_movement_allowed(self, phase: int, movement: str) -> bool:
        return self.phase_rules[phase].get(movement) == MovementStatus.ALLOWED
    
    def get_movement_type(self, movement: str) -> str:
        return self.MOVEMENT_TYPES.get(movement, 'unknown')
    
    def get_phase_name(self, phase: int) -> str:
        return self.phase_names.get(phase, f"Phase {phase}")
    
    def get_next_phase(self, current_phase: int, action: int) -> int:
        if action == 0:  # KEEP
            return current_phase
        elif action == 1:  # NEXT
            return (current_phase + 1) % self.num_phases
        elif action == 2:  # SKIP_TO_NS
            return 0
        elif action == 3:  # SKIP_TO_EW
            return 2
        else:
            return current_phase

    def print_phase_info(self, phase: int):
        print(f"\n{'='*60}")
        print(f"Phase {phase}: {self.get_phase_name(phase)}")
        print(f"{'='*60}")
        print(f"Allowed movements: {self.get_allowed_movements(phase)}")