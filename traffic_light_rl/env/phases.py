"""
Phase definitions for 4-phase traffic light system.

Phase 0: North-South Through Traffic
Phase 1: North-South Protected Left Turns
Phase 2: East-West Through Traffic
Phase 3: East-West Protected Left Turns
"""

from dataclasses import dataclass
from typing import Dict, List, Set


@dataclass
class MovementStatus:
    """Status of a movement in a phase"""
    ALLOWED = "allowed"
    BLOCKED = "blocked"  # Conflict with another movement
    RED = "red"  # Red light


class PhaseManager:
    """
    Manages the 4-phase traffic light system with movement permissions.
    
    Movements are defined as origin_to_destination (from driver's perspective):
    - N_to_S: North to South (straight)
    - N_to_E: North to East (LEFT turn - crosses oncoming traffic)
    - N_to_W: North to West (RIGHT turn - no conflict)
    - S_to_N: South to North (straight)
    - S_to_W: South to West (LEFT turn - crosses oncoming traffic)
    - S_to_E: South to East (RIGHT turn - no conflict)
    - E_to_W: East to West (straight)
    - E_to_S: East to South (LEFT turn - crosses oncoming traffic)
    - E_to_N: East to North (RIGHT turn - no conflict)
    - W_to_E: West to East (straight)
    - W_to_N: West to North (LEFT turn - crosses oncoming traffic)
    - W_to_S: West to South (RIGHT turn - no conflict)
    """
    
    # All possible movements
    ALL_MOVEMENTS = [
        'N_to_S', 'N_to_E', 'N_to_W',
        'S_to_N', 'S_to_W', 'S_to_E',
        'E_to_W', 'E_to_S', 'E_to_N',
        'W_to_E', 'W_to_N', 'W_to_S'
    ]
    
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
        """
        Define the movement rules for each phase.
        
        Returns:
            Dictionary mapping phase -> movement -> status
        """
        rules = {}
        
        # Phase 0: North-South Through Traffic
        # N-S straight and RIGHT turns allowed (right turns don't cross oncoming)
        # N-S LEFT turns blocked (cross oncoming straight traffic)
        # All E-W movements are RED
        rules[0] = {
            'N_to_S': MovementStatus.ALLOWED,  # Straight - allowed
            'S_to_N': MovementStatus.ALLOWED,  # Straight - allowed
            'N_to_W': MovementStatus.ALLOWED,  # Right turn - no conflict
            'S_to_E': MovementStatus.ALLOWED,  # Right turn - no conflict
            'N_to_E': MovementStatus.BLOCKED,  # Left turn - conflicts with S_to_N
            'S_to_W': MovementStatus.BLOCKED,  # Left turn - conflicts with N_to_S
            'E_to_W': MovementStatus.RED,
            'E_to_S': MovementStatus.RED,
            'E_to_N': MovementStatus.RED,
            'W_to_E': MovementStatus.RED,
            'W_to_N': MovementStatus.RED,
            'W_to_S': MovementStatus.RED,
        }
        
        # Phase 1: North-South Protected Left Turns
        # All N-S turns allowed (both left and right), no through traffic
        # This phase allows left turns to proceed safely without oncoming straight traffic
        rules[1] = {
            'N_to_E': MovementStatus.ALLOWED,  # Left turn - protected (no oncoming straight)
            'S_to_W': MovementStatus.ALLOWED,  # Left turn - protected (no oncoming straight)
            'N_to_W': MovementStatus.ALLOWED,  # Right turn - always safe
            'S_to_E': MovementStatus.ALLOWED,  # Right turn - always safe
            'N_to_S': MovementStatus.RED,      # Straight - red
            'S_to_N': MovementStatus.RED,      # Straight - red
            'E_to_W': MovementStatus.RED,
            'E_to_S': MovementStatus.RED,
            'E_to_N': MovementStatus.RED,
            'W_to_E': MovementStatus.RED,
            'W_to_N': MovementStatus.RED,
            'W_to_S': MovementStatus.RED,
        }
        
        # Phase 2: East-West Through Traffic
        # E-W straight and RIGHT turns allowed (right turns don't cross oncoming)
        # E-W LEFT turns blocked (cross oncoming straight traffic)
        # All N-S movements are RED
        rules[2] = {
            'E_to_W': MovementStatus.ALLOWED,  # Straight - allowed
            'W_to_E': MovementStatus.ALLOWED,  # Straight - allowed
            'E_to_N': MovementStatus.ALLOWED,  # Right turn - no conflict
            'W_to_S': MovementStatus.ALLOWED,  # Right turn - no conflict
            'E_to_S': MovementStatus.BLOCKED,  # Left turn - conflicts with W_to_E
            'W_to_N': MovementStatus.BLOCKED,  # Left turn - conflicts with E_to_W
            'N_to_S': MovementStatus.RED,
            'N_to_E': MovementStatus.RED,
            'N_to_W': MovementStatus.RED,
            'S_to_N': MovementStatus.RED,
            'S_to_W': MovementStatus.RED,
            'S_to_E': MovementStatus.RED,
        }
        
        # Phase 3: East-West Protected Left Turns
        # All E-W turns allowed (both left and right), no through traffic
        # This phase allows left turns to proceed safely without oncoming straight traffic
        rules[3] = {
            'E_to_S': MovementStatus.ALLOWED,  # Left turn - protected (no oncoming straight)
            'W_to_N': MovementStatus.ALLOWED,  # Left turn - protected (no oncoming straight)
            'E_to_N': MovementStatus.ALLOWED,  # Right turn - always safe
            'W_to_S': MovementStatus.ALLOWED,  # Right turn - always safe
            'E_to_W': MovementStatus.RED,      # Straight - red
            'W_to_E': MovementStatus.RED,      # Straight - red
            'N_to_S': MovementStatus.RED,
            'N_to_E': MovementStatus.RED,
            'N_to_W': MovementStatus.RED,
            'S_to_N': MovementStatus.RED,
            'S_to_W': MovementStatus.RED,
            'S_to_E': MovementStatus.RED,
        }
        
        return rules
    
    def get_allowed_movements(self, phase: int) -> List[str]:
        """Get list of allowed movements for a phase"""
        return [m for m, status in self.phase_rules[phase].items() 
                if status == MovementStatus.ALLOWED]
    
    def get_blocked_movements(self, phase: int) -> List[str]:
        """Get list of blocked movements (conflicts) for a phase"""
        return [m for m, status in self.phase_rules[phase].items() 
                if status == MovementStatus.BLOCKED]
    
    def get_red_movements(self, phase: int) -> List[str]:
        """Get list of red light movements for a phase"""
        return [m for m, status in self.phase_rules[phase].items() 
                if status == MovementStatus.RED]
    
    def is_movement_allowed(self, phase: int, movement: str) -> bool:
        """Check if a movement is allowed in the given phase"""
        return self.phase_rules[phase].get(movement) == MovementStatus.ALLOWED
    
    def get_movement_type(self, movement: str) -> str:
        """Get the type of movement (straight, left, right)"""
        return self.MOVEMENT_TYPES.get(movement, 'unknown')
    
    def get_phase_name(self, phase: int) -> str:
        """Get the name of a phase"""
        return self.phase_names.get(phase, f"Phase {phase}")
    
    def get_next_phase(self, current_phase: int, action: int) -> int:
        """
        Get the next phase based on action.
        
        Actions:
            0: KEEP - stay in current phase
            1: NEXT - move to next phase (cyclic)
            2: SKIP_TO_NS - jump to phase 0
            3: SKIP_TO_EW - jump to phase 2
        """
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
        """Print detailed information about a phase"""
        print(f"\n{'='*60}")
        print(f"Phase {phase}: {self.get_phase_name(phase)}")
        print(f"{'='*60}")
        print(f"Allowed movements: {self.get_allowed_movements(phase)}")
        print(f"Blocked movements: {self.get_blocked_movements(phase)}")
        print(f"Red light movements: {self.get_red_movements(phase)}")


if __name__ == "__main__":
    # Test the phase manager
    pm = PhaseManager()
    for phase in range(4):
        pm.print_phase_info(phase)
