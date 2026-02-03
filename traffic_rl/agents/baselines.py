import numpy as np
from traffic_rl.agents.base import BaseAgent

class FixedTimeAgent(BaseAgent):
    def __init__(self, durations=None):
        # Default durations matches original: 30s Green, 15s Left
        self.durations = durations or {0: 30, 1: 15, 2: 30, 3: 15}

    def act(self, state, training=False):
        # state[12] = phase, state[13] = duration
        current_phase = int(state[12])
        current_duration = int(state[13])
        
        target = self.durations.get(current_phase, 30)
        
        if current_duration >= target:
            return 1 # NEXT
        return 0 # KEEP

    def update(self, *args, **kwargs): pass
    def save(self, path): pass
    def load(self, path): pass


class AdaptiveAgent(BaseAgent):
    """
    Full port of the original AdaptiveController logic.
    """
    def __init__(self, min_duration=10, max_duration=60, 
                 queue_threshold=5.0, switch_threshold=0.3):
        self.min_dur = min_duration
        self.max_dur = max_duration
        self.q_thresh = queue_threshold
        self.switch_thresh = switch_threshold
        
        # Movements allowed in each phase (Indices match traffic_env.py)
        self.PHASE_MOVEMENTS = {
            0: [0, 3, 2, 5],   # N-S Through (+Right)
            1: [1, 4, 2, 5],   # N-S Left   (+Right)
            2: [6, 9, 8, 11],  # E-W Through (+Right)
            3: [7, 10, 8, 11]  # E-W Left   (+Right)
        }

    def act(self, state, training=False):
        current_phase = int(state[12])
        duration = int(state[13])
        queues = state[:12] # Array of 12 queues
        
        # Respect Min Duration
        if duration < self.min_dur:
            return 0 # KEEP
            
        # Respect Max Duration
        if duration >= self.max_dur:
            return 1 # NEXT
            
        # Calculate Queue Metrics
        curr_q_sum = self._get_phase_queue(queues, current_phase)
        
        next_phase = (current_phase + 1) % 4
        next_q_sum = self._get_phase_queue(queues, next_phase)
        
        total_q = curr_q_sum + next_q_sum
        
        # Standard Switch Logic
        # If current lane is empty-ish AND waiting lane has significant traffic relative to total
        if total_q > 0:
            waiting_ratio = next_q_sum / total_q
            if curr_q_sum < self.q_thresh and waiting_ratio > self.switch_thresh:
                return 1 # NEXT
        
        # Protected Left Turn Optimization
        # If in Left Turn Phase (1 or 3) and empty, switch immediately
        if current_phase in [1, 3]:
            if curr_q_sum < 2:
                return 1
        
        # Skip Logic (Smart jump)
        # If in Left Turn phase, but the cross-traffic Through lane is huge, skip to it
        ns_queue = sum(queues[0:6])
        ew_queue = sum(queues[6:12])
        
        if current_phase == 3: # EW Left
            if ns_queue > 2 * ew_queue:
                return 2 # SKIP TO NS
        elif current_phase == 1: # NS Left
            if ew_queue > 2 * ns_queue:
                return 3 # SKIP TO EW
                
        return 0 # KEEP

    def _get_phase_queue(self, queues, phase):
        indices = self.PHASE_MOVEMENTS.get(phase, [])
        return sum(queues[i] for i in indices)

    def update(self, *args, **kwargs): pass
    def save(self, path): pass
    def load(self, path): pass