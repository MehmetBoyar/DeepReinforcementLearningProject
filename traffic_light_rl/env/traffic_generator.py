"""
Traffic generator for simulating vehicle arrivals.
Uses Poisson distribution for realistic traffic patterns.
"""

import numpy as np
from typing import Dict, Optional


class TrafficGenerator:
    """
    Generates vehicle arrivals using Poisson distribution.
    
    Supports different traffic patterns:
    - uniform: Same arrival rate for all movements
    - directional: Different rates for different directions
    - time_varying: Rates that change over time (rush hour patterns)
    """
    
    # Default arrival rates (vehicles per second)
    DEFAULT_RATES = {
        'N_to_S': 0.15, 'N_to_E': 0.05, 'N_to_W': 0.05,
        'S_to_N': 0.15, 'S_to_W': 0.05, 'S_to_E': 0.05,
        'E_to_W': 0.15, 'E_to_S': 0.05, 'E_to_N': 0.05,
        'W_to_E': 0.15, 'W_to_N': 0.05, 'W_to_S': 0.05,
    }
    
    def __init__(self, 
                 arrival_rates: Optional[Dict[str, float]] = None,
                 pattern: str = 'uniform',
                 seed: Optional[int] = None):
        """
        Initialize the traffic generator.
        
        Args:
            arrival_rates: Custom arrival rates per movement
            pattern: Traffic pattern ('uniform', 'directional', 'time_varying')
            seed: Random seed for reproducibility
        """
        self.arrival_rates = arrival_rates or self.DEFAULT_RATES.copy()
        self.pattern = pattern
        self.base_rates = self.arrival_rates.copy()
        
        if seed is not None:
            np.random.seed(seed)
    
    def generate_arrivals(self, dt: float = 1.0, time_step: int = 0) -> Dict[str, int]:
        """
        Generate vehicle arrivals for all movements.
        
        Args:
            dt: Time step duration in seconds
            time_step: Current simulation time step (for time-varying patterns)
            
        Returns:
            Dictionary mapping movement -> number of new arrivals
        """
        arrivals = {}
        
        # Get current rates (may vary with time)
        current_rates = self._get_current_rates(time_step)
        
        for movement, rate in current_rates.items():
            # Poisson distribution for arrivals
            lambda_param = rate * dt
            arrivals[movement] = np.random.poisson(lambda_param)
        
        return arrivals
    
    def _get_current_rates(self, time_step: int) -> Dict[str, float]:
        """
        Get arrival rates for current time step.
        
        For time-varying patterns, modulates base rates.
        """
        if self.pattern == 'time_varying':
            # Simulate rush hour pattern
            # Peak at time_step 200-400 and 600-800
            multiplier = 1.0
            cycle_position = time_step % 1000
            
            if 200 <= cycle_position < 400 or 600 <= cycle_position < 800:
                multiplier = 1.5  # Rush hour
            elif 400 <= cycle_position < 600:
                multiplier = 0.7  # Off-peak
            
            return {m: r * multiplier for m, r in self.base_rates.items()}
        
        return self.arrival_rates
    
    def set_arrival_rate(self, movement: str, rate: float):
        """Set arrival rate for a specific movement"""
        if movement in self.arrival_rates:
            self.arrival_rates[movement] = rate
            self.base_rates[movement] = rate
    
    def set_all_rates(self, rates: Dict[str, float]):
        """Set all arrival rates"""
        self.arrival_rates = rates.copy()
        self.base_rates = rates.copy()
    
    def scale_rates(self, factor: float):
        """Scale all arrival rates by a factor"""
        for movement in self.arrival_rates:
            self.arrival_rates[movement] *= factor
            self.base_rates[movement] *= factor
    
    def get_expected_arrivals(self, dt: float = 1.0) -> Dict[str, float]:
        """Get expected arrivals per movement (mean of Poisson)"""
        return {m: r * dt for m, r in self.arrival_rates.items()}
    
    def get_total_expected_arrivals(self, dt: float = 1.0) -> float:
        """Get total expected arrivals across all movements"""
        return sum(self.get_expected_arrivals(dt).values())


class CapacityManager:
    """
    Manages vehicle throughput capacities for different movement types.
    """
    
    DEFAULT_CAPACITIES = {
        'straight': 2.0,  # vehicles per second
        'left': 1.5,
        'right': 1.0
    }
    
    def __init__(self, capacities: Optional[Dict[str, float]] = None):
        """Initialize capacity manager"""
        self.capacities = capacities or self.DEFAULT_CAPACITIES.copy()
    
    def get_capacity(self, movement_type: str) -> float:
        """Get capacity for a movement type"""
        return self.capacities.get(movement_type, 1.0)
    
    def process_departures(self, queue: float, movement_type: str, 
                          is_allowed: bool, dt: float = 1.0) -> float:
        """
        Calculate number of vehicles that can depart.
        
        Args:
            queue: Current queue length
            movement_type: Type of movement (straight, left, right)
            is_allowed: Whether movement is allowed in current phase
            dt: Time step duration
            
        Returns:
            Number of vehicles that depart
        """
        if not is_allowed:
            return 0.0
        
        capacity = self.get_capacity(movement_type) * dt
        return min(queue, capacity)


if __name__ == "__main__":
    # Test traffic generator
    gen = TrafficGenerator(seed=42)
    
    print("Expected arrivals per second:")
    for m, rate in gen.get_expected_arrivals().items():
        print(f"  {m}: {rate:.3f}")
    
    print(f"\nTotal expected: {gen.get_total_expected_arrivals():.3f} vehicles/second")
    
    print("\nSample arrivals:")
    for i in range(5):
        arrivals = gen.generate_arrivals()
        total = sum(arrivals.values())
        print(f"  Step {i}: {total} total arrivals")
