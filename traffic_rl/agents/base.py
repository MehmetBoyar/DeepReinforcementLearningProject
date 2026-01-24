from abc import ABC, abstractmethod

class BaseAgent(ABC):
    @abstractmethod
    def act(self, state, training=True):
        """Returns action index"""
        pass

    @abstractmethod
    def update(self, state, action, reward, next_state, done):
        """Performs learning step. Returns loss (float) or None."""
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass