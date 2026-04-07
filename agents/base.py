"""Abstract base class all agents must implement."""

from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """
    Contract:
      obs  = env.reset()
      while not done:
          action = agent.act(obs)   # must return a valid action dict
          obs, reward, done, info = env.step(action)
    """

    @abstractmethod
    def act(self, observation: dict) -> dict:
        """Return one action dict given the current observation."""
        ...

    def reset(self) -> None:
        """Optional: called before each episode."""
        pass

    def name(self) -> str:
        return self.__class__.__name__