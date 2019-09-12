"""Base Policy."""
import abc

import numpy as np


class Policy(abc.ABC):
    """
    Policy base class without Parameterzied.

    Args:
        env_spec (meta_agents.envs.env_spec.EnvSpec): Environment specification.

    """

    def __init__(self, env_spec):
        self._env_spec = env_spec

    def get_action(self, observation):
        """Get action given observation."""
        pass

    @abc.abstractmethod
    def get_actions(self, observations):
        """Get actions given observations."""
        pass

    @property
    def observation_space(self):
        """Observation space."""
        return self._env_spec.observation_space

    @property
    def action_space(self):
        """Policy action space."""
        return self._env_spec.action_space

    @property
    def vectorized(self):
        return True
