"""GaussianMLPPolicy."""
import numpy as np
import torch

from meta_agents.modules import GaussianMLPModule
from meta_agents.policies import Policy
from meta_agents.torch_utils import np_to_torch


class GaussianMLPPolicy(GaussianMLPModule, Policy):
    """
    GaussianMLPPolicy.

    A policy that contains a MLP to make prediction based on a gaussian
    distribution.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        module : GaussianMLPModule to make prediction based on a gaussian
        distribution.
    :return:

    """

    def __init__(self, env_spec, *args, **kwargs):
        Policy.__init__(self, env_spec)

        # TODO add more options for the internal modules
        super().__init__(
            input_dim=self._env_spec.observation_space.flat_dim,
            output_dim=self._env_spec.action_space.flat_dim,
            *args,
            **kwargs)

    def get_actions(self, observations, params=None):
        """Get actions given observations."""
        meta = False
        # This method now handles both the meta case
        # and the single task case.
        if isinstance(observations, list):
            meta = True
            meta_batch_size = len(observations)
            observations = np.concatenate(observations)
            assert len(observations.shape) == 2 and\
                observations.shape[1] == self._env_spec.observation_space.flat_dim
        # numpy to torch
        observations = torch.Tensor(observations)
        with torch.no_grad():
            dist = self.forward(observations, params=params)
            actions = dist.rsample().detach().numpy()

        infos = dict()
        if meta:
            actions = np.split(actions, meta_batch_size, axis=0)
            infos = [
                [dict()] * actions[t].shape[0]
                for t in range(len(actions))
            ]
        return actions, infos

    def get_action(self, observation, params=None):
        with torch.no_grad():
            x = self.forward(observation, params=params)
            return x.numpy(), dict()
