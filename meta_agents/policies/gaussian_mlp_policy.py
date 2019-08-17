"""GaussianMLPPolicy."""
from torch import nn

from meta_agents.modules import GaussianMLPModule
from meta_agents.policies import Policy


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

    def get_actions(self, observations):
        """Get actions given observations."""
        with torch.no_grad():
            dist = self.forward(observations)
            return dist.rsample().detach().numpy(), dict()

    def get_action(self, observation):
        with torch.no_grad():
            x = self.forward(observation)
            return x.numpy(), dict()
