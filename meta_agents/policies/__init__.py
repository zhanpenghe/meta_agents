"""PyTorch Policies."""
from meta_agents.policies.base import Policy
from meta_agents.policies.deterministic_mlp_policy import (
    DeterministicMLPPolicy)
from meta_agents.policies.gaussian_mlp_policy import GaussianMLPPolicy

__all__ = ['DeterministicMLPPolicy', 'GaussianMLPPolicy', 'Policy']
