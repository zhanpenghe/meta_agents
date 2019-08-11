"""Pytorch modules."""

from meta_agents.modules.gaussian_mlp_module import \
    GaussianMLPIndependentStdModule, GaussianMLPModule, \
    GaussianMLPTwoHeadedModule
from meta_agents.modules.mlp_module import MLPModule
from meta_agents.modules.multi_headed_mlp_module import MultiHeadedMLPModule

__all__ = [
    'MLPModule', 'MultiHeadedMLPModule', 'GaussianMLPModule',
    'GaussianMLPIndependentStdModule', 'GaussianMLPTwoHeadedModule'
]
