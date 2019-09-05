"""Utility functions for PyTorch."""
import torch
from torch.distributions import Categorical, Normal, MultivariateNormal


def np_to_torch(array_dict):
    """
    Convert numpy arrays to PyTorch tensors.

     Args:
        dict (dict): Dictionary of data in numpy arrays.

    Returns:
       Dictionary of data in PyTorch tensors.

    """
    for key, value in array_dict.items():
        array_dict[key] = torch.FloatTensor(value)
    return array_dict


def torch_to_np(value_in):
    """
    Convert PyTorch tensors to numpy arrays.

     Args:
        value_in (tuple): Tuple of data in PyTorch tensors.

    Returns:
       Tuple of data in numpy arrays.

    """
    value_out = []
    for v in value_in:
        value_out.append(v.numpy())
    return tuple(value_out)


def detach_distribution(pi):
    if isinstance(pi, Categorical):
        distribution = Categorical(logits=pi.logits.detach())
    elif isinstance(pi, Normal):
        distribution = Normal(loc=pi.loc.detach(), scale=pi.scale.detach())
    elif isinstance(pi, MultivariateNormal):
        distribution = MultivariateNormal(
            loc=pi.loc.detach, covariance_matrix=pi.covariance_matrix.detach())
    else:
        raise NotImplementedError('Only `Categorical` and `Normal` '
                                  'policies are valid policies.')
    return distribution
