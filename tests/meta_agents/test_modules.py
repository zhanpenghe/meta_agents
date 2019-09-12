 
from collections import OrderedDict
import pickle

import numpy as np
import pytest
import torch
import torch.nn as nn

from meta_agents.modules import MLPModule


class TestMLPModel:
    # yapf: disable
    @pytest.mark.parametrize('input_dim, output_dim, hidden_sizes', [
        (5, 1, (1, )),
        (5, 1, (2, )),
        (5, 2, (3, )),
        (5, 2, (1, 1)),
        (5, 3, (2, 2)),
    ])
    # yapf: enable
    def test_output_values(self, input_dim, output_dim, hidden_sizes):
        input_val = torch.ones([1, 5], dtype=torch.float32)
        module = MLPModule(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_nonlinearity=None,
            hidden_sizes=hidden_sizes,
            hidden_w_init=nn.init.ones_,
            output_w_init=nn.init.ones_)
        output = module(input_val)

        expected_output = torch.full([1, output_dim],
                                     fill_value=5 * np.prod(hidden_sizes),
                                     dtype=torch.float32)

        assert torch.all(torch.eq(output, expected_output))

    # yapf: disable
    @pytest.mark.parametrize('input_dim, output_dim, hidden_sizes', [
        (5, 1, (1, )),
        (5, 1, (2, )),
        (5, 2, (3, )),
        (5, 2, (1, 1)),
        (5, 3, (2, 2)),
    ])
    # yapf: enable
    def test_is_pickleable(self, input_dim, output_dim, hidden_sizes):
        input_val = torch.ones([1, 5], dtype=torch.float32)
        module = MLPModule(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_nonlinearity=None,
            hidden_sizes=hidden_sizes,
            hidden_w_init=nn.init.ones_,
            output_w_init=nn.init.ones_)
        output1 = module(input_val)

        h = pickle.dumps(module)
        model_pickled = pickle.loads(h)
        output2 = model_pickled(input_val)

        assert np.array_equal(torch.all(torch.eq(output1, output2)), True)


    def test_forwardpass_with_new_params(self):
        input_val = torch.ones([1, 5], dtype=torch.float32)
        hidden_sizes = (3, 3)
        output_dim = 2

        module = MLPModule(
            input_dim=5,
            output_dim=output_dim,
            hidden_nonlinearity=None,
            hidden_sizes=hidden_sizes,
            hidden_w_init=nn.init.ones_,
            output_w_init=nn.init.ones_)
        output = module(input_val)
        expected_output = torch.full([1, output_dim],
                                    fill_value=5 * np.prod(hidden_sizes),
                                    dtype=torch.float32)

        assert torch.all(torch.eq(output, expected_output))

        new_params = OrderedDict()
        for name, param in module.named_parameters():
            new_params[name] = torch.full_like(param, 2)

        output_with_params = module(input_val, params=new_params)
        expected_output_with_params = torch.Tensor([[446, 446]])
        assert torch.all(torch.eq(output_with_params, expected_output_with_params))
