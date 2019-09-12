import copy
import pickle

import numpy as np
import pytest

from garage.envs import GarageEnv
from meta_agents.policies import GaussianMLPPolicy

from tests.fixtures.envs.dummy import DummyBoxEnv


@pytest.mark.parametrize('policy_cls', [GaussianMLPPolicy])
def test_policy_serialization(policy_cls):
    env = GarageEnv(DummyBoxEnv())
    policy = policy_cls(env_spec=env.spec, hidden_sizes=(8, 8))

    round_trip = pickle.loads(pickle.dumps(policy))
    round_trip2 = copy.deepcopy(policy)
    ori_params = policy.parameters()
    round_trip_params = round_trip.parameters()
    round_trip_params2 = round_trip2.parameters()

    for p, p1, p2 in zip(ori_params, round_trip_params, round_trip_params2):
        assert np.array_equal(
            p1.detach().numpy(), p.detach().numpy())
        assert np.array_equal(
            p2.detach().numpy(), p.detach().numpy())
