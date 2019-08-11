
from collections import deque
import copy

from dowel import logger, tabular
import numpy as np
import torch

from meta_agents.utils import np_to_torch, torch_to_np


def clone_qf(qf):
    # TODO add qfunction cloning
    return qf


class SAC:
    def __init__(self,
                 env_spec,
                 policy,
                 qfs,  # use more than one q-functions to gain more stable performance
                 discount=0.99,
                 reward_scale=1.0,
                 policy_lr=1e-3,
                 policy_optimizer_cls=torch.optim.Adam,
                 qf_lr=1e-3,
                 qf_optimizer_cls=torch.optim.Adam,
                 soft_target_tau=1e-2,
                 target_update_period=1,
                 use_automatic_entropy_tuning=True,
                 target_entropy=None,
                 log_alpha=1.):
        self._env_spec = env_spec
        self._policy = policy

        # Q-functions and target q-functions
        self._qfs = qfs
        self._n_qfs = len(self._qfs)
        self._target_qfs = [clone_qf(qf) for qf in self._qfs]

        self._discount = discount
        self._reward_scale = reward_scale
        self._qf_lr = qf_lr

        self._policy_optimizer = policy_optimizer_cls(
            self.policy.parameters(),
            lr=policy_lr,
        )

        # q-functions optimization
        self._qf_optimizers = []
        self._qf_criterion = torch.nn.MSELoss()
        for qf in self._qfs:
            self._qf_optimizers.append(
                qf_optimizer_cls(qf.parameters(), lr=qf_lr,))

        self._soft_target_tau = soft_target_tau
        self._target_update_period = target_update_period

        # TODO: per-task alpha (task->alpha) for multi-task learning
        self._use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self._use_automatic_entropy_tuning:
            if target_entropy:
                self._target_entropy = target_entropy
            else:
                self._target_entropy = -np.prod(self._env_spec.action_space.shape).item()
            self.log_alpha = torch.zeros(1, requires_grad=True)
            # Use the same type of optimizer as the policy's at this point.
            # TODO consider have an argument for this??
            self.alpha_optimizer = policy_optimizer_cls(
                [self.log_alpha], lr=policy_lr)
        else:
            # TODO check this
            self.log_alpha = log_alpha

    def optimize_policy(self, itr, samples):
        transitions = np_to_torch(samples)
        observations = transitions['observation']
        rewards = transitions['reward']
        actions = transitions['action']
        next_observations = transitions['next_observation']
        terminals = transitions['terminal']

        rewards = rewards.reshape(-1, 1)
        terminals = terminals.reshape(-1, 1)

        next_inputs = next_observations
        inputs = observations
        with torch.no_grad():
            next_actions = self.target_policy(next_inputs)
            target_qvals = self.target_qf(next_inputs, next_actions)

        # Policy loss
        policy_dist = self._policy(observations)
        new_actions = policy_dist.rsample()
        log_pi = policy_dist.log_likelihood(new_actions)

        # get alpha:
        if self._use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self._target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha = 1.

        # TODO: extend this to more than two q-functions
        # TODO: check the correctness of the min usage..
        q_new_actions = torch.min(
            self._qfs[0](obs, new_actions),
            self._qfs[1](obs, new_actions),
        )
        policy_loss = (alpha * log_pi - q_new_actions).mean()

        # Q-functions loss
        q_preds = [qf([obs, actions]) for qf in self._qfs]
        # We need a new set of symbolics!
        new_policy_dist = self._policy(next_observations)
        new_next_actions = policy_dist.rsample()
        next_log_pi = policy_dist.log_likelihood(new_next_actions)

        target_q_values = torch.min(
            self._target_qfs[0](obs, new_next_actions),
            self._target_qfs[1](obs, new_next_actions),
        ) - alpha * next_log_pi

        q_target = self._reward_scale * rewards + (1. - terminals) * self._discount * target_q_values
        q_target = q_target.detach()
        q_losses = [self._qf_criterion(q_pred, q_target) for q_pred in q_preds]

        '''Optimize q-functions'''
        for loss, opt in zip(q_losses, self._qf_optimizers):
            opt.zero_grad()
            loss.backward()
            opt.step()

        '''Optimize policy'''
        self._policy_optimizer.zero_grad()
        policy_loss.backward()
        self._policy_optimizer.step()
