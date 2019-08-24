
from collections import deque
import copy

from dowel import logger, tabular
import numpy as np
import torch

from meta_agents.algos.off_policy_rl_algorithm import OffPolicyRLAlgorithm
from meta_agents.torch_utils import np_to_torch, torch_to_np


class SAC(OffPolicyRLAlgorithm):
    '''
    Soft Actor-Critic
    '''
    def __init__(self,
                 env_spec,
                 policy,
                 qfs,  # use more than one q-functions to gain more stable performance
                 replay_buffer,
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

        super().__init__(
            env_spec=env_spec,
            policy=policy,
            qf=qfs,
            replay_buffer=replay_buffer,
        )
        self._env_spec = env_spec
        self.policy = policy
        self.target_policy = copy.deepcopy(policy)

        # Q-functions and target q-functions
        self._qfs = qfs
        self._n_qfs = len(self._qfs)
        # TODO: Add support to more than 2 qfunctions
        assert self._n_qfs == 2, 'Currently, we only support two q functions'
        self._target_qfs = [copy.deepcopy(qf) for qf in self._qfs]

        self._discount = discount
        self._reward_scale = reward_scale
        self._qf_lr = qf_lr

        self.policy_optimizer = policy_optimizer_cls(
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

    def train_once(self, itr, paths):
        return self.optimize_policy(itr, paths)

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

        # Policy loss
        policy_dist = self.policy(observations)
        new_actions = policy_dist.rsample()
        log_pi = policy_dist.log_prob(new_actions)

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
            self._qfs[0](observations, new_actions),
            self._qfs[1](observations, new_actions),
        )
        policy_loss = (alpha * log_pi - q_new_actions).mean()

        # Q-functions loss
        q_preds = [qf(observations, actions) for qf in self._qfs]
        # We need a new set of symbolics!
        new_policy_dist = self.policy(next_observations)
        new_next_actions = policy_dist.rsample()
        next_log_pi = policy_dist.log_prob(new_next_actions).unsqueeze(-1)

        target_q_values = torch.min(
            self._target_qfs[0](observations, new_next_actions),
            self._target_qfs[1](observations, new_next_actions),
        ) - alpha * next_log_pi

        q_target = self._reward_scale * rewards + (1. - terminals) * self._discount * target_q_values
        q_target = q_target.detach()

        q_losses = [self._qf_criterion(q_pred, q_target) for q_pred in q_preds]

        '''Optimize q-functions'''
        for loss, opt in zip(q_losses, self._qf_optimizers):
            opt.zero_grad()
            loss.backward()
            opt.step()

        q_losses_np = [l.detach().numpy() for l in q_losses]
        mean_q_loss = np.mean(q_losses_np, axis=0)

        '''Optimize policy'''
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        policy_loss_np = policy_loss.detach().numpy()
        if len(policy_loss_np.shape) != 0:
            raise ValueError('The dimension of policy loss is not correct!')
        else:
            policy_loss = policy_loss_np[np.newaxis, ...][0]
        return policy_loss, mean_q_loss
