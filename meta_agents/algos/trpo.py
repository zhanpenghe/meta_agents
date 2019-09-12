from collections import OrderedDict

from dowel import logger, tabular
from garage.np.algos import BatchPolopt
import torch
from torch.distributions.kl import kl_divergence
from torch.nn.utils.convert_parameters import (vector_to_parameters,
                                               parameters_to_vector)

from meta_agents.samplers.single_task_sampler import SingleTaskSampler
from meta_agents.torch_utils import np_to_torch, detach_distribution
from meta_agents.samplers.base import SampleProcessor


def surrogate_loss(samples, policy, old_dist=None):
    assert isinstance(samples, dict)
    assert 'observations' in samples.keys()
    assert 'actions' in samples.keys()
    assert 'advantages' in samples.keys()

    observations = samples['observations']
    actions = samples['actions']
    advantages = samples['advantages']
    # forward pass of policy
    dist = policy(observations)
    # currently lets just detach the logprob
    # as old pi
    if old_dist is None:
        old_dist = detach_distribution(dist)
    
    kl = torch.mean(kl_divergence(dist, old_dist))

    log_likeli_ratio = dist.log_prob(actions) - old_dist.log_prob(actions)
    ratio = torch.exp(log_likeli_ratio)
    surr_loss = -torch.mean(ratio * advantages, dim=0)
    return surr_loss, old_dist, kl


def conjugate_gradient(f_Ax, b, cg_iters=10, residual_tol=1e-10):
    p = b.clone().detach()
    r = b.clone().detach()
    x = torch.zeros_like(b).float()
    rdotr = torch.dot(r, r)

    for i in range(cg_iters):
        z = f_Ax(p).detach()
        v = rdotr / torch.dot(p, z)
        x += v * p
        r -= v * z
        newrdotr = torch.dot(r, r)
        mu = newrdotr / rdotr
        p = r + mu * p

        rdotr = newrdotr
        if rdotr.item() < residual_tol:
            break

    return x.detach()


class TRPO(BatchPolopt):

    def __init__(
        self,
        policy,
        baseline,
        discount=.99,
        max_path_length=200,
        n_samples=1,  # This is super weird and I don't think this
                      # need to exist in on policy.
    ):
        super().__init__(
            policy=policy,
            baseline=baseline,
            discount=discount,
            max_path_length=max_path_length,
            n_samples=n_samples,)

        # We only use our own sampler for consistency between single task
        # and meta learning.
        self.sampler_cls = SingleTaskSampler
        self.preprocessor = SampleProcessor(baseline=self.baseline)

    def train(self, runner, batch_size):
        last_return = None

        for epoch in runner.step_epochs():
            for cycle in range(self.n_samples):
                runner.step_path = runner.obtain_samples(
                    runner.step_itr, batch_size)
                last_return = self.train_once(runner.step_itr,
                                              runner.step_path)
                runner.step_itr += 1

        return last_return

    def train_once(self, itr, paths):
        samples_data = self.preprocessor.process_samples(paths)
        samples = np_to_torch(samples_data)
        self._trpo_step(samples, surrogate_loss, kl_divergence)

    def process_samples(self, itr, paths):
        # We will never use a `process_samples` method under a algo
        # since we have preprocessor in meta_agents
        raise NotImplementedError

    def _trpo_step(self, samples, loss_func, constraint, cg_damping=1e-2,
        ls_backtrack_ratio=.5, cg_iters=10, max_ls_steps=10, max_kl=1e-2,):

        old_loss, old_dist, kl_before = surrogate_loss(samples, self.policy)
        grads = torch.autograd.grad(old_loss, self.policy.parameters())
        grads = parameters_to_vector(grads)

        hessian_vector_product = self.hessian_vector_product(samples, damping=cg_damping)
        step_direction = conjugate_gradient(hessian_vector_product, grads, cg_iters)

        # Compute the Lagrange multiplier
        shs = 0.5 * step_direction.dot(hessian_vector_product(step_direction))
        lagrange_multiplier = torch.sqrt(max_kl / shs)

        grad_step = step_direction * lagrange_multiplier
        old_params = parameters_to_vector(self.policy.parameters())

        # Start line search
        step_size = 1.
        backtrack_step = 0
        for _ in range(max_ls_steps):
            vector_to_parameters(old_params - step_size * grad_step,
                                 self.policy.parameters())

            loss, _, kl = surrogate_loss(samples, self.policy, old_dist=old_dist)

            improve = loss - old_loss
            if (improve.item() < 0.0) and (kl.item() < max_kl):
                break
            step_size *= ls_backtrack_ratio
            backtrack_step += 1
        else:
            vector_to_parameters(old_params, self.policy.parameters())
            logger.log('Failed to update parameters')
        tabular.record('backtrack-iters', backtrack_step)
        tabular.record('loss-before', old_loss.item())
        tabular.record('loss-after', loss.item())
        tabular.record('kl-before', kl_before.item())
        tabular.record('kl-after', kl.item())

    def hessian_vector_product(self, samples_data, damping=1e-2):
        """Hessian-vector product, based on the Perlmutter method."""

        def _product(vector):
            kl = self.kl_divergence(samples_data)
            grads = torch.autograd.grad(kl, self.policy.parameters(),
                create_graph=True)
            flat_grad_kl = parameters_to_vector(grads)

            grad_kl_v = torch.dot(flat_grad_kl, vector)
            grad2s = torch.autograd.grad(grad_kl_v, self.policy.parameters())
            flat_grad2_kl = parameters_to_vector(grad2s)
            return flat_grad2_kl + damping * vector

        return _product

    def kl_divergence(self, samples, old_dist=None):
        loss, old_dist_, kl = surrogate_loss(samples, self.policy)
        if old_dist is None:
            old_dist = old_dist_

        inputs = samples['observations']
        new_dist = self.policy(inputs)
        kl = torch.mean(kl_divergence(new_dist, old_dist))
        return kl

    def adapt_policy(self, loss, step_size=1., create_graph=True):
        grads = torch.autograd.grad(loss,
            self.policy.parameters(), create_graph=create_graph)

        updated_params = OrderedDict()
        for (name, param), grad in zip(self.policy.named_parameters(), grads):
            updated_params[name] = param - step_size * grad

        return updated_params
