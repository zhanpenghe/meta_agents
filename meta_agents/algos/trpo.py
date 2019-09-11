
from garage.np.algos import BatchPolopt
from torch.distributions.kl import kl_divergence

from meta_agents.samplers.single_task_sampler import SingleTaskSampler
from meta_agents.torch_utils import np_to_torch, detach_distribution


def surrogate_loss(samples, policy):
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
    if 'old_dist_info' not in samples.keys():
        old_dist = detach_distribution(dist)
    else:
        old_dist = dist.__class__()(**samples['old_dist_info'])

    log_likeli_ratio = dist.log_prob(actions) - old_dist.log_prob(actions)
    ratio = torch.exp(log_likeli_ratio)
    surr_loss = - torch.mean(ration * advantages, dim=0)
    return surr_loss, dist, old_dist


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

    def train_once(self, samples_data):
        samples = np_to_torch(samples_data)
        # loss, dist, old_dist = surrogate_loss(samples, self.policy)
        # kl_before = kl_divergence(dist, old_dist)
        self._trpo_step(samples, surrogate_loss, kl_divergence)

    def process_samples(self, itr, paths):
        # We will never use a `process_samples` method under a algo
        # since we have preprocessor in meta_agents
        raise NotImplementedError

    def _trpo_step(self, samples, loss_func, constraint):
        
        # Here, we do have detached old_dist so we dont need to do this
        # in the future.
        loss, dist, old_dist = surrogate_loss(samples, self.policy)
        grads = torch.autograd.grad(loss, self.policy.parameters())
        grads = parameters_to_vector(grads)

        hessian_vector_product = self.hessian_vector_product(samples_data)

    def hessian_vector_product(samples_data, damping=1e-2):
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

    def kl_divergence(self, samples_data, old_dists):
        
