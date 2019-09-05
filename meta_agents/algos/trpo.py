
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
        loss, dist, old_dist = surrogate_loss(samples, self.policy)
        kl_before = kl_divergence(dist, old_dist)
        self._trpo_step(loss, kl)

    def process_samples(self, itr, paths):
        # We will never use a `process_samples` method under a algo
        # since we have preprocessor in meta_agents
        raise NotImplementedError

    def _trpo_step(self, loss, constraint):
        pass
