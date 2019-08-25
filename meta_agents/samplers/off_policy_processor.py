import numpy as np

from  meta_agents.samplers.base import SampleProcessor
from meta_agents.utils.utils import discount_cumsum, stack_tensor_dict_list


class OffPolicySampleProcessor(SampleProcessor):

    def __init__(self, discount=0.99):
        self.discount = discount

    def process_samples(self, paths):
        '''
        The replay buffer is weird since its add_transition
        function actually only cares about episode....
        So, this sample processor still maintain the notion
        of episode.

        TODO rewrite replay buffers to make it right

        '''
        samples_data, all_paths = self._compute_samples_data(paths)
        self._log_path_stats(all_paths, log=True)
        return samples_data

    def _compute_samples_data(self, paths):
        all_samples_data = []
        all_paths = []
        for p in paths:
            p["returns"] = discount_cumsum(p["rewards"], self.discount)
            samples_data = dict(
                next_observations=p['observations'][1:, ...],
                observations=p['observations'][:-1, ...],
                actions=p['actions'][:-1, ...],
                rewards=p['rewards'][:-1, np.newaxis],
                dones=p['dones'][:-1, np.newaxis],)
            all_samples_data.append(samples_data)

        assert len(all_samples_data) == len(paths)
        return all_samples_data, paths
