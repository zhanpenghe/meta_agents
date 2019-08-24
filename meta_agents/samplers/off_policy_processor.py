from  meta_agents.samplers.base import SampleProcessor
from meta_agents.utils.utils import stack_tensor_dict_list


class OffPolicySampleProcessor(SampleProcessor):

    def __init__(self):
        pass

    def process_samples(self, paths):
        '''
        The replay buffer is weird since its add_transition
        function actually only cares about episode....
        So, this sample processor still maintain the notion
        of episode.

        TODO rewrite replay buffers to make it right

        '''
        processed_paths = []

        for p in paths:
            samples_data = dict(
                next_observations=p['observations'][1:, ...],
                observations=p['observations'][:-1, ...],
                actions=p['actions'][:-1, ...],
                rewards=p['rewards'][:-1, ...],
                dones=p['dones'][:-1, ...],)
            processed_paths.append(samples_data)

        assert len(processed_paths) == len(paths)
        return processed_paths
