from  meta_agents.samplers.base import SampleProcessor
from meta_agents.utils.utils import stack_tensor_dict_list


class OffPolicySampleProcessor(SampleProcessor):

    def __init__(self):
        pass

    def process_samples(self, paths):
        '''
        paths --> batch
        This method also add a next_observations field to the batch.
        '''
        for p in paths:
            p['next_observation'] = p['observations'][1:, ...]
            p['observation'] = p['observations'][:-1, ...]
            p['action'] = p['actions'][:-1, ...]
            p['reward'] = p['rewards'][:-1, ...]
            p['done'] = p['dones'][:-1, ...]

        return stack_tensor_dict_list(paths)
