from meta_agents.samplers import MetaSampler


class SingleTaskSampler(MetaSampler):

    def __init__(
        self,
        env,
        policy,
        n_rollouts=10,
        max_path_length=100,
        n_envs=None,
        parallel=False,):

        super().__init__(
            env=env,
            policy=policy,
            rollouts_per_meta_task=n_rollouts,
            meta_batch_size=1,
            max_path_length=max_path_length,
            envs_per_task=n_envs,
            parallel=parallel,)

    def update_tasks(self):
        pass

    def obtain_samples(self, *args, **kwargs):
        return super().obtain_samples(*args, **kwargs)[0]
