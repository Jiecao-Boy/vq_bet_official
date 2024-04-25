from gym_custom.envs.registration import registry, register, make, spec

# Custom
# ----------------------------------------

# Practice1 environments
#
register(
    id='Retargeting-v0',
    entry_point='gym_custom.envs.custom.practice_env:PracticeEnv1',
    max_episode_steps=1000,
    reward_threshold=0.0,
    kwargs={},
)