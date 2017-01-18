from gym.envs.registration import register
from gym.scoreboard.registration import add_task, add_group


register(
    id='awentzonline/QBrush-Final-v0',
    entry_point='gym_qbrush.environments.final_quality_reward:FinalQualityRewardEnv',
    reward_threshold=999999.0,
    tags={'wrapper_config.TimeLimit.max_episode_steps': 2000},
    kwargs={
        'canvas_channels': 1
    },
)

register(
    id='awentzonline/QBrush-Step-v0',
    entry_point='gym_qbrush.environments.step_quality_reward:StepQualityRewardEnv',
    reward_threshold=999999.0,
    tags={'wrapper_config.TimeLimit.max_episode_steps': 2000},
    kwargs={
        'canvas_channels': 1
    },
)
