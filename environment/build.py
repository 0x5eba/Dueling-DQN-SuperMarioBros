from environment.reward import RewardChace, ClipRewardEnv
from environment.downsample_env import DownsampleEnv
from environment.frame_stack_env import FrameStackEnv
import gym_super_mario_bros

def build_nes_environment(
    image_size: tuple=(84, 84),
    clip_rewards: bool=True, # sigmoid
    agent_history_length: int=4,
    monitor_dir: str=None
):
    """
    Build and return a configured NES environment.

    Args:
        image_size: the size to down-sample images to
        clip_rewards: whether to clip rewards in {-1, 0, +1}
        agent_history_length: the size of the frame buffer for the agent
        monitor_dir: the directory to save monitor info to if any

    Returns:
        a gym environment configured for this experiment

    """
                                # SuperMarioBros-1-1-v0
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    # add a reward cache for scoring episodes
    env = RewardChace(env)
    # apply a down-sampler for the given game
    downsampler = DownsampleEnv.metadata['SuperMarioBros']
    env = DownsampleEnv(env, image_size, **downsampler)
    # clip the rewards in {-1, 0, +1} if the feature is enabled
    if clip_rewards:
        env = ClipRewardEnv(env)
    # apply the back history of frames if the feature is enabled
    if agent_history_length is not None:
        env = FrameStackEnv(env, agent_history_length)

    return env