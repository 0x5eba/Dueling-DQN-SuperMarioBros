import gym
import numpy as np

class RewardChace(gym.Wrapper):
    """a wrapper that caches rewards of episodes."""
    def __init__(self, env) -> None:
        """
        Initialize a reward caching environment.
        Args:
            env: the environment to wrap
        """
        gym.Wrapper.__init__(self, env)
        self._score = 0
        self.env.unwrapped.episode_rewards = []

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self._score += reward
        if done:
            self.env.unwrapped.episode_rewards.append(self._score)
            self._score = 0
        return state, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)



class ClipRewardEnv(gym.RewardWrapper):
    """An environment that clips rewards in {-1, 0, 1}."""

    def __init__(self, env):
        """Initialize a new reward clipping environment."""
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {-1, 0, +1} using its sign."""
        return np.sign(reward)
