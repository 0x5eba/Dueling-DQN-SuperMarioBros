"""An environment wrapper for down-sampling frames from RGB to smaller B&W."""
import gym
import cv2
import numpy as np


class DownsampleEnv(gym.ObservationWrapper):
    """An environment that down-samples frames."""

    metadata = {
        'SuperMarioBros': {
            'y': (0, 1),
            'x': (0, 1)
        }
    }

    def __init__(self, env, image_size: tuple, y: int, x: int):
        """
        Create a new down-sampler.
        Args:
            image_size: the size to output frames as
            y: the coordinates of the Y padding (y_top, y_bottom)
            x: the coordinates of the X padding (x_left, x_right)
        """
        gym.ObservationWrapper.__init__(self, env)
        self.image_size = image_size
        self.y = y
        self.x = x
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(image_size[1], image_size[0], 1),
            dtype=np.uint8
        )

    def observation(self, frame):
        # crop the image to the playable space
        frame = frame[self.y[0]:-self.y[1], self.x[0]:-self.x[1]]
        # convert the frame from RGB to gray scale
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # resize the frame to the expected shape. bilinear is the default
        # interpolation method
        frame = cv2.resize(frame, self.image_size)
        return frame[:, :, np.newaxis]


# explicitly define the outward facing API of this module
__all__ = [DownsampleEnv.__name__]
