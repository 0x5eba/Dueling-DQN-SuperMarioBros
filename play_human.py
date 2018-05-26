"""A script to play the environment with a keyboard."""
import os
from environment.build import build_nes_environment
from PIL import Image
from datetime import datetime
from gym.utils.play import play
import gym_super_mario_bros
import numpy as np


# return a sorted tuple instead of a sorted list
sorted_tuple = lambda x: tuple(sorted(x))

# create a directory to write screen captures to
output_dir = 'screen_capture'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

counter = 0
game_played = 1
def callback(s, s2, a, r, d, i) -> None:
    """
    Respond to the step callback in the play method.

    Args:
        s: the state before the action was fired
        s2: the state after the action was fired
        a: the action to fire
        r: the reward observed as a result of the action
        d: a flag denoting if the episode is over
        i: the information dictionary from the step

    Returns:
        None

    """

    global counter, game_played
    if counter % 4 == 0:
        dir_numpydata = '{}/{}/{}/'.format(output_dir,"game_"+str(game_played), "episode_"+str(counter))
        os.makedirs(dir_numpydata)
        with open(dir_numpydata + "state2.npy", "w") as f:
            np.save(dir_numpydata + "state2.npy", s2)
        with open(dir_numpydata + "action.npy", "w") as f:
            np.save(dir_numpydata + "action.npy", np.array(a))
        with open(dir_numpydata + "reward.npy", "w") as f:
            np.save(dir_numpydata + "reward.npy", np.array(r))
    counter += 1

    if d:
        counter = 0
        game_played += 1


# Mapping of buttons on the NES joy-pad to keyboard keys
up =    ord('w')
down =  ord('s')
left =  ord('a')
right = ord('d')
A =     ord('o')
B =     ord('p')


# A mapping of pressed key combinations to discrete actions in action space
keys_to_action = {
    (): 0,
    sorted_tuple((left, right, )): 0,
    (up, ): 1,
    (down, ): 2,
    (left, ): 3,
    (right, ): 4,
    sorted_tuple((left, A, )): 5,
    sorted_tuple((left, B, )): 6,
    sorted_tuple((left, A, B, )): 7,
    sorted_tuple((right, A, )): 8,
    sorted_tuple((right, B, )): 9,
    sorted_tuple((right, A, B, )): 10,
    (A, ): 11,
    (B, ): 12,
    sorted_tuple((A, B)): 13
}


# Create the environment and play the game
env = build_nes_environment()

play(env, keys_to_action=keys_to_action, callback=callback)
