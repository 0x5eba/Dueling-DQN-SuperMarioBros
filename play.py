from agent.DQ_agent import DeepQAgent
from agent.utils.base_callback import BaseCallback
from environment.build import build_nes_environment
import pandas as pd
import os, sys

PATH_MODEL = ""

# set up the weights file
weights_file = '{}/weights.h5'.format(PATH_MODEL)
# make sure the weights exist
if not os.path.exists(weights_file):
    print('{} not found!'.format(weights_file))
    sys.exit(-1)

# build the environment
env = build_nes_environment()
# build the agent without any replay memory since we're just playing, load
# the trained weights, and play some games
agent = DeepQAgent(env, replay_memory_size=0)
agent.model.load_weights(weights_file)
agent.target_model.load_weights(weights_file)
agent.play()


# collect the game scores
scores = pd.Series(env.unwrapped.episode_rewards)
scores.to_csv('{}/final_scores.csv'.format(PATH_MODEL))
# print some stats
print('min ', scores.min())
print('mean ', scores.mean())
print('max ', scores.max())

env.close()