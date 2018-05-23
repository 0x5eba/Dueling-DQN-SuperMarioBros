from agent.DQ_agent import DeepQAgent
from agent.utils.base_callback import BaseCallback
from environment.build import build_nes_environment
import os, sys, datetime
import pandas as pd


# setup the output directory
now = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M')
output_dir = '{}/log/{}'.format(".", now)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print('writing results to {}'.format(repr(output_dir)))
weights_file = '{}/weights.h5'.format(output_dir)



env = build_nes_environment()

# build the agent
agent = DeepQAgent(env)
# observe frames to fill the replay memory
agent.observe()

# write some info about the agent's hyperparameters to disk
with open('{}/agent.py'.format(output_dir), 'w') as agent_file:
    agent_file.write(repr(agent))


# train the agent
try:
    callback = BaseCallback(weights_file)
    agent.train(callback=callback)
except KeyboardInterrupt:
    print('canceled training')


# save the training results
scores = pd.Series(callback.scores)
scores.to_csv('{}/scores.csv'.format(output_dir))
losses = pd.Series(callback.losses)
losses.to_csv('{}/losses.csv'.format(output_dir))
# save the weights to disk
agent.model.save_weights(weights_file, overwrite=True)


# close the environment to perform necessary cleanup
env.close()