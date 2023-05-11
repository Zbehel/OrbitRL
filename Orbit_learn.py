from stable_baselines3 import PPO
import os
from custom_env import CustomEnv
import time
import random

from stable_baselines3.common.env_checker import check_env
from model import *

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"
obsdir = f"obs/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

if not os.path.exists(obsdir):
	os.makedirs(obsdir)
env = CustomEnv()
env.reset()

model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log=logdir)
callback = SaveObsCallback(save_dir=obsdir)


TIMESTEPS = 10000
iters = 0
while True:
	iters += 1
	
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, progress_bar=True, tb_log_name=f"PPO",callback=callback)
	model.save(f"{models_dir}/{TIMESTEPS*iters}")
	env.render()

# for episode in range(1,20):
# 	state = env.reset()
# 	done = False
# 	score = 0

# 	while not done:
		
# 		action = env.action_space.sample()
# 		n_state , reward, done, info = env.step(action)
# 		score+=reward
# 		env.render()
# 	print('Episode:{} Score:{}'.format(episode,score))
