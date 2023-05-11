from stable_baselines3 import PPO
import os
from custom_env import CustomEnv
import time
import random

# import evaluate_policy :
from stable_baselines3.common.evaluation import evaluate_policy

from model import *

env = CustomEnv()
env.reset()

# Load last model :


model = PPO.load('models/1683743040/240000.zip',env = env)

evaluate_policy(model,env, n_eval_episodes=10,render=True)