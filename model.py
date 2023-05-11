import numpy as np
import csv
import os
from custom_env import *
from stable_baselines3.common.callbacks import BaseCallback



class SaveObsCallback(BaseCallback):
    def __init__(self, save_dir, verbose=0):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.observations = []
        self.cnt =0


    def _on_step(self):
        obs = self.model.env.envs[0].get_obs()  # get the observations from the environment
        self.observations.append(obs['distanceT'])  # append the observations to the list
        

        return True
    
    def _on_training_end(self):
        save_path = os.path.join(self.save_dir, f'{self.cnt}.csv')
        with open(save_path, 'w', newline='\n') as csvfile:
            writer = csv.writer(csvfile, delimiter='\n')
            writer.writerow(self.observations)
        self.cnt += 1
        self.observations = []
        return True