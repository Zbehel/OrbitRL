import gym
from gym import spaces
import numpy as np

import random

from Const import *
from Class_Def import *

# from main import SatSim
from utils import *

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Planet_Sim")
FONT = pygame.font.SysFont("Times", 16)

class CustomEnv(gym.Env):

    def __init__(self):

        super(CustomEnv, self).__init__()

        self.action_space = spaces.Discrete(8)
        self.observation_space = spaces.Dict({
            'distanceT': spaces.Box(low=0, high=np.inf, shape=(1,),dtype=np.float64),
            'orbit': spaces.Discrete(2)
        })
        
        self.previous_obs = {}
        self.previous_obs={'distanceT' : np.array([0,]),
                           'orbit' : False}
        self.obs= {}
        self.obs={'distanceT' : np.array([0,]),
                           'orbit' : False}

        self.planets = Solar_System()
        self.rockets = []
        self.rockets.append(Launch(2*math.pi*random.random(), 24*random.random(),self.planets[3]))
        self.Flag = 0
        self.Target = random.randint(0,len(self.planets)-1)
        pygame.display.init()
        pygame.font.init()
        self.clock = pygame.time.Clock()
        self.Scale = 250/AU

        self.done = False
        self.reward = 0

    def reset(self):
        
        self.planets = Solar_System()
        self.rockets = []
        self.rockets.append(Launch(2*math.pi*random.random(), 24*random.random(),self.planets[3]))
        self.Target = random.randint(0,len(self.planets)-1)

        self.Flag = 0


        ############# OBSERVATIONS
        self.obs= {'distanceT' : get_distance(self.rockets[-1],self.planets[self.Target])/AU,
                'orbit' : False}
        
        self.done = False
        self.reward = 0
        return self.obs

    def step(self, action):   

        self.previous_obs = self.obs.copy()
        # print(self.obs)
        if action == 1 :
            self.rockets[-1].ChangeR(1)
        elif action == 2 :
            self.rockets[-1].ChangeR(2)
        elif action == 3 :
            self.rockets[-1].ChangeR(3)
        elif action == 4 :
            self.rockets[-1].ChangeR(1)
            self.rockets[-1].ChangeR(2)
        elif action == 5 :
            self.rockets[-1].ChangeR(1)
            self.rockets[-1].ChangeR(3)
        elif action == 6 :
            self.rockets[-1].ChangeR(2)
            self.rockets[-1].ChangeR(3)
        elif action == 7 :
            self.rockets[-1].ChangeR(1)
            self.rockets[-1].ChangeR(2)
            self.rockets[-1].ChangeR(3)
        
        for i in range(Step_p_frame):
            self.planets,self.rockets = update_position(self.planets,self.rockets)

        self.obs = {'distanceT' : get_distance(self.rockets[-1],self.planets[self.Target])/AU,
                'orbit' : is_orbiting(self.rockets[-1], self.planets[self.Target])}

        if self.obs['orbit'] and (self.obs['distanceT']<self.planets[self.Target].radius*1000*10/AU) and (self.rockets[-1].motor_off()):
            self.reward += 50_000.0  # assign a positive reward if the rocket is in orbit around the selected planet
            self.done = True
            print('a')
        elif abs(self.obs['distanceT'] - self.planets[self.Target].radius*1000*10) < abs(self.previous_obs['distanceT'] - self.planets[self.Target].radius*1000*10): 
            self.reward += 1.0
        elif self.obs['distanceT'] >= 150: 
            self.reward -= 50.0
            self.done = True
        elif self.rockets[-1].fuel == 0:
            self.reward -=50.0
            self.done
        else:
            self.reward -= 2.0  # assign a negative reward if the rocket get away from the target
        self.obs['distanceT'] = np.array([get_distance(self.rockets[-1],self.planets[self.Target])/AU,])
        info={}
        return self.obs, self.reward, self.done,info

    def render(self, mode='human'):

        WIN.fill((0,0,0))
        self.clock.tick(999)

        shift_x = - (self.planets[self.Target].x + self.rockets[-1].x) / 2.
        shift_y = - (self.planets[self.Target].y + self.rockets[-1].y) / 2.
        
        self.Scale = 250/get_distance(self.rockets[-1],self.planets[self.Target])

        for planet in self.planets+self.rockets:
            planet.draw(WIN, shift_x, shift_y, self.Scale)
        draw_dist(WIN,shift_x,shift_y,self.rockets[-1],self.planets[self.Target],self.Scale)

        #Show fps :
        fps = self.clock.get_fps()
        fps_text = FONT.render(f"FPS: {int(fps)}", 1, WHITE)
        WIN.blit(fps_text, (WIDTH - fps_text.get_width() - 10, 10))

        speed_text = FONT.render(f"Step per Frame: {int(Step_p_frame)}", 1, WHITE)
        WIN.blit(speed_text, (WIDTH - speed_text.get_width() - 10, 25))
        
        
        Dist_text = FONT.render(f"Distancce to Target :{round(get_distance(self.planets[self.Target],self.rockets[-1])/AU,5)}", 1, WHITE)
        WIN.blit(Dist_text, (WIDTH-Dist_text.get_width()-10, HEIGHT-Dist_text.get_height()-25))
        
        Orbit_text = FONT.render(f"Is Orbiting:{self.obs['orbit']}", 1, WHITE)
        WIN.blit(Orbit_text, (WIDTH-Orbit_text.get_width()-10, HEIGHT-Dist_text.get_height()-10))
        

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        pygame.display.update()

    def get_obs(self):
        # return the observation
        return self.obs
