import numpy as np
import pygame as pg
import math

from Const import*
from Class_Def import Celestial_Core, Planet, SpaceShip


def update_position(planets,rockets):
	
	total_fx = [0] * len(planets+rockets)
	total_fy = [0] * len(planets+rockets)

	for n,i in enumerate(planets+rockets):
		for j in planets+rockets:
			if i != j:
				fx, fy = i.attraction(j)
				total_fx[n-1] += fx
				total_fy[n-1] += fy
		if (n>=len(planets)):
			# print(i.x_vel, i.y_vel)
			if i.reactor1 :
				total_fx[n-1] += - i.accelerator()
			if i.reactor2 :
				total_fx[n-1] += - i.accelerator() * math.cos(2.*math.pi/3.)
				total_fy[n-1] += - i.accelerator() * math.sin(2.*math.pi/3.)
			if i.reactor3 :
				total_fx[n-1] += - i.accelerator() * math.cos(- 2.*math.pi/3.)
				total_fy[n-1] += - i.accelerator() * math.sin(- 2.*math.pi/3.)
				
	for n,i in enumerate(planets+rockets):
		i.x_vel += total_fx[n-1] / i.mass * TIMESTEP
		i.y_vel += total_fy[n-1] / i.mass * TIMESTEP

		i.x += i.x_vel * TIMESTEP
		i.y += i.y_vel * TIMESTEP
		i.orbit.append((i.x, i.y))
		while(len(i.orbit)>5000):
			i.orbit.pop(0)
		
	for i in rockets:
		for j in planets:
			if (i.x - j.x)**2 + (i.y - j.y)**2 < (i.radius + j.radius)**2:
				del i
				rockets.remove(i)
	return planets,rockets


def Launch(Phi, Force, planet):
	
	rocket = SpaceShip(planet.x , planet.y  + 0.00256954*AU, 30_000, 12, RED)
	rocket.x_vel = planet.x_vel -1.022*1000
	rocket.y_vel = planet.y_vel 
	
	return rocket



def get_distance(core1, core2):
	return math.sqrt( (core1.x-core2.x)**2 + (core1.y-core2.y)**2)

def is_orbiting(core1, core2, start=0):
	hist_dist = []
	#if mae between core1 and core 2 get_distance < 1% return True :
	
	for i in range(start,len(core1.orbit)):
		hist_dist.append(math.sqrt( (core1.orbit[i][0] - core2.orbit[i][0])**2 + (core1.orbit[i][1]-core2.orbit[i][1])**2))

	if (max(hist_dist) - min(hist_dist)) < 0.01 * min(hist_dist):
		return True
	else:
		return False
