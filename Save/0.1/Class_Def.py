import pygame
import math

from Const import *
pygame.init()


class Celestial_Core:

	def __init__(self, x, y, mass):
		self.x = x
		self.y = y
		self.mass = mass

		self.orbit = []

		self.x_vel = 0
		self.y_vel = 0

	def draw(self, win, shift_x , shift_y, Scale):
		pass
		

	def attraction(self, other):
		other_x, other_y = other.x, other.y
		distance_x = other_x - self.x
		distance_y = other_y - self.y
		distance = math.sqrt(distance_x ** 2 + distance_y ** 2)


		force = G * self.mass * other.mass / distance**2
		theta = math.atan2(distance_y, distance_x)
		force_x = math.cos(theta) * force
		force_y = math.sin(theta) * force
		return force_x, force_y



class Planet(Celestial_Core):

	def __init__(self, x, y, mass, radius, color):
		super().__init__(x, y, mass)
		self.radius = radius
		self.color = color

	def draw(self, win, shift_x , shift_y, Scale):

		x = (self.x+shift_x) * Scale + WIDTH / 2 
		y = (self.y+shift_y) * Scale + HEIGHT / 2

		#x = (self.x) * Scale + WIDTH / 2  + shift_x
		#y = (self.y) * Scale + HEIGHT / 2 + shift_y

		if len(self.orbit) > 8:
			for point in self.orbit:
				x, y = point
				x = (x+ shift_x) * Scale + WIDTH / 2 
				y = (y+ shift_y) * Scale + HEIGHT / 2 
				
			pygame.draw.lines(win, self.color, False, self.orbit, 2)
		if(x>0):
			pygame.draw.circle(win, self.color, (x, y), self.radius)
		


class SpaceShip(Celestial_Core) :
		
	def __init__(self, x, y, mass, radius, color):
		super().__init__(x, y, mass)
		self.radius = radius
		self.color = color
	
		self.reactor1 = False
		self.reactor2 = False
		self.reactor3 = False

	# Destructor :
	def __del__(self):
		print("Destructor called")


	def ChangeR(self,s_reactor):
		if(s_reactor == 1):
			self.reactor1 = not self.reactor1
		if(s_reactor == 2):
			self.reactor2 = not self.reactor2
		if(s_reactor == 3):
			self.reactor3 = not self.reactor3

	def accelerator(self):
		return ACCELERATING_RATE
		
	def draw(self, win, shift_x , shift_y, Scale):

		x = (self.x+shift_x) * Scale + WIDTH / 2 
		y = (self.y+shift_y) * Scale + HEIGHT / 2


		if len(self.orbit) > 2:
			updated_points = []

			for point in self.orbit:
				x, y = point
				x = (x+ shift_x) * Scale + WIDTH / 2 
				y = (y+ shift_y) * Scale + HEIGHT / 2 
				updated_points.append((x, y))

			pygame.draw.lines(win, WHITE, False, updated_points, 1)
			
		if(x>0):
			pygame.draw.circle(win, self.color, (x, y), self.radius)
			if(self.reactor1):
				pygame.draw.circle(win, YELLOW, (x+12,y),3)
			if(self.reactor2):
				comp_x = 12 * math.cos(2.*math.pi/3.)
				comp_y = 12 * math.sin(2.*math.pi/3.)
				pygame.draw.circle(win, YELLOW, (x+comp_x,y+comp_y),3)
			if(self.reactor3):
				comp_x = 12 * math.cos(-2.*math.pi/3.)
				comp_y = 12 * math.sin(-2.*math.pi/3.)
				pygame.draw.circle(win, YELLOW, (x+comp_x,y+comp_y),3)