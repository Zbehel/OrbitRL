import pygame
import math
import random

from Class_Def import *
from utils import *
from Const import *

pygame.init()

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Planet_Sim")
FONT = pygame.font.SysFont("Times", 16)


def SatSim():
    Scale = 50/AU
    Step_p_frame = 1
    Flag = 0
    shift_x = shift_y = 0
    run = True
    clock = pygame.time.Clock()


    planets = Solar_System()
    #random int between 0 and len(planets):
    Target = random.randint(0,len(planets)-1)
    rockets = []
    rockets.append(Launch(2*math.pi*random.random(), 24*random.random(),planets[3]))
    while run:
        
        clock.tick(FPS)
        WIN.fill((0, 0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            #Actions done once per pressure
            if event.type == pygame.KEYDOWN:
                if(event.key == pygame.K_1):
                    Step_p_frame += 10
                if(event.key == pygame.K_2):
                    Step_p_frame += 1
                if(event.key == pygame.K_3):
                    Step_p_frame -= 1
                if(event.key == pygame.K_4):
                    Step_p_frame -= 10
                if(event.key == pygame.K_a):
                    Flag -=1
                if(event.key == pygame.K_z):
                    Flag +=1
                if(event.key == pygame.K_t):
                    Flag = Target
                if(event.key == pygame.K_r):
                    Target = random.randint(0,len(planets)-1)
                
                # if(event.key == pygame.K_SPACE):
                #     rockets.append(Launch(2*math.pi*random.random(), 24*random.random(),planets[3]))
                
                # if(event.key == pygame.K_BACKSPACE):
                #     if(len(rockets)> 0):
                #         rockets.pop()
                if(event.key == pygame.K_RIGHT):
                    if(Flag>=len(planets)):
                        rockets[Flag-len(planets)].ChangeR(1)
                if(event.key == pygame.K_LEFT):
                    if(Flag>=len(planets)):
                        rockets[Flag-len(planets)].ChangeR(2)
                if(event.key == pygame.K_UP):
                    if(Flag>=len(planets)):
                        rockets[Flag-len(planets)].ChangeR(3)


        if(Flag>(len(planets+rockets)-1)):
             Flag = 0
        if(Flag<0):
            Flag = len(planets+rockets)-1
        # Actions repeated until release
        keys = pygame.key.get_pressed()  #checking pressed keys
        if keys[pygame.K_PAGEUP]:
            Scale *= 1.05
        if keys[pygame.K_PAGEDOWN]:
            Scale /=1.05

            
        for i in range(Step_p_frame):
            planets,rockets = update_position(planets,rockets)
        if(Flag<len(planets)):
            shift_x = - planets[Flag].x
            shift_y = - planets[Flag].y
        else:
            shift_x = - rockets[Flag-len(planets)].x
            shift_y = - rockets[Flag-len(planets)].y
        for planet in planets+rockets:
            planet.draw(WIN, shift_x, shift_y, Scale)

            
        #Show fps :
        fps = clock.get_fps()
        fps_text = FONT.render(f"FPS: {int(fps)}", 1, WHITE)
        WIN.blit(fps_text, (WIDTH - fps_text.get_width() - 10, 10))

        speed_text = FONT.render(f"Step per Frame: {int(Step_p_frame)}", 1, WHITE)
        WIN.blit(speed_text, (WIDTH - speed_text.get_width() - 10, 25))

        Flag_text = FONT.render(f"Followed Planet: {int(Flag)}", 1, WHITE)
        WIN.blit(Flag_text, (10, 10))

        if(Flag<len(planets)):
            Dist_text = FONT.render(f"Obj-Sun Distancce:{round(get_distance(planets[0],planets[Flag])/AU,5)}", 1, WHITE)
        else:
            Dist_text = FONT.render(f"Obj-Sun Distancce:{round(get_distance(planets[0],rockets[Flag-len(planets)])/AU,5)}", 1, WHITE)
        WIN.blit(Dist_text, (WIDTH-Dist_text.get_width()-10, HEIGHT-Dist_text.get_height()-25))
        
        
        if(Flag<len(planets)):
            Orbit_text = FONT.render(f"Is Orbiting:{is_orbiting(planets[3],planets[Flag])}", 1, WHITE)
        else:
            Orbit_text = FONT.render(f"Is Orbiting:{is_orbiting(rockets[Flag-len(planets)],planets[Target])}", 1, WHITE)
        WIN.blit(Orbit_text, (WIDTH-Orbit_text.get_width()-10, HEIGHT-Dist_text.get_height()-10))
        pygame.display.update()

    pygame.quit()


SatSim()