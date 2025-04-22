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
    rockets.append(Launch(planets[3]))
    while run:
        
        clock.tick(FPS)
        WIN.fill((0, 0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            #Actions done once per pressure
            if event.type == pygame.KEYDOWN:
                if(event.key == pygame.K_1):
                    Step_p_frame += 10                          # Speed up time fast >>
                if(event.key == pygame.K_2):
                    Step_p_frame += 1                           # Speed up time slow >
                if(event.key == pygame.K_3):
                    Step_p_frame -= 1                           # Slow down time slow <
                if(event.key == pygame.K_4):
                    Step_p_frame -= 10                          # Slow down time fast <<
                if(event.key == pygame.K_a):
                    Flag -=1                                    # Show previous object
                if(event.key == pygame.K_z):
                    Flag +=1                                    # Show next object
                if(event.key == pygame.K_t):
                    Flag = Target                               # Show target object
                if(event.key == pygame.K_r):
                    Target = random.randint(0,len(planets)-1)   # Set a new target
                
                if(event.key == pygame.K_SPACE):
                    rockets.append(Launch(planets[3]))
                
                if(event.key == pygame.K_BACKSPACE):
                    if(len(rockets)> 0):
                        rockets.pop()
                if(event.key == pygame.K_RIGHT):
                    if(Flag>=len(planets)):
                        rockets[Flag-len(planets)].ChangeR(1)
                if(event.key == pygame.K_LEFT):
                    if(Flag>=len(planets)):
                        rockets[Flag-len(planets)].ChangeR(2)
                if(event.key == pygame.K_UP):
                    if(Flag>=len(planets)):
                        rockets[Flag-len(planets)].ChangeR(3)
                if(event.key == pygame.K_q): run = False



        if(Flag>(len(planets+rockets)-1)):
             Flag = 0
        if(Flag<0):
            Flag = len(planets+rockets)-1
        # Actions repeated until release
        keys = pygame.key.get_pressed()
        if keys[pygame.K_o]:                               # Zoom in
            Scale *= 1.05
        if keys[pygame.K_p]:                             # Zoom out
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

        # Add a pannel to show commands when pressing 'h':
        if keys[pygame.K_h]:
            help_lines = [
                "--- Help ---",
                "1: Speed up time (fast)",
                "2: Speed up time (slow)",
                "3: Slow down time (slow)",
                "4: Slow down time (fast)",
                "a: Focus previous object",
                "z: Focus next object",
                "t: Focus target planet",
                "r: Set random target planet",
                "space: Launch new rocket (from Earth)",
                "backspace: Remove last rocket",
                "o: Zoom in",
                "p: Zoom out",
                "left/right/up: Activate rocket reactors (when focusing rocket)"
            ]
            
            line_height = FONT.get_height()
            start_y = HEIGHT - (len(help_lines) * line_height) - 30 # Start Y position near bottom

            for i, line in enumerate(help_lines):
                line_surface = FONT.render(line, 1, WHITE)
                # Position each line below the previous one
                WIN.blit(line_surface, (10, start_y + i * line_height))


        #Show fps :
        fps = clock.get_fps()
        fps_text = FONT.render(f"FPS: {int(fps)}", 1, WHITE)
        WIN.blit(fps_text, (WIDTH - fps_text.get_width() - 10, 10))

        speed_text = FONT.render(f"Step per Frame: {int(Step_p_frame)}", 1, WHITE)
        WIN.blit(speed_text, (WIDTH - speed_text.get_width() - 10, 25))

        follow_core = Core_position[int(Flag)] if int(Flag)<len(planets) else f"Rocket {int(Flag-len(planets))}"

        Flag_text = FONT.render(f"Followed Planet: {follow_core}", 1, WHITE)
        WIN.blit(Flag_text, (10, 10))
        Target_text = FONT.render(f"Target Planet: {Core_position[int(Target)]}", 1, WHITE)
        WIN.blit(Target_text, (10, 25))
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
        # in the bottom right corner : show press h to show help
        help_text = FONT.render("Press h to show help-commands", 1, WHITE)
        WIN.blit(help_text, (10, HEIGHT - help_text.get_height() - 10))
        pygame.display.update()

    pygame.quit()


SatSim()