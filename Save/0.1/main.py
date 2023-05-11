import pygame
import math
import random

from Class_Def import *
from utils import *

pygame.init()

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Planet_Sim")
FONT = pygame.font.SysFont("Times", 16)



def main():

    random.random()
    # random.randint()
    Step_p_frame = 1
    Flag = 0
    shift_x = shift_y = 0
    Scale = 250 / AU / 5
    run = True
    clock = pygame.time.Clock()

    sun = Planet(0, 0, 1.98892 * 10**30, 30, YELLOW)

    earth = Planet(-1 * AU , 0, 5.9742 * 10**24, 16, BLUE)
    earth.y_vel = 29.783 * 1000 

    moon = Planet((-1 - 0.00256954)*AU , 0, 7.347 * 10**22, 10, WHITE)
    moon.y_vel = (29.783+1.022) * 1000 
    
    mars = Planet(-1.524 * AU, 0, 6.39* 10**23 , 12, RED)
    mars.y_vel = 24.077 * 1000

    mercury = Planet(0.387 * AU, 0, 3.30 * 10**23, 8, DARK_GREY)
    mercury.y_vel = -47.4 * 1000

    venus = Planet(0.723 * AU, 0, 4.8685 * 10**24, 14, WHITE)
    venus.y_vel = -35.02 * 1000
    
    jupiter = Planet(5.19 * AU, 0, 1.898* 10**27, 25, JUP)
    jupiter.y_vel = -13.06 * 1000
    
    saturn = Planet(9.536 * AU, 0, 5.684 * 10**26, 20, GREEN)
    saturn.y_vel = -9.64 * 1000

    neptune = Planet(30.07 * AU, 0, 1.024 * 10**26, 18, BLUE)
    neptune.y_vel = -5.43 * 1000

    rocket = SpaceShip(5*AU, 0, 3_000, 12, RED)
    rocket.y_vel = 15.02 * 1000

    planets = [sun,mercury,venus,earth,moon,mars,jupiter,saturn,neptune]
    rockets = []
    while run:
        
        clock.tick(60)
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
                if(event.key == pygame.K_SPACE):
                    # rockets.append( SpaceShip(0., 0, 3_000, 12, RED))                  
                    # rockets[-1].y_vel = 15.02 * 1000
                    rockets.append(Launch(2*math.pi*random.random(), 24*random.random(),earth))
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
            Dist_text = FONT.render(f"Obj-Sun Distancce:{round(get_distance(sun,planets[Flag])/AU,5)}", 1, WHITE)
        else:
            Dist_text = FONT.render(f"Obj-Sun Distancce:{round(get_distance(sun,rockets[Flag-len(planets)])/AU,5)}", 1, WHITE)
        WIN.blit(Dist_text, (WIDTH-Dist_text.get_width()-10, HEIGHT-Dist_text.get_height()-25))
        
        
        if(Flag<len(planets)):
            Orbit_text = FONT.render(f"Is Orbiting:{is_orbiting(earth,planets[Flag])}", 1, WHITE)
        else:
            Orbit_text = FONT.render(f"Is Orbiting:{is_orbiting(rockets[Flag-len(planets)],planets[0])}", 1, WHITE)
        WIN.blit(Orbit_text, (WIDTH-Orbit_text.get_width()-10, HEIGHT-Dist_text.get_height()-10))
        pygame.display.update()

    pygame.quit()


main()




# # Initialize agent
# model = PPO1(CustomCnnPolicy, vec_env, verbose=0)

# # Train agent
# model.learn(total_timesteps=80000)

# # Plot cumulative reward
# with open(os.path.join(log_dir, "monitor.csv"), 'rt') as fh:    
#     firstline = fh.readline()
#     assert firstline[0] == '#'
#     df = pd.read_csv(fh, index_col=None)['r']
# df.rolling(window=1000).mean().plot()
# plt.show()