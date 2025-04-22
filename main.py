import pygame
import math
import random
import csv

import torch 

from collections import deque

from Agent import *
from Class_Def import *
from utils import *
from Const import *

device = DEVICE

pygame.init()

WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Planet_Sim")
FONT = pygame.font.SysFont("Times", 16)

def SatSim(load_weights=False):
    print(device)
    Scale = 50/AU
    Step_p_frame = 1
    shift_x, shift_y = 0, 0
    run = True
    clock = pygame.time.Clock()

    planets = Solar_System()
    Target = 6 #random.randint(0,len(planets)-1)
    rockets = []
    rockets.append(Launch(planets[3]))


    # --- Initialize Distance Recording ---
    all_distance_acquisitions = []
    current_acquisition = []
    
    # --- Instantiate the Agent ---
    agent = Att_Agent(n_features=NUM_FEATURES, # Use n_features
                  seq_len=SEQ_LENGTH,     # Use seq_len
                  action_size=ACTION_SIZE,
                  replay_memory_capacity=MEMORY_CAPACITY,
                  batch_size=BATCH_SIZE,
                  gamma=GAMMA,
                  eps_start=EPS_START, eps_end=EPS_END, eps_decay=EPS_DECAY,
                  tau=TAU, lr=LR, target_update_freq=TARGET_UPDATE_FREQ,
                  # Add Attention Network specific params if needed (using defaults here)
                  # embed_dim=64, num_heads=4
                  )
   
    # Load weights if demanded
    try:
        agent.load_weights()
    except Exception as e:
        print(f"Error loading weights: {e}")
    # --- Setup Agent Logging ---
    # The agent will create its own timestamped directory within Metrics/MODEL_NAME/
    agent.setup_logging(model_name=MODEL_NAME) # Use the defined MODEL_NAME

    # --- Initialize Episode Tracking Variables ---
    episode_number = 0
    episode_reward = 0.0
    episode_steps = 0
    episode_goal_achieved = False

    # History deques initialization...
    action_hist = deque(maxlen=MAX_HISTORY_LEN)

    Flag = 0
    agent_controlled_rocket_index = 0 
    rocket = rockets[agent_controlled_rocket_index] if rockets else None
    target_planet = planets[Target] if Target < len(planets) else None


    # --- Initialize History Deques ---
    action_hist = deque(maxlen=MAX_HISTORY_LEN)
    rel_x_hist = deque(maxlen=MAX_HISTORY_LEN)
    rel_y_hist = deque(maxlen=MAX_HISTORY_LEN)
    rel_vx_hist = deque(maxlen=MAX_HISTORY_LEN)
    rel_vy_hist = deque(maxlen=MAX_HISTORY_LEN)

    # Fill initial history with padding values (optional, but helps)
    # Use padding values defined in utils or Agent
    initial_pad_value = 0.0
    initial_action_pad = 0
    for _ in range(SEQ_LENGTH): # Start with enough history for the first state
        action_hist.append(initial_action_pad)
        rel_x_hist.append(initial_pad_value)
        rel_y_hist.append(initial_pad_value)
        rel_vx_hist.append(initial_pad_value)
        rel_vy_hist.append(initial_pad_value)

    # Initialize state tuple for the loop start
    state_tuple = None

    # --- Steps count ---
    n_step = 0 
    while run:
            # --- Initialize state ---
        # For simplicity, let's always control the first rocket if it exists
        done = False # Reset done flag each iteration

        agent_controlled_rocket_index = 0 
        rocket = rockets[agent_controlled_rocket_index] if rockets else None
        target_planet = planets[Target]
    
        clock.tick(FPS)
        WIN.fill((0, 0, 0))


        # --- Pygame Event Handling ---
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            #Actions done once per pressure
            if event.type == pygame.KEYDOWN:
                if(event.key == pygame.K_1): Step_p_frame += 10                          # Speed up time fast >>
                if(event.key == pygame.K_2): Step_p_frame += 1                           # Speed up time slow >
                if(event.key == pygame.K_3): Step_p_frame -= 1                           # Slow down time slow <
                if(event.key == pygame.K_4): Step_p_frame -= 10                          # Slow down time fast <<
                if(event.key == pygame.K_a): Flag -=1                                    # Show previous object
                if(event.key == pygame.K_z): Flag +=1                                    # Show next object
                if(event.key == pygame.K_t): Flag = Target                               # Show target object
                if(event.key == pygame.K_r): Target = random.randint(0,len(planets)-1)   # Set a new target
                if(event.key == pygame.K_SPACE):
                    rockets.append(Launch(planets[3]))
                
                if(event.key == pygame.K_BACKSPACE):
                    if(len(rockets)> 0):
                        rockets.pop()
                if event.key == pygame.K_x: 
                    if rocket:
                        done = True

                        # rocket = reset_rocket_state(rocket, planets) Now reset is in the done condition True
                        # --- Clear History Deques on Reset ---
                        action_hist.clear()
                        rel_x_hist.clear()
                        rel_y_hist.clear()
                        rel_vx_hist.clear()
                        rel_vy_hist.clear()
                        # Fill with padding again? Or let the loop handle it?
                        # Re-filling ensures immediate valid sequence length.
                        for _ in range(SEQ_LENGTH):
                            action_hist.append(initial_action_pad)
                            rel_x_hist.append(initial_pad_value)
                            rel_y_hist.append(initial_pad_value)
                            rel_vx_hist.append(initial_pad_value)
                            rel_vy_hist.append(initial_pad_value)
                        # state = get_state(rocket, target_planet) # Remove old state logic
                        state_tuple = None # Indicate state needs recalculation
                        # --- Log previous episode before reset (if it ran at least one step) ---
                        # if episode_steps > 0:
                        #     agent.log_episode_data(episode_number, episode_steps, episode_reward, episode_goal_achieved)
                        # --- Reset Episode Trackers ---
                        episode_reward = 0.0
                        episode_steps = 0
                        episode_goal_achieved = False
                        print("Targeted planet is", Core_position[int(Target)])
                        # Save distance track and reset
                        # all_distance_acquisitions.append(list(current_acquisition)) # Use list() to copy
                        # current_acquisition.clear() # Use clear() for deque/list

                if(event.key == pygame.K_q): 
                    run = False
                    all_distance_acquisitions.append(current_acquisition) # Save the track of distances bwt Rckt & Tgt
                    current_acquisition = []

        if(Flag>(len(planets)+len(rockets)-1)): Flag = 0
        if(Flag<0): Flag = len(planets)+len(rockets)-1
        
        # --- RL Agent Step ---
        reward = 0 # Initialize reward

        if rocket and target_planet: # Ensure we have a rocket and target
            # --- 1. Calculate Current Relative State ---
            # This provides the LATEST snapshot before action selection
            # Uses the state *before* physics update for this step
            current_rel_x, current_rel_y, current_rel_vx, current_rel_vy = calculate_current_relative_state(rocket, target_planet)

            # Handle potential NaN from calculate_current_relative_state if rocket/target missing
            if any(math.isnan(v) for v in [current_rel_x, current_rel_y, current_rel_vx, current_rel_vy]):
                 print("Warning: Invalid current state values (NaN). Skipping agent step.")
                 # Potentially try to recover or skip agent logic for this frame
                 # You might need error handling here depending on how NaNs occur
            else:
                # --- 2. Get State Sequence for Agent ---
                # This uses the history *leading up to* the current point
                state_tuple = get_state_sequence(
                    action_hist, rel_x_hist, rel_y_hist, rel_vx_hist, rel_vy_hist, seq_length=SEQ_LENGTH
                )

                # --- 3. Agent Selects Action ---
                action_tensor = agent.select_action(state_tuple) # Pass tuple (sequence, mask)
                action = action_tensor.item()

                # --- 4. Apply Action to Rocket ---
                rocket.apply_action(action)

                # --- Store state before physics update (for reward calc based on distance change) ---
                prev_dist = get_distance(rocket, target_planet)
                if prev_dist == 0: prev_dist = 1e-6 # Avoid division by zero

                # --- 5. Physics Update ---
                planets, rockets = update_position(planets, rockets) # This updates rocket.x, .y, .vx, .vy

                # Find the controlled rocket again after update (it might have been removed)
                # This assumes the agent always controls the rocket at index 0 if it exists
                controlled_rocket_index = 0
                if controlled_rocket_index < len(rockets):
                    rocket = rockets[controlled_rocket_index] # Update rocket reference
                else:
                    rocket = None # Rocket was destroyed/removed
                    print("Controlled rocket destroyed during physics update.")
                    done = True
                    reward = -GOAL_REWARD * 2 # Heavy penalty for destruction
                    next_state_tuple = None # No next state if destroyed


                 # --- 6. Calculate Reward & Next State (if rocket exists) ---
                if rocket:
                    # --- Increment episode step counter ---
                    episode_steps += 1
                    try:
                        current_distance = get_distance(rocket, target_planet)
                        current_acquisition.append(current_distance) # Save distance for logging

                        if current_distance == 0: current_distance = 1e-6

                        # --- 6a. Calculate Reward Components ---
                        # Distance Reward :
                        reward_distance = float(AU/abs(current_distance - 10*target_planet.radius)) 
                        if current_distance>50*target_planet.radius: 
                            reward_distance *= np.sign(prev_dist - current_distance)


                        # Orbital speed reward & goal check
                        target_mass = target_planet.mass
                        ideal_orbital_speed = math.sqrt(G * target_mass / max(current_distance, 1e-6)) # Avoid div by zero
                        rel_vx_now = rocket.vx - target_planet.vx # Use updated velocities
                        rel_vy_now = rocket.vy - target_planet.vy
                        current_speed_relative = math.sqrt(rel_vx_now**2 + rel_vy_now**2)
                        speed_diff = abs((current_speed_relative - ideal_orbital_speed) / current_speed_relative)
                        reward_speed = REWARD_SPEED_SCALE * (1.0 - speed_diff)

                        reward_action = 0
                        if current_distance<20*target_planet.radius:
                            reward_action = NO_THRUST_REWARD if action == 0 else 0 # Penalize any thrust

                        # Time Penalty
                        reward_time = 0 #-TIME_PENALTY (Far too soon) Probably useless w/ fuel consuption notion.

                        # Goal Reward Check (use updated distance/speed)
                        is_orbiting_goal = (speed_diff * ideal_orbital_speed < ORBIT_SPEED_TOLERANCE) and \
                                        (5*target_planet.radius <= current_distance <= 20*target_planet.radius)
                    #  print(current_distance, target_planet.radius)
                        reactors_off = rocket.motor_off()
                        
                        if is_orbiting_goal and reactors_off:
                             episode_goal_achieved = True # Mark true if goal met anytime in episode
                             reward_goal = GOAL_REWARD
                        else:
                            reward_goal = 0

                        reward = reward_distance + reward_speed + reward_action + reward_time + reward_goal
                        step_reward = reward

                        # --- Accumulate Episode Reward ---
                        episode_reward += step_reward

                        # --- 6b. Check "Done" Conditions ---
                        # Crash into target planet
                        crash_dist_m = (target_planet.radius) #+ CRASH_DISTANCE_THRESHOLD
                        if current_distance < crash_dist_m :
                            print("Crashed into target! Resetting rocket.")
                            print(current_distance, target_planet.radius)
                            done = True
                            reward -= GOAL_REWARD * 0.5 # Penalty for crash
                            # Resetting logic now handles history clearing (see point 5)


                        # Out of bounds (relative to target planet)
                        if current_distance > OUT_OF_BOUNDS_DISTANCE:
                            print("Went out of bounds! Resetting rocket.")
                            done = True
                            reward -= GOAL_REWARD # Penalty for OOB (adjust as needed)
                            # Resetting logic handles history clearing


                        # --- 6c. Update History & Get Next State ---
                        # Calculate the relative state AFTER the physics update
                        next_rel_x, next_rel_y, next_rel_vx, next_rel_vy = calculate_current_relative_state(rocket, target_planet)

                        # Append the state values LEADING TO the action, and the action itself, to history
                        # Use the values calculated in step 1
                        if not math.isnan(current_rel_x): # Check if state was valid before appending
                            action_hist.append(action)
                            rel_x_hist.append(current_rel_x)
                            rel_y_hist.append(current_rel_y)
                            rel_vx_hist.append(current_rel_vx)
                            rel_vy_hist.append(current_rel_vy)

                        # Now generate the next_state sequence using the updated history
                        if not done and not any(math.isnan(v) for v in [next_rel_x, next_rel_y, next_rel_vx, next_rel_vy]):
                            next_state_tuple = get_state_sequence(
                                action_hist, rel_x_hist, rel_y_hist, rel_vx_hist, rel_vy_hist, seq_length=SEQ_LENGTH
                            )
                        else:
                            next_state_tuple = None # Terminal state or invalid next state


                    except (NameError, AttributeError, IndexError, TypeError, ValueError) as e:
                         print(f"Warning: Error during RL step calculation: {e}")
                         import traceback
                         traceback.print_exc() # Print full traceback for debugging
                         reward = 0 # Default reward on error
                         next_state_tuple = None # Cannot determine next state
                         # Decide if this error should terminate the episode (done=True)
                         # done = True # Optional: Terminate on calculation error


                 # --- 7. Store Experience ---
                 # Ensure state_tuple was calculated in this iteration before storing
                if state_tuple is not None:
                    reward_tensor = torch.tensor([reward], dtype=torch.float32, device=device)
                    # Store the state tuple (history before action) and next_state tuple (history after action)
                    agent.store_experience(state_tuple, action_tensor, next_state_tuple, reward_tensor, done)
                else:
                    # This case might happen on the very first step if state isn't pre-initialized
                    # or after an error/reset where state_tuple became None.
                    # Avoid storing experience if the initial state wasn't valid.
                    print("Warning: state_tuple is None, skipping experience storage.") # Optional log
                    pass


                # --- 8. Optimize Agent Model ---
                agent.optimize_model()

                # --- 9. Update Target Network ---
                if agent.tau > 0:
                    agent.update_target_net(soft_update=True)
                # --- Or Hard update less frequently ---
                # elif step_count % agent.target_update_freq == 0:
                #     agent.update_target_net(soft_update=False)


                # --- Handle End of Episode ---
                if done:
                    print(f"Episode finished. Reason: {'Crash' if current_distance < crash_dist_m else 'OOB' if current_distance > OUT_OF_BOUNDS_DISTANCE else 'User Reset'}. Final Reward: {reward}")
                    # Clear History Deques on Reset
                    action_hist.clear(); rel_x_hist.clear(); rel_y_hist.clear(); rel_vx_hist.clear(); rel_vy_hist.clear()
                    # Re-fill padding
                    for _ in range(SEQ_LENGTH):
                        action_hist.append(initial_action_pad)
                        rel_x_hist.append(initial_pad_value)
                        rel_y_hist.append(initial_pad_value)
                        rel_vx_hist.append(initial_pad_value)
                        rel_vy_hist.append(initial_pad_value)
                    # Save distance track and reset
                    all_distance_acquisitions.append(list(current_acquisition))
                    current_acquisition.clear()
                    episode_number += 1
                    episode_reward /= episode_steps # Average reward per step
                    print(f"--- Episode {episode_number} Finished --- Steps: {episode_steps}, Average Reward: {episode_reward:.2f}, Goal: {episode_goal_achieved} ---")
                    # Log data for the completed episode
                    agent.log_episode_data(episode_number, episode_steps, episode_reward, episode_goal_achieved)

                    # Reset rocket state
                    rocket = reset_rocket_state(rocket, planets)
                    # Clear and refill history deques
                    action_hist.clear(); rel_x_hist.clear(); rel_y_hist.clear(); rel_vx_hist.clear(); rel_vy_hist.clear()
                    for _ in range(SEQ_LENGTH):
                        action_hist.append(initial_action_pad); rel_x_hist.append(initial_pad_value); rel_y_hist.append(initial_pad_value); rel_vx_hist.append(initial_pad_value); rel_vy_hist.append(initial_pad_value)

                    # Reset episode trackers
                    episode_reward = 0.0
                    episode_steps = 0
                    episode_goal_achieved = False
                    state_tuple = None # Invalidate state tuple until next loop iteration
                # --- End Reward Calculation ---




        # Add a pannel to show commands when pressing 'h':
        keys = pygame.key.get_pressed()
        if keys[pygame.K_o]: Scale *= 1.05      # Zoom in
        if keys[pygame.K_p]: Scale /= 1.05       # Zoom out
                
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


 

        speed_text = FONT.render(f"Step per Frame: {int(Step_p_frame)}", 1, WHITE)
        WIN.blit(speed_text, (WIDTH - speed_text.get_width() - 10, 25))

        follow_core = Core_position[int(Flag)] if int(Flag)<len(planets) else f"Rocket {int(Flag-len(planets))}"

        Flag_text = FONT.render(f"Followed Planet: {follow_core}", 1, WHITE)
        WIN.blit(Flag_text, (10, 10))

        Target_text = FONT.render(f"Target Planet: {Core_position[int(Target)]}", 1, WHITE)
        WIN.blit(Target_text, (10, 25))

        reward_text_line1 = f"Total Reward: {round(reward, 3)}"
        reward_text_line2 = f" D: {round(reward_distance, 3)}, S: {round(reward_speed, 3)}, A: {round(reward_action, 3)}, T: {round(reward_time, 3)}, G: {round(reward_goal, 3)}"
        reward_text1 = FONT.render(reward_text_line1, 1, WHITE)
        reward_text2 = FONT.render(reward_text_line2, 1, WHITE)
        WIN.blit(reward_text1, (10, 40))
        WIN.blit(reward_text2, (10, 55)) # Position below the first line

        Dist_text = FONT.render(f"Rocket-Tgt Distancce:{round(get_distance(planets[Target],rockets[0])/AU,5)}", 1, WHITE)
        WIN.blit(Dist_text, (WIDTH-Dist_text.get_width()-10, HEIGHT-Dist_text.get_height()-25))        
        
        if(Flag<len(planets)):
            Orbit_text = FONT.render(f"Is Orbiting:{is_orbiting(planets[3],planets[Flag])}", 1, WHITE)
        else:
            Orbit_text = FONT.render(f"Is Orbiting:{is_orbiting(rockets[Flag-len(planets)],planets[Target])}", 1, WHITE)
        WIN.blit(Orbit_text, (WIDTH-Orbit_text.get_width()-10, HEIGHT-Dist_text.get_height()-10))
        # in the bottom right corner : show press h to show help
        help_text = FONT.render("Press h to show help-commands", 1, WHITE)
        WIN.blit(help_text, (10, HEIGHT - help_text.get_height() - 10))
        # --- Show fps ---
        fps = clock.get_fps()
        fps_text = FONT.render(f"FPS: {int(fps)}", 1, WHITE)
        WIN.blit(fps_text, (WIDTH - fps_text.get_width() - 10, HEIGHT-Dist_text.get_height()-40))
        
        # --- Update Display ---
        if n_step%Step_p_frame == 0:
            # --- Draw Planets and Rockets ---
            if Flag<len(planets):
                shift_x = - planets[Flag].x
                shift_y = - planets[Flag].y
            else:
                shift_x = - rockets[Flag-len(planets)].x
                shift_y = - rockets[Flag-len(planets)].y
            for core in planets+rockets:
                core.draw(WIN, shift_x, shift_y, Scale)
            pygame.display.update()
        n_step += 1
    pygame.quit()
    print("Simulation ended.")
    # Optional: Save the trained model
    agent.save_weights_and_distances(all_distance_acquisitions)
    # Close the log file
    agent.close_log()


if __name__ == '__main__':


    SatSim(load_weights=True)