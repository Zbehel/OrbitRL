import pandas as pd
import pygame as pg
import math

import os

import torch 

from collections import deque

from Const import*
from Class_Def import Celestial_Core, Planet, SpaceShip
device = DEVICE

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
			# print(i.vx, i.vy)
			if i.reactor1 :
				total_fx[n-1] += - i.accelerator()
			if i.reactor2 :
				total_fx[n-1] += - i.accelerator() * math.cos(2.*math.pi/3.)
				total_fy[n-1] += - i.accelerator() * math.sin(2.*math.pi/3.)
			if i.reactor3 :
				total_fx[n-1] += - i.accelerator() * math.cos(- 2.*math.pi/3.)
				total_fy[n-1] += - i.accelerator() * math.sin(- 2.*math.pi/3.)
				
	for n,i in enumerate(planets+rockets):
		i.vx += total_fx[n-1] / i.mass * TIMESTEP
		i.vy += total_fy[n-1] / i.mass * TIMESTEP

		i.x += i.vx * TIMESTEP
		i.y += i.vy * TIMESTEP
		i.orbit.append((i.x, i.y))
		while(len(i.orbit)>600):
			i.orbit.pop(0)
		
	for i in rockets:
		for j in planets:
			if (i.x - j.x)**2 + (i.y - j.y)**2 < (i.radius + j.radius)**2:
				del i
				# rockets.remove(i)
	return planets,rockets

def Solar_System():
	sun = Planet(0, 0, 1.98892 * 10**30, 695_700e3, 30, YELLOW)
	
	earth = Planet(-1 * AU , 0, 5.9742 * 10**24, 6_371e3, 16, BLUE)
	earth.vy = 29.783 * 1000 

	moon = Planet((-1 - 0.00256954)*AU , 0, 7.347 * 10**22, 1_737e3, 10, WHITE)
	moon.vy = (29.783+1.022) * 1000 
    
	mars = Planet(-1.524 * AU, 0, 6.39* 10**23 , 3_390e3, 12, RED)
	mars.vy = 24.077 * 1000

	mercury = Planet(0.387 * AU, 0, 3.30 * 10**23, 2_440e3, 8, DARK_GREY)
	mercury.vy = -47.4 * 1000

	venus = Planet(0.723 * AU, 0, 4.8685 * 10**24, 6_052e3, 14, WHITE)
	venus.vy = -35.02 * 1000
    
	jupiter = Planet(5.19 * AU, 0, 1.898* 10**27, 69_911e3, 25, JUP)
	jupiter.vy = -13.06 * 1000
    
	saturn = Planet(9.536 * AU, 0, 5.684 * 10**26, 58_232e3, 20, GREEN)
	saturn.vy = -9.64 * 1000

	neptune = Planet(30.07 * AU, 0, 1.024 * 10**26, 24_622e3, 18, BLUE)
	neptune.vy = -5.43 * 1000

	return  [sun,mercury,venus,earth,moon,mars,jupiter,saturn,neptune]

def Launch(planet):
	
	rocket = SpaceShip(planet.x , planet.y  + 0.00256954*AU, 30_000, 12, RED)
	rocket.vx = planet.vx -1.022*1000
	rocket.vy = planet.vy 
	
	return rocket



def get_distance(core1, core2):
	return math.sqrt( (core1.x-core2.x)**2 + (core1.y-core2.y)**2)

def draw_dist(win, shift_x , shift_y, core1, core2, Scale):

	x1 = (core1.x+shift_x) * Scale + WIDTH / 2 
	y1 = (core1.y+shift_y) * Scale + HEIGHT / 2

	x2 = (core2.x+shift_x) * Scale + WIDTH / 2 
	y2 = (core2.y+shift_y) * Scale + HEIGHT / 2
		
	pg.draw.line(win, WHITE, [x2, y2],[x1, y1], 3)

		

def is_orbiting(core1, core2, start=0):
	hist_dist = []
	#if mae between core1 and core 2 get_distance < 1% return True :
	
	for i in range(start,len(core1.orbit)):
		hist_dist.append(math.sqrt( (core1.orbit[i][0] - core2.orbit[i][0])**2 + (core1.orbit[i][1]-core2.orbit[i][1])**2))

	if (max(hist_dist) - min(hist_dist)) < 0.01 * min(hist_dist):
		return True
	else:
		return False

def magnitude(vx, vy):
	return math.sqrt(vx**2 + vy**2)



# --- Helper function to get the current state ---
def get_state(rocket, target_planet):
    """Calculates the state vector for the agent."""
    if rocket is None or target_planet is None:
        # Return a zero state if objects are missing
        return torch.zeros((1, STATE_SIZE), dtype=torch.float32, device=device)
    # Relative position (normalize by AU for consistency)
    rel_x = (rocket.x - target_planet.x) / AU
    rel_y = (rocket.y - target_planet.y) / AU
    # Relative velocity (needs normalization - find typical max velocity?)
    # Let's estimate a max velocity (e.g., around Earth orbital speed ~30km/s = 3e4 m/s)
    # Or maybe relative to ideal orbital speed? This needs tuning.
    norm_factor_vel = 5e4 # Tune this normalization factor
    rel_vx = (rocket.vx - target_planet.vx) / norm_factor_vel
    rel_vy = (rocket.vy - target_planet.vy) / norm_factor_vel
    
    # State vector: [dx, dy, dvx, dvy]
    state_list = [rel_x, rel_y, rel_vx, rel_vy] 
    
    # Add other components if STATE_SIZE in Const.py is > 4
    # e.g., state_list.append(rocket.fuel / rocket.initial_fuel) # If fuel becomes relevant

    # Convert to PyTorch tensor
    state_tensor = torch.tensor(state_list, dtype=torch.float32, device=device).unsqueeze(0) # Add batch dimension
    return state_tensor



def get_state_sequence(
    action_history: deque,
    rel_pos_x_history: deque,
    rel_pos_y_history: deque,
    rel_vel_x_history: deque,
    rel_vel_y_history: deque,
    seq_length: int = SEQ_LENGTH_DEFAULT,
    padding_value: float = PADDING_VALUE,
    action_padding_value: int = ACTION_PADDING_VALUE
):
    """
    Creates a state tensor representing the history of actions, relative positions,
    and relative velocities, suitable for sequence models like attention networks.

    Args:
        action_history (deque): Deque of past actions taken (integers).
        rel_pos_x_history (deque): Deque of past relative x positions (normalized).
        rel_pos_y_history (deque): Deque of past relative y positions (normalized).
        rel_vel_x_history (deque): Deque of past relative x velocities (normalized).
        rel_vel_y_history (deque): Deque of past relative y velocities (normalized).
        seq_length (int): The desired fixed sequence length for the output tensor.
        padding_value (float): Value used for padding position/velocity sequences.
        action_padding_value (int): Value used for padding the action sequence.

    Returns:
        torch.Tensor: A tensor of shape (1, 5, seq_length) containing the padded sequences.
                      Features are ordered [action, rel_pos_x, rel_pos_y, rel_vel_x, rel_vel_y].
        torch.Tensor: A boolean mask tensor of shape (1, seq_length) where True indicates
                      a real data point and False indicates padding. Useful for attention mechanisms.
    """
    sequences = []
    # Combine all history deques for easier iteration
    histories = [
        action_history, rel_pos_x_history, rel_pos_y_history, rel_vel_x_history, rel_vel_y_history
    ]
    # Define padding value for each sequence type
    paddings = [action_padding_value] + [padding_value] * 4

    actual_len = 0 # Store the actual length before padding

    for i, (history, pad_val) in enumerate(zip(histories, paddings)):
        # Convert deque to list and get the last seq_length items
        recent_history = list(history)[-seq_length:]

        # Store the actual length of the sequence before padding (use any history's length)
        if i == 0:
            actual_len = len(recent_history)

        # Calculate padding needed (prepends padding)
        pad_len = seq_length - len(recent_history)

        # Create the padded sequence, ensuring consistent data types
        if isinstance(pad_val, float):
            # Pad with float, convert history to float
            processed_history = [float(x) for x in recent_history]
            padded_sequence = ([pad_val] * pad_len) + processed_history
        else: # Assuming integer padding (like for actions)
             # Pad with int, convert history to int
            processed_history = [int(x) for x in recent_history]
            padded_sequence = ([pad_val] * pad_len) + processed_history

        sequences.append(padded_sequence)

    # Create the attention mask (True for real data, False for padding)
    mask_pad_len = seq_length - actual_len
    # Mask has shape (1, seq_length)
    mask = torch.tensor(([False] * mask_pad_len) + ([True] * actual_len), device=device, dtype=torch.bool).unsqueeze(0)

    # Stack sequences into a tensor
    # The resulting tensor shape will be (5, seq_length)
    # Ensure the tensor is float32 for neural network input
    state_tensor_stacked = torch.tensor(sequences, dtype=torch.float32, device=device)

    # Add the batch dimension -> (1, 5, seq_length)
    state_tensor_batch = state_tensor_stacked.unsqueeze(0)

    return state_tensor_batch, mask

# --- Placeholder function to calculate current relative state values ---
# You'll call this in your main loop before appending to the history deques.
def calculate_current_relative_state(rocket, target_planet):
    """Calculates the *current* relative state values."""
    if rocket is None or target_planet is None:
        # Return zeros or handle appropriately if objects are missing
        # Returning NaNs might be better to signal invalid state if not handled earlier
        return math.nan, math.nan, math.nan, math.nan

    # Relative position (normalize by AU) [cite: 122]
    rel_x = (rocket.x - target_planet.x) / AU
    rel_y = (rocket.y - target_planet.y) / AU
    # Relative velocity (normalized) [cite: 124]
    rel_vx = (rocket.vx - target_planet.vx) / NORM_FACTOR_VEL
    rel_vy = (rocket.vy - target_planet.vy) / NORM_FACTOR_VEL

    return rel_x, rel_y, rel_vx, rel_vy

# --- Function to reset a specific rocket (useful after crash/OOB) ---
def reset_rocket_state(rocket, planets):
	"""Resets a single rocket's position/velocity near a start planet."""
	start_planet_idx = 3 # Assuming Earth is index 3
	if start_planet_idx >= len(planets): start_planet_idx = 0 
	start_planet = planets[start_planet_idx]


	rocket.x = start_planet.x
	rocket.y = start_planet.y + 0.00256954 * AU
	rocket.vx = start_planet.vx -1.022*1000
	rocket.vy = start_planet.vy 
	# Reset reactors and potentially other state if needed
	rocket.reactor1 = False
	rocket.reactor2 = False
	rocket.reactor3 = False
	# Reset mass/fuel if tracked later
	# rocket.fuel = rocket.initial_fuel 
	# rocket.mass = rocket.structure_mass + rocket.fuel

	print("Rocket state reset.")
	return rocket

# --- Function to reset the environment for a new episode ---
def reset_environment(planets):
    """Resets rocket position/velocity for a new episode."""
    # Choose a starting planet (e.g., Earth)
    start_planet_idx = 3 # Assuming Earth is index 3
    if start_planet_idx >= len(planets):
        start_planet_idx = 0 # Default to Sun if Earth doesn't exist

    start_planet = planets[start_planet_idx]

    # Start in a somewhat random position near the starting planet
    angle = random.uniform(0, 2 * math.pi)
    dist_factor = random.uniform(1.5, 3.0) # Start further away than just radius
    start_dist = start_planet.radius * dist_factor if hasattr(start_planet, 'radius') else 1e7
    
    start_x = start_planet.x + start_dist * math.cos(angle)
    start_y = start_planet.y + start_dist * math.sin(angle)

    # Create the rocket (assuming SpaceShip class exists)
    # Give it some initial velocity, perhaps tangential to start planet
    tangential_speed = math.sqrt(G * start_planet.mass / start_dist) * 0.5 # Start slower than orbit
    start_vx = start_planet.vx - tangential_speed * math.sin(angle)
    start_vy = start_planet.vy + tangential_speed * math.cos(angle)
    
    rocket = SpaceShip(start_x, start_y, 500, 5, BLUE) # Use default parameters
    rocket.vx = start_vx
    rocket.vy = start_vy

    # Reset target (optional, could keep same target)
    # Target = random.randint(0, len(planets) - 1) 

    return [rocket] # Return list containing the single rocket




def save_distance_data(all_acquisitions, save_folder="metrics"):
    """Saves recorded distance data to a CSV file using pandas.
       Each column represents one acquisition (run between resets).
    """
    if not all_acquisitions or not any(all_acquisitions): # Check if empty or contains only empty lists
        print("No distance data recorded to save.")
        return

    try:
        # Create timestamped folder
        filename = os.path.join(save_folder, "distances_log.csv") # New filename maybe

        # Prepare data for DataFrame: dictionary where keys are column names
        data_dict = {}
        for i, acq in enumerate(all_acquisitions):
            # Only include non-empty acquisitions if desired, or handle them all
            if acq: # Ensure the acquisition list is not empty
                 data_dict[f"Acquisition_{i+1}"] = pd.Series(acq)
            # else: # Optionally handle empty acquisitions if needed
            #     data_dict[f"Acquisition_{i+1}"] = pd.Series([]) # Add empty series

        if not data_dict:
             print("All recorded acquisitions were empty. Nothing to save.")
             return

        # Create DataFrame directly from dictionary of Series
        # Pandas handles unequal lengths by padding with NaN automatically
        df = pd.DataFrame(data_dict)

        # Save DataFrame to CSV
        df.to_csv(filename, index=False, float_format='%.2f') # Save without index, format floats

        print(f"Distance data saved to {filename} using pandas.")

    except ImportError:
         print("Error: pandas library not found. Please install it (`pip install pandas`) to use this save function.")
    except Exception as e:
        print(f"Error saving distance data using pandas: {e}")
        import traceback
        traceback.print_exc()