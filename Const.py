import random
import torch

AU = 149.6e9
G = 6.67428e-11
Scale = 250 / AU   # 1AU = 100 pixels
Step_p_frame = 1
TIMESTEP = 3600 # 1 hour
WIDTH, HEIGHT =  600, 300
FPS = 60e3
ACCELERATING_RATE = 1_000_000


WHITE = (255, 255, 255)
JUP = (255,255,75)
YELLOW = (255, 255, 0)
BLUE = (100, 149, 237)
RED = (188, 39, 50)
DARK_GREY = (80, 78, 81)
LIGHT_GREY = (175,173,179)
GREEN = (100,230,72)

random.random()

DEVICE = "mps"

# Weights for reward components 

# --- Constants for Target Distance Reward Function ---
TARGET_DISTANCE_FACTOR = 5.0       # Target distance = TARGET_DISTANCE_FACTOR * planet_radius
# Scale factor for error (e.g., 1 million km = 1e9 meters, or adjust based on typical orbital scales)
DISTANCE_REWARD_SCALE_FACTOR = 1e9
# Alpha controls the sharpness of the peak reward (applied to scaled error)
REWARD_ALPHA = 10.0
# Coeff determines the max bonus reward at the target distance
REWARD_COEFF = 50.0     
""" TESTS ON REWARD DISTANCE
# REWARD_POTENTIAL_SCALE = 1e13
"""
REWARD_DISTANCE_SCALE = 1e-9  # Scales the reward/penalty for change in distance


REWARD_SPEED_SCALE = 1e1    # Scales the reward for being close to orbital speed
COST_PER_THRUST = .1         # Penalty for firing any thruster (action > 0)
TIME_PENALTY = 0.01           # Small penalty for each time step taken
GOAL_REWARD = 100.0           # Large reward for achieving stable orbit with motors off
ORBIT_SPEED_TOLERANCE = 100   # Speed difference tolerance (m/s) to be considered 'in orbit' for reward


# --- DQN Hyperparameters ---
STATE_SIZE = 4          # State vector size: [dx, dy, dvx, dvy] -> Update if state definition changes
ACTION_SIZE = 7         # Actions: 0: none, 1: right, 2: left, 3: up, 4:right+up, 5:left+up, 6:right+left
MEMORY_CAPACITY = 50000 # Replay memory size (adjust based on RAM)
BATCH_SIZE = 128        # Number of experiences to sample for each learning step
GAMMA = 0.99            # Discount factor for future rewards
EPS_START = 0.25        # Starting value for epsilon (exploration rate)
EPS_END = 0.05          # Minimum value for epsilon
EPS_DECAY = 20000       # Controls how fast epsilon decreases (higher means slower decay)
TAU = 0.005             # Soft update parameter for target network weights
LR = 5e-4               # Learning rate for the Adam optimizer
TARGET_UPDATE_FREQ = 10 # How many steps between hard updates of target network (if not using soft updates)
                        # If using soft updates (TAU > 0), this can be ignored or set to 1 for updates every step.

# --- Att Hyperparameters ---
# Define constants (consider moving these to Const.py)
SEQ_LENGTH_DEFAULT = 50
# Velocity normalization factor used in the original get_state [cite: 124]
NORM_FACTOR_VEL = 5e4
PADDING_VALUE = 0.0 # Default padding for numerical sequences
ACTION_PADDING_VALUE = 0 # Default padding for action sequence
# These should match the output of get_state_sequence
NUM_FEATURES = 5  # action, rel_x, rel_y, rel_vx, rel_vy
SEQ_LENGTH = 50   # Example sequence length, should be configurable
MAX_HISTORY_LEN = 5000 # Max length for deques (can be larger than SEQ_LENGTH)
MODEL_NAME = "AttentionDQN" # Name for saving/loading

# --- Simulation Settings for RL ---
MAX_STEPS_PER_EPISODE = 1000 # Limit episode length
CRASH_DISTANCE_THRESHOLD = 1.0E7 # Example: Define a crash distance from planet surface (meters) - needs tuning
OUT_OF_BOUNDS_DISTANCE = 25 * AU # Example: Max distance from Sun before episode ends

Core_position = {
                0:'Sun',
                1:'Mercury',
                2:'Venus',
                3:'Earth',
                4:'Moon',
                5:'Mars',
                6:'Jupiter',
                7:'Saturn',
                8:'Uranus',
                9:'Neptune',
                10:'Pluto',
                }