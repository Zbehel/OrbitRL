# Project: Solar System Navigation with Attention-Based RL Agent

## Description

This project simulates a simplified 2D solar system environment using Pygame and trains a Deep Reinforcement Learning (RL) agent to navigate a spaceship. The agent's goal is typically to reach a target planet or achieve a stable orbit around it. The current implementation utilizes an advanced Deep Q-Network (DQN) architecture featuring an **Attention Mechanism** (`AttentionDQNNetwork`) to process sequential state information, allowing the agent to consider its recent trajectory history when making decisions.

## Features

* **2D Solar System Simulation:** Simulates gravitational interactions between celestial bodies (sun, planets, spaceship) using Newton's law of universal gravitation.
* **Pygame Visualization:** Renders the simulation environment, showing planets, spaceship trajectory, and reactor status. Provides interactive controls.
* **Reinforcement Learning Agent:** Implements an `Agent` class using PyTorch.
    * **AttentionDQN Network:** A custom neural network using 1D convolutions and multi-head self-attention to process sequences of past states (relative position, relative velocity, past actions).
    * **Experience Replay:** Stores past experiences (`state`, `action`, `reward`, `next_state`, `done`, including masks) in a `ReplayMemory` for stable training.
    * **Epsilon-Greedy Exploration:** Balances exploring new actions and exploiting known good actions, with a decaying exploration rate (`epsilon`).
    * **Target Network:** Uses a separate target network for stable Q-value estimation during learning.
* **Sequence-Based State:** The agent uses a history of the last `SEQ_LENGTH` steps (relative position, velocity, and actions) as input, processed via `get_state_sequence`.
* **Customizable Reward Function:** The reward signal guides the agent's learning, composed of components rewarding distance improvement, achieving target orbital speed, and penalizing thrust usage, time, and negative outcomes (crashing, going OOB).
* **Training Loop:** Integrates the simulation, agent interaction, learning steps (`optimize_model`), and target network updates within the main loop.
* **Logging:** Records key training metrics (episode number, steps, total reward, average loss, epsilon, goal achievement) to a CSV file (`training_log.csv`) for monitoring learning progress.
* **Model Saving/Loading:** Automatically saves the trained policy network weights (`agent_policy_weights.pth`) and distance logs (`distances_log.csv`) in timestamped directories within `Metrics/`. Allows loading pre-trained models to continue training or evaluate performance.

## File Structure
.
├── main.txt                # Main script to run the simulation and training loop
├── Agent.txt               # Contains the RL Agent class, AttentionDQNNetwork, ReplayMemory
├── utils.txt               # Utility functions (physics, state calculation, resets, etc.)
├── Class_Def.txt           # Class definitions for Celestial_Core, Planet, SpaceShip
├── Const.txt               # Constants, hyperparameters, reward weights, simulation settings
├── Metrics/                # Directory created to store logs, saved models, and distance data
│   └── AttentionDQN/       # Subdirectory for the specific model type
│       └── YYYYMMDD-HHMMSS/ # Timestamped directory for each run
│           ├── training_log.csv
│           ├── distances_log.csv
│           └── agent_policy_weights.pth
└── README.md               # This file


## Setup

1.  **Prerequisites:** Python 3.x
2.  **Install Libraries:**
    ```bash
    pip install pygame torch numpy pandas
    ```
    * `pygame`: For simulation visualization and interaction.
    * `torch`: For the deep learning framework (neural networks, tensors).
    * `numpy`: Used for numerical operations (e.g., calculating mean loss).
    * `pandas`: Used by `utils.save_distance_data` to save distance logs.

## Usage

1.  **Run the Simulation:**
    ```bash
    python main.py
    ```
    (Assuming you rename the `.txt` files to `.py`)
2.  **Training:** The simulation starts training the agent immediately. Progress is logged to the console and the `training_log.csv` file within the latest `Metrics/AttentionDQN/` subdirectory.
3.  **Loading Weights:** To load the latest trained model instead of starting from scratch, modify the last line in `main.py`:
    ```python
    if __name__ == '__main__':
        # SatSim(load_weights=False) # Start fresh training
        SatSim(load_weights=True) # Load latest saved weights
    ```
4.  **Interactive Controls (during simulation):**
    * `o`/`p`: Zoom In / Zoom Out
    * `1`/`2`: Speed up simulation time
    * `3`/`4`: Slow down simulation time
    * `a`/`z`: Focus view on previous/next celestial body or rocket
    * `t`: Focus view on the target planet
    * `r`: Select a new random target planet
    * `SPACE`: Launch a new rocket from Earth's position
    * `BACKSPACE`: Remove the last launched rocket
    * `x`: Reset the currently controlled rocket's state and log the episode.
    * `q`: Quit the simulation (will also trigger logging and saving).
    * `h`: Display help text with commands in the window.

## Key Components Explained

* **`main.txt`**
    * Orchestrates the entire process.
    * Initializes Pygame, celestial bodies (`Solar_System`), the `Agent`, history deques (`action_hist`, etc.), and logging (`agent.setup_logging`).
    * Contains the main `while run:` loop: Handles Pygame events, calculates state (`calculate_current_relative_state`, `get_state_sequence`), gets actions (`agent.select_action`), applies actions, updates physics (`update_position`), calculates rewards, stores experiences (`agent.store_experience`), triggers learning (`agent.optimize_model`, `agent.update_target_net`), handles episode termination/logging (`agent.log_episode_data`), and updates the display.
    * Calls `agent.save_weights_and_distances` and `agent.close_log` upon quitting.

* **`Agent.txt`**
    * Defines the `Agent` class responsible for learning and decision-making.
    * Initializes the `AttentionDQNNetwork` for `policy_net` and `target_net`.
    * `AttentionDQNNetwork`: Processes sequence input `(batch, features, seq_len)` and masks using `Conv1d`, `MultiheadAttention`, `LayerNorm`, and residual connections to output Q-values.
    * `ReplayMemory`: Stores `Transition` tuples (including sequence/mask tuples) using a `deque`.
    * `select_action`: Implements epsilon-greedy strategy using the sequence/mask state tuple.
    * `store_experience`: Pushes the complete experience tuple to `ReplayMemory`.
    * `optimize_model`: Samples batches, calculates Q-values and targets using sequence/mask inputs, computes loss, performs optimization, and stores step loss.
    * `update_target_net`: Updates the target network.
    * `setup_logging`, `log_episode_data`, `close_log`: Handle CSV logging.
    * `save_weights_and_distances`, `load_weights`: Manage saving/loading network weights and distance data.

* **`utils.txt`**
    * Contains helper functions.
    * `update_position`: Core physics engine (gravity, thrust, simple collisions).
    * `Solar_System`: Creates initial planets.
    * `Launch`: Creates a spaceship.
    * `get_distance`, `is_orbiting`: Geometric helpers.
    * `calculate_current_relative_state`: Computes instantaneous relative state values.
    * `get_state_sequence`: Constructs the padded state sequence tensor and attention mask.
    * `reset_rocket_state`: Resets rocket position/velocity.
    * `save_distance_data`: Saves distance logs using pandas.

* **`Class_Def.txt`**
    * Defines object structure: `Celestial_Core` (base), `Planet` (adds radius/color), `SpaceShip` (adds reactors, `apply_action`). Implements `draw` methods for visualization.

* **`Const.txt`**
    * Centralizes constants: Physical (`G`, `AU`), Simulation (`WIDTH`, `HEIGHT`, `TIMESTEP`), Visualization (`Colors`), RL Hyperparameters (`GAMMA`, `LR`, `EPS_...`, `BATCH_SIZE`, `SEQ_LENGTH`, `NUM_FEATURES`, `ACTION_SIZE`), Reward Weights (`REWARD_..._SCALE`, `COST_PER_THRUST`, etc.), Termination Conditions.

## Agent Logic and Learning Process

### 1. Agent Perception: What the Agent "Sees"

The agent doesn't perceive the simulation visually. Its understanding comes from a sequence of numerical data representing its recent history:

* **Sequence Input:** The agent receives the history of the last `SEQ_LENGTH` time steps.
* **State Features:** For each step in the history, the input includes:
    1.  The **action** taken in that step.
    2.  Normalized **relative X/Y position** to the target.
    3.  Normalized **relative X/Y velocity** to the target.
* **State Construction:** The `get_state_sequence` function compiles this history from deques (`action_hist`, etc.) maintained in `main.txt`, padding the start if the history is short, and produces the `state_sequence` tensor and a `state_mask` tensor (indicating real vs. padded data).
* **Attention Processing:** The `AttentionDQNNetwork` uses its attention layers, guided by the `state_mask`, to focus on the most relevant parts of this historical sequence when determining the value of actions.

### 2. The Reward Process: Guiding the Agent

After the agent acts and the simulation updates, a `reward` signal is calculated in `main.txt` to provide feedback:

* **Reward Components:** The total `step_reward` sums several factors:
    * **Distance Change:** Rewards getting closer, penalizes getting farther.
    * **Orbital Speed Match:** Rewards matching the ideal orbital speed for the current distance.
    * **Action Cost:** Penalizes using thrusters to encourage efficiency.
    * **Time Penalty:** Small penalty per step to encourage speed.
    * **Goal Bonus:** Large reward for achieving stable orbit with thrusters off.
    * **Terminal Penalties:** Large penalties for crashing or going out of bounds.
* **Feedback Loop:** This scalar reward is stored in the `ReplayMemory` with the corresponding state, action, and next state information.

### 3. Impact of Rewards on Decisions and Learning

Rewards drive the learning process, teaching the agent which actions lead to better outcomes:

* **Goal (Q-Learning):** The agent learns to predict **Q-values**, representing the expected total future discounted reward for taking an action in a given state.
* **Learning (`optimize_model`):**
    1.  The agent replays past experiences.
    2.  It calculates a **target Q-value** for each experienced action using the actual `reward` received plus the estimated value of the `next_state` (from the `target_net`).
    3.  The **loss** measures the difference between the `policy_net`'s predicted Q-value and this target.
    4.  The **optimizer** adjusts the `policy_net`'s weights to minimize this loss.
* **Shaping Behavior:** Actions leading to high rewards or states with high estimated future value get their Q-values increased. Actions leading to penalties or low-value states get their Q-values decreased.
* **Decision Making (`select_action`):** During **exploitation**, the agent chooses the action with the highest predicted Q-value. As learning progresses, actions that the reward function has indicated are beneficial (directly or indirectly) will have higher Q-values and thus be chosen more often.

## RL Details Summary (Condensed)

* **State:** Sequence of last `SEQ_LENGTH` steps (past action, rel_pos(x,y), rel_vel(x,y)) + mask.
* **Actions:** 7 discrete thruster combinations.
* **Reward:** Shaped reward encouraging distance reduction, orbital speed matching, efficiency, and goal achievement, with penalties for failure states.
* **Algorithm:** Attention-Based Deep Q-Network (AttentionDQN) with Experience Replay and Target Network.

## Future Work / Improvements

* Implement more realistic physics (e.g., N-body simulation, fuel consumption affecting mass).
* Explore continuous action spaces (variable thrust) using algorithms like DDPG, TD3, or SAC.
* Experiment with different attention mechanisms or Transformer architectures.
* Refine the reward function further.
* Implement more sophisticated goal conditions or multi-stage goals.
* Use hyperparameter optimization tools (e.g., Optuna) to find better hyperparameters.
* Add more complex scenarios (e.g., avoiding asteroids, docking).
* Improve visualization (e.g., show velocity vectors, predicted trajectories).