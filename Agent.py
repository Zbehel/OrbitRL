import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math

from datetime import datetime
import os
from collections import namedtuple, deque
import csv

from utils import save_distance_data
from Const import *

# Use GPU if available, otherwise CPU
device = DEVICE

# Define the structure for experiences stored in replay memory
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))

Att_Transition = namedtuple('Att_Transition',
                            ('state', 'state_mask', 'action', 'next_state', 'next_state_mask', 'reward', 'done'))
class ReplayMemory(object):
    """Stores transitions for experience replay."""

    def __init__(self, capacity):
        """Initialize a ReplayMemory instance.

        Args:
            capacity (int): Maximum size of the replay buffer.
        """
        # Use deque for efficient adding/popping from both ends
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition tuple."""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """Randomly sample a batch of transitions from memory.

        Args:
            batch_size (int): The number of transitions to sample.

        Returns:
            list: A list of Transition tuples.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Return the current size of the memory."""
        return len(self.memory)


class DQNNetwork(nn.Module):
    """Neural Network for approximating Q-values."""

    def __init__(self, n_observations, n_actions):
        """Initialize the DQN Network.

        Args:
            n_observations (int): The size of the state space (input dimensions).
            n_actions (int): The size of the action space (output dimensions).
        """
        super(DQNNetwork, self).__init__()
        # Simple feedforward network
        self.layer1 = nn.Linear(n_observations, 128) # Input layer -> Hidden layer 1
        self.layer2 = nn.Linear(128, 128)             # Hidden layer 1 -> Hidden layer 2
        self.layer3 = nn.Linear(128, n_actions)       # Hidden layer 2 -> Output layer

    def forward(self, x):
        """Defines the forward pass of the network.

        Args:
            x (torch.Tensor): The input state tensor.

        Returns:
            torch.Tensor: The output Q-values for each action.
        """
        x = F.relu(self.layer1(x)) # Apply ReLU activation function
        x = F.relu(self.layer2(x))
        return self.layer3(x)      # Output raw Q-values



class DuelingDQNNetwork(nn.Module):
    """Dueling Network Architecture for Q-values.
       Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
    """
    def __init__(self, n_observations, n_actions):
        """Initialize the Dueling DQN Network.

        Args:
            n_observations (int): The size of the state space.
            n_actions (int): The size of the action space.
        """
        super(DuelingDQNNetwork, self).__init__()
        self.n_actions = n_actions

        # Shared feature learning layers (example: similar size to your original)
        self.feature_layer1 = nn.Linear(n_observations, 128)
        self.feature_layer2 = nn.Linear(128, 128)

        # State Value stream
        self.value_stream_layer1 = nn.Linear(128, 128)
        self.value_stream_output = nn.Linear(128, 1) # Outputs single value V(s)

        # Action Advantage stream
        self.advantage_stream_layer1 = nn.Linear(128, 128)
        self.advantage_stream_output = nn.Linear(128, n_actions) # Outputs advantage A(s,a) for each action

    def forward(self, x):
        """Defines the forward pass."""
        # Pass through shared feature layers
        features = F.relu(self.feature_layer1(x))
        features = F.relu(self.feature_layer2(features))

        # Calculate state value
        value_hidden = F.relu(self.value_stream_layer1(features))
        value = self.value_stream_output(value_hidden) # V(s)

        # Calculate action advantages
        advantage_hidden = F.relu(self.advantage_stream_layer1(features))
        advantages = self.advantage_stream_output(advantage_hidden) # A(s,a)

        # Combine value and advantage streams to get Q-values
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        # The subtraction of the mean advantage ensures identifiability and improves stability
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values

class Agent:
    """Interacts with and learns from the environment using DQN."""

    def __init__(self, state_size, action_size, replay_memory_capacity, batch_size,
                 gamma, eps_start, eps_end, eps_decay, tau, lr, target_update_freq):
        """Initialize the DQN Agent.

        Args:
            state_size (int): Dimension of each state.
            action_size (int): Dimension of each action.
            replay_memory_capacity (int): Size of the replay buffer.
            batch_size (int): Minibatch size for training.
            gamma (float): Discount factor.
            eps_start (float): Starting value of epsilon for epsilon-greedy.
            eps_end (float): Minimum value of epsilon.
            eps_decay (float): Decay rate for epsilon.
            tau (float): Soft update parameter for target network.
            lr (float): Learning rate for the optimizer.
            target_update_freq (int): Frequency (in steps or episodes) for hard target updates (if tau=0).
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = ReplayMemory(replay_memory_capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.lr = lr
        self.target_update_freq = target_update_freq # Note: Used only if tau is 0 (hard updates)

        # --- Networks ---
        # Policy Network: Learns and selects actions
        self.policy_net = DuelingDQNNetwork(state_size, action_size).to(device)
        # Target Network: Provides stable targets for learning
        self.target_net = DuelingDQNNetwork(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict()) # Initialize target with policy weights
        self.target_net.eval()  # Set target network to evaluation mode (no dropout, batchnorm updates)

        # --- Optimizer ---
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)

        self.steps_done = 0  # Counter for epsilon decay

    def select_action(self, state):
        """Selects an action using an epsilon-greedy policy.

        Args:
            state (torch.Tensor): The current state.

        Returns:
            torch.Tensor: The selected action tensor.
        """
        sample = random.random()
        # Calculate current epsilon value based on exponential decay
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        if sample > eps_threshold:
            # Exploitation: Choose the best action from the policy network
            with torch.no_grad():  # Disable gradient calculation during inference
                # policy_net(state) -> Q-values for all actions
                # .max(1)[1] -> index of the max Q-value (the best action)
                # .view(1, 1) -> reshape for compatibility
                action = self.policy_net(state).max(1)[1].view(1, 1)
        else:
            # Exploration: Choose a random action
            action = torch.tensor([[random.randrange(self.action_size)]], device=device, dtype=torch.long)

        return action

    def store_experience(self, state, action, next_state, reward, done):
        """Stores a single experience tuple in the replay memory.

        Args:
            state (torch.Tensor): The starting state.
            action (torch.Tensor): The action taken.
            next_state (torch.Tensor or None): The resulting state (None if terminal).
            reward (torch.Tensor): The reward received.
            done (bool): True if the episode terminated, False otherwise.
        """
        # Convert done flag to tensor before storing
        done_tensor = torch.tensor([done], device=device, dtype=torch.bool)
        self.memory.push(state, action, next_state, reward, done_tensor)

    def optimize_model(self):
        """Performs one step of optimization on the policy network."""
        if len(self.memory) < self.batch_size:
            return  # Not enough samples in memory to form a batch

        # Sample a batch of transitions
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch: Converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # --- Prepare Batch Tensors ---
        # Create a mask for non-final next states (handles terminal states where next_state is None)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                      device=device, dtype=torch.bool)
        # Concatenate non-final next states into a tensor
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        # Concatenate batch elements into tensors
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        # done_batch = torch.cat(batch.done) # Not explicitly used in standard DQN loss below

        # --- Calculate Q(s_t, a) ---
        # Get Q-values from the policy network for the actions that were actually taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # --- Calculate V(s_{t+1}) = max_a Q_target(s_{t+1}, a) ---
        # Initialize next state values to zero (for terminal states)
        next_state_values = torch.zeros(self.batch_size, device=device)
        # Compute max Q-value for non-final next states using the target network
        with torch.no_grad(): # No gradient needed for target calculations
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        # --- Compute Expected Q Values (Bellman Target) ---
        # Target = reward + gamma * V(s_{t+1})
        # For terminal states (where next_state_values is 0), target is just the reward
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # --- Compute Loss ---
        # Huber loss (Smooth L1 loss) is often more stable than MSELoss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1)) # Target needs same shape

        # --- Optimize the Policy Network ---
        self.optimizer.zero_grad()  # Clear previous gradients
        loss.backward()             # Calculate gradients
        # In-place gradient clipping (helps prevent exploding gradients)
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()       # Update network weights

    def update_target_net(self, soft_update=True):
        """Updates the target network weights.

        Args:
            soft_update (bool): If True, uses soft updates (Polyak averaging).
                                If False, performs a hard copy (less common with tau > 0).
        """
        if soft_update and self.tau > 0:
            # Soft update: target_weights = tau * policy_weights + (1 - tau) * target_weights
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + \
                                             target_net_state_dict[key] * (1 - self.tau)
            self.target_net.load_state_dict(target_net_state_dict)
        elif not soft_update:
            # Hard update: Copy weights directly
            # Typically done less frequently (e.g., every target_update_freq steps/episodes)
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_weights_and_distances(self, all_distances_acquisitions,model_name = 'DQN', filename="agent_policy_weights.pth"):
        """Saves the weights of the policy network."""
        
        # You can specify a different weights file path if needed
        datetime_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        models_dir = f"Metrics/{model_name}/{datetime_str}/"
        #Create models directory if it doesn't exist

        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        # Save first the distance data to a CSV file
        save_distance_data(all_distances_acquisitions, save_folder=models_dir)

        filename = os.path.join(models_dir, "agent_policy_weights.pth")
        try:
            # Ensure directory exists if filename includes path
            save_dir = os.path.dirname(filename)
            if save_dir and not os.path.exists(save_dir):
                 os.makedirs(save_dir, exist_ok=True)
                 
            torch.save(self.policy_net.state_dict(), filename)
            print(f"Agent policy weights saved to {filename}")
        except Exception as e:
            print(f"Error saving agent weights: {e}")

    def load_weights(self, model_name = 'DQN/'):
        """Loads weights into the policy and target networks."""
        all_subdirs = [d for d in os.listdir(f'Metrics/{model_name}/')]
        latest_subdir = max(all_subdirs)
        path = os.path.join('Metrics/', model_name)
        path = os.path.join(path, latest_subdir)
        filename = os.path.join(path, "agent_policy_weights.pth")

        # Check if the file exists before attempting to load
        if not os.path.exists(filename):
            print(f"Warning: Weight file not found at {filename}. Starting with initial weights.")
            return False
        try:
            # Load state dict, mapping to the correct device (CPU or GPU)
            state_dict = torch.load(filename, map_location=device) # Use map_location!
            self.policy_net.load_state_dict(state_dict)
            # It's crucial to also update the target network after loading weights
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.policy_net.eval() # Set to eval mode if needed after loading
            self.target_net.eval()
            print(f"Agent policy weights loaded from {filename}")
            return True # Indicate success
        except Exception as e:
            print(f"Error loading agent weights from {filename}: {e}")
            return False # Indicate failure
        


class Att_ReplayMemory(object):
    """Stores transitions (including masks) for experience replay."""

    def __init__(self, capacity):
        """Initialize a ReplayMemory instance."""
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition tuple (state, state_mask, action, next_state, next_state_mask, reward, done)."""
        # Ensure the correct number of arguments are passed
        assert len(args) == 7, "Incorrect number of arguments for Transition tuple"
        self.memory.append(Att_Transition(*args))

    def sample(self, batch_size):
        """Randomly sample a batch of transitions from memory."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Return the current size of the memory."""
        return len(self.memory)


# --- Attention-Based Network ---
class AttentionDQNNetwork(nn.Module):
    """
    DQN Network using Attention to process sequence states.
    Uses learned positional embeddings instead of initial Conv1D layers.
    Input shape: (batch, num_features, seq_length)
    Mask shape: (batch, seq_length)
    Output shape: (batch, n_actions)
    """
    def __init__(self, n_features, seq_len, n_actions, embed_dim=64, num_heads=4):
        """
        Args:
            n_features (int): Number of features in the input sequence (e.g., 5).
            seq_len (int): Length of the input sequence.
            n_actions (int): Number of possible actions.
            embed_dim (int): Dimension for embedding and attention mechanism.
            num_heads (int): Number of attention heads.
        """
        super(AttentionDQNNetwork, self).__init__()
        self.n_features = n_features
        self.seq_len = seq_len
        self.n_actions = n_actions
        self.embed_dim = embed_dim

        # --- Replace Conv1D layers with Input Projection ---
        # Use a Conv1D with kernel_size=1 to project features to embed_dim at each time step
        # Equivalent to applying a Linear layer independently at each time step
        self.input_proj = nn.Conv1d(n_features, embed_dim, kernel_size=1)

        # --- Learned Positional Embedding ---
        # Creates learnable embedding vectors for each position (0 to seq_len-1)
        self.positional_embedding = nn.Embedding(seq_len, embed_dim)

        # --- Attention Layer ---
        # MultiheadAttention expects (seq_len, batch, embed_dim) or (batch, seq_len, embed_dim) if batch_first=True
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True) # Use batch_first=True

        # --- Layer Normalization ---
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # --- Feed-forward layers after attention ---
        self.linear1 = nn.Linear(embed_dim, embed_dim * 2)
        self.linear2 = nn.Linear(embed_dim * 2, n_actions)

    def forward(self, x, attention_mask):
        """
        Forward pass through the network with positional embeddings.

        Args:
            x (torch.Tensor): Input state sequence tensor (batch, n_features, seq_len).
            attention_mask (torch.Tensor): Boolean mask (batch, seq_len). True indicates valid data.

        Returns:
            torch.Tensor: Q-values for each action (batch, n_actions).
        """
        # 1. Project Input Features to Embedding Dimension
        # Input x: (batch, n_features, seq_len)
        x = self.input_proj(x) # Output: (batch, embed_dim, seq_len)
        # Apply activation after projection
        x = F.relu(x)

        # 2. Generate Positional Embeddings
        # Create position indices (0, 1, ..., seq_len-1)
        positions = torch.arange(0, self.seq_len, device=x.device).unsqueeze(0) # Shape: (1, seq_len)
        # Get embeddings for these positions
        pos_emb = self.positional_embedding(positions) # Shape: (1, seq_len, embed_dim)

        # 3. Add Positional Embeddings
        # Permute x to match positional embedding shape for addition
        # x shape becomes (batch, seq_len, embed_dim)
        x = x.permute(0, 2, 1)
        # Add positional embeddings (broadcasts along batch dimension)
        x = x + pos_emb

        # --- The rest of the network remains the same ---

        # 4. Apply Attention
        key_padding_mask = ~attention_mask # Invert mask: True means ignore

        # Apply LayerNorm before attention
        x_norm = self.norm1(x)

        # Self-attention
        attn_output, _ = self.attention(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask)

        # Add & Norm (Residual connection)
        x = x + attn_output
        x = self.norm2(x)

        # 5. Aggregate Sequence Information (e.g., Masked Average Pooling)
        mask_expanded = attention_mask.unsqueeze(-1).float()
        masked_sum = (x * mask_expanded).sum(dim=1)
        valid_counts = mask_expanded.sum(dim=1)
        valid_counts = torch.clamp(valid_counts, min=1.0)
        aggregated_output = masked_sum / valid_counts # Shape: (batch, embed_dim)

        # 6. Final Feed-Forward Layers for Q-Values
        q_values = F.relu(self.linear1(aggregated_output))
        q_values = self.linear2(q_values) # Shape: (batch, n_actions)

        return q_values


class Att_Agent:
    """Interacts with and learns from the environment using Attention DQN."""

    def __init__(self, n_features, seq_len, action_size, replay_memory_capacity, batch_size,
                 gamma, eps_start, eps_end, eps_decay, tau, lr, target_update_freq, embed_dim=64, num_heads=4):
        """Initialize the Attention DQN Agent.

        Args:
            n_features (int): Number of features per time step in the state sequence.
            seq_len (int): The length of the state sequence.
            action_size (int): Dimension of each action.
            replay_memory_capacity (int): Size of the replay buffer.
            batch_size (int): Minibatch size for training.
            gamma (float): Discount factor.
            eps_start (float): Starting value of epsilon for epsilon-greedy.
            eps_end (float): Minimum value of epsilon.
            eps_decay (float): Decay rate for epsilon.
            tau (float): Soft update parameter for target network.
            lr (float): Learning rate for the optimizer.
            target_update_freq (int): Frequency for hard target updates (if tau=0).
            embed_dim (int): Embedding dimension for the Attention network.
            num_heads (int): Number of attention heads for the Attention network.
        """
        self.n_features = n_features
        self.seq_len = seq_len
        self.action_size = action_size
        self.memory = Att_ReplayMemory(replay_memory_capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.lr = lr
        self.target_update_freq = target_update_freq # Used only if tau is 0 (hard updates)
        

        # --- Networks ---
        self.policy_net = AttentionDQNNetwork(n_features, seq_len, action_size, embed_dim, num_heads).to(device)
        self.target_net = AttentionDQNNetwork(n_features, seq_len, action_size, embed_dim, num_heads).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # --- Optimizer ---
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)

        self.steps_done = 0  # Counter for epsilon decay

        # --- Logging Attributes ---
        self.step_losses = []       # To store loss values between episode logs
        self.log_writer = None      # CSV writer object
        self.log_file = None        # File handle for the log
        self.log_file_path = None   # Full path to the log file
        self.log_header_written = False

    def select_action(self, state_tuple):
        """Selects an action using an epsilon-greedy policy based on state sequence and mask.

        Args:
            state_tuple (tuple): A tuple containing (state_sequence, state_mask).
                                 state_sequence shape: (1, n_features, seq_len)
                                 state_mask shape: (1, seq_len)

        Returns:
            torch.Tensor: The selected action tensor (shape [1, 1]).
        """
        state_sequence, state_mask = state_tuple # Unpack the tuple
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        if sample > eps_threshold:
            # Exploitation: Choose the best action from the policy network
            with torch.no_grad():
                # Pass both sequence and mask to the network
                q_values = self.policy_net(state_sequence.to(device), state_mask.to(device))
                # .max(1)[1] -> index of the max Q-value (the best action)
                # .view(1, 1) -> reshape for compatibility
                action = q_values.max(1)[1].view(1, 1)
            # print(f"Action selected: {action.item()} with epsilon: {eps_threshold:.4f}, and sample: {sample:.4f}")
        else:
            # Exploration: Choose a random action
            action = torch.tensor([[random.randrange(self.action_size)]], device=device, dtype=torch.long)
            # print(f"Random action selected: {action.item()} with epsilon: {eps_threshold:.4f}, and sample: {sample:.4f}")

        return action

    def store_experience(self, state_tuple, action, next_state_tuple, reward, done):
        """Stores a single experience tuple (including masks) in the replay memory.

        Args:
            state_tuple (tuple): (state_sequence, state_mask).
            action (torch.Tensor): The action taken (shape [1, 1]).
            next_state_tuple (tuple or None): (next_state_sequence, next_state_mask) or None if terminal.
            reward (torch.Tensor): The reward received (shape [1]).
            done (bool): True if the episode terminated.
        """
        # Unpack tuples or handle None case
        state_sequence, state_mask = state_tuple
        if next_state_tuple is not None:
            next_state_sequence, next_state_mask = next_state_tuple
        else:
            # Use None placeholders if next state is terminal
            # Note: The memory needs to handle these None values during sampling/batching
            next_state_sequence, next_state_mask = None, None

        done_tensor = torch.tensor([done], device=device, dtype=torch.bool)
        # Push all components including masks
        self.memory.push(state_sequence, state_mask, action, next_state_sequence, next_state_mask, reward, done_tensor)


    def optimize_model(self):
        """Performs one step of optimization and stores the loss."""
        if len(self.memory) < self.batch_size:
            return  # Not enough samples

        # ... (sampling and batch preparation logic remains the same) ...
        transitions = self.memory.sample(self.batch_size)
        batch = Att_Transition(*zip(*transitions))
        # ... prepare state_batch, mask_batch, action_batch, reward_batch, non_final masks etc. ...
        non_final_mask_indices = [i for i, ns in enumerate(batch.next_state) if ns is not None]
        state_batch = torch.cat([s for s in batch.state if s is not None])
        state_mask_batch = torch.cat([m for m in batch.state_mask if m is not None])
        if non_final_mask_indices:
            non_final_next_states = torch.cat([batch.next_state[i] for i in non_final_mask_indices])
            non_final_next_state_masks = torch.cat([batch.next_state_mask[i] for i in non_final_mask_indices])
        else:
            non_final_next_states = None
            non_final_next_state_masks = None
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_mask = torch.tensor(tuple(map(lambda ns: ns is not None, batch.next_state)), device=device, dtype=torch.bool)

        # --- Q-Value Calculation ---
        state_action_values = self.policy_net(state_batch, state_mask_batch).gather(1, action_batch)

        # --- Target Q-Value Calculation ---
        next_state_values = torch.zeros(self.batch_size, device=device)
        if non_final_next_states is not None and non_final_next_state_masks is not None:
             with torch.no_grad():
                 max_target_q = self.target_net(non_final_next_states, non_final_next_state_masks).max(1)[0]
                 next_state_values[non_final_mask] = max_target_q
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # --- Compute Loss ---
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # --- Store Loss ---
        self.step_losses.append(loss.item()) # Append scalar loss value

        # --- Optimize ---
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()


    def setup_logging(self, model_name='AttentionDQN', base_dir="Metrics"):
        """Creates logging directory and initializes the CSV log file."""
        try:
            datetime_str = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.log_dir = os.path.join(base_dir, model_name, datetime_str) # Store log dir path
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)

            self.log_file_path = os.path.join(self.log_dir, "training_log.csv")
            # Open file in append mode ('a') to allow continuation if needed, use newline='' for csv
            self.log_file = open(self.log_file_path, mode='a', newline='', encoding='utf-8')
            self.log_writer = csv.writer(self.log_file)

            # Write header only if the file is newly created or empty
            self.log_file.seek(0, os.SEEK_END) # Go to end of file
            if self.log_file.tell() == 0: # Check if file is empty
                 header = ["Episode", "Steps", "TotalReward", "AverageLoss", "Epsilon", "GoalAchieved"]
                 self.log_writer.writerow(header)
                 self.log_header_written = True
                 print(f"Logging initialized at: {self.log_file_path}")

        except Exception as e:
            print(f"Error setting up logging: {e}")
            if self.log_file:
                self.log_file.close() # Ensure file is closed on error
            self.log_writer = None
            self.log_file = None

    # --- New Method: Log Episode Data ---
    def log_episode_data(self, episode, steps, avg_reward, goal_achieved):
        """Calculates metrics and writes a row to the log file for the completed episode."""
        if self.log_writer is None:
            print("Warning: Log writer not initialized. Cannot log episode data.")
            return

        # Calculate average loss for the steps in this episode
        avg_loss = np.mean(self.step_losses) if self.step_losses else 0.0
        self.step_losses = [] # Reset losses for the next episode

        # Calculate current epsilon
        current_epsilon = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)

        try:
            row = [episode, steps, f"{avg_reward:.4f}", f"{avg_loss:.6f}", f"{current_epsilon:.4f}", goal_achieved]
            self.log_writer.writerow(row)
            # Optionally flush data to disk periodically or on each write
            # self.log_file.flush()
        except Exception as e:
            print(f"Error writing log data: {e}")

    # --- New Method: Close Log File ---
    def close_log(self):
        """Closes the log file handle."""
        if self.log_file:
            try:
                self.log_file.close()
                print(f"Log file closed: {self.log_file_path}")
                self.log_file = None
                self.log_writer = None
            except Exception as e:
                print(f"Error closing log file: {e}")

    def update_target_net(self, soft_update=True):
        """Updates the target network weights (same logic as before)."""
        if soft_update and self.tau > 0:
            target_net_state_dict = self.target_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + \
                                             target_net_state_dict[key] * (1 - self.tau)
            self.target_net.load_state_dict(target_net_state_dict)
        elif not soft_update:
            # Hard update (less common if tau > 0)
            self.target_net.load_state_dict(self.policy_net.state_dict())

    # --- save_weights_and_distances and load_weights ---
    # These methods should generally work as they operate on the model's state_dict.
    # You might want to change the default model_name argument or the save directory structure.

    def save_weights_and_distances(self, all_distances_acquisitions, filename="agent_policy_weights.pth"):
        """Saves the weights of the policy network and distance data."""
        base_dir = f'Metrics/{MODEL_NAME}/'
        if not os.path.exists(base_dir):
                print(f"Warning: Base directory {base_dir} not found. Cannot load weights.")
                return False

        all_subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if not all_subdirs:
            print(f"Warning: No subdirectories found in {base_dir}. Cannot load weights.")
            return False

        latest_subdir = max(all_subdirs) # Assumes subdirs are sortable timestamps
        models_dir = os.path.join(base_dir, latest_subdir)
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        # Save distance data (assuming save_distance_data is available)
        # You might need to import or pass this function if it's in utils
        try:
            # Example: Assuming save_distance_data is imported or defined globally
            from utils import save_distance_data
            save_distance_data(all_distances_acquisitions, save_folder=models_dir)
        except ImportError:
            print("Warning: 'save_distance_data' function not found. Skipping distance saving.")
        except Exception as e:
            print(f"Error saving distance data: {e}")


        # Save weights
        weights_filename = os.path.join(models_dir, filename)
        try:
            torch.save(self.policy_net.state_dict(), weights_filename)
            print(f"Agent policy weights saved to {weights_filename}")
        except Exception as e:
            print(f"Error saving agent weights: {e}")

    def load_weights(self):
        # ... (load_weights implementation - check paths match Metrics/ModelName/Timestamp/) ...
        # Make sure it finds the correct directory structure
        try:
            base_dir = f'Metrics/{MODEL_NAME}/'
            if not os.path.exists(base_dir):
                 print(f"Warning: Base directory {base_dir} not found. Cannot load weights.")
                 return False

            all_subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
            if not all_subdirs:
                print(f"Warning: No subdirectories found in {base_dir}. Cannot load weights.")
                return False

            latest_subdir = max(all_subdirs) # Assumes subdirs are sortable timestamps
            path = os.path.join(base_dir, latest_subdir)
            filename = os.path.join(path, "agent_policy_weights.pth") # Standard weight file name

            if not os.path.exists(filename):
                print(f"Warning: Weight file not found at {filename}. Starting with initial weights.")
                return False

            state_dict = torch.load(filename, map_location=device)
            self.policy_net.load_state_dict(state_dict)
            self.target_net.load_state_dict(self.policy_net.state_dict()) # Sync target net
            self.policy_net.eval()
            self.target_net.eval()
            print(f"Agent policy weights loaded from {filename}")
            # Optionally setup logging here if loading a pre-trained model to continue logging
            # self.setup_logging(model_name=model_name) # Needs careful handling if continuing runs
            return True
        except FileNotFoundError:
             print(f"Warning: Directory or file not found during weight loading for {MODEL_NAME}. Starting with initial weights.")
             return False
        except Exception as e:
            print(f"Error loading agent weights for {MODEL_NAME}: {e}")
            return False
    