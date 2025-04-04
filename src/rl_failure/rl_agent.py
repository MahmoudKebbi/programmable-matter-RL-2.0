import os
import sys

# REVIEW: More robust path handling
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # Add current dir
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # Add parent dir

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import deque, namedtuple
import os

# Set up device for GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the experience tuple structure
Experience = namedtuple(
    "Experience", ("state", "action", "reward", "next_state", "done")
)


class ReplayBuffer:
    """Standard Experience Replay buffer."""

    # REVIEW: Kept standard buffer as an option

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        # REVIEW: Ensure data types are consistent upon adding if needed, or handle in sample
        experience = Experience(
            np.float32(state),
            int(action),
            float(reward),
            np.float32(next_state),
            bool(done),
        )
        self.buffer.append(experience)

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)

        # Convert list of tuples to tuple of lists, then to tensors
        states = (
            torch.from_numpy(np.vstack([e.state for e in experiences]))
            .float()
            .to(device)
        )
        actions = (
            torch.from_numpy(np.vstack([e.action for e in experiences]))
            .long()
            .to(device)
        )
        rewards = (
            torch.from_numpy(np.vstack([e.reward for e in experiences]))
            .float()
            .to(device)
        )
        next_states = (
            torch.from_numpy(np.vstack([e.next_state for e in experiences]))
            .float()
            .to(device)
        )
        # Convert boolean done flags to float tensor (0.0 or 1.0)
        dones = (
            torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8))
            .float()
            .to(device)
        )

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """Prioritized Replay Buffer (PER)."""

    # REVIEW: Kept PER implementation

    def __init__(
        self, buffer_size, batch_size, alpha=0.6, beta=0.4, beta_increment=0.001
    ):
        self.buffer = deque(maxlen=buffer_size)
        self.priorities = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        self.epsilon = 1e-5  # Small constant to ensure non-zero priority

    def add(self, state, action, reward, next_state, done):
        """Add experience with maximum priority."""
        experience = Experience(
            np.float32(state),
            int(action),
            float(reward),
            np.float32(next_state),
            bool(done),
        )
        self.buffer.append(experience)
        self.priorities.append(self.max_priority)  # Add with max priority initially

    def sample(self):
        """Sample a batch of experiences based on priorities."""
        if len(self.buffer) < self.batch_size:
            return None, None, None, None, None, None, None  # Not enough samples yet

        # Calculate sampling probabilities (P(i) = p_i^alpha / sum(p_k^alpha))
        priorities = np.array(self.priorities)
        probabilities = priorities**self.alpha
        prob_sum = probabilities.sum()
        if (
            prob_sum == 0
        ):  # Handle edge case where all priorities might be zero initially
            probabilities = np.ones_like(priorities) / len(priorities)
        else:
            probabilities /= prob_sum

        # Sample indices based on probabilities
        indices = np.random.choice(
            len(self.buffer), self.batch_size, p=probabilities, replace=True
        )  # Allow replacement for PER
        experiences = [self.buffer[idx] for idx in indices]

        # Calculate importance sampling (IS) weights (w_i = (N * P(i))^-beta / max(w_j))
        total_samples = len(self.buffer)
        weights = (total_samples * probabilities[indices]) ** -self.beta
        weights /= weights.max()  # Normalize weights
        weights = (
            torch.from_numpy(weights).float().to(device).unsqueeze(1)
        )  # Add dimension for broadcasting

        # Update beta towards 1.0
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Convert sampled experiences to tensors
        states = (
            torch.from_numpy(np.vstack([e.state for e in experiences]))
            .float()
            .to(device)
        )
        actions = (
            torch.from_numpy(np.vstack([e.action for e in experiences]))
            .long()
            .to(device)
        )
        rewards = (
            torch.from_numpy(np.vstack([e.reward for e in experiences]))
            .float()
            .to(device)
        )
        next_states = (
            torch.from_numpy(np.vstack([e.next_state for e in experiences]))
            .float()
            .to(device)
        )
        dones = (
            torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8))
            .float()
            .to(device)
        )

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, td_errors):
        """Update priorities of sampled experiences."""
        # Ensure td_errors is numpy array
        if isinstance(td_errors, torch.Tensor):
            td_errors = td_errors.detach().cpu().numpy()

        priorities = (
            np.abs(td_errors) + self.epsilon
        )  # Add epsilon to ensure non-zero priority
        # REVIEW: Clip priorities to prevent extreme values if necessary, though max_priority tracking helps
        # priorities = np.clip(priorities, self.epsilon, 10.0)

        for idx, priority in zip(indices, priorities):
            # Check index bounds - important if buffer size changes or indices are stale
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority
                self.max_priority = max(self.max_priority, priority)
            else:
                print(f"Warning: Stale or invalid index {idx} in update_priorities.")

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """Standard Deep Q-Network."""

    # REVIEW: Removed BatchNorm1d for potentially better stability in RL.
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, action_size)
        print(f"Initialized QNetwork: state={state_size}, action={action_size}")

    def forward(self, state):
        # REVIEW: Ensure input is handled correctly (e.g., if batch size is 1)
        # The unsqueeze(0) in act() handles the single state case.
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class DuelingQNetwork(nn.Module):
    """Dueling Q-Network Architecture."""

    # REVIEW: Removed BatchNorm1d.
    def __init__(self, state_size, action_size):
        super(DuelingQNetwork, self).__init__()
        self.action_size = action_size

        # Shared feature layers
        self.feature_fc1 = nn.Linear(state_size, 256)
        self.feature_fc2 = nn.Linear(256, 128)

        # State value stream
        self.value_fc1 = nn.Linear(128, 64)
        self.value_fc2 = nn.Linear(64, 1)  # Outputs V(s)

        # Action advantage stream
        self.advantage_fc1 = nn.Linear(128, 64)
        self.advantage_fc2 = nn.Linear(64, action_size)  # Outputs A(s, a)
        print(f"Initialized DuelingQNetwork: state={state_size}, action={action_size}")

    def forward(self, state):
        # Shared feature extraction
        x = F.relu(self.feature_fc1(state))
        features = F.relu(self.feature_fc2(x))

        # Value stream
        value = F.relu(self.value_fc1(features))
        value = self.value_fc2(value)  # Shape: (batch_size, 1)

        # Advantage stream
        advantages = F.relu(self.advantage_fc1(features))
        advantages = self.advantage_fc2(advantages)  # Shape: (batch_size, action_size)

        # Combine value and advantages: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        # Subtracting the mean advantage ensures identifiability
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


class DQNAgent:
    """DQN Agent implementing several enhancements."""

    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=1e-4,  # REVIEW: Adjusted based on train.py
        gamma=0.99,
        buffer_size=100000,  # REVIEW: Adjusted based on train.py
        batch_size=256,  # REVIEW: Adjusted based on train.py
        update_every=4,  # Learn every C steps
        tau=1e-3,  # REVIEW: Using smaller tau for soft updates
        epsilon_start=1.0,
        epsilon_end=0.01,  # REVIEW: Adjusted based on train.py
        epsilon_decay=0.998,  # REVIEW: Adjusted based on train.py
        prioritized_replay=True,  # Use PER
        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_increment=0.0001,  # Slower beta increment might be more stable
        double_dqn=True,  # Use Double DQN
        dueling=True,  # Use Dueling Architecture
        gradient_clipping=1.0,  # REVIEW: Added gradient clipping value
        use_lr_scheduler=False,  # REVIEW: Made LR scheduler optional
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_every = update_every
        self.tau = tau
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.prioritized_replay = prioritized_replay
        self.double_dqn = double_dqn
        self.dueling = dueling
        self.gradient_clipping = gradient_clipping
        self.use_lr_scheduler = use_lr_scheduler

        # --- Q-Networks ---
        NetworkClass = DuelingQNetwork if self.dueling else QNetwork
        self.qnetwork_online = NetworkClass(state_size, action_size).to(device)
        self.qnetwork_target = NetworkClass(state_size, action_size).to(device)
        # Initialize target network with online network's weights
        self.qnetwork_target.load_state_dict(self.qnetwork_online.state_dict())
        self.qnetwork_target.eval()  # Target network is only for inference

        self.optimizer = optim.Adam(self.qnetwork_online.parameters(), lr=learning_rate)

        if self.use_lr_scheduler:
            # REVIEW: Scheduler parameters might need tuning
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=0.5,
                patience=50,
                verbose=True,  # Reduced patience
            )
        else:
            self.scheduler = None

        # --- Replay Buffer ---
        if self.prioritized_replay:
            self.memory = PrioritizedReplayBuffer(
                buffer_size,
                batch_size,
                alpha=per_alpha,
                beta=per_beta_start,
                beta_increment=per_beta_increment,
            )
            print("Using Prioritized Replay Buffer (PER)")
        else:
            self.memory = ReplayBuffer(buffer_size)
            print("Using Standard Replay Buffer")

        # Step counter for triggering learning
        self.t_step = 0

        # Metrics tracking
        self.losses = []
        self.rewards_history = []  # Note: This seems to be tracked in train.py
        self.epsilon_history = []

    def step(self, state, action, reward, next_state, done):
        """Add experience to buffer and trigger learning periodically."""
        self.memory.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get a batch and learn
            if len(self.memory) >= self.batch_size:
                if self.prioritized_replay:
                    experiences = self.memory.sample()
                    if experiences[0] is not None:  # Check if sampling was successful
                        indices, td_errors = self.learn_prioritized(experiences)
                        # Update priorities in the buffer
                        self.memory.update_priorities(indices, td_errors)
                else:
                    experiences = self.memory.sample(self.batch_size)
                    self.learn(experiences)

    def act(self, state, valid_actions_mask=None, eval_mode=False):
        """Choose an action using epsilon-greedy policy with action masking."""
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)

        self.qnetwork_online.eval()  # Set model to evaluation mode for inference
        with torch.no_grad():
            action_values = self.qnetwork_online(state_tensor)  # Get Q-values

            # Apply action masking: set Q-values of invalid actions to -infinity
            if valid_actions_mask is not None:
                # Ensure mask is a tensor on the correct device
                mask_tensor = torch.from_numpy(valid_actions_mask).float().to(device)
                # Add large negative value where mask is 0
                action_values = (
                    action_values + (mask_tensor - 1.0) * 1e9
                )  # Effectively -inf for masked actions

        self.qnetwork_online.train()  # Set model back to training mode

        # Epsilon-greedy action selection
        current_epsilon = (
            self.epsilon if not eval_mode else 0.0
        )  # No exploration in eval mode
        if random.random() < current_epsilon:
            # Exploration: Choose a random *valid* action
            if valid_actions_mask is not None:
                valid_indices = np.where(valid_actions_mask == 1)[0]
                if len(valid_indices) > 0:
                    action = np.random.choice(valid_indices)
                else:
                    # Fallback: if somehow no actions are valid, choose no-op (action 0)
                    # This case needs investigation if it occurs frequently.
                    print(
                        "Warning: No valid actions found during exploration! Choosing action 0."
                    )
                    action = 0
            else:
                # Explore randomly among all actions if no mask provided
                action = random.randrange(self.action_size)
        else:
            # Exploitation: Choose the action with the highest Q-value (masking applied above)
            action = torch.argmax(action_values).item()  # Get index of max Q-value

        return action

    def learn(self, experiences):
        """Learn from a batch of experiences (Standard or Double DQN)."""
        states, actions, rewards, next_states, dones = experiences

        # --- Calculate Target Q-values (Q_targets) ---
        self.qnetwork_target.eval()  # Ensure target network is in eval mode
        with torch.no_grad():  # No gradients needed for target calculations
            if self.double_dqn:
                # Double DQN: Select best action 'a_max' using online network
                online_next_actions = self.qnetwork_online(next_states).argmax(
                    dim=1, keepdim=True
                )
                # Evaluate Q(s', a_max) using target network
                Q_targets_next = self.qnetwork_target(next_states).gather(
                    1, online_next_actions
                )
            else:
                # Standard DQN: Select and evaluate best action using target network
                Q_targets_next = self.qnetwork_target(next_states).max(
                    dim=1, keepdim=True
                )[0]

            # Compute Q targets for current states: R + gamma * Q_target(s', a') * (1 - done)
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # --- Calculate Expected Q-values (Q_expected) ---
        # Get Q-values for the actions actually taken, using the online network
        Q_expected = self.qnetwork_online(states).gather(1, actions)

        # --- Compute Loss ---
        loss = F.mse_loss(Q_expected, Q_targets)
        self.losses.append(loss.item())  # Track loss

        # --- Optimize the Online Network ---
        self.optimizer.zero_grad()
        loss.backward()
        # Apply gradient clipping
        if self.gradient_clipping is not None:
            torch.nn.utils.clip_grad_norm_(
                self.qnetwork_online.parameters(), self.gradient_clipping
            )
        self.optimizer.step()

        # --- Update Target Network ---
        # REVIEW: Using only soft updates now. Removed the periodic hard update.
        self.soft_update(self.qnetwork_online, self.qnetwork_target, self.tau)

        # --- Update Epsilon ---
        self.update_epsilon()

    def learn_prioritized(self, experiences):
        """Learn from a batch of experiences using PER and IS weights."""
        states, actions, rewards, next_states, dones, indices, weights = experiences

        # --- Calculate Target Q-values (Q_targets) ---
        self.qnetwork_target.eval()  # Ensure target network is in eval mode
        self.qnetwork_online.eval()  # Also eval mode for selecting actions in Double DQN
        with torch.no_grad():
            if self.double_dqn:
                # Double DQN: Select action with online, evaluate with target
                online_next_actions = self.qnetwork_online(next_states).argmax(
                    dim=1, keepdim=True
                )
                Q_targets_next = self.qnetwork_target(next_states).gather(
                    1, online_next_actions
                )
            else:
                # Standard DQN: Select and evaluate with target
                Q_targets_next = self.qnetwork_target(next_states).max(
                    dim=1, keepdim=True
                )[0]

            # Compute TD targets
            Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        self.qnetwork_online.train()  # Set online network back to train mode

        # --- Calculate Expected Q-values (Q_expected) ---
        Q_expected = self.qnetwork_online(states).gather(1, actions)

        # --- Compute TD Errors for Priority Update ---
        # Calculate TD errors: |Q_target - Q_expected|
        td_errors = (Q_targets - Q_expected).abs()

        # --- Compute Weighted Loss ---
        # Loss = Mean( IS_weight * (Q_target - Q_expected)^2 )
        # Using Huber loss can be more robust to outliers than MSE
        loss = (
            weights * F.smooth_l1_loss(Q_expected, Q_targets, reduction="none")
        ).mean()  # Huber loss
        # loss = (weights * F.mse_loss(Q_expected, Q_targets, reduction='none')).mean() # MSE loss
        self.losses.append(loss.item())

        # --- Optimize the Online Network ---
        self.optimizer.zero_grad()
        loss.backward()
        if self.gradient_clipping is not None:
            torch.nn.utils.clip_grad_norm_(
                self.qnetwork_online.parameters(), self.gradient_clipping
            )
        self.optimizer.step()

        # --- Update Target Network ---
        self.soft_update(self.qnetwork_online, self.qnetwork_target, self.tau)

        # --- Update Epsilon ---
        self.update_epsilon()

        # Return indices and TD errors for priority updates
        return indices, td_errors.squeeze()  # Squeeze to remove extra dimension

    def soft_update(self, online_model, target_model, tau):
        """Soft update model parameters: target = tau*online + (1-tau)*target"""
        for target_param, online_param in zip(
            target_model.parameters(), online_model.parameters()
        ):
            target_param.data.copy_(
                tau * online_param.data + (1.0 - tau) * target_param.data
            )

    # REVIEW: Removed hard_update as we are using soft updates consistently.

    def update_epsilon(self):
        """Decay epsilon"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.epsilon_history.append(self.epsilon)

    def update_scheduler(self, metric_value):
        """Update LR scheduler if enabled"""
        if self.scheduler:
            self.scheduler.step(metric_value)
            # Optional: Print current LR
            # print(f"Current LR: {self.optimizer.param_groups[0]['lr']}")

    def save(self, filepath):
        """Save model state."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        save_dict = {
            "qnetwork_online_state_dict": self.qnetwork_online.state_dict(),
            "qnetwork_target_state_dict": self.qnetwork_target.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "t_step": self.t_step,
            # REVIEW: Save buffer state if needed for resuming training exactly
            # 'replay_buffer': self.memory.buffer,
            # 'replay_priorities': self.memory.priorities if self.prioritized_replay else None,
            # 'per_beta': self.memory.beta if self.prioritized_replay else None,
        }
        if self.scheduler:
            save_dict["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")

    def load(self, filepath, eval_mode=False):
        """Load model state."""
        if not os.path.exists(filepath):
            print(f"Error: No model found at {filepath}")
            return False

        try:
            checkpoint = torch.load(filepath, map_location=device)
            self.qnetwork_online.load_state_dict(
                checkpoint["qnetwork_online_state_dict"]
            )
            self.qnetwork_target.load_state_dict(
                checkpoint["qnetwork_target_state_dict"]
            )
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # Load epsilon, but set to end value if in eval mode
            self.epsilon = checkpoint["epsilon"] if not eval_mode else self.epsilon_end
            self.t_step = checkpoint.get("t_step", 0)  # Load t_step if available

            if self.scheduler and "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            # REVIEW: Load buffer state if saved and needed
            # if 'replay_buffer' in checkpoint and hasattr(self.memory, 'buffer'):
            #     self.memory.buffer = checkpoint['replay_buffer']
            #     if self.prioritized_replay and 'replay_priorities' in checkpoint:
            #         self.memory.priorities = checkpoint['replay_priorities']
            #         self.memory.beta = checkpoint.get('per_beta', self.memory.beta) # Load beta

            print(f"Model loaded successfully from {filepath}")
            if eval_mode:
                self.qnetwork_online.eval()
                self.qnetwork_target.eval()
                print("Agent set to evaluation mode.")
            else:
                self.qnetwork_online.train()
                # Target network stays in eval mode
                self.qnetwork_target.eval()
            return True

        except Exception as e:
            print(f"Error loading model from {filepath}: {e}")
            return False

    # REVIEW: Removed unused predict_batch and compute_n_step_returns methods.
