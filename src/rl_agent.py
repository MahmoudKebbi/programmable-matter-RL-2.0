import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, List, Tuple
import gymnasium as gym


class PolicyNetwork(nn.Module):
    """Neural network for policy (actor)"""

    def __init__(self, n_blocks: int, grid_size: tuple, hidden_size: int = 128):
        super(PolicyNetwork, self).__init__()
        n, m = grid_size

        # Process grid with CNN
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Calculate CNN output size
        cnn_output_size = 16 * n * m

        # Process block and target positions
        self.block_embedding = nn.Sequential(
            nn.Linear(3 * n_blocks, hidden_size), nn.ReLU()
        )
        self.target_embedding = nn.Sequential(
            nn.Linear(3 * n_blocks, hidden_size), nn.ReLU()
        )

        # Combined features processing
        self.combined = nn.Sequential(
            nn.Linear(cnn_output_size + hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Output heads
        self.block_selector = nn.Linear(hidden_size, n_blocks)
        self.direction_selector = nn.Linear(hidden_size, 9)  # 8 directions + stay

    def forward(self, state):
        # Process grid with CNN
        grid = state["grid"].unsqueeze(1).float()  # Add channel dimension
        grid_features = self.conv(grid)

        # Process blocks and targets
        blocks = state["blocks"].reshape(state["blocks"].size(0), -1).float()
        targets = state["targets"].reshape(state["targets"].size(0), -1).float()

        block_features = self.block_embedding(blocks)
        target_features = self.target_embedding(targets)

        # Combine features
        combined = torch.cat([grid_features, block_features, target_features], dim=1)
        features = self.combined(combined)

        # Output action probabilities
        block_logits = self.block_selector(features)
        direction_logits = self.direction_selector(features)

        return block_logits, direction_logits


class ValueNetwork(nn.Module):
    """Neural network for value function (critic)"""

    def __init__(self, n_blocks: int, grid_size: tuple, hidden_size: int = 128):
        super(ValueNetwork, self).__init__()
        n, m = grid_size

        # Similar architecture to policy network but with value output
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        cnn_output_size = 16 * n * m

        self.block_embedding = nn.Sequential(
            nn.Linear(3 * n_blocks, hidden_size), nn.ReLU()
        )
        self.target_embedding = nn.Sequential(
            nn.Linear(3 * n_blocks, hidden_size), nn.ReLU()
        )

        self.combined = nn.Sequential(
            nn.Linear(cnn_output_size + hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, state):
        # Process grid with CNN
        grid = state["grid"].unsqueeze(1).float()
        grid_features = self.conv(grid)

        # Process blocks and targets
        blocks = state["blocks"].reshape(state["blocks"].size(0), -1).float()
        targets = state["targets"].reshape(state["targets"].size(0), -1).float()

        block_features = self.block_embedding(blocks)
        target_features = self.target_embedding(targets)

        # Combine features
        combined = torch.cat([grid_features, block_features, target_features], dim=1)
        features = self.combined(combined)

        # Output value
        value = self.value_head(features)

        return value


class PPOAgent:
    """PPO agent for programmable matter control"""

    def __init__(
        self,
        env: gym.Env,
        actor_lr: float = 0.0003,
        critic_lr: float = 0.001,
        gamma: float = 0.99,
        clip_ratio: float = 0.2,
        hidden_size: int = 128,
    ):

        self.env = env
        self.gamma = gamma
        self.clip_ratio = clip_ratio

        n_blocks = len(env.reset()["blocks"])
        grid_shape = env.observation_space["grid"].shape

        # Initialize networks
        self.actor = PolicyNetwork(n_blocks, grid_shape, hidden_size)
        self.critic = ValueNetwork(n_blocks, grid_shape, hidden_size)

        # Setup optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # For storing trajectory
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def select_action(self, state, training=True):
        """Select action based on current policy"""
        # Convert numpy arrays to tensors
        state_tensors = {
            "grid": torch.FloatTensor(state["grid"]).unsqueeze(0),
            "blocks": torch.FloatTensor(state["blocks"]).unsqueeze(0),
            "targets": torch.FloatTensor(state["targets"]).unsqueeze(0),
        }

        with torch.no_grad():
            block_logits, direction_logits = self.actor(state_tensors)
            block_probs = F.softmax(block_logits, dim=1)
            direction_probs = F.softmax(direction_logits, dim=1)

            # Sample actions
            if training:
                block_dist = Categorical(block_probs)
                direction_dist = Categorical(direction_probs)

                block_id = block_dist.sample().item()
                direction = direction_dist.sample().item()

                block_log_prob = block_dist.log_prob(torch.tensor(block_id))
                direction_log_prob = direction_dist.log_prob(torch.tensor(direction))

                log_prob = block_log_prob + direction_log_prob
            else:
                # During evaluation, take most likely action
                block_id = torch.argmax(block_probs, dim=1).item()
                direction = torch.argmax(direction_probs, dim=1).item()
                log_prob = 0  # Not needed for evaluation

        action = {"block_id": block_id, "direction": direction}

        return action, log_prob

    def compute_value(self, state):
        """Compute state value using critic network"""
        state_tensors = {
            "grid": torch.FloatTensor(state["grid"]).unsqueeze(0),
            "blocks": torch.FloatTensor(state["blocks"]).unsqueeze(0),
            "targets": torch.FloatTensor(state["targets"]).unsqueeze(0),
        }

        with torch.no_grad():
            value = self.critic(state_tensors).item()

        return value

    def store_transition(self, state, action, log_prob, reward, done, next_state):
        """Store transition in memory"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)

        # Compute and store value
        value = self.compute_value(state)
        self.values.append(value)

    def compute_returns(self, final_value):
        """Compute returns with GAE"""
        returns = []
        advantages = []
        gae = 0

        # Add final value for terminal state
        self.values.append(final_value)

        for i in reversed(range(len(self.rewards))):
            if self.dones[i]:
                delta = self.rewards[i] - self.values[i]
                gae = delta
            else:
                delta = (
                    self.rewards[i] + self.gamma * self.values[i + 1] - self.values[i]
                )
                gae = delta + self.gamma * gae

            returns.insert(0, gae + self.values[i])
            advantages.insert(0, gae)

        return returns, advantages

    def update_policy(self, batch_size=64, epochs=10):
        """Update policy and value networks using PPO"""
        # Calculate returns and advantages
        if self.states:
            # Get value of final state
            if self.dones[-1]:
                final_value = 0
            else:
                final_value = self.compute_value(self.states[-1])

            returns, advantages = self.compute_returns(final_value)

            # Convert to tensors
            returns = torch.FloatTensor(returns)
            advantages = torch.FloatTensor(advantages)
            old_log_probs = torch.FloatTensor(self.log_probs)

            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # Mini-batch training
            indices = np.arange(len(self.states))

            for _ in range(epochs):
                np.random.shuffle(indices)
                for start in range(0, len(indices), batch_size):
                    end = start + batch_size
                    if end > len(indices):
                        end = len(indices)

                    batch_indices = indices[start:end]

                    # Prepare batch data
                    batch_states = [self.states[i] for i in batch_indices]
                    batch_actions = [self.actions[i] for i in batch_indices]
                    batch_returns = returns[batch_indices]
                    batch_advantages = advantages[batch_indices]
                    batch_old_log_probs = old_log_probs[batch_indices]

                    # Convert batch states to tensors
                    batch_state_tensors = {
                        "grid": torch.FloatTensor(
                            np.stack([s["grid"] for s in batch_states])
                        ),
                        "blocks": torch.FloatTensor(
                            np.stack([s["blocks"] for s in batch_states])
                        ),
                        "targets": torch.FloatTensor(
                            np.stack([s["targets"] for s in batch_states])
                        ),
                    }

                    # Actor loss
                    block_logits, direction_logits = self.actor(batch_state_tensors)

                    # Compute new action probabilities
                    new_log_probs = []
                    for i, action in enumerate(batch_actions):
                        block_id = action["block_id"]
                        direction = action["direction"]

                        block_dist = Categorical(F.softmax(block_logits[i], dim=0))
                        direction_dist = Categorical(
                            F.softmax(direction_logits[i], dim=0)
                        )

                        block_log_prob = block_dist.log_prob(torch.tensor(block_id))
                        direction_log_prob = direction_dist.log_prob(
                            torch.tensor(direction)
                        )

                        new_log_probs.append(block_log_prob + direction_log_prob)

                    new_log_probs = torch.stack(new_log_probs)

                    # Compute ratio and clipped ratio
                    ratio = torch.exp(new_log_probs - batch_old_log_probs)
                    clipped_ratio = torch.clamp(
                        ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio
                    )

                    # Compute surrogate losses
                    surr1 = ratio * batch_advantages
                    surr2 = clipped_ratio * batch_advantages

                    actor_loss = -torch.min(surr1, surr2).mean()

                    # Value loss
                    values = self.critic(batch_state_tensors).squeeze()
                    value_loss = F.mse_loss(values, batch_returns)

                    # Update actor
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    self.actor_optimizer.step()

                    # Update critic
                    self.critic_optimizer.zero_grad()
                    value_loss.backward()
                    self.critic_optimizer.step()

            # Clear memory
            self.states = []
            self.actions = []
            self.log_probs = []
            self.rewards = []
            self.dones = []
            self.values = []

    def train(self, num_episodes=1000, max_steps=None, update_interval=2048):
        """Train the agent"""
        total_steps = 0
        best_reward = float("-inf")

        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            step = 0

            while not done:
                action, log_prob = self.select_action(state)
                next_state, reward, done, info = self.env.step(action)

                self.store_transition(state, action, log_prob, reward, done, next_state)

                state = next_state
                episode_reward += reward
                step += 1
                total_steps += 1

                # Update policy after collecting update_interval steps
                if total_steps % update_interval == 0:
                    self.update_policy()

                if max_steps and step >= max_steps:
                    break

            # If episode ended, update policy with remaining steps
            if len(self.states) > 0 and len(self.states) < update_interval:
                self.update_policy()

            # Log progress
            if episode % 10 == 0:
                print(f"Episode {episode}, Reward: {episode_reward:.2f}, Steps: {step}")

            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                self.save_model("best_model")

    def save_model(self, path):
        """Save model weights"""
        torch.save(
            {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()},
            f"{path}.pt",
        )

    def load_model(self, path):
        """Load model weights"""
        checkpoint = torch.load(f"{path}.pt")
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
