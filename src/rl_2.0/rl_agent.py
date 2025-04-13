import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from typing import Dict, List, Tuple
import gymnasium as gym


class FlexiblePolicyNetwork(nn.Module):
    def __init__(self, max_n_blocks, grid_size, hidden_size=128):
        super(FlexiblePolicyNetwork, self).__init__()

        # Initial embeddings for each block
        self.block_embedding = nn.Linear(3, hidden_size)
        self.target_embedding = nn.Linear(3, hidden_size)

        # Message passing layers
        self.message_passing = nn.ModuleList(
            [nn.Linear(hidden_size * 2, hidden_size) for _ in range(3)]
        )

        # Output layers
        self.block_query = nn.Linear(hidden_size, hidden_size)
        self.direction_head = nn.Linear(hidden_size, 9)

    def forward(self, state):
        # Get blocks and reshape to [batch_size, n_blocks, 3]
        batch_size = state["blocks"].size(0)
        blocks = state["blocks"].view(batch_size, -1, 3)
        targets = state["targets"].view(batch_size, -1, 3)
        n_blocks = blocks.size(1)

        # Initial embeddings
        block_features = self.block_embedding(blocks)  # [batch, n_blocks, hidden]
        target_features = self.target_embedding(targets)

        # Block selection is now attention-based
        query = self.block_query(torch.mean(block_features, dim=1, keepdim=True))
        block_scores = torch.bmm(query, block_features.transpose(1, 2)).squeeze(1)

        # Direction selection for each block
        direction_logits = self.direction_head(block_features)  # [batch, n_blocks, 9]

        return block_scores, direction_logits


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
        # Store all parameters as instance attributes
        self.env = env
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        # Add base LR attributes that are referenced elsewhere
        self.base_actor_lr = actor_lr
        self.base_critic_lr = critic_lr
        self.hidden_size = hidden_size

        n_blocks = len(env.reset()["blocks"])
        grid_shape = env.observation_space["grid"].shape

        # Initialize networks
        self.actor = FlexiblePolicyNetwork(n_blocks, grid_shape, hidden_size)
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

        # For tracking progress
        self.episode_rewards = []
        self.training_step = 0
        self.episode_count = 0
        self.level_episodes = {}  # Track episodes per level

        # For curriculum-specific tracking
        self.current_level = 0
        self.exploration_phase = False

        # For tracking stuck state
        self.stuck_count = 0
        self.network_resets = 0

    def select_action(self, state, training=True, boost_exploration=False):
        """Select action based on current policy"""
        # Convert numpy arrays to tensors
        state_tensors = {
            "grid": torch.FloatTensor(state["grid"]).unsqueeze(0),
            "blocks": torch.FloatTensor(state["blocks"]).unsqueeze(0),
            "targets": torch.FloatTensor(state["targets"]).unsqueeze(0),
        }

        with torch.no_grad():
            block_logits, direction_logits = self.actor(state_tensors)

            # Add temperature scaling for exploration
            if training:
                # Base temperature calculation
                self.training_step += 1

                # Default temperature scaling based on training progress
                base_temp = max(0.5, 2.0 - self.episode_count / 2000)

                # Apply boost if requested (for specific levels)
                temperature = 3.0 if boost_exploration else base_temp

                block_logits = block_logits / temperature

                # For FlexiblePolicyNetwork, direction_logits has shape [batch, n_blocks, 9]
                if len(direction_logits.shape) == 3:
                    direction_logits = direction_logits / temperature
                else:
                    # Legacy format support
                    direction_logits = direction_logits / temperature

            block_probs = F.softmax(block_logits, dim=1)

            # Sample block first
            if training:
                block_dist = Categorical(block_probs)
                block_id = block_dist.sample().item()
                block_log_prob = block_dist.log_prob(torch.tensor(block_id))
            else:
                block_id = torch.argmax(block_probs, dim=1).item()
                block_log_prob = 0

            # Now handle direction based on network architecture
            if len(direction_logits.shape) == 3:
                # FlexiblePolicyNetwork: Extract direction logits for the selected block
                # Shape: [batch, n_blocks, 9] -> [batch, 9]
                selected_direction_logits = direction_logits[0, block_id]
                direction_probs = F.softmax(selected_direction_logits, dim=0)

                if training:
                    direction_dist = Categorical(direction_probs)
                    direction = direction_dist.sample().item()
                    direction_log_prob = direction_dist.log_prob(
                        torch.tensor(direction)
                    )
                else:
                    direction = torch.argmax(direction_probs).item()
                    direction_log_prob = 0
            else:
                # Legacy network format
                direction_probs = F.softmax(direction_logits, dim=1)

                if training:
                    direction_dist = Categorical(direction_probs)
                    direction = direction_dist.sample().item()
                    direction_log_prob = direction_dist.log_prob(
                        torch.tensor(direction)
                    )
                else:
                    direction = torch.argmax(direction_probs, dim=1).item()
                    direction_log_prob = 0

            if training:
                log_prob = block_log_prob + direction_log_prob
            else:
                log_prob = 0

        action = {"block_id": block_id, "direction": direction}
        return action, log_prob

    def select_action_with_temp(self, state, temperature=1.0):
        """Select action with explicit temperature control"""
        # Convert numpy arrays to tensors
        state_tensors = {
            "grid": torch.FloatTensor(state["grid"]).unsqueeze(0),
            "blocks": torch.FloatTensor(state["blocks"]).unsqueeze(0),
            "targets": torch.FloatTensor(state["targets"]).unsqueeze(0),
        }

        with torch.no_grad():
            block_logits, direction_logits = self.actor(state_tensors)

            # Apply temperature scaling - higher temp = more exploration
            block_logits = block_logits / temperature

            # Handle different network architectures
            if len(direction_logits.shape) == 3:
                direction_logits = direction_logits / temperature
            else:
                direction_logits = direction_logits / temperature

            # Convert to probabilities
            block_probs = F.softmax(block_logits, dim=1)

            # Sample block
            block_dist = Categorical(block_probs)
            block_id = block_dist.sample().item()
            block_log_prob = block_dist.log_prob(torch.tensor(block_id))

            # Sample direction based on network architecture
            if len(direction_logits.shape) == 3:
                # Extract direction logits for the selected block
                selected_direction_logits = direction_logits[0, block_id]
                direction_probs = F.softmax(selected_direction_logits, dim=0)

                direction_dist = Categorical(direction_probs)
                direction = direction_dist.sample().item()
                direction_log_prob = direction_dist.log_prob(torch.tensor(direction))
            else:
                direction_probs = F.softmax(direction_logits, dim=1)
                direction_dist = Categorical(direction_probs)
                direction = direction_dist.sample().item()
                direction_log_prob = direction_dist.log_prob(torch.tensor(direction))

            log_prob = block_log_prob + direction_log_prob

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

                    # Ensure batch size is at least 1
                    if len(batch_indices) < 1:
                        continue

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

                        # Handle block logits
                        block_probs = F.softmax(block_logits[i], dim=0)
                        block_dist = Categorical(block_probs)
                        block_log_prob = block_dist.log_prob(torch.tensor(block_id))

                        # Handle direction logits based on network architecture
                        if len(direction_logits.shape) == 3:
                            # FlexiblePolicyNetwork format: [batch, n_blocks, 9]
                            selected_direction_logits = direction_logits[i, block_id]
                            direction_probs = F.softmax(
                                selected_direction_logits, dim=0
                            )
                        else:
                            # Legacy format: [batch, 9]
                            direction_probs = F.softmax(direction_logits[i], dim=0)

                        direction_dist = Categorical(direction_probs)
                        direction_log_prob = direction_dist.log_prob(
                            torch.tensor(direction)
                        )

                        new_log_probs.append(block_log_prob + direction_log_prob)

                    # Convert to tensor (make sure there's at least one element)
                    if new_log_probs:
                        new_log_probs = torch.stack(new_log_probs)

                        # Match the dimensions
                        if batch_old_log_probs.shape != new_log_probs.shape:
                            if batch_old_log_probs.numel() == new_log_probs.numel():
                                # Reshape to match if total elements are the same
                                batch_old_log_probs = batch_old_log_probs.view_as(
                                    new_log_probs
                                )
                            else:
                                print(
                                    f"Warning: Shape mismatch - old: {batch_old_log_probs.shape}, new: {new_log_probs.shape}"
                                )
                                continue  # Skip this batch

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
            self.clear_memory()

    def train(self, num_episodes=1000, max_steps=None, update_interval=2048):
        """Train the agent"""
        total_steps = 0
        best_reward = float("-inf")

        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            step = 0

            self.episode_count = episode

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

            # Track episode reward
            self.episode_rewards.append(episode_reward)

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

    def adapt_to_env(self, new_env):
        """Adapt the agent to a new environment (e.g. when curriculum changes)"""
        # Update environment reference
        self.env = new_env

        # Get block count from new environment to check if architecture needs updating
        n_blocks = len(new_env.reset()["blocks"])

        # Determine current block count from actor network
        current_n_blocks = 0
        if hasattr(self.actor, "block_selector"):
            current_n_blocks = self.actor.block_selector.out_features
        else:
            # Try to infer from first layer weights if direct attribute not available
            for name, param in self.actor.named_parameters():
                if "block_selector" in name and "weight" in name:
                    current_n_blocks = param.shape[0]
                    break

        # If block count has changed, need to rebuild networks
        if n_blocks != current_n_blocks:
            print(f"Rebuilding networks for {n_blocks} blocks (was {current_n_blocks})")
            grid_shape = new_env.observation_space["grid"].shape

            # Save old states
            actor_state = self.actor.state_dict()
            critic_state = self.critic.state_dict()

            # Create new networks
            self.actor = FlexiblePolicyNetwork(n_blocks, grid_shape, self.hidden_size)
            self.critic = ValueNetwork(n_blocks, grid_shape, self.hidden_size)

            # Try to load compatible parameters
            try:
                self.actor.load_state_dict(actor_state, strict=False)
                self.critic.load_state_dict(critic_state, strict=False)
            except Exception as e:
                print(f"Parameter loading failed - using new networks. Error: {e}")

            # Reset optimizers
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
            self.critic_optimizer = optim.Adam(
                self.critic.parameters(), lr=self.critic_lr
            )

        # Clear memory buffers
        self.clear_memory()

    def clear_memory(self):
        """Clear the agent's memory buffers"""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []

    def boost_for_level7(self):
        """Apply specialized modifications for level 7"""
        print("🔄 Applying specialized boost for level 7...")

        # 1. Temporarily increase learning rates
        for param_group in self.actor_optimizer.param_groups:
            param_group["lr"] = self.base_actor_lr * 2.0
        for param_group in self.critic_optimizer.param_groups:
            param_group["lr"] = self.base_critic_lr * 2.0

        # 2. Reinitialize final layers with different initialization
        nn.init.orthogonal_(self.actor.block_selector.weight, gain=1.4)
        nn.init.orthogonal_(self.actor.direction_selector.weight, gain=1.4)

        # 3. Clear memory to start fresh
        self.clear_memory()

        # 4. Set exploration phase flag
        self.exploration_phase = True

    def get_exploration_temperature(self, level, stuck_episodes):
        """Calculate appropriate temperature based on level and stuck episodes"""
        base_temp = 1.0  # Default temperature

        if level == 7:
            # The longer we're stuck, the more we explore
            if stuck_episodes < 10:
                return 1.5  # Mild boost
            elif stuck_episodes < 30:
                return 2.0  # Medium boost
            else:
                # Oscillating high temperature to prevent getting stuck in cycles
                return 2.0 + np.sin(stuck_episodes / 5) * 1.0  # 1.0 to 3.0

        return base_temp

    def reset_exploration(self):
        """Reset exploration when advancing to a new level"""
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=self.base_actor_lr
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=self.base_critic_lr
        )

    def boost_exploration(self, stuck_episodes):
        """Dynamically boost exploration based on stuck episodes"""
        # Calculate appropriate temperature based on stuck episodes
        if stuck_episodes < 10:
            return 1.2  # Mild boost
        elif stuck_episodes < 30:
            return 1.5  # Medium boost
        elif stuck_episodes < 60:
            return 2.0  # Strong boost
        else:
            # Oscillating temperature to break patterns
            phase = (stuck_episodes % 20) / 20.0
            return 2.0 + 1.0 * np.sin(phase * 2 * np.pi)

    def select_action_for_stuck_level(self, state, stuck_episodes):
        """Special action selection with dynamic exploration when stuck"""
        # Determine temperature based on stuck duration
        temperature = self.boost_exploration(stuck_episodes)

        # Use temperature-scaled action selection
        return self.select_action_with_temp(state, temperature)

    def reset_network_when_stuck(self, level, stuck_episodes):
        """Progressively reset network parts when stuck for too long"""
        # Only apply resets at certain thresholds
        if stuck_episodes in [50, 80, 110]:
            print(
                f"🔄 Resetting network components for level {level} after {stuck_episodes} episodes"
            )

            # Check if we're using FlexiblePolicyNetwork
            is_flexible_network = isinstance(self.actor, FlexiblePolicyNetwork)

            # Determine which components to reset based on stuck duration
            if stuck_episodes == 50:
                # First reset: Just reinitialize output layers
                if is_flexible_network:
                    nn.init.orthogonal_(self.actor.block_query.weight, gain=1.4)
                    nn.init.orthogonal_(self.actor.direction_head.weight, gain=1.4)
                else:
                    nn.init.orthogonal_(self.actor.block_selector.weight, gain=1.4)
                    nn.init.orthogonal_(self.actor.direction_selector.weight, gain=1.4)

            elif stuck_episodes == 80:
                # Second reset: Reset output layers and optimizer with higher learning rate
                if is_flexible_network:
                    nn.init.orthogonal_(self.actor.block_query.weight, gain=1.6)
                    nn.init.orthogonal_(self.actor.direction_head.weight, gain=1.6)
                else:
                    nn.init.orthogonal_(self.actor.block_selector.weight, gain=1.6)
                    nn.init.orthogonal_(self.actor.direction_selector.weight, gain=1.6)

                # Reset optimizers
                self.actor_optimizer = optim.Adam(
                    self.actor.parameters(), lr=self.base_actor_lr * 1.5
                )
                self.critic_optimizer = optim.Adam(
                    self.critic.parameters(), lr=self.base_critic_lr * 1.5
                )

            elif stuck_episodes == 110:
                # Third reset: More drastic, reset multiple network layers
                for name, module in self.actor.named_modules():
                    if isinstance(module, nn.Linear):
                        nn.init.orthogonal_(module.weight, gain=1.0)

                # Fresh optimizers with higher learning rates
                self.actor_optimizer = optim.Adam(
                    self.actor.parameters(), lr=self.base_actor_lr * 2.0
                )
                self.critic_optimizer = optim.Adam(
                    self.critic.parameters(), lr=self.base_critic_lr * 2.0
                )

            # Clear memory buffers
            self.clear_memory()
            self.network_resets += 1
            return True

        return False

    def apply_learning_rate_cycling(self, stuck_episodes):
        """Apply cycling learning rates to escape local optima"""
        if stuck_episodes > 20:
            # Calculate cycle phase based on stuck episodes
            cycle_phase = (stuck_episodes % 40) / 40.0

            # Sinusoidal learning rate schedule between 0.5x and 2x base rate
            lr_multiplier = 0.5 + 1.5 * (0.5 + 0.5 * np.sin(cycle_phase * 2 * np.pi))

            # Apply to actor optimizer
            for param_group in self.actor_optimizer.param_groups:
                param_group["lr"] = self.base_actor_lr * lr_multiplier

            # Apply to critic optimizer
            for param_group in self.critic_optimizer.param_groups:
                param_group["lr"] = self.base_critic_lr * lr_multiplier

            # Only log occasionally to avoid spamming
            if stuck_episodes % 10 == 0:
                print(
                    f"📊 LR cycling: multiplier={lr_multiplier:.2f}, actor_lr={self.base_actor_lr * lr_multiplier:.6f}"
                )
