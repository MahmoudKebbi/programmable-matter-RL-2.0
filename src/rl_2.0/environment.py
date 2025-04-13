import numpy as np
import gym
from gym import spaces
from typing import List, Tuple, Dict, Optional
from collections import deque


class ProgrammableMatterEnv(gym.Env):
    """
    Programmable Matter Environment for Reinforcement Learning

    State: List of (id, x, y) tuples for each matter element
    Actions: List of (id, dx, dy) movements
    Reward: Based on progress toward goal configuration
    """

    def __init__(
        self,
        n: int,
        m: int,
        initial_positions: List[Tuple[int, int]],
        target_positions: List[Tuple[int, int]],
        obstacles: List[Tuple[int, int]] = None,
    ):

        super(ProgrammableMatterEnv, self).__init__()

        self.n = n  # Grid rows
        self.m = m  # Grid columns

        # Initialize block IDs and positions
        self.blocks = []
        for i, pos in enumerate(initial_positions):
            self.blocks.append((i, pos[0], pos[1]))  # (id, x, y)

        # Store target positions
        self.target_positions = []
        for i, pos in enumerate(target_positions):
            self.target_positions.append((i, pos[0], pos[1]))

        # Store obstacles
        self.obstacles = obstacles if obstacles else []

        # Grid representation (0: empty, 1: block, 2: obstacle)
        self.grid = np.zeros((n, m), dtype=int)
        self._update_grid()

        # Action and observation spaces
        # Action space: For each block, 9 possible actions (stay + 8 directions)
        self.action_space = spaces.Dict(
            {
                "block_id": spaces.Discrete(len(self.blocks)),
                "direction": spaces.Discrete(9),  # 0: stay, 1-8: directions
            }
        )

        # Observation space: Grid state + block positions + target positions
        self.observation_space = spaces.Dict(
            {
                "grid": spaces.Box(low=0, high=2, shape=(n, m), dtype=np.int32),
                "blocks": spaces.Box(
                    low=0, high=max(n, m), shape=(len(self.blocks), 3), dtype=np.int32
                ),
                "targets": spaces.Box(
                    low=0,
                    high=max(n, m),
                    shape=(len(self.target_positions), 3),
                    dtype=np.int32,
                ),
            }
        )

        # Movement directions (Moore neighborhood)
        self.directions = [
            (0, 0),  # Stay
            (-1, 0),  # North
            (1, 0),  # South
            (0, -1),  # West
            (0, 1),  # East
            (-1, -1),  # Northwest
            (-1, 1),  # Northeast
            (1, -1),  # Southwest
            (1, 1),  # Southeast
        ]

        # For tracking episode progress
        self.steps = 0
        self.max_steps = n * m * 4  # Reasonable upper bound

        # For computing rewards
        self.prev_distance = self._compute_total_distance()
        self.initial_distance = self.prev_distance

    def _update_grid(self):
        """Update the grid representation based on blocks and obstacles."""
        self.grid.fill(0)

        # Place blocks
        for _, x, y in self.blocks:
            self.grid[x, y] = 1

        # Place obstacles
        for x, y in self.obstacles:
            self.grid[x, y] = 2

    def _get_state(self):
        """Return the current state observation."""
        # Convert blocks to numpy array
        blocks_array = np.array(
            [(id, x, y) for id, x, y in self.blocks], dtype=np.int32
        )
        targets_array = np.array(
            [(id, x, y) for id, x, y in self.target_positions], dtype=np.int32
        )

        return {
            "grid": self.grid.copy(),
            "blocks": blocks_array,
            "targets": targets_array,
        }

    def _is_connected(self) -> bool:
        """Check if all blocks form a connected component."""
        if not self.blocks:
            return True

        block_positions = {(x, y) for _, x, y in self.blocks}
        visited = set()
        queue = deque(
            [(self.blocks[0][1], self.blocks[0][2])]
        )  # Start BFS from first block
        visited.add((self.blocks[0][1], self.blocks[0][2]))

        # Moore neighborhood
        directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

        while queue:
            x, y = queue.popleft()
            for dx, dy in directions:
                neighbor = (x + dx, y + dy)
                if neighbor in block_positions and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return len(visited) == len(self.blocks)

    def _is_valid_move(self, block_id: int, dx: int, dy: int) -> bool:
        """Check if moving a block is valid."""
        # Find the block
        block = next((b for b in self.blocks if b[0] == block_id), None)
        if not block:
            return False

        _, x, y = block
        new_x, new_y = x + dx, y + dy

        # Check if within bounds
        if not (0 <= new_x < self.n and 0 <= new_y < self.m):
            return False

        # Check for collision with obstacles
        if (new_x, new_y) in self.obstacles:
            return False

        # Check for collision with other blocks
        if any(
            (b_id != block_id and b_x == new_x and b_y == new_y)
            for b_id, b_x, b_y in self.blocks
        ):
            return False

        # Simulate move and check connectivity
        old_blocks = self.blocks.copy()

        # Update the block position
        for i, (b_id, b_x, b_y) in enumerate(self.blocks):
            if b_id == block_id:
                self.blocks[i] = (b_id, new_x, new_y)
                break

        # Check if still connected
        connected = self._is_connected()

        # Restore original positions
        self.blocks = old_blocks

        return connected

    def _compute_total_distance(self) -> float:
        """
        Compute the total Manhattan distance between blocks and their targets.
        Used for reward shaping.
        """
        total = 0
        for (b_id, b_x, b_y), (t_id, t_x, t_y) in zip(
            self.blocks, self.target_positions
        ):
            total += abs(b_x - t_x) + abs(b_y - t_y)
        return total

    def _compute_reward(self) -> float:
        """Improved reward function with better shaping"""
        # Base reward
        reward = 0

        # Distance-based reward
        current_distance = self._compute_total_distance()
        distance_improvement = self.prev_distance - current_distance

        # Increase the reward for getting closer to incentivize progress
        reward += distance_improvement * 2.0  # Double the importance

        # Add a bonus for reaching milestone distances
        if current_distance < self.prev_distance:
            # Progressive bonuses for improving
            if (
                current_distance < self.initial_distance * 0.75
                and self.prev_distance >= self.initial_distance * 0.75
            ):
                reward += 10  # Bonus for reaching 25% closer
            if (
                current_distance < self.initial_distance * 0.5
                and self.prev_distance >= self.initial_distance * 0.5
            ):
                reward += 20  # Bonus for reaching 50% closer
            if (
                current_distance < self.initial_distance * 0.25
                and self.prev_distance >= self.initial_distance * 0.25
            ):
                reward += 30  # Bonus for reaching 75% closer

        # Check if solved
        if current_distance == 0:
            reward += 500  # Much bigger bonus for solving

        # Update previous distance
        self.prev_distance = current_distance

        # Smaller step penalty to encourage exploration
        reward -= 0.05  # Reduced from -0.1

        return reward


    def step(self, action):
        """
        Take a step in the environment.

        Args:
            action: Dictionary with 'block_id' and 'direction' keys

        Returns:
            observation, reward, done, info
        """
        block_id = action["block_id"]
        dir_idx = action["direction"]

        # Convert direction index to dx, dy
        dx, dy = self.directions[dir_idx]

        # Check if the move is valid
        if dx == 0 and dy == 0:  # No movement
            reward = -0.2  # Small penalty for doing nothing
            done = False
        elif self._is_valid_move(block_id, dx, dy):
            # Update block position
            for i, (b_id, b_x, b_y) in enumerate(self.blocks):
                if b_id == block_id:
                    self.blocks[i] = (b_id, b_x + dx, b_y + dy)
                    break

            # Update grid
            self._update_grid()

            # Compute reward
            reward = self._compute_reward()

            # Check if solved - IMPORTANT ADDITION
            current_distance = self._compute_total_distance()
            done = current_distance == 0

            # Add success bonus reward if solved
            if done:
                reward += 500  # Big bonus for solving the puzzle
                print(f"🎉 SOLVED! Steps taken: {self.steps}")
        else:
            # Invalid move
            reward = -1  # Penalty for invalid move
            done = False

        # Update step counter
        self.steps += 1
        if self.steps >= self.max_steps:
            done = True

        # Get current state
        observation = self._get_state()

        # Additional info
        info = {
            "distance": self._compute_total_distance(),
            "steps": self.steps,
            "is_valid_move": dx == 0 and dy == 0 or self._is_valid_move(block_id, dx, dy),
            "solved": done and self.steps < self.max_steps,  # Indicate if solved vs timeout
        }


        return observation, reward, done, info

    def reset(self):
        """Reset the environment to initial state."""
        # Reset blocks to initial positions
        self.blocks = [(i, pos[0], pos[1]) for i, pos in enumerate(self.blocks)]

        # Reset grid
        self._update_grid()

        # Reset step counter
        self.steps = 0

        # Reset previous distance
        self.prev_distance = self._compute_total_distance()
        self.initial_distance = self.prev_distance  # Also reset initial distance

        return self._get_state()

    def render(self, mode="human"):
        """
        Render the environment.

        Args:
            mode (str): 'human' for terminal output, 'rgb_array' for numpy array
        """
        # Create a visual representation with colors
        visual_grid = np.zeros((self.n, self.m, 3), dtype=np.uint8)

        # Set colors: white for empty, blue for blocks, green for targets,
        # purple for blocks on targets, red for obstacles
        empty_color = np.array([255, 255, 255], dtype=np.uint8)  # White
        block_color = np.array([0, 0, 255], dtype=np.uint8)  # Blue
        target_color = np.array([0, 255, 0], dtype=np.uint8)  # Green
        overlap_color = np.array([128, 0, 128], dtype=np.uint8)  # Purple
        obstacle_color = np.array([255, 0, 0], dtype=np.uint8)  # Red

        # Fill with empty color
        visual_grid.fill(255)

        # Get block and target positions
        block_positions = {(x, y): id for id, x, y in self.blocks}
        target_positions = {(x, y): id for id, x, y in self.target_positions}

        # Mark targets
        for x, y in target_positions:
            visual_grid[x, y] = target_color

        # Mark obstacles
        for x, y in self.obstacles:
            visual_grid[x, y] = obstacle_color

        # Mark blocks (and blocks on targets)
        for x, y in block_positions:
            if (x, y) in target_positions:
                visual_grid[x, y] = overlap_color  # Block on target
            else:
                visual_grid[x, y] = block_color

        if mode == "human":
            # Print to terminal
            print("\nGrid State:")
            for i in range(self.n):
                for j in range(self.m):
                    if tuple(visual_grid[i, j]) == tuple(empty_color):
                        print("⬜", end="")
                    elif tuple(visual_grid[i, j]) == tuple(block_color):
                        print("🟦", end="")
                    elif tuple(visual_grid[i, j]) == tuple(target_color):
                        print("🟩", end="")
                    elif tuple(visual_grid[i, j]) == tuple(overlap_color):
                        print("🟪", end="")
                    elif tuple(visual_grid[i, j]) == tuple(obstacle_color):
                        print("🟥", end="")
                print()
            print(
                f"Step: {self.steps}, Distance to target: {self._compute_total_distance()}"
            )

        return visual_grid
