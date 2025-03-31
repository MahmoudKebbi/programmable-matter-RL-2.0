import os
import sys

# REVIEW: More robust path handling
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # Add current dir
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # Add parent dir

import numpy as np
from grid import (
    Grid,
)  # REVIEW: Assuming grid.py is in the same directory or PYTHONPATH is set
from scipy.optimize import (
    linear_sum_assignment,
)  # REVIEW: For optimal distance calculation
from collections import (
    deque,
)  # REVIEW: Added for potential use, though not currently needed here


class GridEnv:
    """
    Reinforcement Learning environment for programmable matter simulation.
    """

    def __init__(
        self,
        n,
        m,
        initial_positions,
        target_positions,
        max_steps=100,
        normalize_coords=True,  # REVIEW: Added option for coordinate normalization
    ):
        """
        Initialize the environment.

        Args:
            n (int): Number of rows in the grid.
            m (int): Number of columns in the grid.
            initial_positions (list): List of (x, y) coordinates for initial matter placement.
            target_positions (list): List of (x, y) coordinates for target configuration.
            max_steps (int): Maximum steps per episode.
            normalize_coords (bool): Whether to normalize coordinates in the observation.
        """
        self.n = n
        self.m = m
        # REVIEW: Ensure target positions are sorted for consistent comparison/distance calc
        self.target_positions = sorted(target_positions)
        # REVIEW: Store initial positions directly for reset
        self._initial_positions = sorted(initial_positions)
        self.max_steps = max_steps
        self.normalize_coords = normalize_coords
        self.grid = Grid(n, m, self._initial_positions)  # Initial grid creation
        self.steps = 0
        self.num_blocks = len(self._initial_positions)

        if len(self._initial_positions) != len(self.target_positions):
            raise ValueError(
                "Initial and target shapes must have the same number of blocks."
            )

        # Define unified action space
        # Actions 0-4: Uniform movement (no-op, up, right, down, left)
        # Actions 5+: Individual movement (block_index * 5 + direction_index)
        # Directions: 0: no-op, 1: up (-1,0), 2: right (0,1), 3: down (1,0), 4: left (0,-1)
        self.uniform_actions = 5
        self.individual_actions_per_block = 5
        self.action_space_size = self.uniform_actions + (
            self.num_blocks * self.individual_actions_per_block
        )

        # Define observation space size dynamically based on _get_observation output
        # REVIEW: Calculate observation space size based on the actual observation format
        self.observation_space_size = len(self._get_observation(calculate_size=True))
        print(
            f"Environment Initialized: Grid={n}x{m}, Blocks={self.num_blocks}, "
            f"Action Space={self.action_space_size}, Obs Space={self.observation_space_size}"
        )

    def reset(self):
        """
        Reset the environment to initial state.

        Returns:
            observation (np.array): The initial state observation.
        """
        # REVIEW: Reset using the stored initial positions
        self.grid = Grid(self.n, self.m, self._initial_positions)
        self.steps = 0
        return self._get_observation()

    def step(self, action):
        """
        Take a step in the environment based on the given action.

        Args:
            action (int): Action index.

        Returns:
            tuple: (observation, reward, done, info)
        """
        if not (0 <= action < self.action_space_size):
            raise ValueError(
                f"Invalid action {action} for action space size {self.action_space_size}"
            )

        self.steps += 1

        # REVIEW: Calculate distance/state metrics *before* taking the action
        old_positions = (
            self.grid.matter_elements.copy()
        )  # Keep copy for potential reward calc
        old_distance = self._optimal_distance_to_target(old_positions)

        # Action mapping
        # Directions: 0: no-op, 1: up (-1,0), 2: right (0,1), 3: down (1,0), 4: left (0,-1)
        # REVIEW: Use a clearer direction mapping
        action_to_direction = {
            0: (0, 0),  # No-op
            1: (-1, 0),  # Up
            2: (0, 1),  # Right
            3: (1, 0),  # Down
            4: (0, -1),  # Left
        }

        success = False
        movement_type = "invalid"  # Default

        # Determine and execute action
        if action < self.uniform_actions:
            # Uniform movement
            movement_type = "uniform"
            direction_index = action
            dx, dy = action_to_direction[direction_index]
            if dx == 0 and dy == 0:
                success = True  # No-op is always "successful" in terms of execution
            else:
                success = self.grid.move(
                    dx, dy, verbose=False
                )  # verbose=True for debugging
        else:
            # Individual block movement
            movement_type = "individual"
            individual_action_index = action - self.uniform_actions
            block_index = individual_action_index // self.individual_actions_per_block
            direction_index = (
                individual_action_index % self.individual_actions_per_block
            )

            if block_index >= self.num_blocks:
                # This case should ideally be prevented by masking or agent's output layer size
                print(
                    f"Warning: Invalid block index {block_index} derived from action {action}."
                )
                success = False
            else:
                dx, dy = action_to_direction[direction_index]
                if dx == 0 and dy == 0:
                    success = True  # No-op for individual block
                else:
                    # Create move dictionary targeting the specific block index
                    # Note: block_index refers to the index in the *sorted* self.grid.matter_elements
                    moves = {block_index: (dx, dy)}
                    success = self.grid.move_individual(
                        moves, verbose=False
                    )  # verbose=True for debugging

        # Get new state and metrics
        observation = self._get_observation()
        new_positions = self.grid.matter_elements
        new_distance = self._optimal_distance_to_target(new_positions)

        # Check termination conditions
        target_reached = set(new_positions) == set(self.target_positions)
        timeout = self.steps >= self.max_steps
        done = target_reached or timeout

        # Calculate reward
        # REVIEW: Simplified reward structure, focusing on distance and validity
        reward = self._calculate_reward(
            success, old_distance, new_distance, target_reached, timeout
        )

        info = {
            "steps": self.steps,
            "success": success,  # Whether the move mechanics were valid (bounds, connectivity, overlap)
            "distance_to_target": new_distance,
            "action": action,
            "movement_type": movement_type,
            "target_reached": target_reached,
            "timeout": timeout,
        }

        return observation, reward, done, info

    def _calculate_reward(
        self, success, old_distance, new_distance, target_reached, timeout
    ):
        """Calculate reward based on state change and termination conditions."""
        if target_reached:
            return 60.0
        if timeout:
            return -20.0
        if not success:
            return -2.0

        distance_improvement = old_distance - new_distance

        # Progressive reward based on distance improvement
        if distance_improvement > 0:
            # More generous reward for improvement
            reward = distance_improvement * 3.0
        elif distance_improvement == 0:
            # NEUTRAL rather than negative for no change
            reward = 0.0  # Don't discourage exploration moves
        else:
            # Much gentler penalty for temporary distance increases
            reward = distance_improvement * 0.5  # Half the penalty

        # Tiny step penalty that won't overshadow other rewards
        reward -= 0.005

        return reward

    def _optimal_distance_to_target(self, current_positions=None):
        """
        Calculate the sum of minimum Manhattan distances between current blocks
        and target positions using optimal assignment (Hungarian algorithm).
        This provides a more accurate measure of "how far" the shape is.

        Args:
            current_positions (list, optional): List of (x,y) tuples.
                                                Defaults to self.grid.matter_elements.

        Returns:
            float: Sum of minimum assigned distances.
        """
        if current_positions is None:
            current_positions = self.grid.matter_elements

        if not current_positions:
            return 0.0
        if len(current_positions) != len(self.target_positions):
            # Should not happen if init checks passed
            print(
                "Warning: Mismatch between current and target block counts in distance calculation."
            )
            return float("inf")

        # Create cost matrix: cost[i][j] = distance between current_pos[i] and target_pos[j]
        cost_matrix = np.zeros((self.num_blocks, self.num_blocks))
        for i, (cx, cy) in enumerate(current_positions):
            for j, (tx, ty) in enumerate(self.target_positions):
                cost_matrix[i, j] = abs(cx - tx) + abs(cy - ty)

        # Use linear_sum_assignment (Hungarian algorithm) to find the optimal assignment
        # It finds the assignment that minimizes the total cost (sum of distances)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # The minimum total distance is the sum of the costs of the optimal assignments
        min_total_distance = cost_matrix[row_ind, col_ind].sum()
        return min_total_distance

    def _get_observation(self, calculate_size=False):
        """
        Create observation vector from the current state.
        REVIEW: Using relative current positions + absolute target positions.
        Includes:
        - Normalized relative positions between current blocks (flattened dx,dy pairs).
        - Normalized absolute target block positions (flattened x,y pairs).
        - Maybe add current centroid? (Optional)
        """
        norm_factor = max(self.n, self.m)
        current_pos_list = (
            self.grid.matter_elements
            if not calculate_size
            else [(0, 0)] * self.num_blocks
        )
        target_pos_list = (
            self.target_positions if not calculate_size else [(0, 0)] * self.num_blocks
        )

        # --- Feature Extraction ---
        # 1. Relative Positions between current blocks
        relative_positions = []
        if self.num_blocks > 1:
            # Use first block as reference? Or centroid? Let's use first block.
            ref_x, ref_y = current_pos_list[0]
            # Add positions relative to the reference block
            for i in range(
                self.num_blocks
            ):  # Include ref block as (0,0) relative? Yes.
                px, py = current_pos_list[i]
                relative_positions.extend([px - ref_x, py - ref_y])
            # Alternative: Pairwise relative positions (as before)
            # for i in range(self.num_blocks):
            #     for j in range(i + 1, self.num_blocks):
            #         p1 = current_pos_list[i]
            #         p2 = current_pos_list[j]
            #         relative_positions.extend([p2[0] - p1[0], p2[1] - p1[1]])
        elif self.num_blocks == 1:
            relative_positions.extend(
                [0, 0]
            )  # Single block relative to itself is (0,0)

        relative_positions_flat = np.array(relative_positions).flatten()

        # 2. Target Positions (Absolute)
        target_positions_flat = np.array(target_pos_list).flatten()

        # --- Normalization ---
        if self.normalize_coords:
            # Normalize relative positions by grid size (max possible difference)
            relative_positions_flat = relative_positions_flat / norm_factor
            # Normalize absolute target positions
            target_positions_flat = target_positions_flat / norm_factor

        # --- Concatenate Features ---
        observation = np.concatenate(
            [
                relative_positions_flat,
                target_positions_flat,
            ]
        ).astype(np.float32)

        return observation

    def render(self, title=""):
        """Render the environment state"""
        print(f"\n--- {title} Step: {self.steps} ---")
        self.grid.display_grid()
        print(f"Current Positions: {self.grid.matter_elements}")
        print(f"Target Positions:  {self.target_positions}")
        print(f"Distance to Target: {self._optimal_distance_to_target():.2f}")
        print("-" * (len(title) + 14))

    def get_valid_actions_mask(self):
        """
        Return a binary mask indicating valid actions (1=valid, 0=invalid).
        Checks bounds, overlaps, and connectivity for each potential move.
        """
        mask = np.zeros(
            self.action_space_size, dtype=np.int8
        )  # Use int8 for efficiency

        # Action mapping (repeated for clarity)
        action_to_direction = {0: (0, 0), 1: (-1, 0), 2: (0, 1), 3: (1, 0), 4: (0, -1)}

        # --- Check Uniform Actions (0-4) ---
        for action_idx, (dx, dy) in action_to_direction.items():
            if dx == 0 and dy == 0:
                mask[action_idx] = 1  # No-op is always valid
            # REVIEW: Use the grid's internal check for uniform moves
            elif self.grid.is_valid_move(dx, dy):
                mask[action_idx] = 1

        # --- Check Individual Actions (5+) ---
        if not self.grid.matter_elements:  # Skip if no blocks
            return mask

        original_positions = self.grid.matter_elements  # Use current state
        num_blocks = len(original_positions)

        for block_idx in range(num_blocks):
            for dir_idx, (dx, dy) in action_to_direction.items():
                action_idx = (
                    self.uniform_actions
                    + block_idx * self.individual_actions_per_block
                    + dir_idx
                )

                if dx == 0 and dy == 0:
                    mask[action_idx] = 1  # No-op for individual block is always valid
                    continue

                # --- Simulate the single block move ---
                new_positions_potential = []
                target_positions_set = set()
                valid_simulation = True

                for i, (x, y) in enumerate(original_positions):
                    move_dx, move_dy = (0, 0)
                    if (
                        i == block_idx
                    ):  # Apply move only to the current block being checked
                        move_dx, move_dy = dx, dy

                    nx, ny = x + move_dx, y + move_dy

                    # Check bounds
                    if not (0 <= nx < self.n and 0 <= ny < self.m):
                        valid_simulation = False
                        break
                    # Check overlaps
                    if (nx, ny) in target_positions_set:
                        valid_simulation = False
                        break

                    target_positions_set.add((nx, ny))
                    new_positions_potential.append((nx, ny))

                if not valid_simulation:
                    continue  # Move invalid due to bounds or overlap

                # Check connectivity of the potential new state
                # REVIEW: Use the grid's helper function, passing sorted list
                new_positions_sorted = sorted(new_positions_potential)
                if self.grid.is_connected_after_move(new_positions_sorted):
                    mask[action_idx] = 1

        # REVIEW: Check if any valid action exists besides no-ops. If not, could indicate a problem.
        if np.sum(mask) <= (1 + self.num_blocks):  # Only no-ops are valid
            # This might happen legitimately if the shape is completely blocked
            # print(f"Warning: Only no-op actions are valid at step {self.steps}")
            pass

        return mask
