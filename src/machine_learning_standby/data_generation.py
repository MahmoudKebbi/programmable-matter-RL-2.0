import numpy as np
import random
import pickle
import os
from tqdm import tqdm
import torch
from torch_geometric.data import Dataset, Data

from grid import Grid
from ai_agent import AI_Agent, state_to_tuple
from ml_model import create_graph_from_state, direction_to_index


class TrainingDataGenerator:
    """
    Generates training data from the A* search agent.
    """

    def __init__(
        self,
        grid_sizes=[(5, 5), (6, 6), (7, 7)],
        num_blocks_range=(3, 8),
        samples_per_size=100,
        max_path_length=30,
    ):
        self.grid_sizes = grid_sizes
        self.num_blocks_range = num_blocks_range
        self.samples_per_size = samples_per_size
        self.max_path_length = max_path_length
        self.data = []

    def generate_random_problem(self, n, m, num_blocks):
        """
        Generate a random problem with connected blocks.

        Args:
            n, m: Grid dimensions
            num_blocks: Number of matter blocks to place

        Returns:
            tuple: (start_state, target_state)
        """
        # Generate a connected start state
        start_state = []
        grid = np.zeros((n, m))

        # Start with a random position
        x, y = random.randint(0, n - 1), random.randint(0, m - 1)
        start_state.append((x, y))
        grid[x, y] = 1

        # Add remaining blocks ensuring connectivity
        block_count = 1
        while block_count < num_blocks:
            # Choose a random existing block
            parent_idx = random.randint(0, len(start_state) - 1)
            px, py = start_state[parent_idx]

            # Try to add a neighboring block
            directions = [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ]
            random.shuffle(directions)

            for dx, dy in directions:
                nx, ny = px + dx, py + dy
                if 0 <= nx < n and 0 <= ny < m and grid[nx, ny] == 0:
                    start_state.append((nx, ny))
                    grid[nx, ny] = 1
                    block_count += 1
                    break
            else:
                # If we couldn't add a neighbor, try another parent
                continue

            if block_count >= num_blocks:
                break

        # Generate a connected target state with same number of blocks
        target_state = []
        grid = np.zeros((n, m))

        # Start with a random position
        x, y = random.randint(0, n - 1), random.randint(0, m - 1)
        target_state.append((x, y))
        grid[x, y] = 1

        # Add remaining blocks ensuring connectivity
        block_count = 1
        while block_count < num_blocks:
            # Choose a random existing block
            parent_idx = random.randint(0, len(target_state) - 1)
            px, py = target_state[parent_idx]

            # Try to add a neighboring block
            directions = [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ]
            random.shuffle(directions)

            for dx, dy in directions:
                nx, ny = px + dx, py + dy
                if 0 <= nx < n and 0 <= ny < m and grid[nx, ny] == 0:
                    target_state.append((nx, ny))
                    grid[nx, ny] = 1
                    block_count += 1
                    break
            else:
                # If we couldn't add a neighbor, try another parent
                continue

            if block_count >= num_blocks:
                break

        return sorted(start_state), sorted(target_state)

    def generate_training_data(self, output_dir="training_data"):
        """Generate and save training data for ML model."""
        os.makedirs(output_dir, exist_ok=True)

        dataset = []
        problems_solved = 0
        attempts = 0

        pbar = tqdm(total=len(self.grid_sizes) * self.samples_per_size)

        for n, m in self.grid_sizes:
            grid_samples = 0

            while (
                grid_samples < self.samples_per_size
                and attempts < self.samples_per_size * 3
            ):
                attempts += 1

                # Determine number of blocks for this problem
                num_blocks = random.randint(
                    self.num_blocks_range[0], self.num_blocks_range[1]
                )

                # Generate random problem
                start_state, target_state = self.generate_random_problem(
                    n, m, num_blocks
                )

                # Run A* search
                agent = AI_Agent(n, m, start_state, target_state)
                solution = agent.plan()

                if solution and len(solution) <= self.max_path_length:
                    # We found a solution - extract training data
                    path_data = self._extract_path_data(
                        start_state, target_state, solution, (n, m)
                    )
                    dataset.extend(path_data)
                    grid_samples += 1
                    problems_solved += 1
                    pbar.update(1)

            print(f"Grid size {n}x{m}: {grid_samples} samples generated")

        pbar.close()
        print(f"Total problems solved: {problems_solved}/{attempts} attempts")
        print(f"Total training examples: {len(dataset)}")

        # Save dataset to disk
        with open(os.path.join(output_dir, "training_data.pkl"), "wb") as f:
            pickle.dump(dataset, f)

        return dataset

    def _extract_path_data(self, start_state, target_state, solution, grid_dims):
        """
        Extract training data from an A* solution path.

        Args:
            start_state: Initial block positions
            target_state: Target block positions
            solution: List of moves from A* search
            grid_dims: Grid dimensions (n, m)

        Returns:
            list: Training examples from this solution path
        """
        path_data = []
        current_state = start_state.copy()

        # Initial state is special - we know the first step in the solution
        first_moves = solution[0] if solution else []

        # Create graph representation of initial state
        initial_graph = create_graph_from_state(current_state, target_state, grid_dims)

        # Extract training labels from the first move
        move_labels = torch.zeros(len(current_state), dtype=torch.long)
        for block_idx, dx, dy in first_moves:
            move_labels[block_idx] = direction_to_index(dx, dy)

        # Compute distance to goal (steps remaining) as heuristic label
        heuristic_label = torch.tensor([len(solution)], dtype=torch.float)

        # Store initial state training example
        initial_graph.y_heuristic = heuristic_label
        initial_graph.y_moves = move_labels
        path_data.append(initial_graph)

        # Process each step in the solution to create more training examples
        for i, moves in enumerate(solution):
            # Apply moves to update current state
            for block_idx, dx, dy in moves:
                x, y = current_state[block_idx]
                current_state[block_idx] = (x + dx, y + dy)

            # Skip the last state since there are no more moves to predict
            if i == len(solution) - 1:
                break

            # Create graph representation
            state_graph = create_graph_from_state(
                current_state, target_state, grid_dims
            )

            # Extract training labels from the next move in solution
            next_moves = solution[i + 1]
            move_labels = torch.zeros(len(current_state), dtype=torch.long)
            for block_idx, dx, dy in next_moves:
                move_labels[block_idx] = direction_to_index(dx, dy)

            # Compute steps remaining as heuristic label
            heuristic_label = torch.tensor([len(solution) - (i + 1)], dtype=torch.float)

            # Store training example
            state_graph.y_heuristic = heuristic_label
            state_graph.y_moves = move_labels
            path_data.append(state_graph)

        return path_data


class ProgrammableMatterDataset(Dataset):
    """PyTorch Geometric Dataset for programmable matter problem."""

    def __init__(self, data_file, transform=None):
        super(ProgrammableMatterDataset, self).__init__(transform)

        with open(data_file, "rb") as f:
            self.data_list = pickle.load(f)

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


if __name__ == "__main__":
    # Example: Generate a small dataset
    generator = TrainingDataGenerator(
        grid_sizes=[(5, 5), (6, 6)], num_blocks_range=(3, 6), samples_per_size=50
    )

    dataset = generator.generate_training_data("example_data")
    print(f"Generated {len(dataset)} training examples")
