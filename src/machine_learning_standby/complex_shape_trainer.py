import os
import random
import numpy as np
import torch
from tqdm import tqdm
import time
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

from grid import Grid
from ai_agent import AI_Agent
from ml_model import ProgrammableMatterGNN, create_graph_from_state, direction_to_index
from data_generation import TrainingDataGenerator
from train_model import ModelTrainer


class ComplexShapeTrainer:
    """
    Specialized trainer for complex programmable matter shapes on large grids.
    Generates and trains on shapes with 20-30 blocks on 40x40 grids.
    """

    def __init__(
        self,
        output_dir="complex_training_data",
        model_dir="complex_models",
        grid_sizes=[(40, 40)],
        block_counts=[30],
        samples_per_config=20,
        max_path_length=50,
        max_time_per_problem=30,  # 0.5 minutes max per problem
        num_workers=4,
    ):

        self.output_dir = output_dir
        self.model_dir = model_dir
        self.grid_sizes = grid_sizes
        self.block_counts = block_counts
        self.samples_per_config = samples_per_config
        self.max_path_length = max_path_length
        self.max_time_per_problem = max_time_per_problem
        self.num_workers = num_workers

        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

        # Training data file path
        self.data_file = os.path.join(output_dir, "complex_training_data.pkl")

    def generate_complex_shape(self, n, m, num_blocks, shape_type="random"):
        """
        Generate a complex shape with the specified number of blocks.

        Args:
            n, m: Grid dimensions
            num_blocks: Number of blocks
            shape_type: Type of shape to generate ("random", "snake", "compact", "sparse")

        Returns:
            List of (x, y) coordinates
        """
        if shape_type == "random":
            return self._generate_random_shape(n, m, num_blocks)
        elif shape_type == "snake":
            return self._generate_snake_shape(n, m, num_blocks)
        elif shape_type == "compact":
            return self._generate_compact_shape(n, m, num_blocks)
        elif shape_type == "sparse":
            return self._generate_sparse_shape(n, m, num_blocks)
        else:
            print(f"Unknown shape type: {shape_type}. Using random.")
            return self._generate_random_shape(n, m, num_blocks)

    def _generate_random_shape(self, n, m, num_blocks):
        """Generate a random connected shape."""
        state = []
        grid = np.zeros((n, m))

        # Start with a random position near center
        center_x, center_y = n // 2, m // 2
        start_x = random.randint(max(0, center_x - 5), min(n - 1, center_x + 5))
        start_y = random.randint(max(0, center_y - 5), min(m - 1, center_y + 5))

        state.append((start_x, start_y))
        grid[start_x, start_y] = 1

        # Add remaining blocks ensuring connectivity
        block_count = 1
        max_attempts = num_blocks * 10
        attempts = 0

        while block_count < num_blocks and attempts < max_attempts:
            attempts += 1

            # Choose a random existing block
            parent_idx = random.randint(0, len(state) - 1)
            px, py = state[parent_idx]

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
                    state.append((nx, ny))
                    grid[nx, ny] = 1
                    block_count += 1
                    attempts = 0  # Reset attempts counter after success
                    break

            if block_count >= num_blocks:
                break

        if block_count < num_blocks:
            print(f"Warning: Could only generate {block_count}/{num_blocks} blocks")

        return sorted(state)

    def _generate_snake_shape(self, n, m, num_blocks):
        """Generate a long snake-like pattern."""
        state = []
        grid = np.zeros((n, m))

        # Start near center
        center_x, center_y = n // 2, m // 2
        x, y = center_x, center_y
        state.append((x, y))
        grid[x, y] = 1

        # Generate snake by following a primarily horizontal path with some vertical deviations
        horizontal_bias = random.choice([True, False])  # Start horizontal or vertical
        direction = 1  # Start in positive direction

        for i in range(1, num_blocks):
            # Every few blocks, consider changing direction
            if i % 8 == 0:
                direction *= -1  # Reverse direction

            if i % 12 == 0:
                horizontal_bias = (
                    not horizontal_bias
                )  # Toggle between horizontal/vertical bias

            # Try to continue in the current biased direction
            tries = 0
            while tries < 8:  # Try all directions if needed
                if horizontal_bias:
                    dx, dy = 0, direction
                else:
                    dx, dy = direction, 0

                # Consider a small chance of moving in non-biased direction
                if random.random() < 0.15:
                    dx, dy = dy, dx  # Switch orientation

                # Try to add a block in the chosen direction
                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < m and grid[nx, ny] == 0:
                    state.append((nx, ny))
                    grid[nx, ny] = 1
                    x, y = nx, ny
                    break

                # If that didn't work, try a different direction
                tries += 1
                if tries >= 4:
                    # After trying cardinal directions, try diagonals
                    directions = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                    dx, dy = random.choice(directions)

                nx, ny = x + dx, y + dy
                if 0 <= nx < n and 0 <= ny < m and grid[nx, ny] == 0:
                    state.append((nx, ny))
                    grid[nx, ny] = 1
                    x, y = nx, ny
                    break

            if tries == 8:
                # If we tried all directions and failed, find any valid neighbor of any existing block
                found = False
                for bx, by in state:
                    for dx, dy in [
                        (-1, 0),
                        (1, 0),
                        (0, -1),
                        (0, 1),
                        (-1, -1),
                        (-1, 1),
                        (1, -1),
                        (1, 1),
                    ]:
                        nx, ny = bx + dx, by + dy
                        if 0 <= nx < n and 0 <= ny < m and grid[nx, ny] == 0:
                            state.append((nx, ny))
                            grid[nx, ny] = 1
                            x, y = nx, ny
                            found = True
                            break
                    if found:
                        break

                if not found:
                    print(f"Warning: Could only generate {i}/{num_blocks} blocks")
                    break

        return sorted(state)

    def _generate_compact_shape(self, n, m, num_blocks):
        """Generate a compact, blob-like shape."""
        state = []
        grid = np.zeros((n, m))
        boundary = []  # List of boundary cells (adjacent to filled cells)

        # Start with a random position near center
        center_x, center_y = n // 2, m // 2
        start_x = random.randint(max(0, center_x - 3), min(n - 1, center_x + 3))
        start_y = random.randint(max(0, center_y - 3), min(m - 1, center_y + 3))

        state.append((start_x, start_y))
        grid[start_x, start_y] = 1

        # Update boundary cells
        for dx, dy in [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]:
            nx, ny = start_x + dx, start_y + dy
            if 0 <= nx < n and 0 <= ny < m and grid[nx, ny] == 0:
                boundary.append((nx, ny))

        # Add remaining blocks preferring cells with more filled neighbors
        block_count = 1
        while block_count < num_blocks and boundary:
            # Score boundary cells by number of filled neighbors
            cell_scores = []
            for x, y in boundary:
                neighbor_count = 0
                for dx, dy in [
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                    (-1, -1),
                    (-1, 1),
                    (1, -1),
                    (1, 1),
                ]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < n and 0 <= ny < m and grid[nx, ny] == 1:
                        neighbor_count += 1
                cell_scores.append((neighbor_count, x, y))

            # Choose a cell with bias toward more neighbors (more compact)
            cell_scores.sort(reverse=True)  # Highest scores first

            # Choose from top 30% scores with some randomness
            top_cells = cell_scores[: max(1, len(cell_scores) // 3)]
            _, x, y = random.choice(top_cells)

            # Fill the chosen cell
            state.append((x, y))
            grid[x, y] = 1
            block_count += 1

            # Remove from boundary
            boundary = [b for b in boundary if b != (x, y)]

            # Add new boundary cells
            for dx, dy in [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ]:
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < n
                    and 0 <= ny < m
                    and grid[nx, ny] == 0
                    and (nx, ny) not in boundary
                ):
                    boundary.append((nx, ny))

        if block_count < num_blocks:
            print(f"Warning: Could only generate {block_count}/{num_blocks} blocks")

        return sorted(state)

    def _generate_sparse_shape(self, n, m, num_blocks):
        """Generate a sparse branching structure."""
        state = []
        grid = np.zeros((n, m))

        # Start with a random position near center
        center_x, center_y = n // 2, m // 2
        start_x = random.randint(max(0, center_x - 3), min(n - 1, center_x + 3))
        start_y = random.randint(max(0, center_y - 3), min(m - 1, center_y + 3))

        state.append((start_x, start_y))
        grid[start_x, start_y] = 1

        # Add several initial branches
        branches = [(start_x, start_y)]
        block_count = 1

        while block_count < num_blocks and branches:
            # Randomly select a branch to extend
            branch_idx = random.randint(0, len(branches) - 1)
            x, y = branches[branch_idx]

            # Choose a random direction
            directions = [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
            ]  # Only cardinal directions for sparse look
            random.shuffle(directions)

            # Try to extend branch
            extended = False
            for dx, dy in directions:
                # Try to go 2-3 cells in this direction
                steps = random.randint(1, min(3, num_blocks - block_count))

                valid_extension = True
                extension = []

                for step in range(1, steps + 1):
                    nx, ny = x + dx * step, y + dy * step
                    if 0 <= nx < n and 0 <= ny < m and grid[nx, ny] == 0:
                        extension.append((nx, ny))
                    else:
                        valid_extension = False
                        break

                if valid_extension and extension:
                    # Add extension to state and grid
                    for nx, ny in extension:
                        state.append((nx, ny))
                        grid[nx, ny] = 1
                        block_count += 1

                    # Add the endpoint as a new potential branch
                    branches.append(extension[-1])
                    extended = True
                    break

            if not extended:
                # Remove this branch point as it can't be extended
                branches.pop(branch_idx)

                # If we're running out of branches but still need blocks,
                # add some random connected blocks
                if len(branches) <= 1 and block_count < num_blocks * 0.8:
                    # Try to find any valid neighbor of any existing block
                    for bx, by in state:
                        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nx, ny = bx + dx, by + dy
                            if 0 <= nx < n and 0 <= ny < m and grid[nx, ny] == 0:
                                state.append((nx, ny))
                                grid[nx, ny] = 1
                                branches.append((nx, ny))
                                block_count += 1
                                break
                        if block_count >= num_blocks:
                            break

        if block_count < num_blocks:
            print(f"Warning: Could only generate {block_count}/{num_blocks} blocks")

        return sorted(state)

    def generate_training_problem(self, shape_type="mixed"):
        """
        Generate a training problem with complex shapes.

        Args:
            shape_type: Type of shape to generate ("random", "snake", "compact", "sparse", or "mixed")

        Returns:
            dict: Problem configuration
        """
        # Select a random grid size
        n, m = random.choice(self.grid_sizes)

        # Select a random block count
        num_blocks = random.choice(self.block_counts)

        # Determine shape types to use
        if shape_type == "mixed":
            start_shape_type = random.choice(["random", "snake", "compact", "sparse"])
            target_shape_type = random.choice(["random", "snake", "compact", "sparse"])
        else:
            start_shape_type = target_shape_type = shape_type

        # Generate start and target shapes
        start_state = self.generate_complex_shape(n, m, num_blocks, start_shape_type)
        target_state = self.generate_complex_shape(n, m, num_blocks, target_shape_type)

        return {
            "grid_size": (n, m),
            "start_state": start_state,
            "target_state": target_state,
            "start_shape_type": start_shape_type,
            "target_shape_type": target_shape_type,
            "num_blocks": len(start_state),
        }

    def solve_problem(self, problem):
        """
        Solve a training problem using A* search.

        Args:
            problem: Dictionary with problem configuration

        Returns:
            tuple: (solution, execution_time, nodes_expanded)
        """
        n, m = problem["grid_size"]
        start_state = problem["start_state"]
        target_state = problem["target_state"]

        # Create AI agent
        agent = AI_Agent(n, m, start_state, target_state)

        # Set a timeout for the A* search
        start_time = time.time()
        solution = agent.plan()
        execution_time = time.time() - start_time

        if solution and len(solution) <= self.max_path_length:
            return solution, execution_time, agent.nodes_expanded
        else:
            return None, execution_time, agent.nodes_expanded

    def extract_training_data(self, start_state, target_state, solution, grid_dims):
        """Extract training data from a solution path."""
        from data_generation import TrainingDataGenerator

        # Use the existing method from TrainingDataGenerator
        generator = TrainingDataGenerator()
        path_data = generator._extract_path_data(
            start_state, target_state, solution, grid_dims
        )

        return path_data

    def process_problem(self, problem):
        """
        Process a single problem for data generation.
        Designed to be called in parallel.

        Returns:
            list: Training examples or None if no solution found
        """
        # Try to solve the problem
        solution, execution_time, nodes_expanded = self.solve_problem(problem)

        if solution:
            print(
                f"Problem solved in {execution_time:.2f}s, {len(solution)} steps, {nodes_expanded} nodes. Extracting training data..."
            )

            # Extract training data
            training_examples = self.extract_training_data(
                problem["start_state"],
                problem["target_state"],
                solution,
                problem["grid_size"],
            )

            return training_examples
        else:
            print(
                f"No solution found after {execution_time:.2f}s, {nodes_expanded} nodes."
            )
            return None

    def generate_training_data(self, num_problems=100, shape_type="mixed"):
        """
        Generate training data from complex shape problems.

        Args:
            num_problems: Number of problems to generate
            shape_type: Type of shape to generate
        """
        import pickle

        # Generate problems
        problems = []
        for i in range(num_problems):
            problems.append(self.generate_training_problem(shape_type))

        print(f"Generated {len(problems)} problems")

        # Process problems in parallel
        dataset = []
        successful_problems = 0

        if self.num_workers > 1:
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [
                    executor.submit(self.process_problem, problem)
                    for problem in problems
                ]

                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Processing problems",
                ):
                    result = future.result()
                    if result:
                        dataset.extend(result)
                        successful_problems += 1
        else:
            for problem in tqdm(problems, desc="Processing problems"):
                result = self.process_problem(problem)
                if result:
                    dataset.extend(result)
                    successful_problems += 1

        print(f"Successfully solved {successful_problems}/{len(problems)} problems")
        print(f"Generated {len(dataset)} training examples")

        # Save dataset
        with open(self.data_file, "wb") as f:
            pickle.dump(dataset, f)

        return dataset

    def train_model(
        self, epochs=50, batch_size=32, learning_rate=0.001, hidden_dim=128
    ):
        """Train a model on the generated complex shape data."""
        trainer = ModelTrainer(
            data_dir=self.output_dir,
            model_dir=self.model_dir,
            batch_size=batch_size,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
        )

        # Load data
        train_size, val_size = trainer.load_data()

        # Initialize model
        model = trainer.initialize_model()

        # Train model
        train_losses, val_losses = trainer.train(epochs=epochs)

        return trainer


def main():
    """Main function to run complex shape training from command line."""
    parser = argparse.ArgumentParser(description="Train ML model on complex shapes")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="complex_training_data",
        help="Directory to save training data",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="complex_models",
        help="Directory to save trained models",
    )
    parser.add_argument(
        "--num-problems", type=int, default=50, help="Number of problems to generate"
    )
    parser.add_argument(
        "--shape-type",
        type=str,
        choices=["random"],
        default="mixed",
        help="Type of shapes to generate",
    )
    parser.add_argument(
        "--block-counts",
        type=str,
        default="15,20,25,30,35,40",
        help="Comma-separated list of block counts",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Training batch size"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers for data generation",
    )

    args = parser.parse_args()

    # Parse block counts
    block_counts = [int(x) for x in args.block_counts.split(",")]

    # Create trainer
    trainer = ComplexShapeTrainer(
        output_dir=args.output_dir,
        model_dir=args.model_dir,
        grid_sizes=[(40, 40)],
        block_counts=block_counts,
        num_workers=args.workers,
    )

    # Generate training data
    dataset = trainer.generate_training_data(
        num_problems=args.num_problems, shape_type=args.shape_type
    )

    # Train model
    trainer.train_model(epochs=args.epochs, batch_size=args.batch_size)

    print("Training complete!")


if __name__ == "__main__":
    main()
