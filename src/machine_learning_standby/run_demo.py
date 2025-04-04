import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import os
import json

from grid import Grid
from ai_agent import AI_Agent
from hybrid_agent import HybridAgent, AdaptiveHybridAgent


def visualize_solution(n, m, start_state, target_state, solution, output_path=None):
    """
    Create an animated visualization of the solution.

    Args:
        n, m: Grid dimensions
        start_state: Initial block positions
        target_state: Target block positions
        solution: List of moves
        output_path: Path to save the animation (optional)
    """
    # Initialize grid
    grid = Grid(n, m, start_state)

    # Set up the figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Programmable Matter Simulation", fontsize=16)

    # Custom colormap
    colors = ["#ffffff", "#3498db", "#2ecc71", "#e74c3c"]
    cmap = ListedColormap(colors)

    # For initial state
    axes[0].set_title("Current State")
    current_grid_plot = axes[0].imshow(grid.grid, cmap=cmap, vmin=0, vmax=3)
    axes[0].set_xticks(range(m))
    axes[0].set_yticks(range(n))
    axes[0].grid(color="gray", linestyle="-", linewidth=0.5)

    # For target state
    target_grid = np.zeros((n, m), dtype=int)
    for x, y in target_state:
        target_grid[x, y] = 2  # Use different color for target

    axes[1].set_title("Target State")
    target_grid_plot = axes[1].imshow(target_grid, cmap=cmap, vmin=0, vmax=3)
    axes[1].set_xticks(range(m))
    axes[1].set_yticks(range(n))
    axes[1].grid(color="gray", linestyle="-", linewidth=0.5)

    # Set text labels for move information
    move_text = fig.text(0.5, 0.04, "Initial state", ha="center", fontsize=12)

    # Create frames for animation
    frames = []
    frames.append((grid.grid.copy(), "Initial state"))

    # Apply each move set
    for i, move_set in enumerate(solution):
        # Apply moves
        for block_idx, dx, dy in move_set:
            x, y = grid.matter_elements[block_idx]
            grid.grid[x, y] = 0
            grid.matter_elements[block_idx] = (x + dx, y + dy)
            grid.grid[x + dx, y + dy] = 1

        # Store current state and move info
        move_description = f"Move {i+1}: " + ", ".join(
            [f"Block {idx}→({dx},{dy})" for idx, dx, dy in move_set]
        )
        frames.append((grid.grid.copy(), move_description))

        # Highlight target positions that have been reached
        overlap_grid = grid.grid.copy()
        for i, (x, y) in enumerate(grid.matter_elements):
            if (x, y) in target_state:
                overlap_grid[x, y] = (
                    3  # Use special color for blocks that reached target position
                )

        frames.append((overlap_grid, move_description + " (checking targets)"))

    # Animation function
    def update_frame(frame_idx):
        grid_state, description = frames[frame_idx]
        current_grid_plot.set_array(grid_state)
        move_text.set_text(description)
        return [current_grid_plot, move_text]

    # Create animation
    anim = animation.FuncAnimation(
        fig, update_frame, frames=len(frames), interval=1000, blit=False
    )

    # Save animation if path provided
    if output_path:
        anim.save(output_path, writer="pillow", fps=1)
        print(f"Animation saved to {output_path}")

    plt.tight_layout()
    plt.show()


def run_example(
    n,
    m,
    start_state,
    target_state,
    model_path=None,
    agent_type="hybrid",
    visualize=True,
):
    """
    Run a programmable matter example with the specified agent.

    Args:
        n, m: Grid dimensions
        start_state: Initial block positions
        target_state: Target block positions
        model_path: Path to the trained model (required for hybrid/adaptive agents)
        agent_type: Type of agent to use ('astar', 'hybrid', or 'adaptive')
        visualize: Whether to create visualization
    """
    print(f"Running {agent_type} agent on {n}x{m} grid with {len(start_state)} blocks")

    # Initialize the appropriate agent
    if agent_type == "astar":
        agent = AI_Agent(n, m, start_state, target_state)
    elif agent_type == "hybrid":
        if model_path is None:
            raise ValueError("Model path is required for hybrid agent")
        agent = HybridAgent(
            n,
            m,
            start_state,
            target_state,
            model_path=model_path,
            ml_weight=0.7,
            use_move_predictions=True,
            smart_pruning=True,
            verbose=True,
        )
    elif agent_type == "adaptive":
        if model_path is None:
            raise ValueError("Model path is required for adaptive agent")
        agent = AdaptiveHybridAgent(
            n, m, start_state, target_state, model_path=model_path, verbose=True
        )
    else:
        raise ValueError(f"Invalid agent type: {agent_type}")

    # Time the solution
    start_time = time.time()
    solution = agent.plan()
    elapsed_time = time.time() - start_time

    # Report results
    if solution:
        print(
            f"Solution found with {len(solution)} moves in {elapsed_time:.2f} seconds"
        )
        print(f"Nodes expanded: {agent.nodes_expanded}")

        # Visualize if requested
        if visualize:
            visualize_solution(
                n,
                m,
                start_state,
                target_state,
                solution,
                output_path=f"{agent_type}_solution.gif",
            )

        return solution, elapsed_time, agent.nodes_expanded
    else:
        print(f"No solution found after {elapsed_time:.2f} seconds")
        print(f"Nodes expanded: {agent.nodes_expanded}")
        return None, elapsed_time, agent.nodes_expanded


def load_example(example_file):
    """Load an example from a JSON file."""
    with open(example_file, "r") as f:
        data = json.load(f)

    n, m = data["grid_size"]
    start_state = [(x, y) for x, y in data["start_state"]]
    target_state = [(x, y) for x, y in data["target_state"]]

    return n, m, start_state, target_state


def create_example_file():
    """Create and save a sample example file."""
    example = {
        "grid_size": [5, 5],
        "start_state": [[0, 0], [1, 0], [1, 1]],  # L shape at top-left
        "target_state": [[3, 3], [4, 3], [4, 4]],  # L shape at bottom-right
    }

    os.makedirs("examples", exist_ok=True)
    with open("examples/sample_example.json", "w") as f:
        json.dump(example, f, indent=2)

    print("Sample example created at examples/sample_example.json")


def main():
    parser = argparse.ArgumentParser(description="Run programmable matter simulations")
    parser.add_argument("--example", type=str, help="Path to example JSON file")
    parser.add_argument(
        "--model",
        type=str,
        default="saved_models/best_model.pt",
        help="Path to trained model for hybrid/adaptive agents",
    )
    parser.add_argument(
        "--agent",
        type=str,
        choices=["astar", "hybrid", "adaptive"],
        default="hybrid",
        help="Agent type to use",
    )
    parser.add_argument("--no-viz", action="store_true", help="Disable visualization")
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create a sample example file and exit",
    )

    args = parser.parse_args()

    if args.create_sample:
        create_example_file()
        return

    if args.example:
        n, m, start_state, target_state = load_example(args.example)
    else:
        # Default example: L shape
        n, m = 5, 5
        start_state = [(0, 0), (1, 0), (1, 1)]  # L shape
        target_state = [(3, 3), (3, 4), (4, 3)]  # L shape at new position

    run_example(
        n,
        m,
        start_state,
        target_state,
        model_path=args.model,
        agent_type=args.agent,
        visualize=not args.no_viz,
    )


if __name__ == "__main__":
    main()
