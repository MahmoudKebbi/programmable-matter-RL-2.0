import pygame
import time
import argparse
import os
import sys
import json
from typing import List, Tuple, Optional, Dict

# Import the original AI agent and grid
from grid import Grid
from ai_agent import AI_Agent

# Import our hybrid agents
from hybrid_agent import HybridAgent, AdaptiveHybridAgent

# We'll use the original visualizer code as a base
from visualizer import Visualizer as BaseVisualizer

# Define Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BLUE = (0, 102, 204)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
ORANGE = (255, 165, 0)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)  # For hybrid agent


class HybridVisualizer(BaseVisualizer):
    """
    Extended visualizer that supports both the original A* agent and our hybrid ML agent.
    Inherits from the original visualizer to maintain compatibility.
    """

    def __init__(
        self,
        grid: Grid,
        target_shape: list,
        model_path: str = None,
        agent_type: str = "astar",
        max_window_size: int = 900,
    ):
        # Initialize the base visualizer
        super().__init__(grid, target_shape, max_window_size)

        # Store ML model path and agent type
        self.model_path = model_path
        self.agent_type = agent_type

        # Add information about ML agent - MOVED THIS BEFORE setup_agents()
        self.ml_enabled = model_path is not None and os.path.exists(model_path)
        if not self.ml_enabled and agent_type != "astar":
            print(
                f"Warning: ML model not found at {model_path}. Falling back to A* agent."
            )
            self.agent_type = "astar"

        # Initialize agents - MOVED AFTER ml_enabled is defined
        self.setup_agents()

        # Additional keyboard shortcuts help
        self.agent_help_displayed = False

    def setup_agents(self):
        """Initialize the appropriate agent based on agent_type."""
        # Initialize A* agent by default (keeping original behavior)
        self.ai_agent = AI_Agent(
            self.grid.n, self.grid.m, self.grid.matter_elements, self.target_shape
        )

        # Initialize ML agents if model path is provided
        if self.ml_enabled:
            if self.agent_type == "hybrid":
                self.hybrid_agent = HybridAgent(
                    self.grid.n,
                    self.grid.m,
                    self.grid.matter_elements,
                    self.target_shape,
                    model_path=self.model_path,
                    ml_weight=0.7,
                    use_move_predictions=True,
                    smart_pruning=True,
                    verbose=True,
                )
            elif self.agent_type == "adaptive":
                self.hybrid_agent = AdaptiveHybridAgent(
                    self.grid.n,
                    self.grid.m,
                    self.grid.matter_elements,
                    self.target_shape,
                    model_path=self.model_path,
                    verbose=True,
                )

    def update_agents(self):
        """Update agents with current grid state."""
        # Update A* agent
        self.ai_agent = AI_Agent(
            self.grid.n, self.grid.m, self.grid.matter_elements, self.target_shape
        )

        # Update ML agent if enabled
        if self.ml_enabled:
            agent_class = (
                AdaptiveHybridAgent if self.agent_type == "adaptive" else HybridAgent
            )
            self.hybrid_agent = agent_class(
                self.grid.n,
                self.grid.m,
                self.grid.matter_elements,
                self.target_shape,
                model_path=self.model_path,
                verbose=True,
            )

    def toggle_agent_type(self):
        """Toggle between different agent types."""
        if not self.ml_enabled:
            self.show_message("ML model not loaded. Using A* agent only.", 120)
            return

        if self.agent_type == "astar":
            self.agent_type = "hybrid"
        elif self.agent_type == "hybrid":
            self.agent_type = "adaptive"
        else:
            self.agent_type = "astar"

        self.show_message(f"Switched to {self.agent_type.upper()} agent", 120)
        self.setup_agents()

    def draw_grid(self):
        """
        Override draw_grid to add ML agent information.
        """
        # Call the original draw_grid method
        super().draw_grid()

        # Add ML agent information if ML is enabled
        if self.ml_enabled:
            font = pygame.font.Font(None, 24)

            # Add agent type info with appropriate color
            if self.agent_type == "astar":
                agent_color = RED
                agent_text = "A* (Original)"
            elif self.agent_type == "hybrid":
                agent_color = PURPLE
                agent_text = "ML Hybrid"
            else:
                agent_color = GREEN
                agent_text = "ML Adaptive"

            agent_info = font.render(f"Agent: {agent_text}", True, agent_color)
            self.screen.blit(agent_info, (self.screen_width - 180, 10))

            # Show agent selection help
            if not self.agent_help_displayed:
                help_text = font.render("Press 'L' to switch agent types", True, BLACK)
                self.screen.blit(help_text, (self.screen_width - 300, 35))

                # Only show once until toggled
                self.agent_help_displayed = True

    def run_selected_agent(self):
        """Run the currently selected agent."""
        print(f"Using {self.agent_type.upper()} agent to find solution...")

        # Reset AI execution state
        self.ai_plan = []
        self.ai_step = 0

        # Display AI computation message
        self.show_message("Computing solution...", 300)
        self.draw_grid()
        pygame.display.flip()

        # Verify correct target shape
        if len(self.target_shape) != len(self.grid.matter_elements):
            self.show_message(
                f"Target must have exactly {len(self.grid.matter_elements)} blocks", 300
            )
            print(f"Target must have exactly {len(self.grid.matter_elements)} blocks")
            self.mode = "manual"
            return

        # Choose the appropriate agent and run plan
        start_time = time.time()

        if self.agent_type == "astar":
            # Use original A* agent (keeping original behavior)
            solution = self.ai_agent.plan()
        else:
            # Use hybrid agent
            solution = self.hybrid_agent.plan()

        execution_time = time.time() - start_time

        # Process the result
        if solution:
            print(
                f"Solution found with {len(solution)} moves in {execution_time:.2f} seconds"
            )
            self.ai_plan = solution
            self.mode = "ai"
            self.ai_step = 0
            self.show_message(f"Solution found: {len(solution)} moves", 180)
        else:
            print("No valid solution found.")
            self.show_message("No valid solution found", 180)

    def run(self):
        """
        Main loop for visualization with keyboard input.
        Extends the original run method with hybrid agent support.
        """
        running = True
        while running:
            self.draw_grid()

            if self.mode == "ai":
                self.execute_ai_plan()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Handle mouse button press
                    self.handle_mouse_click(event.pos, event.button)

                    # Handle mouse wheel
                    if event.button == 4 or event.button == 5:
                        self.handle_mouse_wheel(1 if event.button == 4 else -1)

                elif event.type == pygame.MOUSEMOTION:
                    # Handle mouse motion for scrolling
                    self.handle_mouse_motion(
                        event.pos, event.rel, pygame.mouse.get_pressed()
                    )

                elif event.type == pygame.MOUSEBUTTONUP:
                    # End scrolling
                    if event.button == 2:
                        self.scrolling = False

                elif event.type == pygame.VIDEORESIZE:
                    # Handle window resize
                    self.screen_width = event.w
                    self.screen_height = event.h
                    self.screen = pygame.display.set_mode(
                        (self.screen_width, self.screen_height), pygame.RESIZABLE
                    )

                elif event.type == pygame.KEYDOWN:
                    print(f"Key pressed: {event.key}")

                    if event.key == pygame.K_ESCAPE:
                        running = False

                    if event.key == pygame.K_g:
                        self.grid.display_grid()

                    # Center view on shape
                    if event.key == pygame.K_HOME:
                        self.center_view_on_shape()

                    # Animation speed controls
                    if event.key == pygame.K_PLUS or event.key == pygame.K_KP_PLUS:
                        self.ai_delay = max(0.001, self.ai_delay / 2)
                        print(f"Animation speed increased: delay = {self.ai_delay}s")

                    if event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
                        self.ai_delay = min(1.0, self.ai_delay * 2)
                        print(f"Animation speed decreased: delay = {self.ai_delay}s")

                    # Toggle target selection mode
                    if event.key == pygame.K_m:
                        if self.mode != "target_selection":
                            self.mode = "target_selection"
                            print("Target shape selection mode activated")
                        else:
                            self.mode = "manual"
                            print("Manual mode activated")

                    # Save target shape and update AI
                    if event.key == pygame.K_s and self.mode == "target_selection":
                        # Verify that target shape has exactly the same number of blocks
                        if len(self.target_shape) != len(self.grid.matter_elements):
                            self.show_message(
                                f"Target must have exactly {len(self.grid.matter_elements)} blocks (currently has {len(self.target_shape)})"
                            )
                            print(
                                f"Target must have exactly {len(self.grid.matter_elements)} blocks"
                            )
                        else:
                            print(f"Target shape saved: {self.target_shape}")
                            # Update all agents with new target shape
                            self.update_agents()
                            self.mode = "manual"
                            self.show_message("Target shape saved successfully!")

                    if event.key == pygame.K_t:  # Activate AI mode
                        # First check if target shape has the correct number of blocks
                        if len(self.target_shape) != len(self.grid.matter_elements):
                            self.show_message(
                                f"Target must have exactly {len(self.grid.matter_elements)} blocks (currently has {len(self.target_shape)})"
                            )
                            print(
                                f"Target must have exactly {len(self.grid.matter_elements)} blocks"
                            )
                            continue

                        # Run selected agent
                        self.run_selected_agent()

                    # NEW: Toggle between agent types
                    if event.key == pygame.K_l:
                        self.toggle_agent_type()
                        self.agent_help_displayed = False  # Show help again

                    if event.key == pygame.K_TAB:
                        if self.mode == "manual":
                            self.mode = "individual"
                        elif self.mode == "individual":
                            self.mode = "manual"

                    if self.mode == "manual":
                        # Move the entire structure
                        if event.key == pygame.K_UP:
                            self.grid.move(-1, 0)
                        elif event.key == pygame.K_DOWN:
                            self.grid.move(1, 0)
                        elif event.key == pygame.K_LEFT:
                            self.grid.move(0, -1)
                        elif event.key == pygame.K_RIGHT:
                            self.grid.move(0, 1)
                        elif event.key == pygame.K_q:  # Diagonal up-left
                            self.grid.move(-1, -1)
                        elif event.key == pygame.K_e:  # Diagonal up-right
                            self.grid.move(-1, 1)
                        elif event.key == pygame.K_z:  # Diagonal down-left
                            self.grid.move(1, -1)
                        elif event.key == pygame.K_c:  # Diagonal down-right
                            self.grid.move(1, 1)

                    elif self.mode == "individual":
                        # In individual mode, use R and F to cycle through blocks.
                        if event.key == pygame.K_r:
                            self.selected_index = (self.selected_index - 1) % len(
                                self.grid.matter_elements
                            )
                        elif event.key == pygame.K_f:
                            self.selected_index = (self.selected_index + 1) % len(
                                self.grid.matter_elements
                            )

                        # Move selected block.
                        moves = {}
                        if event.key == pygame.K_w:
                            moves[self.selected_index] = (-1, 0)
                        elif event.key == pygame.K_s:
                            moves[self.selected_index] = (1, 0)
                        elif event.key == pygame.K_a:
                            moves[self.selected_index] = (0, -1)
                        elif event.key == pygame.K_d:
                            moves[self.selected_index] = (0, 1)
                        elif event.key == pygame.K_q:
                            moves[self.selected_index] = (-1, -1)
                        elif event.key == pygame.K_e:
                            moves[self.selected_index] = (-1, 1)
                        elif event.key == pygame.K_z:
                            moves[self.selected_index] = (1, -1)
                        elif event.key == pygame.K_c:
                            moves[self.selected_index] = (1, 1)

                        if moves:
                            self.grid.move_individual(moves)

            pygame.time.delay(30)  # Reduced delay for more responsive UI
        pygame.quit()


def load_or_generate_shape(n, m, blocks=5, filename=None):
    """Load a shape from file or generate one."""
    if filename and os.path.exists(filename):
        with open(filename, "r") as f:
            data = json.load(f)
        return data.get("shape", [])

    # Generate a random shape
    import random

    shape = []
    grid = [[False for _ in range(m)] for _ in range(n)]

    # Start with a random position near center
    x, y = n // 2, m // 2
    shape.append((x, y))
    grid[x][y] = True

    # Add remaining blocks ensuring connectivity
    block_count = 1
    while block_count < blocks:
        # Choose a random existing block
        parent_idx = random.randint(0, len(shape) - 1)
        px, py = shape[parent_idx]

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
            if 0 <= nx < n and 0 <= ny < m and not grid[nx][ny]:
                shape.append((nx, ny))
                grid[nx][ny] = True
                block_count += 1
                break

        if block_count >= blocks:
            break

    # Save the shape if filename provided
    if filename:
        with open(filename, "w") as f:
            json.dump({"shape": shape}, f)

    return shape


def main():
    """Main function to run the hybrid visualizer."""
    parser = argparse.ArgumentParser(
        description="Programmable Matter Hybrid Visualizer"
    )
    parser.add_argument(
        "--grid-size",
        type=str,
        default="40x40",
        help="Grid size in format WIDTHxHEIGHT",
    )
    parser.add_argument(
        "--blocks", type=int, default=30, help="Number of blocks for initial shape"
    )
    parser.add_argument(
        "--shape-file",
        type=str,
        default=None,
        help="JSON file containing initial shape",
    )
    parser.add_argument(
        "--target-file",
        type=str,
        default=None,
        help="JSON file containing target shape",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="complex_models/best_model.pt",
        help="Path to trained ML model",
    )
    parser.add_argument(
        "--agent",
        type=str,
        choices=["astar", "hybrid", "adaptive"],
        default="hybrid",
        help="Initial agent type",
    )

    args = parser.parse_args()

    # Parse grid size
    try:
        width, height = map(int, args.grid_size.split("x"))
    except ValueError:
        print(f"Invalid grid size format: {args.grid_size}. Using default 40x40.")
        width, height = 40, 40

    # Load or generate initial shape
    initial_shape = load_or_generate_shape(width, height, args.blocks, args.shape_file)

    # Load or generate target shape (default to None, will be set in the visualizer)
    target_shape = load_or_generate_shape(width, height, args.blocks, args.target_file)

    # Create grid and visualizer
    grid = Grid(width, height, initial_shape)

    # Ensure model exists and warn if not
    if not os.path.exists(args.model) and args.agent != "astar":
        print(f"Warning: ML model not found at {args.model}. Falling back to A* agent.")
        args.agent = "astar"

    # Create and run visualizer
    visualizer = HybridVisualizer(
        grid=grid,
        target_shape=target_shape,
        model_path=args.model,
        agent_type=args.agent,
    )

    print("\nKeybindings:")
    print("  T - Run AI to find solution")
    print("  L - Switch between agent types (A*, Hybrid ML, Adaptive ML)")
    print("  M - Toggle target shape selection mode")
    print("  S - Save target shape (in target selection mode)")
    print("  +/- - Adjust animation speed")
    print("  ESC - Exit")

    visualizer.run()


if __name__ == "__main__":
    main()
