import pygame
import time
import numpy as np
from src.grid import Grid
from src.ai_agent import AI_Agent
import multiprocessing

# Define Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BLUE = (0, 102, 204)
GREEN = (0, 255, 0)  # Highlight selected block
RED = (255, 0, 0)  # AI execution mode color
ORANGE = (255, 165, 0)  # Target shape color
YELLOW = (255, 255, 0)  # Warning color
PURPLE = (128, 0, 128)  # Obstacle color
CYAN = (0, 255, 255)  # Highlight for waypoints in hierarchical planning
MAGENTA = (255, 0, 255)  # For multiprocessing indicators


class Visualizer:
    """
    Handles rendering the programmable matter grid using Pygame.
    Supports uniform, individual, and AI-based movement modes, obstacles,
    hierarchical planning visualization, and multiprocessing.
    """

    def __init__(
        self,
        grid: Grid,
        target_shape: list,
        obstacles: list = None,
        max_window_size: int = 900,
    ):
        """
        Initializes the Pygame visualization with dynamic cell sizing.

        Args:
            grid (Grid): The Grid object containing the simulation.
            target_shape (list): Target shape to be formed by the AI.
            obstacles (list): List of obstacle positions.
            max_window_size (int): Maximum window dimension in pixels.
        """
        self.grid = grid
        self.target_shape = target_shape
        self.obstacles = obstacles if obstacles is not None else []

        # Calculate appropriate cell size based on grid dimensions
        self.cell_size = self.calculate_cell_size(grid.n, grid.m, max_window_size)

        self.width = grid.m * self.cell_size
        self.height = grid.n * self.cell_size

        # Modes: "manual", "individual", "ai", "target_selection", "obstacle_placement"
        self.mode = "manual"
        self.selected_index = (
            0  # For individual mode, index of currently selected block.
        )

        # Use our optimized AI implementation with obstacles
        self.ai_agent = AI_Agent(
            grid.n, grid.m, self.grid.matter_elements, self.target_shape, self.obstacles
        )
        self.ai_plan = []
        self.ai_step = 0  # Track AI execution progress
        self.ai_delay = 0.01  # Speed control for AI execution
        self.message = ""  # For displaying status messages
        self.message_timer = 0  # For timing message display

        # Hierarchical planning visualization
        self.waypoints = []  # Store waypoints for visualization
        self.current_subproblem = 0  # Track current subproblem
        self.show_waypoints = False  # Toggle for waypoint visualization

        # Performance metrics
        self.plan_time = 0  # Time taken to plan
        self.plan_nodes = 0  # Nodes expanded during planning

        # Multiprocessing status
        self.multiprocessing_active = False
        self.process_count = 0
        self.processes_completed = 0
        self.process_results = {}

        pygame.init()
        # Get info about display
        info = pygame.display.Info()
        self.screen_width = min(self.width, info.current_w - 100)
        self.screen_height = min(self.height, info.current_h - 100)

        # Create window with calculated dimensions
        self.screen = pygame.display.set_mode(
            (self.screen_width, self.screen_height), pygame.RESIZABLE
        )
        pygame.display.set_caption(
            f"Programmable Matter Simulation - Cell Size: {self.cell_size}px"
        )

        # For scrolling in large grids
        self.scroll_x = 0
        self.scroll_y = 0
        self.scrolling = False
        self.scroll_start_pos = None

        # For animation
        self.last_frame_time = time.time()
        self.frame_rate = 30  # Target frame rate
        self.paused = False  # Animation pause toggle

        # Display initial size info
        print(
            f"Grid size: {grid.n}x{grid.m}, Cell size: {self.cell_size}px, Window: {self.screen_width}x{self.screen_height}px"
        )
        if self.obstacles:
            print(f"Obstacles: {len(self.obstacles)} positions")

    def calculate_cell_size(self, n, m, max_size):
        """
        Calculate an appropriate cell size based on grid dimensions.

        Args:
            n (int): Number of rows
            m (int): Number of columns
            max_size (int): Maximum window dimension

        Returns:
            int: Cell size in pixels
        """
        # Calculate cell size that would fit the grid within max_size
        cell_size_h = max_size // m
        cell_size_v = max_size // n

        # Take the smaller dimension to ensure both fit
        cell_size = min(cell_size_h, cell_size_v)

        # Ensure minimum cell size for visibility
        min_cell_size = 10
        cell_size = max(cell_size, min_cell_size)

        return cell_size

    def draw_grid(self):
        """
        Draws grid lines, matter elements, obstacles, waypoints, and target with support for scrolling.
        """
        self.screen.fill(WHITE)

        # Calculate visible area based on scroll position
        visible_cols_start = max(0, self.scroll_x // self.cell_size)
        visible_rows_start = max(0, self.scroll_y // self.cell_size)
        visible_cols_end = min(
            self.grid.m, visible_cols_start + self.screen_width // self.cell_size + 2
        )
        visible_rows_end = min(
            self.grid.n, visible_rows_start + self.screen_height // self.cell_size + 2
        )

        # Draw grid lines for visible area
        for col in range(visible_cols_start, visible_cols_end + 1):
            x = col * self.cell_size - self.scroll_x
            pygame.draw.line(self.screen, GRAY, (x, 0), (x, self.screen_height))

        for row in range(visible_rows_start, visible_rows_end + 1):
            y = row * self.cell_size - self.scroll_y
            pygame.draw.line(self.screen, GRAY, (0, y), (self.screen_width, y))

        # Draw obstacles in visible area
        for row, col in self.obstacles:
            if (
                visible_rows_start <= row < visible_rows_end
                and visible_cols_start <= col < visible_cols_end
            ):
                rect = pygame.Rect(
                    col * self.cell_size - self.scroll_x,
                    row * self.cell_size - self.scroll_y,
                    self.cell_size,
                    self.cell_size,
                )
                if self.mode == "obstacle_placement":
                    pygame.draw.rect(self.screen, PURPLE, rect)
                else:
                    # Solid obstacle
                    pygame.draw.rect(self.screen, PURPLE, rect)

        # Draw target shape in visible area
        for row, col in self.target_shape:
            if (
                visible_rows_start <= row < visible_rows_end
                and visible_cols_start <= col < visible_cols_end
            ):
                rect = pygame.Rect(
                    col * self.cell_size - self.scroll_x,
                    row * self.cell_size - self.scroll_y,
                    self.cell_size,
                    self.cell_size,
                )
                if self.mode == "target_selection":
                    pygame.draw.rect(self.screen, ORANGE, rect)
                else:
                    # Create transparent overlay
                    s = pygame.Surface(
                        (self.cell_size, self.cell_size), pygame.SRCALPHA
                    )
                    s.fill((255, 165, 0, 128))  # Semi-transparent orange
                    self.screen.blit(
                        s,
                        (
                            col * self.cell_size - self.scroll_x,
                            row * self.cell_size - self.scroll_y,
                        ),
                    )

        # Draw waypoints if enabled
        if self.show_waypoints and self.waypoints:
            for i, waypoint in enumerate(self.waypoints):
                # Draw each waypoint with different transparency
                for row, col in waypoint:
                    if (
                        visible_rows_start <= row < visible_rows_end
                        and visible_cols_start <= col < visible_cols_end
                    ):
                        # Use different colors for different waypoints
                        alpha = 128 - 10 * i  # Progressively more transparent
                        alpha = max(50, alpha)  # Ensure minimum visibility

                        s = pygame.Surface(
                            (self.cell_size, self.cell_size), pygame.SRCALPHA
                        )

                        # Current subproblem highlighted differently
                        if i == self.current_subproblem:
                            s.fill((0, 255, 255, alpha))  # Cyan
                        else:
                            s.fill((0, 255, 255, alpha // 2))  # Faded cyan

                        self.screen.blit(
                            s,
                            (
                                col * self.cell_size - self.scroll_x,
                                row * self.cell_size - self.scroll_y,
                            ),
                        )

        # Draw matter elements in visible area
        for i, (row, col) in enumerate(self.grid.matter_elements):
            if (
                visible_rows_start <= row < visible_rows_end
                and visible_cols_start <= col < visible_cols_end
            ):
                if self.mode == "ai":
                    color = RED  # AI mode highlight
                elif self.mode == "individual" and i == self.selected_index:
                    color = GREEN  # Selected block highlight
                else:
                    color = BLUE
                rect = pygame.Rect(
                    col * self.cell_size - self.scroll_x,
                    row * self.cell_size - self.scroll_y,
                    self.cell_size,
                    self.cell_size,
                )
                pygame.draw.rect(self.screen, color, rect)

        # Display current mode and grid info
        font = pygame.font.Font(None, 24)
        mode_text = font.render(
            f"Mode: {self.mode} | Grid: {self.grid.n}x{self.grid.m} | Cell: {self.cell_size}px",
            True,
            BLACK,
        )
        self.screen.blit(mode_text, (10, 10))

        # Add scroll info
        if (
            self.grid.n > self.screen_height // self.cell_size
            or self.grid.m > self.screen_width // self.cell_size
        ):
            scroll_text = font.render("Middle-click & drag to scroll", True, BLACK)
            self.screen.blit(scroll_text, (10, 35))

        # Add instructions based on current mode
        if self.mode == "target_selection":
            total_blocks = len(self.grid.matter_elements)
            selected_blocks = len(self.target_shape)
            color = GREEN if selected_blocks == total_blocks else YELLOW

            help_text = font.render(
                "Click to add/remove target cells. Press 'S' to save.", True, BLACK
            )
            self.screen.blit(help_text, (10, 60))

            count_text = font.render(
                f"Blocks: {selected_blocks}/{total_blocks}", True, color
            )
            self.screen.blit(count_text, (10, 85))

            if selected_blocks != total_blocks:
                warning_text = font.render(
                    "Target must have exactly the same number of blocks as the initial configuration",
                    True,
                    YELLOW,
                )
                self.screen.blit(warning_text, (10, 110))

        # Add instructions for obstacle placement mode
        elif self.mode == "obstacle_placement":
            help_text = font.render(
                "Click to add/remove obstacles. Press 'O' to save.", True, BLACK
            )
            self.screen.blit(help_text, (10, 60))

            count_text = font.render(f"Obstacles: {len(self.obstacles)}", True, PURPLE)
            self.screen.blit(count_text, (10, 85))

        # AI mode info
        elif self.mode == "ai":
            status = "PAUSED" if self.paused else "RUNNING"
            ai_text = font.render(
                f"AI Move: {self.ai_step}/{len(self.ai_plan)} | Delay: {self.ai_delay:.3f}s | {status}",
                True,
                RED,
            )
            self.screen.blit(ai_text, (10, 60))

            if self.waypoints and self.show_waypoints:
                wp_text = font.render(
                    f"Waypoint: {self.current_subproblem + 1}/{len(self.waypoints)}",
                    True,
                    CYAN,
                )
                self.screen.blit(wp_text, (10, 85))

            if self.plan_time > 0:
                perf_text = font.render(
                    f"Planning time: {self.plan_time:.2f}s | Nodes: {self.plan_nodes}",
                    True,
                    BLACK,
                )
                self.screen.blit(perf_text, (10, 110))

        # Help instructions in manual mode
        elif self.mode == "manual":
            help_text = font.render(
                "T: Regular A* | H: Hierarchical | P: Parallel A* | L: Parallel Hierarchical",
                True,
                BLACK,
            )
            self.screen.blit(help_text, (10, 60))

            keys_text = font.render(
                "B: Place Obstacles | M: Set Target | SPACE: Pause/Resume | W: Toggle Waypoints",
                True,
                BLACK,
            )
            self.screen.blit(keys_text, (10, 85))

        # Multiprocessing status if active
        if self.multiprocessing_active:
            mp_text = font.render(
                f"Multiprocessing: {self.processes_completed}/{self.process_count} processes completed",
                True,
                MAGENTA,
            )
            self.screen.blit(mp_text, (10, self.screen_height - 30))

        # Display temporary messages if present
        if self.message and self.message_timer > 0:
            msg_text = font.render(self.message, True, RED)
            self.screen.blit(msg_text, (self.screen_width // 2 - 200, 10))
            self.message_timer -= 1

        pygame.display.flip()

    def show_message(self, message, duration=60):
        """Display a temporary message on screen"""
        self.message = message
        self.message_timer = duration  # frames to display (60 = ~1 second at 60fps)

    def handle_mouse_click(self, pos, button):
        """Handle mouse clicks for target shape selection, obstacle placement, and scrolling"""
        if button == 1:  # Left click
            # Convert mouse position to grid coordinates including scroll
            col = (pos[0] + self.scroll_x) // self.cell_size
            row = (pos[1] + self.scroll_y) // self.cell_size

            # Ensure within grid bounds
            if 0 <= row < self.grid.n and 0 <= col < self.grid.m:
                cell = (row, col)

                # Target selection mode
                if self.mode == "target_selection":
                    # Add or remove the cell from target shape
                    if cell in self.target_shape:
                        self.target_shape.remove(cell)
                        print(f"Removed {cell} from target shape")
                    else:
                        # Check if adding would exceed the number of available blocks
                        if len(self.target_shape) >= len(self.grid.matter_elements):
                            self.show_message(
                                "Cannot add more blocks than available in the initial configuration"
                            )
                            print(
                                "Cannot add more blocks than available in the initial configuration"
                            )
                            return
                        self.target_shape.append(cell)
                        print(f"Added {cell} to target shape")

                # Obstacle placement mode
                elif self.mode == "obstacle_placement":
                    # Check if the position has matter or is a target
                    if cell in self.grid.matter_elements:
                        self.show_message("Cannot place obstacle on matter element")
                        return
                    if cell in self.target_shape:
                        self.show_message("Cannot place obstacle on target position")
                        return

                    # Add or remove obstacle
                    if cell in self.obstacles:
                        self.obstacles.remove(cell)
                        print(f"Removed obstacle at {cell}")
                    else:
                        self.obstacles.append(cell)
                        print(f"Added obstacle at {cell}")

        elif button == 2:  # Middle click
            # Start scrolling
            self.scrolling = True
            self.scroll_start_pos = pos

    def handle_mouse_motion(self, pos, rel, buttons):
        """Handle mouse motion for scrolling"""
        if self.scrolling and buttons[1]:  # Middle button held down
            # Update scroll position
            self.scroll_x = max(
                0, min(self.scroll_x - rel[0], self.width - self.screen_width)
            )
            self.scroll_y = max(
                0, min(self.scroll_y - rel[1], self.height - self.screen_height)
            )
        else:
            self.scrolling = False

    def handle_mouse_wheel(self, y):
        """Handle mouse wheel for scrolling"""
        scroll_amount = 3 * self.cell_size
        if y > 0:
            self.scroll_y = max(0, self.scroll_y - scroll_amount)
        else:
            self.scroll_y = min(
                self.height - self.screen_height, self.scroll_y + scroll_amount
            )

    def execute_ai_plan(self):
        """
        Executes the AI-generated plan step by step.
        Ensures that all individual moves for a given step are executed as one batch.
        """
        if self.paused:
            return

        if self.ai_step < len(self.ai_plan):
            move_set = self.ai_plan[self.ai_step]

            # Ensure move_set is a list of moves
            if isinstance(move_set, tuple) and len(move_set) == 3:
                move_set = [move_set]  # Convert single move into a list

            if not isinstance(move_set, list) or not all(
                len(move) == 3 for move in move_set
            ):
                print(f"Error: Unexpected move format: {move_set}")
                return

            # Group all individual moves into one dictionary
            moves = {i: (dx, dy) for i, dx, dy in move_set}

            # Print to check grouped moves
            print(
                f"Executing shape-changing move {self.ai_step + 1}/{len(self.ai_plan)}: {moves}"
            )

            success = self.grid.move_individual(moves)

            if not success:
                print(
                    "Invalid move! Removing it from the plan and retrying a different approach..."
                )
                self.ai_plan.pop(self.ai_step)  # Remove failed move
                return  # Do not increment `ai_step`, retry from the same step

            # Center view on the shape if shape is outside view
            self.center_view_on_shape()

            # Update current subproblem if using waypoints
            if self.waypoints and self.current_subproblem < len(self.waypoints) - 1:
                # Check if we've reached the next waypoint
                next_waypoint = self.waypoints[self.current_subproblem + 1]
                current_state = self.grid.matter_elements

                # Simple check - if most blocks are close to waypoint, advance
                matches = sum(1 for pos in current_state if pos in next_waypoint)
                if matches >= len(next_waypoint) * 0.7:  # 70% match threshold
                    self.current_subproblem += 1
                    print(
                        f"Reached waypoint {self.current_subproblem + 1}/{len(self.waypoints)}"
                    )

            self.ai_step += 1
        else:
            print("AI execution complete!")
            self.mode = "manual"

        # Maintain consistent timing
        now = time.time()
        elapsed = now - self.last_frame_time
        wait_time = max(0, self.ai_delay - elapsed)
        time.sleep(wait_time)
        self.last_frame_time = now + wait_time

    def center_view_on_shape(self):
        """Center the view on the current shape"""
        if not self.grid.matter_elements:
            return

        # Calculate shape bounds
        min_row = min(row for row, col in self.grid.matter_elements)
        max_row = max(row for row, col in self.grid.matter_elements)
        min_col = min(col for row, col in self.grid.matter_elements)
        max_col = max(col for row, col in self.grid.matter_elements)

        # Calculate center of shape
        center_row = (min_row + max_row) // 2
        center_col = (min_col + max_col) // 2

        # Center view on shape
        self.scroll_x = max(0, center_col * self.cell_size - self.screen_width // 2)
        self.scroll_y = max(0, center_row * self.cell_size - self.screen_height // 2)

        # Ensure we don't scroll beyond grid bounds
        self.scroll_x = min(self.scroll_x, self.width - self.screen_width)
        self.scroll_y = min(self.scroll_y, self.height - self.screen_height)

    def run_hierarchical_planning(self):
        """Run hierarchical planning with timing and show results"""
        print("Starting hierarchical planning...")
        self.show_message("Computing hierarchical plan...")

        # Update AI agent with current state
        self.ai_agent = AI_Agent(
            self.grid.n,
            self.grid.m,
            self.grid.matter_elements,
            self.target_shape,
            self.obstacles,
        )

        # Measure planning time
        start_time = time.time()

        # Run hierarchical planning
        plan, waypoints = self.ai_agent.hierarchical_plan()

        end_time = time.time()
        self.plan_time = end_time - start_time
        self.plan_nodes = self.ai_agent.nodes_expanded

        if plan:
            print(
                f"Hierarchical planning successful: {len(plan)} moves in {self.plan_time:.2f}s"
            )
            self.ai_plan = plan
            self.ai_step = 0
            self.mode = "ai"

            # Store waypoints for visualization
            self.waypoints = waypoints
            self.current_subproblem = 0
            self.show_waypoints = True

            self.show_message(f"Plan found: {len(plan)} moves in {self.plan_time:.2f}s")
            return True
        else:
            print("Hierarchical planning failed")
            self.show_message("Hierarchical planning failed to find a solution")
            return False

    def run_normal_planning(self):
        """Run normal A* planning with timing"""
        print("Starting normal A* planning...")
        self.show_message("Computing plan with A*...")

        # Update AI agent with current state
        self.ai_agent = AI_Agent(
            self.grid.n,
            self.grid.m,
            self.grid.matter_elements,
            self.target_shape,
            self.obstacles,
        )

        # Measure planning time
        start_time = time.time()
        plan = self.ai_agent.plan()
        end_time = time.time()

        self.plan_time = end_time - start_time
        self.plan_nodes = self.ai_agent.nodes_expanded

        if plan:
            print(f"A* planning successful: {len(plan)} moves in {self.plan_time:.2f}s")
            self.ai_plan = plan
            self.ai_step = 0
            self.mode = "ai"
            self.show_message(f"Plan found: {len(plan)} moves in {self.plan_time:.2f}s")
            return True
        else:
            print("A* planning failed")
            self.show_message("A* planning failed to find a solution")
            return False

    def run_parallel_planning(self):
        """Run parallel A* planning with timing"""
        print("Starting parallel A* planning...")
        self.show_message("Computing plan with parallel A*...")

        # Update AI agent with current state
        self.ai_agent = AI_Agent(
            self.grid.n,
            self.grid.m,
            self.grid.matter_elements,
            self.target_shape,
            self.obstacles,
        )

        # Set multiprocessing active
        self.multiprocessing_active = True
        self.process_count = min(8, multiprocessing.cpu_count())
        self.processes_completed = 0

        # Measure planning time
        start_time = time.time()
        plan = self.ai_agent.plan_parallel()
        end_time = time.time()

        self.multiprocessing_active = False
        self.plan_time = end_time - start_time
        self.plan_nodes = self.ai_agent.nodes_expanded

        if plan:
            print(
                f"Parallel A* planning successful: {len(plan)} moves in {self.plan_time:.2f}s"
            )
            self.ai_plan = plan
            self.ai_step = 0
            self.mode = "ai"
            self.show_message(f"Plan found: {len(plan)} moves in {self.plan_time:.2f}s")
            return True
        else:
            print("Parallel A* planning failed")
            self.show_message("Parallel A* planning failed to find a solution")
            return False

    def run_parallel_hierarchical_planning(self):
        """Run parallel hierarchical planning with timing"""
        print("Starting parallel hierarchical planning...")
        self.show_message("Computing parallel hierarchical plan...")

        # Update AI agent with current state
        self.ai_agent = AI_Agent(
            self.grid.n,
            self.grid.m,
            self.grid.matter_elements,
            self.target_shape,
            self.obstacles,
        )

        # Set multiprocessing active
        self.multiprocessing_active = True
        self.process_count = min(8, multiprocessing.cpu_count())
        self.processes_completed = 0

        # Measure planning time
        start_time = time.time()
        plan, waypoints = self.ai_agent.hierarchical_plan_parallel()
        end_time = time.time()

        self.multiprocessing_active = False
        self.plan_time = end_time - start_time
        self.plan_nodes = self.ai_agent.nodes_expanded

        if plan:
            print(
                f"Parallel hierarchical planning successful: {len(plan)} moves in {self.plan_time:.2f}s"
            )
            self.ai_plan = plan
            self.ai_step = 0
            self.mode = "ai"

            # Store waypoints for visualization
            self.waypoints = waypoints
            self.current_subproblem = 0
            self.show_waypoints = True

            self.show_message(f"Plan found: {len(plan)} moves in {self.plan_time:.2f}s")
            return True
        else:
            print("Parallel hierarchical planning failed")
            self.show_message(
                "Parallel hierarchical planning failed to find a solution"
            )
            return False

    def run(self):
        """
        Main loop for visualization with keyboard input.
        Supports AI execution alongside manual control.
        """
        running = True
        while running:
            self.draw_grid()

            if self.mode == "ai" and not self.paused:
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
                    print(f"Key pressed: {pygame.key.name(event.key)}")

                    if event.key == pygame.K_ESCAPE:
                        running = False

                    if event.key == pygame.K_g:
                        self.grid.display_grid()

                    if event.key == pygame.K_SPACE:
                        # Toggle pause/resume
                        self.paused = not self.paused
                        if self.paused:
                            print("Animation paused")
                            self.show_message("Animation paused")
                        else:
                            print("Animation resumed")
                            self.show_message("Animation resumed")

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

                    # Toggle obstacle placement mode
                    if event.key == pygame.K_b:
                        if self.mode != "obstacle_placement":
                            self.mode = "obstacle_placement"
                            print("Obstacle placement mode activated")
                        else:
                            self.mode = "manual"
                            print("Manual mode activated")

                    # Toggle waypoint visibility
                    if event.key == pygame.K_w:
                        self.show_waypoints = not self.show_waypoints
                        print(f"Waypoint visibility: {self.show_waypoints}")

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
                            # Update AI agent with new target shape and obstacles
                            self.ai_agent = AI_Agent(
                                self.grid.n,
                                self.grid.m,
                                self.grid.matter_elements,
                                self.target_shape,
                                self.obstacles,
                            )
                            self.mode = "manual"
                            self.show_message("Target shape saved successfully!")

                    # Save obstacles
                    if event.key == pygame.K_o and self.mode == "obstacle_placement":
                        print(f"Obstacles saved: {len(self.obstacles)} positions")
                        # Update grid and AI agent with new obstacles
                        self.grid.obstacles = set(self.obstacles)
                        self.ai_agent = AI_Agent(
                            self.grid.n,
                            self.grid.m,
                            self.grid.matter_elements,
                            self.target_shape,
                            self.obstacles,
                        )
                        self.mode = "manual"
                        self.show_message(
                            f"{len(self.obstacles)} obstacles saved successfully!"
                        )

                    # Run normal A* planning
                    if event.key == pygame.K_t:
                        # First check if target shape has the correct number of blocks
                        if len(self.target_shape) != len(self.grid.matter_elements):
                            self.show_message(
                                f"Target must have exactly {len(self.grid.matter_elements)} blocks (currently has {len(self.target_shape)})"
                            )
                            print(
                                f"Target must have exactly {len(self.grid.matter_elements)} blocks"
                            )
                            continue

                        self.run_normal_planning()

                    # Run hierarchical planning
                    if event.key == pygame.K_h:
                        # First check if target shape has the correct number of blocks
                        if len(self.target_shape) != len(self.grid.matter_elements):
                            self.show_message(
                                f"Target must have exactly {len(self.grid.matter_elements)} blocks (currently has {len(self.target_shape)})"
                            )
                            print(
                                f"Target must have exactly {len(self.grid.matter_elements)} blocks"
                            )
                            continue

                        self.run_hierarchical_planning()

                    # Run parallel A* planning
                    if event.key == pygame.K_p:
                        # First check if target shape has the correct number of blocks
                        if len(self.target_shape) != len(self.grid.matter_elements):
                            self.show_message(
                                f"Target must have exactly {len(self.grid.matter_elements)} blocks (currently has {len(self.target_shape)})"
                            )
                            print(
                                f"Target must have exactly {len(self.grid.matter_elements)} blocks"
                            )
                            continue

                        self.run_parallel_planning()

                    # Run parallel hierarchical planning
                    if event.key == pygame.K_l:
                        # First check if target shape has the correct number of blocks
                        if len(self.target_shape) != len(self.grid.matter_elements):
                            self.show_message(
                                f"Target must have exactly {len(self.grid.matter_elements)} blocks (currently has {len(self.target_shape)})"
                            )
                            print(
                                f"Target must have exactly {len(self.grid.matter_elements)} blocks"
                            )
                            continue

                        self.run_parallel_hierarchical_planning()

                    # Toggle between manual and individual block mode
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

            # Cap the frame rate
            pygame.time.delay(33)  # ~30 FPS
        pygame.quit()
