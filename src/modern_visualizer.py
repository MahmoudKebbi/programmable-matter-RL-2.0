import random
import pygame
import time
import numpy as np
import os
import math
from pygame import gfxdraw
from src.grid import Grid
from src.puzzle_solver import AI_Agent
import multiprocessing
import pygame.freetype

# Initialize pygame before other imports
pygame.init()


# Enhanced colors with alpha support
class Colors:
    WHITE = (255, 255, 255)
    BLACK = (20, 20, 20)
    GRAY = (100, 100, 110)
    LIGHT_GRAY = (220, 220, 230)
    BLUE = (41, 128, 185)
    BLUE_DARK = (28, 89, 130)
    GREEN = (39, 174, 96)
    RED = (231, 76, 60)
    ORANGE = (230, 126, 34)
    YELLOW = (241, 196, 15)
    PURPLE = (142, 68, 173)
    CYAN = (52, 152, 219)
    MAGENTA = (155, 89, 182)
    BACKGROUND = (245, 247, 250)
    GRID_LINES = (220, 220, 230)
    TEXT_PRIMARY = (44, 62, 80)
    TEXT_SECONDARY = (127, 140, 141)
    TRANSLUCENT_BLACK = (0, 0, 0, 180)
    TRANSLUCENT_WHITE = (255, 255, 255, 180)


# UI Components
# Modified Button class with improved event handling
class Button:
    def __init__(
        self,
        x,
        y,
        width,
        height,
        text,
        color=Colors.BLUE,
        hover_color=None,
        text_color=Colors.WHITE,
        border_radius=5,
        font=None,
        icon=None,
        tooltip=None,
    ):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.color = color
        self.hover_color = hover_color or self.darken_color(color, 0.8)
        self.text_color = text_color
        self.border_radius = border_radius
        self.font = font or pygame.font.Font(None, 24)
        self.icon = icon
        self.tooltip = tooltip
        self.hover = False
        self.active = False  # Pressed state
        self.disabled = False
        self.pulsing = False
        self.pulse_value = 0
        self.pulse_direction = 1

    def darken_color(self, color, factor):
        r, g, b = color
        return (int(r * factor), int(g * factor), int(b * factor))

    def draw(self, surface):
        # Drawing code remains the same
        current_color = self.hover_color if self.hover else self.color

        # Handle pulsing effect
        if self.pulsing:
            self.pulse_value += 0.05 * self.pulse_direction
            if self.pulse_value > 1:
                self.pulse_value = 1
                self.pulse_direction = -1
            elif self.pulse_value < 0:
                self.pulse_value = 0
                self.pulse_direction = 1

            # Pulse between normal color and slightly brighter
            pulse_factor = 1 + 0.2 * self.pulse_value
            current_color = tuple(
                min(255, int(c * pulse_factor)) for c in current_color
            )

        # Disable effect
        if self.disabled:
            current_color = tuple(int(c * 0.5) for c in current_color)

        # Active effect (pressed)
        if self.active:
            current_color = self.darken_color(current_color, 0.8)

        # Draw button background with rounded corners
        rect = pygame.Rect(self.x, self.y, self.width, self.height)
        if self.border_radius > 0:
            pygame.draw.rect(
                surface, current_color, rect, border_radius=self.border_radius
            )
        else:
            pygame.draw.rect(surface, current_color, rect)

        # Draw text
        text_surf = self.font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(
            center=(self.x + self.width // 2, self.y + self.height // 2)
        )
        surface.blit(text_surf, text_rect)

        # Draw icon if available
        if self.icon:
            icon_rect = self.icon.get_rect()
            icon_rect.center = (
                self.x + self.width // 2 - text_rect.width // 2 - 20,
                self.y + self.height // 2,
            )
            surface.blit(self.icon, icon_rect)

    def handle_event(self, event):
        """Improved event handling that works for both press and click actions"""
        if self.disabled:
            return False

        pos = pygame.mouse.get_pos()

        # Always update hover state on mouse movement
        if event.type == pygame.MOUSEMOTION:
            self.hover = self.is_point_inside(pos)
            return False

        # Track mouse down for visual feedback
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:  # Left click
            if self.is_point_inside(pos):
                self.active = True
                return False  # Don't trigger action yet

        # Trigger action on mouse up (only if pressed inside the button)
        elif (
            event.type == pygame.MOUSEBUTTONUP and event.button == 1
        ):  # Left click release
            was_active = self.active
            self.active = False

            # Only trigger if mouse was pressed AND released on this button
            if was_active and self.is_point_inside(pos):
                return True  # Trigger the action!

        return False

    def is_point_inside(self, pos):
        return (
            self.x <= pos[0] <= self.x + self.width
            and self.y <= pos[1] <= self.y + self.height
        )


class Slider:
    def __init__(
        self,
        x,
        y,
        width,
        height,
        min_val,
        max_val,
        value,
        color=Colors.BLUE,
        background_color=Colors.LIGHT_GRAY,
        text="",
        font=None,
    ):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.min_val = min_val
        self.max_val = max_val
        self.value = value
        self.color = color
        self.background_color = background_color
        self.text = text
        self.font = font or pygame.font.Font(None, 20)
        self.dragging = False
        self.hover = False

    def draw(self, surface):
        # Draw background track
        track_rect = pygame.Rect(self.x, self.y + self.height // 2 - 3, self.width, 6)
        pygame.draw.rect(surface, self.background_color, track_rect, border_radius=3)

        # Calculate handle position
        handle_pos = int(
            self.x
            + (self.value - self.min_val) / (self.max_val - self.min_val) * self.width
        )

        # Draw active part of track
        active_rect = pygame.Rect(
            self.x, self.y + self.height // 2 - 3, handle_pos - self.x, 6
        )
        pygame.draw.rect(surface, self.color, active_rect, border_radius=3)

        # Draw handle
        handle_radius = 8
        pygame.draw.circle(
            surface,
            Colors.WHITE,
            (handle_pos, self.y + self.height // 2),
            handle_radius,
        )
        pygame.draw.circle(
            surface,
            self.color,
            (handle_pos, self.y + self.height // 2),
            handle_radius,
            2,
        )

        # Draw text and value
        if self.text:
            text_surf = self.font.render(self.text, True, Colors.TEXT_PRIMARY)
            surface.blit(text_surf, (self.x, self.y - 15))

        value_text = (
            f"{self.value:.3f}" if isinstance(self.value, float) else f"{self.value}"
        )
        value_surf = self.font.render(value_text, True, Colors.TEXT_SECONDARY)
        surface.blit(
            value_surf, (self.x + self.width + 5, self.y + self.height // 2 - 10)
        )

    def handle_event(self, event):
        mouse_pos = pygame.mouse.get_pos()

        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.is_point_inside(mouse_pos):
                self.dragging = True
                self.update_value(mouse_pos[0])
                return True

        elif event.type == pygame.MOUSEMOTION:
            self.hover = self.is_point_inside(mouse_pos)
            if self.dragging:
                self.update_value(mouse_pos[0])
                return True

        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
            if self.is_point_inside(mouse_pos):
                return True

        return False

    def update_value(self, x_pos):
        rel_x = max(0, min(x_pos - self.x, self.width))
        self.value = self.min_val + (rel_x / self.width) * (self.max_val - self.min_val)

    def is_point_inside(self, pos):
        return (
            self.x <= pos[0] <= self.x + self.width
            and self.y - 10 <= pos[1] <= self.y + self.height + 10
        )


class Toggle:
    def __init__(
        self,
        x,
        y,
        width,
        text,
        state=False,
        color_on=Colors.GREEN,
        color_off=Colors.GRAY,
        font=None,
    ):
        self.x = x
        self.y = y
        self.width = width
        self.height = 24
        self.text = text
        self.state = state
        self.color_on = color_on
        self.color_off = color_off
        self.font = font or pygame.font.Font(None, 20)
        self.hover = False

    def draw(self, surface):
        # Draw text
        text_surf = self.font.render(self.text, True, Colors.TEXT_PRIMARY)
        surface.blit(text_surf, (self.x, self.y + 2))

        # Calculate toggle position
        text_width = text_surf.get_width()
        toggle_x = self.x + text_width + 10
        toggle_width = 40
        toggle_height = 20

        # Draw toggle background
        color = self.color_on if self.state else self.color_off
        pygame.draw.rect(
            surface,
            color,
            (toggle_x, self.y + 2, toggle_width, toggle_height),
            border_radius=toggle_height // 2,
        )

        # Draw toggle handle
        handle_x = toggle_x + toggle_width - 16 if self.state else toggle_x + 4
        pygame.draw.circle(
            surface,
            Colors.WHITE,
            (handle_x, self.y + 2 + toggle_height // 2),
            toggle_height // 2 - 2,
        )

    def handle_event(self, event):
        mouse_pos = pygame.mouse.get_pos()

        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.is_point_inside(mouse_pos):
                self.state = not self.state
                return True

        elif event.type == pygame.MOUSEMOTION:
            self.hover = self.is_point_inside(mouse_pos)

        return False

    def is_point_inside(self, pos):
        text_surf = self.font.render(self.text, True, Colors.TEXT_PRIMARY)
        text_width = text_surf.get_width()
        toggle_x = self.x + text_width + 10
        toggle_width = 40

        return (
            toggle_x <= pos[0] <= toggle_x + toggle_width
            and self.y <= pos[1] <= self.y + self.height
        )


class Panel:
    def __init__(
        self, x, y, width, height, color=Colors.WHITE, border_radius=5, alpha=255
    ):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color
        self.border_radius = border_radius
        self.alpha = alpha

    def draw(self, surface):
        # Create a surface with per-pixel alpha
        panel_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        # Fill with color and alpha
        if isinstance(self.color, tuple) and len(self.color) == 3:
            color_with_alpha = (*self.color, self.alpha)
        elif isinstance(self.color, tuple) and len(self.color) == 4:
            color_with_alpha = self.color
        else:
            color_with_alpha = (*self.color, self.alpha)

        # Draw the panel with rounded corners
        pygame.draw.rect(
            panel_surf,
            color_with_alpha,
            (0, 0, self.width, self.height),
            border_radius=self.border_radius,
        )

        # Draw shadow effect
        if self.alpha > 200:
            shadow_surf = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            shadow_color = (0, 0, 0, 30)  # Translucent black
            pygame.draw.rect(
                shadow_surf,
                shadow_color,
                (0, 0, self.width, self.height),
                border_radius=self.border_radius,
            )

            # Offset shadow slightly
            surface.blit(shadow_surf, (self.x + 2, self.y + 2))

        # Blit the panel onto the main surface
        surface.blit(panel_surf, (self.x, self.y))

    def is_point_inside(self, pos):
        return (
            self.x <= pos[0] <= self.x + self.width
            and self.y <= pos[1] <= self.y + self.height
        )


class Visualizer:
    """
    Enhanced visualizer for programmable matter with modern UI elements, effects, and
    advanced visualization features.
    """

    def __init__(
        self,
        grid: Grid,
        target_shape: list,
        obstacles: list = None,
        max_window_size: int = 1200,
    ):
        """
        Initialize the enhanced visualizer with dynamic resizing, themes, and UI elements.
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
        self.selected_index = 0

        # Theme settings
        self.theme = "light"  # Can be "light", "dark", "blue", etc.
        self.show_effects = True  # Enable/disable visual effects

        # Animation and effects settings
        self.particle_systems = []  # Store active particle systems
        self.animations = []  # Store active animations
        self.transition_alpha = 0  # For screen transitions
        self.effect_intensity = 1.0  # Multiplier for effect intensity
        self.last_move_positions = []  # Track positions for trail effects

        # AI agent and planning
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
        self.fps_samples = []  # For tracking FPS
        self.last_fps_update = time.time()
        self.current_fps = 60.0

        # Multiprocessing status
        self.multiprocessing_active = False
        self.process_count = 0
        self.processes_completed = 0
        self.process_results = {}

        # Initialize Pygame with better graphics settings
        os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
        pygame.init()
        pygame.freetype.init()

        # Set up the display with vsync when available
        self.screen_width = min(
            1280, max(self.width + 400, 1024)
        )  # Minimum width for UI
        self.screen_height = min(
            800, max(self.height + 200, 768)
        )  # Minimum height for UI

        # Try to create display with hardware acceleration and vsync
        self.screen = pygame.display.set_mode(
            (self.screen_width, self.screen_height),
            pygame.RESIZABLE | pygame.HWSURFACE | pygame.DOUBLEBUF,
        )

        # Set application icon and title
        pygame.display.set_caption("Programmable Matter Simulator")

        # Load fonts
        self.fonts = {
            "small": pygame.font.Font(None, 18),
            "normal": pygame.font.Font(None, 24),
            "large": pygame.font.Font(None, 32),
            "title": pygame.font.Font(None, 48),
        }

        # For scrolling in large grids
        self.scroll_x = 0
        self.scroll_y = 0
        self.scrolling = False
        self.scroll_start_pos = None

        # UI Layout
        self.sidebar_width = 300
        self.create_ui_elements()

        # For animation
        self.last_frame_time = time.time()
        self.frame_rate = 60  # Target frame rate
        self.delta_time = 1 / 60  # Time between frames
        self.paused = False  # Animation pause toggle

        # Create clock for consistent frame timing
        self.clock = pygame.time.Clock()

        # Display initial size info
        print(
            f"Grid size: {grid.n}x{grid.m}, Cell size: {self.cell_size}px, Window: {self.screen_width}x{self.screen_height}px"
        )
        if self.obstacles:
            print(f"Obstacles: {len(self.obstacles)} positions")

    def create_ui_elements(self):
        """Create UI components like buttons, sliders, etc."""
        # Control Panel UI
        self.panels = {
            "main": Panel(
                self.screen_width - self.sidebar_width,
                0,
                self.sidebar_width,
                self.screen_height,
                color=Colors.WHITE,
                alpha=240,
            )
        }

        # Sidebar buttons - top section
        button_width = 260
        button_height = 40
        button_x = self.screen_width - self.sidebar_width + 20
        button_y = 60
        button_spacing = 15

        self.buttons = {
            "plan_astar": Button(
                button_x,
                button_y,
                button_width,
                button_height,
                "Run A* Planning",
                Colors.BLUE,
            ),
            "plan_hierarchical": Button(
                button_x,
                button_y + button_height + button_spacing,
                button_width,
                button_height,
                "Run Hierarchical Planning",
                Colors.PURPLE,
            ),
            "plan_parallel": Button(
                button_x,
                button_y + 2 * (button_height + button_spacing),
                button_width,
                button_height,
                "Run Parallel A*",
                Colors.ORANGE,
            ),
            "target_selection": Button(
                button_x,
                button_y + 3 * (button_height + button_spacing),
                button_width,
                button_height,
                "Set Target Shape",
                Colors.GREEN,
            ),
            "obstacle_placement": Button(
                button_x,
                button_y + 4 * (button_height + button_spacing),
                button_width,
                button_height,
                "Place Obstacles",
                Colors.RED,
            ),
            "reset": Button(
                button_x,
                button_y + 5 * (button_height + button_spacing),
                button_width,
                button_height,
                "Reset Simulation",
                Colors.GRAY,
            ),
        }

        # Status Panel
        status_panel_y = button_y + 6 * (button_height + button_spacing) + 20
        self.panels["status"] = Panel(
            button_x - 10,
            status_panel_y,
            button_width + 20,
            150,
            color=Colors.LIGHT_GRAY,
            alpha=180,
        )

        # Sliders
        slider_y = status_panel_y + 170
        self.sliders = {
            "speed": Slider(
                button_x,
                slider_y,
                200,
                40,
                0.001,
                0.5,
                self.ai_delay,
                text="Animation Speed",
                font=self.fonts["small"],
            ),
            "effect_intensity": Slider(
                button_x,
                slider_y + 50,
                200,
                40,
                0.0,
                1.0,
                self.effect_intensity,
                text="Effect Intensity",
                font=self.fonts["small"],
            ),
        }

        # Toggles
        toggle_y = slider_y + 120
        self.toggles = {
            "show_waypoints": Toggle(
                button_x, toggle_y, 150, "Show Waypoints", self.show_waypoints
            ),
            "show_effects": Toggle(
                button_x, toggle_y + 30, 150, "Visual Effects", self.show_effects
            ),
        }

    def calculate_cell_size(self, n, m, max_size):
        """
        Calculate an appropriate cell size based on grid dimensions.
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

    def update(self):
        """Update animation and UI state"""
        # Update delta time
        current_time = time.time()
        self.delta_time = current_time - self.last_frame_time
        self.last_frame_time = current_time

        # Limit delta time to avoid physics issues on slow frames
        self.delta_time = min(self.delta_time, 0.05)

        # Update FPS counter
        self.fps_samples.append(1.0 / max(0.001, self.delta_time))
        if len(self.fps_samples) > 60:
            self.fps_samples.pop(0)

        if current_time - self.last_fps_update > 0.5:  # Update FPS every 0.5s
            self.current_fps = sum(self.fps_samples) / len(self.fps_samples)
            self.last_fps_update = current_time

        # Update message timer
        if self.message_timer > 0:
            self.message_timer -= 1

        # Update particle systems
        for i in range(len(self.particle_systems) - 1, -1, -1):
            self.particle_systems[i].update(self.delta_time)
            if self.particle_systems[i].done:
                self.particle_systems.pop(i)

        # Update animations
        for i in range(len(self.animations) - 1, -1, -1):
            self.animations[i].update(self.delta_time)
            if self.animations[i].done:
                self.animations.pop(i)

        # Update UI elements
        # Update slider values
        self.ai_delay = self.sliders["speed"].value
        self.effect_intensity = self.sliders["effect_intensity"].value
        self.show_waypoints = self.toggles["show_waypoints"].state
        self.show_effects = self.toggles["show_effects"].state

        # Execute AI step if in AI mode
        if self.mode == "ai" and not self.paused:
            self.execute_ai_plan()

    def draw_grid(self):
        """
        Draw the grid, blocks, obstacles, and target with enhanced visuals.
        """
        # Fill background for entire screen
        self.screen.fill(Colors.BACKGROUND)

        # Create a surface for the grid that can be moved around
        grid_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        grid_surface.fill((0, 0, 0, 0))  # Transparent background

        # Calculate visible area based on scroll position
        visible_cols_start = max(0, self.scroll_x // self.cell_size)
        visible_rows_start = max(0, self.scroll_y // self.cell_size)
        visible_cols_end = min(
            self.grid.m, visible_cols_start + self.screen_width // self.cell_size + 2
        )
        visible_rows_end = min(
            self.grid.n, visible_rows_start + self.screen_height // self.cell_size + 2
        )

        # Draw grid lines
        for col in range(self.grid.m + 1):
            x = col * self.cell_size
            pygame.draw.line(grid_surface, Colors.GRID_LINES, (x, 0), (x, self.height))

        for row in range(self.grid.n + 1):
            y = row * self.cell_size
            pygame.draw.line(grid_surface, Colors.GRID_LINES, (0, y), (self.width, y))

        # Draw obstacles with enhanced visuals
        for row, col in self.obstacles:
            rect = pygame.Rect(
                col * self.cell_size,
                row * self.cell_size,
                self.cell_size,
                self.cell_size,
            )

            # Draw obstacle with gradient effect
            if self.mode == "obstacle_placement":
                self.draw_rounded_block(
                    grid_surface, rect, Colors.PURPLE, highlight=True
                )
            else:
                self.draw_rounded_block(grid_surface, rect, Colors.PURPLE)

        # Draw target shape with semi-transparency
        for row, col in self.target_shape:
            rect = pygame.Rect(
                col * self.cell_size,
                row * self.cell_size,
                self.cell_size,
                self.cell_size,
            )

            if self.mode == "target_selection":
                self.draw_rounded_block(
                    grid_surface, rect, Colors.ORANGE, highlight=True
                )
            else:
                # Create transparent target indicator
                s = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA)
                s.fill((230, 126, 34, 100))  # Semi-transparent orange

                # Add a subtle border
                pygame.draw.rect(
                    s,
                    (230, 126, 34, 160),
                    (0, 0, self.cell_size, self.cell_size),
                    2,
                    border_radius=3,
                )

                grid_surface.blit(s, (col * self.cell_size, row * self.cell_size))

        # Draw waypoints if enabled
        if self.show_waypoints and self.waypoints:
            for i, waypoint in enumerate(self.waypoints):
                # Draw each waypoint with different transparency
                for row, col in waypoint:
                    alpha = 120 - 10 * i  # Progressively more transparent
                    alpha = max(40, alpha)  # Ensure minimum visibility

                    # Create transparent surface
                    s = pygame.Surface(
                        (self.cell_size, self.cell_size), pygame.SRCALPHA
                    )

                    # Current subproblem highlighted differently
                    if i == self.current_subproblem:
                        pygame.draw.rect(
                            s,
                            (52, 152, 219, alpha),  # Cyan
                            (0, 0, self.cell_size, self.cell_size),
                            border_radius=3,
                        )
                    else:
                        pygame.draw.rect(
                            s,
                            (52, 152, 219, alpha // 2),  # Faded cyan
                            (0, 0, self.cell_size, self.cell_size),
                            border_radius=3,
                        )

                    # Draw waypoint number in center
                    if i == self.current_subproblem:
                        text = self.fonts["small"].render(
                            str(i + 1), True, Colors.WHITE
                        )
                        text_rect = text.get_rect(
                            center=(self.cell_size // 2, self.cell_size // 2)
                        )
                        s.blit(text, text_rect)

                    grid_surface.blit(s, (col * self.cell_size, row * self.cell_size))

        # Draw active particle systems
        if self.show_effects:
            for system in self.particle_systems:
                system.draw(grid_surface)

        # Draw matter elements with enhanced styling
        for i, (row, col) in enumerate(self.grid.matter_elements):
            rect = pygame.Rect(
                col * self.cell_size,
                row * self.cell_size,
                self.cell_size,
                self.cell_size,
            )

            # Choose color based on mode
            if self.mode == "ai":
                color = Colors.RED  # AI mode color
                highlight = True
            elif self.mode == "individual" and i == self.selected_index:
                color = Colors.GREEN  # Selected block highlight
                highlight = True
            else:
                color = Colors.BLUE  # Default color
                highlight = False

            # Draw block with enhanced visuals
            self.draw_rounded_block(
                grid_surface, rect, color, highlight=highlight, index=i
            )

            # Draw connector lines between blocks to show connectivity
            if i < len(self.grid.matter_elements) - 1:
                for j, (other_row, other_col) in enumerate(
                    self.grid.matter_elements[i + 1 :], i + 1
                ):
                    # Check if blocks are adjacent
                    dx = abs(col - other_col)
                    dy = abs(row - other_row)
                    if dx <= 1 and dy <= 1:
                        # Draw connection line
                        start_pos = (
                            col * self.cell_size + self.cell_size // 2,
                            row * self.cell_size + self.cell_size // 2,
                        )
                        end_pos = (
                            other_col * self.cell_size + self.cell_size // 2,
                            other_row * self.cell_size + self.cell_size // 2,
                        )

                        # Adjust line endpoints to start/end at block edges
                        pygame.draw.line(
                            grid_surface, Colors.BLUE_DARK, start_pos, end_pos, 2
                        )

                        # Draw small connecting circle at intersection
                        mid_pos = (
                            (start_pos[0] + end_pos[0]) // 2,
                            (start_pos[1] + end_pos[1]) // 2,
                        )
                        pygame.draw.circle(grid_surface, Colors.BLUE, mid_pos, 3)

        # Draw grid surface to main screen, adjusted for scrolling
        self.screen.blit(grid_surface, (-self.scroll_x, -self.scroll_y))

        # Draw UI panels
        self.draw_ui()

    def draw_rounded_block(self, surface, rect, color, highlight=False, index=None):
        """Draw a block with rounded corners, gradient, and optional index"""
        # Adjust the rectangle to be slightly smaller for better visuals
        padding = 1
        inner_rect = pygame.Rect(
            rect.x + padding,
            rect.y + padding,
            rect.width - 2 * padding,
            rect.height - 2 * padding,
        )

        # Gradient colors (darker at bottom)
        if highlight:
            top_color = tuple(min(255, c + 40) for c in color)
            bottom_color = tuple(max(0, c - 20) for c in color)
        else:
            top_color = color
            bottom_color = tuple(max(0, c - 40) for c in color)

        # Draw main rectangle with rounded corners (using pygame's built-in rounded rect)
        pygame.draw.rect(surface, bottom_color, inner_rect, border_radius=5)

        # Draw a smaller rectangle at the top for gradient effect
        gradient_height = inner_rect.height // 2
        gradient_rect = pygame.Rect(
            inner_rect.x, inner_rect.y, inner_rect.width, gradient_height
        )
        pygame.draw.rect(surface, top_color, gradient_rect, border_radius=5)

        # For a highlight effect, draw a bright line at top
        if highlight:
            highlight_rect = pygame.Rect(
                inner_rect.x, inner_rect.y, inner_rect.width, 2
            )
            pygame.draw.rect(surface, Colors.WHITE, highlight_rect)

        # Draw a subtle border
        pygame.draw.rect(surface, bottom_color, inner_rect, 1, border_radius=5)

        # If index provided, draw it in the center
        if index is not None and self.cell_size >= 20:
            index_text = self.fonts["small"].render(str(index), True, Colors.WHITE)
            index_rect = index_text.get_rect(center=inner_rect.center)
            surface.blit(index_text, index_rect)

    def draw_ui(self):
        """Draw all UI elements including panels, buttons, and status information"""
        # Draw main sidebar panel
        self.panels["main"].draw(self.screen)

        # Draw title
        title_text = self.fonts["title"].render(
            "Matter Solver", True, Colors.TEXT_PRIMARY
        )
        self.screen.blit(title_text, (self.screen_width - self.sidebar_width + 20, 10))

        # Draw all buttons
        for button in self.buttons.values():
            button.draw(self.screen)

        # Draw status panel and info
        self.panels["status"].draw(self.screen)

        # Draw status information
        status_x = self.screen_width - self.sidebar_width + 20
        status_y = self.panels["status"].y + 10
        line_height = 25

        # Mode status
        mode_text = self.fonts["normal"].render(
            f"Mode: {self.mode.title()}", True, Colors.TEXT_PRIMARY
        )
        self.screen.blit(mode_text, (status_x, status_y))

        # Grid info
        grid_text = self.fonts["small"].render(
            f"Grid: {self.grid.n}×{self.grid.m} | Cell size: {self.cell_size}px",
            True,
            Colors.TEXT_SECONDARY,
        )
        self.screen.blit(grid_text, (status_x, status_y + line_height))

        # Block counts
        blocks_text = self.fonts["small"].render(
            f"Blocks: {len(self.grid.matter_elements)} | Target: {len(self.target_shape)}",
            True,
            Colors.TEXT_SECONDARY,
        )
        self.screen.blit(blocks_text, (status_x, status_y + 2 * line_height))

        # Obstacles count
        obstacles_text = self.fonts["small"].render(
            f"Obstacles: {len(self.obstacles)}", True, Colors.TEXT_SECONDARY
        )
        self.screen.blit(obstacles_text, (status_x, status_y + 3 * line_height))

        # AI progress (if applicable)
        if self.ai_plan:
            ai_status = "PAUSED" if self.paused else "RUNNING"
            ai_text = self.fonts["small"].render(
                f"AI Moves: {self.ai_step}/{len(self.ai_plan)} | {ai_status}",
                True,
                Colors.RED if self.mode == "ai" else Colors.TEXT_SECONDARY,
            )
            self.screen.blit(ai_text, (status_x, status_y + 4 * line_height))

            # Planning metrics
            if self.plan_time > 0:
                metrics_text = self.fonts["small"].render(
                    f"Planning: {self.plan_time:.2f}s | Nodes: {self.plan_nodes}",
                    True,
                    Colors.TEXT_SECONDARY,
                )
                self.screen.blit(metrics_text, (status_x, status_y + 5 * line_height))

        # Draw sliders
        for slider in self.sliders.values():
            slider.draw(self.screen)

        # Draw toggles
        for toggle in self.toggles.values():
            toggle.draw(self.screen)

        # Draw FPS counter in corner
        fps_text = self.fonts["small"].render(
            f"{int(self.current_fps)} FPS", True, Colors.TEXT_SECONDARY
        )
        self.screen.blit(fps_text, (self.screen_width - 70, self.screen_height - 20))

        # Draw mode-specific help text
        self.draw_mode_help()

        # Draw message if active
        if self.message and self.message_timer > 0:
            msg_panel = Panel(
                10,
                10,
                len(self.message) * 10 + 20,
                40,
                color=(0, 0, 0, 180),
                border_radius=5,
            )
            msg_panel.draw(self.screen)

            msg_text = self.fonts["normal"].render(self.message, True, Colors.WHITE)
            self.screen.blit(msg_text, (20, 20))

        # Draw multiprocessing status if active
        if self.multiprocessing_active:
            mp_text = self.fonts["normal"].render(
                f"Multiprocessing: {self.processes_completed}/{self.process_count}",
                True,
                Colors.MAGENTA,
            )
            mp_bg = Panel(
                10,
                self.screen_height - 40,
                300,
                30,
                color=(0, 0, 0, 150),
                border_radius=5,
            )
            mp_bg.draw(self.screen)
            self.screen.blit(mp_text, (20, self.screen_height - 35))

    def draw_mode_help(self):
        """Draw help text based on current mode"""
        help_panel_x = 10
        help_panel_y = self.screen_height - 90
        help_panel_width = 400
        help_panel_height = 80

        # Create help panel based on mode
        if self.mode == "target_selection":
            help_panel = Panel(
                help_panel_x,
                help_panel_y,
                help_panel_width,
                help_panel_height,
                color=(0, 0, 0, 150),
                border_radius=5,
            )
            help_panel.draw(self.screen)

            help1 = self.fonts["small"].render(
                "Click to add/remove blocks from target shape", True, Colors.WHITE
            )
            help2 = self.fonts["small"].render(
                "Target must have same number of blocks as initial shape",
                True,
                Colors.WHITE,
            )

            self.screen.blit(help1, (help_panel_x + 10, help_panel_y + 10))
            self.screen.blit(help2, (help_panel_x + 10, help_panel_y + 35))

            # Show warning if counts don't match
            if len(self.target_shape) != len(self.grid.matter_elements):
                warning = self.fonts["small"].render(
                    f"Warning: Target has {len(self.target_shape)} blocks, initial has {len(self.grid.matter_elements)}",
                    True,
                    Colors.YELLOW,
                )
                self.screen.blit(warning, (help_panel_x + 10, help_panel_y + 60))

        elif self.mode == "obstacle_placement":
            help_panel = Panel(
                help_panel_x,
                help_panel_y,
                help_panel_width,
                help_panel_height,
                color=(0, 0, 0, 150),
                border_radius=5,
            )
            help_panel.draw(self.screen)

            help1 = self.fonts["small"].render(
                "Click to add/remove obstacles", True, Colors.WHITE
            )
            help2 = self.fonts["small"].render(
                "Obstacles cannot overlap with blocks or targets", True, Colors.WHITE
            )

            self.screen.blit(help1, (help_panel_x + 10, help_panel_y + 10))
            self.screen.blit(help2, (help_panel_x + 10, help_panel_y + 35))

        elif self.mode == "manual":
            help_panel = Panel(
                help_panel_x,
                help_panel_y,
                help_panel_width,
                help_panel_height,
                color=(0, 0, 0, 150),
                border_radius=5,
            )
            help_panel.draw(self.screen)

            help1 = self.fonts["small"].render(
                "Arrow Keys: Move shape | Tab: Switch to individual mode",
                True,
                Colors.WHITE,
            )
            help2 = self.fonts["small"].render(
                "Space: Pause/Resume | Mouse wheel: Zoom | Middle-click: Pan",
                True,
                Colors.WHITE,
            )

            self.screen.blit(help1, (help_panel_x + 10, help_panel_y + 10))
            self.screen.blit(help2, (help_panel_x + 10, help_panel_y + 35))

        elif self.mode == "individual":
            help_panel = Panel(
                help_panel_x,
                help_panel_y,
                help_panel_width,
                help_panel_height,
                color=(0, 0, 0, 150),
                border_radius=5,
            )
            help_panel.draw(self.screen)

            help1 = self.fonts["small"].render(
                "WASD: Move selected block | R/F: Change selection", True, Colors.WHITE
            )
            help2 = self.fonts["small"].render(
                f"Currently selected: Block {self.selected_index}", True, Colors.GREEN
            )

            self.screen.blit(help1, (help_panel_x + 10, help_panel_y + 10))
            self.screen.blit(help2, (help_panel_x + 10, help_panel_y + 35))

        elif self.mode == "ai":
            help_panel = Panel(
                help_panel_x,
                help_panel_y,
                help_panel_width,
                help_panel_height,
                color=(0, 0, 0, 150),
                border_radius=5,
            )
            help_panel.draw(self.screen)

            help1 = self.fonts["small"].render(
                "Space: Pause/Resume | +/-: Adjust speed", True, Colors.WHITE
            )
            progress = self.ai_step / len(self.ai_plan) if self.ai_plan else 0

            # Draw progress bar
            bar_width = help_panel_width - 20
            bar_height = 15
            bar_x = help_panel_x + 10
            bar_y = help_panel_y + 35

            # Background
            pygame.draw.rect(
                self.screen,
                Colors.GRAY,
                (bar_x, bar_y, bar_width, bar_height),
                border_radius=bar_height // 2,
            )

            # Progress
            if progress > 0:
                pygame.draw.rect(
                    self.screen,
                    Colors.RED,
                    (bar_x, bar_y, int(bar_width * progress), bar_height),
                    border_radius=bar_height // 2,
                )

            # Text
            status = (
                "PAUSED" if self.paused else f"Step {self.ai_step}/{len(self.ai_plan)}"
            )
            progress_text = self.fonts["small"].render(status, True, Colors.WHITE)
            self.screen.blit(
                progress_text,
                (
                    bar_x + bar_width // 2 - progress_text.get_width() // 2,
                    bar_y + bar_height // 2 - progress_text.get_height() // 2,
                ),
            )

            self.screen.blit(help1, (help_panel_x + 10, help_panel_y + 10))

    class ParticleSystem:
        """Simple particle system for visual effects"""

        def __init__(self, x, y, color, count=20, duration=1.0, spread=30):
            self.particles = []
            self.color = color
            self.duration = duration
            self.time = 0
            self.done = False

            for _ in range(count):
                angle = random.random() * math.pi * 2
                speed = random.random() * spread
                self.particles.append(
                    {
                        "x": x,
                        "y": y,
                        "vx": math.cos(angle) * speed,
                        "vy": math.sin(angle) * speed,
                        "size": random.randint(2, 5),
                        "alpha": 255,
                    }
                )

        def update(self, dt):
            self.time += dt
            if self.time >= self.duration:
                self.done = True

            fade_factor = 255 / self.duration

            for p in self.particles:
                p["x"] += p["vx"] * dt
                p["y"] += p["vy"] * dt
                p["alpha"] = max(0, 255 - self.time * fade_factor)

        def draw(self, surface):
            for p in self.particles:
                if p["alpha"] > 0:
                    pygame.draw.circle(
                        surface,
                        (*self.color, p["alpha"]),
                        (int(p["x"]), int(p["y"])),
                        p["size"],
                    )

    def show_message(self, message, duration=60):
        """Display a temporary message on screen with fade effect"""
        self.message = message
        self.message_timer = duration

        # Add visual feedback
        if self.show_effects:
            # Create screen flash
            flash = self.Animation(
                0,
                100,
                0.2,
                lambda v: self.set_transition_alpha(v),
                easing=self.Animation.ease_out,
            )
            fade = self.Animation(
                100,
                0,
                0.4,
                lambda v: self.set_transition_alpha(v),
                easing=self.Animation.ease_in,
                delay=0.2,
            )
            self.animations.append(flash)
            self.animations.append(fade)

    def set_transition_alpha(self, value):
        """Helper for screen transition effects"""
        self.transition_alpha = value

    class Animation:
        """Simple animation manager for tweening values"""

        def __init__(self, start, end, duration, callback, easing=None, delay=0):
            self.start = start
            self.end = end
            self.duration = duration
            self.callback = callback
            self.easing = easing or self.linear
            self.delay = delay
            self.time = -delay
            self.done = False

        def update(self, dt):
            self.time += dt

            if self.time < 0:  # Still in delay
                return

            if self.time >= self.duration:
                self.callback(self.end)
                self.done = True
                return

            # Calculate progress and apply easing
            progress = self.time / self.duration
            eased = self.easing(progress)

            # Calculate and set current value
            value = self.start + (self.end - self.start) * eased
            self.callback(value)

        @staticmethod
        def linear(t):
            return t

        @staticmethod
        def ease_in(t):
            return t * t

        @staticmethod
        def ease_out(t):
            return 1 - (1 - t) * (1 - t)

        @staticmethod
        def ease_in_out(t):
            return t * t * (3 - 2 * t)

    def execute_ai_plan(self):
        """
        Executes the AI-generated plan step by step with visual effects.
        """
        if self.paused or not self.ai_plan or self.ai_step >= len(self.ai_plan):
            return

        # Maintain consistent timing using delta_time
        if self.ai_delay > 0:
            # Only execute move when delay time has been reached
            if not hasattr(self, "move_timer"):
                self.move_timer = 0

            self.move_timer += self.delta_time
            if self.move_timer < self.ai_delay:
                return

            # Reset timer for next move
            self.move_timer = 0

        # Get current move from plan
        move_set = self.ai_plan[self.ai_step]

        # Group all individual moves into one dictionary
        moves = {i: (dx, dy) for i, dx, dy in move_set}

        # Store positions for effects
        self.last_move_positions = [
            (
                self.grid.matter_elements[i][1] * self.cell_size + self.cell_size // 2,
                self.grid.matter_elements[i][0] * self.cell_size + self.cell_size // 2,
            )
            for i in moves.keys()
        ]

        # Execute move
        success = self.grid.move_individual(moves)

        if not success:
            print("Invalid move! Removing it from the plan and retrying...")
            self.ai_plan.pop(self.ai_step)
            return

        # Center view on the shape
        self.center_view_on_shape()

        # Add visual effects if enabled
        if self.show_effects and self.effect_intensity > 0:
            for pos in self.last_move_positions:
                # Scale effects based on intensity
                particles = int(20 * self.effect_intensity)
                # Create particle effect at move location
                self.particle_systems.append(
                    self.ParticleSystem(
                        pos[0],
                        pos[1],
                        Colors.RED if self.mode == "ai" else Colors.BLUE,
                        count=particles,
                        duration=0.7 * self.effect_intensity,
                    )
                )

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

                # Add waypoint effect
                if self.show_effects:
                    for row, col in next_waypoint:
                        self.particle_systems.append(
                            self.ParticleSystem(
                                col * self.cell_size + self.cell_size // 2,
                                row * self.cell_size + self.cell_size // 2,
                                Colors.CYAN,
                                count=30,
                                duration=1.0,
                                spread=50,
                            )
                        )

        # Increment step counter
        self.ai_step += 1

        # Check if we've completed the plan
        if self.ai_step >= len(self.ai_plan):
            print("AI execution complete!")
            self.show_message("Solution complete!")
            self.mode = "manual"

    def handle_mouse_click(self, pos, button):
        """Handle mouse clicks for UI interaction, target selection, obstacle placement, and scrolling"""
        # Check UI elements first
        for name, button_obj in self.buttons.items():
            if button_obj.handle_event(
                pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=button, pos=pos)
            ):
                self.handle_button_click(name)
                return

        # Check sliders
        for name, slider in self.sliders.items():
            if slider.handle_event(
                pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=button, pos=pos)
            ):
                return

        # Check toggles
        for name, toggle in self.toggles.items():
            if toggle.handle_event(
                pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=button, pos=pos)
            ):
                return

        # Check if click is inside sidebar
        if pos[0] > self.screen_width - self.sidebar_width:
            return

        if button == 1:  # Left click
            # Convert mouse position to grid coordinates including scroll
            col = (pos[0] + self.scroll_x) // self.cell_size
            row = (pos[1] + self.scroll_y) // self.cell_size

            # Ensure within grid bounds
            if 0 <= row < self.grid.n and 0 <= col < self.grid.m:
                cell = (row, col)

                # Target selection mode
                if self.mode == "target_selection":
                    self.handle_target_click(cell)
                    return True

                # Obstacle placement mode
                elif self.mode == "obstacle_placement":
                    self.handle_obstacle_click(cell)
                    return True

        elif button == 2:  # Middle click
            # Start scrolling
            self.scrolling = True
            self.scroll_start_pos = pos

    def handle_target_click(self, cell):
        """Handle clicks in target selection mode"""
        # Add or remove the cell from target shape
        if cell in self.target_shape:
            self.target_shape.remove(cell)
            print(f"Removed {cell} from target shape")

            # Add visual effect
            if self.show_effects:
                row, col = cell
                self.particle_systems.append(
                    self.ParticleSystem(
                        col * self.cell_size + self.cell_size // 2 - self.scroll_x,
                        row * self.cell_size + self.cell_size // 2 - self.scroll_y,
                        Colors.RED,
                        count=15,
                    )
                )
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

            # Add visual effect
            if self.show_effects:
                row, col = cell
                self.particle_systems.append(
                    self.ParticleSystem(
                        col * self.cell_size + self.cell_size // 2 - self.scroll_x,
                        row * self.cell_size + self.cell_size // 2 - self.scroll_y,
                        Colors.GREEN,
                        count=15,
                    )
                )

    def handle_obstacle_click(self, cell):
        """Handle clicks in obstacle placement mode"""
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

            # Add visual effect
            if self.show_effects:
                row, col = cell
                self.particle_systems.append(
                    self.ParticleSystem(
                        col * self.cell_size + self.cell_size // 2 - self.scroll_x,
                        row * self.cell_size + self.cell_size // 2 - self.scroll_y,
                        Colors.RED,
                        count=15,
                    )
                )
        else:
            self.obstacles.append(cell)
            print(f"Added obstacle at {cell}")

            # Add visual effect
            if self.show_effects:
                row, col = cell
                self.particle_systems.append(
                    self.ParticleSystem(
                        col * self.cell_size + self.cell_size // 2 - self.scroll_x,
                        row * self.cell_size + self.cell_size // 2 - self.scroll_y,
                        Colors.PURPLE,
                        count=15,
                    )
                )

    def handle_button_click(self, button_name):
        """Handle UI button clicks"""
        if button_name == "plan_astar":
            self.run_normal_planning()

        elif button_name == "plan_hierarchical":
            self.run_hierarchical_planning()

        elif button_name == "plan_parallel":
            self.run_parallel_planning()

        elif button_name == "target_selection":
            self.mode = "target_selection"

        elif button_name == "obstacle_placement":
            self.mode = "obstacle_placement"

        elif button_name == "reset":
            # Reset the simulation
            self.ai_plan = []
            self.ai_step = 0
            self.mode = "manual"
            self.paused = False
            self.particle_systems = []
            self.show_message("Simulation reset")

    def handle_mouse_motion(self, pos, rel, buttons):
        """Handle mouse motion for scrolling"""
        # Handle UI elements hover and drag
        for button in self.buttons.values():
            button.hover = button.is_point_inside(pos)

        for slider in self.sliders.values():
            if slider.dragging:
                slider.handle_event(
                    pygame.event.Event(
                        pygame.MOUSEMOTION, pos=pos, rel=rel, buttons=buttons
                    )
                )
                return

        # Handle scrolling logic
        if self.scrolling and buttons[1]:  # Middle button held down
            # Update scroll position
            self.scroll_x = max(
                0,
                min(
                    self.scroll_x - rel[0],
                    self.width - self.screen_width + self.sidebar_width,
                ),
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
        self.scroll_x = max(
            0,
            center_col * self.cell_size - (self.screen_width - self.sidebar_width) // 2,
        )
        self.scroll_y = max(0, center_row * self.cell_size - self.screen_height // 2)

        # Ensure we don't scroll beyond grid bounds
        self.scroll_x = min(
            self.scroll_x, self.width - (self.screen_width - self.sidebar_width)
        )
        self.scroll_y = min(self.scroll_y, self.height - self.screen_height)

        # Add visual effect for centering
        if self.show_effects:
            center_x = center_col * self.cell_size + self.cell_size // 2 - self.scroll_x
            center_y = center_row * self.cell_size + self.cell_size // 2 - self.scroll_y

            # Create a ripple effect
            for i in range(3):
                delay = i * 0.1

                # Expanding circle animation
                def create_circle_anim(radius, alpha):
                    return self.Animation(
                        0,
                        radius,
                        0.5,
                        lambda v: self.draw_temp_circle(center_x, center_y, v, alpha),
                        easing=self.Animation.ease_out,
                        delay=delay,
                    )

                self.animations.append(create_circle_anim(100, 120))

    def draw_temp_circle(self, x, y, radius, alpha):
        """Helper to draw temporary circles for animations"""
        if radius <= 0:
            return

        # Create a temporary surface for drawing the circle
        temp_surf = pygame.Surface((int(radius * 2), int(radius * 2)), pygame.SRCALPHA)
        pygame.draw.circle(
            temp_surf,
            (255, 255, 255, alpha),
            (int(radius), int(radius)),
            int(radius),
            1,
        )
        self.screen.blit(temp_surf, (int(x - radius), int(y - radius)))

    def run_normal_planning(self):
        """Run normal A* planning with timing and visual effects"""
        # First verify target shape
        if len(self.target_shape) != len(self.grid.matter_elements):
            self.show_message(
                f"Target must have exactly {len(self.grid.matter_elements)} blocks"
            )
            return False

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
        plan = self.ai_agent.plan_auto()
        end_time = time.time()

        self.plan_time = end_time - start_time
        self.plan_nodes = self.ai_agent.nodes_expanded

        if plan:
            print(f"A* planning successful: {len(plan)} moves in {self.plan_time:.2f}s")
            self.ai_plan = plan
            self.ai_step = 0
            self.mode = "ai"
            self.show_message(f"Plan found: {len(plan)} moves in {self.plan_time:.2f}s")

            # Animate success
            if self.show_effects:
                # Create a success particles at each block
                for row, col in self.grid.matter_elements:
                    self.particle_systems.append(
                        self.ParticleSystem(
                            col * self.cell_size + self.cell_size // 2 - self.scroll_x,
                            row * self.cell_size + self.cell_size // 2 - self.scroll_y,
                            Colors.GREEN,
                            count=15,
                            duration=1.0,
                        )
                    )
            return True
        else:
            print("A* planning failed")
            self.show_message("A* planning failed to find a solution")
            return False

    def run_hierarchical_planning(self):
        """Run hierarchical planning with timing and visual effects"""
        # First verify target shape
        if len(self.target_shape) != len(self.grid.matter_elements):
            self.show_message(
                f"Target must have exactly {len(self.grid.matter_elements)} blocks"
            )
            return False

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
            self.toggles["show_waypoints"].state = True

            self.show_message(
                f"Plan found: {len(plan)} moves with {len(waypoints)} waypoints"
            )

            # Add waypoint visualization effects
            if self.show_effects:
                for i, waypoint in enumerate(waypoints):
                    delay = i * 0.2
                    for row, col in waypoint:
                        # Create delayed particle effects for each waypoint
                        x = col * self.cell_size + self.cell_size // 2
                        y = row * self.cell_size + self.cell_size // 2

                        def create_delayed_particles(x, y, delay):
                            """Create particles after a delay"""
                            particles = self.ParticleSystem(
                                x - self.scroll_x,
                                y - self.scroll_y,
                                Colors.CYAN,
                                count=10,
                                duration=0.5,
                            )
                            self.animations.append(
                                self.Animation(
                                    0,
                                    1,
                                    delay,
                                    lambda v: (
                                        self.add_particles(particles)
                                        if v == 1
                                        else None
                                    ),
                                )
                            )

                        create_delayed_particles(x, y, delay)

            return True
        else:
            print("Hierarchical planning failed")
            self.show_message("Hierarchical planning failed to find a solution")
            return False

    def add_particles(self, particles):
        """Helper to add particle system after a delay"""
        self.particle_systems.append(particles)

    def run_parallel_planning(self):
        """Run parallel A* planning with timing and visual effects"""
        # First verify target shape
        if len(self.target_shape) != len(self.grid.matter_elements):
            self.show_message(
                f"Target must have exactly {len(self.grid.matter_elements)} blocks"
            )
            return False

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

            # Animate success with purple particles to indicate parallel processing
            if self.show_effects:
                for row, col in self.grid.matter_elements:
                    self.particle_systems.append(
                        self.ParticleSystem(
                            col * self.cell_size + self.cell_size // 2 - self.scroll_x,
                            row * self.cell_size + self.cell_size // 2 - self.scroll_y,
                            Colors.MAGENTA,  # Purple for parallel
                            count=20,
                            duration=1.2,
                            spread=40,
                        )
                    )
            return True
        else:
            print("Parallel A* planning failed")
            self.show_message("Parallel A* planning failed to find a solution")
            return False

    def run_parallel_hierarchical_planning(self):
        """Run parallel hierarchical planning with timing and visual effects"""
        # First verify target shape
        if len(self.target_shape) != len(self.grid.matter_elements):
            self.show_message(
                f"Target must have exactly {len(self.grid.matter_elements)} blocks"
            )
            return False

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
            self.toggles["show_waypoints"].state = True

            self.show_message(
                f"Plan found with {len(waypoints)} waypoints in {self.plan_time:.2f}s"
            )

            # Combine effects for both parallel and hierarchical processing
            if self.show_effects:
                # Create interleaved cyan and magenta particles
                colors = [Colors.CYAN, Colors.MAGENTA]
                for i, waypoint in enumerate(waypoints):
                    color = colors[i % 2]  # Alternate colors
                    delay = i * 0.15

                    for row, col in waypoint:
                        x = col * self.cell_size + self.cell_size // 2
                        y = row * self.cell_size + self.cell_size // 2

                        def create_delayed_particles(x, y, delay, color):
                            particles = self.ParticleSystem(
                                x - self.scroll_x,
                                y - self.scroll_y,
                                color,
                                count=15,
                                duration=0.6,
                            )
                            self.animations.append(
                                self.Animation(
                                    0,
                                    1,
                                    delay,
                                    lambda v, p=particles: (
                                        self.add_particles(p) if v == 1 else None
                                    ),
                                )
                            )

                        create_delayed_particles(x, y, delay, color)

            return True
        else:
            print("Parallel hierarchical planning failed")
            self.show_message(
                "Parallel hierarchical planning failed to find a solution"
            )
            return False

    def run(self):
        """
        Main loop for visualization with enhanced rendering and interactive UI.
        """
        running = True
        while running:
            # Calculate delta time and update state
            self.update()

            # Draw everything
            self.draw_grid()

            # Draw any screen-wide effects (like transitions)
            if self.transition_alpha > 0:
                # Create a surface for the flash/fade effect
                flash_surf = pygame.Surface(
                    (self.screen_width, self.screen_height), pygame.SRCALPHA
                )
                flash_surf.fill((255, 255, 255, int(self.transition_alpha)))
                self.screen.blit(flash_surf, (0, 0))

            # Update display
            pygame.display.flip()

            # Handle events - will return False if we should quit
            running = self.handle_events()

            # Cap the frame rate
            self.clock.tick(self.frame_rate)

        pygame.quit()

    def handle_events(self):
        """Centralized event handling method"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False  # Signal to quit

            # First pass events to UI elements
            if self.handle_ui_event(event):
                continue  # Event was handled by UI

            # Handle other event types
            if event.type == pygame.MOUSEBUTTONDOWN:
                # Handle scrolling with middle mouse button
                if event.button == 2:  # Middle click
                    self.scrolling = True
                    self.scroll_start_pos = event.pos

                # Handle mouse wheel scrolling
                elif event.button == 4 or event.button == 5:  # Wheel up or down
                    self.handle_mouse_wheel(1 if event.button == 4 else -1)

            elif event.type == pygame.MOUSEMOTION:
                # Handle scrolling drag
                if (
                    self.scrolling and pygame.mouse.get_pressed()[1]
                ):  # Middle button held
                    self.scroll_x = max(
                        0,
                        min(
                            self.scroll_x - event.rel[0],
                            self.width - self.screen_width + self.sidebar_width,
                        ),
                    )
                    self.scroll_y = max(
                        0,
                        min(
                            self.scroll_y - event.rel[1],
                            self.height - self.screen_height,
                        ),
                    )
                else:
                    self.scrolling = False

            elif event.type == pygame.MOUSEBUTTONUP:
                # End scrolling
                if event.button == 2:
                    self.scrolling = False

            elif event.type == pygame.VIDEORESIZE:
                # Handle window resize
                self.screen_width = event.w
                self.screen_height = event.h
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height),
                    pygame.RESIZABLE | pygame.HWSURFACE | pygame.DOUBLEBUF,
                )
                # Recalculate UI positions
                self.sidebar_width = min(300, self.screen_width // 4)
                self.create_ui_elements()

            elif event.type == pygame.KEYDOWN:
                self.handle_keydown(event)

        return True  # Continue running

    def handle_ui_event(self, event):
        """Handle UI-specific events and return True if handled"""
        # Handle button events
        for name, button in self.buttons.items():
            if button.handle_event(event):
                self.handle_button_click(name)
                return True

        # Handle slider events
        for name, slider in self.sliders.items():
            if slider.handle_event(event):
                # Slider value is automatically updated in handle_event
                return True

        # Handle toggle events
        for name, toggle in self.toggles.items():
            if toggle.handle_event(event):
                # Toggle updates its own state
                if name == "show_waypoints":
                    self.show_waypoints = toggle.state
                elif name == "show_effects":
                    self.show_effects = toggle.state
                return True

        # Handle grid cell clicks for target/obstacle placement
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Only process if not clicking sidebar
            if event.pos[0] < self.screen_width - self.sidebar_width:
                # Convert mouse position to grid coordinates
                col = (event.pos[0] + self.scroll_x) // self.cell_size
                row = (event.pos[1] + self.scroll_y) // self.cell_size

                # Ensure within grid bounds
                if 0 <= row < self.grid.n and 0 <= col < self.grid.m:
                    cell = (row, col)

                    # Target selection mode
                    if self.mode == "target_selection":
                        self.handle_target_click(cell)
                        return True

                    # Obstacle placement mode
                    elif self.mode == "obstacle_placement":
                        self.handle_obstacle_click(cell)
                        return True

        return False  # Event not handled by UI

    def handle_keydown(self, event):
        """Handle keyboard input"""
        if event.key == pygame.K_ESCAPE:
            # Show exit confirmation dialog
            dialog_result = self.show_dialog(
                "Exit Confirmation", "Are you sure you want to exit?", ["Yes", "No"]
            )
            if dialog_result == "Yes":
                pygame.event.post(pygame.event.Event(pygame.QUIT))

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
            self.sliders["speed"].value = self.ai_delay
            print(f"Animation speed increased: delay = {self.ai_delay}s")

        if event.key == pygame.K_MINUS or event.key == pygame.K_KP_MINUS:
            self.ai_delay = min(0.5, self.ai_delay * 2)
            self.sliders["speed"].value = self.ai_delay
            print(f"Animation speed decreased: delay = {self.ai_delay}s")

        # Toggle between manual and individual block mode
        if event.key == pygame.K_TAB:
            if self.mode == "manual":
                self.mode = "individual"
                self.show_message("Individual block mode")
            elif self.mode == "individual":
                self.mode = "manual"
                self.show_message("Manual mode")

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
                # Create highlight effect
                if self.show_effects:
                    row, col = self.grid.matter_elements[self.selected_index]
                    self.particle_systems.append(
                        self.ParticleSystem(
                            col * self.cell_size + self.cell_size // 2 - self.scroll_x,
                            row * self.cell_size + self.cell_size // 2 - self.scroll_y,
                            Colors.GREEN,
                            count=10,
                            duration=0.5,
                        )
                    )

            elif event.key == pygame.K_f:
                self.selected_index = (self.selected_index + 1) % len(
                    self.grid.matter_elements
                )
                # Create highlight effect
                if self.show_effects:
                    row, col = self.grid.matter_elements[self.selected_index]
                    self.particle_systems.append(
                        self.ParticleSystem(
                            col * self.cell_size + self.cell_size // 2 - self.scroll_x,
                            row * self.cell_size + self.cell_size // 2 - self.scroll_y,
                            Colors.GREEN,
                            count=10,
                            duration=0.5,
                        )
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
                success = self.grid.move_individual(moves)
                if success and self.show_effects:
                    # Add movement effect
                    row, col = self.grid.matter_elements[self.selected_index]
                    self.particle_systems.append(
                        self.ParticleSystem(
                            col * self.cell_size + self.cell_size // 2 - self.scroll_x,
                            row * self.cell_size + self.cell_size // 2 - self.scroll_y,
                            Colors.BLUE,
                            count=5,
                            duration=0.3,
                        )
                    )
                elif not success:
                    self.show_message("Invalid move!")

        # Quick access keys for planning
        if event.key == pygame.K_t:
            self.run_normal_planning()

        elif event.key == pygame.K_h:
            self.run_hierarchical_planning()

        elif event.key == pygame.K_p:
            self.run_parallel_planning()

        elif event.key == pygame.K_l:
            self.run_parallel_hierarchical_planning()

        # Toggle waypoints
        elif event.key == pygame.K_w and (event.mod & pygame.KMOD_CTRL):
            self.show_waypoints = not self.show_waypoints
            self.toggles["show_waypoints"].state = self.show_waypoints

        # Toggle visual effects
        elif event.key == pygame.K_v:
            self.show_effects = not self.show_effects
            self.toggles["show_effects"].state = self.show_effects

        # Take screenshot
        elif event.key == pygame.K_F12:
            self.take_screenshot()

    def take_screenshot(self):
        """Save a screenshot of the current view"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        pygame.image.save(self.screen, filename)
        self.show_message(f"Screenshot saved as {filename}")

    def show_dialog(self, title, message, options=None):
        """Show a modal dialog with options"""
        if options is None:
            options = ["OK"]

        # Create semi-transparent overlay
        overlay = pygame.Surface(
            (self.screen_width, self.screen_height), pygame.SRCALPHA
        )
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))

        # Create dialog panel
        dialog_width = 400
        dialog_height = 200
        dialog_x = self.screen_width // 2 - dialog_width // 2
        dialog_y = self.screen_height // 2 - dialog_height // 2

        dialog_panel = Panel(
            dialog_x,
            dialog_y,
            dialog_width,
            dialog_height,
            color=Colors.WHITE,
            border_radius=10,
        )
        dialog_panel.draw(self.screen)

        # Draw title
        title_text = self.fonts["large"].render(title, True, Colors.BLACK)
        title_rect = title_text.get_rect(
            centerx=self.screen_width // 2, y=dialog_y + 20
        )
        self.screen.blit(title_text, title_rect)

        # Draw message
        message_text = self.fonts["normal"].render(message, True, Colors.TEXT_PRIMARY)
        message_rect = message_text.get_rect(
            centerx=self.screen_width // 2, y=dialog_y + 70
        )
        self.screen.blit(message_text, message_rect)

        # Create buttons
        buttons = []
        button_width = 80
        total_button_width = button_width * len(options) + 10 * (len(options) - 1)
        start_x = self.screen_width // 2 - total_button_width // 2

        for i, option in enumerate(options):
            button_x = start_x + i * (button_width + 10)
            button = Button(
                button_x,
                dialog_y + dialog_height - 60,
                button_width,
                40,
                option,
                color=Colors.BLUE if i == 0 else Colors.GRAY,
            )
            buttons.append(button)
            button.draw(self.screen)

        pygame.display.flip()

        # Wait for user input
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return None

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = event.pos
                    for i, button in enumerate(buttons):
                        if button.is_point_inside(pos):
                            return options[i]

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return options[-1]  # Return last option (usually Cancel)
                    elif event.key == pygame.K_RETURN:
                        return options[0]  # Return first option (usually OK)
