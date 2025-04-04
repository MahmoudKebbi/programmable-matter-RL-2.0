import numpy as np
from environment import ProgrammableMatterEnv
from typing import List, Tuple, Dict


class ExtendedCurriculumManager:
    """
    Extended comprehensive curriculum learning for programmable matter with scaling complexity:
    1. Translation (same shape, different locations) - increasing distances
    2. Shape transformation (changing shape) - increasing complexity
    3. Combined challenges with obstacles
    4. Advanced scenarios with complex shapes
    5. Scaling scenarios - more blocks, larger grids
    6. Multi-component scenarios - multiple independent shapes
    7. Master challenges - extreme complexity
    """

    def __init__(self, initial_grid_size: int = 10):
        self.initial_grid_size = initial_grid_size
        self.n = initial_grid_size
        self.m = initial_grid_size
        self.level = 0
        self.max_level = 35  # Extended to 35 levels
        self.consecutive_successes = 0
        self.required_successes = 6
        self.success_threshold = -20

        # Shape libraries
        self.simple_shapes = ["square", "line_h", "line_v", "l_shape"]
        self.medium_shapes = ["t_shape", "z_shape", "plus", "u_shape"]
        self.complex_shapes = ["h_shape", "stairs", "cross"]
        self.large_shapes = ["large_square", "spiral", "c_shape", "donut"]

        # Current shape selections
        self.source_shape = "square"
        self.target_shape = "square"

        # Track multi-component configurations
        self.num_components = 1
        self.multi_component_shapes = []

        # Initial position variety
        self.current_initial_positions = [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        ]  # Default square

        # For tracking task type
        self.current_phase = "translation"

        # Create initial environment
        self.env = self.create_environment()

    def get_shape_coords(
        self, shape_type: str, start_x: int, start_y: int
    ) -> List[Tuple[int, int]]:
        """Get coordinates for a specific shape at given position."""
        coords = []

        # Simple shapes
        if shape_type == "square":
            coords = [
                (start_x, start_y),
                (start_x, start_y + 1),
                (start_x + 1, start_y),
                (start_x + 1, start_y + 1),
            ]
        elif shape_type == "line_h":
            coords = [(start_x, start_y + i) for i in range(3)]
        elif shape_type == "line_v":
            coords = [(start_x + i, start_y) for i in range(3)]
        elif shape_type == "l_shape":
            coords = [
                (start_x, start_y),
                (start_x + 1, start_y),
                (start_x + 1, start_y + 1),
            ]

        # Medium shapes
        elif shape_type == "t_shape":
            coords = [
                (start_x, start_y + 1),
                (start_x + 1, start_y),
                (start_x + 1, start_y + 1),
                (start_x + 1, start_y + 2),
            ]
        elif shape_type == "z_shape":
            coords = [
                (start_x, start_y),
                (start_x, start_y + 1),
                (start_x + 1, start_y + 1),
                (start_x + 1, start_y + 2),
            ]
        elif shape_type == "plus":
            coords = [
                (start_x, start_y + 1),
                (start_x + 1, start_y),
                (start_x + 1, start_y + 1),
                (start_x + 1, start_y + 2),
                (start_x + 2, start_y + 1),
            ]
        elif shape_type == "u_shape":
            coords = [
                (start_x, start_y),
                (start_x, start_y + 2),
                (start_x + 1, start_y),
                (start_x + 1, start_y + 1),
                (start_x + 1, start_y + 2),
            ]

        # Complex shapes
        elif shape_type == "h_shape":
            coords = [
                (start_x, start_y),
                (start_x, start_y + 2),
                (start_x + 1, start_y),
                (start_x + 1, start_y + 1),
                (start_x + 1, start_y + 2),
                (start_x + 2, start_y),
                (start_x + 2, start_y + 2),
            ]
        elif shape_type == "stairs":
            coords = [
                (start_x, start_y),
                (start_x, start_y + 1),
                (start_x + 1, start_y + 1),
                (start_x + 1, start_y + 2),
                (start_x + 2, start_y + 2),
            ]
        elif shape_type == "cross":
            coords = [
                (start_x, start_y + 1),
                (start_x + 1, start_y),
                (start_x + 1, start_y + 1),
                (start_x + 1, start_y + 2),
                (start_x + 2, start_y + 1),
            ]

        # Large shapes (for higher levels)
        elif shape_type == "large_square":
            coords = [(start_x + i, start_y + j) for i in range(3) for j in range(3)]
        elif shape_type == "spiral":
            coords = [
                (start_x, start_y),
                (start_x, start_y + 1),
                (start_x, start_y + 2),
                (start_x + 1, start_y + 2),
                (start_x + 2, start_y + 2),
                (start_x + 2, start_y + 1),
                (start_x + 2, start_y),
                (start_x + 1, start_y),
                (start_x + 1, start_y + 1),
            ]
        elif shape_type == "c_shape":
            coords = [
                (start_x, start_y),
                (start_x, start_y + 1),
                (start_x, start_y + 2),
                (start_x + 1, start_y),
                (start_x + 2, start_y),
                (start_x + 2, start_y + 1),
                (start_x + 2, start_y + 2),
            ]
        elif shape_type == "donut":
            coords = [
                (start_x, start_y),
                (start_x, start_y + 1),
                (start_x, start_y + 2),
                (start_x + 1, start_y),
                (start_x + 1, start_y + 2),
                (start_x + 2, start_y),
                (start_x + 2, start_y + 1),
                (start_x + 2, start_y + 2),
            ]

        return coords

    def generate_multi_component(
        self, num_components: int, min_separation: int = 2
    ) -> List[Tuple[int, int]]:
        """Generate a multi-component configuration with appropriate spacing."""
        all_coords = []
        component_shapes = []

        # Select shapes for each component
        shape_pool = self.simple_shapes + self.medium_shapes
        for i in range(num_components):
            shape = np.random.choice(shape_pool)
            component_shapes.append(shape)

        self.multi_component_shapes = component_shapes

        # Place components with proper spacing
        for i, shape in enumerate(component_shapes):
            # Calculate position with proper spacing
            row = (i // 2) * (min_separation + 3)  # 3 is approximate shape size
            col = (i % 2) * (min_separation + 3)

            # Add spacing for larger grids
            if self.n > 15:
                row += 1
                col += 1

            component_coords = self.get_shape_coords(shape, row, col)
            all_coords.extend(component_coords)

        return all_coords

    def update_grid_size(self):
        """Update grid size based on level."""
        # Base size
        base_size = self.initial_grid_size

        # Increase grid size for higher levels
        if self.level >= 21 and self.level < 25:
            self.n = base_size + 4  # 14x14 grid
            self.m = base_size + 4
        elif self.level >= 25 and self.level < 30:
            self.n = base_size + 8  # 18x18 grid
            self.m = base_size + 8
        elif self.level >= 30:
            self.n = base_size + 12  # 22x22 grid
            self.m = base_size + 12

    def get_level_config(self):
        """Get configuration for the current level."""
        # Update grid size based on current level
        self.update_grid_size()

        # Update number of components for multi-component phases
        if self.level >= 25:
            self.num_components = min(1 + (self.level - 25) // 2, 4)  # Max 4 components

        # Set all obstacle types to empty initially
        obstacles = []

        # Phase 1: Translation (Levels 0-5)
        if self.level <= 5:
            self.current_phase = "translation"

            # Use square shape for simplicity in early stages
            self.source_shape = "square"
            self.target_shape = "square"

            # Vary distance based on level
            if self.level <= 1:
                # Very close (1-3 blocks away)
                distance = 2
            elif self.level <= 3:
                # Medium distance
                distance = 4
                obstacles = [] if self.level == 2 else [(2, 2)]
            else:
                # Far distance
                distance = 7
                obstacles = [] if self.level == 4 else [(3, 3), (3, 4), (4, 3)]

            initial_x, initial_y = 1, 1
            target_x = min(initial_x + distance, self.n - 3)
            target_y = min(initial_y + distance, self.m - 3)

        # Phase 2: Shape transformation (Levels 6-8)
        elif self.level <= 8:
            self.current_phase = "transformation"
            distance = 0

            # Change shapes but keep position the same or very close
            shapes = self.simple_shapes
            self.source_shape = shapes[self.level % len(shapes)]
            self.target_shape = shapes[(self.level + 1) % len(shapes)]

            initial_x, initial_y = 2, 2
            target_x = initial_x + distance
            target_y = initial_y + distance

        # Phase 3: Shape changing with translation (Levels 9-14)
        elif self.level <= 14:
            self.current_phase = "transformation_with_translation"

            # Gradually increase complexity and distance
            level_in_phase = self.level - 9

            # Determine shape complexity
            if level_in_phase < 2:
                # Simple shapes with small distance
                shapes = self.simple_shapes
                distance = 3
                obstacles = []
            elif level_in_phase < 4:
                # Simple shapes with medium distance
                shapes = self.simple_shapes
                distance = 5
                obstacles = [(3, 3)] if level_in_phase == 3 else []
            else:
                # Medium shapes with larger distance
                shapes = self.medium_shapes
                distance = 6
                obstacles = [(3, 3), (3, 4), (4, 3)] if level_in_phase >= 5 else []

            # Select shapes
            if level_in_phase % 2 == 0:
                self.source_shape = shapes[level_in_phase % len(shapes)]
                self.target_shape = shapes[(level_in_phase + 1) % len(shapes)]
            else:
                self.source_shape = shapes[(level_in_phase + 1) % len(shapes)]
                self.target_shape = shapes[level_in_phase % len(shapes)]

            # Calculate positions
            initial_x, initial_y = 1, 1
            target_x = min(initial_x + distance, self.n - 4)
            target_y = min(initial_y + distance, self.m - 4)

        # Phase 4: Advanced scenarios (Levels 15-20)
        elif self.level <= 20:
            self.current_phase = "advanced"
            level_in_phase = self.level - 15

            # More complex shapes and obstacle patterns
            if level_in_phase < 2:
                # Medium shapes with complex obstacles
                shapes = self.medium_shapes
                distance = 5
                # Create a wall obstacle
                obstacles = [(i, 4) for i in range(2, 7)]
            elif level_in_phase < 4:
                # Complex shapes
                shapes = self.complex_shapes
                distance = 6
                # Create an L-shaped obstacle
                obstacles = [(i, 5) for i in range(2, 6)] + [
                    (5, i) for i in range(2, 5)
                ]
            else:
                # Very complex with position variations
                shapes = self.complex_shapes + self.medium_shapes
                distance = 7

                # U-shaped obstacle maze
                obstacles = (
                    [(i, 3) for i in range(3, 8)]
                    + [(3, i) for i in range(4, 8)]
                    + [(i, 7) for i in range(4, 8)]
                )

                # Vary initial position
                initial_x, initial_y = 1, 1
                if level_in_phase == 5:
                    # Start from bottom instead of top
                    initial_x, initial_y = self.n - 5, 1

            # Select shapes
            shape_idx = level_in_phase % len(shapes)
            self.source_shape = shapes[shape_idx]
            next_shape_idx = (shape_idx + 1 + level_in_phase) % len(shapes)
            self.target_shape = shapes[next_shape_idx]

            # Calculate positions if not already set
            if "initial_x" not in locals():
                initial_x, initial_y = 1, 1

            target_x = min(initial_x + distance, self.n - 4)
            target_y = min(initial_y + distance, self.m - 4)

        # Phase 5: Scaling up - larger shapes and grids (Levels 21-24)
        elif self.level <= 24:
            self.current_phase = "scaling_up"
            level_in_phase = self.level - 21

            # Use larger shapes
            shapes = self.complex_shapes + self.large_shapes
            distance = 8

            # Complex obstacle patterns
            if level_in_phase < 2:
                # Scattered obstacles
                obstacle_count = 10 + level_in_phase * 4
                possible_positions = [
                    (i, j) for i in range(1, self.n - 1) for j in range(1, self.m - 1)
                ]
                np.random.shuffle(possible_positions)
                obstacles = possible_positions[:obstacle_count]
            else:
                # Maze-like obstacles
                center_x, center_y = self.n // 2, self.m // 2
                obstacles = [
                    (i, center_y) for i in range(2, self.n - 2) if i != center_x
                ]
                obstacles += [
                    (center_x, j) for j in range(2, self.m - 2) if j != center_y
                ]

            # Select shapes
            shape_idx = level_in_phase % len(shapes)
            self.source_shape = shapes[shape_idx]
            self.target_shape = shapes[(shape_idx + 2) % len(shapes)]

            # Calculate positions
            initial_x, initial_y = 1, 1
            target_x = self.n - 6
            target_y = self.m - 6

        # Phase 6: Multi-component challenges (Levels 25-29)
        elif self.level <= 29:
            self.current_phase = "multi_component"
            level_in_phase = self.level - 25

            # Generate multi-component configurations
            components = self.num_components

            # These will be multi-shape configurations
            initial_positions = self.generate_multi_component(components)

            # For multi-component, target is same shapes but rearranged
            np.random.seed(42 + self.level)  # Consistent but different seed
            target_positions = self.generate_multi_component(components)

            # Obstacles depend on level
            obstacle_density = 0.05 + (level_in_phase * 0.02)  # 5-15% of grid
            obstacle_count = int(self.n * self.m * obstacle_density)

            # Place obstacles avoiding blocks
            all_positions = set([(i, j) for i in range(self.n) for j in range(self.m)])
            block_positions = set(initial_positions + target_positions)
            valid_obstacle_positions = list(all_positions - block_positions)
            np.random.shuffle(valid_obstacle_positions)
            obstacles = valid_obstacle_positions[:obstacle_count]

            return initial_positions, target_positions, obstacles

        # Phase 7: Master challenges (Levels 30+)
        else:
            self.current_phase = "master"
            level_in_phase = self.level - 30

            # Extreme challenges with large multi-component configurations
            components = min(3 + level_in_phase, 6)  # Up to 6 separate components

            # Complex shapes with transformations
            shape_pool = self.medium_shapes + self.complex_shapes + self.large_shapes

            # For extreme challenges, sometimes use different shapes for target
            if level_in_phase >= 3:
                # Generate source components
                self.multi_component_shapes = [
                    np.random.choice(shape_pool) for _ in range(components)
                ]
                initial_positions = []

                # Position each component
                for i, shape in enumerate(self.multi_component_shapes):
                    row = (i // 2) * 5
                    col = (i % 3) * 5
                    component_coords = self.get_shape_coords(shape, row, col)
                    initial_positions.extend(component_coords)

                # Generate target with different shapes but same count
                target_shapes = [
                    np.random.choice(shape_pool) for _ in range(components)
                ]
                target_positions = []

                # Position each target component
                for i, shape in enumerate(target_shapes):
                    row = (i // 2) * 5 + (self.n // 2)
                    col = (i % 3) * 5
                    component_coords = self.get_shape_coords(shape, row, col)
                    target_positions.extend(component_coords)
            else:
                # Generate multi-component with same shapes
                initial_positions = self.generate_multi_component(components)

                # Target is rearranged version of same shapes
                np.random.seed(100 + self.level)
                target_positions = self.generate_multi_component(components)

            # Complex obstacles
            obstacle_count = int(self.n * self.m * 0.15)  # 15% obstacles

            # Place obstacles avoiding blocks
            all_positions = set([(i, j) for i in range(self.n) for j in range(self.m)])
            block_positions = set(initial_positions + target_positions)
            valid_obstacle_positions = list(all_positions - block_positions)
            np.random.shuffle(valid_obstacle_positions)
            obstacles = valid_obstacle_positions[:obstacle_count]

            return initial_positions, target_positions, obstacles

        # For levels that don't return directly, get shape coordinates
        initial_positions = self.get_shape_coords(
            self.source_shape, initial_x, initial_y
        )
        target_positions = self.get_shape_coords(self.target_shape, target_x, target_y)

        # Update current initial positions for reference
        self.current_initial_positions = initial_positions

        return initial_positions, target_positions, obstacles

    def create_environment(self):
        """Create environment with appropriate difficulty for current level."""
        # Get configuration for current level
        initial_positions, target_positions, obstacles = self.get_level_config()

        # Log curriculum progression
        print(
            f"\nCurriculum Level {self.level}: Phase: {self.current_phase}, Grid: {self.n}x{self.m}"
        )

        if self.level < 25:  # Standard phases
            print(
                f"  Source Shape: {self.source_shape}, Target Shape: {self.target_shape}"
            )
        else:  # Multi-component phases
            print(
                f"  Components: {self.num_components}, Blocks: {len(initial_positions)}"
            )
            if hasattr(self, "multi_component_shapes") and self.multi_component_shapes:
                print(f"  Shapes: {self.multi_component_shapes}")

        print(f"  Initial: {len(initial_positions)} blocks")
        print(f"  Target: {len(target_positions)} blocks")
        print(f"  Obstacles: {len(obstacles)} obstacles")

        return ProgrammableMatterEnv(
            self.n, self.m, initial_positions, target_positions, obstacles
        )

    def evaluate_progress(self, episode_reward: float, info: dict = None) -> bool:
        """Evaluate if the agent has mastered the current level."""
        # Check if episode was successful
        solved = info.get("solved", False) if info else False

        if solved or episode_reward > self.success_threshold:
            self.consecutive_successes += 1
            print(f"Success! ({self.consecutive_successes}/{self.required_successes})")
        else:
            self.consecutive_successes = 0

        # Advance level if enough successes
        if self.consecutive_successes >= self.required_successes:
            if self.level < self.max_level:
                self.level += 1
                self.consecutive_successes = 0

                # Create new environment with increased difficulty
                self.env = self.create_environment()

                return True  # Level increased

        return False  # Level stayed the same

    def get_environment(self):
        """Get the current environment."""
        return self.env

    def get_level(self) -> int:
        """Get current curriculum level."""
        return self.level

    def get_phase(self) -> str:
        """Get current curriculum phase."""
        return self.current_phase
