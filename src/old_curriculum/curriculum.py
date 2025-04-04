import numpy as np
from environment import ProgrammableMatterEnv
from typing import List, Tuple, Dict


class ComprehensiveCurriculumManager:
    """
    Comprehensive curriculum learning for programmable matter following these phases:
    1. Translation (same shape, different locations) - increasing distances
    2. Shape transformation (changing shape) - increasing complexity
    3. Combined challenges with obstacles
    4. Advanced scenarios with more blocks and complex shapes
    """

    def __init__(self, grid_size: int = 10):
        self.n = grid_size
        self.m = grid_size
        self.level = 0
        self.max_level = 20  # Many more levels for smoother progression
        self.consecutive_successes = 0
        self.required_successes = 3
        self.success_threshold = -20

        # Shape libraries
        self.simple_shapes = ["square", "line_h", "line_v", "l_shape"]
        self.medium_shapes = ["t_shape", "z_shape", "plus", "u_shape"]
        self.complex_shapes = ["h_shape", "stairs", "cross"]

        # Current shape selections
        self.source_shape = "square"
        self.target_shape = "square"

        # Initial position variety
        self.current_initial_positions = [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        ]  # Default square

        # For tracking task type
        self.current_phase = "translation"  # translation, transformation, advanced

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

        return coords

    def get_level_config(self):
        """Get configuration for the current level."""
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
                obstacles = []
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

        # Phase 2: Shape changing without translation (Levels 6-8)
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
            obstacles = []

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
        else:
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
                    initial_x, initial_y = self.n - 4, 1

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

        # Make sure positions are valid
        target_x = min(max(0, target_x), self.n - 4)
        target_y = min(max(0, target_y), self.m - 4)

        # Get shape coordinates
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
        print(f"\nCurriculum Level {self.level}: Phase: {self.current_phase}")
        print(f"  Source Shape: {self.source_shape}, Target Shape: {self.target_shape}")
        print(f"  Initial: {initial_positions}")
        print(f"  Target: {target_positions}")
        if obstacles:
            print(f"  Obstacles: {len(obstacles)} obstacles")
        else:
            print("  No obstacles")

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
