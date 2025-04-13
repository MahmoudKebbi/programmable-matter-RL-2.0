import numpy as np
from collections import deque


class Grid:
    """
    Represents the programmable matter simulation grid.

    Attributes:
        n (int): Number of rows in the grid.
        m (int): Number of columns in the grid.
        grid (np.ndarray): A 2D array representing the grid state.
        matter_elements (list of tuple): List of (x, y) coordinates of matter elements.
        obstacles (set of tuple): Set of (x, y) coordinates of obstacles.
    """

    def __init__(self, n: int, m: int, initial_positions: list, obstacles: list = None):
        self.n = n
        self.m = m
        self.grid = np.zeros((n, m), dtype=int)
        self.matter_elements = []
        self.obstacles = set(obstacles or [])

        # Mark obstacles on grid
        for x, y in self.obstacles:
            if not (0 <= x < n and 0 <= y < m):
                raise ValueError(f"Obstacle position ({x}, {y}) out of bounds.")
            self.grid[x, y] = 2  # 2 represents obstacles

        # Validate initial positions
        seen = set()
        for x, y in initial_positions:
            if (x, y) in seen:
                raise ValueError("Duplicate initial positions.")
            if not (0 <= x < n and 0 <= y < m):
                raise ValueError(f"Initial position ({x}, {y}) out of bounds.")
            if (x, y) in self.obstacles:
                raise ValueError(f"Initial position ({x}, {y}) overlaps with obstacle.")
            seen.add((x, y))
            self.grid[x, y] = 1  # 1 represents matter
            self.matter_elements.append((x, y))

    def display_grid(self):
        """
        Prints the grid state for debugging.
        0: Empty, 1: Matter, 2: Obstacle
        """
        print("\nGrid State:")
        symbols = {0: ".", 1: "M", 2: "X"}
        for row in self.grid:
            print(" ".join(symbols[cell] for cell in row))

    def is_connected(self) -> bool:
        """
        Checks if all matter elements form a single connected component using BFS.
        Diagonally adjacent blocks are considered connected.

        Returns:
            bool: True if all elements are connected, False otherwise.
        """
        if not self.matter_elements:
            return True  # An empty set is trivially connected

        matter_set = set(self.matter_elements)
        visited = set()
        queue = deque()

        # Start BFS from the first matter element.
        start = self.matter_elements[0]
        queue.append(start)
        visited.add(start)

        # Moore neighborhood: 8 directions.
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
            cx, cy = queue.popleft()
            for dx, dy in directions:
                neighbor = (cx + dx, cy + dy)
                if neighbor in matter_set and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return len(visited) == len(self.matter_elements)

    def is_valid_move(self, dx: int, dy: int) -> bool:
        """
        Checks if moving the entire matter block by (dx, dy) is valid:
        1. New positions must be within grid boundaries.
        2. New positions must not overlap with obstacles.
        3. The structure must remain connected.

        Args:
            dx (int): Change in row direction.
            dy (int): Change in column direction.

        Returns:
            bool: True if the uniform move is valid, False otherwise.
        """
        new_positions = [(x + dx, y + dy) for x, y in self.matter_elements]

        # Check if all new positions are within grid bounds and not overlapping obstacles
        if any(
            not (0 <= x < self.n and 0 <= y < self.m) or (x, y) in self.obstacles
            for x, y in new_positions
        ):
            return False

        # Check if new positions are unique (no overlaps)
        if len(set(new_positions)) != len(new_positions):
            return False

        # Check connectivity without modifying self.matter_elements
        return self.is_connected_after_move(new_positions)

    def is_connected_after_move(self, new_positions: list) -> bool:
        """
        Simulates a move and checks if the new state remains connected.
        """
        matter_set = set(new_positions)
        visited = set()
        queue = deque([new_positions[0]])  # Start from any matter element
        visited.add(new_positions[0])

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
            cx, cy = queue.popleft()
            for dx, dy in directions:
                neighbor = (cx + dx, cy + dy)
                if neighbor in matter_set and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return len(visited) == len(matter_set)

    def move(self, dx: int, dy: int, verbose=False) -> bool:
        """
        Moves the entire matter block by (dx, dy) if valid.

        Args:
            dx (int): Change in row direction.
            dy (int): Change in column direction.
            verbose (bool): Whether to print error messages.

        Returns:
            bool: True if move was successful, False otherwise.
        """
        if self.is_valid_move(dx, dy):
            # Clear previous positions.
            for x, y in self.matter_elements:
                self.grid[x, y] = 0

            # Update positions.
            self.matter_elements = [(x + dx, y + dy) for x, y in self.matter_elements]

            # Write new positions.
            for x, y in self.matter_elements:
                self.grid[x, y] = 1

            return True  # Move was successful

        else:
            if verbose:  # Only print if verbose is True
                print(
                    "Invalid move! It would break connectivity, go out of bounds, or hit obstacles."
                )
            return False  # Move failed

    def move_individual(self, moves: dict, verbose=False) -> bool:
        """
        Moves individual blocks based on the provided dictionary of moves.

        Args:
            moves (dict): Dictionary mapping block indices to (dx, dy) moves.
            verbose (bool): Whether to print error messages.

        Returns:
            bool: True if move was successful, False otherwise.
        """
        new_positions = []
        seen = set()
        for i, (x, y) in enumerate(self.matter_elements):
            dx, dy = moves.get(i, (0, 0))
            nx, ny = x + dx, y + dy

            if (nx, ny) in seen:
                if verbose:
                    print("Invalid move: overlapping elements.")
                return False  # Move is invalid due to overlap

            if (
                not (0 <= nx < self.n and 0 <= ny < self.m)
                or (nx, ny) in self.obstacles
            ):
                if verbose:
                    print(
                        f"Invalid move: position ({nx}, {ny}) is out of bounds or occupied by obstacle."
                    )
                return False  # Move is invalid due to out-of-bounds or obstacle

            seen.add((nx, ny))
            new_positions.append((nx, ny))

        new_positions = sorted(new_positions)

        # Check connectivity
        original = self.matter_elements
        self.matter_elements = new_positions
        if not self.is_connected():
            self.matter_elements = original  # Revert to original positions
            if verbose:
                print("Invalid move: would break connectivity.")
            return False  # Move is invalid due to breaking connectivity

        # Update grid
        self.grid.fill(0)
        # Restore obstacles
        for x, y in self.obstacles:
            self.grid[x, y] = 2
        # Place matter in new positions
        for x, y in new_positions:
            self.grid[x, y] = 1

        return True  # Move was successful
