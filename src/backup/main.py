import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backup.grid import Grid
from src.backup.visualizer import Visualizer


def main():
    """
    Entry point for the programmable matter simulation.
    Initializes a 30x30 grid with a starting shape and a target shape.
    """
    n, m = 30, 30
    start_positions = [
        # Row 9 (10 positions)
        (9, 0),
        (9, 1),
        (9, 2),
        (9, 3),
        (9, 4),
        (9, 5),
        (9, 6),
        (9, 7),
        (9, 8),
        (9, 9),
        # Row 8 (10 positions)
        (8, 0),
        (8, 1),
        (8, 2),
        (8, 3),
        (8, 4),
        (8, 5),
        (8, 6),
        (8, 7),
        (8, 8),
        (8, 9),
        # Row 7 (10 positions)
        (7, 0),
        (7, 1),
        (7, 2),
        (7, 3),
        (7, 4),
        (7, 5),
        (7, 6),
        (7, 7),
        (7, 8),
        (7, 9),
    ]  # 3x10 starting shape with 30 positions

    target_positions = [
        # Original shape (20 positions)
        (3, 4),
        (3, 5),
        (4, 3),
        (4, 4),
        (4, 5),
        (4, 6),
        (5, 2),
        (5, 3),
        (5, 6),
        (5, 7),
        (6, 2),
        (6, 3),
        (6, 6),
        (6, 7),
        (7, 3),
        (7, 4),
        (7, 5),
        (7, 6),
        (8, 4),
        (8, 5),
        # Additional positions to reach 30 total
        (2, 4),
        (2, 5),  # Extending upward
        (9, 4),
        (9, 5),  # Extending downward
        (5, 1),
        (6, 1),  # Extending left
        (5, 8),
        (6, 8),  # Extending right
        (3, 3),
        (3, 6),  # Filling corners
    ]  # Expanded target shape with 30 positions

    grid = Grid(n, m, start_positions)
    visualizer = Visualizer(grid, target_positions)
    visualizer.run()


if __name__ == "__main__":
    main()
