import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backup.grid import Grid
from moder_visualizer import Visualizer


def main():
    """
    Entry point for the programmable matter simulation.
    Initializes a 30x30 grid with a starting shape and a target shape.
    """
    n, m = 50,50
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


# import os
# import sys

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from src.backup.grid import Grid
# from moder_visualizer import Visualizer


# def main():
#     """
#     Entry point for the programmable matter simulation.
#     Initializes a 50x50 grid with 50 starting and 50 target positions.
#     """
#     n, m = 50, 50

#     # 5x10 starting grid with 50 positions
#     start_positions = [
#         # Row 9 (10 positions)
#         (9, 0),
#         (9, 1),
#         (9, 2),
#         (9, 3),
#         (9, 4),
#         (9, 5),
#         (9, 6),
#         (9, 7),
#         (9, 8),
#         (9, 9),
#         # Row 8 (10 positions)
#         (8, 0),
#         (8, 1),
#         (8, 2),
#         (8, 3),
#         (8, 4),
#         (8, 5),
#         (8, 6),
#         (8, 7),
#         (8, 8),
#         (8, 9),
#         # Row 7 (10 positions)
#         (7, 0),
#         (7, 1),
#         (7, 2),
#         (7, 3),
#         (7, 4),
#         (7, 5),
#         (7, 6),
#         (7, 7),
#         (7, 8),
#         (7, 9),
#         # Row 6 (10 positions) - Added
#         (6, 0),
#         (6, 1),
#         (6, 2),
#         (6, 3),
#         (6, 4),
#         (6, 5),
#         (6, 6),
#         (6, 7),
#         (6, 8),
#         (6, 9),
#         # Row 5 (10 positions) - Added
#         (5, 0),
#         (5, 1),
#         (5, 2),
#         (5, 3),
#         (5, 4),
#         (5, 5),
#         (5, 6),
#         (5, 7),
#         (5, 8),
#         (5, 9),
#     ]  # 5x10 starting shape with 50 positions

#     # Expanded heart-like shape with 50 positions
#     target_positions = [
#         # Original core shape (20 positions)
#         (3, 4),
#         (3, 5),
#         (4, 3),
#         (4, 4),
#         (4, 5),
#         (4, 6),
#         (5, 2),
#         (5, 3),
#         (5, 6),
#         (5, 7),
#         (6, 2),
#         (6, 3),
#         (6, 6),
#         (6, 7),
#         (7, 3),
#         (7, 4),
#         (7, 5),
#         (7, 6),
#         (8, 4),
#         (8, 5),
#         # First expansion (10 positions)
#         (2, 4),
#         (2, 5),  # Extending upward
#         (9, 4),
#         (9, 5),  # Extending downward
#         (5, 1),
#         (6, 1),  # Extending left
#         (5, 8),
#         (6, 8),  # Extending right
#         (3, 3),
#         (3, 6),  # Filling corners
#         # Additional expansions to reach 50 total (20 more positions)
#         # Upper expansion
#         (1, 4),
#         (1, 5),  # Top extension
#         (2, 3),
#         (2, 6),  # Upper corners
#         # Lower expansion
#         (10, 4),
#         (10, 5),  # Bottom extension
#         (9, 3),
#         (9, 6),  # Lower corners
#         # Side expansions
#         (4, 2),
#         (7, 2),  # Left side
#         (4, 7),
#         (7, 7),  # Right side
#         # Outer corners
#         (3, 2),
#         (3, 7),
#         (8, 3),
#         (8, 6),
#         # Fill remaining gaps for symmetry
#         (2, 2),
#         (2, 7),
#         (8, 2),
#         (8, 7),
#         (4, 1),
#         (7, 1),
#         (4, 8),
#         (7, 8),
#     ]  # Expanded target shape with 50 positions

#     grid = Grid(n, m, start_positions)
#     visualizer = Visualizer(grid, target_positions)
#     visualizer.run()


# if __name__ == "__main__":
#     main()
