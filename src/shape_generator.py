import numpy as np
import random
from collections import deque

# REVIEW: Using Moore (8-way) neighborhood for generation, matching grid checks.
DIRECTIONS_8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

# REVIEW: 4-way connectivity option (might generate simpler shapes)
DIRECTIONS_4 = [(0, 1), (1, 0), (0, -1), (-1, 0)]


def generate_random_connected_shape(
    grid_size,
    num_blocks,
    min_distance_from_edge=1,
    connectivity=8,  # REVIEW: Added option for 4 or 8 way connectivity
    max_attempts_per_block=20,  # REVIEW: Limit attempts per block to avoid infinite loops
):
    """
    Generates a random connected shape using a randomized growth approach.

    Args:
        grid_size (tuple): (rows N, columns M) of the grid.
        num_blocks (int): Number of blocks in the shape.
        min_distance_from_edge (int): Minimum distance from grid edges.
        connectivity (int): 4 or 8, for neighborhood definition.
        max_attempts_per_block (int): Max attempts to add a single block.

    Returns:
        list: List of (row, col) coordinates for the shape, or None if failed.
    """
    n, m = grid_size
    if num_blocks <= 0:
        return []
    if num_blocks > (n - 2 * min_distance_from_edge) * (m - 2 * min_distance_from_edge):
        print("Warning: Requesting more blocks than available space allows.")
        return None  # Cannot generate shape

    directions = DIRECTIONS_8 if connectivity == 8 else DIRECTIONS_4

    # Try multiple times to generate a valid shape
    for _ in range(50):  # Max attempts to generate the whole shape
        # Start with a random valid position
        try:
            start_x = random.randint(
                min_distance_from_edge, n - 1 - min_distance_from_edge
            )
            start_y = random.randint(
                min_distance_from_edge, m - 1 - min_distance_from_edge
            )
        except ValueError:  # Happens if grid is too small for min_distance
            print(
                f"Error: Grid size {grid_size} too small for min_distance_from_edge {min_distance_from_edge}."
            )
            return None

        shape = [(start_x, start_y)]
        shape_set = {(start_x, start_y)}  # Use set for faster checking
        possible_neighbors = set()  # Potential locations to add next block

        # Initial neighbors
        for dx, dy in directions:
            nx, ny = start_x + dx, start_y + dy
            if (
                min_distance_from_edge <= nx < n - min_distance_from_edge
                and min_distance_from_edge <= ny < m - min_distance_from_edge
            ):
                possible_neighbors.add((nx, ny))

        # Grow the shape
        blocks_added = 1
        while blocks_added < num_blocks:
            if not possible_neighbors:
                # Failed to grow the shape to the desired size from this start point
                break  # Break inner loop, try a new starting point

            # Choose a random neighbor to add
            next_pos = random.choice(list(possible_neighbors))
            possible_neighbors.remove(next_pos)

            # Check if it's already in the shape (shouldn't happen with this logic, but safe check)
            if next_pos in shape_set:
                continue

            # Add the block
            shape.append(next_pos)
            shape_set.add(next_pos)
            blocks_added += 1

            # Add its valid neighbors to the possible expansion set
            nx, ny = next_pos
            for dx, dy in directions:
                nnx, nny = nx + dx, ny + dy
                new_neighbor = (nnx, nny)
                if (
                    min_distance_from_edge <= nnx < n - min_distance_from_edge
                    and min_distance_from_edge <= nny < m - min_distance_from_edge
                    and new_neighbor not in shape_set
                ):  # Only add if valid and not already part of shape
                    possible_neighbors.add(new_neighbor)

        # If we successfully generated the shape, return it (sorted)
        if blocks_added == num_blocks:
            return sorted(shape)  # Return sorted list

    # If all attempts failed
    print(
        f"Warning: Failed to generate connected shape with {num_blocks} blocks after multiple attempts."
    )
    return None


def get_shape_centroid(shape):
    """Calculates the centroid (average position) of a shape."""
    if not shape:
        return (0, 0)
    cx = sum(p[0] for p in shape) / len(shape)
    cy = sum(p[1] for p in shape) / len(shape)
    return (cx, cy)


def is_shapes_distant(shape1, shape2, min_distance):
    """
    Check if the minimum Manhattan distance between any block in shape1
    and any block in shape2 is at least min_distance.
    """
    if not shape1 or not shape2:
        return True  # No overlap if one shape is empty

    min_dist_found = float("inf")
    for x1, y1 in shape1:
        for x2, y2 in shape2:
            dist = abs(x1 - x2) + abs(y1 - y2)
            min_dist_found = min(min_dist_found, dist)
            if min_dist_found < min_distance:  # Early exit if condition violated
                return False
    return True  # Minimum distance requirement met


def generate_distinct_shapes(
    grid_size,
    num_blocks,
    num_shapes=2,
    min_distance=3,  # Min distance between blocks of different shapes
    min_centroid_distance=None,  # Optional: Min distance between shape centroids
    max_attempts=100,  # Max attempts to find a distinct shape
    connectivity=8,
    min_distance_from_edge=1,
):
    """
    Generates multiple distinct shapes that are sufficiently far apart.

    Args:
        grid_size (tuple): (rows N, columns M).
        num_blocks (int): Number of blocks per shape.
        num_shapes (int): How many shapes to generate.
        min_distance (int): Minimum Manhattan distance between blocks of different shapes.
        min_centroid_distance (float, optional): Minimum distance between shape centroids.
        max_attempts (int): Max attempts to generate each distinct shape.
        connectivity (int): Connectivity (4 or 8) for shape generation.
        min_distance_from_edge (int): Min distance from edge for generation.


    Returns:
        list: List of shapes (each shape is a list of (row, col) tuples),
              or None if failed to generate the required number of distinct shapes.
    """
    shapes = []
    attempts = 0
    while len(shapes) < num_shapes and attempts < max_attempts * num_shapes:
        attempts += 1
        new_shape = generate_random_connected_shape(
            grid_size, num_blocks, min_distance_from_edge, connectivity
        )

        if new_shape is None:  # Generation failed
            continue

        # Check distance constraints against existing shapes
        is_distinct_enough = True
        for existing_shape in shapes:
            # 1. Check minimum block distance
            if not is_shapes_distant(new_shape, existing_shape, min_distance):
                is_distinct_enough = False
                break
            # 2. Check minimum centroid distance (optional)
            if min_centroid_distance is not None:
                c1 = get_shape_centroid(new_shape)
                c2 = get_shape_centroid(existing_shape)
                centroid_dist = np.sqrt(
                    (c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2
                )  # Euclidean
                # centroid_dist = abs(c1[0]-c2[0]) + abs(c1[1]-c2[1]) # Manhattan
                if centroid_dist < min_centroid_distance:
                    is_distinct_enough = False
                    break

        if is_distinct_enough:
            shapes.append(new_shape)
            attempts = 0  # Reset attempts counter for the *next* shape

    if len(shapes) == num_shapes:
        return shapes
    else:
        print(
            f"Warning: Failed to generate {num_shapes} distinct shapes satisfying constraints."
        )
        return None  # Indicate failure
