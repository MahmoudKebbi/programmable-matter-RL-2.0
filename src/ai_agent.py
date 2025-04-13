import heapq
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import distance_transform_edt
from typing import List, Tuple, Optional, Dict, Set, Any
from collections import deque
import time
import multiprocessing as mp
from copy import deepcopy

# Global direction constants
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

# Connectivity check cache for optimization
connectivity_cache = {}


def state_to_tuple(state: List[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    """Convert state to a hashable tuple for caching and visited set."""
    return tuple(sorted(state))


def is_state_connected(state: List[Tuple[int, int]]) -> bool:
    """Check if all blocks in the state are connected."""
    state_tuple = tuple(sorted(state))

    # Check cache first
    if state_tuple in connectivity_cache:
        return connectivity_cache[state_tuple]

    if not state:
        return True

    state_set = set(state)
    visited = set()
    queue = deque([state[0]])
    visited.add(state[0])

    while queue:
        x, y = queue.popleft()
        for dx, dy in DIRECTIONS:
            neighbor = (x + dx, y + dy)
            if neighbor in state_set and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    result = len(visited) == len(state)
    connectivity_cache[state_tuple] = result
    return result


def is_still_connected_after_move(
    state: List[Tuple[int, int]], block_idx: int, new_pos: Tuple[int, int]
) -> bool:
    """Check if state remains connected after moving a single block."""
    if len(state) <= 1:
        return True

    state_set = set(state)
    orig_pos = state[block_idx]

    # If the block has no connections, it can't be moved (except for single block puzzles)
    has_connections = False
    for dx, dy in DIRECTIONS:
        neighbor = (orig_pos[0] + dx, orig_pos[1] + dy)
        if neighbor in state_set:
            has_connections = True
            break

    if not has_connections and len(state) > 1:
        return False

    # Check if new position will be connected to at least one existing block
    x, y = new_pos
    will_be_connected = False
    for dx, dy in DIRECTIONS:
        neighbor = (x + dx, y + dy)
        if neighbor in state_set and neighbor != orig_pos:
            will_be_connected = True
            break

    if not will_be_connected and len(state) > 1:
        return False

    # Full connectivity check
    new_state = state.copy()
    new_state[block_idx] = new_pos
    return is_state_connected(new_state)


def find_articulation_points(state: List[Tuple[int, int]]) -> Set[int]:
    """Find blocks that, if removed, would disconnect the structure."""
    if len(state) <= 2:
        return set()

    articulation_points = set()
    state_set = set(state)

    for i, pos in enumerate(state):
        # Remove this block temporarily
        reduced_state = state.copy()
        reduced_state.pop(i)

        # If remaining blocks aren't connected, this is an articulation point
        if not is_state_connected(reduced_state):
            articulation_points.add(i)

    return articulation_points


def compute_neighbor_map(state: List[Tuple[int, int]]) -> Dict[int, List[int]]:
    """Create a map of each block's neighboring blocks."""
    neighbor_map = {i: [] for i in range(len(state))}
    state_set = set(state)

    for i, (x, y) in enumerate(state):
        for dx, dy in DIRECTIONS:
            neighbor = (x + dx, y + dy)
            if neighbor in state_set:
                j = state.index(neighbor)
                if j != i:  # Don't add self as neighbor
                    neighbor_map[i].append(j)

    return neighbor_map


def heuristic(state: List[Tuple[int, int]], target: List[Tuple[int, int]]) -> int:
    """Improved heuristic function using the Hungarian algorithm."""
    cost_matrix = np.zeros((len(state), len(target)))
    for i, (x1, y1) in enumerate(state):
        for j, (x2, y2) in enumerate(target):
            cost_matrix[i, j] = abs(x1 - x2) + abs(y1 - y2)

    # Use Hungarian algorithm to find optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_cost = cost_matrix[row_ind, col_ind].sum()

    return total_cost


def get_successors(
    state: List[Tuple[int, int]],
    n: int,
    m: int,
    obstacles: Set[Tuple[int, int]] = None,
    move_priority: str = None,
):
    """Generate successors with optimizations and obstacle avoidance."""
    obstacles = obstacles or set()  # Default to empty set if None
    successors = []
    state_set = set(state)

    # Pre-compute neighbor information
    neighbor_map = compute_neighbor_map(state)

    # Collection for storing successors by type
    uniform_moves = []
    individual_moves = []
    pair_moves = []
    group_moves = []

    # 1. Uniform moves (all blocks move the same direction)
    for dx, dy in DIRECTIONS:
        new_state = [(x + dx, y + dy) for x, y in state]
        # Check bounds and obstacles
        if all(
            0 <= x < n and 0 <= y < m and (x, y) not in obstacles for x, y in new_state
        ):
            uniform_moves.append((new_state, [(i, dx, dy) for i in range(len(state))]))

    # Find leaf blocks and articulation points
    leaves = [i for i, neighbors in neighbor_map.items() if len(neighbors) <= 1]
    articulation_points = find_articulation_points(state)

    # 2. Single block moves
    movable_blocks = [i for i in range(len(state)) if i not in articulation_points]
    if not movable_blocks and leaves:
        movable_blocks = leaves

    for i in movable_blocks:
        x, y = state[i]
        for dx, dy in DIRECTIONS:
            new_x, new_y = x + dx, y + dy
            new_pos = (new_x, new_y)

            # Check bounds, collisions, and obstacles
            if (
                not (0 <= new_x < n and 0 <= new_y < m)
                or new_pos in state_set
                or new_pos in obstacles
            ):
                continue

            # Connectivity check
            if is_still_connected_after_move(state, i, new_pos):
                new_state = state.copy()
                new_state[i] = new_pos
                individual_moves.append((new_state, [(i, dx, dy)]))

    # 3. Multi-block moves (for pairs of adjacent blocks)
    if len(state) >= 2:
        for i, neighbors in neighbor_map.items():
            for j in neighbors:
                if i < j:  # Process each pair only once
                    for dx, dy in DIRECTIONS:
                        new_state = state.copy()

                        # Move both blocks
                        new_i_pos = (state[i][0] + dx, state[i][1] + dy)
                        new_j_pos = (state[j][0] + dx, state[j][1] + dy)

                        # Check bounds, collisions, and obstacles
                        if (
                            not (0 <= new_i_pos[0] < n and 0 <= new_i_pos[1] < m)
                            or not (0 <= new_j_pos[0] < n and 0 <= new_j_pos[1] < m)
                            or new_i_pos in obstacles
                            or new_j_pos in obstacles
                        ):
                            continue

                        # Check if target positions are occupied by other blocks
                        if (
                            new_i_pos in state_set
                            and new_i_pos != state[i]
                            and new_i_pos != state[j]
                        ) or (
                            new_j_pos in state_set
                            and new_j_pos != state[i]
                            and new_j_pos != state[j]
                        ):
                            continue

                        new_state[i] = new_i_pos
                        new_state[j] = new_j_pos

                        # Verify connectivity
                        if is_state_connected(new_state):
                            pair_moves.append((new_state, [(i, dx, dy), (j, dx, dy)]))

    # 4. Group moves (3-4 blocks)
    if len(state) >= 3:
        for start_idx in range(len(state)):
            for group_size in [3, 4]:
                if group_size > len(state):
                    continue

                visited = {start_idx}
                queue = deque([start_idx])
                group = [start_idx]

                while queue and len(group) < group_size:
                    current = queue.popleft()
                    for neighbor_idx in neighbor_map[current]:
                        if neighbor_idx not in visited:
                            visited.add(neighbor_idx)
                            group.append(neighbor_idx)
                            queue.append(neighbor_idx)
                            if len(group) == group_size:
                                break

                if len(group) == group_size:
                    for dx, dy in DIRECTIONS:
                        new_state = state.copy()
                        valid = True
                        moves = []

                        for idx in group:
                            new_x, new_y = state[idx][0] + dx, state[idx][1] + dy
                            new_pos = (new_x, new_y)

                            # Check bounds and obstacles
                            if (
                                not (0 <= new_x < n and 0 <= new_y < m)
                                or new_pos in obstacles
                            ):
                                valid = False
                                break

                            # Check if position is occupied by a block not in the group
                            if new_pos in state_set:
                                block_at_pos = state.index(new_pos)
                                if block_at_pos not in group:
                                    valid = False
                                    break

                            new_state[idx] = new_pos
                            moves.append((idx, dx, dy))

                        if valid and is_state_connected(new_state):
                            group_moves.append((new_state, moves))

    # Combine successors based on move priority
    if move_priority == "uniform":
        successors = uniform_moves + individual_moves + pair_moves + group_moves
    elif move_priority == "individual":
        successors = individual_moves + uniform_moves + pair_moves + group_moves
    else:
        # Default priority: Mix them to allow for a variety of moves
        successors = uniform_moves + individual_moves + pair_moves + group_moves

    return successors


class AI_Agent:
    def __init__(
        self,
        n: int,
        m: int,
        start_state: List[Tuple[int, int]],
        target_state: List[Tuple[int, int]],
        obstacles: List[Tuple[int, int]] = None,
    ):
        self.n = n
        self.m = m
        self.start_state = sorted(start_state)
        self.target_state = sorted(target_state)
        self.obstacles = set(obstacles or [])

        # Validate that obstacles don't overlap with start or target states
        for obs in self.obstacles:
            if obs in self.start_state or obs in self.target_state:
                raise ValueError(
                    f"Obstacle at {obs} overlaps with start or target state"
                )

        # Clear global caches at the start of each search
        global connectivity_cache
        connectivity_cache = {}

        # Set up memoization structures
        self.visited = set()
        self.g_scores = {}

        # Initialize heuristic data
        self.initial_h = heuristic(self.start_state, self.target_state)

        # Track statistics
        self.nodes_expanded = 0
        self.max_frontier_size = 0

    def estimate_min_moves(self):
        """Estimate minimum number of moves needed."""
        # This is a rough estimate based on the initial heuristic
        return max(1, self.initial_h // min(len(self.start_state), 4))

    def plan(
        self, max_iters=None, depth_lim=None
    ) -> Optional[List[List[Tuple[int, int, int]]]]:
        """
        Find a sequence of moves to transform start_state into target_state, avoiding obstacles.

        Args:
            max_iters: Maximum number of iterations (states to explore)
            depth_lim: Maximum depth (moves) to explore

        Returns:
            List of moves or None if no solution found
        """
        # Use provided limits or fall back to default calculations
        max_iterations = (
            max_iters
            if max_iters is not None
            else min(200000, 20000 * len(self.start_state))
        )
        depth_limit = (
            depth_lim
            if depth_lim is not None
            else max(30, self.estimate_min_moves() * 3)
        )

        start_tuple = state_to_tuple(self.start_state)
        target_tuple = state_to_tuple(self.target_state)

        # Quick check for already solved puzzle
        if start_tuple == target_tuple:
            return []

        self.g_scores = {start_tuple: 0}
        frontier = []

        # Use the optimized heuristic
        h_val = self.initial_h

        # Create a tiebreaker based on blocks in wrong positions
        wrong_pos_count = sum(
            1 for a, b in zip(self.start_state, self.target_state) if a != b
        )

        # Push initial state: (f_value, tiebreaker, state_tuple, path so far)
        heapq.heappush(frontier, (h_val, wrong_pos_count, start_tuple, []))

        # Tracking best partial solution
        best_h = h_val
        best_state = None
        best_path = None

        iterations = 0
        while frontier and iterations < max_iterations:
            iterations += 1

            # Update max frontier size statistic
            self.max_frontier_size = max(self.max_frontier_size, len(frontier))

            # Pop best state from frontier
            f, _, current_tuple, path = heapq.heappop(frontier)

            # Check for goal
            if current_tuple == target_tuple:
                print(f"Solution found after {iterations} iterations")
                self.nodes_expanded = iterations
                return path

            # Skip if already visited
            if current_tuple in self.visited:
                continue

            # Mark as visited
            self.visited.add(current_tuple)
            self.nodes_expanded += 1

            # Check depth limit
            if len(path) >= depth_limit:
                continue

            # Convert to list for successor generation
            current_list = list(current_tuple)

            # Track best partial solution
            current_h = heuristic(current_list, self.target_state)
            if current_h < best_h:
                best_h = current_h
                best_state = current_list
                best_path = path

            # Generate successors using optimized function - PASS OBSTACLES HERE
            successors = get_successors(current_list, self.n, self.m, self.obstacles)

            for succ, moves in successors:
                succ_tuple = state_to_tuple(succ)

                if succ_tuple in self.visited:
                    continue

                new_cost = len(path) + 1
                if new_cost < self.g_scores.get(succ_tuple, float("inf")):
                    self.g_scores[succ_tuple] = new_cost

                    # Compute heuristic and f-value
                    h_val = heuristic(succ, self.target_state)
                    f_val = new_cost + h_val

                    # Compute tiebreaker
                    wrong_pos_count = sum(
                        1 for a, b in zip(succ, self.target_state) if a != b
                    )

                    heapq.heappush(
                        frontier, (f_val, wrong_pos_count, succ_tuple, path + [moves])
                    )

        # Report statistics
        print(f"Search exhausted after {iterations} iterations")
        print(f"Nodes expanded: {self.nodes_expanded}")
        print(f"Max frontier size: {self.max_frontier_size}")

        # Return partial solution if we found one
        if best_path and best_h < self.initial_h:
            print(f"No solution found, but reached a state with heuristic {best_h}")
            return best_path

        return None

    def hierarchical_plan(
        self,
    ) -> Tuple[Optional[List[List[Tuple[int, int, int]]]], List[List[Tuple[int, int]]]]:
        """
        Hierarchical planning approach:
        1. Create an abstract path ignoring connectivity
        2. Generate waypoints along this path
        3. Solve subproblems between waypoints

        Returns:
            Tuple of (complete plan of moves, list of waypoint states)
        """
        print("Starting hierarchical planning...")

        # 1. Generate a distance map for pathfinding around obstacles
        distance_map = self.generate_distance_map()

        # 2. Find abstract path using simplified constraints
        abstract_path = self.plan_abstract_path(distance_map)
        if not abstract_path:
            print("Failed to find abstract path")
            return None, []

        print(f"Abstract path with {len(abstract_path)} states found")

        # 3. Extract waypoints from abstract path
        waypoints = self.extract_waypoints(abstract_path, max_distance=4)
        print(f"Generated {len(waypoints)} waypoints")

        # 4. Solve subproblems between waypoints
        plan = self.solve_subproblems(waypoints)
        return plan, waypoints

    def generate_distance_map(self):
        """Create a distance map that shows distance from obstacles"""
        # Create grid representation for obstacles
        grid = np.zeros((self.n, self.m), dtype=bool)

        # Mark obstacles
        for x, y in self.obstacles:
            if 0 <= x < self.n and 0 <= y < self.m:  # Ensure within bounds
                grid[x, y] = True

        # Calculate distance transform (distance to nearest obstacle)
        distance_map = distance_transform_edt(~grid)
        return distance_map

    def plan_abstract_path(self, distance_map):
        """
        Find a path from start to target with improved robustness.
        """
        print("Planning abstract path...")

        # Calculate centroids for high-level planning
        start_centroid = self.calculate_centroid(self.start_state)
        target_centroid = self.calculate_centroid(self.target_state)

        # Use A* to find path between centroids first (ignoring shape)
        centroid_path = self.find_centroid_path(
            start_centroid, target_centroid, distance_map
        )

        if not centroid_path:
            print("Could not find centroid path - trying direct shape mapping")
            # Fall back to direct shape mapping
            return [self.start_state, self.target_state]

        # Now expand centroid path to full shape path
        shape_path = [self.start_state]

        # Add intermediate shapes based on centroid path
        for i in range(1, len(centroid_path) - 1):
            # Create intermediate shape by translating from previous position
            prev_centroid = centroid_path[i - 1]
            current_centroid = centroid_path[i]

            dx = int(current_centroid[0] - prev_centroid[0])
            dy = int(current_centroid[1] - prev_centroid[1])

            # Translate previous shape
            prev_shape = shape_path[-1]
            new_shape = [(x + dx, y + dy) for x, y in prev_shape]

            # Ensure shape is valid (within bounds and avoids obstacles)
            if all(
                0 <= x < self.n and 0 <= y < self.m and (x, y) not in self.obstacles
                for x, y in new_shape
            ):
                shape_path.append(new_shape)

        # Always add target state at end
        if shape_path[-1] != self.target_state:
            shape_path.append(self.target_state)

        print(f"Abstract path created with {len(shape_path)} waypoints")
        return shape_path

    def find_centroid_path(self, start, target, distance_map):
        """Find path between centroids using A* with obstacle avoidance"""
        # Convert centroids to integer coordinates
        start = (int(start[0]), int(start[1]))
        target = (int(target[0]), int(target[1]))

        # Check if start or target are invalid
        if not (0 <= start[0] < self.n and 0 <= start[1] < self.m):
            start = (
                min(max(0, start[0]), self.n - 1),
                min(max(0, start[1]), self.m - 1),
            )
        if not (0 <= target[0] < self.n and 0 <= target[1] < self.m):
            target = (
                min(max(0, target[0]), self.n - 1),
                min(max(0, target[1]), self.m - 1),
            )

        # Standard A* implementation
        open_set = [(self.manhattan_distance(start, target), 0, start, [start])]
        closed_set = set()

        iterations = 0
        max_iterations = 10000

        while open_set and iterations < max_iterations:
            iterations += 1
            f, g, current, path = heapq.heappop(open_set)

            if current == target:
                return path

            if current in closed_set:
                continue

            closed_set.add(current)

            # Consider all 8 directions
            for dx, dy in DIRECTIONS:
                nx, ny = current[0] + dx, current[1] + dy

                # Check bounds
                if not (0 <= nx < self.n and 0 <= ny < self.m):
                    continue

                # Check obstacle
                if (nx, ny) in self.obstacles:
                    continue

                # Reward paths that stay away from obstacles
                safety_bonus = (
                    distance_map[nx, ny] if 0 <= nx < self.n and 0 <= ny < self.m else 0
                )
                move_cost = 1.0 / (0.1 + safety_bonus)

                # Add to open set
                new_g = g + move_cost
                new_h = self.manhattan_distance((nx, ny), target)
                new_f = new_g + new_h

                heapq.heappush(open_set, (new_f, new_g, (nx, ny), path + [(nx, ny)]))

        # If we can't find a perfect path, try a direct line
        return [start, target]

    def calculate_centroid(self, state):
        """Calculate the centroid of a shape"""
        if not state:
            return (0, 0)
        x_sum = sum(pos[0] for pos in state)
        y_sum = sum(pos[1] for pos in state)
        return (x_sum / len(state), y_sum / len(state))

    def manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two points"""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def abstract_state_hash(self, state):
        """Create a hash for the abstract state (ignoring exact position)"""
        # Normalize to origin
        if not state:
            return ()
        min_x = min(pos[0] for pos in state)
        min_y = min(pos[1] for pos in state)
        normalized = tuple(sorted((x - min_x, y - min_y) for x, y in state))
        return normalized

    def shape_distance(self, shape1, shape2):
        """
        Calculate a distance metric between two shapes
        This is a simplified Hungarian algorithm implementation
        """
        # Calculate minimum bipartite matching between blocks
        total_dist = 0
        for block1 in shape1:
            min_dist = float("inf")
            for block2 in shape2:
                dist = self.manhattan_distance(block1, block2)
                min_dist = min(min_dist, dist)
            total_dist += min_dist
        return total_dist

    def generate_abstract_moves(self, current_state, distance_map):
        """
        Generate possible abstract moves:
        - Uniform translations in 8 directions
        - Simple reconfigurations that maintain rough shape
        """
        moves = []
        shape = current_state["shape"]
        centroid = current_state["centroid"]

        # Uniform translations
        for dx, dy in DIRECTIONS:
            new_shape = [(x + dx, y + dy) for x, y in shape]

            # Check bounds and obstacles
            if all(
                0 <= x < self.n and 0 <= y < self.m and (x, y) not in self.obstacles
                for x, y in new_shape
            ):

                # Calculate move safety based on distance map
                safety = 0
                for x, y in new_shape:
                    if 0 <= x < self.n and 0 <= y < self.m:  # Ensure within bounds
                        safety += distance_map[x, y]
                safety /= len(new_shape)

                # Prefer paths further from obstacles
                move_cost = 1.0 / (0.1 + safety)

                new_centroid = (centroid[0] + dx, centroid[1] + dy)
                moves.append(
                    ({"centroid": new_centroid, "shape": new_shape}, move_cost)
                )

        # Add more complex reconfiguration moves here for shapes that need to change
        # This would depend on your specific problem requirements

        return moves

    def extract_waypoints(self, abstract_path, max_distance=4):
        """
        Extract key waypoints from the abstract path.
        Ensures waypoints are not too far apart.
        """
        if not abstract_path:
            return []

        waypoints = [abstract_path[0]]  # Start with the initial state

        for i in range(1, len(abstract_path)):
            current = abstract_path[i]
            prev_waypoint = waypoints[-1]

            # Add a waypoint if we've moved far enough
            if self.shape_distance(current, prev_waypoint) >= max_distance:
                waypoints.append(current)

        # Always include the final state
        if abstract_path[-1] != waypoints[-1]:
            waypoints.append(abstract_path[-1])

        return waypoints

    def solve_subproblems(self, waypoints):
        """
        Solve a series of smaller problems between waypoints.
        Each subproblem is much easier to solve than the full problem.
        """
        if not waypoints or len(waypoints) < 2:
            return None

        full_plan = []
        current_state = self.start_state

        # Keep track of overall time for timeout
        start_time = time.time()
        max_total_time = 60  # Total allowed time in seconds

        for i, target_waypoint in enumerate(waypoints[1:]):
            print(f"Solving subproblem {i+1}/{len(waypoints)-1}")

            # Check for timeout
            if time.time() - start_time > max_total_time:
                print("Timeout reached in hierarchical planning")
                if full_plan:  # Return partial plan if we have one
                    return full_plan
                return None

            # Create subproblem with smaller iteration limit
            subproblem = AI_Agent(
                self.n, self.m, current_state, target_waypoint, self.obstacles
            )

            # Configure subproblem for quicker search
            max_iterations = min(10000, 5000 * len(current_state))
            depth_limit = max(
                20, 2 * self.shape_distance(current_state, target_waypoint)
            )

            # Override subproblem parameters for faster search
            subproblem.visited = set()
            subproblem.g_scores = {}

            # Solve the subproblem
            sub_plan = subproblem.plan(max_iters=max_iterations, depth_lim=depth_limit)

            if sub_plan:
                print(f"Subproblem {i+1} solved with {len(sub_plan)} moves")
                full_plan.extend(sub_plan)

                # Apply the moves to get the new current state
                current_state = self.apply_moves(current_state, sub_plan)
            else:
                print(f"Failed to solve subproblem {i+1}")
                # Try direct A* for this subproblem with increased limits
                direct_subproblem = AI_Agent(
                    self.n, self.m, current_state, target_waypoint, self.obstacles
                )
                sub_plan = direct_subproblem.plan(max_iters=25000, depth_lim=40)

                if sub_plan:
                    print(f"Direct solution found with {len(sub_plan)} moves")
                    full_plan.extend(sub_plan)
                    current_state = self.apply_moves(current_state, sub_plan)
                else:
                    print("Couldn't solve subproblem, stopping hierarchical planning")
                    break

        # Check if we reached the final target
        if self.shape_distance(current_state, self.target_state) < 1:
            print(f"Hierarchical planning successful: total {len(full_plan)} moves")
            return full_plan
        else:
            # Try to complete the plan with a final subproblem
            final_subproblem = AI_Agent(
                self.n, self.m, current_state, self.target_state, self.obstacles
            )
            final_plan = final_subproblem.plan(max_iters=25000, depth_lim=40)

            if final_plan:
                full_plan.extend(final_plan)
                print(
                    f"Plan completed with final adjustment: total {len(full_plan)} moves"
                )
                return full_plan

        # Return partial plan if we have one
        if full_plan:
            print(f"Returning partial plan with {len(full_plan)} moves")
            return full_plan

        return None

    def apply_moves(self, state, plan):
        """Apply a series of moves to a state to get the new state"""
        result = deepcopy(state)

        for move_set in plan:
            # Create a mapping of block index to new position
            moves_dict = {}
            for block_idx, dx, dy in move_set:
                if block_idx < len(result):
                    x, y = result[block_idx]
                    moves_dict[block_idx] = (x + dx, y + dy)

            # Apply the moves
            for idx, new_pos in moves_dict.items():
                if idx < len(result):
                    result[idx] = new_pos

        return result

    def plan_parallel(self, num_processes=None):
        """
        Use multiple processes to search for a solution in parallel.
        Each process uses different search parameters for better coverage.

        Args:
            num_processes: Number of processes to use (defaults to CPU count)

        Returns:
            The best plan found by any of the processes
        """
        if num_processes is None:
            num_processes = mp.cpu_count()

        print(f"Starting parallel planning with {num_processes} processes...")
        start_time = time.time()

        # Create different search configurations
        configs = []

        # 1. Different heuristic weights
        for weight in [1.0, 1.2, 1.5, 2.0]:
            configs.append(
                {
                    "weight": weight,
                    "strategy": "astar",
                    "max_iters": min(100000, 15000 * len(self.start_state)),
                    "depth_lim": max(25, self.estimate_min_moves() * 2.5),
                }
            )

        # 2. Different search strategies
        configs.append(
            {
                "weight": 1.0,
                "strategy": "beam",
                "beam_width": 1000,
                "max_iters": min(120000, 18000 * len(self.start_state)),
            }
        )

        configs.append(
            {
                "weight": 1.0,
                "strategy": "greedy",
                "max_iters": min(150000, 20000 * len(self.start_state)),
            }
        )

        # 3. Different movement priorities
        configs.append({"weight": 1.0, "strategy": "astar", "move_priority": "uniform"})

        configs.append(
            {"weight": 1.0, "strategy": "astar", "move_priority": "individual"}
        )

        # Adjust config count to match process count
        if len(configs) > num_processes:
            configs = configs[:num_processes]
        elif len(configs) < num_processes:
            # Duplicate some configs to use all available processes
            configs = configs * (num_processes // len(configs) + 1)
            configs = configs[:num_processes]

        # Prepare inputs for each worker process
        inputs = []
        for i, config in enumerate(configs):
            inputs.append(
                (
                    self.n,
                    self.m,
                    self.start_state,
                    self.target_state,
                    list(self.obstacles),
                    config,
                    i,
                )
            )

        # Run processes in parallel
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(self._search_worker_wrapper, inputs)

        # Process results
        best_plan = None
        best_plan_length = float("inf")

        for process_id, plan, stats in results:
            if plan is not None:
                if len(plan) < best_plan_length:
                    best_plan = plan
                    best_plan_length = len(plan)
                print(f"Process {process_id} found solution with {len(plan)} moves")
            else:
                print(f"Process {process_id} failed to find a solution")

        end_time = time.time()
        if best_plan:
            print(
                f"Parallel planning found solution with {best_plan_length} moves in {end_time - start_time:.2f}s"
            )
            self.nodes_expanded = sum(
                stats.get("nodes_expanded", 0) for _, _, stats in results
            )
        else:
            print(
                f"Parallel planning failed to find a solution in {end_time - start_time:.2f}s"
            )

        return best_plan

    @staticmethod
    def _search_worker_wrapper(args):
        """Static wrapper method to allow multiprocessing with instance methods"""
        n, m, start, target, obstacles, config, process_id = args
        try:
            # Create a new instance for this process
            agent = AI_Agent(n, m, start, target, obstacles)

            # Apply configuration
            weight = config.get("weight", 1.0)
            strategy = config.get("strategy", "astar")
            max_iters = config.get("max_iters", None)
            depth_lim = config.get("depth_lim", None)
            beam_width = config.get("beam_width", None)
            move_priority = config.get("move_priority", None)

            # Run search with this configuration
            plan = agent._search_with_config(
                weight, strategy, max_iters, depth_lim, beam_width, move_priority
            )

            # Return process ID, plan, and statistics
            stats = {
                "nodes_expanded": agent.nodes_expanded,
                "max_frontier": agent.max_frontier_size,
            }

            return process_id, plan, stats

        except Exception as e:
            print(f"Error in process {process_id}: {e}")
            return process_id, None, {}

    def _search_with_config(
        self,
        weight=1.0,
        strategy="astar",
        max_iters=None,
        depth_lim=None,
        beam_width=None,
        move_priority=None,
    ):
        """
        Perform search with specific configuration parameters.

        Args:
            weight: Weight for heuristic (higher = more greedy)
            strategy: "astar", "beam", or "greedy"
            max_iters: Maximum iterations
            depth_lim: Maximum depth
            beam_width: Maximum frontier size for beam search
            move_priority: "uniform", "individual", or None for balanced

        Returns:
            Plan if found, None otherwise
        """
        # Reset state for this search
        self.visited = set()
        self.g_scores = {}
        self.nodes_expanded = 0
        self.max_frontier_size = 0

        # Use provided limits or fall back to defaults
        max_iterations = (
            max_iters
            if max_iters is not None
            else min(200000, 20000 * len(self.start_state))
        )
        depth_limit = (
            depth_lim
            if depth_lim is not None
            else max(30, self.estimate_min_moves() * 3)
        )

        start_tuple = state_to_tuple(self.start_state)
        target_tuple = state_to_tuple(self.target_state)

        # Quick check for already solved puzzle
        if start_tuple == target_tuple:
            return []

        self.g_scores = {start_tuple: 0}
        frontier = []

        # Use the optimized heuristic
        h_val = self.initial_h

        # Create a tiebreaker based on blocks in wrong positions
        wrong_pos_count = sum(
            1 for a, b in zip(self.start_state, self.target_state) if a != b
        )

        # Push initial state: (f_value, tiebreaker, state_tuple, path so far)
        if strategy == "greedy":
            # Greedy best-first search only uses heuristic
            heapq.heappush(frontier, (h_val, wrong_pos_count, start_tuple, []))
        else:
            # A* uses g + h
            heapq.heappush(frontier, (h_val, wrong_pos_count, start_tuple, []))

        # Tracking best partial solution
        best_h = h_val
        best_state = None
        best_path = None

        iterations = 0
        while frontier and iterations < max_iterations:
            iterations += 1

            # Update max frontier size statistic
            self.max_frontier_size = max(self.max_frontier_size, len(frontier))

            # Apply beam width limit if beam search
            if strategy == "beam" and beam_width and len(frontier) > beam_width:
                frontier = heapq.nsmallest(beam_width, frontier)
                heapq.heapify(frontier)

            # Pop best state from frontier
            f, _, current_tuple, path = heapq.heappop(frontier)

            # Check for goal
            if current_tuple == target_tuple:
                self.nodes_expanded = iterations
                return path

            # Skip if already visited
            if current_tuple in self.visited:
                continue

            # Mark as visited
            self.visited.add(current_tuple)
            self.nodes_expanded += 1

            # Check depth limit
            if len(path) >= depth_limit:
                continue

            # Convert to list for successor generation
            current_list = list(current_tuple)

            # Track best partial solution
            current_h = heuristic(current_list, self.target_state)
            if current_h < best_h:
                best_h = current_h
                best_state = current_list
                best_path = path

            # Generate successors with move priority
            successors = get_successors(
                current_list, self.n, self.m, self.obstacles, move_priority
            )

            for succ, moves in successors:
                succ_tuple = state_to_tuple(succ)

                if succ_tuple in self.visited:
                    continue

                new_cost = len(path) + 1
                if new_cost < self.g_scores.get(succ_tuple, float("inf")):
                    self.g_scores[succ_tuple] = new_cost

                    # Compute heuristic and f-value
                    h_val = heuristic(succ, self.target_state)

                    # Apply heuristic weight
                    if strategy == "greedy":
                        f_val = h_val  # Greedy best-first search
                    else:
                        f_val = new_cost + (h_val * weight)  # Weighted A*

                    # Compute tiebreaker
                    wrong_pos_count = sum(
                        1 for a, b in zip(succ, self.target_state) if a != b
                    )

                    heapq.heappush(
                        frontier, (f_val, wrong_pos_count, succ_tuple, path + [moves])
                    )

        # Return partial solution if we found one
        if best_path and best_h < self.initial_h:
            return best_path

        return None

    def hierarchical_plan_parallel(self, num_processes=None):
        """
        Run hierarchical planning with parallel subproblem solving.

        Args:
            num_processes: Number of processes to use (defaults to CPU count)

        Returns:
            Tuple of (plan, waypoints)
        """
        print("Starting parallel hierarchical planning...")

        # 1. Generate a distance map for pathfinding around obstacles
        distance_map = self.generate_distance_map()

        # 2. Find abstract path using simplified constraints
        abstract_path = self.plan_abstract_path(distance_map)
        if not abstract_path:
            print("Failed to find abstract path")
            return None, []

        print(f"Abstract path with {len(abstract_path)} states found")

        # 3. Extract waypoints from abstract path
        waypoints = self.extract_waypoints(abstract_path, max_distance=4)
        print(f"Generated {len(waypoints)} waypoints")

        # Setup process pool
        if num_processes is None:
            num_processes = min(mp.cpu_count(), len(waypoints) - 1)
            if num_processes <= 0:
                num_processes = 1

        if len(waypoints) < 2:
            return None, []

        # 4. Solve subproblems in parallel
        subproblems = []
        current_state = self.start_state

        # Create subproblems between each pair of consecutive waypoints
        for i in range(1, len(waypoints)):
            subproblems.append(
                (
                    self.n,
                    self.m,
                    (
                        current_state if i == 1 else waypoints[i - 1]
                    ),  # Start from previous waypoint after first subproblem
                    waypoints[i],  # Target is current waypoint
                    list(self.obstacles),
                    {
                        "max_iters": min(10000, 5000 * len(self.start_state)),
                        "depth_lim": max(
                            20, 2 * self.shape_distance(current_state, waypoints[i])
                        ),
                    },
                    i - 1,  # subproblem ID
                )
            )

        # Solve subproblems in parallel
        with mp.Pool(processes=num_processes) as pool:
            results = pool.map(self._solve_subproblem, subproblems)

        # Process results and combine plans
        full_plan = []
        plans_succeeded = True

        for i, (subproblem_id, sub_plan, success) in enumerate(results):
            if success and sub_plan:
                print(f"Subproblem {subproblem_id} solved with {len(sub_plan)} moves")
                full_plan.extend(sub_plan)
            else:
                print(f"Subproblem {subproblem_id} failed")
                plans_succeeded = False
                break

        if plans_succeeded:
            print(f"Parallel hierarchical planning successful: {len(full_plan)} moves")
            return full_plan, waypoints
        else:
            print("Parallel hierarchical planning failed")
            # Try sequential fallback
            return self.solve_subproblems(waypoints), waypoints

    @staticmethod
    def _solve_subproblem(args):
        """Static method to solve a subproblem in a worker process"""
        n, m, start, target, obstacles, config, subproblem_id = args

        try:
            # Create agent for this subproblem
            agent = AI_Agent(n, m, start, target, obstacles)

            # Get config parameters
            max_iters = config.get("max_iters")
            depth_lim = config.get("depth_lim")

            # Try to solve subproblem
            sub_plan = agent.plan(max_iters=max_iters, depth_lim=depth_lim)

            if sub_plan:
                return subproblem_id, sub_plan, True
            else:
                return subproblem_id, None, False

        except Exception as e:
            print(f"Error in subproblem {subproblem_id}: {e}")
            return subproblem_id, None, False
