import heapq
import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import List, Tuple, Optional, Dict, Set
from collections import deque

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

# Connectivity check optimizations
connectivity_cache = {}


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


def state_to_tuple(state: List[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    """Convert state to a hashable tuple."""
    return tuple(sorted(state))


# Optimized heuristic functions
def heuristic(state: List[Tuple[int, int]], target: List[Tuple[int, int]]) -> int:
    """Improved heuristic function using greedy matching."""
    cost_matrix = np.zeros((len(state), len(target)))
    for i, (x1, y1) in enumerate(state):
        for j, (x2, y2) in enumerate(target):
            cost_matrix[i, j] = abs(x1 - x2) + abs(y1 - y2)

    # Use Hungarian algorithm to find optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_cost = cost_matrix[row_ind, col_ind].sum()

    return total_cost


# Fast successor generation
def get_successors(state: List[Tuple[int, int]], n: int, m: int):
    """Generate successors with optimizations."""
    successors = []
    state_set = set(state)

    # Pre-compute neighbor information
    neighbor_map = compute_neighbor_map(state)

    # 1. Uniform moves (all blocks move the same direction)
    for dx, dy in DIRECTIONS:
        new_state = [(x + dx, y + dy) for x, y in state]
        if all(0 <= x < n and 0 <= y < m for x, y in new_state):
            successors.append((new_state, [(i, dx, dy) for i in range(len(state))]))

    # Find leaf blocks (blocks with only one neighbor)
    leaves = [i for i, neighbors in neighbor_map.items() if len(neighbors) <= 1]

    # Find articulation points (blocks that cannot be moved alone)
    articulation_points = find_articulation_points(state)

    # 2. Single block moves
    movable_blocks = [i for i in range(len(state)) if i not in articulation_points]
    if (
        not movable_blocks and leaves
    ):  # If all blocks are articulation points, try leaf nodes
        movable_blocks = leaves

    for i in movable_blocks:
        x, y = state[i]
        for dx, dy in DIRECTIONS:
            new_x, new_y = x + dx, y + dy
            new_pos = (new_x, new_y)

            # Quick bounds and collision check
            if not (0 <= new_x < n and 0 <= new_y < m) or new_pos in state_set:
                continue

            # Quick connectivity check
            if is_still_connected_after_move(state, i, new_pos):
                new_state = state.copy()
                new_state[i] = new_pos
                successors.append((new_state, [(i, dx, dy)]))

    # 3. Multi-block moves (for pairs of adjacent blocks)
    if len(state) >= 2:
        # Find adjacent block pairs
        for i, neighbors in neighbor_map.items():
            for j in neighbors:
                if i < j:  # Process each pair only once
                    for dx, dy in DIRECTIONS:
                        new_state = state.copy()

                        # Move both blocks
                        new_i_pos = (state[i][0] + dx, state[i][1] + dy)
                        new_j_pos = (state[j][0] + dx, state[j][1] + dy)

                        # Check bounds and collisions
                        if not (
                            0 <= new_i_pos[0] < n and 0 <= new_i_pos[1] < m
                        ) or not (0 <= new_j_pos[0] < n and 0 <= new_j_pos[1] < m):
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
                            successors.append((new_state, [(i, dx, dy), (j, dx, dy)]))

    # 4. For 3-4 block groups, find connected groups and try moving them
    if len(state) >= 3:
        # Use BFS to find connected groups of 3-4 blocks
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

                            # Check bounds
                            if not (0 <= new_x < n and 0 <= new_y < m):
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
                            successors.append((new_state, moves))

    return successors


class AI_Agent:
    def __init__(
        self,
        n: int,
        m: int,
        start_state: List[Tuple[int, int]],
        target_state: List[Tuple[int, int]],
    ):
        self.n = n
        self.m = m
        self.start_state = sorted(start_state)
        self.target_state = sorted(target_state)

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

    def plan(self) -> Optional[List[List[Tuple[int, int, int]]]]:
        """Find a sequence of moves to transform start_state into target_state."""
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

        # Adaptive iteration limit based on problem complexity
        max_iterations = min(200000, 20000 * len(self.start_state))
        depth_limit = max(30, self.estimate_min_moves() * 3)

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

            # Generate successors using optimized function
            successors = get_successors(current_list, self.n, self.m)

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

        # If no solution found but we found a better state than the start
        if best_state and best_h < self.initial_h:
            print(f"No solution found, but reached a state with heuristic {best_h}")
            # You could return the partial solution here
            # return best_path

        return None


# Example usage
if __name__ == "__main__":
    # Example puzzle: move a 3-block L shape to target position
    n, m = 5, 5
    start_state = [(0, 0), (1, 0), (1, 1)]
    target_state = [(3, 3), (3, 4), (4, 3)]

    agent = AI_Agent(n, m, start_state, target_state)
    solution = agent.plan()

    if solution:
        print(f"Solution found with {len(solution)} moves:")
        for i, move_set in enumerate(solution):
            print(f"Move {i+1}: {move_set}")
    else:
        print("No solution found")
