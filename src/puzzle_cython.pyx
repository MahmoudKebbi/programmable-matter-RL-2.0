# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import numpy as np
cimport numpy as np
from collections import deque
import heapq
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import distance_transform_edt

# Initialize NumPy C API
np.import_array()

# Global direction constants
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
cdef int[:, :] C_DIRECTIONS = np.array(DIRECTIONS, dtype=np.int32)

# Cache for connectivity checks
connectivity_cache = {}

cpdef tuple state_to_tuple(list state):
    """Convert state to a hashable tuple for caching."""
    return tuple(sorted(state))

cpdef bint is_state_connected(list state):
    """Check if all blocks in the state are connected."""
    if not state:
        return True
        
    # Check cache first
    cdef tuple state_tuple = state_to_tuple(state)
    if state_tuple in connectivity_cache:
        return connectivity_cache[state_tuple]
    
    # Python containers - no cdef
    state_set = set(state)
    visited = set()
    queue = deque([state[0]])
    visited.add(state[0])
    
    cdef tuple pos, neighbor
    cdef int x, y, dx, dy, i
    
    while queue:
        pos = queue.popleft()
        x, y = pos
        
        for i in range(8):
            dx = C_DIRECTIONS[i, 0]
            dy = C_DIRECTIONS[i, 1]
            neighbor = (x + dx, y + dy)
            
            if neighbor in state_set and neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    cdef bint result = len(visited) == len(state)
    connectivity_cache[state_tuple] = result
    return result

cpdef bint is_still_connected_after_move(list state, int block_idx, tuple new_pos):
    """Check if state remains connected after moving a single block."""
    if len(state) <= 1:
        return True
        
    # Python containers - no cdef
    state_set = set(state)
    cdef tuple orig_pos = state[block_idx]
    
    # Check if block has any connections
    cdef bint has_connections = False
    cdef int x = orig_pos[0]
    cdef int y = orig_pos[1]
    cdef int dx, dy, i
    cdef tuple neighbor
    
    for i in range(8):
        dx = C_DIRECTIONS[i, 0]
        dy = C_DIRECTIONS[i, 1]
        neighbor = (x + dx, y + dy)
        if neighbor in state_set:
            has_connections = True
            break
            
    if not has_connections and len(state) > 1:
        return False
        
    # Check if new position will be connected
    x = new_pos[0]
    y = new_pos[1]
    cdef bint will_be_connected = False
    
    for i in range(8):
        dx = C_DIRECTIONS[i, 0]
        dy = C_DIRECTIONS[i, 1]
        neighbor = (x + dx, y + dy)
        if neighbor in state_set and neighbor != orig_pos:
            will_be_connected = True
            break
            
    if not will_be_connected and len(state) > 1:
        return False
        
    # Do a full connectivity check
    new_state = state.copy()
    new_state[block_idx] = new_pos
    return is_state_connected(new_state)

cpdef dict compute_neighbor_map(list state):
    """Create a map of each block's neighboring blocks."""
    neighbor_map = {}
    state_set = set(state)
    cdef int i, j, dx, dy
    cdef tuple pos, neighbor
    
    for i in range(len(state)):
        neighbor_map[i] = []
        pos = state[i]
        
        for j in range(8):
            dx = C_DIRECTIONS[j, 0]
            dy = C_DIRECTIONS[j, 1]
            neighbor = (pos[0] + dx, pos[1] + dy)
            
            if neighbor in state_set:
                neighbor_idx = state.index(neighbor)
                if neighbor_idx != i:  # Don't add self as neighbor
                    neighbor_map[i].append(neighbor_idx)
                    
    return neighbor_map

cpdef set find_articulation_points(list state):
    """Find blocks that, if removed, would disconnect the structure."""
    if len(state) <= 2:
        return set()
    
    articulation_points = set()
    cdef int i
    
    for i in range(len(state)):
        # Skip if this is a leaf block (has 0 or 1 neighbors)
        neighbors = compute_neighbor_map(state).get(i, [])
        if len(neighbors) <= 1:
            continue
            
        # Remove this block temporarily
        reduced_state = state.copy()
        reduced_state.pop(i)
        
        # If remaining blocks aren't connected, this is an articulation point
        if not is_state_connected(reduced_state):
            articulation_points.add(i)
            
    return articulation_points

cpdef int heuristic(list state, list target):
    """Optimized heuristic function using the Hungarian algorithm."""
    cdef np.ndarray[double, ndim=2] cost_matrix = np.zeros((len(state), len(target)))
    
    cdef int i, j
    for i in range(len(state)):
        for j in range(len(target)):
            x1, y1 = state[i]
            x2, y2 = target[j]
            cost_matrix[i, j] = abs(x1 - x2) + abs(y1 - y2)
    
    # Use Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    total_cost = int(cost_matrix[row_ind, col_ind].sum())
    
    return total_cost

cpdef double shape_distance(list shape1, list shape2):
    """Calculate distance between two shapes (bipartite matching)."""
    cdef int i, j
    cdef double total_dist = 0
    cdef double min_dist, dist
    
    for i in range(len(shape1)):
        min_dist = float('inf')
        for j in range(len(shape2)):
            dist = abs(shape1[i][0] - shape2[j][0]) + abs(shape1[i][1] - shape2[j][1])
            if dist < min_dist:
                min_dist = dist
        total_dist += min_dist
    
    return total_dist

cpdef tuple calculate_centroid(list state):
    """Calculate the centroid of a shape."""
    if not state:
        return (0, 0)
    
    cdef double x_sum = 0
    cdef double y_sum = 0
    cdef int i
    
    for i in range(len(state)):
        x_sum += state[i][0]
        y_sum += state[i][1]
    
    return (x_sum / len(state), y_sum / len(state))

cpdef list get_successors(list state, int n, int m, set obstacles=None, str move_priority=None):
    """Generate successors with optimizations."""
    if obstacles is None:
        obstacles = set()
        
    successors = []
    state_set = set(state)
    
    # Pre-compute neighbor information
    neighbor_map = compute_neighbor_map(state)
    
    # Storage for different types of moves
    uniform_moves = []
    individual_moves = []
    pair_moves = []
    group_moves = []
    
    # 1. Uniform moves (all blocks move the same direction)
    cdef int i, j, dx, dy
    cdef tuple pos, new_pos
    cdef bint valid
    
    for i in range(8):  # 8 directions
        dx = C_DIRECTIONS[i, 0]
        dy = C_DIRECTIONS[i, 1]
        new_state = []
        valid = True
        
        for j in range(len(state)):
            pos = state[j]
            new_x, new_y = pos[0] + dx, pos[1] + dy
            new_pos = (new_x, new_y)
            
            # Check bounds and obstacles
            if not (0 <= new_x < n and 0 <= new_y < m) or new_pos in obstacles:
                valid = False
                break
                
            new_state.append(new_pos)
            
        if valid:
            # Create move list for uniform move
            moves = [(j, dx, dy) for j in range(len(state))]
            uniform_moves.append((new_state, moves))
    
    # 2. Single block moves
    # Find articulation points
    articulation_points = find_articulation_points(state)
    
    # Movable blocks are those that aren't articulation points
    movable_blocks = []
    for i in range(len(state)):
        if i not in articulation_points:
            movable_blocks.append(i)
            
    # If all blocks are articulation points, use leaves instead
    if not movable_blocks:
        for i in range(len(state)):
            if len(neighbor_map.get(i, [])) <= 1:
                movable_blocks.append(i)
    
    for i in movable_blocks:
        pos = state[i]
        for j in range(8):
            dx = C_DIRECTIONS[j, 0]
            dy = C_DIRECTIONS[j, 1]
            new_x, new_y = pos[0] + dx, pos[1] + dy
            new_pos = (new_x, new_y)
            
            # Check bounds, collisions, and obstacles
            if not (0 <= new_x < n and 0 <= new_y < m) or new_pos in state_set or new_pos in obstacles:
                continue
                
            # Check connectivity
            if is_still_connected_after_move(state, i, new_pos):
                new_state = state.copy()
                new_state[i] = new_pos
                individual_moves.append((new_state, [(i, dx, dy)]))
    
    # 3. Pair moves (move two adjacent blocks together)
    if len(state) >= 2:
        for i in range(len(state)):
            for j in neighbor_map.get(i, []):
                if i < j:  # Process each pair only once
                    for d in range(8):
                        dx = C_DIRECTIONS[d, 0]
                        dy = C_DIRECTIONS[d, 1]
                        
                        # New positions for both blocks
                        new_i_pos = (state[i][0] + dx, state[i][1] + dy)
                        new_j_pos = (state[j][0] + dx, state[j][1] + dy)
                        
                        # Check bounds and obstacles
                        if (not (0 <= new_i_pos[0] < n and 0 <= new_i_pos[1] < m) or 
                            not (0 <= new_j_pos[0] < n and 0 <= new_j_pos[1] < m) or
                            new_i_pos in obstacles or new_j_pos in obstacles):
                            continue
                        
                        # Check collisions with other blocks
                        if ((new_i_pos in state_set and new_i_pos != state[i] and new_i_pos != state[j]) or
                            (new_j_pos in state_set and new_j_pos != state[i] and new_j_pos != state[j])):
                            continue
                            
                        # Create new state
                        new_state = state.copy()
                        new_state[i] = new_i_pos
                        new_state[j] = new_j_pos
                        
                        # Check connectivity
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
                    for neighbor_idx in neighbor_map.get(current, []):
                        if neighbor_idx not in visited:
                            visited.add(neighbor_idx)
                            group.append(neighbor_idx)
                            queue.append(neighbor_idx)
                            if len(group) == group_size:
                                break
                                
                if len(group) == group_size:
                    for d in range(8):
                        dx = C_DIRECTIONS[d, 0]
                        dy = C_DIRECTIONS[d, 1]
                        new_state = state.copy()
                        valid = True
                        moves = []
                        
                        for idx in group:
                            new_x, new_y = state[idx][0] + dx, state[idx][1] + dy
                            new_pos = (new_x, new_y)
                            
                            # Check bounds and obstacles
                            if not (0 <= new_x < n and 0 <= new_y < m) or new_pos in obstacles:
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
    
    # Combine moves based on priority
    if move_priority == "uniform":
        successors = uniform_moves + individual_moves + pair_moves + group_moves
    elif move_priority == "individual":
        successors = individual_moves + uniform_moves + pair_moves + group_moves
    else:
        # Default priority
        successors = uniform_moves + individual_moves + pair_moves + group_moves
        
    return successors

cpdef np.ndarray[double, ndim=2] generate_distance_map(int n, int m, set obstacles):
    """Create a distance map that shows distance from obstacles."""
    cdef np.ndarray[np.int32_t, ndim=2] grid = np.zeros((n, m), dtype=np.int32)
    
    # Mark obstacles
    for x, y in obstacles:
        if 0 <= x < n and 0 <= y < m:  # Ensure within bounds
            grid[x, y] = 1
            
    # Calculate distance transform
    cdef np.ndarray[double, ndim=2] distance_map = distance_transform_edt(~grid.astype(bool))
    return distance_map

cpdef int manhattan_distance_int(tuple pos1, tuple pos2):
    """Calculate Manhattan distance between two points."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

cpdef list apply_moves(list state, list plan):
    """Apply a series of moves to a state to get the new state."""
    result = state.copy()
    
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

cpdef void clear_cache():
    """Clear the connectivity cache."""
    global connectivity_cache
    connectivity_cache = {}