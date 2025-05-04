import heapq
from collections import deque
from typing import List, Tuple, Dict, Set

# Import your Cython optimized functions
from puzzle_cython import apply_moves, manhattan_distance_int as manhattan_distance


class TunnelNavigator:
    """Handles specialized tunnel navigation for programmable matter."""

    def __init__(self, n, m, obstacles, topology_analyzer):
        self.n = n
        self.m = m
        self.obstacles = set(obstacles)
        self.topology = topology_analyzer

    def navigate_tunnel(self, tunnel_idx, start_state, target_chamber):
        """
        Navigate through a tunnel by moving one block at a time.

        Args:
            tunnel_idx: Index of tunnel in topology analyzer's tunnel list
            start_state: Current state of blocks
            target_chamber: Index of chamber on the target side of tunnel

        Returns:
            List of moves to navigate all blocks through the tunnel
        """
        if tunnel_idx >= len(self.topology.tunnels):
            return None

        tunnel = self.topology.tunnels[tunnel_idx]
        print(
            f"Navigating through tunnel: {len(tunnel['path'])} cells, width: {tunnel['width']}"
        )

        # Extract tunnel data
        tunnel_path = tunnel["path"]
        entrance = tunnel["entrance"]
        exit = tunnel["exit"]

        # Make sure exit is on target side
        chamber_a, chamber_b = tunnel["chambers"]
        if chamber_b == target_chamber:
            # Tunnel is oriented correctly
            pass
        elif chamber_a == target_chamber:
            # Swap entrance and exit
            entrance, exit = exit, entrance
        else:
            print("Tunnel doesn't connect to target chamber")
            return None

        print(f"Tunnel entrance at {entrance}, exit at {exit}")

        # Start navigation process
        current_blocks = list(start_state)
        full_plan = []

        # Split blocks into blocks to move and blocks that have crossed
        remaining_blocks = current_blocks.copy()
        crossed_blocks = []

        # Process blocks one by one through the tunnel
        while remaining_blocks:
            # Find block closest to entrance
            closest_idx = self._find_closest_block(remaining_blocks, entrance)

            # Move block to tunnel entrance
            print(f"Moving block {closest_idx} to tunnel entrance...")
            entrance_plan = self._plan_to_tunnel_entrance(
                remaining_blocks, closest_idx, entrance, crossed_blocks
            )

            if not entrance_plan:
                print("Failed to move block to tunnel entrance")
                return None

            # Apply the plan to all blocks
            full_plan.extend(entrance_plan)
            remaining_blocks = apply_moves(remaining_blocks, entrance_plan)

            # Now get updated position of the block we're moving
            moving_block = remaining_blocks[closest_idx]

            # Create moves to navigate through tunnel
            print("Moving block through tunnel...")
            tunnel_plan = self._navigate_through_tunnel(moving_block, tunnel_path, exit)

            if not tunnel_plan:
                print("Failed to navigate through tunnel")
                return None

            # Apply the tunnel plan
            full_plan.extend(tunnel_plan)

            # Update block positions
            block_after_tunnel = apply_moves([moving_block], tunnel_plan)[0]

            # Move block to crossed list
            crossed_blocks.append(block_after_tunnel)
            remaining_blocks.pop(closest_idx)

            print(
                f"Successfully moved block through tunnel. {len(remaining_blocks)} blocks remaining"
            )

        # All blocks have crossed, return the plan
        return full_plan

    def _find_closest_block(self, blocks, target_pos):
        """Find the index of the block closest to a position."""
        min_dist = float("inf")
        closest_idx = 0

        for i, pos in enumerate(blocks):
            dist = manhattan_distance(pos, target_pos)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        return closest_idx

    def _plan_to_tunnel_entrance(self, blocks, block_idx, entrance, crossed_blocks):
        """Plan path for a block to reach tunnel entrance."""
        # Target position near the entrance
        staging_pos = self._find_staging_position(entrance, blocks, crossed_blocks)

        # Create a set of positions to avoid
        obstacles = self.obstacles.copy()
        for i, block in enumerate(blocks):
            if i != block_idx:  # Don't avoid the block we're moving
                obstacles.add(block)
        for block in crossed_blocks:
            obstacles.add(block)

        # A* search for path
        start_pos = blocks[block_idx]
        target_pos = staging_pos

        return self._astar_for_block(start_pos, target_pos, obstacles)

    def _find_staging_position(self, entrance, blocks, crossed_blocks):
        """Find a position near tunnel entrance that's accessible."""
        x, y = entrance
        occupied = set(blocks).union(crossed_blocks).union(self.obstacles)

        # Try positions at increasing distances from entrance
        for dist in range(0, 5):
            for dx in range(-dist, dist + 1):
                for dy in range(-dist, dist + 1):
                    # Only consider positions at exactly distance 'dist'
                    if abs(dx) + abs(dy) == dist:
                        pos = (x + dx, y + dy)
                        if (
                            0 <= pos[0] < self.n
                            and 0 <= pos[1] < self.m
                            and pos not in occupied
                        ):
                            # Verify we can reach the entrance from here
                            if self._check_entrance_accessibility(
                                pos, entrance, occupied
                            ):
                                return pos

        # Fallback - return the entrance itself
        return entrance

    def _check_entrance_accessibility(self, pos, entrance, obstacles):
        """Check if we can reach the tunnel entrance from this position."""
        # Simple BFS to check reachability
        visited = set([pos])
        queue = deque([pos])

        while queue:
            x, y = queue.popleft()

            if (x, y) == entrance:
                return True

            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                next_pos = (nx, ny)

                if (
                    0 <= nx < self.n
                    and 0 <= ny < self.m
                    and next_pos not in obstacles
                    and next_pos not in visited
                ):
                    visited.add(next_pos)
                    queue.append(next_pos)

        return False

    def _navigate_through_tunnel(self, block_pos, tunnel_path, exit_pos):
        """Create a sequence of moves to navigate through the tunnel."""
        # Find path through tunnel
        path = self._find_path_through_tunnel(block_pos, tunnel_path, exit_pos)

        if not path:
            return None

        # Convert path to moves
        moves = []
        current_pos = block_pos

        for next_pos in path[1:]:  # Skip the first position (current position)
            dx = next_pos[0] - current_pos[0]
            dy = next_pos[1] - current_pos[1]

            # Create move (always for block 0 since we're processing one block at a time)
            moves.append([(0, dx, dy)])
            current_pos = next_pos

        return moves

    def _find_path_through_tunnel(self, start_pos, tunnel_path, target_pos):
        """Find a valid path through the tunnel from start to target."""
        # Use A* search through the tunnel
        tunnel_set = set(tunnel_path)

        # Add a buffer zone around the tunnel
        extended_tunnel = tunnel_set.copy()
        for x, y in tunnel_path:
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < self.n
                    and 0 <= ny < self.m
                    and (nx, ny) not in self.obstacles
                ):
                    extended_tunnel.add((nx, ny))

        # Run A* search
        open_set = [
            (manhattan_distance(start_pos, target_pos), 0, start_pos, [start_pos])
        ]
        closed_set = set()

        while open_set:
            _, cost, current, path = heapq.heappop(open_set)

            if current == target_pos:
                return path

            if current in closed_set:
                continue

            closed_set.add(current)

            # Explore neighbors
            x, y = current
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                next_pos = (nx, ny)

                if (
                    0 <= nx < self.n
                    and 0 <= ny < self.m
                    and next_pos not in self.obstacles
                    and next_pos not in closed_set
                    and (next_pos in extended_tunnel or next_pos == target_pos)
                ):

                    new_cost = cost + 1
                    new_h = manhattan_distance(next_pos, target_pos)
                    new_f = new_cost + new_h

                    heapq.heappush(
                        open_set, (new_f, new_cost, next_pos, path + [next_pos])
                    )

        return None

    def _astar_for_block(self, start_pos, target_pos, obstacles):
        """A* search for a single block, returning moves for that block."""
        open_set = [(manhattan_distance(start_pos, target_pos), 0, start_pos, [])]
        closed_set = set()

        while open_set:
            _, cost, current, path = heapq.heappop(open_set)

            if current == target_pos:
                return path

            if current in closed_set:
                continue

            closed_set.add(current)

            # Explore neighbors
            x, y = current
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                next_pos = (nx, ny)

                if (
                    0 <= nx < self.n
                    and 0 <= ny < self.m
                    and next_pos not in obstacles
                    and next_pos not in closed_set
                ):

                    new_cost = cost + 1
                    new_h = manhattan_distance(next_pos, target_pos)
                    new_f = new_cost + new_h

                    # Create move for the block (always index 0 since we're working with one block)
                    new_move = [(0, dx, dy)]

                    heapq.heappush(
                        open_set, (new_f, new_cost, next_pos, path + [new_move])
                    )

        return None
