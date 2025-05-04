import numpy as np
from collections import deque
import heapq

# Import the Cython functions you already have
from puzzle_cython import (
    manhattan_distance_int as manhattan_distance,
    generate_distance_map,
)


class TopologyAnalyzer:
    """Analyzes the topological structure of the environment to find paths through tunnels."""

    def __init__(self, n, m, obstacles):
        self.n = n
        self.m = m
        self.obstacles = set(obstacles)
        self.grid = self._create_grid()
        self.chambers = []
        self.tunnels = []
        self.narrow_passages = []

    def _create_grid(self):
        """Create binary grid representation with obstacles marked."""
        grid = np.zeros((self.n, self.m), dtype=bool)
        for x, y in self.obstacles:
            if 0 <= x < self.n and 0 <= y < self.m:
                grid[x, y] = True
        return grid

    def analyze(self):
        """Perform full topological analysis of the environment."""
        print("Analyzing environment topology...")

        # Find chambers (open areas)
        self._find_chambers()
        print(f"Found {len(self.chambers)} open chambers")

        # Find tunnels and narrow passages
        self.tunnels = self._detect_tunnels()
        print(f"Found {len(self.tunnels)} tunnels")

        # Create navigation graph connecting chambers through tunnels
        graph = self.get_navigation_graph()

        return self.chambers, self.tunnels, graph

    def _find_chambers(self):
        """Find distinct open chambers in the environment."""
        # Initialize visited grid
        visited = np.zeros((self.n, self.m), dtype=bool)

        # Mark obstacles as visited
        for x, y in self.obstacles:
            if 0 <= x < self.n and 0 <= y < self.m:
                visited[x, y] = True

        # BFS to find chambers
        for i in range(self.n):
            for j in range(self.m):
                if not visited[i, j]:
                    chamber = self._explore_chamber(i, j, visited)
                    if len(chamber) > 1:  # Ignore tiny chambers
                        self.chambers.append(chamber)

        return self.chambers

    def _explore_chamber(self, start_x, start_y, visited):
        """Explore a single chamber with BFS."""
        chamber = []
        queue = deque([(start_x, start_y)])

        while queue:
            x, y = queue.popleft()

            if visited[x, y]:
                continue

            visited[x, y] = True
            chamber.append((x, y))

            # Explore neighbors
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy

                if (
                    0 <= nx < self.n
                    and 0 <= ny < self.m
                    and not visited[nx, ny]
                    and (nx, ny) not in self.obstacles
                ):
                    queue.append((nx, ny))

        return chamber

    def _detect_tunnels(self):
        """Detect tunnels in the environment."""
        tunnels = []
        visited = set(self.obstacles)  # Mark obstacles as visited

        # Check each cell that could be a tunnel
        for i in range(self.n):
            for j in range(self.m):
                if (i, j) not in visited and self._is_tunnel_cell((i, j)):
                    tunnel = self._trace_tunnel((i, j), visited)
                    if tunnel and len(tunnel["path"]) >= 2:
                        tunnels.append(tunnel)

        return tunnels

    def _is_tunnel_cell(self, pos):
        """Check if a cell could be part of a tunnel."""
        x, y = pos

        if (x, y) in self.obstacles:
            return False

        # Count obstacles in surrounding cells
        obstacle_count = 0
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if (
                nx < 0
                or nx >= self.n
                or ny < 0
                or ny >= self.m
                or (nx, ny) in self.obstacles
            ):
                obstacle_count += 1

        # A tunnel cell has at least 2 adjacent obstacles/boundaries
        return obstacle_count >= 2

    def _trace_tunnel(self, start_pos, visited):
        """Trace the path of a tunnel."""
        tunnel_path = [start_pos]
        local_visited = set([start_pos])
        visited.add(start_pos)
        queue = deque([start_pos])

        while queue:
            x, y = queue.popleft()

            # Check all four directions
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                next_pos = (nx, ny)

                if (
                    0 <= nx < self.n
                    and 0 <= ny < self.m
                    and next_pos not in self.obstacles
                    and next_pos not in local_visited
                    and self._is_tunnel_cell(next_pos)
                ):

                    local_visited.add(next_pos)
                    visited.add(next_pos)
                    tunnel_path.append(next_pos)
                    queue.append(next_pos)

        # Find endpoints of tunnel
        if len(tunnel_path) >= 2:
            entrance, exit = self._find_tunnel_endpoints(tunnel_path)
            if entrance and exit:
                chambers_connected = self._get_chambers_connected_by_tunnel(
                    entrance, exit
                )
                return {
                    "path": tunnel_path,
                    "entrance": entrance,
                    "exit": exit,
                    "width": self._calculate_tunnel_width(tunnel_path),
                    "chambers": chambers_connected,
                }

        return None

    def _find_tunnel_endpoints(self, tunnel_path):
        """Find the entrance and exit points of a tunnel."""
        # Create a set of tunnel positions for quick lookup
        tunnel_set = set(tunnel_path)

        # Find cells with connections to non-tunnel areas
        endpoints = []

        for pos in tunnel_path:
            x, y = pos
            external_connections = []

            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                neighbor = (nx, ny)

                if (
                    0 <= nx < self.n
                    and 0 <= ny < self.m
                    and neighbor not in self.obstacles
                    and neighbor not in tunnel_set
                ):
                    external_connections.append(neighbor)

            if external_connections:
                endpoints.append((pos, external_connections))

        # If we don't have enough endpoints, the tunnel might be a dead end
        if len(endpoints) < 2:
            return None, None

        # Find the two endpoints that are farthest apart
        max_distance = 0
        best_endpoints = None

        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                pos1 = endpoints[i][0]
                pos2 = endpoints[j][0]
                dist = manhattan_distance(pos1, pos2)

                if dist > max_distance:
                    max_distance = dist
                    best_endpoints = (pos1, pos2)

        return best_endpoints

    def _calculate_tunnel_width(self, tunnel_path):
        """Estimate the width of a tunnel."""
        # For narrow tunnels, this will typically be 1
        # For wider tunnels, we estimate based on the maximum number of
        # cells in a cross-section perpendicular to the tunnel's main axis

        # Simple heuristic: check groups of cells that share the same x or y coordinate
        x_counts = {}
        y_counts = {}

        for x, y in tunnel_path:
            x_counts[x] = x_counts.get(x, 0) + 1
            y_counts[y] = y_counts.get(y, 0) + 1

        max_x_count = max(x_counts.values()) if x_counts else 0
        max_y_count = max(y_counts.values()) if y_counts else 0

        return max(1, min(max_x_count, max_y_count))

    def _get_chambers_connected_by_tunnel(self, entrance, exit):
        """Determine which chambers are connected by this tunnel."""
        entrance_chamber = -1
        exit_chamber = -1

        # Find which chamber contains each endpoint
        for i, chamber in enumerate(self.chambers):
            chamber_set = set(chamber)

            # Check entrance
            if entrance in chamber_set:
                entrance_chamber = i
            elif any(manhattan_distance(entrance, pos) <= 2 for pos in chamber):
                entrance_chamber = i

            # Check exit
            if exit in chamber_set:
                exit_chamber = i
            elif any(manhattan_distance(exit, pos) <= 2 for pos in chamber):
                exit_chamber = i

        return [entrance_chamber, exit_chamber]

    def get_navigation_graph(self):
        """Build a navigation graph connecting chambers through tunnels."""
        graph = {}

        # Add nodes for chambers
        for i, chamber in enumerate(self.chambers):
            centroid_x = sum(pos[0] for pos in chamber) / len(chamber)
            centroid_y = sum(pos[1] for pos in chamber) / len(chamber)

            graph[f"chamber_{i}"] = {
                "pos": (int(centroid_x), int(centroid_y)),
                "size": len(chamber),
                "connections": [],
            }

        # Add connections through tunnels
        for i, tunnel in enumerate(self.tunnels):
            chamber_a, chamber_b = tunnel["chambers"]

            # Skip if tunnel doesn't connect two valid chambers
            if chamber_a < 0 or chamber_b < 0 or chamber_a == chamber_b:
                continue

            # Add bidirectional connections
            chamber_a_id = f"chamber_{chamber_a}"
            chamber_b_id = f"chamber_{chamber_b}"

            if chamber_a_id in graph and chamber_b_id in graph:
                graph[chamber_a_id]["connections"].append(
                    {"to": chamber_b_id, "via_tunnel": i, "width": tunnel["width"]}
                )

                graph[chamber_b_id]["connections"].append(
                    {"to": chamber_a_id, "via_tunnel": i, "width": tunnel["width"]}
                )

        return graph

    def get_chamber_containing_position(self, pos):
        """Find which chamber contains a position."""
        for i, chamber in enumerate(self.chambers):
            if pos in chamber:
                return i
            # Check nearby positions
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if (pos[0] + dx, pos[1] + dy) in chamber:
                        return i
        return -1
