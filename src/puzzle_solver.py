import heapq
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import distance_transform_edt
from typing import List, Tuple, Optional, Dict, Set, Any
from collections import deque
import time
import multiprocessing as mp
from copy import deepcopy
from topology_analyzer import TopologyAnalyzer
from tunnel_navigator import TunnelNavigator

# Import optimized Cython functions
from puzzle_cython import (
    state_to_tuple,
    is_state_connected,
    is_still_connected_after_move,
    compute_neighbor_map,
    find_articulation_points,
    heuristic,
    get_successors,
    shape_distance,
    calculate_centroid,
    generate_distance_map,
    manhattan_distance_int as manhattan_distance,
    apply_moves,
    clear_cache,
)

# Global direction constants
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]


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

        # Clear caches at the start of each search
        clear_cache()

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

            # Generate successors using Cython-optimized function
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
        distance_map = generate_distance_map(self.n, self.m, self.obstacles)

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

    def plan_abstract_path(self, distance_map):
        """
        Find a path from start to target with improved robustness.
        """
        print("Planning abstract path...")

        # Calculate centroids for high-level planning
        start_centroid = calculate_centroid(self.start_state)
        target_centroid = calculate_centroid(self.target_state)

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
        open_set = [(manhattan_distance(start, target), 0, start, [start])]
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
                new_h = manhattan_distance((nx, ny), target)
                new_f = new_g + new_h

                heapq.heappush(open_set, (new_f, new_g, (nx, ny), path + [(nx, ny)]))

        # If we can't find a perfect path, try a direct line
        return [start, target]

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
            if shape_distance(current, prev_waypoint) >= max_distance:
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
                self.n, self.m, current_state, target_waypoint, list(self.obstacles)
            )

            # Configure subproblem for quicker search
            max_iterations = min(10000, 5000 * len(current_state))
            depth_limit = max(20, 2 * shape_distance(current_state, target_waypoint))

            # Solve the subproblem
            sub_plan = subproblem.plan(max_iters=max_iterations, depth_lim=depth_limit)

            if sub_plan:
                print(f"Subproblem {i+1} solved with {len(sub_plan)} moves")
                full_plan.extend(sub_plan)

                # Apply the moves to get the new current state
                current_state = apply_moves(current_state, sub_plan)
            else:
                print(f"Failed to solve subproblem {i+1}")
                # Try direct A* for this subproblem with increased limits
                direct_subproblem = AI_Agent(
                    self.n, self.m, current_state, target_waypoint, list(self.obstacles)
                )
                sub_plan = direct_subproblem.plan(max_iters=25000, depth_lim=40)

                if sub_plan:
                    print(f"Direct solution found with {len(sub_plan)} moves")
                    full_plan.extend(sub_plan)
                    current_state = apply_moves(current_state, sub_plan)
                else:
                    print("Couldn't solve subproblem, stopping hierarchical planning")
                    break

        # Check if we reached the final target
        if shape_distance(current_state, self.target_state) < 1:
            print(f"Hierarchical planning successful: total {len(full_plan)} moves")
            return full_plan
        else:
            # Try to complete the plan with a final subproblem
            final_subproblem = AI_Agent(
                self.n, self.m, current_state, self.target_state, list(self.obstacles)
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

    def plan_parallel(self, num_processes=6):
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
                stats.get("nodes_expanded", 0) for _, _, stats in results if stats
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
        distance_map = generate_distance_map(self.n, self.m, self.obstacles)

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
                            20, 2 * shape_distance(current_state, waypoints[i])
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

    def plan_smart(self, max_time=60):
        """
        Smart planning that can handle tunnel obstacles.

        Args:
            max_time: Maximum planning time in seconds

        Returns:
            List of moves or None if no solution found
        """
        print("Starting smart planning with tunnel navigation...")
        start_time = time.time()

        # First analyze the environment topology
        topology = TopologyAnalyzer(self.n, self.m, self.obstacles)
        chambers, tunnels, graph = topology.analyze()

        # Find which chambers contain start and target states
        start_chamber = topology.get_chamber_containing_position(self.start_state[0])
        target_chamber = topology.get_chamber_containing_position(self.target_state[0])

        print(f"Start chamber: {start_chamber}, Target chamber: {target_chamber}")

        # If start and target are in the same chamber, use regular planning
        if start_chamber == target_chamber:
            print("Start and target in same chamber, using regular planning")
            return self.plan()

        # Check if we need to navigate through tunnels
        if tunnels and start_chamber != target_chamber:
            # Try to find a path through chambers
            path = self._find_chamber_path(graph, start_chamber, target_chamber)

            if path:
                print(f"Found path through chambers: {path}")

                # Initialize plan
                full_plan = []
                current_state = self.start_state

                # Navigate through each tunnel in the path
                for i in range(1, len(path)):
                    from_chamber = int(path[i - 1].split("_")[1])
                    to_chamber = int(path[i].split("_")[1])

                    # Find tunnel connecting these chambers
                    tunnel_idx = self._find_connecting_tunnel(
                        graph, from_chamber, to_chamber
                    )

                    if tunnel_idx >= 0:
                        print(
                            f"Navigating tunnel {tunnel_idx} from chamber {from_chamber} to {to_chamber}"
                        )

                        # Create tunnel navigator
                        navigator = TunnelNavigator(
                            self.n, self.m, self.obstacles, topology
                        )

                        # Navigate the tunnel
                        tunnel_plan = navigator.navigate_tunnel(
                            tunnel_idx, current_state, to_chamber
                        )

                        if tunnel_plan:
                            full_plan.extend(tunnel_plan)
                            current_state = apply_moves(current_state, tunnel_plan)
                        else:
                            print(f"Failed to navigate tunnel {tunnel_idx}")
                            break
                    else:
                        print(
                            f"No tunnel found between chambers {from_chamber} and {to_chamber}"
                        )
                        break

                # Final navigation to target shape
                if full_plan:
                    print("Planning final arrangement to target shape...")
                    final_agent = AI_Agent(
                        self.n,
                        self.m,
                        current_state,
                        self.target_state,
                        list(self.obstacles),
                    )
                    final_plan = final_agent.plan(max_iters=10000, depth_lim=30)

                    if final_plan:
                        full_plan.extend(final_plan)
                        print(f"Complete plan found with {len(full_plan)} moves")
                        return full_plan

        # Fall back to hierarchical planning
        print("Tunnel navigation unsuccessful, trying hierarchical planning")
        plan, _ = self.hierarchical_plan()

        # If hierarchical planning failed, try regular planning
        if not plan:
            print("Hierarchical planning failed, trying regular planning")
            plan = self.plan()

        elapsed = time.time() - start_time
        print(f"Smart planning completed in {elapsed:.2f}s")

        return plan

    def _find_chamber_path(self, graph, start_chamber, target_chamber):
        """Find a path through chambers using BFS."""
        if start_chamber < 0 or target_chamber < 0:
            return None

        start_node = f"chamber_{start_chamber}"
        target_node = f"chamber_{target_chamber}"

        if start_node not in graph or target_node not in graph:
            return None

        # BFS to find path
        visited = set()
        queue = deque([(start_node, [start_node])])

        while queue:
            node, path = queue.popleft()

            if node == target_node:
                return path

            if node in visited:
                continue

            visited.add(node)

            # Add neighbors
            for conn in graph[node]["connections"]:
                next_node = conn["to"]
                if next_node not in visited:
                    queue.append((next_node, path + [next_node]))

        return None

    def _find_connecting_tunnel(self, graph, chamber1, chamber2):
        """Find tunnel index that connects two chambers."""
        chamber1_node = f"chamber_{chamber1}"
        chamber2_node = f"chamber_{chamber2}"

        if chamber1_node in graph:
            for conn in graph[chamber1_node]["connections"]:
                if conn["to"] == chamber2_node:
                    return conn["via_tunnel"]

        return -1

    def plan_auto(self, max_time=60):
        """
        Intelligently navigates obstacles while preserving formation shape whenever possible.

        Args:
            max_time: Maximum planning time in seconds

        Returns:
            List of moves or None if no solution found
        """
        print(
            f"Starting formation-preserving planning (date: 2025-04-23 13:19:56, user: MahmoudKebbi)"
        )
        start_time = time.time()

        # First, detect obstacles between start and target
        obstacle_analysis = self._analyze_obstacles_between()

        if not obstacle_analysis["has_obstacles"]:
            print("No significant obstacles detected, using regular planning")
            return self.plan()

        # FORCE going around as the default strategy
        print("Default strategy: Preserving formation by going AROUND obstacles")

        # Generate path around obstacles
        around_path = self._generate_formation_preserving_path(obstacle_analysis)

        # Plan movement along this path
        plan = self._execute_formation_movement(around_path)

        if plan:
            elapsed = time.time() - start_time
            print(
                f"Formation-preserving plan successful with {len(plan)} moves in {elapsed:.2f}s"
            )
            return plan

        # Fallback to regular planning only if formation preservation fails
        print("Formation preservation failed, falling back to regular planning")
        return self.plan()

    def _estimate_tunnel_cost(self, obstacle_analysis):
        """Estimate the cost (number of moves) to go through a tunnel."""
        if not obstacle_analysis["has_tunnel"]:
            return float("inf")  # Can't go through

        # Get basic data
        num_blocks = len(self.start_state)
        tunnel_width = obstacle_analysis.get("tunnel_width", 1)

        # Check if tunnel is too narrow for practical navigation
        if tunnel_width < 1:
            return float("inf")

        # MUCH higher penalty for very narrow tunnels
        narrowness_factor = (
            4.0 if tunnel_width == 1 else (2.0 if tunnel_width == 2 else 1.2)
        )

        # Estimate formations needed
        # Line formation cost: higher for more blocks
        line_cost = num_blocks * 3 * narrowness_factor

        # Through tunnel cost: distance to traverse plus extra for narrow tunnels
        wall_thickness = obstacle_analysis.get("wall_width", 3)
        through_cost = wall_thickness * narrowness_factor * 2

        # Reform cost: higher for more blocks
        reform_cost = num_blocks * 3

        # Additional penalties
        # Higher cost for small tunnels relative to number of blocks
        block_to_tunnel_ratio = num_blocks / max(1, tunnel_width)
        size_penalty = block_to_tunnel_ratio * 5

        # Calculate distance from start to tunnel
        start_centroid = obstacle_analysis["start_centroid"]
        tunnel_pos = obstacle_analysis["tunnel_position"]
        start_to_tunnel = manhattan_distance(start_centroid, tunnel_pos)

        # Final estimate (with increased overall cost)
        total_cost = (
            line_cost + through_cost + reform_cost + size_penalty + start_to_tunnel
        )

        # Print detailed breakdown for debugging
        print(f"  Tunnel cost breakdown:")
        print(f"    - Line formation: {line_cost:.1f}")
        print(f"    - Through tunnel: {through_cost:.1f}")
        print(f"    - Reform shape: {reform_cost:.1f}")
        print(f"    - Size penalty: {size_penalty:.1f}")
        print(f"    - Distance to tunnel: {start_to_tunnel:.1f}")

        return total_cost

    def _estimate_around_cost(self, obstacle_analysis):
        """
        Estimate the cost to go around an obstacle and find the path around.

        Returns:
            Tuple of (estimated cost, path around obstacle)
        """
        # Get grid dimensions
        n, m = self.n, self.m

        # Extract wall extents
        wall_left = obstacle_analysis.get("wall_left", 0)
        wall_right = obstacle_analysis.get("wall_right", n - 1)
        wall_top = obstacle_analysis.get("wall_top", 0)
        wall_bottom = obstacle_analysis.get("wall_bottom", m - 1)

        # Ensure we have reasonable values
        wall_left = max(0, min(wall_left, n - 1))
        wall_right = max(0, min(wall_right, n - 1))
        wall_top = max(0, min(wall_top, m - 1))
        wall_bottom = max(0, min(wall_bottom, m - 1))

        # Calculate wall dimensions
        wall_width = wall_right - wall_left + 1
        wall_height = wall_bottom - wall_top + 1

        # If wall is very small, going around is cheap
        if wall_width <= 3 or wall_height <= 3:
            return 10, [(0, 0), (n - 1, m - 1)]  # Dummy path, will be recalculated

        start_centroid = obstacle_analysis["start_centroid"]
        target_centroid = obstacle_analysis["target_centroid"]

        # Check if start and target are on opposite sides of the wall
        start_x, start_y = int(start_centroid[0]), int(start_centroid[1])
        target_x, target_y = int(target_centroid[0]), int(target_centroid[1])

        # Determine if we need to go around horizontally or vertically
        horizontal_wall = wall_width > wall_height
        start_side, target_side = None, None

        if horizontal_wall:
            # Check which side of the horizontal wall
            start_side = "above" if start_y < wall_top else "below"
            target_side = "above" if target_y < wall_top else "below"
        else:
            # Check which side of the vertical wall
            start_side = "left" if start_x < wall_left else "right"
            target_side = "left" if target_x < wall_left else "right"

        # If start and target are on the same side, going around is unnecessary
        if start_side == target_side:
            # Direct path - just need to avoid the wall
            direct_cost = manhattan_distance(start_centroid, target_centroid) * 1.2
            direct_path = [start_centroid, target_centroid]
            return direct_cost, direct_path

        # Choose the best direction to go around
        paths = []

        # Create a padding to ensure we stay away from the wall
        padding = 2

        if horizontal_wall:
            # Going around left
            if wall_left > padding:
                left_point = (wall_left - padding, (wall_top + wall_bottom) // 2)
                left_path = [start_centroid, left_point, target_centroid]
                left_distance = manhattan_distance(
                    start_centroid, left_point
                ) + manhattan_distance(left_point, target_centroid)
                paths.append((left_distance, left_path))

            # Going around right
            if wall_right < n - padding - 1:
                right_point = (wall_right + padding, (wall_top + wall_bottom) // 2)
                right_path = [start_centroid, right_point, target_centroid]
                right_distance = manhattan_distance(
                    start_centroid, right_point
                ) + manhattan_distance(right_point, target_centroid)
                paths.append((right_distance, right_path))
        else:
            # Going around top
            if wall_top > padding:
                top_point = ((wall_left + wall_right) // 2, wall_top - padding)
                top_path = [start_centroid, top_point, target_centroid]
                top_distance = manhattan_distance(
                    start_centroid, top_point
                ) + manhattan_distance(top_point, target_centroid)
                paths.append((top_distance, top_path))

            # Going around bottom
            if wall_bottom < m - padding - 1:
                bottom_point = ((wall_left + wall_right) // 2, wall_bottom + padding)
                bottom_path = [start_centroid, bottom_point, target_centroid]
                bottom_distance = manhattan_distance(
                    start_centroid, bottom_point
                ) + manhattan_distance(bottom_point, target_centroid)
                paths.append((bottom_distance, bottom_path))

        if not paths:
            # No valid path around, try direct path
            wall_side_x = (
                wall_left - padding if start_x < wall_left else wall_right + padding
            )
            wall_side_y = (
                wall_top - padding if start_y < wall_top else wall_bottom + padding
            )

            # Create a point that follows the wall boundary
            side_point = (
                min(max(0, wall_side_x), n - 1),
                min(max(0, wall_side_y), m - 1),
            )

            direct_path = [start_centroid, side_point, target_centroid]
            direct_distance = manhattan_distance(
                start_centroid, side_point
            ) + manhattan_distance(side_point, target_centroid)

            return direct_distance * 1.5, direct_path

        # Find the shortest path
        paths.sort(key=lambda p: p[0])
        path_length, best_path = paths[0]

        # REDUCED PENALTY: Estimate the actual move cost
        num_blocks = len(self.start_state)

        # Less penalty for formation movement
        formation_factor = 1.0 + (num_blocks / 15)  # Reduced from 10 to 15

        # Reduced distance penalty
        distance_factor = 1.0 + (path_length / 30)  # Reduced from 20 to 30

        # Final cost estimate (generally lower than before)
        cost_estimate = (
            path_length * formation_factor * distance_factor * 0.8
        )  # 20% discount

        # Print detailed breakdown for debugging
        print(f"  Around cost breakdown:")
        print(f"    - Path length: {path_length:.1f}")
        print(f"    - Formation factor: {formation_factor:.2f}")
        print(f"    - Distance factor: {distance_factor:.2f}")
        print(f"    - Total estimate: {cost_estimate:.1f}")

        return cost_estimate, best_path

    def _analyze_obstacles_between(self):
        """
        Improved analysis of obstacles between start and target positions.

        Returns:
            Dictionary with obstacle analysis information
        """
        # Calculate centroids
        start_centroid = calculate_centroid(self.start_state)
        target_centroid = calculate_centroid(self.target_state)

        # Create grid representation
        grid = np.zeros((self.n, self.m), dtype=bool)
        for x, y in self.obstacles:
            if 0 <= x < self.n and 0 <= y < self.m:
                grid[x, y] = True

        # Generate a line between centroids to check for obstacles
        line_points = self._get_line_points(
            int(start_centroid[0]),
            int(start_centroid[1]),
            int(target_centroid[0]),
            int(target_centroid[1]),
        )

        # Check for obstacle runs along the line
        obstacle_runs = []
        current_run = None

        for i, (x, y) in enumerate(line_points):
            if 0 <= x < self.n and 0 <= y < self.m:
                if grid[x, y]:  # Hit an obstacle
                    if current_run is None:
                        current_run = {"start": i, "start_pos": (x, y)}
                elif current_run is not None:  # End of obstacle run
                    current_run["end"] = i
                    current_run["end_pos"] = (x, y)
                    current_run["length"] = i - current_run["start"]
                    obstacle_runs.append(current_run)
                    current_run = None

        # If we end in an obstacle, close the run
        if current_run is not None:
            current_run["end"] = len(line_points) - 1
            current_run["end_pos"] = line_points[-1]
            current_run["length"] = current_run["end"] - current_run["start"]
            obstacle_runs.append(current_run)

        # Look for significant obstacle runs (walls)
        significant_walls = [run for run in obstacle_runs if run["length"] >= 3]

        result = {
            "has_obstacles": len(significant_walls) > 0,
            "start_centroid": start_centroid,
            "target_centroid": target_centroid,
            "grid": grid,
            "total_distance": manhattan_distance(start_centroid, target_centroid),
        }

        if not significant_walls:
            return result  # No significant obstacles

        # Find the most significant wall
        wall = max(significant_walls, key=lambda r: r["length"])
        result["wall"] = wall

        # Determine wall orientation
        start_x, start_y = wall["start_pos"]
        end_x, end_y = wall["end_pos"]

        if abs(end_y - start_y) > abs(end_x - start_x):
            orientation = "horizontal"  # Wall runs horizontally
        else:
            orientation = "horizontal"  # Wall runs vertically

        result["orientation"] = orientation

        # Find the dimensions of the wall by exploring outward
        # This is similar to flood fill but constrained to the wall
        wall_cells = set()
        visited = set()
        queue = deque([wall["start_pos"]])

        while queue:
            x, y = queue.popleft()

            if (x, y) in visited:
                continue

            visited.add((x, y))

            if 0 <= x < self.n and 0 <= y < self.m and grid[x, y]:
                wall_cells.add((x, y))

                # Add neighbors
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if (nx, ny) not in visited:
                        queue.append((nx, ny))

        # Calculate wall bounds from discovered cells
        wall_xs = [x for x, y in wall_cells]
        wall_ys = [y for x, y in wall_cells]

        if wall_cells:
            result.update(
                {
                    "wall_left": min(wall_xs),
                    "wall_right": max(wall_xs),
                    "wall_top": min(wall_ys),
                    "wall_bottom": max(wall_ys),
                    "wall_width": max(wall_xs) - min(wall_xs) + 1,
                    "wall_height": max(wall_ys) - min(wall_ys) + 1,
                }
            )

        # Look for tunnel passage in the wall
        tunnel_position = self._find_tunnel_in_wall(wall, orientation, grid)

        if tunnel_position:
            result["has_tunnel"] = True
            result["tunnel_position"] = tunnel_position
            result["tunnel_width"] = self._estimate_tunnel_width(
                tunnel_position, orientation, grid
            )
        else:
            result["has_tunnel"] = False

        return result

    def _plan_around_obstacle(self, path_around):
        """
        Plan a movement path around an obstacle using waypoints.

        Args:
            path_around: List of waypoints to navigate around the obstacle

        Returns:
            List of moves or None if planning failed
        """
        if not path_around:
            return None

        print(f"Planning movement around obstacle using {len(path_around)} waypoints")

        # Add more intermediate waypoints for smoother path
        waypoints = self._add_intermediate_waypoints(path_around)

        # Create states at each waypoint
        waypoint_states = []
        for point in waypoints:
            # Create formation centered at this waypoint
            formation = self._create_formation_at_position(point)
            waypoint_states.append(formation)

        # Always make sure target state is the final waypoint
        if waypoint_states and self.target_state != waypoint_states[-1]:
            waypoint_states.append(self.target_state)

        # Do hierarchical planning through the waypoints
        return self.solve_subproblems(waypoint_states)

    def _add_intermediate_waypoints(self, path):
        """Add intermediate waypoints for smoother navigation."""
        if len(path) <= 2:
            return path

        result = [path[0]]

        # Add intermediate points between distant waypoints
        for i in range(1, len(path)):
            prev = path[i - 1]
            current = path[i]

            distance = manhattan_distance(prev, current)

            # If points are far apart, add intermediate points
            if distance > 8:
                # Number of points to add
                num_points = min(3, int(distance / 5))

                for j in range(1, num_points + 1):
                    # Interpolate position
                    factor = j / (num_points + 1)
                    x = int(prev[0] + (current[0] - prev[0]) * factor)
                    y = int(prev[1] + (current[1] - prev[1]) * factor)
                    result.append((x, y))

            result.append(current)

        return result

    def _get_line_points(self, x1, y1, x2, y2):
        """
        Get all points along a line between (x1, y1) and (x2, y2) using Bresenham's algorithm.

        Args:
            x1, y1: Starting point coordinates
            x2, y2: Ending point coordinates

        Returns:
            List of points (x, y) along the line
        """
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        # Determine direction of line
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1

        # Decision variable
        err = dx - dy

        # Current position
        x, y = x1, y1

        while True:
            # Add current point
            points.append((x, y))

            # If we've reached the end point, we're done
            if x == x2 and y == y2:
                break

            # Calculate next position
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        return points

    def _generate_formation_preserving_path(self, obstacle_analysis):
        """
        Generate a path that goes around obstacles while keeping the formation intact.

        This prioritizes going wide around obstacles rather than through them.
        """
        # Extract data
        start_centroid = obstacle_analysis["start_centroid"]
        target_centroid = obstacle_analysis["target_centroid"]

        # Convert centroids to integer coordinates
        sx, sy = int(start_centroid[0]), int(start_centroid[1])
        tx, ty = int(target_centroid[0]), int(target_centroid[1])

        # Find the direction from start to target
        dx = tx - sx
        dy = ty - sy

        # Determine if we need to go horizontally or vertically around obstacles
        horizontal_first = abs(dx) > abs(dy)

        # Create waypoints for path around
        waypoints = [start_centroid]

        if horizontal_first:
            # Go horizontally first, then vertically
            # Check if we should go horizontally left or right of the grid
            if obstacle_analysis.get("orientation") == "vertical":
                # There's a vertical wall
                wall_x = obstacle_analysis.get("wall_left", self.n // 2)

                # Determine which side to go around
                if sx < wall_x and tx < wall_x:
                    # Both start and target are on left side
                    mid_x = min(sx, tx) - 2  # Go further left
                elif sx > wall_x and tx > wall_x:
                    # Both start and target are on right side
                    mid_x = max(sx, tx) + 2  # Go further right
                else:
                    # Need to go around
                    left_space = wall_x
                    right_space = self.n - 1 - wall_x

                    if left_space >= right_space:
                        # More space on left
                        mid_x = max(0, wall_x - int(self.n * 0.25))
                    else:
                        # More space on right
                        mid_x = min(self.n - 1, wall_x + int(self.n * 0.25))
            else:
                # No clear vertical wall, just go toward the target x-coordinate
                mid_x = tx

            # Add horizontal movement waypoint
            waypoints.append((mid_x, sy))

            # Add target
            waypoints.append(target_centroid)
        else:
            # Go vertically first, then horizontally
            # Check if we should go vertically above or below
            if obstacle_analysis.get("orientation") == "horizontal":
                # There's a horizontal wall
                wall_y = obstacle_analysis.get("wall_top", self.m // 2)

                # Determine which side to go around
                if sy < wall_y and ty < wall_y:
                    # Both start and target are above
                    mid_y = min(sy, ty) - 2  # Go further up
                elif sy > wall_y and ty > wall_y:
                    # Both start and target are below
                    mid_y = max(sy, ty) + 2  # Go further down
                else:
                    # Need to go around
                    top_space = wall_y
                    bottom_space = self.m - 1 - wall_y

                    if top_space >= bottom_space:
                        # More space on top
                        mid_y = max(0, wall_y - int(self.m * 0.25))
                    else:
                        # More space on bottom
                        mid_y = min(self.m - 1, wall_y + int(self.m * 0.25))
            else:
                # No clear horizontal wall, just go toward the target y-coordinate
                mid_y = ty

            # Add vertical movement waypoint
            waypoints.append((sx, mid_y))

            # Add target
            waypoints.append(target_centroid)

        # Add corner waypoint if we have obstacles
        if len(waypoints) == 3 and (
            obstacle_analysis.get("orientation") == "horizontal"
            or obstacle_analysis.get("orientation") == "vertical"
        ):
            # Insert corner waypoint
            corner = (waypoints[1][0], waypoints[2][1])
            waypoints.insert(2, corner)

        # Ensure path goes wide around obstacles
        return self._ensure_path_avoids_obstacles(waypoints, obstacle_analysis["grid"])

    def _ensure_path_avoids_obstacles(self, waypoints, obstacle_grid):
        """Adjust the path to ensure it stays well clear of obstacles."""
        if len(waypoints) <= 2:
            return waypoints

        adjusted_waypoints = [waypoints[0]]

        # Check each segment for obstacle proximity
        for i in range(1, len(waypoints)):
            prev = waypoints[i - 1]
            current = waypoints[i]

            # Find points along this segment
            segment_points = self._get_line_points(
                int(prev[0]), int(prev[1]), int(current[0]), int(current[1])
            )

            # Check if any point is too close to obstacles
            too_close = False
            for x, y in segment_points:
                if 0 <= x < self.n and 0 <= y < self.m:
                    # Check if any obstacle is within 2 cells
                    for dx in range(-2, 3):
                        for dy in range(-2, 3):
                            nx, ny = x + dx, y + dy
                            if (
                                0 <= nx < self.n
                                and 0 <= ny < self.m
                                and obstacle_grid[nx, ny]
                            ):
                                too_close = True
                                break
                        if too_close:
                            break
                if too_close:
                    break

            if too_close:
                # Adjust waypoint to move further from obstacles
                cx, cy = current

                # Calculate vector from center of obstacles to waypoint
                obs_center_x = self.n // 2
                obs_center_y = self.m // 2

                vx = cx - obs_center_x
                vy = cy - obs_center_y

                # Normalize and scale vector
                magnitude = max(1, (vx**2 + vy**2) ** 0.5)
                vx = vx / magnitude * 4  # Push out by 4 units
                vy = vy / magnitude * 4

                # Create adjusted waypoint
                adjusted = (
                    max(0, min(self.n - 1, int(cx + vx))),
                    max(0, min(self.m - 1, int(cy + vy))),
                )

                adjusted_waypoints.append(adjusted)
            else:
                adjusted_waypoints.append(current)

        return adjusted_waypoints

    def _execute_formation_movement(self, path):
        """
        Execute movement that preserves formation shape along the given path.

        Args:
            path: List of waypoints to navigate around obstacles

        Returns:
            List of moves or None if planning failed
        """
        if not path or len(path) < 2:
            return None

        print(f"Executing formation movement along {len(path)} waypoints")
        print(f"  Path: {path}")

        # Create intermediate waypoints for smoother movement
        smooth_path = self._add_intermediate_waypoints(path)

        # Create formation states at each waypoint, preserving shape
        waypoint_states = []

        # Start with initial state
        formation_shape = self._extract_formation_shape(self.start_state)

        for point in smooth_path:
            # Create a copy of the formation at this waypoint
            formation_at_point = self._place_formation_at(formation_shape, point)

            if formation_at_point:
                waypoint_states.append(formation_at_point)
            else:
                print(f"  Cannot place formation at {point}, skipping")

        # Ensure target state is last
        if waypoint_states and waypoint_states[-1] != self.target_state:
            waypoint_states.append(self.target_state)

        # Skip the first state if it's too close to start state
        if len(waypoint_states) > 2 and waypoint_states[0] == self.start_state:
            waypoint_states = waypoint_states[1:]

        # Now plan movement between these states
        print(f"Planning movement through {len(waypoint_states)} formation states")

        # Use hierarchical planning with extra time for subproblems
        return self.solve_subproblems(waypoint_states)

    def _extract_formation_shape(self, state):
        """Extract the relative shape of a formation."""
        if not state:
            return []

        # Calculate centroid
        centroid = calculate_centroid(state)

        # Calculate relative positions
        shape = []
        for x, y in state:
            rel_x = x - centroid[0]
            rel_y = y - centroid[1]
            shape.append((rel_x, rel_y))

        return shape

    def _place_formation_at(self, formation_shape, center):
        """Place a formation at the specified center point, avoiding obstacles."""
        if not formation_shape:
            return []

        center_x, center_y = center

        # Convert back to absolute coordinates
        formation = []
        for rel_x, rel_y in formation_shape:
            x = int(center_x + rel_x)
            y = int(center_y + rel_y)

            # Check if position is valid
            if 0 <= x < self.n and 0 <= y < self.m and (x, y) not in self.obstacles:
                formation.append((x, y))
            else:
                # Find nearby valid position
                found = False
                for d in range(1, 5):
                    for dx in range(-d, d + 1):
                        for dy in range(-d, d + 1):
                            if abs(dx) + abs(dy) == d:  # Manhattan distance = d
                                nx, ny = x + dx, y + dy
                                if (
                                    0 <= nx < self.n
                                    and 0 <= ny < self.m
                                    and (nx, ny) not in self.obstacles
                                    and (nx, ny) not in formation
                                ):
                                    formation.append((nx, ny))
                                    found = True
                                    break
                        if found:
                            break
                    if found:
                        break

                if not found:
                    # Try anywhere valid
                    for test_x in range(self.n):
                        for test_y in range(self.m):
                            if (test_x, test_y) not in self.obstacles and (
                                test_x,
                                test_y,
                            ) not in formation:
                                formation.append((test_x, test_y))
                                break
                        if len(formation) == len(formation_shape):
                            break

        # Ensure we have the correct number of blocks
        while len(formation) < len(formation_shape):
            for x in range(self.n):
                for y in range(self.m):
                    if (x, y) not in self.obstacles and (x, y) not in formation:
                        formation.append((x, y))
                        break
                if len(formation) == len(formation_shape):
                    break

        return formation

    def plan_smart_formation(self):
        """
        Plan formation-preserving movement around walls.

        Returns:
            Plan of moves
        """
        print("Starting formation-preserving wall navigation planning...")
        start_time = time.time()

        # Analyze the environment for wall obstacles
        wall_analysis = self._analyze_wall_obstacles()

        # If we found a simple wall, use specialized formation planning
        if wall_analysis["is_simple_wall"]:
            print(f"Detected simple wall obstacle: {wall_analysis['wall_type']}")
            plan = self._plan_formation_around_wall(wall_analysis)

            if plan:
                elapsed = time.time() - start_time
                print(
                    f"Formation navigation successful with {len(plan)} moves in {elapsed:.2f}s"
                )
                return plan

        # If not a simple wall or formation planning failed, fall back to plan_auto
        print("Complex obstacle structure detected, falling back to standard planning")
        return self.plan_auto(max_time=30)

    def _analyze_wall_obstacles(self):
        """
        Analyzes the environment to detect and characterize wall obstacles.

        Returns:
            Dictionary with wall analysis information
        """
        # Get centroids of start and target positions
        start_centroid = calculate_centroid(self.start_state)
        target_centroid = calculate_centroid(self.target_state)

        # Create grid representation
        grid = np.zeros((self.n, self.m), dtype=bool)
        for x, y in self.obstacles:
            if 0 <= x < self.n and 0 <= y < self.m:
                grid[x, y] = True

        # Get points along direct line between centroids
        line_points = self._get_line_points(
            int(start_centroid[0]),
            int(start_centroid[1]),
            int(target_centroid[0]),
            int(target_centroid[1]),
        )

        # Count obstacles along the line
        obstacle_counts = []
        current_run = 0

        for x, y in line_points:
            if 0 <= x < self.n and 0 <= y < self.m and grid[x, y]:
                current_run += 1
            elif current_run > 0:
                obstacle_counts.append(current_run)
                current_run = 0

        # Add final run if any
        if current_run > 0:
            obstacle_counts.append(current_run)

        # Initialize result dictionary
        result = {
            "is_simple_wall": False,
            "wall_type": "none",
            "grid": grid,
            "start_centroid": start_centroid,
            "target_centroid": target_centroid,
        }

        # Analyze obstacle pattern
        if len(obstacle_counts) == 1 and 1 <= obstacle_counts[0] <= 10:
            # Single wall with reasonable thickness
            result["is_simple_wall"] = True
            result["wall_thickness"] = obstacle_counts[0]

            # Find wall extents
            wall_cells = self._find_connected_wall(line_points, grid)

            if wall_cells:
                wall_xs = [x for x, y in wall_cells]
                wall_ys = [y for x, y in wall_cells]

                result["wall_left"] = min(wall_xs)
                result["wall_right"] = max(wall_xs)
                result["wall_top"] = min(wall_ys)
                result["wall_bottom"] = max(wall_ys)
                result["wall_height"] = max(wall_ys) - min(wall_ys) + 1
                result["wall_width"] = max(wall_xs) - min(wall_xs) + 1

                # Determine wall type by its actual dimensions
                result["wall_type"] = (
                    "horizontal"
                    if result["wall_width"] > result["wall_height"]
                    else "vertical"
                )
                print(
                    f"Wall dimensions: {result['wall_width']}x{result['wall_height']}, type: {result['wall_type']}"
                )

        return result

    def _plan_formation_around_wall(self, wall_analysis):
        """
        Simple three-stage wall navigation: 
        1. Move to the side
        2. Then up/down
        3. Then to target
        """
        print("Using simple three-stage wall navigation")

        # Extract wall information
        wall_type = wall_analysis["wall_type"]
        start_centroid = wall_analysis["start_centroid"]
        target_centroid = wall_analysis["target_centroid"]

        # Get the formation shape to preserve
        formation_shape = self._extract_formation_shape(self.start_state)

        # Calculate formation dimensions for safety bounds
        rel_x_coords = [x for x, y in formation_shape]
        rel_y_coords = [y for x, y in formation_shape]

        formation_width = max(rel_x_coords) - min(rel_x_coords) + 1
        formation_height = max(rel_y_coords) - min(rel_y_coords) + 1

        # Safety margin - how far to stay from walls
        safety_margin = max(5, formation_height * 3)

        # For vertical walls (extends up-down)
        if wall_type == "vertical":
            wall_top = wall_analysis.get("wall_top", 0)
            wall_bottom = wall_analysis.get("wall_bottom", self.m - 1)
            start_y = int(start_centroid[1])
            target_y = int(target_centroid[1])

            # Determine whether to go above or below the wall
            # Check if both start and target are on same side of wall
            if (start_y < wall_top and target_y < wall_top) or (start_y > wall_bottom and target_y > wall_bottom):
                # Both on same side, just ensure we're far enough away
                if start_y < wall_top:  # Both above wall
                    side_y = max(0, min(start_y, target_y) - safety_margin)
                else:  # Both below wall
                    side_y = min(self.m - 1, max(start_y, target_y) + safety_margin)
            else:  # ONLY calculate this if they're on DIFFERENT sides
                # Decide based on available space and positions
                if wall_top <= self.m - 1 - wall_bottom:  # More space above wall
                    # Use a percentage of grid height for better scaling
                    grid_height_percentage = 0.60  # Increased to 60%
                    min_clearance = 15  # Absolute minimum clearance in units
                    side_y = max(0, wall_top - max(safety_margin, min_clearance, int(self.m * grid_height_percentage)))
                else:  # More space below wall
                    grid_height_percentage = 0.60  # Increased to 60%
                    min_clearance = 15  # Absolute minimum clearance in units
                    side_y = min(self.m - 1, wall_bottom + max(safety_margin, min_clearance, int(self.m * grid_height_percentage)))

            # Stage 1: Move VERTICALLY first to clear the wall
            print(f"  Stage 1: Move vertically to {side_y}")
            stage1_point = (int(start_centroid[0]), int(self.m-(formation_width)-5))

            # Stage 2: Move HORIZONTALLY to target x-coordinate, still at safe y
            stage2_point = (int(target_centroid[0]), int(self.m - (formation_width) - 5))

        # For horizontal walls (extends left-to-right)
        else:
            wall_left = wall_analysis.get("wall_left", 0)
            wall_right = wall_analysis.get("wall_right", self.n - 1)
            start_x = int(start_centroid[0])
            target_x = int(target_centroid[0])

            # Safety margin for horizontal movement
            safety_margin = max(10, formation_width * 2)

            # Determine whether to go left or right of the wall
            # Check if both start and target are on same side of wall
            if (start_x < wall_left and target_x < wall_left) or (start_x > wall_right and target_x > wall_right):
                # Both on same side, just ensure we're far enough away
                if start_x < wall_left:  # Both left of wall
                    side_x = max(0, min(start_x, target_x) - safety_margin)
                else:  # Both right of wall
                    side_x = min(self.n - 1, max(start_x, target_x) + safety_margin)
            # Otherwise, decide based on available space
            elif wall_left <= self.n - 1 - wall_right:  # More space on left
                side_x = max(0, wall_left - safety_margin)
            else:  # More space on right
                side_x = min(self.n - 1, wall_right + safety_margin)

            # Stage 1: Move HORIZONTALLY first to clear the wall
            stage1_point = (int(side_x), int(start_centroid[1]))

            # Stage 2: Move VERTICALLY to target y-coordinate, still at safe x
            stage2_point = (int(side_x), int(target_centroid[1]))

        # Create formations at the three key points
        stage1_formation = self._place_formation_at(formation_shape, stage1_point)
        stage2_formation = self._place_formation_at(formation_shape, stage2_point)

        # Create the simple three-stage plan
        formations = [
            self.start_state,
            stage1_formation,
            stage2_formation, 
            self.target_state,
        ]

        # Log the plan
        print(f"Three-stage navigation path:")
        print(f"  Start: {tuple(int(x) for x in start_centroid)}")
        print(f"  Stage 1: {stage1_point}")
        print(f"  Stage 2: {stage2_point}")
        print(f"  Target: {tuple(int(x) for x in target_centroid)}")

        # Solve each step separately and merge
        return self._solve_three_stage_navigation(formations)

    def _find_connected_wall(self, line_points, grid):
        """Find all connected obstacle cells that form the wall"""
        # Start with obstacles along the line
        wall_cells = set()
        seed_points = []

        for x, y in line_points:
            if 0 <= x < self.n and 0 <= y < self.m and grid[x, y]:
                seed_points.append((x, y))
                wall_cells.add((x, y))

        # Expand to connected obstacles using BFS
        if not seed_points:
            return wall_cells

        queue = deque(seed_points)
        while queue:
            x, y = queue.popleft()

            # Check 4-connected neighbors
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy

                if (
                    0 <= nx < self.n
                    and 0 <= ny < self.m
                    and grid[nx, ny]
                    and (nx, ny) not in wall_cells
                ):
                    wall_cells.add((nx, ny))
                    queue.append((nx, ny))

        return wall_cells

    def _solve_three_stage_navigation(self, formations):
        """
        Solve each stage of the three-stage navigation separately.

        Args:
            formations: List of formation states for each stage

        Returns:
            List of moves or None if planning failed
        """
        if not formations or len(formations) < 2:
            return None

        print(f"Solving three-stage navigation with {len(formations)} formations")

        full_plan = []
        current_state = formations[0]

        # Process each stage separately with appropriate limits
        for i, target_formation in enumerate(formations[1:], 1):
            print(f"Solving stage {i}/{len(formations)-1}")

            # Create subproblem for this stage
            subproblem = AI_Agent(
                self.n, self.m, current_state, target_formation, list(self.obstacles)
            )

            # Use higher limits for each critical stage
            max_iterations = 20000  # Smaller than before to solve faster
            depth_limit = 35  # Reduced but still sufficient

            # Solve this stage
            sub_plan = subproblem.plan(max_iters=max_iterations, depth_lim=depth_limit)

            if sub_plan:
                print(f"  Stage {i} solved with {len(sub_plan)} moves")
                full_plan.extend(sub_plan)
                current_state = apply_moves(current_state, sub_plan)
            else:
                print(f"  Failed to solve stage {i}, trying parallel planning")
                # Try parallel planning with fewer processes for quicker results
                parallel_subproblem = AI_Agent(
                    self.n,
                    self.m,
                    current_state,
                    target_formation,
                    list(self.obstacles),
                )
                sub_plan = parallel_subproblem.plan_parallel(num_processes=2)

                if sub_plan:
                    print(f"  Parallel solution found with {len(sub_plan)} moves")
                    full_plan.extend(sub_plan)
                    current_state = apply_moves(current_state, sub_plan)
                else:
                    print("  Stage failed - trying direct planning to final target")
                    # Last resort - try to go directly to target
                    if i < len(formations) - 1:
                        direct_subproblem = AI_Agent(
                            self.n,
                            self.m,
                            current_state,
                            formations[-1],
                            list(self.obstacles),
                        )
                        final_plan = direct_subproblem.plan(
                            max_iters=20000, depth_lim=40
                        )

                        if final_plan:
                            print(
                                f"  Direct plan to target found with {len(final_plan)} moves"
                            )
                            full_plan.extend(final_plan)
                            return full_plan

                    print("  All attempts to solve stage failed")
                    return None  # If we can't solve a stage, return failure

        return full_plan

    def _solve_wall_navigation_subproblems(self, formations):
        """
        Specialized subproblem solver for wall navigation with higher limits.

        Args:
            formations: List of formation states

        Returns:
            List of moves or None if planning failed
        """
        if not formations or len(formations) < 2:
            return None

        print(f"Solving wall navigation with {len(formations)} key formations")

        full_plan = []
        current_state = formations[0]

        # Process each subproblem with higher limits
        for i, target_formation in enumerate(formations[1:], 1):
            print(f"Solving wall navigation subproblem {i}/{len(formations)-1}")

            # Create subproblem with HIGHER iteration limits for wall navigation
            subproblem = AI_Agent(
                self.n, self.m, current_state, target_formation, list(self.obstacles)
            )

            # Use much higher limits for these critical subproblems
            max_iterations = 30000  # Increased from 10000
            depth_limit = 50  # Increased from 20

            # Solve the subproblem
            sub_plan = subproblem.plan(max_iters=max_iterations, depth_lim=depth_limit)

            if sub_plan:
                print(f"  Subproblem {i} solved with {len(sub_plan)} moves")
                full_plan.extend(sub_plan)
                current_state = apply_moves(current_state, sub_plan)
            else:
                print(f"  Failed to solve subproblem {i}, trying parallel planning")
                # Try parallel planning as a last resort
                parallel_subproblem = AI_Agent(
                    self.n,
                    self.m,
                    current_state,
                    target_formation,
                    list(self.obstacles),
                )
                sub_plan = parallel_subproblem.plan_parallel(num_processes=4)

                if sub_plan:
                    print(f"  Parallel solution found with {len(sub_plan)} moves")
                    full_plan.extend(sub_plan)
                    current_state = apply_moves(current_state, sub_plan)
                else:
                    print("  All attempts to solve subproblem failed")
                    return None  # If we can't solve a subproblem, return failure

        return full_plan

    def _create_formation_states(self, waypoints):
        """
        Create formation states at each waypoint that preserve the original formation shape.

        Args:
            waypoints: List of waypoint coordinates

        Returns:
            List of states (formations) at each waypoint
        """
        # Extract the relative shape from the start state
        formation_shape = self._extract_formation_shape(self.start_state)

        # Create a state at each waypoint
        states = [self.start_state]  # Start with the initial state

        for i, waypoint in enumerate(
            waypoints[1:], 1
        ):  # Skip the first waypoint (start position)
            # Create a formation at this waypoint, preserving shape
            formation = self._place_formation_at(formation_shape, waypoint)

            if formation and len(formation) == len(self.start_state):
                # Only add if it's a valid formation with the right number of blocks
                states.append(formation)
            else:
                print(f"Could not create valid formation at waypoint {i}, skipping")

        return states

    def _generate_formation_move(self, state, dx, dy):
        """
        Generate moves that maintain formation shape.

        Args:
            state: Current state
            dx, dy: Direction to move

        Returns:
            List of moves or None if move is not possible
        """
        # Check if moving the entire formation would cause collisions
        new_state = [(x + dx, y + dy) for x, y in state]

        # Check for obstacles or out of bounds
        for x, y in new_state:
            if not (0 <= x < self.n and 0 <= y < self.m) or (x, y) in self.obstacles:
                return None  # Movement not possible

        # Create moves for all blocks
        moves = []
        for i in range(len(state)):
            moves.append([(i, dx, dy)])

        return moves

    def _move_blocks_individually(self, state, dx, dy):
        """
        Try to move blocks individually in the given direction,
        prioritizing those that need to move.

        Args:
            state: Current state
            dx, dy: Direction to move

        Returns:
            List of moves or None if movement not possible
        """
        moves = []
        moved_positions = set()
        new_state = state.copy()

        # Try to move each block
        for i, (x, y) in enumerate(state):
            new_x, new_y = x + dx, y + dy

            # Check if new position is valid
            if (
                0 <= new_x < self.n
                and 0 <= new_y < self.m
                and (new_x, new_y) not in self.obstacles
                and (new_x, new_y) not in moved_positions
            ):

                # This block can move
                moves.append([(i, dx, dy)])
                moved_positions.add((new_x, new_y))
                new_state[i] = (new_x, new_y)

        return moves if moves else None

    def _add_intermediate_waypoints(self, path):
        """Add intermediate waypoints for smoother navigation."""
        if len(path) <= 2:
            return path

        result = [path[0]]

        # Add intermediate points between distant waypoints
        for i in range(1, len(path)):
            prev = path[i - 1]
            current = path[i]

            # Calculate distance
            dx = current[0] - prev[0]
            dy = current[1] - prev[1]
            distance = abs(dx) + abs(dy)  # Manhattan distance

            # If points are far apart, add intermediate points
            if distance > 6:
                # Number of points to add
                num_points = min(3, max(1, int(distance / 6)))

                for j in range(1, num_points + 1):
                    # Interpolate position
                    factor = j / (num_points + 1)
                    x = int(prev[0] + dx * factor)
                    y = int(prev[1] + dy * factor)
                    result.append((x, y))

            result.append(current)

        return result
