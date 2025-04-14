import heapq
import numpy as np
import time
import multiprocessing as mp
from copy import deepcopy

# Import optimized Cython functions
from src.puzzle_cython import (
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


class AI_Agent:
    def __init__(
        self,
        n: int,
        m: int,
        start_state: list,
        target_state: list,
        obstacles: list = None,
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

    def plan(self, max_iters=None, depth_lim=None):
        """
        Find a sequence of moves to transform start_state into target_state, avoiding obstacles.

        Args:
            max_iters: Maximum number of iterations (states to explore)
            depth_lim: Maximum depth (moves) to explore

        Returns:
            List of moves or None if no solution found
        """
        print("Running Cython-optimized A* planning...")

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

    def hierarchical_plan(self):
        """
        Hierarchical planning approach:
        1. Create an abstract path ignoring connectivity
        2. Generate waypoints along this path
        3. Solve subproblems between waypoints

        Returns:
            Tuple of (complete plan of moves, list of waypoint states)
        """
        print("Starting hierarchical planning with Cython acceleration...")

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
            for dx, dy in [
                (0, 1),
                (1, 0),
                (0, -1),
                (-1, 0),
                (1, 1),
                (1, -1),
                (-1, 1),
                (-1, -1),
            ]:
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

        print(
            f"Starting parallel planning with {num_processes} processes using Cython..."
        )
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
        print("Starting parallel hierarchical planning with Cython...")

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
