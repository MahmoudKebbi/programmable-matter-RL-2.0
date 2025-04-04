import torch
import numpy as np
import heapq
import time
from typing import List, Tuple, Optional, Dict, Set
from collections import deque

from grid import Grid
from ai_agent import AI_Agent, state_to_tuple, heuristic as original_heuristic
from ml_model import (
    ProgrammableMatterGNN,
    create_graph_from_state,
    direction_to_index,
    index_to_direction,
)


class HybridAgent:
    """
    Hybrid agent that combines ML predictions with A* search for programmable matter.

    The hybrid approach uses ML in several ways:
    1. ML-enhanced heuristic function for A* search
    2. Intelligent move prioritization based on GNN predictions
    3. Smart pruning of the search space
    """

    def __init__(
        self,
        n: int,
        m: int,
        start_state: List[Tuple[int, int]],
        target_state: List[Tuple[int, int]],
        model_path: str,
        ml_weight: float = 0.7,
        use_move_predictions: bool = True,
        smart_pruning: bool = True,
        verbose: bool = False,
    ):
        self.n = n
        self.m = m
        self.start_state = sorted(start_state)
        self.target_state = sorted(target_state)
        self.grid_dims = (n, m)
        self.ml_weight = ml_weight
        self.use_move_predictions = use_move_predictions
        self.smart_pruning = smart_pruning
        self.verbose = verbose

        # Load the ML model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model(model_path)

        # Statistics and diagnostics
        self.nodes_expanded = 0
        self.max_frontier_size = 0
        self.ml_suggestions_used = 0
        self.ml_suggestions_total = 0
        self.ml_time = 0
        self.search_time = 0

        # Initialize the search data structures
        self.visited = set()
        self.g_scores = {}

        # Store ML predictions cache for efficiency
        self.ml_predictions_cache = {}

    def _load_model(self, model_path: str):
        """Load the trained ML model."""
        try:
            # Determine feature dimension from the first state
            sample_graph = create_graph_from_state(
                self.start_state, self.target_state, self.grid_dims
            )
            node_features = sample_graph.x.shape[1]

            # Initialize the model with the correct dimensions
            self.model = ProgrammableMatterGNN(
                node_features=node_features,
                hidden_dim=64,
                output_dim=9,  # 8 directions + no move
            ).to(self.device)

            # Load the saved weights
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.eval()

            if self.verbose:
                print(f"ML model loaded from {model_path}")

        except Exception as e:
            print(f"Error loading model: {e}")
            print("Falling back to standard A* search")
            self.model = None

    def get_ml_predictions(self, state: List[Tuple[int, int]]):
        """
        Get ML model predictions for the current state.

        Returns:
            tuple: (heuristic_value, move_predictions, block_priority)
        """
        # Check if we have cached predictions for this state
        state_tuple = state_to_tuple(state)
        if state_tuple in self.ml_predictions_cache:
            return self.ml_predictions_cache[state_tuple]

        if self.model is None:
            # If model failed to load, return None
            return None, None, None

        start_time = time.time()

        try:
            # Create graph representation
            graph = create_graph_from_state(state, self.target_state, self.grid_dims)
            graph = graph.to(self.device)

            # Get model predictions
            with torch.no_grad():
                heuristic_pred, move_logits, block_priority = self.model(
                    graph.unsqueeze(0)
                )

            # Convert to more usable format
            heuristic_value = heuristic_pred.item()
            move_predictions = torch.softmax(move_logits, dim=1).cpu().numpy()
            block_priorities = torch.sigmoid(block_priority).cpu().numpy().flatten()

            # Cache the predictions
            self.ml_predictions_cache[state_tuple] = (
                heuristic_value,
                move_predictions,
                block_priorities,
            )

            self.ml_time += time.time() - start_time
            return heuristic_value, move_predictions, block_priorities

        except Exception as e:
            if self.verbose:
                print(f"Error during ML prediction: {e}")
            return None, None, None

    def hybrid_heuristic(self, state: List[Tuple[int, int]]) -> float:
        """
        Hybrid heuristic function that combines traditional heuristic with ML predictions.
        """
        # Calculate traditional heuristic value
        traditional_h = original_heuristic(state, self.target_state)

        # Get ML prediction
        ml_pred, _, _ = self.get_ml_predictions(state)

        if ml_pred is not None:
            # Combine the two heuristic values with the specified weight
            return (1.0 - self.ml_weight) * traditional_h + self.ml_weight * ml_pred
        else:
            # Fall back to traditional heuristic if ML prediction failed
            return traditional_h

    def get_successors_with_ml(self, state: List[Tuple[int, int]]):
        """Generate successors with guidance from ML model."""
        from ai_agent import get_successors as original_get_successors

        # Get all possible successors using the original function
        all_successors = original_get_successors(state, self.n, self.m)

        # Get ML move predictions
        _, move_predictions, block_priorities = self.get_ml_predictions(state)

        if not self.use_move_predictions or move_predictions is None:
            # If we're not using ML move predictions, return all successors
            return all_successors

        # Process the ML predictions to prioritize and prune successors
        successors_with_scores = []

        for succ_state, moves in all_successors:
            # Score this successor state based on ML predictions
            score = 0

            for block_idx, dx, dy in moves:
                # Find the direction index for this move
                dir_idx = direction_to_index(dx, dy)

                # Score is based on how likely the model thinks this move is
                if block_idx < len(move_predictions):
                    # Probability of this move from ML model
                    move_prob = move_predictions[block_idx][dir_idx]

                    # Higher probability = higher score
                    # Also consider block priority if available
                    block_priority = 1.0
                    if block_priorities is not None and block_idx < len(
                        block_priorities
                    ):
                        block_priority = block_priorities[block_idx]

                    score += move_prob * block_priority

            # If multiple blocks moved, average the score
            if moves:
                score /= len(moves)

            successors_with_scores.append((score, succ_state, moves))

        # Sort successors by score (highest first)
        successors_with_scores.sort(reverse=True)

        # Apply smart pruning if enabled
        if self.smart_pruning:
            # Keep only the top 50% of successors, but always keep at least 3
            keep_count = max(3, len(successors_with_scores) // 2)
            successors_with_scores = successors_with_scores[:keep_count]

        # Extract successors without scores
        prioritized_successors = [(s, m) for _, s, m in successors_with_scores]

        # Update stats
        self.ml_suggestions_total += 1
        if len(prioritized_successors) < len(all_successors):
            self.ml_suggestions_used += 1

        return prioritized_successors

    def plan(self) -> Optional[List[List[Tuple[int, int, int]]]]:
        """
        Find a sequence of moves to transform start_state into target_state
        using the hybrid ML + A* approach.
        """
        start_time = time.time()

        # Convert states to tuples for hashing
        start_tuple = state_to_tuple(self.start_state)
        target_tuple = state_to_tuple(self.target_state)

        # Quick check for already solved puzzle
        if start_tuple == target_tuple:
            return []

        # Initialize search data structures
        self.g_scores = {start_tuple: 0}
        frontier = []
        self.visited = set()

        # Compute initial heuristic
        h_val = self.hybrid_heuristic(self.start_state)

        # Create a tiebreaker based on blocks in wrong positions
        wrong_pos_count = sum(
            1 for a, b in zip(self.start_state, self.target_state) if a != b
        )

        # Push initial state: (f_value, tiebreaker, state_tuple, path so far)
        heapq.heappush(frontier, (h_val, wrong_pos_count, start_tuple, []))

        # Set up search limits
        max_iterations = min(200000, 20000 * len(self.start_state))
        depth_limit = max(30, int(h_val * 1.5))

        # Track best partial solution
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
                self.search_time = time.time() - start_time
                if self.verbose:
                    print(f"Solution found after {iterations} iterations")
                    print(
                        f"Search time: {self.search_time:.2f}s, ML time: {self.ml_time:.2f}s"
                    )
                    print(f"Nodes expanded: {self.nodes_expanded}")
                    print(
                        f"ML suggestions used: {self.ml_suggestions_used}/{self.ml_suggestions_total}"
                    )
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
            current_h = self.hybrid_heuristic(current_list)
            if current_h < best_h:
                best_h = current_h
                best_state = current_list
                best_path = path

            # Generate successors using ML-guided approach
            successors = self.get_successors_with_ml(current_list)

            for succ, moves in successors:
                succ_tuple = state_to_tuple(succ)

                if succ_tuple in self.visited:
                    continue

                new_cost = len(path) + 1
                if new_cost < self.g_scores.get(succ_tuple, float("inf")):
                    self.g_scores[succ_tuple] = new_cost

                    # Compute heuristic and f-value
                    h_val = self.hybrid_heuristic(succ)
                    f_val = new_cost + h_val

                    # Compute tiebreaker
                    wrong_pos_count = sum(
                        1 for a, b in zip(succ, self.target_state) if a != b
                    )

                    heapq.heappush(
                        frontier, (f_val, wrong_pos_count, succ_tuple, path + [moves])
                    )

        # Search ended without finding solution
        self.search_time = time.time() - start_time

        if self.verbose:
            print(f"Search exhausted after {iterations} iterations")
            print(f"Search time: {self.search_time:.2f}s, ML time: {self.ml_time:.2f}s")
            print(f"Nodes expanded: {self.nodes_expanded}")
            print(f"Max frontier size: {self.max_frontier_size}")

        # Return partial solution if we made progress
        if best_state and best_h < h_val:
            if self.verbose:
                print(f"No solution found, but reached state with heuristic {best_h}")
            return best_path

        return None


class AdaptiveHybridAgent(HybridAgent):
    """
    Extended hybrid agent that adapts ML usage based on puzzle complexity.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Adaptive parameters
        self.complexity_threshold = 15  # Estimated steps threshold
        self.ml_weight_easy = 0.5
        self.ml_weight_hard = 0.8
        self.pruning_easy = False
        self.pruning_hard = True

        # Estimate problem complexity
        self._assess_problem_complexity()

    def _assess_problem_complexity(self):
        """Assess problem complexity and adjust agent parameters."""
        # Get initial heuristic estimate
        initial_h = self.hybrid_heuristic(self.start_state)

        # Determine if this is a complex problem
        is_complex = initial_h > self.complexity_threshold

        # Adjust parameters based on complexity
        if is_complex:
            self.ml_weight = self.ml_weight_hard
            self.smart_pruning = self.pruning_hard
            if self.verbose:
                print(
                    f"Complex problem (h={initial_h:.2f}): Using ML weight {self.ml_weight} and pruning={self.smart_pruning}"
                )
        else:
            self.ml_weight = self.ml_weight_easy
            self.smart_pruning = self.pruning_easy
            if self.verbose:
                print(
                    f"Simple problem (h={initial_h:.2f}): Using ML weight {self.ml_weight} and pruning={self.smart_pruning}"
                )


# Example usage
if __name__ == "__main__":
    # Test puzzle
    n, m = 5, 5
    start_state = [(0, 0), (1, 0), (1, 1)]
    target_state = [(3, 3), (3, 4), (4, 3)]

    # Path to the trained model
    model_path = "../saved_models/best_model.pt"

    # Initialize hybrid agent
    agent = HybridAgent(
        n,
        m,
        start_state,
        target_state,
        model_path,
        ml_weight=0.7,
        use_move_predictions=True,
        smart_pruning=True,
        verbose=True,
    )

    # Find solution
    solution = agent.plan()

    if solution:
        print(f"Solution found with {len(solution)} moves:")

        # Apply the solution to visualize
        grid = Grid(n, m, start_state)
        print("Initial state:")
        grid.display_grid()

        for i, move_set in enumerate(solution):
            print(f"Move {i+1}: {move_set}")

            # Apply moves
            for block_idx, dx, dy in move_set:
                x, y = grid.matter_elements[block_idx]
                grid.grid[x, y] = 0
                grid.matter_elements[block_idx] = (x + dx, y + dy)
                grid.grid[x + dx, y + dy] = 1

            print(f"After move {i+1}:")
            grid.display_grid()
    else:
        print("No solution found")
