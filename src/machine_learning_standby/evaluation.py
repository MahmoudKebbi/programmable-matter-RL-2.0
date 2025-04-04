import time
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import argparse

from grid import Grid
from ai_agent import AI_Agent
from hybrid_agent import HybridAgent, AdaptiveHybridAgent


class AgentEvaluator:
    """
    System for evaluating and comparing different programmable matter agents.
    """

    def __init__(self, model_path, output_dir="evaluation_results"):
        self.model_path = model_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Statistics trackers
        self.results = defaultdict(list)

    def generate_test_cases(
        self,
        num_cases=50,
        grid_sizes=[(5, 5), (7, 7), (10, 10)],
        block_counts=[3, 4, 5, 7, 10],
    ):
        """Generate a diverse set of test cases."""
        test_cases = []

        for n, m in grid_sizes:
            for num_blocks in block_counts:
                if num_blocks > n * m // 2:  # Skip if too many blocks for grid
                    continue

                # Generate several cases for each configuration
                cases_per_config = max(
                    1, num_cases // (len(grid_sizes) * len(block_counts))
                )

                for _ in range(cases_per_config):
                    # Generate start state
                    start_state = self._generate_connected_state(n, m, num_blocks)

                    # Generate target state
                    target_state = self._generate_connected_state(n, m, num_blocks)

                    # Store test case
                    test_cases.append(
                        {
                            "grid_size": (n, m),
                            "start_state": start_state,
                            "target_state": target_state,
                        }
                    )

        print(f"Generated {len(test_cases)} test cases")
        return test_cases

    def _generate_connected_state(self, n, m, num_blocks):
        """Generate a randomly connected state with the given number of blocks."""
        import random

        state = []
        grid = np.zeros((n, m))

        # Start with a random position
        x, y = random.randint(0, n - 1), random.randint(0, m - 1)
        state.append((x, y))
        grid[x, y] = 1

        # Add remaining blocks ensuring connectivity
        block_count = 1
        while block_count < num_blocks:
            # Choose a random existing block
            parent_idx = random.randint(0, len(state) - 1)
            px, py = state[parent_idx]

            # Try to add a neighboring block
            directions = [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ]
            random.shuffle(directions)

            for dx, dy in directions:
                nx, ny = px + dx, py + dy
                if 0 <= nx < n and 0 <= ny < m and grid[nx, ny] == 0:
                    state.append((nx, ny))
                    grid[nx, ny] = 1
                    block_count += 1
                    break
            else:
                # If we couldn't add a neighbor, try another parent
                continue

            if block_count >= num_blocks:
                break

        return sorted(state)

    def run_evaluation(self, test_cases, time_limit=60, parallel=True):
        """
        Run evaluation on all test cases using different agent implementations.

        Args:
            test_cases: List of test case dictionaries
            time_limit: Maximum time in seconds allowed for each agent on a test case
            parallel: Whether to run evaluations in parallel
        """
        if parallel:
            with ProcessPoolExecutor() as executor:
                # Map each test case to a future
                futures = [
                    executor.submit(self._evaluate_single_case, case, i, time_limit)
                    for i, case in enumerate(test_cases)
                ]

                # Collect results as they complete
                for future in tqdm(futures, total=len(futures), desc="Evaluating"):
                    result = future.result()
                    if result:
                        for agent_type, metrics in result.items():
                            for metric, value in metrics.items():
                                self.results[f"{agent_type}_{metric}"].append(value)
        else:
            # Run sequentially
            for i, case in enumerate(tqdm(test_cases, desc="Evaluating")):
                result = self._evaluate_single_case(case, i, time_limit)
                if result:
                    for agent_type, metrics in result.items():
                        for metric, value in metrics.items():
                            self.results[f"{agent_type}_{metric}"].append(value)

        # Summarize results
        self._summarize_results()

        return self.results

    def _evaluate_single_case(self, case, case_index, time_limit):
        """Evaluate a single test case with multiple agents."""
        n, m = case["grid_size"]
        start_state = case["start_state"]
        target_state = case["target_state"]

        results = {}

        # Define agent configurations to test
        agent_configs = [
            {"name": "astar", "class": AI_Agent, "kwargs": {}},
            {
                "name": "hybrid",
                "class": HybridAgent,
                "kwargs": {
                    "model_path": self.model_path,
                    "ml_weight": 0.7,
                    "use_move_predictions": True,
                    "smart_pruning": True,
                    "verbose": False,
                },
            },
            {
                "name": "adaptive",
                "class": AdaptiveHybridAgent,
                "kwargs": {"model_path": self.model_path, "verbose": False},
            },
        ]

        # Test each agent
        for config in agent_configs:
            agent_name = config["name"]
            agent_class = config["class"]

            try:
                # Initialize agent
                agent = agent_class(
                    n, m, start_state.copy(), target_state.copy(), **config["kwargs"]
                )

                # Set timeout for this run
                start_time = time.time()

                # Run planning
                solution = agent.plan()

                # Record execution time
                execution_time = time.time() - start_time

                # Check for timeout
                if execution_time > time_limit:
                    results[agent_name] = {
                        "solved": False,
                        "solution_length": float("inf"),
                        "execution_time": time_limit,
                        "nodes_expanded": agent.nodes_expanded,
                        "max_frontier_size": agent.max_frontier_size,
                        "grid_size": n * m,
                        "num_blocks": len(start_state),
                        "complexity": len(start_state) * (n + m),
                    }
                    continue

                # Record results
                results[agent_name] = {
                    "solved": solution is not None,
                    "solution_length": len(solution) if solution else float("inf"),
                    "execution_time": execution_time,
                    "nodes_expanded": agent.nodes_expanded,
                    "max_frontier_size": agent.max_frontier_size,
                    "grid_size": n * m,
                    "num_blocks": len(start_state),
                    "complexity": len(start_state) * (n + m),
                }

                # For hybrid agents, record ML-specific metrics
                if hasattr(agent, "ml_time"):
                    results[agent_name]["ml_time"] = agent.ml_time
                    results[agent_name]["search_time"] = agent.search_time
                    results[agent_name][
                        "ml_suggestions_used"
                    ] = agent.ml_suggestions_used
                    results[agent_name][
                        "ml_suggestions_total"
                    ] = agent.ml_suggestions_total

                # Save the solution for this test case
                if solution:
                    case_dir = os.path.join(self.output_dir, f"case_{case_index}")
                    os.makedirs(case_dir, exist_ok=True)
                    with open(
                        os.path.join(case_dir, f"{agent_name}_solution.json"), "w"
                    ) as f:
                        # Convert moves to JSON serializable format
                        solution_json = [
                            [list(move) for move in step] for step in solution
                        ]
                        json.dump(
                            {
                                "grid_size": list(case["grid_size"]),
                                "start_state": [
                                    list(pos) for pos in case["start_state"]
                                ],
                                "target_state": [
                                    list(pos) for pos in case["target_state"]
                                ],
                                "solution": solution_json,
                                "metrics": results[agent_name],
                            },
                            f,
                            indent=2,
                        )

            except Exception as e:
                print(f"Error evaluating {agent_name} on case {case_index}: {e}")
                results[agent_name] = {
                    "solved": False,
                    "error": str(e),
                    "grid_size": n * m,
                    "num_blocks": len(start_state),
                }

        return results

    def _summarize_results(self):
        """Create summary statistics and visualizations from results."""
        # Convert results to DataFrame
        results_df = pd.DataFrame()

        # Extract agent types and metrics
        agent_types = set()
        metrics = set()
        for key in self.results.keys():
            agent_type, metric = key.split("_", 1)
            agent_types.add(agent_type)
            metrics.add(metric)

        # Create a DataFrame from results
        data = {}
        for agent_type in agent_types:
            for metric in metrics:
                key = f"{agent_type}_{metric}"
                if key in self.results:
                    data[key] = self.results[key]

        results_df = pd.DataFrame(data)

        # Save results to CSV
        results_df.to_csv(os.path.join(self.output_dir, "results.csv"))

        # Create summary statistics
        summary = {}
        for agent_type in agent_types:
            summary[agent_type] = {
                "solved_rate": np.mean(
                    [
                        1 if x else 0
                        for x in self.results.get(f"{agent_type}_solved", [0])
                    ]
                ),
                "avg_solution_length": np.mean(
                    [
                        x
                        for x in self.results.get(f"{agent_type}_solution_length", [0])
                        if x != float("inf")
                    ]
                ),
                "avg_execution_time": np.mean(
                    self.results.get(f"{agent_type}_execution_time", [0])
                ),
                "avg_nodes_expanded": np.mean(
                    self.results.get(f"{agent_type}_nodes_expanded", [0])
                ),
                "avg_frontier_size": np.mean(
                    self.results.get(f"{agent_type}_max_frontier_size", [0])
                ),
            }

            # ML-specific metrics
            if f"{agent_type}_ml_time" in self.results:
                ml_time = self.results[f"{agent_type}_ml_time"]
                search_time = self.results[f"{agent_type}_search_time"]
                summary[agent_type]["avg_ml_time"] = np.mean(ml_time)
                summary[agent_type]["avg_search_time"] = np.mean(search_time)
                summary[agent_type]["avg_ml_time_ratio"] = np.mean(
                    [m / max(1e-6, m + s) for m, s in zip(ml_time, search_time)]
                )

        # Save summary to JSON
        with open(os.path.join(self.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        # Print summary
        print("\nEvaluation Summary:")
        for agent_type, metrics in summary.items():
            print(f"\n{agent_type.upper()} Agent:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")

        # Create visualizations
        self._generate_visualizations()

    def _generate_visualizations(self):
        """Generate visualizations from evaluation results."""
        # Prepare data
        agent_types = list({key.split("_")[0] for key in self.results.keys()})

        # 1. Success rate by grid size
        try:
            plt.figure(figsize=(10, 6))
            for agent in agent_types:
                solved = np.array(self.results.get(f"{agent}_solved", []))
                grid_sizes = np.array(self.results.get(f"{agent}_grid_size", []))

                # Group by grid size
                unique_sizes = np.unique(grid_sizes)
                success_rates = []

                for size in unique_sizes:
                    mask = grid_sizes == size
                    success_rate = np.mean(solved[mask])
                    success_rates.append(success_rate)

                plt.plot(unique_sizes, success_rates, marker="o", label=agent)

            plt.xlabel("Grid Size (n×m)")
            plt.ylabel("Success Rate")
            plt.title("Success Rate by Grid Size")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir, "success_rate_by_grid_size.png"))
            plt.close()
        except Exception as e:
            print(f"Error generating success rate visualization: {e}")

        # 2. Execution time by problem complexity
        try:
            plt.figure(figsize=(10, 6))
            for agent in agent_types:
                times = np.array(self.results.get(f"{agent}_execution_time", []))
                complexity = np.array(self.results.get(f"{agent}_complexity", []))
                solved = np.array(self.results.get(f"{agent}_solved", []))

                # Only plot cases where the agent solved the problem
                mask = solved == True
                plt.scatter(complexity[mask], times[mask], alpha=0.7, label=agent)

                # Add trend line
                if np.sum(mask) > 1:
                    z = np.polyfit(complexity[mask], times[mask], 1)
                    p = np.poly1d(z)
                    plt.plot(complexity[mask], p(complexity[mask]), "--", alpha=0.5)

            plt.xlabel("Problem Complexity (blocks × (rows + columns))")
            plt.ylabel("Execution Time (s)")
            plt.title("Execution Time vs Problem Complexity")
            plt.legend()
            plt.grid(True)
            plt.savefig(
                os.path.join(self.output_dir, "execution_time_by_complexity.png")
            )
            plt.close()
        except Exception as e:
            print(f"Error generating execution time visualization: {e}")

        # 3. Nodes Expanded Comparison
        try:
            plt.figure(figsize=(10, 6))
            data = []
            labels = []

            for agent in agent_types:
                nodes = self.results.get(f"{agent}_nodes_expanded", [])
                if nodes:
                    data.append(nodes)
                    labels.append(agent)

            if data:
                plt.boxplot(data, labels=labels)
                plt.ylabel("Nodes Expanded")
                plt.title("Search Efficiency Comparison")
                plt.grid(True, axis="y")
                plt.savefig(
                    os.path.join(self.output_dir, "nodes_expanded_comparison.png")
                )
            plt.close()
        except Exception as e:
            print(f"Error generating nodes expanded visualization: {e}")

        # 4. Solution Length Comparison
        try:
            plt.figure(figsize=(10, 6))

            solution_lengths = {}
            for agent in agent_types:
                lengths = np.array(self.results.get(f"{agent}_solution_length", []))
                solved = np.array(self.results.get(f"{agent}_solved", []))

                # Only include cases where the agent found a solution
                lengths = [
                    l for l, s in zip(lengths, solved) if s and l != float("inf")
                ]

                if lengths:
                    solution_lengths[agent] = lengths

            if solution_lengths:
                plt.boxplot(
                    [
                        solution_lengths[agent]
                        for agent in agent_types
                        if agent in solution_lengths
                    ],
                    labels=[
                        agent for agent in agent_types if agent in solution_lengths
                    ],
                )
                plt.ylabel("Solution Length")
                plt.title("Solution Quality Comparison")
                plt.grid(True, axis="y")
                plt.savefig(
                    os.path.join(self.output_dir, "solution_length_comparison.png")
                )
            plt.close()
        except Exception as e:
            print(f"Error generating solution length visualization: {e}")


def main():
    """Main function for running evaluations from command line."""
    parser = argparse.ArgumentParser(description="Evaluate programmable matter agents")
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the trained ML model"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--num-cases", type=int, default=50, help="Number of test cases to generate"
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        default=60,
        help="Time limit in seconds for each agent on a test case",
    )
    parser.add_argument(
        "--no-parallel", action="store_true", help="Disable parallel evaluation"
    )

    args = parser.parse_args()

    # Create evaluator
    evaluator = AgentEvaluator(args.model_path, args.output_dir)

    # Generate test cases
    test_cases = evaluator.generate_test_cases(num_cases=args.num_cases)

    # Run evaluation
    evaluator.run_evaluation(
        test_cases, time_limit=args.time_limit, parallel=not args.no_parallel
    )


if __name__ == "__main__":
    main()
