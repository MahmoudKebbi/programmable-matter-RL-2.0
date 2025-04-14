import time
from puzzle_solver import AI_Agent


def main():
    # Define puzzle parameters
    n, m = 10, 10  # Grid size

    # Simple example: move a 2x2 square from one corner to another
    start_state = [(1, 1), (1, 2), (2, 1), (2, 2)]
    target_state = [(7, 7), (7, 8), (8, 7), (8, 8)]

    # Add some obstacles in the middle
    obstacles = [(5, 5), (5, 6), (6, 5), (6, 6)]

    print("Creating AI agent...")
    agent = AI_Agent(n, m, start_state, target_state, obstacles)

    # Try direct A* search
    print("\n=== Running A* search ===")
    start_time = time.time()
    solution = agent.plan()
    end_time = time.time()

    if solution:
        print(f"Solution found with {len(solution)} moves")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
        print(f"Nodes expanded: {agent.nodes_expanded}")

        # Print a few moves
        for i, move_set in enumerate(solution[:5]):
            print(f"Move {i+1}: {move_set}")
        if len(solution) > 5:
            print(f"... ({len(solution)-5} more moves)")
    else:
        print("No solution found with direct A* search")

        # Try hierarchical planning
        print("\n=== Trying hierarchical planning ===")
        start_time = time.time()
        solution, waypoints = agent.hierarchical_plan()
        end_time = time.time()

        if solution:
            print(f"Hierarchical solution found with {len(solution)} moves")
            print(f"Time taken: {end_time - start_time:.2f} seconds")
            print(f"Waypoints: {len(waypoints)}")
        else:
            print("\n=== Trying parallel planning as last resort ===")
            solution = agent.plan_parallel()

            if solution:
                print(f"Parallel solution found with {len(solution)} moves")
            else:
                print("Failed to find any solution")


if __name__ == "__main__":
    main()
