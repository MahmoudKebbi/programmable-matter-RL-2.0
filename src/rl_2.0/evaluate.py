import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from environment import ProgrammableMatterEnv
from rl_agent import PPOAgent


def evaluate_agent(env, agent, num_episodes=10, render=True):
    """Evaluate agent performance"""
    total_rewards = []
    total_steps = []
    solved = 0

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0
        done = False

        while not done:
            # Select action without exploration
            action, _ = agent.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)

            if render and episode == 0:
                env.render()

            state = next_state
            episode_reward += reward
            steps += 1

            if steps >= 200:  # Prevent infinite loops
                break

        if info["distance"] == 0:
            solved += 1

        total_rewards.append(episode_reward)
        total_steps.append(steps)

        print(
            f"Episode {episode+1}: Reward = {episode_reward:.2f}, Steps = {steps}, Solved = {info['distance'] == 0}"
        )

    success_rate = solved / num_episodes
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(total_steps)

    print(f"\nEvaluation Results:")
    print(f"Success Rate: {success_rate:.2f}")
    print(f"Average Reward: {avg_reward:.2f}")
    print(f"Average Steps: {avg_steps:.2f}")

    return success_rate, avg_reward, avg_steps


def visualize_trajectory(env, agent, save_path="trajectory.gif"):
    """Generate animation of agent solving the problem"""
    state = env.reset()
    done = False
    frames = []
    states = []
    actions = []

    # Collect trajectory
    while not done:
        frames.append(env.render(mode="rgb_array"))
        states.append(state.copy())

        action, _ = agent.select_action(state, training=False)
        actions.append(action)

        state, _, done, info = env.step(action)

        if len(frames) > 100:  # Prevent infinite loops
            break

    # Final state
    frames.append(env.render(mode="rgb_array"))

    # Create animation
    fig, ax = plt.subplots(figsize=(8, 8))

    def update(frame_idx):
        ax.clear()
        img = ax.imshow(frames[frame_idx], cmap="viridis")

        # Show action info if available
        if frame_idx < len(actions):
            action = actions[frame_idx]
            ax.set_title(
                f"Step {frame_idx}: Move block {action['block_id']} in direction {action['direction']}"
            )
        else:
            ax.set_title("Final state")

        ax.set_axis_off()
        return [img]

    ani = FuncAnimation(fig, update, frames=len(frames), interval=500)
    ani.save(save_path, writer="pillow", fps=2)
    plt.close()

    return states, actions, frames


def main():
    # Load the same environment as in training
    n, m = 10, 10
    initial_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    target_positions = [(8, 8), (8, 9), (9, 8), (9, 9)]
    obstacles = [(3, 3), (3, 4), (3, 5), (3, 6), (4, 3), (5, 3), (6, 3)]

    env = ProgrammableMatterEnv(n, m, initial_positions, target_positions, obstacles)

    # Create agent and load trained model
    agent = PPOAgent(env)
    agent.load_model("best_model")

    # Evaluate and visualize
    evaluate_agent(env, agent, num_episodes=5)
    visualize_trajectory(env, agent)

    # Test on a new configuration
    print("\nTesting on a new configuration:")
    new_initial = [(2, 2), (2, 3), (3, 2)]
    new_target = [(7, 7), (7, 8), (8, 7)]

    new_env = ProgrammableMatterEnv(n, m, new_initial, new_target, obstacles)
    evaluate_agent(new_env, agent, num_episodes=3)
    visualize_trajectory(new_env, agent, save_path="new_trajectory.gif")


if __name__ == "__main__":
    main()
