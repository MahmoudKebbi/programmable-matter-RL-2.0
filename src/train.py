import numpy as np
from environment import ProgrammableMatterEnv
from rl_agent import PPOAgent
import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import torch


def plot_training_progress(rewards, avg_rewards, steps):
    """Plot training progress"""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rewards, label="Episode Reward")
    plt.plot(avg_rewards, label="Avg Reward (100 ep)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.title("Training Rewards")

    plt.subplot(1, 2, 2)
    plt.plot(steps)
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Episode Length")

    plt.tight_layout()
    plt.savefig("training_progress.png")
    plt.close()


def visualize_solution(env, agent, save_path="solution.gif"):
    """Generate animation of the trained agent solving the problem"""
    state = env.reset()
    done = False
    frames = [env.render(mode="rgb_array")]

    while not done:
        action, _ = agent.select_action(state, training=False)
        state, _, done, _ = env.step(action)
        frames.append(env.render(mode="rgb_array"))

    # Create animation
    fig, ax = plt.subplots()

    def update(frame):
        ax.clear()
        ax.imshow(frame)
        ax.set_axis_off()
        return [ax]

    ani = FuncAnimation(fig, update, frames=frames, blit=True)
    ani.save(save_path, writer="pillow", fps=5)
    plt.close()


def main():
    # Environment parameters
    n, m = 10, 10  # Grid size

    # Example configuration
    initial_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]  # 2x2 square
    target_positions = [(8, 8), (8, 9), (9, 8), (9, 9)]  # Move to bottom-right

    # Add some obstacles
    obstacles = [(3, 3), (3, 4), (3, 5), (3, 6), (4, 3), (5, 3), (6, 3)]

    # Create environment
    env = ProgrammableMatterEnv(n, m, initial_positions, target_positions, obstacles)

    # Create agent
    agent = PPOAgent(
        env,
        actor_lr=0.0003,
        critic_lr=0.001,
        gamma=0.99,
        clip_ratio=0.2,
        hidden_size=256,
    )

    # Training parameters
    num_episodes = 5000
    max_steps = 200
    update_interval = 1024

    # For tracking progress
    all_rewards = []
    all_steps = []
    avg_rewards = []

    # Train agent
    print("Starting training...")
    start_time = time.time()

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0
        done = False

        while not done and steps < max_steps:
            action, log_prob = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.store_transition(state, action, log_prob, reward, done, next_state)

            state = next_state
            episode_reward += reward
            steps += 1

            # Update policy periodically
            if (steps + episode * max_steps) % update_interval == 0:
                agent.update_policy()

        # Update with remaining data
        if len(agent.states) > 0:
            agent.update_policy()

        # Track progress
        all_rewards.append(episode_reward)
        all_steps.append(steps)

        # Calculate running average
        if episode >= 99:
            avg_rewards.append(sum(all_rewards[-100:]) / 100)
        else:
            avg_rewards.append(sum(all_rewards) / (episode + 1))

        # Print progress
        if episode % 20 == 0:
            print(
                f"Episode {episode}/{num_episodes}, Reward: {episode_reward:.2f}, Steps: {steps}, Avg Reward: {avg_rewards[-1]:.2f}"
            )

            # Plot progress
            if episode % 100 == 0:
                plot_training_progress(all_rewards, avg_rewards, all_steps)
                agent.save_model(f"model_checkpoint_ep{episode}")

    # Training complete
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds")

    # Save final model
    agent.save_model("final_model")

    # Plot final progress
    plot_training_progress(all_rewards, avg_rewards, all_steps)

    # Generate visualization of solution
    visualize_solution(env, agent)

    print("Training complete!")


if __name__ == "__main__":
    main()
