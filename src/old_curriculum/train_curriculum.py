import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from rl_agent import PPOAgent
from curriculum import CurriculumManager


def main():
    # Create curriculum manager
    curriculum = CurriculumManager()

    # Get initial environment
    env = curriculum.get_environment()

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
    all_levels = []

    # Train agent
    print("Starting curriculum learning training...")
    start_time = time.time()

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0
        done = False

        # Record current level
        current_level = curriculum.get_level()
        all_levels.append(current_level)

        while not done and steps < max_steps:
            # Select action
            action, log_prob = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            # Store transition
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

        # Print progress
        if episode % 10 == 0:
            avg_reward = sum(all_rewards[-10:]) / min(10, len(all_rewards))
            print(
                f"Episode {episode}/{num_episodes}, Level: {current_level}, "
                f"Reward: {episode_reward:.2f}, Avg Reward: {avg_reward:.2f}, "
                f"Steps: {steps}, Successes: {curriculum.consecutive_successes}/{curriculum.required_successes}"
            )

        # Evaluate curriculum progress
        level_increased = curriculum.evaluate_progress(episode_reward, info)
        if level_increased:
            # Update environment reference in agent
            env = curriculum.get_environment()
            agent.env = env

            # Save a checkpoint when advancing levels
            agent.save_model(f"level{current_level}_model")

            print(f"===== ADVANCING TO LEVEL {curriculum.get_level()} =====")

            # Optionally: adjust learning parameters for new level
            if current_level == 0:  # Just completed easiest level
                # Reduce learning rate for finer tuning as tasks get harder
                for param_group in agent.actor_optimizer.param_groups:
                    param_group["lr"] *= 0.7
                for param_group in agent.critic_optimizer.param_groups:
                    param_group["lr"] *= 0.7

        # Save checkpoint
        if episode % 200 == 0 and episode > 0:
            agent.save_model(f"model_checkpoint_ep{episode}")

            # Save curriculum stats as CSV for later analysis
            np.savetxt(
                f"training_stats_ep{episode}.csv",
                np.column_stack((all_rewards, all_steps, all_levels)),
                delimiter=",",
                header="reward,steps,level",
                comments="",
            )

    # Training complete
    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time:.2f} seconds")

    # Save final model
    agent.save_model("final_model")

    # Generate final stats plot (just once at the end)
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(all_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode Rewards")

    plt.subplot(1, 3, 2)
    plt.plot(all_steps)
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Episode Length")

    plt.subplot(1, 3, 3)
    plt.plot(all_levels)
    plt.xlabel("Episode")
    plt.ylabel("Curriculum Level")
    plt.yticks([0, 1, 2, 3])
    plt.title("Curriculum Progression")

    plt.tight_layout()
    plt.savefig("final_training_summary.png")
    print("Saved final training summary plot to final_training_summary.png")

    print("Training complete!")


if __name__ == "__main__":
    main()
