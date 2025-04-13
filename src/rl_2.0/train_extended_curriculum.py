import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import matplotlib.pyplot as plt
from rl_agent import PPOAgent
from extended_curriculum import (
    UltraGradualCurriculumManager,
    shape_micro_level_reward,
)


def main():
    # Create ultra-gradual curriculum manager
    curriculum = UltraGradualCurriculumManager(initial_grid_size=12)

    # Get initial environment
    env = curriculum.get_environment()

    # Create agent with improved parameters
    agent = PPOAgent(
        env,
        actor_lr=0.0003,
        critic_lr=0.001,
        gamma=0.99,
        clip_ratio=0.2,
        hidden_size=256,
    )  # Larger network for better generalization

    # Training parameters
    num_episodes = 30000
    max_steps = 150  # Maximum steps per episode
    update_interval = 1024

    # For tracking progress
    all_rewards = []
    all_steps = []
    all_levels = []
    all_phases = []
    all_block_counts = []
    all_stuck_episodes = []

    # For tracking best models per level
    best_rewards_per_level = {}

    # Train agent
    print("Starting ultra-gradual curriculum learning with anti-stuck mechanisms...")
    start_time = time.time()

    for episode in range(num_episodes):
        # Emergency skip for persistently stuck micro-levels
        if hasattr(curriculum, "micro_level") and curriculum.micro_level is not None:
            if getattr(curriculum, "micro_attempts", 0) > 100:
                curriculum.emergency_skip_micro_level()
                env = curriculum.get_environment()
                agent.adapt_to_env(env)
                agent.reset_network_for_micro_level()  # Complete reset (implement this method)
                continue

        state = env.reset()
        episode_reward = 0
        steps = 0
        done = False

        # Record current stats
        current_level = curriculum.get_level()
        current_phase = curriculum.get_phase()
        current_block_count = len(env.blocks)
        stuck_episodes = curriculum.stuck_episodes

        # Save stats
        all_levels.append(current_level)
        all_phases.append(current_phase)
        all_block_counts.append(current_block_count)
        all_stuck_episodes.append(stuck_episodes)

        # Track agent's current level
        agent.current_level = current_level

        # Check if stuck and apply network resets if needed
        if stuck_episodes > 0:
            agent.reset_network_when_stuck(current_level, stuck_episodes)
            agent.apply_learning_rate_cycling(stuck_episodes)

        while not done and steps < max_steps:
            # Determine appropriate action selection method based on current state

            # For micro-levels with high stuck count, use very high temperature
            if (
                hasattr(curriculum, "micro_level")
                and curriculum.micro_level is not None
                and stuck_episodes > 0
            ):
                temperature = 5.0  # Much higher temperature for stuck micro-levels
                action, log_prob = agent.select_action_with_temp(state, temperature)
                if steps == 0 and stuck_episodes % 5 == 0:
                    print(
                        f"🔍 Using extreme exploration (temp=5.0) for stuck micro-level {current_level}"
                    )

            # For regular stuck levels, use dynamic temperature
            elif stuck_episodes > 0:
                # Get dynamic temperature based on how long we've been stuck
                temperature = agent.get_exploration_temperature(
                    current_level, stuck_episodes
                )
                action, log_prob = agent.select_action_with_temp(state, temperature)

                # First step of episode - print once every few episodes
                if steps == 0 and stuck_episodes % 10 == 0:
                    print(
                        f"🔍 Using boosted exploration (temp={temperature:.1f}) for stuck level {current_level}"
                    )
            else:
                # Normal action selection for unstuck levels
                action, log_prob = agent.select_action(state)

            next_state, reward, done, info = env.step(action)

            # Apply reward shaping for micro-levels
            if (
                hasattr(curriculum, "micro_level")
                and curriculum.micro_level is not None
            ):
                shaped_reward = shape_micro_level_reward(
                    reward, state, next_state, info
                )
            else:
                shaped_reward = reward

            # Store transition
            agent.store_transition(
                state, action, log_prob, shaped_reward, done, next_state
            )

            state = next_state
            episode_reward += reward  # Use original reward for tracking
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
        agent.episode_rewards.append(episode_reward)

        # Print progress
        if episode % 10 == 0:
            avg_reward = sum(all_rewards[-10:]) / min(10, len(all_rewards))
            print(
                f"Episode {episode}/{num_episodes}, Level: {current_level}, Phase: {current_phase}"
            )
            print(
                f"  Grid: {curriculum.n}x{curriculum.m}, Blocks: {current_block_count}"
            )
            print(
                f"  Reward: {episode_reward:.2f}, Avg: {avg_reward:.2f}, Steps: {steps}"
            )
            print(
                f"  Successes: {curriculum.consecutive_successes}/{curriculum.required_successes}"
                + (f", Stuck: {stuck_episodes}" if stuck_episodes > 0 else "")
            )

        # Save best model for current level
        if (
            current_level not in best_rewards_per_level
            or episode_reward > best_rewards_per_level[current_level]
        ):
            best_rewards_per_level[current_level] = episode_reward
            agent.save_model(f"best_level{current_level}_model")

        # Evaluate curriculum progress
        level_increased = curriculum.evaluate_progress(episode_reward, info)
        if level_increased:
            # Update environment reference in agent
            env = curriculum.get_environment()
            agent.adapt_to_env(env)

            # Save a checkpoint when advancing levels
            agent.save_model(f"checkpoint_level{current_level}_model")

            print(f"===== ADVANCING TO LEVEL {curriculum.get_level()} =====")

            # Reset exploration when advancing to a new level
            agent.reset_exploration()

        # Save checkpoint periodically
        if episode % 500 == 0 and episode > 0:
            agent.save_model(f"model_checkpoint_ep{episode}")

            # Save stats
            np.savetxt(
                f"training_stats_ep{episode}.csv",
                np.column_stack(
                    (
                        all_rewards,
                        all_steps,
                        all_levels,
                        all_block_counts,
                        all_stuck_episodes,
                    )
                ),
                delimiter=",",
                header="reward,steps,level,block_count,stuck_episodes",
                comments="",
            )

            # Plot progress
            plot_training_progress(
                all_rewards,
                all_steps,
                all_levels,
                all_block_counts,
                all_stuck_episodes,
                curriculum.max_level,
                episode,
            )

    # Training complete
    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Training completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")

    # Save final model
    agent.save_model("final_model")

    print("Training complete!")


def plot_training_progress(
    all_rewards,
    all_steps,
    all_levels,
    all_block_counts,
    all_stuck_episodes,
    max_level,
    episode,
):
    plt.figure(figsize=(15, 12))

    plt.subplot(3, 2, 1)
    plt.plot(all_rewards[-1000:])  # Show recent rewards
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Recent Episode Rewards")

    plt.subplot(3, 2, 2)
    plt.plot(all_steps[-1000:])  # Show recent steps
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title("Recent Episode Length")

    plt.subplot(3, 2, 3)
    plt.plot(all_levels)
    plt.xlabel("Episode")
    plt.ylabel("Curriculum Level")
    plt.yticks(range(0, max_level + 1, 5))
    plt.title("Curriculum Progression")

    plt.subplot(3, 2, 4)
    plt.plot(all_block_counts)
    plt.xlabel("Episode")
    plt.ylabel("Block Count")
    plt.title("Block Count Progression")

    plt.subplot(3, 2, 5)
    plt.plot(all_stuck_episodes)
    plt.xlabel("Episode")
    plt.ylabel("Stuck Episodes")
    plt.title("Stuck Episode Counter")

    # Histogram of rewards
    plt.subplot(3, 2, 6)
    plt.hist(all_rewards[-1000:], bins=20)
    plt.xlabel("Reward")
    plt.ylabel("Frequency")
    plt.title("Reward Distribution (Recent)")

    plt.tight_layout()
    plt.savefig(f"training_progress_ep{episode}.png")
    plt.close()


if __name__ == "__main__":
    main()
