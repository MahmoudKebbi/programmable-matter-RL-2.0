import numpy as np
import torch
import time
import matplotlib.pyplot as plt
from rl_agent import PPOAgent
from extended_curriculum import ExtendedCurriculumManager


def main():
    # Create extended curriculum manager - start with larger grid
    curriculum = ExtendedCurriculumManager(initial_grid_size=12)

    # Get initial environment
    env = curriculum.get_environment()

    # Create agent
    agent = PPOAgent(
        env,
        actor_lr=0.0003,
        critic_lr=0.001,
        gamma=0.99,
        clip_ratio=0.2,
        hidden_size=384,
    )  # Larger network for complex problems

    # Training parameters
    num_episodes = 30000  # Increased for more comprehensive training
    max_steps = 150  # Allow more steps for complex tasks
    update_interval = 1024

    # For tracking progress
    all_rewards = []
    all_steps = []
    all_levels = []
    all_phases = []
    all_grid_sizes = []
    all_block_counts = []

    # Train agent
    print("Starting extended curriculum learning training...")
    start_time = time.time()

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        steps = 0
        done = False

        # Record current stats
        current_level = curriculum.get_level()
        current_phase = curriculum.get_phase()
        current_grid_size = curriculum.n
        current_block_count = len(env.blocks)

        # Save stats
        all_levels.append(current_level)
        all_phases.append(current_phase)
        all_grid_sizes.append(current_grid_size)
        all_block_counts.append(current_block_count)

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
                f"Episode {episode}/{num_episodes}, Level: {current_level}, Phase: {current_phase}"
            )
            print(
                f"  Grid: {current_grid_size}x{current_grid_size}, Blocks: {current_block_count}"
            )
            print(
                f"  Reward: {episode_reward:.2f}, Avg: {avg_reward:.2f}, Steps: {steps}"
            )
            print(
                f"  Successes: {curriculum.consecutive_successes}/{curriculum.required_successes}"
            )

        # Evaluate curriculum progress
        level_increased = curriculum.evaluate_progress(episode_reward, info)
        if level_increased:
            # Update environment reference in agent
            env = curriculum.get_environment()
            agent.env = env
            agent.adapt_to_env(env)

            # Save a checkpoint when advancing levels
            agent.save_model(f"level{current_level}_model")

            print(f"===== ADVANCING TO LEVEL {curriculum.get_level()} =====")

            # Adjust learning parameters based on phase transitions
            if current_phase != curriculum.get_phase():
                print(
                    f"===== PHASE TRANSITION: {current_phase} -> {curriculum.get_phase()} ====="
                )

                # For multi-component and master challenges, adjust learning
                if curriculum.get_phase() in ["multi_component", "master"]:
                    print("Adjusting learning parameters for advanced challenges...")

                    # Reset optimizer parameters for new phase
                    for param_group in agent.actor_optimizer.param_groups:
                        param_group["lr"] = (
                            0.0002  # Lower learning rate for fine tuning
                        )
                    for param_group in agent.critic_optimizer.param_groups:
                        param_group["lr"] = 0.0007

        # Save checkpoint
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
                        all_grid_sizes,
                        all_block_counts,
                    )
                ),
                delimiter=",",
                header="reward,steps,level,grid_size,block_count",
                comments="",
            )

            # Save progression plot
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
            plt.yticks(range(0, curriculum.max_level + 1, 5))
            plt.title("Curriculum Progression")

            # Plot phase transitions
            plt.subplot(3, 2, 4)
            phase_values = {
                "translation": 1,
                "transformation": 2,
                "transformation_with_translation": 3,
                "advanced": 4,
                "scaling_up": 5,
                "multi_component": 6,
                "master": 7,
            }
            phase_nums = [phase_values.get(p, 0) for p in all_phases]
            plt.plot(phase_nums)
            plt.xlabel("Episode")
            plt.ylabel("Training Phase")
            plt.yticks(
                list(phase_values.values()),
                [
                    "Translation",
                    "Transform",
                    "Transform+Move",
                    "Advanced",
                    "Scaling Up",
                    "Multi-component",
                    "Master",
                ],
            )
            plt.title("Training Phase Progression")

            # Plot grid size progression
            plt.subplot(3, 2, 5)
            plt.plot(all_grid_sizes)
            plt.xlabel("Episode")
            plt.ylabel("Grid Size")
            plt.title("Grid Size Progression")

            # Plot block count progression
            plt.subplot(3, 2, 6)
            plt.plot(all_block_counts)
            plt.xlabel("Episode")
            plt.ylabel("Block Count")
            plt.title("Block Count Progression")

            plt.tight_layout()
            plt.savefig(f"training_progress_ep{episode}.png")
            plt.close()

    # Training complete
    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Training completed in {int(hours)}h {int(minutes)}m {seconds:.2f}s")

    # Save final model
    agent.save_model("final_model")

    print("Training complete!")


if __name__ == "__main__":
    main()
