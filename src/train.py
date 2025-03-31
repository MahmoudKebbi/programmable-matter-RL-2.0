import os
import sys

# REVIEW: More robust path handling
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # Add current dir
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)  # Add parent dir

import numpy as np
import matplotlib.pyplot as plt
import time
import torch
from collections import deque

# REVIEW: Assuming components are in 'src' or PYTHONPATH is set correctly
try:
    from src.grid_env import GridEnv
    from src.rl_agent import DQNAgent
    from src.shape_generator import (
        generate_distinct_shapes,
        generate_random_connected_shape,
    )
except ImportError:
    print(
        "Error importing modules. Ensure grid_env, rl_agent, shape_generator are accessible."
    )
    print("Attempting import from current directory...")
    from grid_env import GridEnv
    from rl_agent import DQNAgent
    from shape_generator import (
        generate_distinct_shapes,
        generate_random_connected_shape,
    )


def train_agent_with_random_shapes(
    agent,
    grid_size=(10, 10),
    num_blocks=4,  # This is the FIXED number of blocks we will use
    num_episodes=5000,
    max_steps_per_episode=150,
    print_every=20,
    save_every=200,
    model_dir="models",
    model_base_name="dqn_matter_agent",
    use_curriculum=True,  # Flag to enable/disable non-block-count curriculum aspects
    target_update_metric="avg_score",  # 'score' or 'avg_score' for LR scheduler
    warmup_episodes=100,  # Episodes before starting epsilon decay
):
    """Train agent with randomly generated shapes for each episode."""
    print("Starting training with random shapes...")
    scores_deque = deque(maxlen=100)  # For tracking rolling average
    all_scores = []
    avg_scores_list = []
    success_rates = []  # Track success rate over last 100 episodes

    model_path = os.path.join(model_dir, f"{model_base_name}.pth")
    os.makedirs(model_dir, exist_ok=True)

    # --- Curriculum Parameters ---
    # REVIEW: Keep num_blocks fixed to the value passed to the function (e.g., 4).
    # The agent's network state and action sizes are fixed based on this.
    current_num_blocks = num_blocks  # Use the fixed number of blocks always.
    current_max_steps = 50 if use_curriculum else max_steps_per_episode
    min_shape_distance = 1  # Start with easier distances

    start_time_total = time.time()

    for episode in range(1, num_episodes + 1):
        episode_start_time = time.time()

        # --- Curriculum Update (Only for non-state/action changing parameters) ---
        if use_curriculum:
            # Example: Increase difficulty gradually for steps and distance
            # REMOVED: Block count changes that caused the runtime error.
            if episode == 500:
                current_max_steps = 75
                print(
                    f"Curriculum Update @ Ep {episode}: max_steps={current_max_steps}"
                )
            elif episode == 1500:
                current_max_steps = 100
                print(
                    f"Curriculum Update @ Ep {episode}: max_steps={current_max_steps}"
                )
            elif episode == 3000:
                current_max_steps = max_steps_per_episode  # Reach target max_steps
                min_shape_distance = 2
                print(
                    f"Curriculum Update @ Ep {episode}: max_steps={current_max_steps}, min_dist={min_shape_distance}"
                )
            # Add more stages as needed

        # --- Generate Episode Task ---
        try:
            # Generate shapes with the FIXED number of blocks
            shapes = generate_distinct_shapes(
                grid_size=grid_size,
                num_blocks=current_num_blocks,  # This is always the fixed num_blocks
                num_shapes=2,
                min_distance=min_shape_distance,  # Use curriculum distance
                max_attempts=50,  # Avoid getting stuck in generation
            )
            if shapes is None or len(shapes) < 2:
                print(
                    f"Warning: Failed to generate distinct shapes for episode {episode}. Skipping."
                )
                continue  # Skip episode if generation fails

            initial_positions = shapes[0]
            target_positions = shapes[1]

            # Create environment with the FIXED number of blocks
            # State and action size will be consistent with agent initialization
            env = GridEnv(
                n=grid_size[0],
                m=grid_size[1],
                initial_positions=initial_positions,
                target_positions=target_positions,
                max_steps=current_max_steps,  # Use curriculum max_steps
                normalize_coords=True,  # Match agent's expected input if using normalization
            )
            state = env.reset()  # State will now consistently have the expected size
        except ValueError as e:
            print(f"Error creating environment for episode {episode}: {e}. Skipping.")
            continue  # Skip if initial state is invalid (e.g., disconnected)
        except Exception as e:
            print(f"Unexpected error during episode setup {episode}: {e}. Skipping.")
            continue

        score = 0
        done = False
        steps_taken = 0
        target_reached_flag = False

        # --- Run Episode ---
        while not done and steps_taken < current_max_steps:
            # Get valid actions mask (will be the correct size now)
            valid_actions_mask = env.get_valid_actions_mask()

            # Select action (only start decaying epsilon after warmup)
            explore = episode > warmup_episodes
            # State size now matches network input size
            action = agent.act(state, valid_actions_mask, eval_mode=not explore)

            # Take step
            next_state, reward, done, info = env.step(action)

            # Store experience and potentially learn
            agent.step(state, action, reward, next_state, done)

            # Update state and score
            state = next_state
            score += reward
            steps_taken += 1

            if done:
                target_reached_flag = info.get("target_reached", False)
                break  # Exit loop immediately if done

        # --- Post-Episode Processing ---
        scores_deque.append(score)
        all_scores.append(score)
        avg_score = np.mean(scores_deque)
        avg_scores_list.append(avg_score)

        # Track success rate (target reached)
        current_success = 1 if target_reached_flag else 0
        success_rates.append(current_success)
        avg_success_rate = np.mean(success_rates[-100:])  # Rolling success rate

        # Update epsilon (only after warmup)
        if episode > warmup_episodes:
            agent.update_epsilon()

        # Update LR scheduler if enabled
        if agent.scheduler:
            metric = avg_score if target_update_metric == "avg_score" else score
            agent.update_scheduler(metric)

        episode_duration = time.time() - episode_start_time

        # --- Logging ---
        if episode % print_every == 0 or episode == num_episodes:
            print(
                f"Ep: {episode}/{num_episodes} | "
                f"Score: {score:.2f} | Avg Score(100): {avg_score:.2f} | "
                f"Success Rate(100): {avg_success_rate:.2%} | Steps: {steps_taken} | "
                f"Epsilon: {agent.epsilon:.4f} | Target Reached: {target_reached_flag} | "
                f"Time: {episode_duration:.2f}s"
            )

        # --- Save Checkpoint ---
        if episode % save_every == 0 or episode == num_episodes:
            checkpoint_path = os.path.join(
                model_dir, f"{model_base_name}_ep{episode}.pth"
            )
            agent.save(checkpoint_path)
            # Save the latest as well for convenience
            agent.save(model_path)

    # --- End of Training ---
    total_training_time = time.time() - start_time_total
    print(f"\nTraining finished after {num_episodes} episodes.")
    print(f"Total training time: {total_training_time / 3600:.2f} hours.")
    print(f"Final model saved to: {model_path}")

    # Plot results
    plot_training_results(
        all_scores,
        avg_scores_list,
        avg_success_rate_history=success_rates,
        plot_avg_success=True,
    )

    return all_scores, avg_scores_list


def evaluate_agent(
    agent,
    grid_size=(10, 10),
    num_blocks=4,
    num_eval_episodes=20,
    max_steps=150,
    model_path=None,
):
    """Evaluate the trained agent's performance."""
    print("\n--- Starting Evaluation ---")
    if model_path:
        if not agent.load(model_path, eval_mode=True):
            print("Evaluation aborted: Could not load model.")
            return 0.0, 0.0
    else:
        # Ensure agent is in eval mode if no path provided
        agent.epsilon = 0.0  # Turn off exploration
        agent.qnetwork_online.eval()
        agent.qnetwork_target.eval()
        print("Evaluating agent with current in-memory weights.")

    successful_episodes = 0
    total_scores = []

    for episode in range(1, num_eval_episodes + 1):
        print(f"\nEvaluation Episode {episode}/{num_eval_episodes}")
        try:
            # Generate task with the standard number of blocks
            shapes = generate_distinct_shapes(
                grid_size, num_blocks, num_shapes=2, min_distance=2
            )
            if shapes is None or len(shapes) < 2:
                print(
                    "Warning: Failed to generate shapes for evaluation episode. Skipping."
                )
                continue

            initial_positions = shapes[0]
            target_positions = shapes[1]

            env = GridEnv(
                n=grid_size[0],
                m=grid_size[1],
                initial_positions=initial_positions,
                target_positions=target_positions,
                max_steps=max_steps,
                normalize_coords=True,
            )
            state = env.reset()
            env.render(title="Eval Start")

        except Exception as e:
            print(f"Error setting up evaluation episode {episode}: {e}. Skipping.")
            continue

        score = 0
        done = False
        steps = 0
        target_reached = False

        while not done and steps < max_steps:
            valid_actions_mask = env.get_valid_actions_mask()
            action = agent.act(
                state, valid_actions_mask, eval_mode=True
            )  # Use eval_mode=True

            next_state, reward, done, info = env.step(action)
            state = next_state
            score += reward
            steps += 1

            if done:
                target_reached = info.get("target_reached", False)
                break

        total_scores.append(score)
        if target_reached:
            successful_episodes += 1
            print(
                f"Result: Success! Target reached in {steps} steps. Score: {score:.2f}"
            )
        else:
            print(
                f"Result: Failure. Target not reached within {max_steps} steps. Score: {score:.2f}"
            )
        env.render(title="Eval End")
        print("-" * 30)

    avg_score = np.mean(total_scores) if total_scores else 0.0
    success_rate = (
        successful_episodes / num_eval_episodes if num_eval_episodes > 0 else 0.0
    )

    print("\n--- Evaluation Summary ---")
    print(f"Episodes: {num_eval_episodes}")
    print(
        f"Success Rate: {success_rate:.2%} ({successful_episodes}/{num_eval_episodes})"
    )
    print(f"Average Score: {avg_score:.2f}")
    print("--------------------------")

    return success_rate, avg_score


def plot_training_results(
    scores,
    avg_scores,
    eval_scores=None,
    eval_every=None,
    avg_success_rate_history=None,
    plot_avg_success=False,
):
    """Plot training progress."""
    fig, ax1 = plt.subplots(figsize=(12, 7))

    episodes = np.arange(1, len(scores) + 1)

    # Plot episode scores (primary y-axis)
    color = "tab:blue"
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Score", color=color)
    ax1.plot(episodes, scores, label="Episode Score", alpha=0.3, color=color)
    ax1.plot(
        episodes, avg_scores, label="Avg Score (100 eps)", linewidth=2, color="darkblue"
    )
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.7)

    # Plot evaluation scores if provided
    if eval_scores and eval_every:
        eval_episodes = np.arange(eval_every, len(scores) + 1, eval_every)
        eval_episodes = eval_episodes[: len(eval_scores)]
        ax1.plot(
            eval_episodes,
            eval_scores,
            "ro-",
            label=f"Eval Score (every {eval_every} eps)",
            linewidth=2,
        )

    # Plot success rate (secondary y-axis)
    if plot_avg_success and avg_success_rate_history:
        ax2 = ax1.twinx()
        color = "tab:green"
        ax2.set_ylabel("Avg Success Rate (100 eps)", color=color)
        success_deque = deque(maxlen=100)
        rolling_avg_success = []
        for success in avg_success_rate_history:
            success_deque.append(success)
            rolling_avg_success.append(np.mean(success_deque))

        # Ensure rolling_avg_success has same length as episodes for plotting
        if len(rolling_avg_success) < len(episodes):
            # Pad beginning with NaN or first value if needed, though history should match episodes
            padding = [rolling_avg_success[0]] * (
                len(episodes) - len(rolling_avg_success)
            )
            rolling_avg_success = padding + rolling_avg_success

        ax2.plot(
            episodes,
            rolling_avg_success[: len(episodes)],
            label="Avg Success Rate (100 eps)",
            linewidth=2,
            color=color,
            linestyle=":",
        )
        ax2.tick_params(axis="y", labelcolor=color)
        ax2.set_ylim(0, 1.05)

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="center right")
    else:
        ax1.legend(loc="center right")

    ax1.set_title("Training Progress")
    fig.tight_layout()
    plt.savefig("training_results.png")
    print("Training plot saved to training_results.png")
    plt.show()


def visualize_agent(env, agent, episodes=1, pause=0.3):
    """Visualize agent performance in the environment."""
    print("\n--- Starting Visualization ---")
    agent.epsilon = 0.0
    agent.qnetwork_online.eval()

    for episode in range(episodes):
        try:
            state = env.reset()
        except Exception as e:
            print(f"Error resetting env for visualization: {e}")
            return

        total_reward = 0
        done = False
        step = 0

        print(f"\nVisualization Episode {episode+1}/{episodes}")
        env.render(title="Initial State")
        time.sleep(pause * 2)

        while not done:
            valid_actions_mask = env.get_valid_actions_mask()
            action = agent.act(state, valid_actions_mask, eval_mode=True)

            next_state, reward, done, info = env.step(action)
            state = next_state
            total_reward += reward
            step += 1

            env.render(title=f"Step {step}, Action: {action}")
            print(f"Reward: {reward:.2f}, Cumulative: {total_reward:.2f}")
            time.sleep(pause)

            if done:
                success = info.get("target_reached", False)
                print(f"\nEpisode finished in {step} steps.")
                print(f"Total Reward: {total_reward:.2f}")
                print(f"Target Reached: {success}")
                break
            if step >= env.max_steps:
                print("\nMax steps reached.")
                break
    print("--- Visualization Finished ---")


if __name__ == "__main__":
    # --- Configuration ---
    GRID_SIZE = (10, 10)
    NUM_BLOCKS = 4  # Fixed number of blocks for agent architecture
    MAX_STEPS = 150
    NUM_EPISODES = 8000
    MODEL_DIR = "models_reviewed"
    MODEL_BASE_NAME = f"dqn_matter_{NUM_BLOCKS}blocks"
    LOAD_EXISTING_MODEL = False
    TRAIN_AGENT = True
    EVALUATE_AGENT = True
    VISUALIZE_EXAMPLE = True

    # --- Environment Setup (for getting sizes) ---
    try:
        print(f"Initializing temporary environment with {NUM_BLOCKS} blocks to get *new* sizes...")
        # Generate shapes for the temporary env
        temp_initial = generate_random_connected_shape(GRID_SIZE, NUM_BLOCKS)
        temp_target = generate_random_connected_shape(GRID_SIZE, NUM_BLOCKS)
        if not temp_initial or not temp_target: raise ValueError("Shape generation failed")

        # Create the temp env (it will now use the modified _get_observation)
        temp_env = GridEnv(
            n=GRID_SIZE[0], m=GRID_SIZE[1],
            initial_positions=temp_initial,
            target_positions=temp_target,
            max_steps=MAX_STEPS,
            normalize_coords=True # Ensure consistency
        )
        # Get the NEW state size
        state_size = temp_env.observation_space_size
        action_size = temp_env.action_space_size # Action size should be unchanged
        print(f"Determined *NEW* State Size: {state_size}, Action Size: {action_size}")
        del temp_env
    except Exception as e:
        print(f"Fatal Error: Could not initialize temporary environment: {e}")
        sys.exit(1)

    # --- Agent Initialization ---
    # The agent will now be initialized with the correct state_size for the new observation format
    agent = DQNAgent(
        state_size=state_size, # Use the newly calculated state_size
        action_size=action_size,
        learning_rate=1e-4,
        gamma=0.99,
        buffer_size=100000,
        batch_size=128,
        update_every=4,
        tau=1e-3,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.9995,  # Slower decay
        prioritized_replay=True,
        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_increment=0.0001,
        double_dqn=True,
        dueling=True,
        gradient_clipping=1.0,
        use_lr_scheduler=False,
    )

    model_path = os.path.join(MODEL_DIR, f"{MODEL_BASE_NAME}.pth")

    # --- Load Existing Model (Optional) ---
    if LOAD_EXISTING_MODEL:
        if os.path.exists(model_path):
            print(f"Attempting to load model from: {model_path}")
            agent.load(model_path, eval_mode=False)
        else:
            print(
                f"Model file not found at {model_path}. Starting training from scratch."
            )

    # --- Training ---
    if TRAIN_AGENT:
        train_agent_with_random_shapes(
            agent=agent,
            grid_size=GRID_SIZE,
            num_blocks=NUM_BLOCKS,  # Pass the fixed number of blocks
            num_episodes=NUM_EPISODES,
            max_steps_per_episode=MAX_STEPS,
            print_every=50,
            save_every=500,
            model_dir=MODEL_DIR,
            model_base_name=MODEL_BASE_NAME,
            use_curriculum=True,  # Enable non-block-count curriculum
            warmup_episodes=500,
        )

    # --- Evaluation ---
    if EVALUATE_AGENT:
        final_model_path = os.path.join(MODEL_DIR, f"{MODEL_BASE_NAME}.pth")
        if not os.path.exists(final_model_path):
            checkpoints = sorted(
                [
                    f
                    for f in os.listdir(MODEL_DIR)
                    if f.startswith(MODEL_BASE_NAME) and f.endswith(".pth")
                ]
            )
            if checkpoints:
                final_model_path = os.path.join(MODEL_DIR, checkpoints[-1])
                print(
                    f"Final model not found, using last checkpoint: {final_model_path}"
                )
            else:
                print("No model found for evaluation.")
                final_model_path = None

        if final_model_path:
            evaluate_agent(
                agent=agent,
                grid_size=GRID_SIZE,
                num_blocks=NUM_BLOCKS,  # Evaluate with the same num_blocks
                num_eval_episodes=30,
                max_steps=MAX_STEPS,
                model_path=final_model_path,
            )
        else:
            print("Skipping evaluation as no model file was found.")

    # --- Visualization ---
    if VISUALIZE_EXAMPLE:
        print("\nVisualizing agent on a new random task...")
        try:
            vis_shapes = generate_distinct_shapes(
                GRID_SIZE, NUM_BLOCKS, num_shapes=2, min_distance=2
            )
            if vis_shapes:
                vis_env = GridEnv(
                    n=GRID_SIZE[0],
                    m=GRID_SIZE[1],
                    initial_positions=vis_shapes[0],
                    target_positions=vis_shapes[1],
                    max_steps=MAX_STEPS,
                    normalize_coords=True,
                )
                final_model_path = os.path.join(MODEL_DIR, f"{MODEL_BASE_NAME}.pth")
                # Find last checkpoint if final doesn't exist
                if not os.path.exists(final_model_path):
                    checkpoints = sorted(
                        [
                            f
                            for f in os.listdir(MODEL_DIR)
                            if f.startswith(MODEL_BASE_NAME) and f.endswith(".pth")
                        ]
                    )
                    if checkpoints:
                        final_model_path = os.path.join(MODEL_DIR, checkpoints[-1])

                if os.path.exists(final_model_path):
                    agent.load(final_model_path, eval_mode=True)
                    visualize_agent(vis_env, agent, episodes=1, pause=0.2)
                else:
                    print("Cannot visualize: Model file not found.")
            else:
                print("Cannot visualize: Failed to generate shapes.")
        except Exception as e:
            print(f"Error during visualization setup: {e}")
