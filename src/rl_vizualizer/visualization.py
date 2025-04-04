import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
from IPython.display import display, clear_output
import time
from matplotlib.colors import ListedColormap
import io
from PIL import Image


class TrainingVisualizer:
    """Class to visualize the training progress of the RL agent"""

    def __init__(self, n, m, update_interval=20):
        self.fig, self.axs = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle("Programmable Matter RL Training Progress", fontsize=16)

        # Initialize plots
        (self.reward_line,) = self.axs[0, 0].plot([], [], "b-", label="Episode Reward")
        (self.avg_reward_line,) = self.axs[0, 0].plot(
            [], [], "r-", label="Avg Reward (100 ep)"
        )
        self.axs[0, 0].set_xlabel("Episode")
        self.axs[0, 0].set_ylabel("Reward")
        self.axs[0, 0].set_title("Training Rewards")
        self.axs[0, 0].legend()

        (self.steps_line,) = self.axs[0, 1].plot([], [], "g-")
        self.axs[0, 1].set_xlabel("Episode")
        self.axs[0, 1].set_ylabel("Steps")
        self.axs[0, 1].set_title("Episode Length")

        # Grid visualization
        self.grid_img = self.axs[1, 0].imshow(
            np.zeros((n, m)), cmap="viridis", vmin=0, vmax=2
        )
        self.axs[1, 0].set_title("Current State")
        self.axs[1, 0].set_axis_off()

        # Action probabilities visualization
        self.axs[1, 1].set_title("Action Probabilities")
        self.axs[1, 1].set_axis_off()
        self.action_text = self.axs[1, 1].text(
            0.5,
            0.5,
            "",
            horizontalalignment="center",
            verticalalignment="center",
            transform=self.axs[1, 1].transAxes,
            fontsize=12,
        )

        # Data storage
        self.rewards = []
        self.avg_rewards = []
        self.steps = []
        self.episodes = []

        # Update settings
        self.update_interval = update_interval

        # Initial render
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        self.fig.canvas.draw()

    def update_plots(self, episode, reward, step, grid, action_info=None):
        """Update all visualization components"""
        # Store data
        self.rewards.append(reward)
        self.steps.append(step)
        self.episodes.append(episode)

        # Calculate running average
        if len(self.rewards) >= 100:
            avg_reward = np.mean(self.rewards[-100:])
        else:
            avg_reward = np.mean(self.rewards)
        self.avg_rewards.append(avg_reward)

        # Update reward plot
        self.reward_line.set_data(self.episodes, self.rewards)
        self.avg_reward_line.set_data(self.episodes, self.avg_rewards)

        # Update steps plot
        self.steps_line.set_data(self.episodes, self.steps)

        # Auto-scale axes
        for ax in [self.axs[0, 0], self.axs[0, 1]]:
            ax.relim()
            ax.autoscale_view()

        # Update grid visualization
        self.grid_img.set_array(grid)

        # Update action probabilities
        if action_info:
            info_text = (
                f"Selected block: {action_info['block_id']}\n"
                f"Direction: {action_info['direction']}\n"
                f"Episode: {episode}, Step: {step}\n"
                f"Reward: {reward:.2f}, Avg Reward: {avg_reward:.2f}"
            )
            self.action_text.set_text(info_text)

        # Refresh canvas
        self.fig.canvas.draw()

        # For interactive environments like Jupyter
        if episode % self.update_interval == 0:
            filename = f"training_progress_ep{episode}.png"
            self.fig.savefig(filename)
            print(f"Saved visualization to {filename}")

    def save(self, path="training_progress.png"):
        """Save current visualization to file"""
        self.fig.savefig(path)

    def create_grid_visualization(self, grid, blocks, targets, obstacles=None):
        """Create a color-coded visualization of the grid state"""
        # Create a visual grid representation
        visual_grid = np.zeros_like(grid)

        # Mark cells with different values:
        # 0: Empty, 1: Block, 2: Target, 3: Block on target, 4: Obstacle

        # Mark targets
        for _, x, y in targets:
            visual_grid[x, y] = 2

        # Mark obstacles
        if obstacles:
            for x, y in obstacles:
                visual_grid[x, y] = 4

        # Mark blocks (and blocks on targets)
        for _, x, y in blocks:
            if visual_grid[x, y] == 2:  # If on target
                visual_grid[x, y] = 3
            else:
                visual_grid[x, y] = 1

        # Create a custom colormap
        colors = ["white", "blue", "green", "purple", "red"]
        cmap = ListedColormap(colors)

        return visual_grid, cmap

    def generate_animation(self, frames, filename="training_animation.gif"):
        """Generate an animation from saved frames"""
        fig, ax = plt.subplots(figsize=(8, 8))

        def update(frame):
            ax.clear()
            ax.imshow(frame)
            ax.set_axis_off()
            return [ax]

        ani = animation.FuncAnimation(fig, update, frames=frames, interval=200)
        ani.save(filename, writer="pillow", fps=5)
        plt.close()
