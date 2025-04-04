import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import json

from ml_model import ProgrammableMatterGNN
from data_generation import ProgrammableMatterDataset


class ModelTrainer:
    """
    Handles training and evaluation of the GNN model for programmable matter.
    """

    def __init__(
        self,
        data_dir="training_data",
        model_dir="saved_models",
        batch_size=32,
        learning_rate=0.001,
        hidden_dim=64,
    ):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_dim = hidden_dim

        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)

        # Initialize metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.heuristic_errors = []
        self.move_accuracies = []

    def load_data(self):
        """Load and split the dataset into training and validation sets."""
        data_file = os.path.join(self.data_dir, "../example_data/training_data.pkl")
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Training data file not found: {data_file}")

        # Load the full dataset
        full_dataset = ProgrammableMatterDataset(data_file)

        # Split into training and validation sets
        dataset_size = len(full_dataset)
        train_size = int(0.8 * dataset_size)
        val_size = dataset_size - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )

        print(f"Loaded {train_size} training samples and {val_size} validation samples")

        # Extract a sample to determine node feature size
        sample = full_dataset[0]
        self.node_features = sample.x.shape[1]

        return train_size, val_size

    def initialize_model(self):
        """Initialize the GNN model and optimizer."""
        # Determine if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Initialize model
        self.model = ProgrammableMatterGNN(
            node_features=self.node_features,
            hidden_dim=self.hidden_dim,
            output_dim=9,  # 8 directions + no move
        ).to(self.device)

        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        # Loss functions
        self.heuristic_loss_fn = nn.MSELoss()
        self.move_loss_fn = nn.CrossEntropyLoss()
        self.priority_loss_fn = nn.MSELoss()  # We'll use this later

        return self.model

    def train(self, epochs=30):
        """Train the model."""
        best_val_loss = float("inf")

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            heuristic_loss_sum = 0
            move_loss_sum = 0
            correct_moves = 0
            total_moves = 0

            progress_bar = tqdm(
                self.train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"
            )
            for batch in progress_bar:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()

                # Forward pass
                heuristic_pred, move_logits, _ = self.model(batch)

                # Calculate losses
                heuristic_loss = self.heuristic_loss_fn(
                    heuristic_pred, batch.y_heuristic
                )
                move_loss = self.move_loss_fn(move_logits, batch.y_moves)

                # Combined loss (you can adjust the weighting)
                loss = heuristic_loss + move_loss * 5.0

                # Backward pass and optimization
                loss.backward()
                self.optimizer.step()

                # Update metrics
                train_loss += loss.item() * batch.num_graphs
                heuristic_loss_sum += heuristic_loss.item() * batch.num_graphs
                move_loss_sum += move_loss.item() * batch.num_graphs

                # Calculate move prediction accuracy
                pred_moves = torch.argmax(move_logits, dim=1)
                correct_moves += (pred_moves == batch.y_moves).sum().item()
                total_moves += batch.y_moves.size(0)

                # Update progress bar
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "h_loss": f"{heuristic_loss.item():.4f}",
                        "move_loss": f"{move_loss.item():.4f}",
                    }
                )

            # Calculate epoch metrics
            train_loss /= len(self.train_loader.dataset)
            heuristic_loss_avg = heuristic_loss_sum / len(self.train_loader.dataset)
            move_loss_avg = move_loss_sum / len(self.train_loader.dataset)
            move_accuracy = correct_moves / total_moves if total_moves > 0 else 0

            self.train_losses.append(train_loss)

            print(
                f"Epoch {epoch+1} Training: Loss: {train_loss:.4f}, "
                f"Heuristic Loss: {heuristic_loss_avg:.4f}, "
                f"Move Loss: {move_loss_avg:.4f}, "
                f"Move Accuracy: {move_accuracy:.4f}"
            )

            # Validation phase
            val_loss, val_heuristic_error, val_move_accuracy = self.evaluate()

            self.val_losses.append(val_loss)
            self.heuristic_errors.append(val_heuristic_error)
            self.move_accuracies.append(val_move_accuracy)

            # Save model if it's the best so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(
                    os.path.join(self.model_dir, "best_model.pt"), epoch + 1
                )
                print(f"New best model saved with validation loss: {val_loss:.4f}")

            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                self.save_model(
                    os.path.join(self.model_dir, f"checkpoint_epoch_{epoch+1}.pt"),
                    epoch + 1,
                )

        # Final model save
        self.save_model(os.path.join(self.model_dir, "final_model.pt"), epochs)

        # Plot training curves
        self.plot_training_curves()

        return self.train_losses, self.val_losses

    def evaluate(self):
        """Evaluate the model on the validation set."""
        self.model.eval()
        val_loss = 0
        heuristic_error_sum = 0
        correct_moves = 0
        total_moves = 0

        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validation")
            for batch in progress_bar:
                batch = batch.to(self.device)

                # Forward pass
                heuristic_pred, move_logits, _ = self.model(batch)

                # Calculate losses
                heuristic_loss = self.heuristic_loss_fn(
                    heuristic_pred, batch.y_heuristic
                )
                move_loss = self.move_loss_fn(move_logits, batch.y_moves)

                # Combined loss
                loss = heuristic_loss + move_loss * 5.0

                # Update metrics
                val_loss += loss.item() * batch.num_graphs
                heuristic_error_sum += (
                    torch.abs(heuristic_pred - batch.y_heuristic).sum().item()
                )

                # Calculate move prediction accuracy
                pred_moves = torch.argmax(move_logits, dim=1)
                correct_moves += (pred_moves == batch.y_moves).sum().item()
                total_moves += batch.y_moves.size(0)

                # Update progress bar
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Calculate average metrics
        val_loss /= len(self.val_loader.dataset)
        heuristic_error = heuristic_error_sum / len(self.val_loader.dataset)
        move_accuracy = correct_moves / total_moves if total_moves > 0 else 0

        print(
            f"Validation: Loss: {val_loss:.4f}, "
            f"Heuristic Error: {heuristic_error:.4f}, "
            f"Move Accuracy: {move_accuracy:.4f}"
        )

        return val_loss, heuristic_error, move_accuracy

    def save_model(self, path, epoch):
        """Save the model and training state."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "heuristic_errors": self.heuristic_errors,
            "move_accuracies": self.move_accuracies,
        }
        torch.save(checkpoint, path)

    def load_model(self, path):
        """Load a saved model."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load metrics if they exist
        if "train_losses" in checkpoint:
            self.train_losses = checkpoint["train_losses"]
        if "val_losses" in checkpoint:
            self.val_losses = checkpoint["val_losses"]
        if "heuristic_errors" in checkpoint:
            self.heuristic_errors = checkpoint["heuristic_errors"]
        if "move_accuracies" in checkpoint:
            self.move_accuracies = checkpoint["move_accuracies"]

        print(f"Model loaded from {path} (epoch {checkpoint['epoch']})")

        return checkpoint["epoch"]

    def plot_training_curves(self):
        """Plot training and validation curves."""
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))

        # Plot training and validation loss
        axs[0, 0].plot(self.train_losses, label="Training Loss")
        axs[0, 0].plot(self.val_losses, label="Validation Loss")
        axs[0, 0].set_title("Loss Curves")
        axs[0, 0].set_xlabel("Epoch")
        axs[0, 0].set_ylabel("Loss")
        axs[0, 0].legend()
        axs[0, 0].grid(True)

        # Plot heuristic error
        axs[0, 1].plot(self.heuristic_errors, color="orange")
        axs[0, 1].set_title("Heuristic Error")
        axs[0, 1].set_xlabel("Epoch")
        axs[0, 1].set_ylabel("Mean Absolute Error")
        axs[0, 1].grid(True)

        # Plot move accuracy
        axs[1, 0].plot(self.move_accuracies, color="green")
        axs[1, 0].set_title("Move Prediction Accuracy")
        axs[1, 0].set_xlabel("Epoch")
        axs[1, 0].set_ylabel("Accuracy")
        axs[1, 0].grid(True)

        # Save the plots
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, "training_curves.png"))
        plt.close()


if __name__ == "__main__":
    # Example usage
    trainer = ModelTrainer(
        data_dir="training_data",
        model_dir="saved_models",
        batch_size=64,
        learning_rate=0.001,
        hidden_dim=128,
    )

    # Load data
    train_size, val_size = trainer.load_data()

    # Initialize model
    model = trainer.initialize_model()

    # Train model
    train_losses, val_losses = trainer.train(epochs=50)

    print("Training complete!")
