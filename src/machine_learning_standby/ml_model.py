import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch


class ProgrammableMatterGNN(nn.Module):
    """
    Graph Neural Network for programmable matter simulation.
    - Learns block-to-block relationships
    - Captures connectivity constraints implicitly
    - Produces move predictions and heuristic estimates
    """

    def __init__(self, node_features=8, hidden_dim=64, output_dim=9):
        super(ProgrammableMatterGNN, self).__init__()

        # Node feature processing
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        # Global graph-level features
        self.global_fc1 = nn.Linear(hidden_dim, hidden_dim)

        # Output heads
        self.heuristic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),  # Scalar heuristic value
        )

        self.move_prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),  # 8 directions + 1 for "no move"
        )

        self.block_priority_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),  # Priority score for each block
        )

    def forward(self, data):
        # Node features and graph structure
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Node-level message passing
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)

        # Graph-level pooling for global features
        x_global = global_mean_pool(x, batch)
        x_global = F.relu(self.global_fc1(x_global))

        # Output predictions
        heuristic = self.heuristic_head(x_global)
        move_logits = self.move_prediction_head(x)
        block_priority = self.block_priority_head(x)

        return heuristic, move_logits, block_priority


def create_graph_from_state(current_state, target_state, grid_dims):
    """
    Convert a state representation to a graph for the GNN.

    Args:
        current_state: List of (x,y) coordinates of the current blocks
        target_state: List of (x,y) coordinates of the target blocks
        grid_dims: Tuple of (n,m) grid dimensions

    Returns:
        torch_geometric.data.Data: Graph representation
    """
    n, m = grid_dims
    num_blocks = len(current_state)

    # Create node features
    node_features = []
    for i, ((cx, cy), (tx, ty)) in enumerate(zip(current_state, target_state)):
        # Features per node: [x_curr, y_curr, x_target, y_target,
        #                    x_rel, y_rel, x_normalized, y_normalized]
        x_rel = tx - cx  # Relative x displacement to target
        y_rel = ty - cy  # Relative y displacement to target

        # Normalized positions (0-1 scale)
        x_norm = cx / (n - 1) if n > 1 else 0.5
        y_norm = cy / (m - 1) if m > 1 else 0.5

        node_features.append([cx, cy, tx, ty, x_rel, y_rel, x_norm, y_norm])

    # Create edges based on Moore neighborhood (8-way connectivity)
    edges = []

    # Create a map for fast lookups
    block_positions = {pos: idx for idx, pos in enumerate(current_state)}

    # Add edges between adjacent blocks
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for i, (x, y) in enumerate(current_state):
        for dx, dy in directions:
            neighbor = (x + dx, y + dy)
            if neighbor in block_positions:
                j = block_positions[neighbor]
                # Add both directions for undirected graph
                edges.append([i, j])

    # Convert to PyTorch tensors
    x = torch.tensor(node_features, dtype=torch.float)

    # Handle the case with no edges
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        # Create an empty edge index tensor with the right shape
        edge_index = torch.zeros((2, 0), dtype=torch.long)

    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index)

    return data


def direction_to_index(dx, dy):
    """Convert a direction to an index (0-8)."""
    direction_map = {
        (-1, 0): 0,  # North
        (1, 0): 1,  # South
        (0, -1): 2,  # West
        (0, 1): 3,  # East
        (-1, -1): 4,  # Northwest
        (-1, 1): 5,  # Northeast
        (1, -1): 6,  # Southwest
        (1, 1): 7,  # Southeast
        (0, 0): 8,  # No movement
    }
    return direction_map.get((dx, dy), 8)  # Default to "no movement"


def index_to_direction(idx):
    """Convert an index (0-8) to a direction."""
    index_map = {
        0: (-1, 0),  # North
        1: (1, 0),  # South
        2: (0, -1),  # West
        3: (0, 1),  # East
        4: (-1, -1),  # Northwest
        5: (-1, 1),  # Northeast
        6: (1, -1),  # Southwest
        7: (1, 1),  # Southeast
        8: (0, 0),  # No movement
    }
    return index_map.get(idx, (0, 0))  # Default to no movement
