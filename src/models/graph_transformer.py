"""
Graph Transformer for Drug (Molecule) Encoding
Implements attention mechanism over molecular graphs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool
from torch_geometric.data import Batch


class GraphTransformerLayer(nn.Module):
    """Single Graph Transformer Layer with multi-head attention"""
    
    def __init__(self, 
                 hidden_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 use_edge_features: bool = True,
                 edge_dim: int = 12):
        super(GraphTransformerLayer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # TransformerConv: graph attention with multi-head
        self.conv = TransformerConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            dropout=dropout,
            edge_dim=edge_dim if use_edge_features else None,
            beta=True,  # Use gating mechanism
            concat=True  # Concatenate heads
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x: Node features [num_nodes, hidden_dim]
            edge_index: Edge indices [2, num_edges]
            edge_attr: Edge features [num_edges, edge_dim]
        
        Returns:
            Updated node features [num_nodes, hidden_dim]
        """
        # Self-attention with residual
        x_res = x
        x = self.conv(x, edge_index, edge_attr)
        x = self.norm1(x + x_res)
        
        # Feedforward with residual
        x_res = x
        x = self.ffn(x)
        x = self.norm2(x + x_res)
        
        return x


class GraphTransformerEncoder(nn.Module):
    """
    Graph Transformer Encoder for Molecular Graphs
    
    Architecture:
        Input (SMILES → Graph) → Embedding → N × Transformer Layers → Global Pooling → Output
    """
    
    def __init__(self,
                 atom_features: int = 78,
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 use_edge_features: bool = True,
                 edge_dim: int = 12):
        super(GraphTransformerEncoder, self).__init__()
        
        self.atom_features = atom_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection
        self.atom_embedding = nn.Linear(atom_features, hidden_dim)
        
        # Stacked Graph Transformer Layers
        self.layers = nn.ModuleList([
            GraphTransformerLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_edge_features=use_edge_features,
                edge_dim=edge_dim
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, data):
        """
        Args:
            data: PyG Batch object with:
                - x: Node features [num_nodes, atom_features]
                - edge_index: Edge indices [2, num_edges]
                - edge_attr: Edge features [num_edges, edge_dim]
                - batch: Batch assignment [num_nodes]
        
        Returns:
            Graph embedding [batch_size, hidden_dim]
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Embed atom features
        x = self.atom_embedding(x)  # [num_nodes, hidden_dim]
        
        # Apply Graph Transformer layers
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        
        # Global pooling (mean over all nodes in each graph)
        x = global_mean_pool(x, batch)  # [batch_size, hidden_dim]
        
        # Final projection
        x = self.output_projection(x)
        
        return x


def test_graph_transformer():
    """Test Graph Transformer with dummy data"""
    from torch_geometric.data import Data, Batch
    
    # Create dummy molecular graph
    num_atoms = 10
    x = torch.randn(num_atoms, 78)  # Atom features
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long)
    edge_attr = torch.randn(6, 12)  # Bond features
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    batch = Batch.from_data_list([data, data])  # Batch of 2 molecules
    
    # Initialize model
    model = GraphTransformerEncoder(
        atom_features=78,
        hidden_dim=128,
        num_layers=4,
        num_heads=8
    )
    
    # Forward pass
    output = model(batch)
    
    print("Graph Transformer Test:")
    print(f"Input: {batch.num_graphs} molecules, {batch.num_nodes} total atoms")
    print(f"Output shape: {output.shape}")  # Should be [2, 128]
    print(f"✓ Test passed!")


if __name__ == "__main__":
    test_graph_transformer()
