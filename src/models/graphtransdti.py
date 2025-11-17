"""
GraphTransDTI: Complete Model for Drug-Target Interaction Prediction

Architecture:
    Drug (SMILES) → Graph Transformer → Drug Embedding
    Protein (Sequence) → CNN + BiLSTM → Protein Embedding
    [Drug, Protein] → Cross-Attention → Fused Representation
    Fused → MLP → Binding Affinity (Regression)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .graph_transformer import GraphTransformerEncoder
from .protein_encoder import ProteinEncoder
from .cross_attention import CrossAttentionFusion


class GraphTransDTI(nn.Module):
    """
    Complete GraphTransDTI Model
    
    Key Innovation:
    - Graph Transformer for drugs (captures global molecular structure)
    - CNN + BiLSTM for proteins (captures local motifs + long-range dependencies)
    - Cross-Attention fusion (learns drug-protein interaction patterns)
    """
    
    def __init__(self, config: dict):
        super(GraphTransDTI, self).__init__()
        
        self.config = config
        
        # Drug Encoder - Graph Transformer
        drug_config = config['model']['drug_encoder']
        self.drug_encoder = GraphTransformerEncoder(
            atom_features=config['drug']['atom_features'],
            hidden_dim=drug_config['hidden_dim'],
            num_layers=drug_config['num_layers'],
            num_heads=drug_config['num_heads'],
            dropout=drug_config['dropout'],
            use_edge_features=drug_config['use_edge_features'],
            edge_dim=config['drug']['edge_features']
        )
        
        # Protein Encoder - CNN + BiLSTM
        protein_config = config['model']['protein_encoder']
        self.protein_encoder = ProteinEncoder(
            vocab_size=config['protein']['vocab_size'],
            embedding_dim=config['protein']['embedding_dim'],
            cnn_filters=protein_config['cnn_filters'],
            kernel_sizes=protein_config['cnn_kernel_sizes'],
            lstm_hidden_dim=protein_config['lstm_hidden_dim'],
            lstm_num_layers=protein_config['lstm_num_layers'],
            lstm_dropout=protein_config['lstm_dropout'],
            bidirectional=protein_config['bidirectional']
        )
        
        # Cross-Attention Fusion
        fusion_config = config['model']['cross_attention']
        self.cross_attention = CrossAttentionFusion(
            hidden_dim=fusion_config['hidden_dim'],
            num_heads=fusion_config['num_heads'],
            dropout=fusion_config['dropout'],
            fusion_method="concat"
        )
        
        # Prediction Head (MLP)
        predictor_config = config['model']['predictor']
        hidden_dims = predictor_config['hidden_dims']
        dropout = predictor_config['dropout']
        
        layers = []
        input_dim = fusion_config['hidden_dim']
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = hidden_dim
        
        # Final output layer (regression)
        layers.append(nn.Linear(input_dim, predictor_config['output_dim']))
        
        self.predictor = nn.Sequential(*layers)
        
    def forward(self, drug_data, protein_seq):
        """
        Forward pass through GraphTransDTI
        
        Args:
            drug_data: PyG Batch object containing molecular graphs
                - x: Node features [num_nodes, atom_features]
                - edge_index: Edge indices [2, num_edges]
                - edge_attr: Edge features [num_edges, edge_dim]
                - batch: Batch assignment [num_nodes]
            protein_seq: Protein sequences [batch_size, max_protein_len]
        
        Returns:
            predictions: Binding affinity predictions [batch_size, 1]
        """
        # Encode drug (molecular graph)
        drug_emb = self.drug_encoder(drug_data)  # [batch_size, hidden_dim]
        
        # Encode protein (amino acid sequence)
        protein_emb = self.protein_encoder(protein_seq, return_sequence=False)  # [batch_size, hidden_dim]
        
        # Cross-attention fusion
        fused = self.cross_attention(drug_emb, protein_emb)  # [batch_size, hidden_dim]
        
        # Predict binding affinity
        predictions = self.predictor(fused)  # [batch_size, 1]
        
        return predictions
    
    def get_embeddings(self, drug_data, protein_seq):
        """
        Get intermediate embeddings (for analysis/visualization)
        
        Returns:
            dict: {
                'drug_emb': [batch_size, hidden_dim],
                'protein_emb': [batch_size, hidden_dim],
                'fused_emb': [batch_size, hidden_dim]
            }
        """
        with torch.no_grad():
            drug_emb = self.drug_encoder(drug_data)
            protein_emb = self.protein_encoder(protein_seq, return_sequence=False)
            fused = self.cross_attention(drug_emb, protein_emb)
        
        return {
            'drug_emb': drug_emb,
            'protein_emb': protein_emb,
            'fused_emb': fused
        }


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_graphtransdti():
    """Test complete GraphTransDTI model"""
    from torch_geometric.data import Data, Batch
    import yaml
    
    # Load config
    config = {
        'drug': {
            'atom_features': 78,
            'edge_features': 12
        },
        'protein': {
            'vocab_size': 26,
            'embedding_dim': 128
        },
        'model': {
            'drug_encoder': {
                'hidden_dim': 128,
                'num_layers': 4,
                'num_heads': 8,
                'dropout': 0.1,
                'use_edge_features': True
            },
            'protein_encoder': {
                'cnn_filters': [32, 64, 96],
                'cnn_kernel_sizes': [4, 8, 12],
                'lstm_hidden_dim': 128,
                'lstm_num_layers': 2,
                'lstm_dropout': 0.1,
                'bidirectional': True
            },
            'cross_attention': {
                'hidden_dim': 128,
                'num_heads': 8,
                'dropout': 0.1
            },
            'predictor': {
                'hidden_dims': [256, 128, 64],
                'dropout': 0.2,
                'output_dim': 1
            }
        }
    }
    
    # Create dummy data
    batch_size = 4
    
    # Drug graphs
    graphs = []
    for _ in range(batch_size):
        num_atoms = 10
        x = torch.randn(num_atoms, 78)
        edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
        edge_attr = torch.randn(4, 12)
        graphs.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))
    
    drug_batch = Batch.from_data_list(graphs)
    
    # Protein sequences
    protein_seq = torch.randint(1, 26, (batch_size, 100))
    
    # Initialize model
    model = GraphTransDTI(config)
    
    # Forward pass
    predictions = model(drug_batch, protein_seq)
    
    # Get embeddings
    embeddings = model.get_embeddings(drug_batch, protein_seq)
    
    print("=" * 60)
    print("GraphTransDTI Model Test")
    print("=" * 60)
    print(f"Input: {batch_size} drug-protein pairs")
    print(f"Output shape: {predictions.shape}")  # Should be [4, 1]
    print(f"\nEmbeddings:")
    print(f"  Drug: {embeddings['drug_emb'].shape}")
    print(f"  Protein: {embeddings['protein_emb'].shape}")
    print(f"  Fused: {embeddings['fused_emb'].shape}")
    print(f"\nTotal parameters: {count_parameters(model):,}")
    print("=" * 60)
    print("✓ Test passed!")


if __name__ == "__main__":
    test_graphtransdti()
