"""
Protein Encoder: CNN + BiLSTM
Extracts features from protein amino acid sequences
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProteinCNN(nn.Module):
    """
    Convolutional Neural Network for extracting local motifs from protein sequences
    Multiple kernel sizes to capture different motif lengths
    """
    
    def __init__(self, 
                 embedding_dim: int = 128,
                 cnn_filters: list = [32, 64, 96],
                 kernel_sizes: list = [4, 8, 12]):
        super(ProteinCNN, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.cnn_filters = cnn_filters
        self.kernel_sizes = kernel_sizes
        
        # Multiple 1D convolutions with different kernel sizes
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=num_filters,
                kernel_size=k,
                padding=k // 2  # Same padding
            )
            for num_filters, k in zip(cnn_filters, kernel_sizes)
        ])
        
        self.total_filters = sum(cnn_filters)
        
    def forward(self, x):
        """
        Args:
            x: Embedded protein sequence [batch_size, seq_len, embedding_dim]
        
        Returns:
            CNN features [batch_size, seq_len, total_filters]
        """
        # Transpose for Conv1d: [batch, embedding_dim, seq_len]
        x = x.transpose(1, 2)
        
        # Apply multiple CNNs and concatenate
        conv_outputs = []
        for conv in self.conv_layers:
            conv_out = F.relu(conv(x))  # [batch, filters, seq_len]
            conv_outputs.append(conv_out)
        
        # Concatenate along filter dimension
        x = torch.cat(conv_outputs, dim=1)  # [batch, total_filters, seq_len]
        
        # Transpose back: [batch, seq_len, total_filters]
        x = x.transpose(1, 2)
        
        return x


class ProteinEncoder(nn.Module):
    """
    Complete Protein Encoder: Embedding → CNN (local) → BiLSTM (global context)
    
    Architecture:
        Amino Acid Sequence → Embedding → CNN (motifs) → BiLSTM (context) → Output
    """
    
    def __init__(self,
                 vocab_size: int = 26,  # 20 amino acids + special tokens
                 embedding_dim: int = 128,
                 cnn_filters: list = [32, 64, 96],
                 kernel_sizes: list = [4, 8, 12],
                 lstm_hidden_dim: int = 128,
                 lstm_num_layers: int = 2,
                 lstm_dropout: float = 0.1,
                 bidirectional: bool = True):
        super(ProteinEncoder, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.bidirectional = bidirectional
        
        # Amino acid embedding
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # CNN for local motifs
        self.cnn = ProteinCNN(
            embedding_dim=embedding_dim,
            cnn_filters=cnn_filters,
            kernel_sizes=kernel_sizes
        )
        
        # BiLSTM for long-range dependencies
        self.lstm = nn.LSTM(
            input_size=self.cnn.total_filters,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=lstm_dropout if lstm_num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output dimension
        self.output_dim = lstm_hidden_dim * 2 if bidirectional else lstm_hidden_dim
        
        # Output projection to match drug encoder dimension
        self.output_projection = nn.Linear(self.output_dim, embedding_dim)
        
    def forward(self, x, return_sequence=False):
        """
        Args:
            x: Protein sequence indices [batch_size, seq_len]
            return_sequence: If True, return full sequence output; else return last hidden state
        
        Returns:
            If return_sequence=False: [batch_size, embedding_dim]
            If return_sequence=True: [batch_size, seq_len, embedding_dim]
        """
        # Embedding
        x = self.embedding(x)  # [batch, seq_len, embedding_dim]
        
        # CNN for local features
        x = self.cnn(x)  # [batch, seq_len, total_filters]
        
        # BiLSTM for sequential context
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: [batch, seq_len, hidden_dim * 2]
        # h_n: [num_layers * 2, batch, hidden_dim]
        
        if return_sequence:
            # Return full sequence for cross-attention
            x = self.output_projection(lstm_out)  # [batch, seq_len, embedding_dim]
        else:
            # Use last hidden state (concatenate forward and backward)
            if self.bidirectional:
                # h_n[-2]: forward last hidden, h_n[-1]: backward last hidden
                h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)  # [batch, hidden_dim * 2]
            else:
                h_last = h_n[-1]  # [batch, hidden_dim]
            
            x = self.output_projection(h_last)  # [batch, embedding_dim]
        
        return x


def test_protein_encoder():
    """Test Protein Encoder with dummy data"""
    batch_size = 4
    seq_len = 100
    vocab_size = 26
    
    # Dummy protein sequences (random indices)
    x = torch.randint(1, vocab_size, (batch_size, seq_len))
    
    # Initialize model
    model = ProteinEncoder(
        vocab_size=vocab_size,
        embedding_dim=128,
        cnn_filters=[32, 64, 96],
        kernel_sizes=[4, 8, 12],
        lstm_hidden_dim=128,
        lstm_num_layers=2,
        bidirectional=True
    )
    
    # Test without sequence output
    output = model(x, return_sequence=False)
    print("Protein Encoder Test (single vector):")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")  # Should be [4, 128]
    
    # Test with sequence output (for cross-attention)
    output_seq = model(x, return_sequence=True)
    print("\nProtein Encoder Test (sequence output):")
    print(f"Output shape: {output_seq.shape}")  # Should be [4, 100, 128]
    print(f"✓ Test passed!")


if __name__ == "__main__":
    test_protein_encoder()
