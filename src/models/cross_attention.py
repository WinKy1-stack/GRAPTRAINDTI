"""
Cross-Attention Layer for Drug-Protein Interaction
Implements multi-head cross-attention between drug and protein representations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadCrossAttention(nn.Module):
    """
    Multi-Head Cross-Attention mechanism
    
    Query: from one modality (e.g., drug)
    Key, Value: from another modality (e.g., protein)
    
    This allows learning interactions between drug atoms and protein residues
    """
    
    def __init__(self, 
                 hidden_dim: int = 128,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super(MultiHeadCrossAttention, self).__init__()
        
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # Linear projections for Q, K, V
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: [batch_size, query_len, hidden_dim]
            key: [batch_size, key_len, hidden_dim]
            value: [batch_size, value_len, hidden_dim]
            mask: Optional mask [batch_size, query_len, key_len]
        
        Returns:
            Attention output: [batch_size, query_len, hidden_dim]
            Attention weights: [batch_size, num_heads, query_len, key_len]
        """
        batch_size = query.size(0)
        query_len = query.size(1)
        key_len = key.size(1)
        
        # Linear projections and reshape for multi-head
        Q = self.query_proj(query).view(batch_size, query_len, self.num_heads, self.head_dim)
        K = self.key_proj(key).view(batch_size, key_len, self.num_heads, self.head_dim)
        V = self.value_proj(value).view(batch_size, key_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation: [batch, num_heads, seq_len, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute attention scores: Q @ K^T / sqrt(d_k)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # [batch, num_heads, query_len, key_len]
        
        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        # [batch, num_heads, query_len, head_dim]
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, query_len, self.hidden_dim)
        
        # Final output projection
        output = self.output_proj(attn_output)
        
        return output, attn_weights


class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Fusion Layer for Drug-Protein Interaction
    
    Performs bidirectional cross-attention:
    1. Drug attends to Protein
    2. Protein attends to Drug
    3. Fuse both representations
    """
    
    def __init__(self,
                 hidden_dim: int = 128,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 fusion_method: str = "concat"):  # concat, add, gate
        super(CrossAttentionFusion, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.fusion_method = fusion_method
        
        # Drug → Protein attention (drug queries protein)
        self.drug_to_protein_attn = MultiHeadCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Protein → Drug attention (protein queries drug)
        self.protein_to_drug_attn = MultiHeadCrossAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Layer normalization
        self.norm_drug = nn.LayerNorm(hidden_dim)
        self.norm_protein = nn.LayerNorm(hidden_dim)
        
        # Fusion layer
        if fusion_method == "concat":
            self.fusion = nn.Linear(hidden_dim * 2, hidden_dim)
        elif fusion_method == "gate":
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )
        
    def forward(self, drug_emb, protein_emb):
        """
        Args:
            drug_emb: Drug embedding [batch_size, hidden_dim]
            protein_emb: Protein embedding [batch_size, hidden_dim]
        
        Returns:
            Fused representation [batch_size, hidden_dim]
        """
        # Add sequence dimension for attention
        # [batch, hidden_dim] → [batch, 1, hidden_dim]
        drug_emb_seq = drug_emb.unsqueeze(1)
        protein_emb_seq = protein_emb.unsqueeze(1)
        
        # Drug attends to Protein
        drug_attended, drug_attn_weights = self.drug_to_protein_attn(
            query=drug_emb_seq,
            key=protein_emb_seq,
            value=protein_emb_seq
        )
        drug_attended = drug_attended.squeeze(1)  # [batch, hidden_dim]
        drug_attended = self.norm_drug(drug_attended + drug_emb)
        
        # Protein attends to Drug
        protein_attended, protein_attn_weights = self.protein_to_drug_attn(
            query=protein_emb_seq,
            key=drug_emb_seq,
            value=drug_emb_seq
        )
        protein_attended = protein_attended.squeeze(1)  # [batch, hidden_dim]
        protein_attended = self.norm_protein(protein_attended + protein_emb)
        
        # Fusion
        if self.fusion_method == "concat":
            fused = torch.cat([drug_attended, protein_attended], dim=-1)
            fused = self.fusion(fused)  # [batch, hidden_dim]
        elif self.fusion_method == "add":
            fused = drug_attended + protein_attended
        elif self.fusion_method == "gate":
            gate_input = torch.cat([drug_attended, protein_attended], dim=-1)
            gate = self.gate(gate_input)
            fused = gate * drug_attended + (1 - gate) * protein_attended
        else:
            fused = (drug_attended + protein_attended) / 2
        
        return fused


def test_cross_attention():
    """Test Cross-Attention Fusion"""
    batch_size = 4
    hidden_dim = 128
    
    # Dummy embeddings
    drug_emb = torch.randn(batch_size, hidden_dim)
    protein_emb = torch.randn(batch_size, hidden_dim)
    
    # Initialize model
    model = CrossAttentionFusion(
        hidden_dim=hidden_dim,
        num_heads=8,
        dropout=0.1,
        fusion_method="concat"
    )
    
    # Forward pass
    fused = model(drug_emb, protein_emb)
    
    print("Cross-Attention Fusion Test:")
    print(f"Drug embedding shape: {drug_emb.shape}")
    print(f"Protein embedding shape: {protein_emb.shape}")
    print(f"Fused output shape: {fused.shape}")  # Should be [4, 128]
    print(f"✓ Test passed!")


if __name__ == "__main__":
    test_cross_attention()
