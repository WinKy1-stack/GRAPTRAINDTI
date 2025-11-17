"""
Featurizer for Drug-Target Interaction data
Converts raw SMILES and protein sequences to model-ready tensors
"""
import torch
import numpy as np
from rdkit import Chem
from typing import List, Dict, Tuple, Optional
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.smiles_to_graph import smiles_to_graph


# Amino acid vocabulary
AMINO_ACIDS = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
    'X', 'U', 'O'  # X=unknown, U=selenocysteine, O=pyrrolysine
]

# Create amino acid to index mapping
AMINO_ACID_TO_IDX = {aa: idx + 1 for idx, aa in enumerate(AMINO_ACIDS)}  # 0 reserved for padding
AMINO_ACID_TO_IDX['<PAD>'] = 0
AMINO_ACID_TO_IDX['<UNK>'] = len(AMINO_ACIDS) + 1


class DTIFeaturizer:
    """
    Featurizer for Drug-Target Interaction pairs
    """
    
    def __init__(self, 
                 max_drug_atoms: int = 100,
                 max_protein_len: int = 1000):
        """
        Args:
            max_drug_atoms: Maximum number of atoms in drug molecule
            max_protein_len: Maximum length of protein sequence
        """
        self.max_drug_atoms = max_drug_atoms
        self.max_protein_len = max_protein_len
        self.aa_vocab = AMINO_ACID_TO_IDX
        
    def featurize_drug(self, smiles: str):
        """
        Convert SMILES to graph
        
        Args:
            smiles: SMILES string
        
        Returns:
            PyG Data object or None if invalid
        """
        return smiles_to_graph(smiles, max_atoms=self.max_drug_atoms)
    
    def featurize_protein(self, sequence: str) -> torch.Tensor:
        """
        Convert protein sequence to indices
        
        Args:
            sequence: Protein amino acid sequence
        
        Returns:
            Tensor of indices [max_protein_len]
        """
        # Convert to uppercase
        sequence = sequence.upper()
        
        # Truncate if too long
        if len(sequence) > self.max_protein_len:
            sequence = sequence[:self.max_protein_len]
        
        # Convert to indices
        indices = []
        for aa in sequence:
            if aa in self.aa_vocab:
                indices.append(self.aa_vocab[aa])
            else:
                indices.append(self.aa_vocab['<UNK>'])
        
        # Pad to max length
        while len(indices) < self.max_protein_len:
            indices.append(self.aa_vocab['<PAD>'])
        
        return torch.tensor(indices, dtype=torch.long)
    
    def featurize_pair(self, smiles: str, protein_seq: str, label: float = None):
        """
        Featurize a drug-protein pair
        
        Args:
            smiles: Drug SMILES string
            protein_seq: Protein sequence
            label: Binding affinity (optional)
        
        Returns:
            dict: {
                'drug': PyG Data object,
                'protein': Tensor [max_protein_len],
                'label': float (if provided)
            }
            Returns None if SMILES is invalid
        """
        drug_graph = self.featurize_drug(smiles)
        if drug_graph is None:
            return None
        
        protein_tensor = self.featurize_protein(protein_seq)
        
        result = {
            'drug': drug_graph,
            'protein': protein_tensor,
            'smiles': smiles,
            'protein_seq': protein_seq[:50] + '...' if len(protein_seq) > 50 else protein_seq
        }
        
        if label is not None:
            result['label'] = torch.tensor([label], dtype=torch.float)
        
        return result


def collate_dti_batch(batch: List[Dict]) -> Dict:
    """
    Custom collate function for DataLoader
    
    Args:
        batch: List of featurized samples
    
    Returns:
        dict: Batched data
    """
    from torch_geometric.data import Batch
    
    # Filter out None samples
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        return None
    
    # Batch drug graphs
    drug_graphs = [item['drug'] for item in batch]
    drug_batch = Batch.from_data_list(drug_graphs)
    
    # Stack protein sequences
    protein_seqs = torch.stack([item['protein'] for item in batch])
    
    # Stack labels if available
    if 'label' in batch[0]:
        labels = torch.stack([item['label'] for item in batch])
    else:
        labels = None
    
    result = {
        'drug': drug_batch,
        'protein': protein_seqs,
        'smiles': [item['smiles'] for item in batch],
        'protein_seq': [item['protein_seq'] for item in batch]
    }
    
    if labels is not None:
        result['label'] = labels
    
    return result


def test_featurizer():
    """Test the featurizer"""
    print("Testing DTIFeaturizer...")
    print("=" * 60)
    
    featurizer = DTIFeaturizer(max_drug_atoms=100, max_protein_len=1000)
    
    # Test samples
    test_smiles = "CCO"  # Ethanol
    test_protein = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWSTPSELGHAGLNGDILVWNPVLEDAFELSSMGIRVDADTLKHQLALTGDEDRLELEWHQALLRGEMPQTIGGGIGQSRLTMLLLQLPHIGQVQAGVWPAAVRESVPSLL"  # Example protein
    test_label = 5.2  # Example binding affinity
    
    # Featurize
    sample = featurizer.featurize_pair(test_smiles, test_protein, test_label)
    
    if sample:
        print(f"SMILES: {test_smiles}")
        print(f"Protein: {test_protein[:50]}...")
        print(f"Label: {test_label}")
        print(f"\nDrug graph:")
        print(f"  Num atoms: {sample['drug'].x.size(0)}")
        print(f"  Num bonds: {sample['drug'].edge_index.size(1)}")
        print(f"  Node features: {sample['drug'].x.shape}")
        print(f"  Edge features: {sample['drug'].edge_attr.shape}")
        print(f"\nProtein tensor:")
        print(f"  Shape: {sample['protein'].shape}")
        print(f"  First 20 indices: {sample['protein'][:20].tolist()}")
        print(f"\nLabel:")
        print(f"  Shape: {sample['label'].shape}")
        print(f"  Value: {sample['label'].item()}")
        print("\n✓ Test passed!")
    else:
        print("✗ Test failed - invalid SMILES")
    
    print("=" * 60)


if __name__ == "__main__":
    test_featurizer()
