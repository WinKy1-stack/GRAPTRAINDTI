"""
Convert SMILES string to PyTorch Geometric graph
Using RDKit for molecular featurization
"""
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from torch_geometric.data import Data
from typing import Optional


def one_hot_encoding(x, allowable_set):
    """One-hot encode a value against an allowable set"""
    if x not in allowable_set:
        x = allowable_set[-1]  # Use last element as "unknown"
    return list(map(lambda s: x == s, allowable_set))


def get_atom_features(atom):
    """
    Extract atom features from RDKit atom object
    
    Returns 78-dimensional feature vector:
    - Atom type (one-hot, 44 types)
    - Degree (one-hot, 11 values: 0-10)
    - Formal charge (one-hot, 11 values: -5 to +5)
    - Hybridization (one-hot, 6 types)
    - Aromaticity (binary)
    - Total H (one-hot, 5 values: 0-4)
    - Chirality (binary)
    """
    # Atom type (44 common elements)
    atom_type = one_hot_encoding(
        atom.GetSymbol(),
        ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'B', 'Si', 
         'Se', 'Te', 'As', 'Al', 'Zn', 'Sn', 'H', 'Fe', 'Cu', 'Ca',
         'Na', 'K', 'Mg', 'Li', 'Mn', 'Co', 'Ni', 'Cr', 'Ti', 'V',
         'Mo', 'W', 'Ag', 'Au', 'Pt', 'Pd', 'Hg', 'Cd', 'Pb', 'Bi',
         'Sb', 'Ge', 'Unknown']
    )
    
    # Degree (0-10)
    degree = one_hot_encoding(atom.GetDegree(), list(range(11)))
    
    # Formal charge (-5 to +5)
    formal_charge = one_hot_encoding(
        atom.GetFormalCharge(), 
        list(range(-5, 6))
    )
    
    # Hybridization
    hybridization = one_hot_encoding(
        str(atom.GetHybridization()),
        ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'UNSPECIFIED']
    )
    
    # Aromaticity
    aromaticity = [atom.GetIsAromatic()]
    
    # Total number of Hs
    total_h = one_hot_encoding(atom.GetTotalNumHs(), list(range(5)))
    
    # Chirality
    chirality = [atom.HasProp('_ChiralityPossible')]
    
    features = (
        atom_type + degree + formal_charge + 
        hybridization + aromaticity + total_h + chirality
    )
    
    return np.array(features, dtype=np.float32)


def get_bond_features(bond):
    """
    Extract bond features from RDKit bond object
    
    Returns 12-dimensional feature vector:
    - Bond type (one-hot, 4 types)
    - Conjugated (binary)
    - In ring (binary)
    - Stereo (one-hot, 6 types)
    """
    # Bond type
    bond_type = one_hot_encoding(
        str(bond.GetBondType()),
        ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
    )
    
    # Conjugated
    conjugated = [bond.GetIsConjugated()]
    
    # In ring
    in_ring = [bond.IsInRing()]
    
    # Stereo configuration
    stereo = one_hot_encoding(
        str(bond.GetStereo()),
        ['STEREONONE', 'STEREOANY', 'STEREOZ', 'STEREOE', 
         'STEREOCIS', 'STEREOTRANS']
    )
    
    features = bond_type + conjugated + in_ring + stereo
    
    return np.array(features, dtype=np.float32)


def smiles_to_graph(smiles: str, max_atoms: int = 100) -> Optional[Data]:
    """
    Convert SMILES string to PyTorch Geometric Data object
    
    Args:
        smiles (str): SMILES string
        max_atoms (int): Maximum number of atoms (for padding/truncation)
    
    Returns:
        Data: PyG Data object with node features and edge indices
              Returns None if SMILES is invalid
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    # Add explicit hydrogens (optional - may increase graph size)
    # mol = Chem.AddHs(mol)
    
    # Get atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(get_atom_features(atom))
    
    # Truncate if too many atoms
    num_atoms = len(atom_features)
    if num_atoms > max_atoms:
        atom_features = atom_features[:max_atoms]
        num_atoms = max_atoms
    
    # If no atoms, return None
    if num_atoms == 0:
        return None
    
    x = torch.tensor(atom_features, dtype=torch.float)
    
    # Get edge indices and features
    edge_indices = []
    edge_features = []
    
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        # Skip if atoms are beyond max_atoms
        if i >= max_atoms or j >= max_atoms:
            continue
        
        # Add both directions (undirected graph)
        edge_indices.append([i, j])
        edge_indices.append([j, i])
        
        bond_feat = get_bond_features(bond)
        edge_features.append(bond_feat)
        edge_features.append(bond_feat)  # Same for reverse edge
    
    # Handle molecules with no bonds (single atom)
    if len(edge_indices) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 12), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
    
    # Create PyG Data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        smiles=smiles
    )
    
    return data


def test_smiles_to_graph():
    """Test the SMILES to graph conversion"""
    test_smiles = [
        "CCO",  # Ethanol
        "CC(=O)O",  # Acetic acid
        "c1ccccc1",  # Benzene
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
    ]
    
    print("Testing SMILES to Graph conversion:")
    print("=" * 60)
    
    for smi in test_smiles:
        data = smiles_to_graph(smi)
        if data is not None:
            print(f"\nSMILES: {smi}")
            print(f"  Num atoms: {data.x.size(0)}")
            print(f"  Num bonds: {data.edge_index.size(1)}")
            print(f"  Node features shape: {data.x.shape}")
            print(f"  Edge features shape: {data.edge_attr.shape}")
        else:
            print(f"\nSMILES: {smi} - INVALID")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_smiles_to_graph()
