"""
DAVIS Dataset Loader
Drug-Target Binding Affinity Dataset
"""
import os
import json
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, List
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataloader.featurizer import DTIFeaturizer, collate_dti_batch


class DAVISDataset(Dataset):
    """
    DAVIS Dataset for Drug-Target Interaction
    
    Dataset structure (expected):
        data/davis/
            ├── ligands_can.txt  (SMILES)
            ├── proteins.txt  (sequences)
            └── Y  (affinity matrix as pickle)
    
    Reference:
        Davis et al. (2011) "Comprehensive analysis of kinase inhibitor selectivity"
    """
    
    def __init__(self,
                 data_dir: str = "./data/davis",
                 split: str = "train",
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.1,
                 test_ratio: float = 0.1,
                 featurizer: DTIFeaturizer = None,
                 seed: int = 42):
        """
        Args:
            data_dir: Path to DAVIS data directory
            split: 'train', 'val', or 'test'
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            featurizer: DTIFeaturizer instance
            seed: Random seed for splitting
        """
        super(DAVISDataset, self).__init__()
        
        self.data_dir = data_dir
        self.split = split
        self.featurizer = featurizer if featurizer else DTIFeaturizer()
        
        # Load data
        self.smiles_list, self.proteins_list, self.affinity_matrix = self._load_data()
        
        # Create pairs
        self.pairs = self._create_pairs()
        
        # Split dataset
        self.pairs = self._split_dataset(train_ratio, val_ratio, test_ratio, seed)
        
        print(f"[INFO] DAVIS {split.upper()} dataset loaded: {len(self.pairs)} pairs")
    
    def _load_data(self) -> Tuple[List[str], List[str], np.ndarray]:
        """Load SMILES, proteins, and affinity matrix from DeepDTA format"""
        
        ligands_file = os.path.join(self.data_dir, "ligands_can.txt")
        proteins_file = os.path.join(self.data_dir, "proteins.txt")
        affinity_file = os.path.join(self.data_dir, "Y")
        
        if not all([os.path.exists(ligands_file), 
                    os.path.exists(proteins_file), 
                    os.path.exists(affinity_file)]):
            raise FileNotFoundError(
                f"DAVIS data files not found in {self.data_dir}\n"
                "Expected files: ligands_can.txt, proteins.txt, Y\n"
                "See data/DATA_DOWNLOAD_GUIDE.md for download instructions."
            )
        
        # Load SMILES from JSON format (DeepDTA uses dict format)
        with open(ligands_file, 'r') as f:
            ligands_dict = json.load(f)
            smiles_list = list(ligands_dict.values())
        
        # Load protein sequences from JSON format
        with open(proteins_file, 'r') as f:
            proteins_dict = json.load(f)
            proteins_list = list(proteins_dict.values())
        
        # Load affinity matrix (pickle format)
        with open(affinity_file, 'rb') as f:
            affinity_matrix = pickle.load(f, encoding='latin1')
        
        print(f"[INFO] Loaded {len(smiles_list)} drugs, {len(proteins_list)} proteins")
        print(f"[INFO] Affinity matrix shape: {affinity_matrix.shape}")
        
        return smiles_list, proteins_list, affinity_matrix
    
    def _create_pairs(self) -> List[Tuple[int, int, float]]:
        """
        Create valid drug-protein pairs
        Only include pairs with known affinity (not NaN)
        
        Returns:
            List of (drug_idx, protein_idx, affinity)
        """
        pairs = []
        
        for i in range(len(self.smiles_list)):
            for j in range(len(self.proteins_list)):
                affinity = self.affinity_matrix[i, j]
                
                # Skip NaN values
                if not np.isnan(affinity):
                    pairs.append((i, j, affinity))
        
        print(f"[INFO] Created {len(pairs)} valid drug-protein pairs")
        
        return pairs
    
    def _split_dataset(self, 
                       train_ratio: float, 
                       val_ratio: float, 
                       test_ratio: float, 
                       seed: int) -> List[Tuple]:
        """Split dataset into train/val/test"""
        
        np.random.seed(seed)
        indices = np.random.permutation(len(self.pairs))
        
        n_train = int(len(self.pairs) * train_ratio)
        n_val = int(len(self.pairs) * val_ratio)
        
        if self.split == "train":
            indices = indices[:n_train]
        elif self.split == "val":
            indices = indices[n_train:n_train + n_val]
        elif self.split == "test":
            indices = indices[n_train + n_val:]
        else:
            raise ValueError(f"Invalid split: {self.split}")
        
        return [self.pairs[i] for i in indices]
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        """
        Get a single drug-protein pair
        
        Returns:
            dict: Featurized sample
        """
        drug_idx, protein_idx, affinity = self.pairs[idx]
        
        smiles = self.smiles_list[drug_idx]
        protein_seq = self.proteins_list[protein_idx]
        
        # Featurize
        sample = self.featurizer.featurize_pair(smiles, protein_seq, affinity)
        
        return sample


def get_davis_dataloader(data_dir: str = "./data/davis",
                         split: str = "train",
                         batch_size: int = 64,
                         num_workers: int = 4,
                         shuffle: bool = True,
                         **kwargs) -> DataLoader:
    """
    Get DAVIS DataLoader
    
    Args:
        data_dir: Path to DAVIS data
        split: 'train', 'val', or 'test'
        batch_size: Batch size
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle data
        **kwargs: Additional arguments for DAVISDataset
    
    Returns:
        DataLoader
    """
    dataset = DAVISDataset(data_dir=data_dir, split=split, **kwargs)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_dti_batch,
        pin_memory=True
    )
    
    return dataloader


def test_davis_loader():
    """Test DAVIS dataset loader"""
    print("\nTesting DAVIS Dataset Loader...")
    print("=" * 60)
    
    try:
        dataloader = get_davis_dataloader(
            data_dir="./data/davis",
            split="train",
            batch_size=4,
            num_workers=0,
            shuffle=True
        )
        
        # Get one batch
        batch = next(iter(dataloader))
        
        print(f"Batch keys: {batch.keys()}")
        print(f"Drug batch: {batch['drug']}")
        print(f"  Num graphs: {batch['drug'].num_graphs}")
        print(f"  Total atoms: {batch['drug'].num_nodes}")
        print(f"Protein shape: {batch['protein'].shape}")
        print(f"Label shape: {batch['label'].shape}")
        print(f"\nSample SMILES: {batch['smiles'][0]}")
        print(f"Sample protein: {batch['protein_seq'][0]}")
        print(f"Sample label: {batch['label'][0].item():.3f}")
        print("\n✓ Test passed!")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        print("Note: Make sure DAVIS data is downloaded and placed in ./data/davis/")
    
    print("=" * 60)


if __name__ == "__main__":
    test_davis_loader()
