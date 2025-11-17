"""
Evaluation script for GraphTransDTI
Test on DAVIS dataset for generalization
"""
import os
import sys
import yaml
import torch
import numpy as np
from tqdm import tqdm
import argparse

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import GraphTransDTI
from dataloader import get_kiba_dataloader, get_davis_dataloader
from utils import set_seed, get_device, calculate_metrics, print_metrics


class Evaluator:
    """
    Evaluator for GraphTransDTI
    """
    
    def __init__(self, config: dict, checkpoint_path: str):
        """
        Initialize evaluator
        
        Args:
            config: Configuration dictionary
            checkpoint_path: Path to model checkpoint
        """
        self.config = config
        
        # Set seed
        set_seed(config['experiment']['seed'])
        
        # Device
        self.device = get_device(prefer_cuda=(config['experiment']['device'] == 'cuda'))
        
        # Load model
        self.model = GraphTransDTI(config).to(self.device)
        
        # Load checkpoint
        print(f"[INFO] Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[INFO] Loaded model from epoch {checkpoint['epoch']}")
        
        self.model.eval()
    
    def evaluate(self, test_loader, dataset_name: str = "Test"):
        """
        Evaluate on test set
        
        Args:
            test_loader: Test data loader
            dataset_name: Name of dataset for display
        
        Returns:
            dict: Evaluation metrics
        """
        print(f"\n{'='*60}")
        print(f"Evaluating on {dataset_name} Dataset")
        print(f"{'='*60}")
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Evaluating {dataset_name}"):
                if batch is None:
                    continue
                
                # Move to device
                drug_batch = batch['drug'].to(self.device)
                protein_seq = batch['protein'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                predictions = self.model(drug_batch, protein_seq)
                
                # Store for metrics
                all_predictions.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Calculate metrics
        all_predictions = np.concatenate(all_predictions, axis=0).flatten()
        all_labels = np.concatenate(all_labels, axis=0).flatten()
        
        metrics = calculate_metrics(all_labels, all_predictions)
        print_metrics(metrics, prefix=dataset_name)
        
        return metrics, all_predictions, all_labels


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate GraphTransDTI")
    parser.add_argument('--config', type=str, default='./config.yaml',
                       help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='davis',
                       choices=['kiba', 'davis'],
                       help='Dataset to evaluate on')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n" + "=" * 60)
    print("GraphTransDTI Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Split: {args.split.upper()}")
    print("=" * 60)
    
    # Initialize evaluator
    evaluator = Evaluator(config, args.checkpoint)
    
    # Get test loader
    if args.dataset == 'kiba':
        test_loader = get_kiba_dataloader(
            data_dir=os.path.join(config['data']['data_dir'], 'kiba'),
            split=args.split,
            batch_size=config['training']['batch_size'],
            num_workers=config['data']['num_workers'],
            shuffle=False,
            train_ratio=config['data']['train_ratio'],
            val_ratio=config['data']['val_ratio'],
            test_ratio=config['data']['test_ratio'],
            seed=config['experiment']['seed']
        )
    elif args.dataset == 'davis':
        test_loader = get_davis_dataloader(
            data_dir=os.path.join(config['data']['data_dir'], 'davis'),
            split=args.split,
            batch_size=config['training']['batch_size'],
            num_workers=config['data']['num_workers'],
            shuffle=False,
            train_ratio=config['data']['train_ratio'],
            val_ratio=config['data']['val_ratio'],
            test_ratio=config['data']['test_ratio'],
            seed=config['experiment']['seed']
        )
    
    # Evaluate
    metrics, predictions, labels = evaluator.evaluate(
        test_loader,
        dataset_name=f"{args.dataset.upper()} {args.split.upper()}"
    )
    
    # Save results
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    
    results_path = os.path.join(
        results_dir,
        f"evaluation_{args.dataset}_{args.split}.npz"
    )
    
    np.savez(
        results_path,
        predictions=predictions,
        labels=labels,
        metrics=metrics
    )
    
    print(f"\n[INFO] Results saved to {results_path}")


if __name__ == "__main__":
    main()
