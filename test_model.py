"""
Quick test script to verify model works before training
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import yaml
from models import GraphTransDTI
from dataloader import get_kiba_dataloader

print("=" * 60)
print("TESTING GRAPHTRANSDTI MODEL")
print("=" * 60)

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Override for quick test
config['training']['batch_size'] = 4
config['data']['num_workers'] = 0

print("\n1. Checking GPU...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

print("\n2. Initializing model...")
model = GraphTransDTI(config).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"   Total parameters: {total_params:,}")

print("\n3. Loading test batch...")
train_loader = get_kiba_dataloader(
    data_dir='./data/kiba',
    split='train',
    batch_size=4,
    num_workers=0,
    shuffle=False,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42
)

# Get one batch
batch = next(iter(train_loader))
drug_batch = batch['drug'].to(device)
protein_seq = batch['protein'].to(device)
labels = batch['label'].to(device)

print(f"   Drug batch: {drug_batch}")
print(f"   Protein shape: {protein_seq.shape}")
print(f"   Labels shape: {labels.shape}")

print("\n4. Forward pass test...")
model.eval()
with torch.no_grad():
    predictions = model(drug_batch, protein_seq)
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Sample predictions: {predictions[:3].squeeze()}")
    print(f"   Sample labels: {labels[:3].squeeze()}")

print("\n5. Loss computation test...")
criterion = torch.nn.MSELoss()
loss = criterion(predictions, labels)
print(f"   MSE Loss: {loss.item():.4f}")

print("\n6. Backward pass test...")
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Forward
predictions = model(drug_batch, protein_seq)
loss = criterion(predictions, labels)

# Backward
optimizer.zero_grad()
loss.backward()
optimizer.step()

print(f"   Loss after backward: {loss.item():.4f}")
print("   ✓ Gradients computed successfully")

print("\n" + "=" * 60)
print("✓ ALL TESTS PASSED!")
print("=" * 60)
print("\nModel is ready for training!")
print("Run: python src/train.py")
