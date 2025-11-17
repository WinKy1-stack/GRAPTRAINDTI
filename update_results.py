import json

# Load existing results
with open('results/results_summary.json', 'r') as f:
    results = json.load(f)

# Load normalized DAVIS results
with open('results/davis_normalized/results.json', 'r') as f:
    davis_normalized = json.load(f)

# Add to main results
results['davis_normalized'] = davis_normalized

# Save updated results
with open('results/results_summary.json', 'w') as f:
    json.dump(results, f, indent=2)

print("✓ Updated results_summary.json with normalized DAVIS results")
print("\n" + "="*70)
print("COMPARISON: RAW vs NORMALIZED DAVIS")
print("="*70)
print(f"RMSE:")
print(f"  Raw:        {results['davis']['test_metrics']['rmse']:.2f}")
print(f"  Normalized: {results['davis_normalized']['metrics']['rmse']:.4f}")
print(f"  Improvement: {results['davis']['test_metrics']['rmse'] / results['davis_normalized']['metrics']['rmse']:.1f}x better")
print(f"\nPearson:")
print(f"  Raw:        {results['davis']['test_metrics']['pearson']:.4f}")
print(f"  Normalized: {results['davis_normalized']['metrics']['pearson']:.4f}")
print(f"\nCI:")
print(f"  Raw:        {results['davis']['test_metrics']['ci']:.4f}")
print(f"  Normalized: {results['davis_normalized']['metrics']['ci']:.4f}")
print("="*70)
